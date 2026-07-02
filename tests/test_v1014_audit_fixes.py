"""Regression tests for v10.14.0: the 2026-07-02 external audit fixes.

Covers the confirmed findings from the independent v10.13.0 audit:
hybrid search discarding vec-only hits when FTS is empty (H4), embed_meta
back-compat column backfill + doctor surviving legacy schemas (H2), atomic
content+vector writes (H1/L9/M3), CML exploder boundary false-splits (H3),
remediate-oversized children bypassing the exploder (M9), valid_from
enforcement + valid_until off-by-one (M5/L6), model provenance (M1),
embed-fill backfill command (M2), merge lineage in nyx_insights (M4),
CONTRADICT blast-radius guard (M6), weave bridge embedding (L5), stale
mem_by_id reload (L7), julianday parameterization (S4), and server/CLI
hardening (L1/L2/L3/L8/M8).
"""

import sqlite3

import mnemos.core as core_mod
from mnemos.core import Mnemos
from mnemos.splitter import explode_cml_chain, _sep_free
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _store(tmp_path, name="m.db"):
    return SQLiteStore(db_path=str(tmp_path / name), namespace="t")


def _mnemos(tmp_path, **kw):
    return Mnemos(store=_store(tmp_path), namespace="t",
                  enable_contradiction_detection=False, enable_rerank=False, **kw)


def _vec(seed=0.001):
    return [seed] * DIMS


def _fake_embed(texts, prefix="passage"):
    return [_vec() for _ in texts]


def _failing_embed(texts, prefix="passage"):
    return []


class TestH4HybridVecOnlyFallthrough:
    def test_hybrid_returns_vec_hits_when_fts_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = _mnemos(tmp_path)
        mid = m.store_memory("dev", "the cat sleeps near the heater",
                             skip_dedup=True)["id"]
        res = m.search("zzzunmatchable qqqtoken", search_mode="hybrid",
                       auto_widen=False)
        ids = [r["id"] for r in res["results"]]
        assert mid in ids, (
            "hybrid search must fall through to vector-only hits when FTS "
            "matches nothing (cross-lingual retrieval case)"
        )

    def test_hybrid_fts_only_path_still_works(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        mid = m.store_memory("dev", "unique heaterword content",
                             skip_dedup=True)["id"]
        res = m.search("heaterword", search_mode="hybrid", auto_widen=False)
        ids = [r["id"] for r in res["results"]]
        assert mid in ids


def _legacy_vec_db(path):
    """A pre-v10.6 embed_meta schema: no text_hash, no model column."""
    import sqlite_vec
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute(
        f"CREATE VIRTUAL TABLE embed_vec USING vec0(embedding float[{DIMS}])"
    )
    for table in ("embed_meta", "embed_meta_arch"):
        conn.execute(f"""
            CREATE TABLE {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                embedded_at TEXT,
                UNIQUE(source_db, source_id)
            )
        """)
    conn.commit()
    conn.close()


class TestH2EmbedMetaBackfill:
    def test_store_with_embedding_on_pre_text_hash_db(self, tmp_path):
        db = str(tmp_path / "legacy.db")
        _legacy_vec_db(db)
        store = SQLiteStore(db_path=db, namespace="t")
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="legacy alpha"),
            embedding=_vec(), text_hash="h-legacy",
        )
        row = store._get_conn().execute(
            "SELECT text_hash, model FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()
        assert row["text_hash"] == "h-legacy"

    def test_embed_meta_arch_backfilled_too(self, tmp_path):
        db = str(tmp_path / "legacy2.db")
        _legacy_vec_db(db)
        store = SQLiteStore(db_path=db, namespace="t")
        cols = {r[1] for r in store._get_conn().execute(
            "PRAGMA table_info(embed_meta_arch)").fetchall()}
        assert {"text_hash", "model"} <= cols

    def test_doctor_survives_embed_status_failure(self, tmp_path, monkeypatch):
        m = _mnemos(tmp_path)
        def _boom():
            raise sqlite3.OperationalError("no such column: text_hash")
        monkeypatch.setattr(m, "embed_status", _boom)
        report = m.doctor(migrate=False)
        assert any("coverage check failed" in i for i in report["issues"])


class _FlakyConn:
    """Proxy that raises on the first statement matching a prefix, to
    simulate a crash at an exact point inside a multi-statement write."""

    def __init__(self, real, fail_prefix):
        self._real = real
        self._fail_prefix = fail_prefix

    def execute(self, sql, *a, **kw):
        if sql.strip().startswith(self._fail_prefix):
            raise RuntimeError(f"simulated crash at: {self._fail_prefix}")
        return self._real.execute(sql, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestH1AtomicContentVectorWrite:
    def test_store_rolls_back_content_when_vector_write_fails(
            self, tmp_path, monkeypatch):
        import pytest
        store = _store(tmp_path)
        store._get_conn()

        def _boom(mid, embedding, text_hash=None, **kw):
            raise RuntimeError("simulated vector-write failure")
        monkeypatch.setattr(store, "_store_embedding", _boom)
        with pytest.raises(RuntimeError):
            store.store_memory(
                Memory(namespace="t", project="dev", content="atomic alpha"),
                embedding=_vec(), text_hash="h",
            )
        n = store._get_conn().execute(
            "SELECT COUNT(*) FROM memories").fetchone()[0]
        assert n == 0, "content row must not survive a failed vector write"

    def test_update_rolls_back_content_when_vector_write_fails(
            self, tmp_path, monkeypatch):
        import pytest
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="original"),
            embedding=_vec(), text_hash="h1",
        )

        def _boom(mid, embedding, text_hash=None, **kw):
            raise RuntimeError("simulated vector-write failure")
        monkeypatch.setattr(store, "_store_embedding", _boom)
        with pytest.raises(RuntimeError):
            store.update_memory(mid, {"content": "revised"},
                                embedding=_vec(0.002), text_hash="h2")
        conn = store._get_conn()
        row = conn.execute(
            "SELECT content FROM memories WHERE id=?", (mid,)).fetchone()
        assert row["content"] == "original"
        meta = conn.execute(
            "SELECT text_hash FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()
        assert meta["text_hash"] == "h1", (
            "old vector metadata must remain consistent with the "
            "rolled-back content"
        )

    def test_reembed_still_replaces_existing_vector(self, tmp_path):
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="replace me"),
            embedding=_vec(), text_hash="h1",
        )
        store.update_memory(mid, {"content": "replaced"},
                            embedding=_vec(0.5), text_hash="h2")
        conn = store._get_conn()
        assert conn.execute(
            "SELECT COUNT(*) FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT text_hash FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()["text_hash"] == "h2"


class TestM3AtomicArchiveMove:
    def test_move_is_atomic_when_active_delete_fails(
            self, tmp_path, monkeypatch):
        import pytest
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="to archive"),
            embedding=_vec(), text_hash="h",
        )
        real_conn = store._get_conn()
        flaky = _FlakyConn(real_conn, "DELETE FROM embed_vec WHERE")
        monkeypatch.setattr(store, "_get_conn", lambda: flaky)
        with pytest.raises(RuntimeError):
            store.move_embedding_to_archive(mid)
        assert real_conn.execute(
            "SELECT COUNT(*) FROM embed_meta_arch").fetchone()[0] == 0, (
            "a crash between arch insert and active delete must not leave "
            "the vector in both indexes"
        )
        assert real_conn.execute(
            "SELECT COUNT(*) FROM embed_meta").fetchone()[0] == 1


class TestH1QdrantCompensation:
    def _fake_qdrant_store(self, tmp_path, client):
        from mnemos.storage.qdrant_store import QdrantStore

        class _FakeModels:
            @staticmethod
            def PointStruct(**kw):
                return kw

        qs = object.__new__(QdrantStore)
        qs.namespace = "t"
        qs._sqlite = _store(tmp_path, "q.db")
        qs._client = client
        qs._collection = "test"
        qs._qmodels = _FakeModels
        return qs

    def test_store_compensates_when_upsert_fails(self, tmp_path):
        import pytest

        class _DownClient:
            def upsert(self, **kw):
                raise RuntimeError("qdrant unreachable")

        qs = self._fake_qdrant_store(tmp_path, _DownClient())
        with pytest.raises(RuntimeError):
            qs.store_memory(
                Memory(namespace="t", project="dev", content="net split"),
                embedding=_vec(), text_hash="h",
            )
        n = qs._sqlite._get_conn().execute(
            "SELECT COUNT(*) FROM memories").fetchone()[0]
        assert n == 0, (
            "a failed Qdrant upsert must not leave a vector-less content row"
        )

    def test_upsert_payload_carries_text_hash(self, tmp_path):
        captured = {}

        class _CapturingClient:
            def upsert(self, collection_name, points):
                captured["points"] = points

        qs = self._fake_qdrant_store(tmp_path, _CapturingClient())
        qs.store_memory(
            Memory(namespace="t", project="dev", content="hash travels"),
            embedding=_vec(), text_hash="h-q",
        )
        assert captured["points"][0]["payload"].get("text_hash") == "h-q"


class TestH3ExploderBoundaries:
    def test_drive_letter_mid_clause_not_split(self):
        blob = "F:free space on C: drive is low; D:clean it up"
        assert explode_cml_chain(blob) == (
            "F:free space on C: drive is low\nD:clean it up"
        )

    def test_punctuation_glued_prefix_not_split(self):
        blob = "F:build uses x (P:prefer HE-AAC v2); D:ship"
        assert explode_cml_chain(blob) == (
            "F:build uses x (P:prefer HE-AAC v2)\nD:ship"
        )

    def test_quoted_prefix_not_split(self):
        blob = 'F:the header says "W:warning" verbatim; D:keep it'
        assert explode_cml_chain(blob) == (
            'F:the header says "W:warning" verbatim\nD:keep it'
        )

    def test_period_boundary_still_splits(self):
        blob = "F:alpha done. D:beta next"
        assert explode_cml_chain(blob) == "F:alpha done.\nD:beta next"

    def test_semicolon_no_space_still_splits(self):
        assert explode_cml_chain("F:alpha;D:beta") == "F:alpha\nD:beta"

    def test_sep_free_no_longer_strips_periods(self):
        assert "." in _sep_free("v1.2.3 release")


class TestL8FtsTriggerGuard:
    _TRIGGER_SQL = ("SELECT sql FROM sqlite_master WHERE type='trigger' "
                    "AND name='memories_au'")

    def test_update_trigger_has_when_guard(self, tmp_path):
        store = _store(tmp_path)
        sql = store._get_conn().execute(self._TRIGGER_SQL).fetchone()["sql"]
        head = sql.upper().split("BEGIN")[0]
        assert "WHEN" in head, (
            "without a WHEN guard every access-count bump re-tokenizes the "
            "whole row into FTS"
        )

    def test_legacy_unguarded_trigger_is_upgraded(self, tmp_path):
        db = str(tmp_path / "legacy-trigger.db")
        store = SQLiteStore(db_path=db, namespace="t")
        conn = store._get_conn()
        conn.execute("DROP TRIGGER memories_au")
        conn.execute("""
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, project, tags)
                VALUES('delete', old.id, old.content, old.project, old.tags);
                INSERT INTO memories_fts(rowid, content, project, tags)
                VALUES (new.id, new.content, new.project, new.tags);
            END
        """)
        conn.commit()
        store.close()
        store2 = SQLiteStore(db_path=db, namespace="t")
        sql = store2._get_conn().execute(self._TRIGGER_SQL).fetchone()["sql"]
        head = sql.upper().split("BEGIN")[0]
        assert "WHEN" in head

    def test_fts_still_syncs_on_content_update(self, tmp_path):
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="gammaword original"))
        store.update_memory(mid, {"content": "deltaword revised"})
        assert store.search_fts("deltaword") == [mid]
        assert store.search_fts("gammaword") == []


class TestL3ReadMsgGuard:
    def test_malformed_line_is_skipped_not_fatal(self, monkeypatch):
        import io
        import mnemos.mcp_server as srv
        monkeypatch.setattr(srv.sys, "stdin", io.StringIO("this is not json\n"))
        assert srv.read_msg() is srv.SKIP_MSG

    def test_eof_still_returns_none(self, monkeypatch):
        import io
        import mnemos.mcp_server as srv
        monkeypatch.setattr(srv.sys, "stdin", io.StringIO(""))
        assert srv.read_msg() is None

    def test_valid_json_still_parses(self, monkeypatch):
        import io
        import mnemos.mcp_server as srv
        monkeypatch.setattr(srv.sys, "stdin", io.StringIO('{"a": 1}\n'))
        assert srv.read_msg() == {"a": 1}


class TestL2SanitizedToolErrors:
    def test_error_reports_class_and_truncates(self, monkeypatch):
        import io
        import json as j
        import mnemos.mcp_server as srv

        class _ExplodingMnemos:
            def __getattr__(self, name):
                raise RuntimeError(
                    "boom at /root/very/secret/db.sqlite " + "y" * 1000)

        req = j.dumps({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "memory_get", "arguments": {"id": 1}},
        })
        out = io.StringIO()
        monkeypatch.setattr(srv, "build_mnemos", lambda: _ExplodingMnemos())
        monkeypatch.setattr(srv.sys, "stdin", io.StringIO(req + "\n"))
        monkeypatch.setattr(srv.sys, "stdout", out)
        srv.main()
        responses = [j.loads(l) for l in out.getvalue().splitlines()]
        resp = [r for r in responses if r.get("id") == 7][0]
        err = j.loads(resp["result"]["content"][0]["text"])["error"]
        assert err.startswith("RuntimeError:")
        assert len(err) <= 400, "error text must be truncated"
        assert resp["result"]["isError"] is True


class TestM8CliUtf8Output:
    def test_non_utf8_stdout_is_reconfigured(self, monkeypatch):
        import mnemos.cli as cli

        class _FakeStream:
            encoding = "cp1252"

            def __init__(self):
                self.reconfigured = None

            def reconfigure(self, encoding=None, errors=None):
                self.reconfigured = (encoding, errors)

        fake_out, fake_err = _FakeStream(), _FakeStream()
        monkeypatch.setattr(cli.sys, "stdout", fake_out)
        monkeypatch.setattr(cli.sys, "stderr", fake_err)
        cli._ensure_utf8_output()
        assert fake_out.reconfigured == ("utf-8", "replace")
        assert fake_err.reconfigured == ("utf-8", "replace")

    def test_utf8_stdout_left_alone(self, monkeypatch):
        import mnemos.cli as cli

        class _FakeStream:
            encoding = "utf-8"

            def __init__(self):
                self.reconfigured = None

            def reconfigure(self, encoding=None, errors=None):
                self.reconfigured = (encoding, errors)

        fake = _FakeStream()
        monkeypatch.setattr(cli.sys, "stdout", fake)
        monkeypatch.setattr(cli.sys, "stderr", fake)
        cli._ensure_utf8_output()
        assert fake.reconfigured is None


class TestL1BulkRewriteTimeout:
    def test_catastrophic_regex_returns_error_not_hang(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        monkeypatch.setenv("MNEMOS_BULK_REWRITE_TIMEOUT", "1")
        m = _mnemos(tmp_path)
        m.store_memory("dev", "a" * 64 + "b", skip_dedup=True)
        res = m.bulk_rewrite(pattern=r"(a+)+$", replacement="x",
                             use_regex=True, dry_run=True)
        assert "timed out" in res.get("error", ""), (
            "a caller-supplied catastrophic regex must be bounded, not hang "
            "the single-threaded MCP loop"
        )


class TestM4MergeLineage:
    def test_apply_merge_records_nyx_insights_lineage(
            self, tmp_path, monkeypatch):
        import mnemos.consolidation.phases as phases
        monkeypatch.setenv("MNEMOS_NAMESPACE", "t")
        monkeypatch.setattr(phases, "fastembed_embed",
                            lambda texts, prefix="passage": [_vec()])
        store = _store(tmp_path)
        ids = [
            store.store_memory(Memory(namespace="t", project="dev", content=c))
            for c in ("one fact", "two fact")
        ]
        mem_by_id = {
            mid: {"project": "dev", "tags": "x", "importance": 5,
                  "consolidation_lock": 0, "verified": 0, "type": "fact",
                  "last_confirmed": None}
            for mid in ids
        }
        conn = store._get_conn()
        new_id = phases.apply_merge(conn, ids, "merged fact", mem_by_id)
        assert new_id is not None
        sources = store.get_merged_sources(new_id, valid_only=False)
        assert sorted(m.id for m in sources) == sorted(ids), (
            "expand_merged reads lineage from nyx_insights; the primary "
            "merge path must write it, tags alone are invisible to the join"
        )


class TestM6ContradictBlastRadius:
    def _setup(self, tmp_path, monkeypatch, older_extra):
        import math
        import mnemos.consolidation.phases as phases
        monkeypatch.setenv("MNEMOS_NAMESPACE", "t")
        store = _store(tmp_path)
        a = store.store_memory(
            Memory(namespace="t", project="dev", content="older fact"))
        b = store.store_memory(
            Memory(namespace="t", project="dev", content="newer fact"))
        va = [1.0] + [0.0] * (DIMS - 1)
        vb = [0.7, math.sqrt(1 - 0.49)] + [0.0] * (DIMS - 2)
        embeds = {a: va, b: vb}
        mem_by_id = {
            a: {"project": "dev", "type": "fact",
                "created_at": "2026-01-01 00:00:00", "content": "older fact",
                **older_extra},
            b: {"project": "dev", "type": "fact",
                "created_at": "2026-06-01 00:00:00", "content": "newer fact",
                "verified": 0, "importance": 5},
        }
        monkeypatch.setattr(
            phases, "opus_chat",
            lambda *a2, **k: "SUPERSEDED|steered verdict from injected content")
        return phases, store, a, b, embeds, mem_by_id

    def test_verified_memory_never_auto_archived(self, tmp_path, monkeypatch):
        phases, store, a, b, embeds, mem_by_id = self._setup(
            tmp_path, monkeypatch, {"verified": 1, "importance": 5})
        conn = store._get_conn()
        phases.phase_contradict(conn, embeds, mem_by_id,
                                is_surge=False, execute=True)
        row = conn.execute(
            "SELECT status FROM memories WHERE id=?", (a,)).fetchone()
        assert row["status"] == "active", (
            "an LLM verdict derived from untrusted memory content must not "
            "be able to archive a verified memory"
        )

    def test_high_importance_memory_never_auto_archived(
            self, tmp_path, monkeypatch):
        phases, store, a, b, embeds, mem_by_id = self._setup(
            tmp_path, monkeypatch, {"verified": 0, "importance": 9})
        conn = store._get_conn()
        phases.phase_contradict(conn, embeds, mem_by_id,
                                is_surge=False, execute=True)
        row = conn.execute(
            "SELECT status FROM memories WHERE id=?", (a,)).fetchone()
        assert row["status"] == "active"

    def test_ordinary_memory_still_superseded(self, tmp_path, monkeypatch):
        phases, store, a, b, embeds, mem_by_id = self._setup(
            tmp_path, monkeypatch, {"verified": 0, "importance": 5})
        conn = store._get_conn()
        stats = phases.phase_contradict(conn, embeds, mem_by_id,
                                        is_surge=False, execute=True)
        row = conn.execute(
            "SELECT status FROM memories WHERE id=?", (a,)).fetchone()
        assert row["status"] == "archived"
        assert stats["superseded"] == 1


class TestL5BridgeEmbedding:
    def test_bridge_insight_is_embedded_at_creation(
            self, tmp_path, monkeypatch):
        import mnemos.consolidation.phases as phases
        monkeypatch.setenv("MNEMOS_NAMESPACE", "t")
        monkeypatch.setattr(phases, "fastembed_embed",
                            lambda texts, prefix="passage": [_vec()])
        store = _store(tmp_path)
        conn = store._get_conn()
        cid = phases.store_bridge_insight(
            conn, 1, 2, "these two connect via a shared mechanism")
        assert cid is not None
        row = conn.execute(
            "SELECT COUNT(*) FROM embed_meta WHERE source_db='memory' "
            "AND source_id=?", (cid,)
        ).fetchone()[0]
        assert row == 1, (
            "weave bridges stored active but never embedded are FTS-only "
            "forever: embed at creation like every other active memory"
        )


class TestM2EmbedFill:
    def test_store_result_warns_on_embed_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        res = m.store_memory("dev", "no vector here", skip_dedup=True)
        assert res["embedded"] is False
        assert "embed-fill" in res.get("warning", ""), (
            "a silent FTS-only store is exactly what the audit flagged; the "
            "caller must be told and pointed at the repair"
        )

    def test_embed_fill_backfills_missing_active_vectors(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        mid = m.store_memory("dev", "vectorless orphan", skip_dedup=True)["id"]
        assert m.embed_status()["missing"] == 1
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        res = m.embed_fill()
        assert res["filled"] == 1
        assert m.embed_status()["missing"] == 0
        row = m.store._get_conn().execute(
            "SELECT text_hash FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()
        assert row["text_hash"]

    def test_embed_fill_dry_run_does_not_mutate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        m.store_memory("dev", "vectorless again", skip_dedup=True)
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        res = m.embed_fill(dry_run=True)
        assert res["missing"] == 1
        assert m.embed_status()["missing"] == 1


class TestM1ModelProvenance:
    def test_store_embedding_records_model(self, tmp_path):
        from mnemos.constants import FASTEMBED_MODEL
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="model tracked"),
            embedding=_vec(), text_hash="h",
        )
        row = store._get_conn().execute(
            "SELECT model FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()
        assert row["model"] == FASTEMBED_MODEL

    def test_archive_move_preserves_model(self, tmp_path):
        from mnemos.constants import FASTEMBED_MODEL
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="model travels"),
            embedding=_vec(), text_hash="h",
        )
        assert store.move_embedding_to_archive(mid)
        row = store._get_conn().execute(
            "SELECT model FROM embed_meta_arch WHERE source_id=?", (mid,)
        ).fetchone()
        assert row["model"] == FASTEMBED_MODEL

    def test_doctor_flags_foreign_model_vectors(self, tmp_path):
        m = _mnemos(tmp_path)
        m.store.store_memory(
            Memory(namespace="t", project="dev", content="foreign vector"),
            embedding=_vec(), text_hash="h",
        )
        conn = m.store._get_conn()
        conn.execute("UPDATE embed_meta SET model='some-other-model-v9'")
        conn.commit()
        report = m.doctor(migrate=False)
        assert any("some-other-model-v9" in i for i in report["issues"]), (
            "doctor must flag vectors embedded by a different model: same-dim "
            "swaps silently corrupt every KNN comparison"
        )


class TestM5L6TemporalValidity:
    def test_future_valid_from_excluded_when_valid_only(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        m.store_memory("dev", "futureword fact", valid_from="2999-01-01",
                       skip_dedup=True)
        res = m.search("futureword", search_mode="fts", valid_only=True,
                       auto_widen=False)
        assert res["results"] == [], (
            "a not-yet-valid memory must not pass valid_only=True"
        )

    def test_expired_today_excluded_when_valid_only(
            self, tmp_path, monkeypatch):
        from datetime import date
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        m.store_memory("dev", "expiredword fact",
                       valid_until=date.today().isoformat(), skip_dedup=True)
        res = m.search("expiredword", search_mode="fts", valid_only=True,
                       auto_widen=False)
        assert res["results"] == [], (
            "valid_until set to today by Phase-4 EVOLVED means expired NOW, "
            "not tomorrow"
        )

    def test_currently_valid_window_included(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        m = _mnemos(tmp_path)
        mid = m.store_memory("dev", "activeword fact",
                             valid_from="2020-01-01", valid_until="2999-01-01",
                             skip_dedup=True)["id"]
        res = m.search("activeword", search_mode="fts", valid_only=True,
                       auto_widen=False)
        assert [r["id"] for r in res["results"]] == [mid]


class TestM9SplitChildrenAtomic:
    def _packed_content(self):
        packed = "F:alpha fact; D:beta decision"
        filler = "\n".join(
            f"L:filler line {i} with enough padding text to add up quickly"
            for i in range(120)
        )
        return packed + "\n" + filler

    def test_remediate_children_get_per_line_explode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = _mnemos(tmp_path)
        content = self._packed_content()
        assert len(content) > 4000
        m.store.store_memory(
            Memory(namespace="t", project="dev", content=content))
        res = m.remediate_oversized(min_size=4000)
        assert res["split"] == 1
        rows = m.store._get_conn().execute(
            "SELECT content FROM memories WHERE tags LIKE '%split-from:%'"
        ).fetchall()
        joined = "\n".join(r["content"] for r in rows)
        assert "F:alpha fact; D:beta decision" not in joined, (
            "remediate children must not inherit packed multi-statement lines"
        )
        assert "F:alpha fact" in joined
        assert "D:beta decision" in joined

    def test_live_store_split_children_get_per_line_explode(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = _mnemos(tmp_path)
        res = m.store_memory("dev", self._packed_content(), skip_dedup=True)
        assert res.get("status") == "stored-split"
        rows = m.store._get_conn().execute(
            "SELECT content FROM memories").fetchall()
        joined = "\n".join(r["content"] for r in rows)
        assert "F:alpha fact; D:beta decision" not in joined
        assert "F:alpha fact" in joined
        assert "D:beta decision" in joined
