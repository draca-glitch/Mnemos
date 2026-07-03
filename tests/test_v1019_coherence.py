"""Tests for v10.19.0: doctor content/vector coherence check.

embed_meta.text_hash has recorded what every embedding was computed from
since v10.6; doctor now re-derives it from current content and flags
divergence (content mutated without a re-embed, e.g. direct SQL writes
bypassing the API). No embedder model is loaded: hashes are plain digests
and the migrate path uses a stubbed embed().
"""

import mnemos.core as core_mod
from mnemos.core import Mnemos
from mnemos.embed import prep_memory_text, text_hash
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _mnemos(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t",
                  enable_contradiction_detection=False, enable_rerank=False)


def _seed_coherent(m, mid=1, content="F: the disk is 512GB"):
    conn = m.store._get_conn()
    conn.execute(
        "INSERT INTO memories (id, namespace, project, content, tags, type, "
        "layer, status) VALUES (?, 't', 'dev', ?, '', 'fact', 'semantic', "
        "'active')", (mid, content))
    thash = text_hash(prep_memory_text(
        "dev", content, "", mem_type="fact", layer="semantic"))
    conn.execute(
        "INSERT INTO embed_meta (source_db, source_id, text_hash, model) "
        "VALUES ('memory', ?, ?, 'test-model')", (mid, thash))
    conn.commit()
    return conn


class TestCoherenceCheck:
    def test_coherent_store_passes(self, tmp_path):
        m = _mnemos(tmp_path)
        _seed_coherent(m)
        report = m.doctor()
        assert any("coherence" in c and "verified" in c
                   for c in report["checks"])
        assert not any("coherence" in i.lower() for i in report["issues"])

    def test_tampered_content_is_flagged(self, tmp_path):
        m = _mnemos(tmp_path)
        conn = _seed_coherent(m)
        conn.execute(
            "UPDATE memories SET content = 'F: the disk is 9TB (tampered)' "
            "WHERE id = 1")
        conn.commit()
        report = m.doctor()
        flagged = [i for i in report["issues"] if "coherence" in i]
        assert flagged and "[1]" in flagged[0]

    def test_migrate_reembeds_and_heals(self, tmp_path, monkeypatch):
        m = _mnemos(tmp_path)
        conn = _seed_coherent(m)
        conn.execute(
            "UPDATE memories SET content = 'F: the disk is 9TB (tampered)' "
            "WHERE id = 1")
        conn.commit()
        monkeypatch.setattr(core_mod, "embed",
                            lambda texts, prefix="passage":
                            [[0.001] * DIMS for _ in texts])
        report = m.doctor(migrate=True)
        assert any("re-embedded 1/1" in a
                   for a in report["migrations_applied"])
        clean = m.doctor()
        assert not any("coherence" in i for i in clean["issues"])

    def test_pre_tracking_rows_are_skipped(self, tmp_path):
        m = _mnemos(tmp_path)
        conn = m.store._get_conn()
        conn.execute(
            "INSERT INTO memories (id, namespace, project, content, type, "
            "layer, status) VALUES (2, 't', 'dev', 'F: old row', 'fact', "
            "'semantic', 'active')")
        conn.execute(
            "INSERT INTO embed_meta (source_db, source_id, text_hash, model) "
            "VALUES ('memory', 2, NULL, NULL)")
        conn.commit()
        report = m.doctor()
        assert not any("coherence" in i for i in report["issues"])
