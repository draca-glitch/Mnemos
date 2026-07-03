"""Tests for v10.17.0: the zero-LLM daily consolidation cycle.

Covers mutual top-k candidacy, the NLI cluster admission gate, the
mechanical line-union merge, phase-4 judge queue mode, weave hygiene
(staleness guard, novelty gate, episodic bridges) and the retrieval-log
useful-loop. Evidence base: benchmarks/weave-bench and
benchmarks/merge-bench (2026-07-03).

Model-dependent scoring is faked via monkeypatch; no NLI checkpoint or
embedder is used by this suite.
"""

import sqlite3

import mnemos.nli as nli
import mnemos.consolidation.phases as phases
import mnemos.consolidation.mechanical as mechanical
from mnemos.consolidation.mechanical import mechanical_merge_cluster
from mnemos.consolidation.phases import (
    mutual_topk_adjacency, nli_cluster_gate, find_clusters, phase_contradict,
)
from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _store(tmp_path, name="m.db"):
    return SQLiteStore(db_path=str(tmp_path / name), namespace="t")


class TestMutualTopkAdjacency:
    def test_mutual_pair_is_adjacent(self):
        sim = [[1.0, 0.9, 0.2],
               [0.9, 1.0, 0.3],
               [0.2, 0.3, 1.0]]
        adj = mutual_topk_adjacency(sim, k=1)
        assert adj[0][1] and adj[1][0]

    def test_one_sided_neighbor_is_not_adjacent(self):
        # 2's nearest is 1, but 1's top-1 is 0: not mutual at k=1.
        sim = [[1.0, 0.9, 0.1],
               [0.9, 1.0, 0.5],
               [0.1, 0.5, 1.0]]
        adj = mutual_topk_adjacency(sim, k=1)
        assert not adj[1][2] and not adj[2][1]

    def test_find_clusters_mutual_topk_ignores_absolute_threshold(self):
        # High absolute similarity everywhere; only the mutual top-1 pair
        # clusters, the rest of the compressed-space noise does not.
        mem = {1: {"project": "p"}, 2: {"project": "p"}, 3: {"project": "p"},
               4: {"project": "p"}}
        sim = [[1.0, 0.97, 0.80, 0.80],
               [0.97, 1.0, 0.80, 0.80],
               [0.80, 0.80, 1.0, 0.78],
               [0.80, 0.80, 0.78, 1.0]]
        clusters = find_clusters([1, 2, 3, 4], sim, 0.75, mem,
                                 candidacy="mutual-topk", top_k=1)
        assert clusters == [[1, 2]]

    def test_find_clusters_threshold_mode_is_legacy(self):
        mem = {1: {"project": "p"}, 2: {"project": "p"}}
        sim = [[1.0, 0.9], [0.9, 1.0]]
        clusters = find_clusters([1, 2], sim, 0.88, mem,
                                 candidacy="threshold")
        assert clusters == [[1, 2]]


class TestClusterGate:
    def _mem(self):
        return {
            1: {"content": "F: alpha is 5\nF: beta is 6"},
            2: {"content": "F: alpha is 5 (confirmed)\nF: gamma is 7"},
            3: {"content": "F: totally unrelated topic"},
        }

    def test_gate_ejects_member_without_shared_fact(self, monkeypatch):
        monkeypatch.setattr(nli, "is_available", lambda: True)

        def fake_dup(a, b, top_k=8):
            return 0.95 if ("alpha" in a and "alpha" in b) else 0.2

        monkeypatch.setattr(nli, "line_max_duplicate", fake_dup)
        admitted = nli_cluster_gate([1, 2, 3], self._mem())
        assert admitted == [1, 2]

    def test_gate_dissolves_noise_cluster(self, monkeypatch):
        monkeypatch.setattr(nli, "is_available", lambda: True)
        monkeypatch.setattr(nli, "line_max_duplicate",
                            lambda a, b, top_k=8: 0.3)
        assert nli_cluster_gate([1, 2, 3], self._mem()) == []

    def test_gate_off_passes_through(self, monkeypatch):
        monkeypatch.setenv("MNEMOS_CLUSTER_GATE", "off")
        assert nli_cluster_gate([1, 2, 3], self._mem()) == [1, 2, 3]

    def test_gate_without_nli_passes_through(self, monkeypatch):
        monkeypatch.delenv("MNEMOS_CLUSTER_GATE", raising=False)
        monkeypatch.setattr(nli, "is_available", lambda: False)
        assert nli_cluster_gate([1, 2, 3], self._mem()) == [1, 2, 3]


class TestMechanicalMerge:
    def _mem(self, a_content, b_content):
        return {
            1: {"id": 1, "created_at": "2026-01-01", "content": a_content},
            2: {"id": 2, "created_at": "2026-02-01", "content": b_content},
        }

    def test_exact_duplicate_collapses_without_nli_scoring(self, monkeypatch):
        monkeypatch.setattr(mechanical.nli, "is_available", lambda: True)
        calls = []

        def fake_bident(a, b):
            calls.append((a, b))
            return 0.0

        monkeypatch.setattr(mechanical.nli, "bidirectional_entailment",
                            fake_bident)
        mem = self._mem("F: the server has 64GB of memory installed",
                        "F: the server has 64GB of memory installed\n"
                        "F: the backup runs nightly at 03:45 via cron")
        merged = mechanical_merge_cluster([1, 2], mem)
        assert merged.count("64GB") == 1
        assert "03:45" in merged

    def test_paraphrase_duplicate_keeps_newer_phrasing(self, monkeypatch):
        monkeypatch.setattr(mechanical.nli, "is_available", lambda: True)
        monkeypatch.setattr(mechanical.nli, "bidirectional_entailment",
                            lambda a, b: 0.95)
        mem = self._mem("F: the old phrasing of the shared server fact",
                        "F: the newer phrasing of the shared server fact")
        merged = mechanical_merge_cluster([1, 2], mem)
        assert merged == "F: the newer phrasing of the shared server fact"

    def test_below_tau_keeps_both_lines(self, monkeypatch):
        monkeypatch.setattr(mechanical.nli, "is_available", lambda: True)
        monkeypatch.setattr(mechanical.nli, "bidirectional_entailment",
                            lambda a, b: 0.85)
        mem = self._mem("F: route follow-up through the cataract surgeon",
                        "F: aggressive eye procedures are a hard call here")
        merged = mechanical_merge_cluster([1, 2], mem)
        assert "cataract" in merged and "aggressive" in merged

    def test_short_lines_dedup_by_exact_match_only(self, monkeypatch):
        monkeypatch.setattr(mechanical.nli, "is_available", lambda: True)
        monkeypatch.setattr(mechanical.nli, "bidirectional_entailment",
                            lambda a, b: 0.99)
        mem = self._mem("7. Horror (light)", "8. Comedy (focus)")
        merged = mechanical_merge_cluster([1, 2], mem)
        assert "Horror" in merged and "Comedy" in merged

    def test_returns_none_without_nli(self, monkeypatch):
        monkeypatch.setattr(mechanical.nli, "is_available", lambda: False)
        mem = self._mem("F: aaa", "F: bbb")
        assert mechanical_merge_cluster([1, 2], mem) is None


def _phase4_fixture(tmp_path):
    """Small store with two same-project fact memories in the finder band."""
    store = _store(tmp_path)
    conn = store._get_conn()
    for mid, content in ((1, "F: the disk is 512GB"),
                         (2, "F: the disk is 256GB")):
        conn.execute(
            "INSERT INTO memories (id, namespace, project, content, type, "
            "status, created_at) VALUES (?, 't', 'server', ?, 'fact', "
            "'active', ?)",
            (mid, content, f"2026-0{mid}-01 00:00:00"),
        )
    conn.commit()
    mem_by_id = {
        r["id"]: dict(r) for r in conn.execute(
            "SELECT * FROM memories").fetchall()
    }
    # Two vectors at ~0.7 cosine: inside [CONTRADICT_MIN_SIM, MAX_SIM].
    v1 = [1.0] + [0.0] * (DIMS - 1)
    v2 = [0.7, 0.7141428] + [0.0] * (DIMS - 2)
    embeddings = {1: __import__("numpy").array(v1, dtype="float32"),
                  2: __import__("numpy").array(v2, dtype="float32")}
    return store, conn, mem_by_id, embeddings


class TestJudgeQueue:
    def test_queue_mode_records_candidate_links(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOS_NYX_CONTRADICT_FINDER", raising=False)
        store, conn, mem_by_id, embeddings = _phase4_fixture(tmp_path)
        stats = phase_contradict(conn, embeddings, mem_by_id,
                                 is_surge=False, execute=True, judge="queue")
        assert stats["queued"] == 1
        row = conn.execute(
            "SELECT source_id, target_id FROM memory_links "
            "WHERE relation_type='contradiction-candidate'").fetchone()
        assert row is not None

    def test_compatible_verdict_is_remembered(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOS_NYX_CONTRADICT_FINDER", raising=False)
        store, conn, mem_by_id, embeddings = _phase4_fixture(tmp_path)
        monkeypatch.setattr(
            phases, "opus_chat",
            lambda *a, **k: "CLASSIFICATION: COMPATIBLE\nEXPLANATION: fine")
        first = phase_contradict(conn, embeddings, mem_by_id,
                                 is_surge=False, execute=True, judge="llm")
        assert first["compatible"] == 1
        cleared = conn.execute(
            "SELECT COUNT(*) FROM memory_links "
            "WHERE relation_type='contradiction-cleared'").fetchone()[0]
        assert cleared == 1
        second = phase_contradict(conn, embeddings, mem_by_id,
                                  is_surge=False, execute=True, judge="llm")
        assert second["candidates"] == 0

    def test_queue_mode_skips_cleared_pairs(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOS_NYX_CONTRADICT_FINDER", raising=False)
        store, conn, mem_by_id, embeddings = _phase4_fixture(tmp_path)
        conn.execute(
            "INSERT INTO memory_links (source_id, target_id, relation_type, "
            "strength) VALUES (1, 2, 'contradiction-cleared', 0.1)")
        conn.commit()
        stats = phase_contradict(conn, embeddings, mem_by_id,
                                 is_surge=False, execute=True, judge="queue")
        assert stats["queued"] == 0

    def test_llm_mode_consumes_queued_candidates(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOS_NYX_CONTRADICT_FINDER", raising=False)
        store, conn, mem_by_id, embeddings = _phase4_fixture(tmp_path)
        conn.execute(
            "INSERT INTO memory_links (source_id, target_id, relation_type, "
            "strength) VALUES (1, 2, 'contradiction-candidate', 0.7)")
        conn.commit()
        monkeypatch.setattr(
            phases, "opus_chat",
            lambda *a, **k: "CLASSIFICATION: COMPATIBLE\nEXPLANATION: fine")
        stats = phase_contradict(conn, embeddings, mem_by_id,
                                 is_surge=False, execute=True, judge="llm")
        left = conn.execute(
            "SELECT COUNT(*) FROM memory_links "
            "WHERE relation_type='contradiction-candidate'").fetchone()[0]
        assert left == 0
        assert stats["candidates"] >= 1


class TestScanCache:
    """nli_scan_cache memoization for the phase-4 finder (10.18.0)."""

    def _fixture(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOS_NYX_CONTRADICT_FINDER", "nli")
        store, conn, mem_by_id, embeddings = _phase4_fixture(tmp_path)
        from mnemos.consolidation.orchestrator import _migrate_nyx_schema
        _migrate_nyx_schema(conn)
        monkeypatch.setattr(phases.nli, "is_available", lambda: True)
        calls = []

        def fake_score(a, b, top_k=8):
            calls.append((a, b))
            return 0.5  # below threshold: not flagged, not queued

        monkeypatch.setattr(phases.nli, "line_max_contradiction", fake_score)
        return conn, mem_by_id, embeddings, calls

    def test_second_run_scores_nothing(self, tmp_path, monkeypatch):
        conn, mem_by_id, embeddings, calls = self._fixture(tmp_path, monkeypatch)
        phase_contradict(conn, embeddings, mem_by_id, is_surge=False,
                         execute=True, judge="queue")
        assert len(calls) == 1
        phase_contradict(conn, embeddings, mem_by_id, is_surge=False,
                         execute=True, judge="queue")
        assert len(calls) == 1  # cache hit, no new scoring

    def test_content_change_invalidates(self, tmp_path, monkeypatch):
        conn, mem_by_id, embeddings, calls = self._fixture(tmp_path, monkeypatch)
        phase_contradict(conn, embeddings, mem_by_id, is_surge=False,
                         execute=True, judge="queue")
        mem_by_id[1]["content"] = "F: the disk is 1TB now"
        phase_contradict(conn, embeddings, mem_by_id, is_surge=False,
                         execute=True, judge="queue")
        assert len(calls) == 2  # hash mismatch forced a re-score

    def test_dry_run_does_not_write_cache(self, tmp_path, monkeypatch):
        conn, mem_by_id, embeddings, calls = self._fixture(tmp_path, monkeypatch)
        phase_contradict(conn, embeddings, mem_by_id, is_surge=False,
                         execute=False, judge="queue")
        rows = conn.execute("SELECT COUNT(*) FROM nli_scan_cache").fetchone()[0]
        assert rows == 0


class TestWeaveHygiene:
    def _weave_fixture(self, tmp_path):
        store = _store(tmp_path)
        conn = store._get_conn()
        rows = (
            (1, "server", "F: the epsilon server runs the memory system"),
            (2, "dev", "F: the memory system project ships benchmarks"),
            (3, "dev", "F: an unrelated third memory for padding"),
            (4, "food", "F: a fourth memory to clear the size floor"),
        )
        for mid, project, content in rows:
            conn.execute(
                "INSERT INTO memories (id, namespace, project, content, "
                "type, status, created_at) VALUES (?, 't', ?, ?, 'fact', "
                "'active', '2026-01-01 00:00:00')",
                (mid, project, content),
            )
        conn.commit()
        mem_by_id = {
            r["id"]: dict(r) for r in conn.execute(
                "SELECT * FROM memories").fetchall()
        }
        import numpy as np
        base = np.zeros(DIMS, dtype="float32")
        base[0] = 1.0
        near = np.zeros(DIMS, dtype="float32")
        near[0], near[1] = 0.8, 0.6
        far1 = np.zeros(DIMS, dtype="float32")
        far1[2] = 1.0
        far2 = np.zeros(DIMS, dtype="float32")
        far2[3] = 1.0
        embeddings = {1: base, 2: near, 3: far1, 4: far2}
        return store, conn, mem_by_id, embeddings

    def test_staleness_guard_excludes_superseded_sources(self, tmp_path,
                                                         monkeypatch):
        store, conn, mem_by_id, embeddings = self._weave_fixture(tmp_path)
        conn.execute(
            "INSERT INTO memory_links (source_id, target_id, relation_type, "
            "strength) VALUES (1, 3, 'superseded_by', 0.9)")
        conn.commit()
        monkeypatch.setattr(
            phases, "opus_chat",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                "stale source must not reach the LLM")))
        stats = phases.phase_weave(conn, embeddings, mem_by_id,
                                   is_surge=False, execute=True)
        assert stats["pairs_evaluated"] == 0

    def test_novelty_gate_keeps_link_skips_redundant_insight(self, tmp_path,
                                                             monkeypatch):
        store, conn, mem_by_id, embeddings = self._weave_fixture(tmp_path)
        monkeypatch.setattr(
            phases, "opus_chat",
            lambda *a, **k: "LINK_TYPE: reflects\nSTRENGTH: 0.8\n"
                            "INSIGHT: a restatement of the sources")
        monkeypatch.setattr(phases.nli, "is_available", lambda: True)
        monkeypatch.setattr(phases.nli, "p_entailment", lambda p, h: 0.95)
        stats = phases.phase_weave(conn, embeddings, mem_by_id,
                                   is_surge=False, execute=True)
        assert stats["links_created"] >= 1
        assert stats.get("insights_skipped_redundant", 0) >= 1
        assert stats["insights_stored"] == 0

    def test_bridge_insight_lands_on_episodic_layer(self, tmp_path,
                                                    monkeypatch):
        store, conn, mem_by_id, embeddings = self._weave_fixture(tmp_path)
        monkeypatch.setattr(phases, "fastembed_embed",
                            lambda texts: [[0.001] * DIMS for _ in texts])
        cid = phases.store_bridge_insight(conn, 1, 2, "an insight sentence")
        layer = conn.execute(
            "SELECT layer FROM memories WHERE id=?", (cid,)).fetchone()[0]
        assert layer == "episodic"


class TestMemArenaOptOut:
    """MNEMOS_DISABLE_MEM_ARENA (10.17.1, contributed by balaianu/Mnemos,
    extended here to the NLI ONNX sessions)."""

    def test_default_is_no_session_options(self, monkeypatch):
        monkeypatch.setattr(nli, "DISABLE_MEM_ARENA", False)
        assert nli._onnx_session_options() is None

    def test_flag_disables_cpu_mem_arena(self, monkeypatch):
        monkeypatch.setattr(nli, "DISABLE_MEM_ARENA", True)
        so = nli._onnx_session_options()
        assert so is not None
        assert so.enable_cpu_mem_arena is False

    def test_fastembed_still_exposes_the_option(self):
        # The embedder/reranker pass enable_cpu_mem_arena as a fastembed
        # kwarg; this guards the contract across fastembed upgrades.
        from fastembed.common.onnx_model import OnnxModel
        assert "enable_cpu_mem_arena" in OnnxModel.EXPOSED_SESSION_OPTIONS


class TestUsefulLoop:
    def test_get_marks_recent_retrievals_useful(self, tmp_path):
        store = _store(tmp_path)
        m = Mnemos(store=store, namespace="t", enable_retrieval_log=True,
                   enable_contradiction_detection=False, enable_rerank=False)
        conn = store._get_conn()
        conn.execute(
            "INSERT INTO memories (id, namespace, project, content, type, "
            "status) VALUES (7, 't', 'dev', 'F: a memory', 'fact', 'active')")
        conn.execute(
            "INSERT INTO retrieval_log (memory_id, query) VALUES (7, 'q')")
        conn.commit()
        m.get(7)
        useful = conn.execute(
            "SELECT useful FROM retrieval_log WHERE memory_id=7").fetchone()[0]
        assert useful == 1

    def test_get_without_logging_leaves_rows_alone(self, tmp_path):
        store = _store(tmp_path)
        m = Mnemos(store=store, namespace="t", enable_retrieval_log=False,
                   enable_contradiction_detection=False, enable_rerank=False)
        conn = store._get_conn()
        conn.execute(
            "INSERT INTO memories (id, namespace, project, content, type, "
            "status) VALUES (8, 't', 'dev', 'F: a memory', 'fact', 'active')")
        conn.execute(
            "INSERT INTO retrieval_log (memory_id, query) VALUES (8, 'q')")
        conn.commit()
        m.get(8)
        useful = conn.execute(
            "SELECT useful FROM retrieval_log WHERE memory_id=8").fetchone()[0]
        assert useful is None
