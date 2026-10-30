"""v10.43.0: citation feedback into the retrieval log and the report on it.

The only usefulness signal before this release was memory_get shortly after
a search. Agents that answer from the search snippet never call get, so the
Epsilon store had 363 verdicts on 11 902 logged retrievals after seven
months. mark_cited takes (query, memory_id) pairs from a harness hook that
scans the reply for cited ids; retrieval_report turns the judged events into
recall@k and MRR.
"""

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore


def _store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")


def _mnemos(store, log=True):
    return Mnemos(store=store, namespace="t", enable_retrieval_log=log,
                  enable_contradiction_detection=False, enable_rerank=False)


def _seed_event(conn, query, ids, at="2026-10-30 12:00:00"):
    conn.executemany(
        "INSERT INTO retrieval_log (memory_id, query, retrieved_at) VALUES (?, ?, ?)",
        [(mid, query, at) for mid in ids])
    conn.commit()


class TestMarkCited:
    def test_marks_matching_pair_only(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        conn = store._get_conn()
        _seed_event(conn, "wheel width", [11, 12, 13])
        out = m.mark_cited([("wheel width", 12)], since="2026-10-30 00:00:00")
        assert out == {"cited": 1, "uncited": 0}
        rows = dict(conn.execute(
            "SELECT memory_id, useful FROM retrieval_log").fetchall())
        assert rows == {11: None, 12: 1, 13: None}

    def test_uncited_zero_judges_the_rest_of_the_query(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        conn = store._get_conn()
        _seed_event(conn, "wheel width", [11, 12, 13])
        _seed_event(conn, "other query", [21])
        out = m.mark_cited([("wheel width", 12)], since="2026-10-30 00:00:00",
                           uncited_zero=True)
        assert out == {"cited": 1, "uncited": 2}
        rows = dict(conn.execute(
            "SELECT memory_id, useful FROM retrieval_log").fetchall())
        assert rows == {11: 0, 12: 1, 13: 0, 21: None}

    def test_since_bounds_the_rows_touched(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        conn = store._get_conn()
        _seed_event(conn, "q", [1], at="2026-01-01 00:00:00")
        _seed_event(conn, "q", [1], at="2026-10-30 12:00:00")
        out = m.mark_cited([("q", 1)], since="2026-10-30 00:00:00")
        assert out["cited"] == 1
        olds = conn.execute(
            "SELECT useful FROM retrieval_log WHERE retrieved_at < '2026-06-01'").fetchone()[0]
        assert olds is None

    def test_a_later_citation_overrides_zero(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        conn = store._get_conn()
        _seed_event(conn, "q", [1, 2])
        m.mark_cited([("q", 1)], since="2026-10-30 00:00:00", uncited_zero=True)
        m.mark_cited([("q", 2)], since="2026-10-30 00:00:00")
        rows = dict(conn.execute(
            "SELECT memory_id, useful FROM retrieval_log").fetchall())
        assert rows == {1: 1, 2: 1}

    def test_inert_when_logging_off(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store, log=False)
        conn = store._get_conn()
        _seed_event(conn, "q", [1])
        out = m.mark_cited([("q", 1)], since="2026-10-30 00:00:00")
        assert out["retrieval_log"] == "off"
        assert conn.execute("SELECT useful FROM retrieval_log").fetchone()[0] is None


class TestRetrievalReport:
    def test_recall_and_mrr_over_cited_events(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        conn = store._get_conn()
        # event A: hit at rank 1; event B: hit at rank 4; event C: judged, no hit;
        # event D: never judged
        _seed_event(conn, "a", [1, 2, 3, 4, 5], at="2026-10-30 10:00:00")
        _seed_event(conn, "b", [6, 7, 8, 9, 10], at="2026-10-30 10:01:00")
        _seed_event(conn, "c", [11, 12], at="2026-10-30 10:02:00")
        _seed_event(conn, "d", [13], at="2026-10-30 10:03:00")
        m.mark_cited([("a", 1), ("b", 9)], since="2026-10-30 00:00:00", uncited_zero=True)
        store.mark_retrievals_uncited(["c"], since="2026-10-30 00:00:00")
        r = m.retrieval_report(days=365)
        assert r["events"] == 4
        assert r["judged_events"] == 3
        assert r["cited_events"] == 2
        assert r["surfaced_not_cited_events"] == 1
        assert r["rows"] == 13
        assert r["recall_at_1"] == 0.5
        assert r["recall_at_3"] == 0.5
        assert r["recall_at_5"] == 1.0
        assert r["recall_at_10"] == 1.0
        assert r["mrr"] == round((1.0 + 0.25) / 2, 4)

    def test_report_without_verdicts_has_no_recall_keys(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        _seed_event(store._get_conn(), "a", [1, 2])
        r = m.retrieval_report(days=365)
        assert r["events"] == 1 and r["cited_events"] == 0
        assert "mrr" not in r

    def test_rank_is_insertion_order_within_an_event(self, tmp_path):
        store = _store(tmp_path)
        m = _mnemos(store)
        conn = store._get_conn()
        # the same memory id can appear in two events; rank is per event
        _seed_event(conn, "a", [5, 1], at="2026-10-30 10:00:00")
        _seed_event(conn, "b", [1, 5], at="2026-10-30 10:05:00")
        m.mark_cited([("a", 1), ("b", 1)], since="2026-10-30 00:00:00")
        r = m.retrieval_report(days=365)
        assert r["recall_at_1"] == 0.5
        assert r["mrr"] == round((0.5 + 1.0) / 2, 4)
