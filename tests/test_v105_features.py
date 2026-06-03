"""Regression tests for v10.5.1: Phase 6 bookkeeping respects execute.

The Nyx Phase 6 mutators (decay_access_counts, cleanup_stale_links,
cleanup_orphan_vectors) previously committed unconditionally. A dry run
therefore decayed access counts and demoted importance on the live store
while log_consolidation_run (correctly execute-gated) recorded nothing,
leaving the store mutated but the run unlogged. These guard that a dry
run reports would-be counts without mutating, and that execute mutates.
"""

import sqlite3

from mnemos.consolidation.orchestrator import decay_access_counts, cleanup_stale_links


def _decay_conn(access_count=4, importance=8):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, status TEXT, "
        "access_count INTEGER, importance INTEGER, last_accessed TEXT)"
    )
    conn.execute(
        "INSERT INTO memories (status, access_count, importance, last_accessed) "
        "VALUES ('active', ?, ?, datetime('now','-30 days'))",
        (access_count, importance),
    )
    conn.commit()
    mid = conn.execute("SELECT id FROM memories").fetchone()[0]
    return conn, mid


class TestDecayRespectsExecute:
    def test_dry_run_reports_but_does_not_mutate(self):
        conn, mid = _decay_conn(access_count=4, importance=8)
        decayed, demoted = decay_access_counts(conn, execute=False)
        assert decayed == 1  # would-be count is reported
        row = conn.execute("SELECT access_count, importance FROM memories WHERE id=?", (mid,)).fetchone()
        assert row == (4, 8)  # nothing written

    def test_execute_decays(self):
        conn, mid = _decay_conn(access_count=4, importance=5)
        decayed, _ = decay_access_counts(conn, execute=True)
        assert decayed == 1
        assert conn.execute("SELECT access_count FROM memories WHERE id=?", (mid,)).fetchone()[0] == 3

    def test_nothing_eligible_is_zero(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY, status TEXT, "
            "access_count INTEGER, importance INTEGER, last_accessed TEXT)"
        )
        # accessed today -> not eligible
        conn.execute(
            "INSERT INTO memories (status, access_count, importance, last_accessed) "
            "VALUES ('active', 4, 5, datetime('now'))"
        )
        conn.commit()
        assert decay_access_counts(conn, execute=True) == (0, 0)


class TestStaleLinksRespectExecute:
    def _stale_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE memory_links (source_id INTEGER, target_id INTEGER)")
        conn.execute("INSERT INTO memories (id, status) VALUES (1, 'active')")
        conn.execute("INSERT INTO memory_links (source_id, target_id) VALUES (1, 2)")  # target 2 absent -> stale
        conn.commit()
        return conn

    def test_dry_run_counts_without_deleting(self):
        conn = self._stale_conn()
        assert cleanup_stale_links(conn, execute=False) == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0] == 1

    def test_execute_deletes(self):
        conn = self._stale_conn()
        assert cleanup_stale_links(conn, execute=True) == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0] == 0
