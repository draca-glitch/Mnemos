"""Regression tests for v10.10.0: WAL-safe DB backup + doctor integrity check.

Root cause of the 2026-06-27 prod corruption: the live WAL-mode memory.db was
file-copied (shutil.copy2 in doctor, plus an operator `cp` cascade) without
checkpointing the -wal/-shm, so a restored snapshot replayed mismatched WAL
frames and the memories btree corrupted (btreeInitPage error 11, rowids out of
order, concentrated in the newest pages). These tests pin the fix:

  - SQLiteStore.backup() makes a consistent STANDALONE snapshot (VACUUM INTO)
    that includes committed-but-WAL-resident rows regardless of checkpoint
    state, needing no sidecar -wal/-shm to read.
  - doctor() runs an integrity check so this corruption class is actually
    detected (the prior doctor never checked, so search just blew up).
  - doctor()'s pre-migration backup uses that safe primitive, not a raw copy.
"""
import os
import sqlite3

import mnemos.core as core_mod
from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _fake_embed(texts, prefix="passage"):
    return [[0.001] * DIMS for _ in texts]


def _m(tmp_path, monkeypatch):
    monkeypatch.setattr(core_mod, "embed", _fake_embed)
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    # Mirror prod (memory.db is persistent-WAL) and pin committed rows in the
    # WAL with no auto-checkpoint into the main file, so an unsafe .db-only copy
    # provably loses them and the WAL-safety of backup() is tested
    # deterministically rather than by luck of checkpoint timing.
    conn = store._get_conn()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    return Mnemos(store=store, namespace="t", enable_rerank=False)


class TestWalSafeBackup:
    def test_backup_includes_uncheckpointed_wal_rows(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        for i in range(20):
            m.store_memory(project="dev", content=f"F: fact number {i}", skip_dedup=True)
        live = m.store._get_conn().execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

        dest = str(tmp_path / "backup.db")
        m.store.backup(dest)

        # The snapshot must stand alone: a naive copy of just the .db file would
        # be missing the committed rows still resident in the un-checkpointed
        # WAL, and would need the matching -wal/-shm to be consistent.
        assert not os.path.exists(dest + "-wal")
        snap = sqlite3.connect(dest)
        try:
            assert snap.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert snap.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == live
        finally:
            snap.close()

    def test_backup_returns_destination_path(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: one", skip_dedup=True)
        dest = str(tmp_path / "out.db")
        assert m.store.backup(dest) == dest


class TestDoctorIntegrityCheck:
    def test_doctor_runs_integrity_check_on_clean_db(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: alpha", skip_dedup=True)
        report = m.doctor(migrate=False)
        joined = " ".join(report["checks"]).lower()
        assert "integrity" in joined
        assert report["status"] == "healthy"

    def test_doctor_premigration_backup_is_consistent(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: beta", skip_dedup=True)
        live = m.store._get_conn().execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

        # Simulate pointing Mnemos at an old DB missing an aux table, and stop
        # init_schema from auto-healing it so the doctor backup branch runs.
        conn = m.store._get_conn()
        conn.execute("DROP TABLE IF EXISTS tool_usage")
        conn.commit()
        monkeypatch.setattr(type(m.store), "init_schema", lambda self: None)

        report = m.doctor(migrate=True)
        assert report.get("backup") and os.path.exists(report["backup"])
        snap = sqlite3.connect(report["backup"])
        try:
            assert snap.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert snap.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == live
        finally:
            snap.close()
