"""
Archive-side embedding lifecycle (v10.24.0).

Forensic audit 2026-07-07 found the tier-2 index leaks: consolidation
archived memories without moving their vectors (51 gaps in prod, clustered
on Nyx days), hard delete left arch rows behind (3 orphans), embed_meta
tables created before v10.6 carry a UTC embedded_at default while the
memories table uses localtime, and the arch vec table never got the
declared-PK schema the active side already handles.

These tests pin the fixed lifecycle: every archive path moves the vector,
hard delete purges both indexes, reindex-archived re-embeds legacy-hash
rows, and doctor detects plus migrates the two schema-era drifts.
"""
import os
import sqlite3
import tempfile

import pytest


@pytest.fixture
def mnemos_with_tmpdb():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from mnemos.core import Mnemos
    from mnemos.storage.sqlite_store import SQLiteStore
    m = Mnemos(store=SQLiteStore(db_path=path))
    yield m
    m.close()
    for suffix in ("", "-journal", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)


def _active_has(store, mid):
    return store._get_conn().execute(
        "SELECT 1 FROM embed_meta WHERE source_db='memory' AND source_id=?", (mid,)
    ).fetchone() is not None


def _arch_has(store, mid):
    return store._get_conn().execute(
        "SELECT 1 FROM embed_meta_arch WHERE source_db='memory' AND source_id=?", (mid,)
    ).fetchone() is not None


def _arch_vec_count(store):
    conn = store._get_conn()
    from mnemos.storage.sqlite_store import _arch_join_col
    col = _arch_join_col(conn)
    return conn.execute(f"SELECT COUNT({col}) FROM embed_vec_arch").fetchone()[0]


def test_new_store_arch_vec_has_declared_pk(mnemos_with_tmpdb):
    conn = mnemos_with_tmpdb.store._get_conn()
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='embed_vec_arch'"
    ).fetchone()[0]
    assert "id INTEGER PRIMARY KEY" in sql


def test_soft_delete_moves_vector_to_arch(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: soft delete moves the vector")["id"]
    assert _active_has(m.store, mid)
    m.store.delete_memory(mid, hard=False)
    assert not _active_has(m.store, mid)
    assert _arch_has(m.store, mid)
    assert _arch_vec_count(m.store) == 1


def test_hard_delete_purges_arch_rows(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: hard delete purges everything")["id"]
    m.store.delete_memory(mid, hard=False)          # now lives in arch index
    assert _arch_has(m.store, mid)
    m.store.delete_memory(mid, hard=True)
    assert not _arch_has(m.store, mid)
    assert _arch_vec_count(m.store) == 0
    conn = m.store._get_conn()
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE id=?", (mid,)).fetchone()[0] == 0


def test_archive_memory_helper_moves_vector(mnemos_with_tmpdb):
    from mnemos.consolidation.phases import archive_memory
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: helper archives and moves")["id"]
    conn = m.store._get_conn()
    archive_memory(conn, mid, tag_suffix="merged-into-999")
    row = conn.execute("SELECT status, tags FROM memories WHERE id=?", (mid,)).fetchone()
    assert row[0] == "archived"
    assert "merged-into-999" in row[1]
    assert not _active_has(m.store, mid)
    assert _arch_has(m.store, mid)


def test_archived_index_roundtrip_on_declared_pk_schema(mnemos_with_tmpdb):
    """search_vec_archived must work against the new declared-PK arch table."""
    from mnemos.embed import embed
    m = mnemos_with_tmpdb
    mid = m.store_memory(
        project="dev", content="F: the Eiffel Tower is in Paris, built 1889")["id"]
    m.store.delete_memory(mid, hard=False)
    q = embed(["Eiffel Tower Paris"], prefix="search_query")[0]
    hits = m.store.search_vec_archived(q, namespace=m.namespace, limit=5)
    assert any(sid == mid for sid, _dist in hits)


def test_reindex_archived_reembeds_legacy_hashes(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: legacy hash row gets re-embedded")["id"]
    m.store.delete_memory(mid, hard=False)
    conn = m.store._get_conn()
    conn.execute(
        "UPDATE embed_meta_arch SET text_hash=substr(text_hash,1,16) WHERE source_id=?",
        (mid,))
    conn.commit()
    report = m.reindex_archived()
    assert report["legacy_reembedded"] == 1
    thash = conn.execute(
        "SELECT text_hash FROM embed_meta_arch WHERE source_id=?", (mid,)).fetchone()[0]
    assert len(thash) == 64


def test_doctor_detects_and_migrates_utc_embed_timestamps(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: timestamps unify to localtime")["id"]
    conn = m.store._get_conn()
    # Recreate embed_meta with the pre-v10.6-era UTC default, preserving rows,
    # to simulate a store whose table predates the localtime DDL.
    conn.executescript("""
        CREATE TABLE embed_meta_old AS SELECT * FROM embed_meta;
        DROP TABLE embed_meta;
        CREATE TABLE embed_meta (
            id INTEGER PRIMARY KEY,
            source_db TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            text_hash TEXT,
            model TEXT,
            embedded_at TEXT DEFAULT (datetime('now')),
            UNIQUE(source_db, source_id)
        );
        INSERT INTO embed_meta SELECT id, source_db, source_id, text_hash, model,
            datetime('now') FROM embed_meta_old;
        DROP TABLE embed_meta_old;
    """)
    conn.commit()
    utc_before = conn.execute(
        "SELECT embedded_at FROM embed_meta WHERE source_id=?", (mid,)).fetchone()[0]
    expected_local = conn.execute(
        "SELECT datetime(?, 'localtime')", (utc_before,)).fetchone()[0]

    report = m.doctor()
    assert any("UTC" in i for i in report["issues"])

    report = m.doctor(migrate=True)
    assert any("embedded_at" in mig for mig in report["migrations_applied"])
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='embed_meta'").fetchone()[0]
    assert "localtime" in sql
    after = conn.execute(
        "SELECT embedded_at FROM embed_meta WHERE source_id=?", (mid,)).fetchone()[0]
    assert after == expected_local
    # idempotent: second run reports nothing to do
    report = m.doctor()
    assert not any("UTC" in i for i in report["issues"])


def test_doctor_cleans_arch_orphans(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: orphan arch rows get cleaned")["id"]
    m.store.delete_memory(mid, hard=False)
    conn = m.store._get_conn()
    # Simulate the pre-v10.24 hard-delete leak: memory row gone, arch rows behind
    conn.execute("DELETE FROM memories WHERE id=?", (mid,))
    conn.commit()
    report = m.doctor()
    assert any("orphan" in i.lower() and "arch" in i.lower() for i in report["issues"])
    report = m.doctor(migrate=True)
    assert any("orphan" in mig.lower() for mig in report["migrations_applied"])
    assert not _arch_has(m.store, mid)
    assert _arch_vec_count(m.store) == 0


def test_doctor_reports_archived_without_arch_vector(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = m.store_memory(project="dev", content="F: gap is reported not auto-embedded")["id"]
    conn = m.store._get_conn()
    # Archive the raw way (the old leak): status flips, no vector move
    conn.execute("UPDATE memories SET status='archived' WHERE id=?", (mid,))
    # Doctor never auto-embeds; the leak shows as advisory pointing at reindex-archived
    meta_id = conn.execute(
        "SELECT id FROM embed_meta WHERE source_id=?", (mid,)).fetchone()[0]
    col = m.store._get_vec_join_col()
    conn.execute(f"DELETE FROM embed_vec WHERE {col}=?", (meta_id,))
    conn.execute("DELETE FROM embed_meta WHERE source_id=?", (mid,))
    conn.commit()
    report = m.doctor()
    assert any("reindex-archived" in i for i in report["issues"])
