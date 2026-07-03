"""
Tier-2 archived recall (v10.7.0).

Archived memories keep their vectors in a separate index (embed_vec_arch) so
expand_merged can vector-search the originals directly, while primary search
stays active-only with no regression.

The shared-fixture DB accumulates across tests, so these assert on the specific
memory (moved / recalled) rather than on absolute index counts.
"""
import os
import tempfile

import pytest


@pytest.fixture
def mnemos_with_tmpdb():
    # Construct the store explicitly. The old env-var route
    # (os.environ["MNEMOS_DB"] = path before Mnemos()) silently broke when
    # any earlier-collected test imported mnemos first: DEFAULT_DB_PATH is
    # frozen at import time, so these tests accumulated into the developer's
    # default DB (observed 2026-07-03: 419 fixture memories in
    # ~/.mnemos/memory.db, and enough archived Eiffel copies to crowd the
    # fresh one out of tier-2 KNN and flake the recall assert).
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


def _seed(m):
    r1 = m.store_memory(
        project="dev",
        content="F: The Eiffel Tower is in Paris, capital of France, built 1889",
    )
    m.store_memory(
        project="dev",
        content="F: Mnemos uses sqlite-vec for 1024-dim e5-large vector storage",
    )
    m.store_memory(
        project="dev",
        content="F: Stockholm is the capital of Sweden, built on islands",
    )
    return r1["id"]


def _active_has(store, mid):
    return store._get_conn().execute(
        "SELECT 1 FROM embed_meta WHERE source_db='memory' AND source_id=?", (mid,)
    ).fetchone() is not None


def _arch_has(store, mid):
    return store._get_conn().execute(
        "SELECT 1 FROM embed_meta_arch WHERE source_db='memory' AND source_id=?", (mid,)
    ).fetchone() is not None


def test_cleanup_moves_archived_vector_not_deletes(mnemos_with_tmpdb):
    from mnemos.consolidation.orchestrator import cleanup_orphan_vectors
    m = mnemos_with_tmpdb
    mid = _seed(m)
    assert _active_has(m.store, mid)
    m.delete(mid)  # soft archive; vector still in the active index
    cleanup_orphan_vectors(m.store._get_conn(), execute=True)
    assert not _active_has(m.store, mid)  # left the active (primary) index
    assert _arch_has(m.store, mid)        # entered the archived (tier-2) index


def test_primary_search_excludes_archived(mnemos_with_tmpdb):
    from mnemos.consolidation.orchestrator import cleanup_orphan_vectors
    m = mnemos_with_tmpdb
    mid = _seed(m)
    m.delete(mid)
    cleanup_orphan_vectors(m.store._get_conn(), execute=True)
    res = m.search("Eiffel Tower Paris France", search_mode="vec", expand_merged=False)
    assert mid not in [r["id"] for r in res["results"]]
    assert "tier2_recall" not in res


def test_tier2_recall_surfaces_archived(mnemos_with_tmpdb):
    from mnemos.consolidation.orchestrator import cleanup_orphan_vectors
    m = mnemos_with_tmpdb
    mid = _seed(m)
    m.delete(mid)
    cleanup_orphan_vectors(m.store._get_conn(), execute=True)
    res = m.search("Eiffel Tower Paris France", search_mode="vec", expand_merged=True)
    t2 = res.get("tier2_recall", [])
    assert mid in [t["id"] for t in t2]
    assert all("vec_distance" in t for t in t2)


def test_reindex_archived_backfills_missing(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    mid = _seed(m)
    m.delete(mid)
    # Simulate a pre-10.7.0 store: the archived memory's vector exists in
    # neither index. reindex_archived should re-embed it into the archived index.
    conn = m.store._get_conn()
    join_col = m.store._get_vec_join_col()
    meta = conn.execute("SELECT id FROM embed_meta WHERE source_id=?", (mid,)).fetchone()
    if meta:
        conn.execute(f"DELETE FROM embed_vec WHERE {join_col}=?", (meta["id"],))
        conn.execute("DELETE FROM embed_meta WHERE source_id=?", (mid,))
    am = conn.execute("SELECT id FROM embed_meta_arch WHERE source_id=?", (mid,)).fetchone()
    if am:
        conn.execute("DELETE FROM embed_vec_arch WHERE rowid=?", (am["id"],))
        conn.execute("DELETE FROM embed_meta_arch WHERE source_id=?", (mid,))
    conn.commit()
    assert mid in {x.id for x in m.store.archived_missing_embeddings()}
    out = m.reindex_archived()
    assert out["embedded_now"] >= 1
    assert mid not in {x.id for x in m.store.archived_missing_embeddings()}
    assert m.reindex_archived()["embedded_now"] == 0  # idempotent
