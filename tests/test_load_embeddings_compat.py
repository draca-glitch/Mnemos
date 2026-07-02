"""Regression: load_embeddings must handle a rowid-keyed embed_vec (the fresh
sqlite_store schema), not only the legacy explicit-'id' PK schema.

Bug (2026-07-01, found on the NUC while wiring the local Nyx cycle):
consolidation/phases.py::load_embeddings hardcoded `WHERE id = ?`, so
`consolidate` crashed with `sqlite3.OperationalError: no such column: id` on
every fresh install, whose embed_vec is `vec0(embedding float[N])` (rowid-keyed,
no id column). Epsilon's own DB never hit it because its embed_vec dates from the
v7/v8 explicit-id era, so this branch was untested until now.
"""
import pytest

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore
from mnemos.consolidation.phases import load_embeddings, _vec_join_col


@pytest.fixture
def m(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t", enable_rerank=False,
                  enable_contradiction_detection=False)


def test_fresh_db_is_rowid_keyed(m):
    # Precondition the bug depends on: a fresh sqlite_store DB has no 'id' column.
    m.store_memory("test", "F:seed so embed_vec exists")
    assert _vec_join_col(m.store._get_conn()) == "rowid"


def test_load_embeddings_on_rowid_schema(m):
    # Distinct memories across projects; vec-dedup may still drop some, so assert
    # against the count that actually landed rather than a fixed number.
    m.store_memory("test", "F:the earth completes one orbit of the sun per year")
    m.store_memory("dev", "R:prompt cache allows at most four breakpoints per request")
    m.store_memory("brf", "C:Anna Svensson, board treasurer, anna@example.se, 070-1234567")
    conn = m.store._get_conn()
    stored = conn.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    assert stored >= 1
    # Regression: this raised 'no such column: id' on the rowid schema before the
    # fix. It must run and return exactly one real 1024-dim embedding per stored memory.
    embeddings, mergeable, mem_by_id = load_embeddings(conn)
    assert len(embeddings) == stored
    assert len(mem_by_id) == stored
    for vec in embeddings.values():
        assert vec.shape[0] == 1024
