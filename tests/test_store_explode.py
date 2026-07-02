"""Regression: store paths normalize single-line prefix-chained CML into one
fact per line, so a weak local merge LLM (or a caller) chaining facts with ';'
does not persist a single-line multi-fact blob.

Bug (2026-07-02): explode_cml_chain existed but was wired into no runtime path.
apply_merge and store_memory only ran the size-guard splitter (len > 4000), so
short ';'-chained merges/stores were persisted single-line. The exploder now
runs before the size guard on both write paths.
"""
import pytest

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore


@pytest.fixture
def m(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t", enable_rerank=False,
                  enable_contradiction_detection=False)


def test_store_explodes_single_line_chain(m):
    r = m.store_memory("dev", "F:api binds to localhost port 8080; D:chose postgres over mysql")
    assert m.get(r["id"])["content"] == "F:api binds to localhost port 8080\nD:chose postgres over mysql"


def test_store_leaves_single_fact_untouched(m):
    content = "F:the api server binds to localhost on port 8080"
    r = m.store_memory("dev", content)
    assert m.get(r["id"])["content"] == content


def test_store_leaves_intra_fact_semicolon_untouched(m):
    # A ';' not followed by a CML prefix is a sub-clause, not a fact boundary.
    content = "F:qwen pool routes epsilon, linnuc, plex; least-busy node wins"
    r = m.store_memory("dev", content)
    assert m.get(r["id"])["content"] == content
