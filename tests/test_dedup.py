"""Regression: store-time dedup must not blanket-block distinct memories when
the cross-encoder reranker is unavailable.

Bug (2026-07-02): Mnemos._dedup fell back to a fixed score=0.75 for any coarse
FTS/CML/vec candidate when rerank was off, so distinct memories that merely
shared a domain were flagged duplicates and silently not stored (11 of 16
dropped in a fresh-install shakedown). The fallback now scores by real vector
distance and blocks only within VEC_DEDUP_MAX_DISTANCE.
"""
import pytest

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore


@pytest.fixture
def m(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t", enable_rerank=False,
                  enable_contradiction_detection=False)


def _stored(res):
    return res.get("status") in ("stored", "stored-split")


def test_distinct_memories_store_without_rerank(m):
    # Unrelated facts across domains must all store (the old fallback blocked
    # every one after the first that shared any coarse signal).
    a = m.store_memory("server", "F:epsilon runs ubuntu 24.04 with 64gb ecc ram")
    b = m.store_memory("health", "F:vitamin d dose is 4000 iu daily in winter")
    c = m.store_memory("work", "F:the barclays fire alarm ppm runs monthly for asset ms277")
    assert _stored(a) and _stored(b) and _stored(c)
    assert m.stats()["active"] == 3


def test_near_identical_still_deduped_without_rerank(m):
    a = m.store_memory("dev", "F:the api server listens on port 8080 and binds to localhost")
    b = m.store_memory("dev", "F:the api server listens on port 8080 and binds to localhost")
    assert _stored(a)
    # A (near-)identical memory must still be caught: tiny vector distance is a
    # strong signal even without the reranker.
    assert not _stored(b)
    assert "existing_id" in b
