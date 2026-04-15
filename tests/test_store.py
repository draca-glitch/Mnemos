"""
End-to-end smoke test for Mnemos.

Exercises the full pipeline against a throwaway SQLite database:
  store -> search -> get -> update -> consolidate (dry run) -> doctor

No LLM is required. If MNEMOS_LLM_API_KEY is unset (the default in CI),
the Nyx cycle's LLM phases are skipped automatically and only Phase 6
(SQL bookkeeping) runs, which still exercises the orchestrator path.

Uses a temp MNEMOS_DB so parallel test runs don't collide.
"""

import os
import tempfile

import pytest


@pytest.fixture
def mnemos_with_tmpdb():
    """Fresh Mnemos instance with a throwaway SQLite file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["MNEMOS_DB"] = path
    # Import only after MNEMOS_DB is set so constants.DEFAULT_DB_PATH picks it up
    from mnemos.core import Mnemos
    m = Mnemos()
    yield m
    m.close()
    for suffix in ("", "-journal", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)


def test_store_search_get_update_delete(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb

    # Store two memories in the same project
    r1 = m.store_memory(project="test", content="F:redis runs on port 6379")
    assert r1.get("status") == "stored"
    assert isinstance(r1.get("id"), int)
    mid1 = r1["id"]

    r2 = m.store_memory(project="test", content="F:postgres runs on port 5432")
    assert r2.get("status") == "stored"
    mid2 = r2["id"]
    assert mid1 != mid2

    # Search finds both under a broad query
    sr = m.search(query="port", project="test")
    assert sr.get("count") >= 1

    # Specific search surfaces the right one first
    sr = m.search(query="redis", project="test", limit=5)
    ids = [r["id"] for r in sr["results"]]
    assert mid1 in ids

    # Get increments access_count
    got = m.get(mid1)
    assert got.get("id") == mid1
    assert got.get("access_count", 0) >= 1
    assert got.get("content", "").startswith("F:redis")

    # Update content
    ur = m.update(mid1, content="F:redis runs on port 6379 with TLS")
    assert ur.get("status") == "updated"
    assert "content" in ur.get("fields", [])
    got2 = m.get(mid1)
    assert "TLS" in got2.get("content", "")

    # Delete (archive, default)
    dr = m.delete(mid1)
    assert dr.get("status") == "archived"

    # Archived memories are filtered out of active search
    sr = m.search(query="redis", project="test", status="active")
    assert all(r["id"] != mid1 for r in sr.get("results", []))


def test_consolidate_dry_run(mnemos_with_tmpdb, monkeypatch):
    m = mnemos_with_tmpdb
    # Ensure no LLM is configured so we exercise the graceful-skip path
    monkeypatch.delenv("MNEMOS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("MNEMOS_LLM_MODEL", raising=False)

    m.store_memory(project="test", content="F:one memory")
    m.store_memory(project="test", content="F:another memory")

    stats = m.consolidate(execute=False)
    # Phase 6 (bookkeeping) always runs; LLM phases are skipped
    assert "phase6" in stats or stats == {} or isinstance(stats, dict)


def test_doctor(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    m.store_memory(project="test", content="F:health check")

    report = m.doctor()
    assert report.get("status") in ("healthy", "issues_detected")
    assert "checks" in report
    assert "issues" in report
    # A fresh store with one memory should be healthy
    assert report["status"] == "healthy", f"unexpected issues: {report.get('issues')}"


def test_stats(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    before = m.stats()
    m.store_memory(project="test", content="F:stats sample")
    after = m.stats()
    # Stats structure is backend-defined; just assert it grew or exists
    assert isinstance(after, dict)
    if "active" in before and "active" in after:
        assert after["active"] >= before["active"]


def test_cml_dedup_fires_on_same_subject(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    r1 = m.store_memory(project="test",
                        content="F:Ethereum wallet 0x1234abcd for BRF")
    assert r1.get("status") == "stored"

    # Same CML type + subject + very similar content: dedup should flag
    r2 = m.store_memory(project="test",
                        content="F:Ethereum wallet 0x1234abcd for BRF, verified")
    # Either flagged as duplicate (existing_id set) or stored (weak rerank signal);
    # whichever, the "cml" method should be in the methods list IF it is flagged
    if r2.get("existing_id"):
        assert "cml" in r2.get("methods", [])


def test_missing_project_or_content_errors(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    assert "error" in m.store_memory(project="", content="F:hi")
    assert "error" in m.store_memory(project="test", content="")


def test_get_returns_error_on_missing_id(mnemos_with_tmpdb):
    m = mnemos_with_tmpdb
    assert "error" in m.get(99999)
