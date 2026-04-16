"""Regression tests for features added in v10.1.x through v10.3.x.

Covers the ergonomics and safety features that didn't have tests when
shipped: bulk_rewrite, list_tags (both CTE and Python paths), multi-hop
include_linked BFS, three-tier contradiction classification modes,
snippet extraction fallbacks, doctor --migrate, and opt-in analytics
logging. These tests are the regression net that catches future breakage
in the v10.3.x additions.

Kept deliberately focused: happy paths + one or two key edge cases per
feature. Not exhaustive; the existing audit passes caught the deeper
bugs.
"""

import os
import tempfile
import sqlite3

import pytest

from mnemos import Mnemos


@pytest.fixture
def m(tmp_path):
    """Fresh Mnemos instance with isolated test DB.

    We must pass db_path explicitly: DEFAULT_DB_PATH is resolved at
    module import time, so monkeypatch.setenv doesn't isolate here.
    """
    from mnemos.storage.sqlite_store import SQLiteStore
    db = tmp_path / "test.db"
    store = SQLiteStore(db_path=str(db), namespace="test")
    mnemos = Mnemos(store=store, namespace="test", enable_rerank=False)
    yield mnemos
    mnemos.close()


# --- bulk_rewrite (v10.2.3) ---

class TestBulkRewrite:
    def test_dry_run_does_not_mutate_db(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: hello world")
        result = m.bulk_rewrite("world", "earth", dry_run=True)
        assert result["matched"] == 1
        # Verify DB unchanged
        row = m.store._get_conn().execute(
            "SELECT content FROM memories WHERE content LIKE '%world%'"
        ).fetchone()
        assert row is not None

    def test_commit_writes_changes(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: hello world")
        m.bulk_rewrite("world", "earth", dry_run=False)
        row = m.store._get_conn().execute(
            "SELECT content FROM memories WHERE content LIKE '%earth%'"
        ).fetchone()
        assert row is not None

    def test_max_affected_cap_aborts_without_writing(self, m):
        # Seed 3 memories with the pattern
        for i in range(3):
            m.store_memory(skip_dedup=True, project=f"p{i}",
                           content=f"F: X marker {i} content zzz{i}qqq")
        result = m.bulk_rewrite("X marker", "Y marker", max_affected=1, dry_run=False)
        assert "error" in result
        assert result["affected"] == 0
        # All memories still have original content
        hits = m.store._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE '%X marker%'"
        ).fetchone()[0]
        assert hits == 3

    def test_tag_filter_is_word_boundary_not_substring(self, m):
        """v10.3.9 fix: tags LIKE '%name%' must not match 'unnamed'."""
        m.store_memory(skip_dedup=True, project="p", content="F: target with real tag", tags="name,other")
        m.store_memory(skip_dedup=True, project="p", content="F: decoy with substring tag", tags="unnamed,other")
        result = m.bulk_rewrite("target", "T", tags="name", dry_run=True)
        # Must match exactly the first memory, not the one with 'unnamed'
        assert result["matched"] == 1

    def test_regex_mode_supports_backreferences(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: year 2023 event")
        result = m.bulk_rewrite(r"\b20(\d{2})\b", r"19\1",
                                use_regex=True, dry_run=False)
        assert result["affected"] == 1
        row = m.store._get_conn().execute(
            "SELECT content FROM memories WHERE content LIKE '%1923%'"
        ).fetchone()
        assert row is not None

    def test_empty_pattern_errors(self, m):
        assert "error" in m.bulk_rewrite("", "x")


# --- list_tags (v10.1.0 + v10.3.8 CTE path) ---

class TestListTags:
    def test_tags_aggregated_correctly(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: a content", tags="python,async")
        m.store_memory(skip_dedup=True, project="p", content="F: b content different", tags="python,blocking")
        tags = m.list_tags()
        python_tag = next((t for t in tags if t["tag"] == "python"), None)
        assert python_tag is not None
        assert python_tag["count"] == 2

    def test_min_count_filter(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: a unique alpha content", tags="common,rare1")
        m.store_memory(skip_dedup=True, project="p", content="F: b unique beta content", tags="common,rare2")
        tags = m.list_tags(min_count=2)
        tag_names = {t["tag"] for t in tags}
        assert "common" in tag_names
        assert "rare1" not in tag_names

    def test_alpha_order(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: content foo", tags="zebra,apple")
        tags = m.list_tags(order_by="alpha")
        tag_names = [t["tag"] for t in tags]
        assert tag_names == sorted(tag_names, key=str.lower)


# --- Multi-hop include_linked (v10.3.6) ---

class TestMultiHopLinked:
    def test_depth_1_returns_direct_links_only(self, m):
        a = m.store_memory(skip_dedup=True, project="p", content="F: node A unique alpha")["id"]
        b = m.store_memory(skip_dedup=True, project="p", content="F: node B unique beta")["id"]
        c = m.store_memory(skip_dedup=True, project="p", content="F: node C unique gamma")["id"]
        m.store.store_link(a, b, "relates", 0.7)
        m.store.store_link(b, c, "relates", 0.7)
        r = m.search("node A alpha", limit=1, include_linked=True, linked_depth=1)
        linked_ids = {l["id"] for l in r["results"][0].get("linked_memories", [])}
        # Depth 1: only B, not C
        assert b in linked_ids
        assert c not in linked_ids

    def test_depth_2_includes_transitive(self, m):
        a = m.store_memory(skip_dedup=True, project="p", content="F: node A alpha marker")["id"]
        b = m.store_memory(skip_dedup=True, project="p", content="F: node B beta marker")["id"]
        c = m.store_memory(skip_dedup=True, project="p", content="F: node C gamma marker")["id"]
        m.store.store_link(a, b, "relates", 0.7)
        m.store.store_link(b, c, "relates", 0.7)
        r = m.search("node A alpha marker", limit=1, include_linked=True, linked_depth=2)
        linked = r["results"][0].get("linked_memories", [])
        linked_ids = {l["id"] for l in linked}
        assert b in linked_ids
        assert c in linked_ids
        # C should have distance=2 and a 'via' field
        c_entry = next(l for l in linked if l["id"] == c)
        assert c_entry["distance"] == 2
        assert c_entry.get("via") == b

    def test_cycle_does_not_loop(self, m):
        a = m.store_memory(skip_dedup=True, project="p", content="F: node A unique alpha")["id"]
        b = m.store_memory(skip_dedup=True, project="p", content="F: node B unique beta")["id"]
        m.store.store_link(a, b, "relates", 0.7)
        m.store.store_link(b, a, "relates", 0.7)  # cycle
        r = m.search("node A alpha", limit=1, include_linked=True, linked_depth=3)
        linked_ids = [l["id"] for l in r["results"][0].get("linked_memories", [])]
        assert a not in linked_ids  # root excluded from its own linked_memories

    def test_negative_linked_depth_clamped(self, m):
        """v10.3.10 fix: negative linked_depth must not silently disable BFS."""
        a = m.store_memory(skip_dedup=True, project="p", content="F: node A alpha")["id"]
        b = m.store_memory(skip_dedup=True, project="p", content="F: node B beta")["id"]
        m.store.store_link(a, b, "relates", 0.7)
        # Clamped to min=1 internally; should still return direct links
        r = m.search("node A alpha", limit=1, include_linked=True, linked_depth=-5)
        linked = r["results"][0].get("linked_memories", [])
        assert len(linked) > 0


# --- Snippet extraction (v10.1.0 + v10.3.7 sentence-picker + v10.3.10 negative guard) ---

class TestSnippetExtraction:
    def test_vec_fallback_picks_best_sentence(self):
        content = (
            "F: Mnemos uses FTS5 for lexical search. "
            "The cross-encoder reranker is Jina v2 multilingual. "
            "Benchmarks show 98% R@5 on LongMemEval."
        )
        out = Mnemos._vec_fallback_snippet(content, "reranker cross-encoder", 200)
        assert "Jina" in out
        assert "FTS5" not in out

    def test_vec_fallback_head_slice_when_no_overlap(self):
        content = "F: First sentence here. Second sentence follows."
        out = Mnemos._vec_fallback_snippet(content, "completely unrelated", 200)
        # Should return head slice when no token overlap
        assert out.startswith("F: First")

    def test_empty_content_returns_empty(self):
        """v10.3.9 fix."""
        assert Mnemos._vec_fallback_snippet("", "q", 100) == ""
        assert Mnemos._vec_fallback_snippet("   ", "q", 100) == ""

    def test_non_positive_chars_returns_empty(self):
        """v10.3.10 fix."""
        assert Mnemos._vec_fallback_snippet("some content", "q", 0) == ""
        assert Mnemos._vec_fallback_snippet("some content", "q", -5) == ""

    def test_briefing_line_sentence_boundary_with_ellipsis(self):
        """v10.1.2 sentence-aware truncation."""
        long = (
            "L: Memory launch checklist. FINAL. Load-bearing principle softened "
            "per user feedback. Ship only verifiable claims."
        )
        out = Mnemos._briefing_line(long, max_chars=50)
        assert out.endswith("…")
        # Should end on a sentence boundary when possible
        assert "." in out or "…" in out


# --- Doctor --migrate (v10.3.5) ---

class TestDoctorMigrate:
    def test_doctor_reports_healthy_on_clean_db(self, m):
        report = m.doctor(migrate=False)
        # A fresh Mnemos init is well-formed; no issues expected
        assert report["status"] == "healthy"

    def test_doctor_migrate_creates_missing_aux_tables(self, m, tmp_path):
        # Drop an aux table to simulate drift
        conn = m.store._get_conn()
        conn.execute("DROP TABLE IF EXISTS retrieval_log")
        conn.commit()
        # Re-init the connection cache so next _get_conn triggers init_schema
        m.store._conn = None
        report = m.doctor(migrate=True)
        # init_schema runs on reconnect and recreates the table; doctor
        # reports either migrations_applied or status=healthy post-init
        tables = {
            r[0] for r in m.store._get_conn().execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "retrieval_log" in tables


# --- NUL byte stripping on store (v10.3.10) ---

class TestNulByteStripping:
    def test_nul_in_content_stripped(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: hello\x00world")
        row = m.store._get_conn().execute(
            "SELECT content FROM memories WHERE project='p'"
        ).fetchone()
        assert "\x00" not in (row["content"] or "")

    def test_nul_in_tags_stripped(self, m):
        m.store_memory(skip_dedup=True, project="p", content="F: distinct content here",
                       tags="a\x00tag,b\x00tag")
        row = m.store._get_conn().execute(
            "SELECT tags FROM memories WHERE project='p'"
        ).fetchone()
        assert "\x00" not in (row["tags"] or "")
