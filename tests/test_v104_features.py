"""Regression tests for v10.4.0 features.

Covers:
  - cemelify() happy path (mocked chat())
  - cemelify() fallbacks: None response, empty string, exception
  - _needs_cemelify() filter
  - MNEMOS_DISABLE_LLM=1 suppresses the loud-fail
  - Loud-fail without disable raises RuntimeError
  - OpenAI URL + unset MNEMOS_LLM_MODEL → default 'gpt-4o-mini'
  - Non-OpenAI URL has no implicit default
  - Phase 0.5 skips consolidation_lock=1 memories

Style mirrors tests/test_v103_features.py: tmp_path-isolated Mnemos fixture,
focused happy-path + key edge cases, no network, chat() monkeypatched.
"""

from unittest.mock import patch

import pytest

from mnemos import Mnemos
from mnemos.cemelify import cemelify, _needs_cemelify
from mnemos.consolidation import llm as llm_module


@pytest.fixture
def m(tmp_path):
    from mnemos.storage.sqlite_store import SQLiteStore
    db = tmp_path / "test.db"
    store = SQLiteStore(db_path=str(db), namespace="test")
    mnemos = Mnemos(store=store, namespace="test", enable_rerank=False)
    yield mnemos
    mnemos.close()


# --- Cemelify (v10.4.0) ---

class TestCemelify:
    def test_happy_path_mocked_chat(self):
        with patch("mnemos.consolidation.llm.chat",
                   return_value="F: mnemos uses sqlite-vec"):
            out = cemelify("Mnemos uses sqlite-vec for vector storage.")
        assert out == "F: mnemos uses sqlite-vec"

    def test_fallback_on_none_response(self):
        with patch("mnemos.consolidation.llm.chat", return_value=None):
            original = "some raw content that would have been cemelified"
            out = cemelify(original)
        assert out == original

    def test_fallback_on_empty_string_response(self):
        with patch("mnemos.consolidation.llm.chat", return_value="   "):
            out = cemelify("raw content")
        assert out == "raw content"

    def test_fallback_on_chat_exception(self):
        def boom(*a, **kw):
            raise RuntimeError("network down")
        with patch("mnemos.consolidation.llm.chat", side_effect=boom):
            out = cemelify("raw content")
        assert out == "raw content"

    def test_empty_input_returns_empty(self):
        assert cemelify("") == ""
        assert cemelify("   ") == "   "


class TestNeedsCemelify:
    def test_non_cml_prose_is_candidate(self):
        assert _needs_cemelify("This is raw prose.") is True

    def test_short_cml_is_not_candidate(self):
        assert _needs_cemelify("F: mnemos uses sqlite-vec") is False

    def test_long_cml_triggers_anyway(self):
        long_cml = "F: " + ("x " * 500)  # > 800 chars
        assert _needs_cemelify(long_cml) is True

    def test_empty_or_none(self):
        assert _needs_cemelify("") is False
        assert _needs_cemelify(None) is False


# --- MNEMOS_DISABLE_LLM + loud-fail (v10.4.0) ---

class TestLoudFailAndDisable:
    def test_disable_llm_runs_sql_only_silently(self, m, monkeypatch):
        """MNEMOS_DISABLE_LLM=1 restores the pre-v10.4.0 silent behavior:
        LLM phases dropped, Phase 6 bookkeeping runs, no exception."""
        monkeypatch.delenv("MNEMOS_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MNEMOS_LLM_MODEL", raising=False)
        monkeypatch.setenv("MNEMOS_DISABLE_LLM", "1")
        m.store_memory(skip_dedup=True, project="p", content="F: seed memory")
        stats = m.consolidate(execute=True, phases={1, 2, 3, 6})
        assert "phase6" in stats
        assert "phase2" not in stats

    def test_loud_fail_without_disable(self, m, monkeypatch):
        """Without MNEMOS_DISABLE_LLM, an unconfigured LLM must raise
        RuntimeError rather than silently skipping phases."""
        monkeypatch.delenv("MNEMOS_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MNEMOS_LLM_MODEL", raising=False)
        monkeypatch.delenv("MNEMOS_DISABLE_LLM", raising=False)
        m.store_memory(skip_dedup=True, project="p", content="F: seed memory")
        with pytest.raises(RuntimeError, match="No LLM configured"):
            m.consolidate(execute=True, phases={1, 2, 3, 6})


# --- OpenAI default model preset (v10.4.0) ---

class TestOpenAIDefaultModel:
    def test_openai_url_defaults_to_gpt_4o_mini(self, monkeypatch):
        monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-test")
        monkeypatch.delenv("MNEMOS_LLM_MODEL", raising=False)
        monkeypatch.delenv("MNEMOS_LLM_FAST_MODEL", raising=False)
        monkeypatch.setenv(
            "MNEMOS_LLM_API_URL",
            "https://api.openai.com/v1/chat/completions",
        )
        cfg = llm_module._get_config()
        assert cfg["model"] == "gpt-4o-mini"
        assert llm_module.is_configured() is True

    def test_openai_default_url_no_env_also_defaults(self, monkeypatch):
        """Even with no MNEMOS_LLM_API_URL set, the package default points
        at OpenAI, so the model default kicks in."""
        monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-test")
        monkeypatch.delenv("MNEMOS_LLM_MODEL", raising=False)
        monkeypatch.delenv("MNEMOS_LLM_API_URL", raising=False)
        cfg = llm_module._get_config()
        assert cfg["model"] == "gpt-4o-mini"

    def test_non_openai_url_has_no_default_model(self, monkeypatch):
        monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-test")
        monkeypatch.delenv("MNEMOS_LLM_MODEL", raising=False)
        monkeypatch.setenv(
            "MNEMOS_LLM_API_URL",
            "http://localhost:11434/v1/chat/completions",
        )
        cfg = llm_module._get_config()
        assert cfg["model"] is None
        assert llm_module.is_configured() is False

    def test_explicit_model_wins_on_openai(self, monkeypatch):
        monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MNEMOS_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv(
            "MNEMOS_LLM_API_URL",
            "https://api.openai.com/v1/chat/completions",
        )
        cfg = llm_module._get_config()
        assert cfg["model"] == "gpt-4o"


# --- Phase 0.5 skips consolidation_lock memories (v10.4.0) ---

class TestPhase05SkipsLocked:
    def _build_mem_by_id(self, conn, mid):
        row = conn.execute(
            "SELECT id, content, consolidation_lock FROM memories WHERE id=?",
            (mid,),
        ).fetchone()
        return {
            row["id"]: {
                "id": row["id"],
                "content": row["content"],
                "consolidation_lock": row["consolidation_lock"],
            }
        }

    def test_consolidation_lock_memory_is_not_cemelified(self, m):
        long_prose = "This is prose. " * 100  # ~1500 chars, no CML prefix
        locked_id = m.store_memory(
            skip_dedup=True,
            project="p",
            content=long_prose,
            consolidation_lock=True,
        )["id"]

        from mnemos.consolidation.orchestrator import _phase_cemelify
        conn = m.store._get_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        mem_by_id = self._build_mem_by_id(conn, locked_id)

        called_with = []
        def spy(content, max_tokens=512):
            called_with.append(content)
            return "F: this should not happen"
        with patch("mnemos.cemelify.cemelify", side_effect=spy):
            stats = _phase_cemelify(conn, m.store, mem_by_id, execute=True)

        assert stats["candidates"] == 0
        assert called_with == []

    def test_unlocked_long_prose_is_cemelified(self, m):
        long_prose = "This is prose. " * 100
        unlocked_id = m.store_memory(
            skip_dedup=True, project="p", content=long_prose,
        )["id"]

        from mnemos.consolidation.orchestrator import _phase_cemelify
        conn = m.store._get_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        mem_by_id = self._build_mem_by_id(conn, unlocked_id)

        with patch("mnemos.cemelify.cemelify", return_value="F: cemelified prose"):
            stats = _phase_cemelify(conn, m.store, mem_by_id, execute=True)

        assert stats["candidates"] == 1
        assert stats["cemelified"] == 1
        updated = conn.execute(
            "SELECT content FROM memories WHERE id=?", (unlocked_id,),
        ).fetchone()
        assert updated["content"] == "F: cemelified prose"
