"""Per-phase LLM config resolution (llm._get_config).

Covers the per-phase API-key override (added 2026-07-02): a single phase (e.g.
MERGE) can target a cloud provider with a real token while the other phases
stay on a local endpoint whose key is a throwaway, so the secret never reaches
the local endpoint.
"""
import os

from mnemos.consolidation.llm import _get_config


def _clear(monkeypatch):
    for k in list(os.environ):
        if k.startswith("MNEMOS_LLM_"):
            monkeypatch.delenv(k, raising=False)


def test_per_phase_key_override(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-local")
    monkeypatch.setenv("MNEMOS_LLM_API_KEY_MERGE", "sk-cloud")
    assert _get_config(phase="MERGE")["key"] == "sk-cloud"
    assert _get_config(phase="WEAVE")["key"] == "sk-local"
    assert _get_config()["key"] == "sk-local"


def test_per_phase_key_falls_back_to_global(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-local")
    assert _get_config(phase="MERGE")["key"] == "sk-local"


def test_per_phase_omit_temperature(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "k")
    monkeypatch.setenv("MNEMOS_LLM_OMIT_TEMPERATURE_MERGE", "1")
    assert _get_config(phase="MERGE")["omit_temperature"] is True
    assert _get_config(phase="WEAVE")["omit_temperature"] is False
    assert _get_config()["omit_temperature"] is False


def test_per_phase_url_and_model_still_resolve(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "k")
    monkeypatch.setenv("MNEMOS_LLM_API_URL", "http://local/v1/chat/completions")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "qwen-pool")
    monkeypatch.setenv("MNEMOS_LLM_API_URL_MERGE", "https://api.anthropic.com/v1/chat/completions")
    monkeypatch.setenv("MNEMOS_LLM_MODEL_MERGE", "claude-sonnet-5")
    cfg = _get_config(phase="MERGE")
    assert cfg["url"] == "https://api.anthropic.com/v1/chat/completions"
    assert cfg["model"] == "claude-sonnet-5"
