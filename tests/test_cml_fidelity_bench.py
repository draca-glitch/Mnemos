"""Check the CML transformation cache without calling a model provider."""

import pytest

from benchmarks import cml_fidelity_bench as bench


@pytest.mark.parametrize("prompt_name", ["MERGE_SYSTEM", "CEMELIFY_USER_TEMPLATE"])
def test_cemelify_cache_invalidates_when_prompt_changes(tmp_path, monkeypatch, prompt_name):
    monkeypatch.setattr(bench, "CACHE_DIR", str(tmp_path))
    responses = iter(["F: original response", "F: revised response"])
    monkeypatch.setattr(bench, "llm_chat", lambda *args, **kwargs: next(responses))
    prose = "The local service listens on port 8080."

    assert bench.cemelify(prose) == "F: original response"
    assert bench.cemelify(prose) == "F: original response"

    monkeypatch.setattr(bench, prompt_name, getattr(bench, prompt_name) + "\nRevised instructions.")

    assert bench.cemelify(prose) == "F: revised response"
    assert bench.cemelify(prose) == "F: revised response"
