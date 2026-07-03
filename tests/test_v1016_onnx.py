"""Tests for v10.16.0: ONNX int8 backend for the NLI layer.

Backend selection: ONNX (onnxruntime, already in the dependency tree via
fastembed) is preferred when exported models are present under
MNEMOS_NLI_ONNX_DIR; the torch scorer remains as fallback. No model files
are loaded by this suite; scorer classes are stubbed.
"""

import os

import mnemos.nli as nli


class _Recorder:
    """Stub scorer class recording construction args."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _Recorder.instances.append(self)

    def score(self, premise, hypothesis):
        return (0.5, 0.5)


def _fresh(monkeypatch):
    _Recorder.instances = []
    monkeypatch.setattr(nli, "_scorers", {})


class TestOnnxModelDir:
    def test_env_dir_with_model_found(self, tmp_path, monkeypatch):
        d = tmp_path / "en"
        d.mkdir()
        (d / "model.onnx").touch()
        monkeypatch.setenv("MNEMOS_NLI_ONNX_DIR", str(tmp_path))
        assert nli._onnx_model_dir(multilingual=False) == str(d)

    def test_missing_model_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOS_NLI_ONNX_DIR", str(tmp_path))
        assert nli._onnx_model_dir(multilingual=False) is None
        assert nli._onnx_model_dir(multilingual=True) is None


class TestBackendSelection:
    def test_prefers_onnx_when_model_present(self, tmp_path, monkeypatch):
        _fresh(monkeypatch)
        monkeypatch.setattr(nli, "_onnx_model_dir",
                            lambda multilingual=False: str(tmp_path))
        monkeypatch.setattr(nli, "_OnnxNliScorer", _Recorder)
        scorer = nli._get_scorer(multilingual=False)
        assert isinstance(scorer, _Recorder)
        assert str(tmp_path) in scorer.args

    def test_falls_back_to_torch_without_onnx(self, monkeypatch):
        _fresh(monkeypatch)
        monkeypatch.setattr(nli, "_onnx_model_dir",
                            lambda multilingual=False: None)
        monkeypatch.setattr(nli, "_TorchNliScorer", _Recorder)
        scorer = nli._get_scorer(multilingual=False)
        assert isinstance(scorer, _Recorder)

    def test_is_available_true_with_onnx_models_and_no_torch(self, monkeypatch):
        # stub the import probe too: CI has neither torch nor transformers,
        # and this test is about the model-presence logic, not the env
        monkeypatch.setattr(nli, "_torch_available", lambda: False)
        monkeypatch.setattr(nli, "_onnx_runtime_available", lambda: True)
        monkeypatch.setattr(nli, "_onnx_model_dir",
                            lambda multilingual=False: "/somewhere")
        assert nli.is_available() is True

    def test_is_available_false_with_neither_backend(self, monkeypatch):
        monkeypatch.setattr(nli, "_torch_available", lambda: False)
        monkeypatch.setattr(nli, "_onnx_runtime_available", lambda: True)
        monkeypatch.setattr(nli, "_onnx_model_dir",
                            lambda multilingual=False: None)
        assert nli.is_available() is False


class TestGracefulScorerFailure:
    def test_score_functions_return_none_when_scorer_raises(self, monkeypatch):
        monkeypatch.setattr(nli, "is_available", lambda: True)

        def boom(multilingual=False):
            raise RuntimeError("model load failed")

        monkeypatch.setattr(nli, "_get_scorer", boom)
        assert nli.p_contradiction("a", "b") is None
        assert nli.bidirectional_entailment("a", "b") is None
        assert nli.line_max_contradiction("a", "b") is None


class TestBackendPin:
    def test_backend_pin_torch_ignores_onnx_models(self, tmp_path, monkeypatch):
        _fresh(monkeypatch)
        monkeypatch.setenv("MNEMOS_NLI_BACKEND", "torch")
        monkeypatch.setattr(nli, "_onnx_model_dir",
                            lambda multilingual=False: str(tmp_path))
        monkeypatch.setattr(nli, "_TorchNliScorer", _Recorder)
        assert isinstance(nli._get_scorer(multilingual=False), _Recorder)

    def test_backend_pin_onnx_never_falls_back_to_torch(self, monkeypatch):
        _fresh(monkeypatch)
        monkeypatch.setenv("MNEMOS_NLI_BACKEND", "onnx")
        monkeypatch.setattr(nli, "_onnx_model_dir",
                            lambda multilingual=False: None)
        monkeypatch.setattr(nli, "is_available", lambda: True)
        # pinned to onnx with no models: scoring degrades to None, no torch load
        assert nli.p_contradiction("a", "b") is None
