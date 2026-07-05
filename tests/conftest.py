"""Hermetic test environment.

The suite's baseline is a clean CI machine: no MNEMOS_* configuration and no
locally exported NLI model. On a developer box that runs a live Mnemos
deployment, the login shell typically exports production settings (e.g.
MNEMOS_NLI_BACKEND=onnx, MNEMOS_CONTRADICT_MODE=nli, MNEMOS_DB pointing at the
real store) and ~/.cache/mnemos/nli-onnx holds a real model export. Those leak
into the tests two ways: mnemos.constants bakes several env vars into module
constants at import time, and nli._onnx_model_dir falls back to the cache dir
at call time. Symptoms (2026-07-05): backend-pinning made the torch-fallback
test raise, and live NLI contradiction detection wrote extra memory links that
broke the multi-hop depth and rerank-guard assertions.

This scrub runs at conftest import, which pytest guarantees happens before any
test module (and therefore mnemos itself) is imported, so the import-time
constants also see the clean environment. It intentionally deletes MNEMOS_DB
as well, so a misconfigured test can never touch a real store.
"""

import os
import tempfile

for _var in [v for v in os.environ if v.startswith("MNEMOS_")]:
    del os.environ[_var]

# An empty dir, so a developer's real NLI export can't be picked up via the
# ~/.cache/mnemos/nli-onnx fallback. Tests that need a model dir set their own.
os.environ["MNEMOS_NLI_ONNX_DIR"] = tempfile.mkdtemp(prefix="mnemos-test-nli-")
