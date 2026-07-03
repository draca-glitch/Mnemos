"""NLI scoring layer for the store decision path and the Nyx phase-4 finder.

Replaces the cross-encoder reranker for dedup confirmation and contradiction
detection. A reranker answers "are these about the same topic?"; an NLI model
answers "do these say the same / opposite things?", which is the question the
store decision layer actually asks (benchmarks/nli-bench, 2026-07-02:
contradiction AUC 0.939 vs 0.69 for the reranker; dedup AUC 0.983 vs 0.95).

Language routing is agnostic, not tied to any specific language: content that
reads as English goes to an English checkpoint (ANLI+FEVER-hardened, the
strongest benched); everything else goes to a multilingual XNLI checkpoint
(~100 languages). Routing uses a cheap English-stopword heuristic; text with
no prose signal at all (paths, versions, numbers) defaults to English.

Backends, in preference order (v10.16.0):
  1. ONNX int8 (onnxruntime, already in the dependency tree via fastembed;
     tokenizer via transformers, no torch). Models are local exports under
     MNEMOS_NLI_ONNX_DIR (default ~/.cache/mnemos/nli-onnx/{en,multi});
     produce them once with scripts/export_nli_onnx.py.
  2. torch + transformers fallback (install extra: mnemos[nli-torch]),
     loading the HF checkpoints directly.
Every entry point degrades gracefully (returns None) when neither backend
is usable.
"""

import os
import re
import threading

from .embed import embed
from .constants import NLI_EN_MODEL, NLI_MULTI_MODEL, NLI_MAX_LENGTH

_EN_STOPWORDS = re.compile(
    r"\b(the|and|is|are|was|were|of|to|in|for|with|on|at|by|from|that|this|"
    r"it|as|be|has|have|had|not|but|or|an|when|which|will|would|should|"
    r"there|their|then|than|these|those|its|into|about|after|before|only)\b",
    re.IGNORECASE,
)
_NON_ASCII_LETTER = re.compile(r"[^\x00-\x7f]")

_scorers = {}
_scorer_lock = threading.Lock()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _onnx_model_dir(multilingual=False):
    """Directory containing an exported model.onnx for the routed model,
    or None. Layout: <MNEMOS_NLI_ONNX_DIR>/{en,multi}/model.onnx."""
    base = os.environ.get(
        "MNEMOS_NLI_ONNX_DIR",
        os.path.expanduser("~/.cache/mnemos/nli-onnx"))
    d = os.path.join(base, "multi" if multilingual else "en")
    return d if os.path.exists(os.path.join(d, "model.onnx")) else None


def _onnx_runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _onnx_available() -> bool:
    if not _onnx_runtime_available():
        return False
    return (_onnx_model_dir(multilingual=False) is not None
            or _onnx_model_dir(multilingual=True) is not None)


def is_available() -> bool:
    """True when at least one NLI backend (ONNX export or torch) is usable."""
    return _onnx_available() or _torch_available()


def is_english(text: str) -> bool:
    """Cheap routing heuristic: does this text read as English prose?

    English function words present -> English. No function words but
    non-ASCII letters present -> not English (covers diacritics and
    non-Latin scripts). No prose signal at all -> default English, which
    is harmless: such texts are identifiers and numbers either model
    reads the same way.
    """
    words = text.split()
    if not words:
        return True
    hits = len(_EN_STOPWORDS.findall(text))
    if hits >= 2 or hits / len(words) > 0.15:
        return True
    if _NON_ASCII_LETTER.search(text):
        return False
    return True


class _TorchNliScorer:
    """Lazy transformers-backed scorer. score() returns (P(entail), P(contra))."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.idx = {}
        for i, label in self.model.config.id2label.items():
            label = label.lower()
            if "entail" in label:
                self.idx["entail"] = i
            elif "contra" in label:
                self.idx["contra"] = i

    def score(self, premise, hypothesis):
        with self._torch.no_grad():
            enc = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                 truncation=True, max_length=NLI_MAX_LENGTH)
            probs = self._torch.softmax(self.model(**enc).logits[0], dim=-1)
        return float(probs[self.idx["entail"]]), float(probs[self.idx["contra"]])


class _OnnxNliScorer:
    """onnxruntime-backed scorer over a local export (no torch needed)."""

    def __init__(self, onnx_dir):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer, AutoConfig
        self._np = np
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        cfg = AutoConfig.from_pretrained(onnx_dir)
        self.idx = {}
        for i, label in cfg.id2label.items():
            label = label.lower()
            if "entail" in label:
                self.idx["entail"] = int(i)
            elif "contra" in label:
                self.idx["contra"] = int(i)
        self.session = ort.InferenceSession(
            os.path.join(onnx_dir, "model.onnx"),
            providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}

    def score(self, premise, hypothesis):
        np = self._np
        enc = self.tokenizer(premise, hypothesis, return_tensors="np",
                             truncation=True, max_length=NLI_MAX_LENGTH)
        feed = {k: v for k, v in enc.items() if k in self._input_names}
        logits = self.session.run(None, feed)[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return float(probs[self.idx["entail"]]), float(probs[self.idx["contra"]])


def _get_scorer(multilingual=False):
    """ONNX-first scorer resolution. MNEMOS_NLI_BACKEND pins a backend:
    auto (default) prefers a local ONNX export and falls back to torch;
    onnx and torch use only that backend (raising when unusable, which the
    public score functions turn into a graceful None)."""
    key = "multi" if multilingual else "en"
    backend = os.environ.get("MNEMOS_NLI_BACKEND", "auto").lower()
    with _scorer_lock:
        if key not in _scorers:
            onnx_dir = (None if backend == "torch"
                        else _onnx_model_dir(multilingual=multilingual))
            if onnx_dir is not None:
                _scorers[key] = _OnnxNliScorer(onnx_dir)
            elif backend == "onnx":
                raise RuntimeError(
                    "MNEMOS_NLI_BACKEND=onnx but no exported model found "
                    "(run scripts/export_nli_onnx.py)")
            else:
                model_id = NLI_MULTI_MODEL if multilingual else NLI_EN_MODEL
                _scorers[key] = _TorchNliScorer(model_id)
        return _scorers[key]


def _score_pair(a, b):
    """Both directions through the routed scorer. Returns ((e1,c1),(e2,c2))."""
    multilingual = not (is_english(a) and is_english(b))
    scorer = _get_scorer(multilingual=multilingual)
    return scorer.score(a, b), scorer.score(b, a)


def p_contradiction(a, b):
    """Max-direction P(contradiction). None when no backend is usable.

    Max over both directions is mandatory: real contradictions can score
    asymmetrically (benched 0.44 one direction, 0.99 the other).
    """
    if not is_available():
        return None
    try:
        (_, c1), (_, c2) = _score_pair(a, b)
    except Exception:
        return None
    return max(c1, c2)


def bidirectional_entailment(a, b):
    """Min-direction P(entailment): duplicate = each side entails the other.

    None when no backend is usable.
    """
    if not is_available():
        return None
    try:
        (e1, _), (e2, _) = _score_pair(a, b)
    except Exception:
        return None
    return min(e1, e2)


def p_entailment(premise, hypothesis):
    """One-direction P(entailment): does the premise text state the
    hypothesis? Used by the phase-3 novelty gate (an insight entailed by
    either source alone is a restatement). None when no backend is usable.
    """
    if not is_available():
        return None
    try:
        multilingual = not (is_english(premise) and is_english(hypothesis))
        scorer = _get_scorer(multilingual=multilingual)
        e, _ = scorer.score(premise, hypothesis)
    except Exception:
        return None
    return e


def _lines(text):
    return [ln.strip() for ln in text.split("\n")
            if ln.strip() and ln.strip() != "---"]


def line_max_contradiction(a, b, top_k=8):
    """Line-level contradiction finder: max P(contra) over the top_k
    cosine-preselected line pairs of two records.

    Recall-first: isolating conflicting statements from surrounding lines
    rescues contradictions that blob-level scoring buries (benched: the
    diagnosis conflict scored 0.58 blob-level, 0.9956 line-level). None
    when the runtime is unavailable or either record has no lines.
    """
    if not is_available():
        return None
    lines_a, lines_b = _lines(a), _lines(b)
    if not lines_a or not lines_b:
        return None
    vecs = embed(lines_a + lines_b, prefix="passage")
    if not vecs or len(vecs) != len(lines_a) + len(lines_b):
        pairs = [(la, lb) for la in lines_a for lb in lines_b][:top_k]
    else:
        va, vb = vecs[:len(lines_a)], vecs[len(lines_a):]
        scored = []
        for i, la in enumerate(lines_a):
            for j, lb in enumerate(lines_b):
                cos = sum(x * y for x, y in zip(va[i], vb[j]))
                scored.append((cos, la, lb))
        scored.sort(key=lambda t: t[0], reverse=True)
        pairs = [(la, lb) for _, la, lb in scored[:top_k]]
    best = 0.0
    try:
        for la, lb in pairs:
            multilingual = not (is_english(la) and is_english(lb))
            scorer = _get_scorer(multilingual=multilingual)
            _, c1 = scorer.score(la, lb)
            _, c2 = scorer.score(lb, la)
            best = max(best, c1, c2)
    except Exception:
        return None
    return best


def line_max_duplicate(a, b, top_k=8):
    """Line-level shared-fact finder: max min-direction P(entail) over the
    top_k cosine-preselected line pairs of two records.

    Two records share a fact when some line in one and some line in the
    other bidirectionally entail each other. This is the phase-2 cluster
    admission signal (weave-bench gate replay: both 2026-07-03 production
    noise clusters scored below 0.57 on every member pair and dissolve at
    the 0.70 gate). None when the runtime is unavailable or either record
    has no lines.
    """
    if not is_available():
        return None
    lines_a, lines_b = _lines(a), _lines(b)
    if not lines_a or not lines_b:
        return None
    vecs = embed(lines_a + lines_b, prefix="passage")
    if not vecs or len(vecs) != len(lines_a) + len(lines_b):
        pairs = [(la, lb) for la in lines_a for lb in lines_b][:top_k]
    else:
        va, vb = vecs[:len(lines_a)], vecs[len(lines_a):]
        scored = []
        for i, la in enumerate(lines_a):
            for j, lb in enumerate(lines_b):
                cos = sum(x * y for x, y in zip(va[i], vb[j]))
                scored.append((cos, la, lb))
        scored.sort(key=lambda t: t[0], reverse=True)
        pairs = [(la, lb) for _, la, lb in scored[:top_k]]
    best = 0.0
    try:
        for la, lb in pairs:
            multilingual = not (is_english(la) and is_english(lb))
            scorer = _get_scorer(multilingual=multilingual)
            e1, _ = scorer.score(la, lb)
            e2, _ = scorer.score(lb, la)
            best = max(best, min(e1, e2))
    except Exception:
        return None
    return best
