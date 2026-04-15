"""
FastEmbed wrapper for Mnemos.

Uses multilingual-e5-large (1024-dim) ONNX model. Loads once at startup,
~7ms per embedding on CPU. The e5 model uses prefix tokens to distinguish
between document passages and search queries; we handle this transparently.
"""

import hashlib

from .constants import FASTEMBED_MODEL, FASTEMBED_CACHE, FASTEMBED_DIMS

_instance = None


def _get_model():
    global _instance
    if _instance is None:
        from fastembed import TextEmbedding
        _instance = TextEmbedding(
            model_name=FASTEMBED_MODEL,
            cache_dir=FASTEMBED_CACHE,
        )
    return _instance


def embed(texts, prefix="passage"):
    """Embed a list of texts. prefix is 'passage' for docs, 'query' for queries.

    Returns a list of lists of floats (1024-dim, L2-normalized).
    Returns empty list on failure.
    """
    if not texts:
        return []
    if isinstance(texts, str):
        texts = [texts]
    # e5 expects "passage: " or "query: " prefix
    prefixed = [f"{prefix}: {t}" for t in texts]
    try:
        import math
        model = _get_model()
        # L2-normalize each vector so cosine similarity can be computed as a
        # simple dot product and L2 distance stays bounded in [0, 2]. Recent
        # fastembed versions no longer normalize e5-large output, so we do it
        # here explicitly, all downstream thresholds (dedup, contradiction
        # detection) assume unit-norm vectors.
        out = []
        for vec in model.embed(prefixed):
            v = list(vec)
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                v = [x / norm for x in v]
            out.append(v)
        return out
    except Exception as e:
        import sys
        print(f"FastEmbed error: {e}", file=sys.stderr)
        return []


def text_hash(text: str) -> str:
    """SHA256 hash of text, used to detect changes for re-embedding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prep_memory_text(project, content, tags="", mem_type="", layer=""):
    """Build the canonical text representation used for embedding a memory.

    Combines project, type, layer, content, and tags into a single string
    so the embedding captures all the metadata that affects retrieval.
    """
    parts = [project]
    if mem_type and mem_type != "fact":
        parts.append(f"[{mem_type}]")
    if layer and layer != "semantic":
        parts.append(f"[{layer}]")
    parts.append(content)
    if tags:
        parts.append(tags)
    return " ".join(parts).strip()
