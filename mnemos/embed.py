"""
FastEmbed wrapper for Mnemos.

Uses multilingual-e5-large (1024-dim) ONNX model. Loads once at startup,
~7ms per embedding on CPU. The e5 model uses prefix tokens to distinguish
between document passages and search queries; we handle this transparently.
"""

import hashlib
import sys
import threading
import time
from pathlib import Path

from .constants import (
    FASTEMBED_MODEL, FASTEMBED_CACHE, FASTEMBED_DIMS, DISABLE_MEM_ARENA,
)
from . import _resource

_instance = None
_last_used = 0.0
_lock = threading.Lock()

# Files younger than this are assumed to belong to an active download and
# are left alone. 1 hour is conservative: a 2.2G model download finishes in
# minutes on any reasonable connection, and concurrent mnemos processes
# (MCP server + CLI + consolidation cron) can overlap safely.
_INCOMPLETE_STALE_SECONDS = 3600


def _clean_broken_cache():
    """Remove orphaned download artifacts that prevent model loading.

    huggingface_hub leaves ``.incomplete`` temp files when a download is
    interrupted and never garbage-collects them. A failed initial download
    can also leave ``refs/main`` empty (0 bytes), which causes
    ``snapshot_download(local_files_only=True)`` to return the ``snapshots/``
    parent dir instead of the hash subdir — onnxruntime then fails with
    NoSuchFile because ``snapshots/model.onnx`` doesn't exist. Both
    conditions block the model from ever loading and compound with each
    retry: every failed attempt adds another dead partial, filling the disk
    and making the next download fail even sooner.

    Safe with concurrent downloads: only ``.incomplete`` files older than
    ``_INCOMPLETE_STALE_SECONDS`` are removed (an active download keeps its
    file's mtime fresh). For empty ``refs/main``, only the file itself is
    deleted — not the model dir — so already-downloaded small files survive
    and ``snapshot_download`` raises ``LocalEntryNotFoundError`` (which
    fastembed catches and falls through to a network re-download).
    """
    cache = Path(FASTEMBED_CACHE)
    if not cache.is_dir():
        return

    now = time.time()
    for f in cache.glob("models--*/blobs/*.incomplete"):
        try:
            if now - f.stat().st_mtime > _INCOMPLETE_STALE_SECONDS:
                f.unlink()
        except OSError:
            pass

    for refs_main in cache.glob("models--*/refs/main"):
        try:
            if refs_main.stat().st_size == 0:
                refs_main.unlink()
                print(
                    f"mnemos: removed empty refs/main for "
                    f"{refs_main.parent.parent.name}",
                    file=sys.stderr,
                )
        except OSError:
            pass


def _get_model():
    global _instance, _last_used
    with _lock:
        if _instance is None:
            _resource.guard_memory()
            _clean_broken_cache()
            from fastembed import TextEmbedding
            kwargs = {
                "model_name": FASTEMBED_MODEL,
                "cache_dir": FASTEMBED_CACHE,
            }
            if DISABLE_MEM_ARENA:
                kwargs["enable_cpu_mem_arena"] = False
            _instance = TextEmbedding(**kwargs)
        _last_used = time.monotonic()
        return _instance


def maybe_unload(force=False):
    """Drop the embedder if it has been idle longer than MNEMOS_MODEL_IDLE_TTL.

    Returns True if a model was unloaded. The next embed() pays a one-off
    reload (about 1-2s on a fast CPU, more on small hardware). Opt-in: with the
    default TTL of 0 this never fires. An in-flight embed() holds its own local
    reference to the model, so unloading here cannot pull the ONNX session out
    from under a query that is already running (CPython refcounting keeps it
    alive until that call returns).
    """
    global _instance
    with _lock:
        if _instance is not None and (
            force or (_resource.IDLE_TTL and time.monotonic() - _last_used > _resource.IDLE_TTL)
        ):
            _instance = None
            _resource.trim()
            return True
    return False


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


# Nyx consolidation rewrites these bookkeeping tags every cycle (merge/split
# markers). They carry no retrieval signal, so folding them into the embed-text
# only churns the vector and, worse, invalidates the coherence hash on every
# store that has ever been consolidated. Excluded from the embed-text since
# v10.22.0 so the canonical text is stable across consolidation.
_NYX_TAG_EXACT = frozenset({"consolidated", "nyx-split", "nyx-cycle",
                            "synthesized", "bridge"})
_NYX_TAG_PREFIX = ("merged-into", "split-from", "split-part")


def stable_tags(tags: str) -> str:
    """Drop Nyx-internal bookkeeping tags, keep retrieval-relevant ones."""
    if not tags:
        return ""
    kept = []
    for t in tags.split(","):
        s = t.strip()
        if not s:
            continue
        low = s.lower()
        if low in _NYX_TAG_EXACT or low.startswith(_NYX_TAG_PREFIX):
            continue
        kept.append(s)
    return ", ".join(kept)


def prep_memory_text(project, content, tags="", mem_type="", layer=""):
    """Build the canonical text representation used for embedding a memory.

    Combines project, type, layer, content, and retrieval-relevant tags so the
    embedding captures the metadata that affects retrieval. Nyx bookkeeping tags
    (merge/split markers) are excluded: they churn every consolidation cycle and
    carry no retrieval signal, so including them only destabilizes the vector
    and the coherence hash.
    """
    parts = [project]
    if mem_type and mem_type != "fact":
        parts.append(f"[{mem_type}]")
    if layer and layer != "semantic":
        parts.append(f"[{layer}]")
    parts.append(content)
    stable = stable_tags(tags)
    if stable:
        parts.append(stable)
    return " ".join(parts).strip()
