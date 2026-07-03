"""
Resource-awareness helpers for Mnemos: optional idle model unloading and
memory-pressure guarding.

All behaviour here is opt-in via environment variables and defaults to a no-op,
so existing deployments are unaffected. It exists so a lean memory system can
run alongside other services on a small box without permanently pinning the
embedder and reranker in RAM.

    MNEMOS_MODEL_IDLE_TTL   seconds a model may sit idle before the reaper
                            unloads it on its next tick. 0 (default) = never
                            unload, models stay resident as before.
    MNEMOS_MIN_FREE_MB      refuse to load a model when MemAvailable is below
                            this floor, so the search path degrades to a
                            lighter stage instead of risking an OOM kill.
                            0 (default) = off.
    MNEMOS_DISABLE_MEM_ARENA  disable the ONNX Runtime CPU memory arena on
                            every session Mnemos creates (embedder, reranker,
                            NLI scorers), bounding RSS during ACTIVE periods
                            where the idle reaper never gets a window.
                            ~10-15% slower inference. 0 (default) = off.
                            (Lives in constants.py; listed here because this
                            docstring is the resource-controls index.)

Reclaim mechanics (measured): dropping the model reference plus gc frees the
ONNX session and returns its arena to the OS; malloc_trim then hands back the
glibc-side residue. Both steps are needed for RSS to actually fall.
"""

import ctypes
import gc
import os

IDLE_TTL = int(os.environ.get("MNEMOS_MODEL_IDLE_TTL", "0"))
MIN_FREE_MB = int(os.environ.get("MNEMOS_MIN_FREE_MB", "0"))


def available_mb():
    """Best-effort MemAvailable in MB from /proc/meminfo.

    Returns None when it cannot be read (e.g. non-Linux), which callers treat
    as "no pressure" so the guard never blocks where it cannot measure.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None


def guard_memory():
    """Raise MemoryError if free memory is below MNEMOS_MIN_FREE_MB.

    Call this before loading a model. Callers in the search path already catch
    the failure and drop to a lighter stage (vec-only, then FTS5), so this
    turns a potential OOM into graceful degradation. No-op when MIN_FREE_MB=0.
    """
    if MIN_FREE_MB:
        avail = available_mb()
        if avail is not None and avail < MIN_FREE_MB:
            raise MemoryError(f"{avail:.0f} MB available < {MIN_FREE_MB} MB floor")


def trim():
    """Run gc then return freed glibc arena to the OS via malloc_trim.

    gc frees the ONNX session (and its arena); malloc_trim reclaims the
    glibc-held residue gc leaves behind. Safe no-op off glibc.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
