"""
Mnemos: persistent memory system for AI agents.

Named after Mnemosyne (Greek: μνήμη, memory).
Benchmark: LongMemEval R@5 = 98.1% (hybrid mode).

Quick start:
    from mnemos import Mnemos
    m = Mnemos()  # uses SQLite default
    m.store(project="dev", content="F:Mnemos uses sqlite-vec")
    results = m.search("vector storage")
"""

__version__ = "10.3.2"

from .core import Mnemos
from .storage.base import MnemosStore, Memory

__all__ = ["Mnemos", "MnemosStore", "Memory", "__version__"]
