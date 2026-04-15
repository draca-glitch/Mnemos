"""Mnemos storage backends."""

from .base import MnemosStore, Memory, SearchResult
from .sqlite_store import SQLiteStore

__all__ = ["MnemosStore", "Memory", "SearchResult", "SQLiteStore"]

# Optional backends - imported lazily so missing dependencies don't break imports
def get_qdrant_store(*args, **kwargs):
    from .qdrant_store import QdrantStore
    return QdrantStore(*args, **kwargs)


def get_postgres_store(*args, **kwargs):
    from .postgres_store import PostgresStore
    return PostgresStore(*args, **kwargs)
