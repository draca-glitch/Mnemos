"""
PostgreSQL + pgvector storage backend for Mnemos.

STATUS: STUB. Not implemented. Contributions welcome.

This stub demonstrates how to add a new backend without touching the rest of
Mnemos. Implement the abstract methods from MnemosStore and pgvector with HNSW
will give you sub-50ms vector search at million-scale memories with full
ACID transactions, MVCC concurrency, and battle-tested operations.

Implementation guide:
  1. Use psycopg2 or asyncpg for the connection
  2. Use pgvector extension: CREATE EXTENSION vector;
  3. CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
  4. Use tsvector for FTS (or external Elastic for better BM25)
  5. Implement all abstract methods from MnemosStore

See sqlite_store.py for the reference SQL schema and method implementations.
"""

from .base import MnemosStore


class PostgresStore(MnemosStore):
    """PostgreSQL + pgvector backend (NOT IMPLEMENTED)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PostgresStore is a stub. Implementation welcome, see docstring "
            "and sqlite_store.py for reference. Open a PR at "
            "https://github.com/draca-glitch/mnemos"
        )

    def init_schema(self): raise NotImplementedError()
    def close(self): pass
    def store_memory(self, memory, embedding=None): raise NotImplementedError()
    def get_memory(self, mid, increment_access=True): raise NotImplementedError()
    def update_memory(self, mid, fields, embedding=None): raise NotImplementedError()
    def delete_memory(self, mid, hard=False): raise NotImplementedError()
    def search_fts(self, query, **kwargs): raise NotImplementedError()
    def search_vec(self, embedding, **kwargs): raise NotImplementedError()
    def get_memories_by_ids(self, ids): raise NotImplementedError()
    def count_active(self, namespace=None): raise NotImplementedError()
    def store_link(self, source_id, target_id, relation_type, strength=0.5): raise NotImplementedError()
    def get_links(self, memory_ids): raise NotImplementedError()
    def store_nyx_insight(self, memory_id, source_ids, insight_type, consolidation_type="aggregation"): raise NotImplementedError()
    def get_merged_sources(self, memory_id): raise NotImplementedError()
    def stats(self, namespace=None): raise NotImplementedError()
