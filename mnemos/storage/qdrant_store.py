"""
Qdrant storage backend for Mnemos.

For deployments where vector search needs to scale beyond ~25K memories or
where you want to share Qdrant infrastructure across multiple Mnemos
instances. Uses HNSW indexing for sub-50ms search at million-scale.

Metadata (FTS, links, nyx_insights) is still stored in SQLite; Qdrant
handles only the vector layer. This hybrid is the same architecture used
in production at https://github.com/draca-glitch (mail/docs indexing via
Qdrant + memory metadata via SQLite).

Status: REFERENCE IMPLEMENTATION. The author runs Mnemos in this hybrid
mode in production, but this open-source class is a clean rewrite, not
yet feature-complete with all SQLiteStore methods. Contributions welcome.
"""

from typing import Optional

from .base import MnemosStore, Memory
from .sqlite_store import SQLiteStore
from ..constants import FASTEMBED_DIMS, DEFAULT_NAMESPACE


class QdrantStore(MnemosStore):
    """Hybrid backend: SQLite for metadata + FTS, Qdrant for vector search.

    This is a thin wrapper that delegates metadata operations to SQLiteStore
    and vector operations to a Qdrant collection. Use this when you need
    HNSW-scale vector search but want to keep the relational metadata model.
    """

    def __init__(
        self,
        sqlite_path: Optional[str] = None,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "mnemos_memories",
        namespace: str = DEFAULT_NAMESPACE,
        api_key: Optional[str] = None,
    ):
        super().__init__(namespace=namespace)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError:
            raise ImportError(
                "QdrantStore requires qdrant-client. Install with: "
                "pip install qdrant-client"
            )

        self._sqlite = SQLiteStore(db_path=sqlite_path, namespace=namespace)
        self._client = QdrantClient(url=qdrant_url, api_key=api_key)
        self._collection = collection
        self._qmodels = qmodels

    def init_schema(self):
        """Create SQLite tables and Qdrant collection if missing."""
        self._sqlite._get_conn()
        self._sqlite.init_schema()

        collections = {c.name for c in self._client.get_collections().collections}
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._qmodels.VectorParams(
                    size=FASTEMBED_DIMS,
                    distance=self._qmodels.Distance.COSINE,
                ),
            )

    def close(self):
        self._sqlite.close()
        try:
            self._client.close()
        except Exception:
            pass

    # --- Delegate metadata operations to SQLite ---

    def store_memory(self, memory: Memory, embedding=None) -> int:
        mid = self._sqlite.store_memory(memory, embedding=None)  # skip sqlite-vec
        if embedding is not None:
            self._upsert_vector(mid, embedding, memory)
        return mid

    def _upsert_vector(self, mid: int, embedding: list, memory: Memory):
        """Push the vector to Qdrant with full metadata payload for filtering."""
        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._qmodels.PointStruct(
                    id=mid,
                    vector=list(embedding),
                    payload={
                        "namespace": memory.namespace or self.namespace,
                        "project": memory.project,
                        "subcategory": memory.subcategory,
                        "type": memory.type,
                        "layer": memory.layer,
                        "status": memory.status,
                        "valid_until": memory.valid_until,
                    },
                )
            ],
        )

    def get_memory(self, mid, increment_access=True):
        return self._sqlite.get_memory(mid, increment_access=increment_access)

    def update_memory(self, mid, fields, embedding=None):
        ok = self._sqlite.update_memory(mid, fields, embedding=None)
        if ok and embedding is not None:
            mem = self._sqlite.get_memory(mid, increment_access=False)
            if mem:
                self._upsert_vector(mid, embedding, mem)
        return ok

    def delete_memory(self, mid, hard=False):
        ok = self._sqlite.delete_memory(mid, hard=hard)
        if ok and hard:
            try:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=self._qmodels.PointIdsList(points=[mid]),
                )
            except Exception:
                pass
        return ok

    def search_fts(self, *args, **kwargs):
        return self._sqlite.search_fts(*args, **kwargs)

    def search_vec(self, embedding, namespace=None, project=None, subcategory=None,
                   layer=None, type_filter=None, status="active", valid_only=False,
                   limit=50):
        ns = namespace or self.namespace
        # Build Qdrant filter
        must = [
            self._qmodels.FieldCondition(
                key="namespace",
                match=self._qmodels.MatchValue(value=ns),
            )
        ]
        if status != "all":
            must.append(self._qmodels.FieldCondition(
                key="status", match=self._qmodels.MatchValue(value=status),
            ))
        if project:
            must.append(self._qmodels.FieldCondition(
                key="project", match=self._qmodels.MatchValue(value=project),
            ))
        if subcategory:
            must.append(self._qmodels.FieldCondition(
                key="subcategory", match=self._qmodels.MatchValue(value=subcategory),
            ))
        if layer:
            must.append(self._qmodels.FieldCondition(
                key="layer", match=self._qmodels.MatchValue(value=layer),
            ))
        if type_filter:
            must.append(self._qmodels.FieldCondition(
                key="type", match=self._qmodels.MatchValue(value=type_filter),
            ))

        results = self._client.search(
            collection_name=self._collection,
            query_vector=list(embedding),
            query_filter=self._qmodels.Filter(must=must),
            limit=limit,
        )
        # Qdrant returns (1 - cosine_similarity) as a "score"; convert to distance
        return [(int(r.id), 1.0 - r.score) for r in results]

    def get_memories_by_ids(self, ids):
        return self._sqlite.get_memories_by_ids(ids)

    def count_active(self, namespace=None):
        return self._sqlite.count_active(namespace=namespace)

    def store_link(self, *args, **kwargs):
        return self._sqlite.store_link(*args, **kwargs)

    def get_links(self, *args, **kwargs):
        return self._sqlite.get_links(*args, **kwargs)

    def store_nyx_insight(self, *args, **kwargs):
        return self._sqlite.store_nyx_insight(*args, **kwargs)

    def get_merged_sources(self, *args, **kwargs):
        return self._sqlite.get_merged_sources(*args, **kwargs)

    def stats(self, namespace=None):
        return self._sqlite.stats(namespace=namespace)
