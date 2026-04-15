"""
Mnemos storage abstraction: backend-agnostic interface for memory persistence.

A MnemosStore implements all CRUD + search + link operations needed by Mnemos
core. Implementations live in the same package: SQLiteStore (default),
QdrantStore (for production-scale vector search), PostgresStore (community).

All operations are scoped to a `namespace` (default "default"). Multi-user
deployments can use namespaces to isolate data without true multi-tenancy in
the storage layer. Authentication is the responsibility of the transport
layer (MCP server, HTTP server, etc.), never the storage layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Memory:
    """A single memory entry."""
    id: Optional[int] = None
    project: str = ""
    content: str = ""
    tags: str = ""
    importance: int = 5
    type: str = "fact"
    layer: str = "semantic"
    status: str = "active"
    verified: int = 0
    consolidation_lock: int = 0
    subcategory: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    namespace: str = "default"
    access_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_accessed: Optional[str] = None
    last_confirmed: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SearchResult:
    """A single search hit with optional score and metadata."""
    memory: Memory
    score: float = 0.0
    rank: int = 0
    source: str = ""  # 'fts', 'vec', 'rerank', etc.
    widened: bool = False  # came from auto-widen pass
    links: list = field(default_factory=list)


class MnemosStore(ABC):
    """Abstract base class for Mnemos storage backends.

    All implementations must support the operations below. Optional capabilities
    (like vector search) can return empty results if not implemented; the core
    will gracefully fall back to FTS-only mode.
    """

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace

    # --- Lifecycle ---

    @abstractmethod
    def init_schema(self) -> None:
        """Create tables/indexes/collections if missing. Idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release any open connections."""

    # --- CRUD ---

    @abstractmethod
    def store_memory(self, memory: Memory, embedding: Optional[list] = None) -> int:
        """Insert a new memory. Returns assigned ID. Embedding is optional."""

    @abstractmethod
    def get_memory(self, mid: int, increment_access: bool = True) -> Optional[Memory]:
        """Fetch a memory by ID. Optionally bumps access count + last_accessed."""

    @abstractmethod
    def update_memory(self, mid: int, fields: dict, embedding: Optional[list] = None) -> bool:
        """Update specified fields. If content/tags/type/layer change, re-embed."""

    @abstractmethod
    def delete_memory(self, mid: int, hard: bool = False) -> bool:
        """Soft delete (status='archived') or hard delete with vector cleanup."""

    # --- Search ---

    @abstractmethod
    def search_fts(
        self,
        query: str,
        namespace: Optional[str] = None,
        project: Optional[str] = None,
        subcategory: Optional[str] = None,
        layer: Optional[str] = None,
        type_filter: Optional[str] = None,
        status: str = "active",
        valid_only: bool = False,
        limit: int = 50,
        and_mode: bool = True,
    ) -> list:
        """Full-text search with BM25 ranking. Returns list of memory IDs in rank order."""

    @abstractmethod
    def search_vec(
        self,
        embedding: list,
        namespace: Optional[str] = None,
        project: Optional[str] = None,
        subcategory: Optional[str] = None,
        layer: Optional[str] = None,
        type_filter: Optional[str] = None,
        status: str = "active",
        valid_only: bool = False,
        limit: int = 50,
    ) -> list:
        """Vector similarity search. Returns list of (memory_id, distance) tuples."""

    def supports_vec(self) -> bool:
        """Whether this backend has working vector search."""
        return True

    @abstractmethod
    def get_memories_by_ids(self, ids: list) -> dict:
        """Bulk fetch by IDs. Returns {id: Memory} mapping."""

    @abstractmethod
    def count_active(self, namespace: Optional[str] = None) -> int:
        """Count active memories in namespace."""

    # --- Links (relationships between memories) ---

    @abstractmethod
    def store_link(self, source_id: int, target_id: int, relation_type: str, strength: float = 0.5) -> None:
        """Store a relationship between two memories."""

    @abstractmethod
    def get_links(self, memory_ids: list) -> dict:
        """Fetch all links involving the given memory IDs.
        Returns {memory_id: [{linked_id, relation, strength}, ...]}.
        """

    # --- Consolidation support ---

    @abstractmethod
    def store_nyx_insight(
        self,
        memory_id: int,
        source_ids: list,
        insight_type: str,
        consolidation_type: str = "aggregation",
    ) -> None:
        """Record a consolidation event linking source memories to merged result."""

    @abstractmethod
    def get_merged_sources(self, memory_id: int, valid_only: bool = True) -> list:
        """Return source `Memory` objects that were merged into this memory.

        When `valid_only` is True, exclude sources whose `valid_until` is in
        the past (i.e. those marked as superseded by the consolidation
        pipeline). This protects tier-2 recall from surfacing outdated facts.
        """

    # --- Statistics ---

    @abstractmethod
    def stats(self, namespace: Optional[str] = None) -> dict:
        """Return summary statistics: count, active, archived, by_project, etc."""
