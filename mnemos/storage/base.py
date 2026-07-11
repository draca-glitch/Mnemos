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
    def store_memory(self, memory: Memory, embedding: Optional[list] = None,
                     text_hash: Optional[str] = None) -> int:
        """Insert a new memory. Returns assigned ID. Embedding is optional;
        text_hash records the canonical embed-text hash for staleness checks."""

    @abstractmethod
    def get_memory(self, mid: int, increment_access: bool = True) -> Optional[Memory]:
        """Fetch a memory by ID. Optionally bumps access count + last_accessed."""

    @abstractmethod
    def update_memory(self, mid: int, fields: dict, embedding: Optional[list] = None,
                      text_hash: Optional[str] = None) -> bool:
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

    # --- Tier-2 archived index (v10.7.0, optional; SQLite implements it) ---

    def search_vec_archived(self, embedding, **kwargs) -> list:
        """KNN over the archived vector index. Backends without it return []."""
        return []

    def _store_archived_embedding(self, mid: int, embedding: list, text_hash=None) -> None:
        """Store a vector in the archived index. No-op by default."""

    def move_embedding_to_archive(self, mid: int) -> bool:
        """Move a vector from the active to the archived index. No-op by default."""
        return False

    def archived_missing_embeddings(self, namespace=None) -> list:
        """Archived memories lacking an archived-index vector. Empty by default."""
        return []

    def archived_embed_count(self) -> int:
        return 0

    # --- Statistics ---

    @abstractmethod
    def stats(self, namespace: Optional[str] = None) -> dict:
        """Return summary statistics: count, active, archived, by_project, etc."""

    # --- Tag discovery (optional, backends may no-op) ---

    def list_tags(
        self,
        namespace: Optional[str] = None,
        project: Optional[str] = None,
        min_count: int = 1,
        order_by: str = "count",
        limit: int = 500,
    ) -> list:
        """Return all unique tags with usage counts.

        Default implementation returns an empty list; backends that can
        efficiently aggregate tags should override. Format:
        [{"tag": "...", "count": int, "example_id": int}, ...]
        """
        return []

    # --- Snippet extraction (optional, used by search when snippet_chars is set) ---

    def get_snippets(
        self,
        ids: list,
        query: str,
        chars: int = 200,
    ) -> dict:
        """Return query-matched snippets for the given memory IDs.

        Default implementation returns empty dict; backends that support
        FTS snippet extraction (like SQLite FTS5) should override.
        Format: {memory_id: snippet_string}.
        """
        return {}

    # --- Retrieval logging (opt-in, used when MNEMOS_RETRIEVAL_LOG=1) ---

    def log_retrieval(
        self,
        query: str,
        memory_ids: list,
        session_id: Optional[str] = None,
    ) -> None:
        """Persist a search event: the query and the memory IDs it returned.

        Default implementation is a no-op. Backends that want retrieval-log
        semantics (SQLite-based stores) override with an INSERT. Callers
        pass `session_id=None` to omit; future analytics may group by it.
        """
        return None

    # --- Tool usage logging (opt-in, used when MNEMOS_TOOL_USAGE_LOG=1) ---

    def log_tool_usage(self, tool_name: str) -> None:
        """Persist a single tool call: name + timestamp, nothing else.

        Default implementation is a no-op. SQLite backend writes one row
        to `tool_usage`. Useful for health diagnostics without parsing
        MCP server stdin/stdout logs. Carries no user content.
        """
        return None

    # --- Consolidation run logging (always-on, written from Nyx orchestrator) ---

    def log_consolidation_run(
        self,
        clusters_found: int = 0,
        clusters_merged: int = 0,
        memories_archived: int = 0,
        memories_created: int = 0,
        details: str = "",
        phase_details: str = "{}",
    ) -> None:
        """Persist one row summarizing a Nyx-cycle run.

        Called by the orchestrator at the end of each execute=True run so
        health checks, last-run lookups, and post-hoc debugging have a
        durable audit trail. Default implementation is a no-op; SQLite
        backend writes to the `consolidation_log` table.
        """
        return None

    # --- Oversized memory remediation (v10.8.0, used by remediate_oversized) ---

    def find_oversized_memories(
        self,
        namespace: str,
        min_size: int = 4000,
        max_size: Optional[int] = None,
        include_archived: bool = False,
        limit: Optional[int] = None,
    ) -> list:
        """Find memories with content longer than min_size.

        Returns list of memory IDs, ordered by content length descending.
        Excludes memories with consolidation_lock=1 and memories that already
        have split children (tags containing 'split-from:#'). Default
        implementation returns an empty list; backends that support content
        length queries should override.
        """
        return []

    # --- CML subject dedup (used by _unified_dedup CML tier) ---

    def find_cml_subject_matches(
        self,
        namespace: str,
        project: str,
        pattern: str,
        limit: int = 5,
    ) -> list:
        """Find memories whose content starts with a CML subject pattern.

        pattern is a LIKE pattern (e.g. 'D:foo%'). Returns list of
        dicts with keys: id, content, project. Default implementation
        returns an empty list.
        """
        return []

    # --- Retrieval log useful-loop (v10.17.0, used by get()) ---

    def mark_retrieval_useful(self, memory_id: int) -> None:
        """Mark recent retrieval_log entries for a memory as useful.

        Called when a memory is accessed via get() shortly after being
        returned by a search. No-op by default; backends with retrieval
        logging override.
        """
        return None

    # --- Content-based memory search (used by bulk_rewrite) ---

    def find_memories_by_content(
        self,
        namespace: str,
        project: Optional[str] = None,
        tags: Optional[str] = None,
        content_pattern: Optional[str] = None,
    ) -> list:
        """Find active memories matching content/tags/project filters.

        content_pattern is a raw LIKE pattern (caller provides wildcards),
        or None for no content filter. Returns list of dicts with keys:
        id, content. Default implementation returns an empty list.
        """
        return []

    # --- Briefing (used by briefing()) ---

    def get_briefing_memories(
        self,
        namespace: str,
        project: Optional[str] = None,
        limit: int = 30,
    ) -> list:
        """Return top memories by importance + recency for briefing.

        Returns list of dicts with keys: id, project, content, importance,
        type. Default implementation returns an empty list.
        """
        return []

    # --- Digest (used by digest()) ---

    def get_digest_memories(
        self,
        namespace: str,
        days: int = 7,
        project: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Return recent memories from the last N days.

        Returns list of dicts with keys: id, project, content, created_at,
        importance. Default implementation returns an empty list.
        """
        return []

    # --- Project map (used by map()) ---

    def get_project_map(self, namespace: str) -> list:
        """Return project/subcategory counts for active memories.

        Returns list of dicts with keys: project, subcategory, n.
        Default implementation returns an empty list.
        """
        return []

    # --- Embedding backfill (used by embed_fill()) ---

    def get_unembedded_memories(
        self,
        namespace: str,
        limit: Optional[int] = None,
    ) -> list:
        """Return active memories that have no embedding vector.

        Returns list of dicts with keys: id, project, content, tags, type,
        layer. Default implementation returns an empty list.
        """
        return []

    # --- Embedding coverage (used by embed_status()) ---

    def get_embed_coverage(self, namespace: str) -> dict:
        """Return embedding coverage statistics.

        Returns dict with keys: active, embedded, missing, stale, unverified,
        coverage. Default implementation returns zeros.
        """
        return {
            "active": 0, "embedded": 0, "missing": 0,
            "stale": 0, "unverified": 0, "coverage": 0.0,
        }

    # --- Health check (used by doctor()) ---

    def health_check(self, namespace: str, migrate: bool = False) -> dict:
        """Run backend-specific health checks and optional migrations.

        Returns dict with keys: checks (list of str), issues (list of str),
        migrations_applied (list of str). The core layer adds its own
        checks (coherence, provenance) on top of this. Default
        implementation returns an empty report.
        """
        return {"checks": [], "issues": [], "migrations_applied": []}

    # --- Content/vector coherence (v10.19.0, used by doctor()) ---

    def get_coherence_mismatches(self, namespace: str) -> dict:
        """Check that stored embeddings match current memory content.

        Compares each active memory's embed-text hash against the hash
        recorded when its vector was stored. Returns dict with keys:
        checkable (int), mismatched_ids (list of int). Default
        implementation returns empty results.
        """
        return {"checkable": 0, "mismatched_ids": []}

    def reembed_mismatched(self, namespace: str, mismatched_ids: list,
                           embed_fn, text_hash_fn, prep_fn) -> int:
        """Re-embed memories whose content no longer matches their vector.

        Called by doctor --migrate. Returns count of successfully re-embedded
        memories. Default implementation is a no-op returning 0.
        """
        return 0

    # --- Vector provenance (used by doctor()) ---

    def get_vector_provenance(self) -> list:
        """Return vector model provenance statistics.

        Returns list of dicts with keys: model, count. Rows with model=None
        predate provenance tracking. Default implementation returns an
        empty list.
        """
        return []

    # --- Archive-side embedding lifecycle (v10.24.0, used by doctor()) ---

    def check_archive_lifecycle(self, namespace: str, migrate: bool,
                                report: dict) -> None:
        """Check archive-side embedding lifecycle (v10.24.0).

        Detects and optionally migrates three archive-side drifts:
          - embed_meta tables with UTC embedded_at default (pre-localtime DDL)
          - orphan tier-2 embedding rows (hard deletes before v10.24.0)
          - incomplete tier-2 archive index (gaps and legacy hashes)

        Mutates `report` in place: appends to report["issues"],
        report["checks"], report["migrations_applied"], and may set
        report["backup"]. Default implementation is a no-op.
        """
        pass
