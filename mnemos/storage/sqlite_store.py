"""
SQLite storage backend for Mnemos.

Single-file storage with FTS5 (BM25) and sqlite-vec (vector search) in one
unified database. Atomic transactions, no synchronization between stores.

Performance characteristics (1024-dim vectors, brute-force):
  - 1K memories:    ~5ms search
  - 5K memories:   ~25ms search
  - 10K memories:  ~45ms search
  - 25K memories:  ~75ms search
  - 50K memories: ~265ms search (cache boundary)
  - 100K memories: ~475ms search

Recommended for personal/small-team deployments up to ~10K memories on SSD,
~5K on HDD. Beyond that, use QdrantStore or PostgresStore.
"""

import os
import sqlite3
import struct
from typing import Optional

from .base import MnemosStore, Memory, SearchResult
from ..constants import (
    DEFAULT_DB_PATH, DEFAULT_NAMESPACE, FASTEMBED_DIMS, FASTEMBED_MODEL,
    BM25_WEIGHTS, RANKING_ORDER_SQL, BM25_CALL,
)


def _serialize_vec(vec):
    """Pack a Python list/tuple of floats into the binary format sqlite-vec expects."""
    return struct.pack(f"{len(vec)}f", *vec)


def _vec_join_col_conn(conn, table):
    """Join/PK column for a vec0 table on this connection: 'id' when the
    schema declares `id INTEGER PRIMARY KEY`, 'rowid' otherwise. Conn-level
    twin of SQLiteStore._get_vec_join_col for callers that only hold a
    connection (consolidation phases, external plumbing)."""
    try:
        conn.execute(f"SELECT id FROM {table} LIMIT 0").fetchone()
        return "id"
    except sqlite3.OperationalError:
        return "rowid"


def _arch_join_col(conn):
    return _vec_join_col_conn(conn, "embed_vec_arch")


def _store_archived_embedding_conn(conn, mid, embedding, text_hash=None,
                                   commit=True, model=None,
                                   source_key="memory"):
    """Insert or replace a memory's embedding in the tier-2 archived index.

    Handles both embed_vec_arch schemas (implicit rowid vs declared id PK),
    same policy as _store_embedding on the active side."""
    join_col = _arch_join_col(conn)
    existing = conn.execute(
        "SELECT id FROM embed_meta_arch WHERE source_db = ? AND source_id = ?",
        (source_key, mid),
    ).fetchone()
    if existing:
        conn.execute(f"DELETE FROM embed_vec_arch WHERE {join_col} = ?",
                     (existing[0],))
        conn.execute(
            "DELETE FROM embed_meta_arch WHERE source_db = ? AND source_id = ?",
            (source_key, mid),
        )
    if join_col == "id":
        cur = conn.execute(
            "INSERT INTO embed_meta_arch (source_db, source_id, text_hash, model) VALUES (?, ?, ?, ?)",
            (source_key, mid, text_hash, model),
        )
        meta_id = cur.lastrowid
        conn.execute(
            "INSERT INTO embed_vec_arch(id, embedding) VALUES (?, ?)",
            (meta_id, _serialize_vec(embedding)),
        )
    else:
        cur = conn.execute(
            "INSERT INTO embed_vec_arch(embedding) VALUES (?)",
            (_serialize_vec(embedding),),
        )
        vec_id = cur.lastrowid
        conn.execute(
            "INSERT INTO embed_meta_arch (id, source_db, source_id, text_hash, model) VALUES (?, ?, ?, ?, ?)",
            (vec_id, source_key, mid, text_hash, model),
        )
    if commit:
        conn.commit()


def move_embedding_to_archive_conn(conn, mid, source_key="memory"):
    """Move a memory's vector from the active index into the archived index.

    Conn-level implementation shared by the store method, the consolidation
    phases (which archive via raw UPDATEs and previously leaked vectors,
    the 51-gap finding of the 2026-07-07 forensic audit) and external
    plumbing. Returns False when there is no active embedding to move."""
    active_col = _vec_join_col_conn(conn, "embed_vec")
    meta = conn.execute(
        "SELECT id, text_hash, model FROM embed_meta WHERE source_db = ? AND source_id = ?",
        (source_key, mid),
    ).fetchone()
    if not meta:
        return False
    meta_id, meta_hash, meta_model = meta[0], meta[1], meta[2]
    vrow = conn.execute(
        f"SELECT vec_to_json(embedding) AS j FROM embed_vec WHERE {active_col} = ?",
        (meta_id,),
    ).fetchone()
    if not vrow or vrow[0] is None:
        return False
    import json as _json
    # Insert-into-arch and delete-from-active commit together: a crash in
    # between must not leave the vector in both indexes (invisible to the
    # reconcile check, which only looks for missing arch copies).
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        _store_archived_embedding_conn(conn, mid, _json.loads(vrow[0]),
                                       meta_hash, commit=False,
                                       model=meta_model, source_key=source_key)
        conn.execute(f"DELETE FROM embed_vec WHERE {active_col} = ?", (meta_id,))
        conn.execute(
            "DELETE FROM embed_meta WHERE source_db = ? AND source_id = ?",
            (source_key, mid),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return True


def _ensure_vec_db(path, dims=FASTEMBED_DIMS):
    """Open a SQLite connection with the sqlite-vec extension loaded."""
    import sqlite_vec
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS embed_vec USING vec0(embedding float[{dims}])")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embed_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_db TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            text_hash TEXT,
            model TEXT,
            embedded_at TEXT DEFAULT (datetime('now', 'localtime')),
            UNIQUE(source_db, source_id)
        )
    """)
    # Tier-2 recall index (v10.7.0): a SEPARATE vector index for archived
    # memories. Keeping archived vectors out of embed_vec is deliberate, the
    # primary KNN over-fetches k=limit*3 and post-filters status, so mixing
    # archived (the bulk of the corpus) into embed_vec would crowd out active
    # hits before the filter runs. A separate index lets primary search stay
    # active-only with zero regression while expand_merged can vector-search
    # the archived originals directly instead of only reaching them by lineage.
    # Declared-PK schema (v10.24.0) to match what _get_vec_join_col already
    # handles on the active side; pre-existing implicit-rowid arch tables
    # keep working through _arch_join_col detection, mirroring the active
    # table's both-schemas-in-the-wild policy.
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS embed_vec_arch USING vec0(id INTEGER PRIMARY KEY, embedding float[{dims}])")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embed_meta_arch (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_db TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            text_hash TEXT,
            model TEXT,
            embedded_at TEXT DEFAULT (datetime('now', 'localtime')),
            UNIQUE(source_db, source_id)
        )
    """)
    # Back-compat column backfill, same treatment the memories table gets in
    # init_schema. CREATE TABLE IF NOT EXISTS is a no-op on a pre-v10.6
    # embed_meta (no text_hash/model), while _store_embedding writes
    # text_hash unconditionally, so without this every store/update on an
    # older DB throws "no such column". ALTER ADD COLUMN cannot carry a
    # non-constant default, hence nullable embedded_at here.
    for table in ("embed_meta", "embed_meta_arch"):
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col in ("text_hash", "model", "embedded_at"):
            if col not in existing:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
                except Exception:
                    pass
    return conn


class SQLiteStore(MnemosStore):
    """SQLite + FTS5 + sqlite-vec storage backend."""

    SOURCE_KEY = "memory"

    def __init__(self, db_path: Optional[str] = None, namespace: str = DEFAULT_NAMESPACE):
        super().__init__(namespace=namespace)
        self.db_path = db_path or DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        # Lazy-resolved join column for embed_vec. sqlite-vec vec0 virtual
        # tables expose either `rowid` (when only `embedding` column is
        # declared) or `id` (when declared as `id INTEGER PRIMARY KEY`).
        # Both schemas exist in the wild; we detect at first use and cache.
        self._vec_join_col: Optional[str] = None

    # --- Connection management ---

    def _get_conn(self):
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                return self._conn
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                self._conn = None
        self._conn = _ensure_vec_db(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Wait out cross-process write contention instead of raising SQLITE_BUSY.
        # The MCP server, the (now prod-default) CLI, and the Nyx consolidation
        # run are separate processes against one DB; WAL handles reader/writer
        # but write-vs-write still needs a timeout or one side's write throws.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA cache_size=-200000")  # 200MB page cache
        self.init_schema()
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def backup(self, dest_path: str) -> str:
        """Write a WAL-safe, consistent standalone snapshot of the DB.

        Uses VACUUM INTO, which captures the full committed state (including
        rows still resident in an un-checkpointed WAL) into a single
        defragmented file that needs no -wal/-shm sidecar to read. Safe while
        the DB is live. A raw file copy of a WAL-mode .db is NOT safe: it can
        omit WAL-resident rows and, when restored next to a stale -wal/-shm,
        replays mismatched frames and corrupts the btree (the failure mode that
        took out prod on 2026-06-27). Always use this instead of `cp`.

        Atomic: writes to a temp sibling then os.replace()s it into place, so a
        failed VACUUM (disk full, I/O error) never destroys an existing prior
        backup at dest_path. Resolves dest to an absolute path and creates the
        parent dir if missing. Returns the absolute path written.
        """
        dest_path = os.path.abspath(dest_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        conn = self._get_conn()
        conn.commit()  # VACUUM requires no open transaction
        tmp = dest_path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)  # VACUUM INTO refuses to write an existing file
        try:
            conn.execute("VACUUM INTO ?", (tmp,))
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise  # dest_path was never touched, prior backup survives
        os.replace(tmp, dest_path)  # atomic on the same filesystem
        return dest_path

    # --- Schema ---

    def init_schema(self):
        conn = self._conn
        if conn is None:
            return
        # Main memories table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL DEFAULT 'default',
                project TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                importance INTEGER DEFAULT 5,
                status TEXT DEFAULT 'active',
                type TEXT DEFAULT 'fact',
                layer TEXT DEFAULT 'semantic',
                verified INTEGER DEFAULT 0,
                consolidation_lock INTEGER DEFAULT 0,
                subcategory TEXT DEFAULT NULL,
                valid_from TEXT DEFAULT NULL,
                valid_until TEXT DEFAULT NULL,
                access_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                last_accessed TEXT DEFAULT NULL,
                last_confirmed TEXT DEFAULT NULL,
                nyx_processed TEXT DEFAULT NULL
            )
        """)
        # Pre-v10 DBs may predate columns that v10.x adds (notably namespace
        # and nyx_processed). CREATE TABLE IF NOT EXISTS is idempotent but
        # doesn't ALTER an existing schema, so we explicitly patch missing
        # columns before any CREATE INDEX that might reference them. Without
        # this, pointing Mnemos at an older DB threw "no such column:
        # namespace" at the first index create - a cryptic first-run failure
        # that's now a silent migration.
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        _column_backfills = [
            ("namespace",         "TEXT NOT NULL DEFAULT 'default'"),
            ("type",              "TEXT DEFAULT 'fact'"),
            ("nyx_processed",     "TEXT DEFAULT NULL"),
            ("subcategory",       "TEXT DEFAULT NULL"),
            ("valid_from",        "TEXT DEFAULT NULL"),
            ("valid_until",       "TEXT DEFAULT NULL"),
            ("layer",             "TEXT DEFAULT 'semantic'"),
            ("consolidation_lock", "INTEGER DEFAULT 0"),
            ("verified",          "INTEGER DEFAULT 0"),
            ("last_confirmed",    "TEXT DEFAULT NULL"),
            ("last_accessed",     "TEXT DEFAULT NULL"),
            ("updated_at",        "TEXT DEFAULT (datetime('now', 'localtime'))"),
        ]
        for col, coldef in _column_backfills:
            if col not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {coldef}")
                except Exception:
                    # ALTER ADD COLUMN can fail on exotic cases (e.g. a
                    # NOT NULL DEFAULT on a table with a trigger); log
                    # and press on - the index below will throw with a
                    # clearer error if the column truly didn't land.
                    pass
        # Indexes (only created after the backfill pass above so older DBs
        # don't trip on a missing column reference)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_mem_namespace ON memories(namespace)",
            "CREATE INDEX IF NOT EXISTS idx_mem_project ON memories(project)",
            "CREATE INDEX IF NOT EXISTS idx_mem_subcategory ON memories(subcategory)",
            "CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)",
            "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)",
            "CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance)",
            "CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_mem_last_accessed ON memories(last_accessed)",
            "CREATE INDEX IF NOT EXISTS idx_mem_valid_until ON memories(valid_until)",
        ]:
            conn.execute(idx_sql)
        # FTS5
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, project, tags,
                content=memories,
                content_rowid=id,
                tokenize='porter unicode61 remove_diacritics 2'
            )
        """)
        # FTS triggers
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, project, tags)
                VALUES (new.id, new.content, new.project, new.tags);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, project, tags)
                VALUES('delete', old.id, old.content, old.project, old.tags);
            END
        """)
        # Guarded UPDATE trigger: without the WHEN clause every UPDATE fully
        # re-tokenized the row into FTS, and get_memory bumps access_count on
        # every read, i.e. a full FTS reindex per read. Upgrade in place when
        # an existing DB still carries the unguarded trigger (CREATE TRIGGER
        # IF NOT EXISTS would silently keep it).
        existing_au = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='memories_au'"
        ).fetchone()
        if existing_au and "WHEN" not in existing_au[0].upper().split("BEGIN")[0]:
            conn.execute("DROP TRIGGER memories_au")
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
            WHEN new.content IS NOT old.content
              OR new.project IS NOT old.project
              OR new.tags IS NOT old.tags
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, project, tags)
                VALUES('delete', old.id, old.content, old.project, old.tags);
                INSERT INTO memories_fts(rowid, content, project, tags)
                VALUES (new.id, new.content, new.project, new.tags);
            END
        """)
        # Memory links (relationships)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT DEFAULT 'related',
                strength REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)")
        # Nyx insights (consolidation history)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nyx_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                source_ids TEXT NOT NULL,
                insight_type TEXT DEFAULT 'merge',
                consolidation_type TEXT DEFAULT 'aggregation',
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nyx_memory ON nyx_insights(memory_id)")
        # Consolidation audit log (always-on). Every Nyx-cycle run emits one
        # row here via store.log_consolidation_run() - health checks,
        # "last run" lookups, and post-hoc debugging all read this table.
        # Schema aligned with Epsilon's v8 layout so a shared DB works.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT DEFAULT (datetime('now', 'localtime')),
                clusters_found INTEGER DEFAULT 0,
                clusters_merged INTEGER DEFAULT 0,
                memories_archived INTEGER DEFAULT 0,
                memories_created INTEGER DEFAULT 0,
                access_decayed INTEGER DEFAULT 0,
                importance_demoted INTEGER DEFAULT 0,
                details TEXT DEFAULT '',
                phase_details TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consolidation_run_at ON consolidation_log(run_at)")
        # Nyx-cycle persistent state between runs (last-run markers, etc.)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nyx_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        # Retrieval log (opt-in; populated when MNEMOS_RETRIEVAL_LOG=1). Schema
        # matches the Epsilon custom-server layout so the same table serves
        # both deployments if a DB is shared during migration.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                retrieved_at TEXT DEFAULT (datetime('now', 'localtime')),
                useful INTEGER DEFAULT NULL,
                session_id TEXT DEFAULT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_memory ON retrieval_log(memory_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_useful ON retrieval_log(useful)")
        # Tool-usage log (opt-in; populated when MNEMOS_TOOL_USAGE_LOG=1).
        # Only tool name + timestamp - no content, no query text, no IDs.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                called_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_called_at ON tool_usage(called_at)")
        conn.commit()

    # --- CRUD ---

    def store_memory(self, memory: Memory, embedding: Optional[list] = None,
                     text_hash: Optional[str] = None) -> int:
        # Content and vector commit together: a crash between the two used to
        # leave a keyword-findable but vector-invisible memory (FTS triggers
        # are transactional with the INSERT, the vector write was not).
        # BEGIN IMMEDIATE takes the write lock up front so the read-then-write
        # inside _store_embedding cannot race a concurrent embedder of the
        # same id into an IntegrityError.
        conn = self._get_conn()
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                """INSERT INTO memories
                   (namespace, project, content, tags, importance, type, layer,
                    verified, subcategory, valid_from, valid_until, consolidation_lock)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.namespace or self.namespace,
                    memory.project, memory.content, memory.tags,
                    memory.importance, memory.type, memory.layer,
                    memory.verified, memory.subcategory,
                    memory.valid_from, memory.valid_until,
                    memory.consolidation_lock,
                ),
            )
            mid = cur.lastrowid
            if embedding is not None:
                self._store_embedding(mid, embedding, text_hash=text_hash,
                                      commit=False)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return mid

    def _get_vec_join_col(self) -> str:
        """Detect which column embed_vec uses for the PK/join.

        Returns 'id' for schemas declared as `vec0(id INTEGER PRIMARY KEY,
        embedding float[N])` or 'rowid' for `vec0(embedding float[N])`.
        Cached after first resolution per connection.
        """
        if self._vec_join_col is not None:
            return self._vec_join_col
        conn = self._get_conn()
        try:
            conn.execute("SELECT id FROM embed_vec LIMIT 0").fetchone()
            self._vec_join_col = "id"
        except sqlite3.OperationalError:
            self._vec_join_col = "rowid"
        return self._vec_join_col

    def _store_embedding(self, mid: int, embedding: list,
                         text_hash: Optional[str] = None,
                         commit: bool = True):
        """Insert or replace the embedding for a memory.

        Handles both embed_vec schemas (implicit rowid vs explicit id)
        so the same code works against fresh Mnemos-created DBs and
        against DBs created by other tooling that declares an explicit
        `id INTEGER PRIMARY KEY` column on the vec0 virtual table.

        text_hash records the canonical embed-text's hash so staleness
        (content updated, re-embed failed, old vector retained) is
        detectable later; without it a stale vector is indistinguishable
        from a fresh one.

        commit=False joins the caller's transaction instead of committing
        here, so content and vector land atomically.
        """
        conn = self._get_conn()
        join_col = self._get_vec_join_col()
        # Remove existing if any
        existing = conn.execute(
            "SELECT id FROM embed_meta WHERE source_db = ? AND source_id = ?",
            (self.SOURCE_KEY, mid),
        ).fetchone()
        if existing:
            conn.execute(
                f"DELETE FROM embed_vec WHERE {join_col} = ?",
                (existing["id"],),
            )
            conn.execute(
                "DELETE FROM embed_meta WHERE source_db = ? AND source_id = ?",
                (self.SOURCE_KEY, mid),
            )
        if join_col == "id":
            # Explicit-id schema: pre-insert into embed_meta, then use its
            # auto-assigned id as the vec0 row id
            cur = conn.execute(
                "INSERT INTO embed_meta (source_db, source_id, text_hash, model) VALUES (?, ?, ?, ?)",
                (self.SOURCE_KEY, mid, text_hash, FASTEMBED_MODEL),
            )
            meta_id = cur.lastrowid
            conn.execute(
                "INSERT INTO embed_vec(id, embedding) VALUES (?, ?)",
                (meta_id, _serialize_vec(embedding)),
            )
        else:
            # Implicit-rowid schema: insert vec first, use its rowid as the
            # pre-declared id for embed_meta
            cur = conn.execute(
                "INSERT INTO embed_vec(embedding) VALUES (?)",
                (_serialize_vec(embedding),),
            )
            vec_id = cur.lastrowid
            conn.execute(
                "INSERT INTO embed_meta (id, source_db, source_id, text_hash, model) VALUES (?, ?, ?, ?, ?)",
                (vec_id, self.SOURCE_KEY, mid, text_hash, FASTEMBED_MODEL),
            )
        if commit:
            conn.commit()

    def get_memory(self, mid: int, increment_access: bool = True) -> Optional[Memory]:
        conn = self._get_conn()
        if increment_access:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "last_accessed = datetime('now', 'localtime'), "
                "last_confirmed = datetime('now', 'localtime') WHERE id = ?",
                (mid,),
            )
            conn.commit()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    # Whitelist of column names that update_memory will write to. Anything
    # else passed in `fields` is silently dropped, prevents arbitrary
    # column injection from a misbehaving caller.
    _UPDATABLE_COLUMNS = frozenset({
        "project", "content", "tags", "importance", "status",
        "type", "layer", "verified", "consolidation_lock",
        "subcategory", "valid_from", "valid_until",
        "last_confirmed", "last_accessed", "access_count",
    })

    def update_memory(self, mid: int, fields: dict, embedding: Optional[list] = None,
                      text_hash: Optional[str] = None) -> bool:
        conn = self._get_conn()
        # Filter to known columns so an unexpected key cannot inject SQL
        safe_fields = {k: v for k, v in (fields or {}).items() if k in self._UPDATABLE_COLUMNS}
        if not safe_fields and embedding is None:
            return False
        # Single transaction for content + vector: a crash between them used
        # to leave the new content searchable via FTS with the OLD vector and
        # text_hash still attached.
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            rowcount = 0
            if safe_fields:
                set_clause = ", ".join(f"{k} = ?" for k in safe_fields.keys())
                params = list(safe_fields.values()) + [mid]
                cur = conn.execute(
                    f"UPDATE memories SET {set_clause}, updated_at = datetime('now', 'localtime') WHERE id = ?",
                    params,
                )
                rowcount = cur.rowcount
            else:
                # Embedding-only update, verify the row exists so we don't lie about success
                row = conn.execute("SELECT 1 FROM memories WHERE id = ?", (mid,)).fetchone()
                if row is None:
                    conn.rollback()
                    return False
                rowcount = 1
            if embedding is not None:
                self._store_embedding(mid, embedding, text_hash=text_hash,
                                      commit=False)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return rowcount > 0

    def delete_memory(self, mid: int, hard: bool = False) -> bool:
        conn = self._get_conn()
        if hard:
            join_col = self._get_vec_join_col()
            # Remove embedding first
            existing = conn.execute(
                "SELECT id FROM embed_meta WHERE source_db = ? AND source_id = ?",
                (self.SOURCE_KEY, mid),
            ).fetchone()
            if existing:
                conn.execute(
                    f"DELETE FROM embed_vec WHERE {join_col} = ?",
                    (existing["id"],),
                )
                conn.execute(
                    "DELETE FROM embed_meta WHERE source_db = ? AND source_id = ?",
                    (self.SOURCE_KEY, mid),
                )
            # Tier-2 rows too: hard delete used to stop at the active index,
            # leaving arch meta/vec rows behind forever (3 orphans found in
            # prod by the 2026-07-07 audit).
            arch = conn.execute(
                "SELECT id FROM embed_meta_arch WHERE source_db = ? AND source_id = ?",
                (self.SOURCE_KEY, mid),
            ).fetchone()
            if arch:
                arch_col = _arch_join_col(conn)
                conn.execute(
                    f"DELETE FROM embed_vec_arch WHERE {arch_col} = ?",
                    (arch["id"],),
                )
                conn.execute(
                    "DELETE FROM embed_meta_arch WHERE source_db = ? AND source_id = ?",
                    (self.SOURCE_KEY, mid),
                )
            # Links referencing a hard-deleted memory would otherwise persist
            # as phantom rows forever (nothing else ever prunes by absence).
            conn.execute(
                "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
                (mid, mid),
            )
            conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
            conn.commit()
        else:
            conn.execute(
                "UPDATE memories SET status = 'archived', updated_at = datetime('now', 'localtime') WHERE id = ?",
                (mid,),
            )
            conn.commit()
            # Move the vector into the tier-2 index so the archived memory
            # stays reachable by expand_merged. Never blocks the archive
            # itself; reindex-archived is the catch-up for failures here.
            try:
                self.move_embedding_to_archive(mid)
            except Exception:
                pass
        return True

    # --- Search ---

    def search_fts(self, query, namespace=None, project=None, subcategory=None,
                   layer=None, type_filter=None, status="active", valid_only=False,
                   limit=50, and_mode=True):
        from ..query import clean_fts_query
        conn = self._get_conn()
        ns = namespace or self.namespace

        for fts_mode in (("AND", "OR") if and_mode else ("OR",)):
            fts_query = clean_fts_query(query, mode=fts_mode)
            if not fts_query:
                continue
            where = " AND m.namespace = ?"
            params = [fts_query, ns]
            if status != "all":
                where += " AND m.status = ?"
                params.append(status)
            if project:
                where += " AND m.project = ?"
                params.append(project)
            if subcategory:
                where += " AND m.subcategory = ?"
                params.append(subcategory)
            if layer:
                where += " AND m.layer = ?"
                params.append(layer)
            if type_filter:
                where += " AND m.type = ?"
                params.append(type_filter)
            if valid_only:
                where += (" AND (m.valid_until IS NULL OR m.valid_until > date('now', 'localtime'))"
                          " AND (m.valid_from IS NULL OR m.valid_from <= date('now', 'localtime'))")
            params.append(limit)

            try:
                rows = conn.execute(f"""
                    SELECT m.id
                    FROM memories_fts fts
                    JOIN memories m ON m.id = fts.rowid
                    WHERE memories_fts MATCH ?{where}
                    {RANKING_ORDER_SQL}
                    LIMIT ?
                """, params).fetchall()
                if rows:
                    return [r["id"] for r in rows]
            except sqlite3.OperationalError:
                continue
        return []

    def search_vec(self, embedding, namespace=None, project=None, subcategory=None,
                   layer=None, type_filter=None, status="active", valid_only=False,
                   limit=50):
        conn = self._get_conn()
        ns = namespace or self.namespace
        join_col = self._get_vec_join_col()

        # Brute-force vec search (sqlite-vec)
        rows = conn.execute(
            f"""SELECT em.source_id, ev.distance
               FROM embed_vec ev
               JOIN embed_meta em ON em.id = ev.{join_col}
               WHERE em.source_db = ? AND ev.embedding MATCH ? AND k = ?
               ORDER BY ev.distance""",
            (self.SOURCE_KEY, _serialize_vec(embedding), limit * 3),
        ).fetchall()

        if not rows:
            return []

        # Filter by namespace + other criteria
        candidate_ids = [r["source_id"] for r in rows]
        ph = ",".join("?" for _ in candidate_ids)
        filter_params = list(candidate_ids) + [ns]
        filter_clause = " AND namespace = ?"
        if status != "all":
            filter_clause += " AND status = ?"
            filter_params.append(status)
        if project:
            filter_clause += " AND project = ?"
            filter_params.append(project)
        if subcategory:
            filter_clause += " AND subcategory = ?"
            filter_params.append(subcategory)
        if layer:
            filter_clause += " AND layer = ?"
            filter_params.append(layer)
        if type_filter:
            filter_clause += " AND type = ?"
            filter_params.append(type_filter)
        if valid_only:
            filter_clause += (" AND (valid_until IS NULL OR valid_until > date('now', 'localtime'))"
                              " AND (valid_from IS NULL OR valid_from <= date('now', 'localtime'))")

        active = set(
            r[0] for r in conn.execute(
                f"SELECT id FROM memories WHERE id IN ({ph}){filter_clause}",
                filter_params,
            ).fetchall()
        )
        results = [(r["source_id"], r["distance"]) for r in rows if r["source_id"] in active]
        return results[:limit]

    # --- Tier-2 archived index (v10.7.0) ---

    def _store_archived_embedding(self, mid: int, embedding: list,
                                  text_hash: Optional[str] = None,
                                  commit: bool = True,
                                  model: Optional[str] = FASTEMBED_MODEL):
        """Store/replace a memory's embedding in the tier-2 archived index.

        Mirrors _store_embedding; handles both arch vec schemas (implicit
        rowid on pre-v10.24 stores, declared id PK on new ones).
        """
        _store_archived_embedding_conn(self._get_conn(), mid, embedding,
                                       text_hash=text_hash, commit=commit,
                                       model=model, source_key=self.SOURCE_KEY)

    def move_embedding_to_archive(self, mid: int) -> bool:
        """Move a memory's vector from the active index into the archived index.

        Called when a memory is archived: its content now lives in a
        consolidated active memory (or it was soft-deleted), but we keep the
        original's vector for tier-2 recall instead of deleting it. Returns
        False if the memory had no active embedding to move.
        """
        return move_embedding_to_archive_conn(self._get_conn(), mid,
                                              source_key=self.SOURCE_KEY)

    def search_vec_archived(self, embedding, namespace=None, project=None,
                            subcategory=None, layer=None, type_filter=None,
                            valid_only=True, limit=20):
        """KNN over the tier-2 archived index. Always status='archived'.

        Independent of whether an archived memory's consolidated parent ranked
        in primary search, so consolidation that dropped a detail no longer
        makes that detail unrecallable.
        """
        conn = self._get_conn()
        ns = namespace or self.namespace
        arch_col = _arch_join_col(conn)
        rows = conn.execute(
            f"""SELECT em.source_id, ev.distance
               FROM embed_vec_arch ev
               JOIN embed_meta_arch em ON em.id = ev.{arch_col}
               WHERE em.source_db = ? AND ev.embedding MATCH ? AND k = ?
               ORDER BY ev.distance""",
            (self.SOURCE_KEY, _serialize_vec(embedding), limit * 3),
        ).fetchall()
        if not rows:
            return []
        candidate_ids = [r["source_id"] for r in rows]
        ph = ",".join("?" for _ in candidate_ids)
        params = list(candidate_ids) + [ns]
        clause = " AND namespace = ? AND status = 'archived'"
        if project:
            clause += " AND project = ?"
            params.append(project)
        if subcategory:
            clause += " AND subcategory = ?"
            params.append(subcategory)
        if layer:
            clause += " AND layer = ?"
            params.append(layer)
        if type_filter:
            clause += " AND type = ?"
            params.append(type_filter)
        if valid_only:
            clause += (" AND (valid_until IS NULL OR valid_until > date('now', 'localtime'))"
                       " AND (valid_from IS NULL OR valid_from <= date('now', 'localtime'))")
        keep = set(
            r[0] for r in conn.execute(
                f"SELECT id FROM memories WHERE id IN ({ph}){clause}", params
            ).fetchall()
        )
        return [(r["source_id"], r["distance"]) for r in rows if r["source_id"] in keep][:limit]

    def archived_embed_count(self) -> int:
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM embed_meta_arch WHERE source_db = ?", (self.SOURCE_KEY,)
        ).fetchone()[0]

    def archived_missing_embeddings(self, namespace=None) -> list:
        """Archived memory rows that have no vector in the tier-2 index yet."""
        conn = self._get_conn()
        ns = namespace or self.namespace
        rows = conn.execute(
            """SELECT m.* FROM memories m
               WHERE m.status = 'archived' AND m.namespace = ?
               AND NOT EXISTS (
                   SELECT 1 FROM embed_meta_arch em
                   WHERE em.source_db = ? AND em.source_id = m.id
               )""",
            (ns, self.SOURCE_KEY),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def archived_legacy_hash_rows(self, namespace=None) -> list:
        """Archived memory rows whose tier-2 meta carries a legacy hash.

        Pre-v10.6 plumbing wrote 16-char truncated hashes; those vectors
        also predate the canonical embed-text formula, so reindex-archived
        re-embeds them instead of just re-hashing.
        """
        conn = self._get_conn()
        ns = namespace or self.namespace
        rows = conn.execute(
            """SELECT m.* FROM memories m
               JOIN embed_meta_arch em ON em.source_db = ? AND em.source_id = m.id
               WHERE m.status = 'archived' AND m.namespace = ?
               AND (em.text_hash IS NULL OR length(em.text_hash) < 64)""",
            (self.SOURCE_KEY, ns),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_ids(self, ids: list) -> dict:
        if not ids:
            return {}
        conn = self._get_conn()
        ph = ",".join("?" for _ in ids)
        rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({ph})", ids).fetchall()
        return {r["id"]: self._row_to_memory(r) for r in rows}

    def count_active(self, namespace: Optional[str] = None) -> int:
        conn = self._get_conn()
        ns = namespace or self.namespace
        return conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active' AND namespace = ?", (ns,)
        ).fetchone()[0]

    # --- Links ---

    def store_link(self, source_id, target_id, relation_type, strength=0.5):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation_type, strength) "
            "VALUES (?, ?, ?, ?)",
            (source_id, target_id, relation_type, strength),
        )
        conn.commit()

    def get_links(self, memory_ids):
        if not memory_ids:
            return {}
        conn = self._get_conn()
        ph = ",".join("?" for _ in memory_ids)
        id_list = list(memory_ids)
        rows = conn.execute(
            f"""SELECT source_id, target_id, relation_type, strength
                FROM memory_links
                WHERE source_id IN ({ph}) OR target_id IN ({ph})
                ORDER BY strength DESC""",
            id_list + id_list,
        ).fetchall()
        link_map = {}
        id_set = set(memory_ids)
        for src, tgt, rtype, strength in rows:
            for a, b in [(src, tgt), (tgt, src)]:
                if a in id_set:
                    link_map.setdefault(a, []).append({
                        "linked_id": b,
                        "relation": rtype,
                        "strength": round(strength or 0.5, 2),
                    })
        return link_map

    # --- Consolidation ---

    def store_nyx_insight(self, memory_id, source_ids, insight_type, consolidation_type="aggregation"):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO nyx_insights (memory_id, source_ids, insight_type, consolidation_type) "
            "VALUES (?, ?, ?, ?)",
            (memory_id, ",".join(str(s) for s in source_ids), insight_type, consolidation_type),
        )
        conn.commit()

    def get_merged_sources(self, memory_id, valid_only=True):
        """Return source memories that were merged into the given memory.

        If `valid_only` is True (default), filters out sources that have been
        explicitly superseded via `valid_until` set to a past date. This
        protects tier-2 recall from surfacing outdated facts that the
        consolidation pipeline marked as no longer current.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT source_ids, consolidation_type FROM nyx_insights WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if not row:
            return []
        source_ids = [int(x) for x in row["source_ids"].split(",")]
        if not source_ids:
            return []
        ph = ",".join("?" for _ in source_ids)
        where_extra = ""
        if valid_only:
            where_extra = (" AND (valid_until IS NULL OR valid_until > date('now', 'localtime'))"
                           " AND (valid_from IS NULL OR valid_from <= date('now', 'localtime'))")
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({ph}){where_extra} ORDER BY created_at DESC",
            source_ids,
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # --- Stats ---

    def stats(self, namespace=None):
        conn = self._get_conn()
        ns = namespace or self.namespace
        active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active' AND namespace=?", (ns,)
        ).fetchone()[0]
        archived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='archived' AND namespace=?", (ns,)
        ).fetchone()[0]
        by_project = dict(conn.execute(
            "SELECT project, COUNT(*) FROM memories WHERE status='active' AND namespace=? GROUP BY project",
            (ns,),
        ).fetchall())
        return {
            "active": active,
            "archived": archived,
            "total": active + archived,
            "by_project": by_project,
            "namespace": ns,
        }

    # --- Consolidation run logging ---

    def log_consolidation_run(self, clusters_found=0, clusters_merged=0,
                              memories_archived=0, memories_created=0,
                              details="", phase_details="{}",
                              access_decayed=0, importance_demoted=0):
        """Insert one audit row summarizing a Nyx cycle run.

        access_decayed/importance_demoted come from Phase 6 bookkeeping, so
        SQL-only runs (no LLM, no clusters) still record meaningful counts in
        the main columns instead of leaving an all-zero-looking row.

        Failures are swallowed: the audit log is a best-effort side
        channel, never a hard dependency of cycle correctness. Callers
        can rely on the run itself having completed even if logging
        fails for storage reasons.
        """
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO consolidation_log "
                "(clusters_found, clusters_merged, memories_archived, "
                " memories_created, access_decayed, importance_demoted, "
                " details, phase_details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (clusters_found, clusters_merged, memories_archived,
                 memories_created, access_decayed, importance_demoted,
                 details, phase_details),
            )
            conn.commit()
        except Exception:
            pass

    # --- Tool usage logging ---

    def log_tool_usage(self, tool_name):
        """Insert one row recording an MCP tool call.

        Failures swallowed - tool_usage is diagnostics only, never a hard
        dependency of server correctness.
        """
        if not tool_name:
            return
        try:
            conn = self._get_conn()
            # called_at is set explicitly rather than relying on the column
            # default: tables created by pre-package deployments declare
            # called_at NOT NULL with no default, and relying on the default
            # there makes every insert fail (and get swallowed) silently.
            conn.execute(
                "INSERT INTO tool_usage (tool_name, called_at) "
                "VALUES (?, datetime('now', 'localtime'))",
                (tool_name,),
            )
            conn.commit()
        except Exception:
            pass

    # --- Retrieval logging ---

    def log_retrieval(self, query, memory_ids, session_id=None):
        """Insert one row per returned memory for a search event.

        Called from Mnemos.search() when MNEMOS_RETRIEVAL_LOG=1. Failures
        are swallowed: logging is a best-effort side-channel, never a
        hard dependency of search correctness. The caller's search result
        is unaffected by logging outcomes.
        """
        if not query or not memory_ids:
            return
        try:
            conn = self._get_conn()
            conn.executemany(
                "INSERT INTO retrieval_log (memory_id, query, session_id) "
                "VALUES (?, ?, ?)",
                [(mid, query, session_id) for mid in memory_ids],
            )
            conn.commit()
        except Exception:
            pass

    # --- Tag discovery ---

    def list_tags(self, namespace=None, project=None, min_count=1,
                  order_by="count", limit=500):
        """Aggregate unique tags from the tags CSV column via a recursive
        CTE that splits the CSV server-side. Scales better than the
        previous Python-side fetch-all-rows-and-split approach for large
        deployments; at modest sizes (<5K memories) the difference is
        negligible either way.

        Tags are stored as comma-separated strings. The CTE walks each
        row's `tags` column, emits one row per tag, trims whitespace,
        drops empties. GROUP BY counts and MIN(id) picks the smallest-id
        memory per tag as the example anchor.
        """
        conn = self._get_conn()
        ns = namespace or self.namespace
        where = "m.status = 'active' AND m.namespace = ? AND m.tags != ''"
        params: list = [ns]
        if project:
            where += " AND m.project = ?"
            params.append(project)

        order_sql = (
            'ORDER BY LOWER(tag) ASC'
            if order_by == "alpha"
            else 'ORDER BY "count" DESC, LOWER(tag) ASC'
        )

        try:
            rows = conn.execute(
                f"""
                WITH RECURSIVE split(mid, tag, rest) AS (
                    -- Seed: one row per memory, tag empty, rest = full tags CSV
                    -- appended with ',' so the final tag has a comma to split on.
                    SELECT m.id, '', m.tags || ','
                    FROM memories m
                    WHERE {where}

                    UNION ALL

                    -- Recur: carve off the next tag up to the first comma,
                    -- leave the remainder for the next iteration.
                    SELECT mid,
                           substr(rest, 1, instr(rest, ',') - 1),
                           substr(rest, instr(rest, ',') + 1)
                    FROM split
                    WHERE rest != '' AND instr(rest, ',') > 0
                )
                SELECT TRIM(tag)         AS tag,
                       COUNT(*)          AS "count",
                       MIN(mid)          AS example_id
                FROM split
                WHERE TRIM(tag) != ''
                GROUP BY TRIM(tag)
                HAVING COUNT(*) >= ?
                {order_sql}
                LIMIT ?
                """,
                params + [min_count, limit],
            ).fetchall()
        except sqlite3.OperationalError:
            # Recursive CTE depth limit or similar - fall back to Python-side
            # aggregation for safety. SQLite's default recursion limit is
            # high (1000), so this should only trip on pathological tag
            # strings.
            return self._list_tags_python_fallback(
                ns, project, min_count, order_by, limit,
            )
        return [dict(r) for r in rows]

    def _list_tags_python_fallback(self, namespace, project, min_count,
                                   order_by, limit):
        """Python-side fallback path, identical semantics to the CTE query."""
        conn = self._get_conn()
        where = "status = 'active' AND namespace = ? AND tags != ''"
        params = [namespace]
        if project:
            where += " AND project = ?"
            params.append(project)
        rows = conn.execute(
            f"SELECT id, tags FROM memories WHERE {where}", params,
        ).fetchall()
        counts = {}
        examples = {}
        for r in rows:
            for raw in (r["tags"] or "").split(","):
                tag = raw.strip()
                if not tag:
                    continue
                counts[tag] = counts.get(tag, 0) + 1
                if tag not in examples or r["id"] < examples[tag]:
                    examples[tag] = r["id"]
        items = [
            {"tag": t, "cnt": c, "example_id": examples[t]}
            for t, c in counts.items() if c >= min_count
        ]
        if order_by == "alpha":
            items.sort(key=lambda x: x["tag"].lower())
        else:
            items.sort(key=lambda x: (-x["cnt"], x["tag"].lower()))
        # Normalize key naming to match CTE output
        return [
            {"tag": it["tag"], "count": it["cnt"], "example_id": it["example_id"]}
            for it in items[:limit]
        ]

    # --- Snippet extraction (FTS5 snippet() built-in) ---

    def get_snippets(self, ids, query, chars=200):
        """Return FTS5-extracted snippets for memory IDs that match the query.

        Uses SQLite's built-in snippet() function which returns a window
        around the match with configurable token count. Memories that
        don't match the FTS query are absent from the result (caller
        should fall back to content head).
        """
        if not ids:
            return {}
        from ..query import clean_fts_query
        conn = self._get_conn()
        fts_query = clean_fts_query(query, mode="OR")
        if not fts_query:
            return {}
        # snippet() token count is approximate. ~6 chars/token average.
        tokens = max(8, min(64, chars // 6))
        ph = ",".join("?" for _ in ids)
        params = [fts_query] + list(ids)
        try:
            rows = conn.execute(
                f"""SELECT fts.rowid AS id,
                           snippet(memories_fts, 0, '⟪', '⟫', ' … ', {tokens}) AS snip
                    FROM memories_fts fts
                    WHERE memories_fts MATCH ? AND fts.rowid IN ({ph})""",
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {r["id"]: r["snip"] for r in rows if r["snip"]}

    # --- Oversized memory remediation (v10.8.0) ---

    def find_oversized_memories(
        self,
        namespace: str,
        min_size: int = 4000,
        max_size: Optional[int] = None,
        include_archived: bool = False,
        limit: Optional[int] = None,
    ) -> list:
        import re as _re
        conn = self._get_conn()
        clause = "length(content) > ?"
        params = [min_size]
        if max_size:
            clause += " AND length(content) <= ?"
            params.append(max_size)
        params.append(namespace)
        status_clause = (
            "status IN ('active','archived')" if include_archived
            else "status='active'"
        )
        rows = conn.execute(
            f"SELECT id FROM memories WHERE {status_clause} "
            f"AND consolidation_lock=0 AND {clause} AND namespace=? "
            f"ORDER BY length(content) DESC",
            params,
        ).fetchall()
        ids = [r["id"] for r in rows]
        done = set()
        for r in conn.execute(
            "SELECT tags FROM memories "
            "WHERE tags LIKE '%split-from:#%' AND namespace=?",
            (namespace,),
        ):
            for gid in _re.findall(r"split-from:#(\d+)", r["tags"] or ""):
                done.add(int(gid))
        ids = [i for i in ids if i not in done]
        if limit:
            ids = ids[:limit]
        return ids

    # --- CML subject dedup ---

    def find_cml_subject_matches(
        self,
        namespace: str,
        project: str,
        pattern: str,
        limit: int = 5,
    ) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, content, project FROM memories "
            "WHERE status='active' AND namespace=? AND project=? "
            "AND content LIKE ? LIMIT ?",
            (namespace, project, pattern, limit),
        ).fetchall()
        return [
            {"id": r["id"], "content": r["content"], "project": r["project"]}
            for r in rows
        ]

    # --- Retrieval log useful-loop (v10.17.0) ---

    def mark_retrieval_useful(self, memory_id: int) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "UPDATE retrieval_log SET useful = 1 "
                "WHERE memory_id = ? AND useful IS NULL "
                "AND retrieved_at >= datetime('now', 'localtime', '-24 hours')",
                (memory_id,),
            )
            conn.commit()
        except Exception:
            pass

    # --- Content-based memory search (used by bulk_rewrite) ---

    def find_memories_by_content(
        self,
        namespace: str,
        project: Optional[str] = None,
        tags: Optional[str] = None,
        content_pattern: Optional[str] = None,
    ) -> list:
        conn = self._get_conn()
        where = "status = 'active' AND namespace = ?"
        params: list = [namespace]
        if project:
            where += " AND project = ?"
            params.append(project)
        if tags:
            where += " AND (',' || tags || ',') LIKE ?"
            params.append(f"%,{tags},%")
        if content_pattern:
            where += " AND content LIKE ?"
            params.append(content_pattern)
        rows = conn.execute(
            f"SELECT id, content FROM memories WHERE {where}", params,
        ).fetchall()
        return [{"id": r["id"], "content": r["content"]} for r in rows]

    # --- Briefing ---

    def get_briefing_memories(
        self,
        namespace: str,
        project: Optional[str] = None,
        limit: int = 30,
    ) -> list:
        conn = self._get_conn()
        where = "WHERE m.status = 'active' AND m.namespace = ?"
        params = [namespace]
        if project:
            where += " AND m.project = ?"
            params.append(project)
        try:
            rows = conn.execute(
                f"SELECT id, project, content, importance, type "
                f"FROM memories m {where} "
                f"ORDER BY importance DESC, last_accessed DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        except Exception:
            return []
        return [
            {"id": r["id"], "project": r["project"], "content": r["content"],
             "importance": r["importance"], "type": r["type"]}
            for r in rows
        ]

    # --- Digest ---

    def get_digest_memories(
        self,
        namespace: str,
        days: int = 7,
        project: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        conn = self._get_conn()
        where = (
            "WHERE status = 'active' AND namespace = ? "
            "AND julianday('now', 'localtime') - julianday(created_at) <= ?"
        )
        params = [namespace, days]
        if project:
            where += " AND project = ?"
            params.append(project)
        rows = conn.execute(
            f"SELECT id, project, content, created_at, importance "
            f"FROM memories {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Project map ---

    def get_project_map(self, namespace: str) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT project, subcategory, COUNT(*) as n "
            "FROM memories WHERE status='active' AND namespace=? "
            "GROUP BY project, subcategory ORDER BY project, n DESC",
            (namespace,),
        ).fetchall()
        return [
            {"project": r["project"], "subcategory": r["subcategory"], "n": r["n"]}
            for r in rows
        ]

    # --- Embedding backfill (used by embed_fill) ---

    def get_unembedded_memories(
        self,
        namespace: str,
        limit: Optional[int] = None,
    ) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT m.id, m.project, m.content, m.tags, m.type, m.layer "
            "FROM memories m WHERE m.status='active' AND m.namespace=? "
            "AND NOT EXISTS (SELECT 1 FROM embed_meta em "
            " WHERE em.source_db='memory' AND em.source_id=m.id) "
            "ORDER BY m.id",
            (namespace,),
        ).fetchall()
        if limit:
            rows = rows[:limit]
        return [
            {"id": r["id"], "project": r["project"], "content": r["content"],
             "tags": r["tags"], "type": r["type"], "layer": r["layer"]}
            for r in rows
        ]

    # --- Embedding coverage (used by embed_status) ---

    def get_embed_coverage(self, namespace: str) -> dict:
        from ..embed import text_hash as embed_text_hash, prep_memory_text
        conn = self._get_conn()
        active = conn.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE status='active' AND namespace=?",
            (namespace,),
        ).fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM embed_meta em "
            "JOIN memories m ON m.id = em.source_id "
            "WHERE em.source_db = 'memory' AND m.status='active' "
            "AND m.namespace=?",
            (namespace,),
        ).fetchone()[0]
        stale = 0
        unverified = 0
        rows = conn.execute(
            "SELECT m.project, m.content, m.tags, m.type, m.layer, "
            "em.text_hash FROM embed_meta em "
            "JOIN memories m ON m.id = em.source_id "
            "WHERE em.source_db = 'memory' AND m.status='active' "
            "AND m.namespace=?",
            (namespace,),
        ).fetchall()
        for r in rows:
            if not r["text_hash"]:
                unverified += 1
                continue
            current = embed_text_hash(prep_memory_text(
                r["project"], r["content"], r["tags"] or "",
                mem_type=r["type"] or "", layer=r["layer"] or "",
            ))
            stored = r["text_hash"]
            if current != stored and not current.startswith(stored):
                stale += 1
        return {
            "active": active,
            "embedded": embedded,
            "missing": active - embedded,
            "stale": stale,
            "unverified": unverified,
            "coverage": round(embedded / active, 4) if active else 0.0,
        }

    # --- Health check (used by doctor) ---

    def health_check(self, namespace: str, migrate: bool = False) -> dict:
        conn = self._get_conn()
        report = {
            "namespace": namespace,
            "checks": [],
            "issues": [],
            "migrations_applied": [],
        }

        # --- Integrity check ---
        try:
            from ..core import _summarize_quick_check
            ok, summary = _summarize_quick_check(
                conn.execute("PRAGMA quick_check").fetchall()
            )
            if ok:
                report["checks"].append("Integrity check passed (quick_check)")
            else:
                report["issues"].append(f"Integrity check FAILED: {summary}")
        except Exception as e:
            report["issues"].append(f"Integrity check error: {e}")

        # --- Schema drift detection ---
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(memories)").fetchall()}
        required = {"id", "project", "content", "type", "layer",
                    "subcategory", "valid_from", "valid_until", "namespace"}
        missing_cols = required - cols
        if missing_cols:
            report["issues"].append(
                f"Missing columns in memories: {sorted(missing_cols)}")
        else:
            report["checks"].append("Schema is up to date (v10)")

        # --- Empty-store detection (v10.23.0) ---
        try:
            active = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE status = 'active' AND namespace = ?",
                (namespace,)).fetchone()[0]
            if active > 0:
                report["checks"].append(
                    f"Store populated: {active} active memories "
                    f"in namespace '{namespace}'")
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM memories").fetchone()[0]
                if total == 0:
                    report["issues"].append(
                        "Store is empty: 0 memories in this database. "
                        "Expected only for a brand-new store; otherwise "
                        "MNEMOS_DB is probably pointing at the wrong file")
                else:
                    rows = conn.execute(
                        "SELECT namespace, COUNT(*) AS n FROM memories "
                        "GROUP BY namespace ORDER BY n DESC").fetchall()
                    listing = ", ".join(
                        f"'{r[0]}': {r[1]}" for r in rows)
                    report["issues"].append(
                        f"No active memories in namespace "
                        f"'{namespace}' but the database holds "
                        f"{total} memories ({listing}). Check "
                        "MNEMOS_NAMESPACE if these should be visible here")
        except Exception as e:
            report["issues"].append(f"Empty-store check failed: {e}")

        # --- Missing aux tables ---
        existing_tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected_aux = {"retrieval_log", "tool_usage",
                        "consolidation_log", "nyx_state"}
        missing_tables = expected_aux - existing_tables
        if missing_tables:
            report["issues"].append(
                f"Missing aux tables: {sorted(missing_tables)}")

        # --- Backup before migrations ---
        if migrate and (missing_cols or missing_tables) and \
                hasattr(self, "db_path"):
            import time
            src = self.db_path
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = f"{src}.bak-pre-doctor-migrate-{ts}"
            try:
                self.backup(backup)
                report["backup"] = backup
            except Exception as e:
                report["issues"].append(
                    f"Backup failed; aborting migrations: {e}")
                migrate = False

        # --- Column migration ---
        if migrate and missing_cols:
            try:
                self.init_schema()
                post_cols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(memories)").fetchall()}
                backfilled = sorted(missing_cols & post_cols)
                if backfilled:
                    report["migrations_applied"].append(
                        f"Backfilled columns: {backfilled}")
            except Exception as e:
                report["issues"].append(f"Column migration failed: {e}")

        # --- Table migration ---
        if migrate and missing_tables:
            try:
                self.init_schema()
                post_tables = {
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table'").fetchall()
                }
                created = sorted(missing_tables & post_tables)
                if created:
                    report["migrations_applied"].append(
                        f"Created aux tables: {created}")
            except Exception as e:
                report["issues"].append(f"Table migration failed: {e}")

        # --- FTS sanity ---
        mem_count = conn.execute(
            "SELECT COUNT(*) FROM memories").fetchone()[0]
        try:
            fts_count = conn.execute(
                "SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            if mem_count != fts_count:
                report["issues"].append(
                    f"FTS index out of sync: {mem_count} memories "
                    f"vs {fts_count} FTS rows")
                if migrate:
                    try:
                        conn.execute(
                            "INSERT INTO memories_fts(memories_fts) "
                            "VALUES('rebuild')")
                        conn.commit()
                        new_count = conn.execute(
                            "SELECT COUNT(*) FROM memories_fts"
                        ).fetchone()[0]
                        report["migrations_applied"].append(
                            f"Rebuilt FTS index ({new_count} rows)")
                    except Exception as e:
                        report["issues"].append(f"FTS rebuild failed: {e}")
            else:
                report["checks"].append(f"FTS index synced ({fts_count} rows)")
        except Exception as e:
            report["issues"].append(f"FTS check failed: {e}")

        # --- Re-check issues post-migration ---
        if migrate and report["migrations_applied"]:
            post_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(memories)").fetchall()}
            post_tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table'").fetchall()
            }
            resolved = []
            for issue in report["issues"]:
                if "Missing columns" in issue and not (required - post_cols):
                    continue
                if "Missing aux tables" in issue and \
                        not (expected_aux - post_tables):
                    continue
                resolved.append(issue)
            report["issues"] = resolved

        return report

    # --- Content/vector coherence (v10.19.0) ---

    def get_coherence_mismatches(self, namespace: str) -> dict:
        from ..embed import text_hash as embed_text_hash, prep_memory_text
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT m.id, m.project, m.content, m.tags, m.type, m.layer, "
            "e.text_hash FROM memories m JOIN embed_meta e "
            "ON e.source_db='memory' AND e.source_id = m.id "
            "WHERE m.status='active' AND m.namespace = ?",
            (namespace,),
        ).fetchall()
        mismatched = []
        checkable = 0
        for r in rows:
            if not r["text_hash"]:
                continue
            checkable += 1
            expect = embed_text_hash(prep_memory_text(
                r["project"], r["content"] or "", r["tags"] or "",
                mem_type=r["type"] or "", layer=r["layer"] or ""))
            if expect != r["text_hash"]:
                mismatched.append(r["id"])
        return {"checkable": checkable, "mismatched_ids": mismatched}

    def reembed_mismatched(self, namespace: str, mismatched_ids: list,
                           embed_fn, text_hash_fn, prep_fn) -> int:
        conn = self._get_conn()
        repaired = 0
        for mid in mismatched_ids:
            r = conn.execute(
                "SELECT project, content, tags, type, layer "
                "FROM memories WHERE id = ?", (mid,)).fetchone()
            text = prep_fn(
                r["project"], r["content"] or "", r["tags"] or "",
                mem_type=r["type"] or "", layer=r["layer"] or "")
            vecs = embed_fn([text], prefix="passage")
            if vecs and vecs[0]:
                self._store_embedding(
                    mid, vecs[0], text_hash=text_hash_fn(text))
                repaired += 1
        return repaired

    # --- Vector provenance ---

    def get_vector_provenance(self) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT model, COUNT(*) AS n FROM embed_meta "
            "WHERE source_db = 'memory' GROUP BY model"
        ).fetchall()
        return [{"model": r["model"], "count": r["n"]} for r in rows]

    # --- Archive-side embedding lifecycle (v10.24.0) ---

    def check_archive_lifecycle(self, namespace: str, migrate: bool,
                                report: dict) -> None:
        """Check and optionally migrate archive-side embedding drift.

        Detects: UTC-dialect embed_meta tables, orphan tier-2 rows from
        pre-v10.24 hard deletes, and incomplete tier-2 archive index.
        Mutates `report` in place.
        """
        conn = self._get_conn()
        existing_tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        # Timestamp dialect: detected from the table DDL itself, so the
        # check is idempotent and needs no migration marker.
        dialect_tables = []
        for table in ("embed_meta", "embed_meta_arch"):
            if table not in existing_tables:
                continue
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()
            if ddl and ddl[0] and "localtime" not in ddl[0]:
                dialect_tables.append(table)

        arch_orphans = []
        if "embed_meta_arch" in existing_tables:
            arch_orphans = conn.execute(
                "SELECT e.id, e.source_id FROM embed_meta_arch e "
                "LEFT JOIN memories m ON m.id = e.source_id "
                "WHERE m.id IS NULL"
            ).fetchall()

        # Backup before these migrations too (the early backup only
        # triggers on column/table drift).
        if migrate and (dialect_tables or arch_orphans) \
                and "backup" not in report and hasattr(self, "db_path"):
            import time as _time
            backup = (f"{self.db_path}"
                      f".bak-pre-doctor-migrate-{_time.strftime('%Y%m%d-%H%M%S')}")
            try:
                self.backup(backup)
                report["backup"] = backup
            except Exception as e:
                report["issues"].append(
                    f"Backup failed; skipping archive-lifecycle migrations: {e}")
                dialect_tables, arch_orphans = [], []

        if dialect_tables and migrate:
            for table in dialect_tables:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_utc_old")
                conn.execute(f"""
                    CREATE TABLE {table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_db TEXT NOT NULL,
                        source_id INTEGER NOT NULL,
                        text_hash TEXT,
                        model TEXT,
                        embedded_at TEXT DEFAULT (datetime('now', 'localtime')),
                        UNIQUE(source_db, source_id)
                    )""")
                conn.execute(
                    f"INSERT INTO {table} (id, source_db, source_id, text_hash, model, embedded_at) "
                    f"SELECT id, source_db, source_id, text_hash, model, "
                    f"CASE WHEN embedded_at IS NULL THEN NULL "
                    f"ELSE datetime(embedded_at, 'localtime') END "
                    f"FROM {table}_utc_old")
                conn.execute(f"DROP TABLE {table}_utc_old")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_source "
                         "ON embed_meta(source_db, source_id)")
            conn.commit()
            report["migrations_applied"].append(
                f"Converted embedded_at to localtime and rebuilt with "
                f"localtime default: {dialect_tables}")
        elif dialect_tables:
            report["issues"].append(
                f"embedded_at stored in UTC dialect in {dialect_tables} "
                "while the memories table uses localtime; timestamps "
                "cross-compare wrong by the UTC offset. Run doctor "
                "--migrate to convert")

        if arch_orphans and migrate:
            arch_col = _arch_join_col(conn)
            for row in arch_orphans:
                conn.execute(
                    f"DELETE FROM embed_vec_arch WHERE {arch_col} = ?",
                    (row[0],))
                conn.execute(
                    "DELETE FROM embed_meta_arch WHERE id = ?", (row[0],))
            conn.commit()
            report["migrations_applied"].append(
                f"Removed {len(arch_orphans)} orphan tier-2 embedding rows "
                "(memories hard-deleted before v10.24.0 left them behind)")
        elif arch_orphans:
            report["issues"].append(
                f"{len(arch_orphans)} orphan arch embedding rows whose "
                "memory no longer exists (pre-v10.24 hard-delete leak). "
                "Run doctor --migrate to clean")

        if "embed_meta_arch" in existing_tables:
            gaps = conn.execute(
                "SELECT COUNT(*) FROM memories m "
                "WHERE m.status = 'archived' AND m.namespace = ? "
                "AND NOT EXISTS (SELECT 1 FROM embed_meta_arch em "
                "WHERE em.source_db = 'memory' AND em.source_id = m.id)",
                (namespace,)).fetchone()[0]
            legacy = conn.execute(
                "SELECT COUNT(*) FROM memories m "
                "JOIN embed_meta_arch em ON em.source_db = 'memory' "
                "AND em.source_id = m.id "
                "WHERE m.status = 'archived' AND m.namespace = ? "
                "AND (em.text_hash IS NULL OR length(em.text_hash) < 64)",
                (namespace,)).fetchone()[0]
            if gaps or legacy:
                # Never auto-embedded here, same policy as active-side
                # coverage: embedding cost/time is caller-controlled.
                report["issues"].append(
                    f"Tier-2 archive index incomplete: {gaps} archived "
                    f"memories without a vector, {legacy} with a legacy "
                    "pre-canonical hash. Run `mnemos reindex-archived`")
            else:
                arch_n = conn.execute(
                    "SELECT COUNT(*) FROM embed_meta_arch "
                    "WHERE source_db = 'memory'").fetchone()[0]
                report["checks"].append(
                    f"Tier-2 archive index: complete ({arch_n} vectors)")

    # --- Helpers ---

    @staticmethod
    def _row_to_memory(row) -> Memory:
        return Memory(
            id=row["id"],
            namespace=row["namespace"],
            project=row["project"],
            content=row["content"],
            tags=row["tags"] or "",
            importance=row["importance"],
            status=row["status"],
            type=row["type"] or "fact",
            layer=row["layer"] or "semantic",
            verified=row["verified"] or 0,
            consolidation_lock=row["consolidation_lock"] or 0,
            subcategory=row["subcategory"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            access_count=row["access_count"] or 0,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            last_confirmed=row["last_confirmed"],
        )
