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
    DEFAULT_DB_PATH, DEFAULT_NAMESPACE, FASTEMBED_DIMS,
    BM25_WEIGHTS, RANKING_ORDER_SQL, BM25_CALL,
)


def _serialize_vec(vec):
    """Pack a Python list/tuple of floats into the binary format sqlite-vec expects."""
    return struct.pack(f"{len(vec)}f", *vec)


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
        # Indexes
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
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
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
        conn.commit()

    # --- CRUD ---

    def store_memory(self, memory: Memory, embedding: Optional[list] = None) -> int:
        conn = self._get_conn()
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
        conn.commit()
        if embedding is not None:
            self._store_embedding(mid, embedding)
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

    def _store_embedding(self, mid: int, embedding: list):
        """Insert or replace the embedding for a memory.

        Handles both embed_vec schemas (implicit rowid vs explicit id)
        so the same code works against fresh Mnemos-created DBs and
        against DBs created by other tooling that declares an explicit
        `id INTEGER PRIMARY KEY` column on the vec0 virtual table.
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
                "INSERT INTO embed_meta (source_db, source_id) VALUES (?, ?)",
                (self.SOURCE_KEY, mid),
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
                "INSERT INTO embed_meta (id, source_db, source_id) VALUES (?, ?, ?)",
                (vec_id, self.SOURCE_KEY, mid),
            )
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

    def update_memory(self, mid: int, fields: dict, embedding: Optional[list] = None) -> bool:
        conn = self._get_conn()
        # Filter to known columns so an unexpected key cannot inject SQL
        safe_fields = {k: v for k, v in (fields or {}).items() if k in self._UPDATABLE_COLUMNS}
        if not safe_fields and embedding is None:
            return False
        rowcount = 0
        if safe_fields:
            set_clause = ", ".join(f"{k} = ?" for k in safe_fields.keys())
            params = list(safe_fields.values()) + [mid]
            cur = conn.execute(
                f"UPDATE memories SET {set_clause}, updated_at = datetime('now', 'localtime') WHERE id = ?",
                params,
            )
            rowcount = cur.rowcount
            conn.commit()
        else:
            # Embedding-only update, verify the row exists so we don't lie about success
            row = conn.execute("SELECT 1 FROM memories WHERE id = ?", (mid,)).fetchone()
            if row is None:
                return False
            rowcount = 1
        if embedding is not None:
            self._store_embedding(mid, embedding)
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
            conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
        else:
            conn.execute(
                "UPDATE memories SET status = 'archived', updated_at = datetime('now', 'localtime') WHERE id = ?",
                (mid,),
            )
        conn.commit()
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
                where += " AND (m.valid_until IS NULL OR m.valid_until >= date('now', 'localtime'))"
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
            filter_clause += " AND (valid_until IS NULL OR valid_until >= date('now', 'localtime'))"

        active = set(
            r[0] for r in conn.execute(
                f"SELECT id FROM memories WHERE id IN ({ph}){filter_clause}",
                filter_params,
            ).fetchall()
        )
        results = [(r["source_id"], r["distance"]) for r in rows if r["source_id"] in active]
        return results[:limit]

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
            where_extra = " AND (valid_until IS NULL OR valid_until >= date('now', 'localtime'))"
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
        """Aggregate unique tags from the tags CSV column across active memories.

        Tags are stored as comma-separated strings. This method splits,
        normalizes (strip whitespace, drop empties), aggregates counts,
        and returns one row per unique tag with an example memory ID.
        """
        conn = self._get_conn()
        ns = namespace or self.namespace
        where = "status = 'active' AND namespace = ? AND tags != ''"
        params = [ns]
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
                examples.setdefault(tag, r["id"])
        items = [
            {"tag": t, "count": c, "example_id": examples[t]}
            for t, c in counts.items() if c >= min_count
        ]
        if order_by == "alpha":
            items.sort(key=lambda x: x["tag"].lower())
        else:
            items.sort(key=lambda x: (-x["count"], x["tag"].lower()))
        return items[:limit]

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
