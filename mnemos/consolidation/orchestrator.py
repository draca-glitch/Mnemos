"""
Nyx cycle orchestrator for Mnemos.

Runs the 6-phase consolidation pipeline. Phase 6 (Bookkeeping) always runs
since it's pure SQL. Phases 1-4 require an LLM and are skipped automatically
if no LLM is configured (see mnemos.consolidation.llm).

Usage:
    from mnemos.consolidation import run_nyx_cycle
    from mnemos.storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    stats = run_nyx_cycle(store, execute=True)
"""

import json
import sqlite3
import time

from .llm import is_configured as llm_is_configured
from ..constants import IMPORTANCE_THRESHOLDS

SOURCE_KEY = "memory"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --- Schema migration ---

def _migrate_nyx_schema(conn):
    """Add Nyx cycle tables. Safe to run repeatedly."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT DEFAULT (datetime('now', 'localtime')),
            clusters_found INTEGER DEFAULT 0,
            clusters_merged INTEGER DEFAULT 0,
            memories_archived INTEGER DEFAULT 0,
            memories_created INTEGER DEFAULT 0,
            details TEXT,
            phase_details TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nyx_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links(relation_type)")
    conn.commit()


# --- Phase 6: Bookkeeping (no LLM required) ---

def cleanup_orphan_vectors(conn):
    """Remove vectors for archived/deleted memories."""
    active_ids = {r[0] for r in conn.execute(
        "SELECT id FROM memories WHERE status = 'active'"
    ).fetchall()}
    embedded = conn.execute(
        "SELECT id, source_id FROM embed_meta WHERE source_db = ?", (SOURCE_KEY,)
    ).fetchall()
    removed = 0
    for meta_id, source_id in embedded:
        if source_id not in active_ids:
            conn.execute("DELETE FROM embed_vec WHERE rowid = ?", (meta_id,))
            conn.execute("DELETE FROM embed_meta WHERE id = ?", (meta_id,))
            removed += 1
    if removed:
        conn.commit()
    return removed


def decay_access_counts(conn):
    """Decay access_count by 1 for memories not accessed in 7+ days."""
    rows = conn.execute("""
        SELECT id, access_count, importance FROM memories
        WHERE status = 'active' AND access_count > 0
          AND (last_accessed IS NULL
               OR julianday('now','localtime') - julianday(last_accessed) >= 7)
    """).fetchall()

    decayed = demoted = 0
    for mid, count, imp in rows:
        new_count = count - 1
        target = 5
        for thresh, level in IMPORTANCE_THRESHOLDS:
            if new_count >= thresh:
                target = level
                break
        if imp in (6, 7, 8) and imp > target:
            conn.execute(
                "UPDATE memories SET access_count = ?, importance = ? WHERE id = ?",
                (new_count, target, mid),
            )
            demoted += 1
        else:
            conn.execute("UPDATE memories SET access_count = ? WHERE id = ?", (new_count, mid))
        decayed += 1
    if decayed:
        conn.commit()
    return decayed, demoted


def cleanup_stale_links(conn):
    """Remove links where either endpoint is archived."""
    try:
        n = conn.execute("""
            DELETE FROM memory_links
            WHERE source_id NOT IN (SELECT id FROM memories WHERE status='active')
               OR target_id NOT IN (SELECT id FROM memories WHERE status='active')
        """).rowcount
        if n:
            conn.commit()
        return n
    except Exception:
        return 0


# --- Main orchestrator ---

def run_nyx_cycle(
    store,
    execute: bool = False,
    phases: "set[int] | None" = None,
    surge: bool = False,
    project: "str | None" = None,
):
    """Run the Nyx cycle.

    Args:
        store: An SQLiteStore (or anything that wraps one, like QdrantStore)
        execute: If False, dry run only
        phases: Set of phase numbers to run. Default {0,1,2,3,5}, add 4 for synthesis
        surge: Force surge mode (higher LLM call limits)
        project: Filter Phase 2 (Dedup) to this project

    Returns: dict of phase stats
    """
    # Get the underlying SQLite connection (works for SQLiteStore and QdrantStore
    if hasattr(store, "_get_conn"):
        conn = store._get_conn()
    elif hasattr(store, "_sqlite") and hasattr(store._sqlite, "_get_conn"):
        conn = store._sqlite._get_conn()
    else:
        raise ValueError(
            "run_nyx_cycle requires a backend with a SQLite connection. "
            "SQLiteStore or QdrantStore are supported)."
        )

    conn.row_factory = sqlite3.Row
    _migrate_nyx_schema(conn)

    if phases is None:
        phases = {1, 2, 3, 4, 6}  # default: consolidation, no synthesis

    has_llm = llm_is_configured()
    llm_phases = {2, 3, 4, 5}  # phases that need an LLM

    if not has_llm and (phases & llm_phases):
        log("No LLM configured (MNEMOS_LLM_API_KEY and MNEMOS_LLM_MODEL both required), skipping LLM-dependent phases.")
        log(f"  Skipped phases: {sorted(phases & llm_phases)}")
        log("  Phase 6 (Bookkeeping) will still run.")
        phases = phases - llm_phases

    mode = "EXECUTE" if execute else "DRY RUN"
    log(f"Mnemos Nyx Cycle starting ({mode}, phases={sorted(phases)})")
    start_time = time.time()

    phase_stats = {}

    # --- LLM-dependent phases ---
    if phases & llm_phases:
        from .phases import (
            load_embeddings, phase_triage, phase_dedup,
            phase_weave, phase_contradict, phase_synthesize,
            _is_nyx_generated,
        )

        last_run = conn.execute(
            "SELECT MAX(run_at) FROM consolidation_log"
        ).fetchone()[0]
        log(f"Last run: {last_run or 'never'}")

        log("Loading embeddings...")
        all_embeddings, mergeable_embeddings, mem_by_id = load_embeddings(
            conn, project=project
        )

        if len(all_embeddings) < 2:
            log(f"Only {len(all_embeddings)} embeddings, need >=2 for clustering. Skipping LLM phases.")
        else:
            new_ids = []
            is_surge = surge
            if 1 in phases:
                new_ids, surge_detected = phase_triage(conn, mem_by_id, last_run)
                is_surge = is_surge or surge_detected

            if 2 in phases:
                phase_stats["phase2"] = phase_dedup(
                    conn, mergeable_embeddings, mem_by_id, is_surge, execute=execute
                )
                if execute:
                    p1 = phase_stats["phase2"]
                    if (p1.get("tight_merged", 0) + p1.get("topic_merged", 0)) > 0:
                        log("Reloading embeddings after merges...")
                        all_embeddings, mergeable_embeddings, mem_by_id = load_embeddings(
                            conn, project=project
                        )

            if 3 in phases:
                phase_stats["phase3"] = phase_weave(
                    conn, all_embeddings, mem_by_id, is_surge, execute=execute
                )

            if 4 in phases:
                phase_stats["phase4"] = phase_contradict(
                    conn, mergeable_embeddings, mem_by_id, is_surge, execute=execute
                )

            if 5 in phases:
                # Only synthesize if enough new ORIGINAL material
                should_synth = True
                if not is_surge and last_run:
                    new_originals = [
                        mid for mid in new_ids
                        if not _is_nyx_generated(mem_by_id.get(mid, {}))
                    ]
                    if len(new_originals) < 5:
                        log(f"Phase 5: Skipping synthesis ({len(new_originals)} new originals, need 5+)")
                        should_synth = False
                if should_synth:
                    phase_stats["phase5"] = phase_synthesize(
                        conn, all_embeddings, mem_by_id, execute=execute
                    )

    # --- Phase 6: Bookkeeping (always runs) ---
    if 6 in phases:
        log("Phase 6: Bookkeeping...")
        orphaned = cleanup_orphan_vectors(conn)
        if orphaned:
            log(f"  Cleaned {orphaned} orphaned vectors")

        decayed, demoted = decay_access_counts(conn)
        if decayed:
            log(f"  Decayed {decayed} access counts ({demoted} demoted)")

        stale_links = cleanup_stale_links(conn)
        if stale_links:
            log(f"  Removed {stale_links} stale links")

        phase_stats["phase6"] = {
            "orphans_cleaned": orphaned,
            "access_decayed": decayed,
            "importance_demoted": demoted,
            "stale_links_removed": stale_links,
        }

    elapsed = time.time() - start_time
    log(f"Nyx cycle complete in {elapsed:.1f}s")

    if execute:
        p1 = phase_stats.get("phase2", {})
        try:
            conn.execute(
                "INSERT INTO consolidation_log "
                "(clusters_found, clusters_merged, memories_archived, "
                " memories_created, details, phase_details) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    p1.get("tight_found", 0) + p1.get("topic_found", 0),
                    p1.get("tight_merged", 0) + p1.get("topic_merged", 0),
                    p1.get("archived", 0),
                    p1.get("created", 0),
                    f"mnemos-nyx-cycle, phases={sorted(phases)}",
                    json.dumps(phase_stats, default=str),
                ),
            )
            conn.commit()
        except Exception as e:
            log(f"Warning: failed to log consolidation run: {e}")

    return phase_stats
