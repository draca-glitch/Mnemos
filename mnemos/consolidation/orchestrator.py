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
    """Ensure Nyx-cycle-adjacent schema exists. Mostly a no-op post-v10.2.1:
    consolidation_log and nyx_state now live in SQLiteStore.init_schema so
    they exist from first DB connection (not just first Nyx run). This
    function is kept as a safety net for older DBs that predate v10.2.1
    and may not have seen a fresh init, plus for the idx_links_type index
    which is not redundantly created elsewhere.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT DEFAULT (datetime('now', 'localtime')),
            clusters_found INTEGER DEFAULT 0,
            clusters_merged INTEGER DEFAULT 0,
            memories_archived INTEGER DEFAULT 0,
            memories_created INTEGER DEFAULT 0,
            details TEXT DEFAULT '',
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

def _vec_join_col(conn):
    """Detect whether embed_vec uses 'id' (explicit PK) or 'rowid' (implicit).

    Matches the auto-detection logic in storage/sqlite_store.py so
    consolidation works against both schemas.
    """
    try:
        conn.execute("SELECT id FROM embed_vec LIMIT 0").fetchone()
        return "id"
    except sqlite3.OperationalError:
        return "rowid"


def cleanup_orphan_vectors(conn):
    """Remove vectors for archived/deleted memories."""
    active_ids = {r[0] for r in conn.execute(
        "SELECT id FROM memories WHERE status = 'active'"
    ).fetchall()}
    embedded = conn.execute(
        "SELECT id, source_id FROM embed_meta WHERE source_db = ?", (SOURCE_KEY,)
    ).fetchall()
    join_col = _vec_join_col(conn)
    removed = 0
    for meta_id, source_id in embedded:
        if source_id not in active_ids:
            conn.execute(f"DELETE FROM embed_vec WHERE {join_col} = ?", (meta_id,))
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


# --- Phase 0.5: Cemelify (v10.4.0, requires LLM) ---

def _phase_cemelify(conn, store, mem_by_id, execute: bool = False):
    """Rewrite non-CML or over-long active memories into CML form via LLM.

    Candidates: content does not start with a CML prefix (F:/D:/C:/L:/P:/W:),
    OR content length > 800 chars. Skips memories with consolidation_lock=1
    (prose-protection convention, matches Phase 2 Dedup semantics).

    Each candidate is passed through cemelify(), which falls back to the
    original on LLM failure. We detect "no change" and skip the write to
    avoid spurious updated_at churn. Updates persist via store.update_memory;
    re-embed happens on the next bookkeeping pass (deferred by design to keep
    this phase cheap).

    Returns a stats dict: candidates / cemelified / unchanged / failed.
    """
    from ..cemelify import cemelify, _needs_cemelify

    candidates = []
    for mid, m in mem_by_id.items():
        if m.get("consolidation_lock"):
            continue
        if _needs_cemelify(m.get("content")):
            candidates.append(mid)

    log(f"Phase 0.5 (Cemelify): {len(candidates)} candidates")
    if not candidates:
        return {"candidates": 0, "cemelified": 0, "unchanged": 0, "failed": 0}

    cemelified = unchanged = failed = 0
    for i, mid in enumerate(candidates):
        if i and i % 100 == 0:
            log(f"  Phase 0.5 progress: {i}/{len(candidates)}")
        original = mem_by_id[mid].get("content") or ""
        try:
            new_content = cemelify(original)
        except Exception:
            failed += 1
            continue
        if not new_content or new_content == original:
            unchanged += 1
            continue
        if execute:
            try:
                store.update_memory(mid, {"content": new_content})
                cemelified += 1
                # Reflect change so downstream phases (Dedup, Weave) in this
                # same run operate on the cemelified content.
                mem_by_id[mid]["content"] = new_content
            except Exception:
                failed += 1
        else:
            cemelified += 1  # dry-run accounting

    log(f"Phase 0.5 (Cemelify) complete: cemelified={cemelified}, "
        f"unchanged={unchanged}, failed={failed}")
    return {
        "candidates": len(candidates),
        "cemelified": cemelified,
        "unchanged": unchanged,
        "failed": failed,
    }


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

    import os as _os
    disable_llm = _os.environ.get("MNEMOS_DISABLE_LLM") == "1"
    has_llm = llm_is_configured() and not disable_llm
    llm_phases = {2, 3, 4, 5}  # phases that need an LLM

    if not has_llm and (phases & llm_phases):
        # v10.4.0: loud-fail by default. Silent degradation is dangerous
        # (users assume Nyx ran). MNEMOS_DISABLE_LLM=1 explicitly opts into
        # SQL-only operation for deployments that intentionally do not
        # configure an LLM.
        if disable_llm:
            phases = phases - llm_phases
        else:
            raise RuntimeError(
                "No LLM configured (MNEMOS_LLM_API_KEY required, and "
                "MNEMOS_LLM_MODEL required for non-OpenAI endpoints). "
                f"Nyx cycle would skip phases {sorted(phases & llm_phases)}. "
                "Set the env vars, or pass MNEMOS_DISABLE_LLM=1 to run "
                "SQL-only phases (Phase 6 bookkeeping) intentionally."
            )

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

            # Phase 0.5: Cemelify (v10.4.0). Rewrites prose-form active
            # memories into CML before clustering, so Dedup operates on
            # canonical content. Skips consolidation_lock=1 memories.
            phase_stats["phase0_5"] = _phase_cemelify(
                conn, store, mem_by_id, execute=execute
            )

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
            store.log_consolidation_run(
                clusters_found=p1.get("tight_found", 0) + p1.get("topic_found", 0),
                clusters_merged=p1.get("tight_merged", 0) + p1.get("topic_merged", 0),
                memories_archived=p1.get("archived", 0),
                memories_created=p1.get("created", 0),
                details=f"mnemos-nyx-cycle, phases={sorted(phases)}",
                phase_details=json.dumps(phase_stats, default=str),
            )
        except Exception as e:
            log(f"Warning: failed to log consolidation run: {e}")

    return phase_stats
