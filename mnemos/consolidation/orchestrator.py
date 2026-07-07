"""
Nyx cycle orchestrator for Mnemos.

Runs the Nyx consolidation pipeline. Phases 1 (triage), 2 (dedup with the
mechanical engine), 4 (contradiction finder in queue mode) and 6
(bookkeeping) run with zero LLM calls; phases 3 (weave) and 5 (synthesize),
plus the phase-4 llm judge, form the optional LLM tier and are skipped with
a warning when no LLM is configured (see mnemos.consolidation.llm).

Usage:
    from mnemos.consolidation import run_nyx_cycle
    from mnemos.storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    stats = run_nyx_cycle(store, execute=True)
"""

import json
import os
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
            access_decayed INTEGER DEFAULT 0,
            importance_demoted INTEGER DEFAULT 0,
            details TEXT DEFAULT '',
            phase_details TEXT DEFAULT '{}'
        )
    """)
    # Backfill the v10.5.0 bookkeeping columns on DBs whose consolidation_log
    # predates them. SQLite has no ADD COLUMN IF NOT EXISTS, so the swallowed
    # retry is the idempotency. Lets SQL-only runs log decay/demote counts.
    for _col in ("access_decayed", "importance_demoted"):
        try:
            conn.execute(f"ALTER TABLE consolidation_log ADD COLUMN {_col} INTEGER DEFAULT 0")
        except Exception:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nyx_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links(relation_type)")
    # v10.18.0: memoization for the phase-4 NLI finder. A pair's score is a
    # pure function of the two contents, so it is cached keyed on content
    # hashes; the nightly finder only re-scores pairs whose content changed
    # or that were never scored (backfill under the per-run budget).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nli_scan_cache (
            pair_min INTEGER NOT NULL,
            pair_max INTEGER NOT NULL,
            a_hash TEXT NOT NULL,
            b_hash TEXT NOT NULL,
            p_contra REAL NOT NULL,
            scanned_at TEXT DEFAULT (datetime('now', 'localtime')),
            PRIMARY KEY (pair_min, pair_max)
        )
    """)
    conn.commit()


# --- Phase 6: Bookkeeping (no LLM required) ---

def cleanup_scan_cache(conn, execute=True):
    """Drop nli_scan_cache rows whose pair references a non-active memory.

    Mirrors cleanup_stale_links: archived and deleted memories can never
    re-enter the phase-4 candidate set, so their cached scores are dead
    weight. Returns the number of rows removed (or that would be).
    """
    where = (
        "NOT EXISTS (SELECT 1 FROM memories a WHERE a.id = nli_scan_cache.pair_min "
        "AND a.status = 'active') OR "
        "NOT EXISTS (SELECT 1 FROM memories b WHERE b.id = nli_scan_cache.pair_max "
        "AND b.status = 'active')"
    )
    try:
        count = conn.execute(
            f"SELECT COUNT(*) FROM nli_scan_cache WHERE {where}"
        ).fetchone()[0]
        if count and execute:
            conn.execute(f"DELETE FROM nli_scan_cache WHERE {where}")
            conn.commit()
        return count
    except Exception:
        return 0


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


def cleanup_orphan_vectors(conn, execute: bool = True):
    """Tier-2 aware vector hygiene (v10.7.0).

    Vectors for archived memories are MOVED into the archived index
    (embed_vec_arch) so tier-2 recall can vector-search the originals directly,
    rather than deleted. A vector is only removed outright when its memory row
    no longer exists at all (a hard delete). Returns the count of vectors moved
    or removed. Only mutates when execute=True.
    """
    from ..storage.sqlite_store import _serialize_vec
    status_by_id = {
        r[0]: r[1] for r in conn.execute("SELECT id, status FROM memories").fetchall()
    }
    embedded = conn.execute(
        "SELECT id, source_id, text_hash, model FROM embed_meta WHERE source_db = ?", (SOURCE_KEY,)
    ).fetchall()
    join_col = _vec_join_col(conn)
    moved = removed = 0
    for meta_id, source_id, thash, emodel in embedded:
        st = status_by_id.get(source_id)
        if st == "active":
            continue
        if not execute:
            moved += 1  # would be handled (moved or removed)
            continue
        if st == "archived":
            vrow = conn.execute(
                f"SELECT vec_to_json(embedding) AS j FROM embed_vec WHERE {join_col} = ?",
                (meta_id,),
            ).fetchone()
            if vrow and vrow[0] is not None:
                emb = json.loads(vrow[0])
                from ..storage.sqlite_store import _store_archived_embedding_conn
                _store_archived_embedding_conn(conn, source_id, emb,
                                               text_hash=thash, commit=False,
                                               model=emodel, source_key=SOURCE_KEY)
                moved += 1
            conn.execute(f"DELETE FROM embed_vec WHERE {join_col} = ?", (meta_id,))
            conn.execute("DELETE FROM embed_meta WHERE id = ?", (meta_id,))
        else:
            # memory row gone entirely -> drop the vector
            conn.execute(f"DELETE FROM embed_vec WHERE {join_col} = ?", (meta_id,))
            conn.execute("DELETE FROM embed_meta WHERE id = ?", (meta_id,))
            removed += 1
    if (moved or removed) and execute:
        conn.commit()
    return moved + removed


def decay_access_counts(conn, execute: bool = True):
    """Decay access_count by 1 for memories not accessed in 7+ days.

    Returns (decayed, demoted) counts. Only mutates when execute=True; a
    dry run computes the same counts without writing, so a preview never
    silently demotes importance or decays counts on the live store.
    """
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
        will_demote = imp in (6, 7, 8) and imp > target
        if execute:
            if will_demote:
                conn.execute(
                    "UPDATE memories SET access_count = ?, importance = ? WHERE id = ?",
                    (new_count, target, mid),
                )
            else:
                conn.execute("UPDATE memories SET access_count = ? WHERE id = ?", (new_count, mid))
        if will_demote:
            demoted += 1
        decayed += 1
    if decayed and execute:
        conn.commit()
    return decayed, demoted


def cleanup_stale_links(conn, execute: bool = True):
    """Remove links where either endpoint is archived.

    Returns the count of stale links. Only deletes when execute=True; a
    dry run counts them via SELECT without mutating.
    """
    where = (
        "source_id NOT IN (SELECT id FROM memories WHERE status='active') "
        "OR target_id NOT IN (SELECT id FROM memories WHERE status='active')"
    )
    try:
        if execute:
            n = conn.execute(f"DELETE FROM memory_links WHERE {where}").rowcount
            if n:
                conn.commit()
        else:
            n = conn.execute(f"SELECT COUNT(*) FROM memory_links WHERE {where}").fetchone()[0]
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

    import os as _os
    from ..constants import MERGE_ENGINE_DEFAULT, CONTRADICT_JUDGE_DEFAULT
    disable_llm = _os.environ.get("MNEMOS_DISABLE_LLM") == "1"
    has_llm = llm_is_configured() and not disable_llm

    # v10.17.0: which phases need an LLM depends on the configured engines.
    # Weave and synthesize always generate text. Phase 2 needs one only in
    # the legacy llm merge engine (mechanical is selection, not generation).
    # Phase 4 needs one only when the judge resolves to llm; queue mode
    # records contradiction-candidate links for a later LLM-tier run.
    merge_engine = _os.environ.get(
        "MNEMOS_MERGE_ENGINE", MERGE_ENGINE_DEFAULT).lower()
    judge_mode = _os.environ.get(
        "MNEMOS_CONTRADICT_JUDGE", CONTRADICT_JUDGE_DEFAULT).lower()
    if judge_mode == "auto":
        judge_mode = "llm" if has_llm else "queue"
    llm_phases = {3, 5}
    if merge_engine == "llm":
        llm_phases.add(2)
    if judge_mode == "llm":
        llm_phases.add(4)
    log(f"Engines: merge={merge_engine} contradict-judge={judge_mode} "
        f"(LLM-requiring phases requested: {sorted(phases & llm_phases) or 'none'})")

    if not has_llm and (phases & llm_phases):
        # v10.17.0: the core cycle no longer needs an LLM, so a missing key
        # skips only the enrichment-tier phases instead of failing the run
        # (v10.4.0 loud-fail predates the zero-LLM daily cycle). The WARNING
        # line is grep-able by ops wrappers that expect the LLM tier to run.
        skipped = sorted(phases & llm_phases)
        phases = phases - llm_phases
        if disable_llm:
            log(f"MNEMOS_DISABLE_LLM=1: skipping LLM-tier phases {skipped}")
        else:
            log(f"WARNING: no LLM configured; skipping LLM-tier phases "
                f"{skipped}. The core cycle continues; set "
                "MNEMOS_LLM_API_KEY to enable the enrichment tier.")

    mode = "EXECUTE" if execute else "DRY RUN"
    log(f"Mnemos Nyx Cycle starting ({mode}, phases={sorted(phases)})")
    start_time = time.time()

    phase_stats = {}

    # Last-run timestamp: needed by Phase 1 triage and recorded by the audit
    # log. Hoisted out of the LLM block so triage can run without an LLM.
    last_run = conn.execute(
        "SELECT MAX(run_at) FROM consolidation_log"
    ).fetchone()[0]
    log(f"Last run: {last_run or 'never'}")

    # Surge/new-id state. Populated by Phase 1 if it runs, consumed by Phase 5.
    new_ids = []
    is_surge = surge

    # --- Phase 1: Triage (pure SQL, no LLM required, runs standalone) ---
    # Detect memories created since the last run and decide surge mode. Lives
    # outside the LLM block so SQL-only runs still triage, not just Phase 6.
    if 1 in phases:
        from .phases import load_memory_meta, phase_triage
        meta = load_memory_meta(conn, project=project,
                                namespace=getattr(store, 'namespace', None))
        new_ids, surge_detected = phase_triage(conn, meta, last_run)
        is_surge = is_surge or surge_detected
        phase_stats["phase1"] = {"new_memories": len(new_ids), "surge": is_surge}

    # --- Embedding-dependent phases (2-5; LLM only where the engine needs it) ---
    if phases & {2, 3, 4, 5}:
        from .phases import (
            load_embeddings, phase_dedup,
            phase_weave, phase_contradict, phase_synthesize,
            _is_nyx_generated,
        )

        log("Loading embeddings...")
        all_embeddings, mergeable_embeddings, mem_by_id = load_embeddings(
            conn, project=project,
            namespace=getattr(store, 'namespace', None)
        )

        if len(all_embeddings) < 2:
            log(f"Only {len(all_embeddings)} embeddings, need >=2 for clustering. Skipping phases 2-5.")
        else:

            # Phase 0.5 (Cemelify) was REMOVED in v10.20.0. Rewriting
            # already-stored memories every cycle is generation against
            # content whose fidelity is the product; it drifted exact
            # strings on weaker models, was disabled on every known
            # deployment, and (final straw) ran ungated by the phase list,
            # firing LLM calls on zero-LLM runs of key-configured hosts.
            # Prose entering the store is shaped at store/ingest time
            # instead; stored content is never rewritten in place.

            if 2 in phases:
                phase_stats["phase2"] = phase_dedup(
                    conn, mergeable_embeddings, mem_by_id, is_surge, execute=execute
                )
                if execute:
                    p1 = phase_stats["phase2"]
                    if (p1.get("tight_merged", 0) + p1.get("topic_merged", 0)) > 0:
                        log("Reloading embeddings after merges...")
                        all_embeddings, mergeable_embeddings, mem_by_id = load_embeddings(
                            conn, project=project,
                            namespace=getattr(store, 'namespace', None)
                        )

            if 3 in phases:
                phase_stats["phase3"] = phase_weave(
                    conn, all_embeddings, mem_by_id, is_surge, execute=execute
                )

            if 4 in phases:
                # all_embeddings, not mergeable: protected memories
                # (decisions, verified, importance>=9) are exactly the ones
                # worth contradiction-scanning, and load_embeddings keeps
                # them in the full set for weave/contradict by design.
                # They were never wired through, which is why phase 4
                # reported "not enough decision/fact memories" on stores
                # whose facts are predominantly protected.
                phase_stats["phase4"] = phase_contradict(
                    conn, all_embeddings, mem_by_id, is_surge,
                    execute=execute, judge=judge_mode
                )
                if execute:
                    p4 = phase_stats["phase4"]
                    if p4.get("superseded", 0) > 0 and 5 in phases:
                        # Same staleness problem the post-phase-2 reload
                        # solves: SUPERSEDED archived rows in the DB while
                        # mem_by_id still says active, letting synthesis cite
                        # a just-archived source.
                        log("Reloading embeddings after supersedes...")
                        all_embeddings, mergeable_embeddings, mem_by_id = load_embeddings(
                            conn, project=project,
                            namespace=getattr(store, 'namespace', None)
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

    # --- Phase 6: Bookkeeping (runs even without an LLM; respects execute) ---
    if 6 in phases:
        log("Phase 6: Bookkeeping...")
        verb = "" if execute else "would "
        orphaned = cleanup_orphan_vectors(conn, execute=execute)
        if orphaned:
            log(f"  {verb}cleaned {orphaned} orphaned vectors")

        decayed, demoted = decay_access_counts(conn, execute=execute)
        if decayed:
            log(f"  {verb}decayed {decayed} access counts ({demoted} demoted)")

        stale_links = cleanup_stale_links(conn, execute=execute)
        if stale_links:
            log(f"  {verb}removed {stale_links} stale links")

        dead_cache = cleanup_scan_cache(conn, execute=execute)
        if dead_cache:
            log(f"  {verb}dropped {dead_cache} dead scan-cache rows")

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
        p6 = phase_stats.get("phase6", {})
        try:
            store.log_consolidation_run(
                clusters_found=p1.get("tight_found", 0) + p1.get("topic_found", 0),
                clusters_merged=p1.get("tight_merged", 0) + p1.get("topic_merged", 0),
                memories_archived=p1.get("archived", 0),
                memories_created=p1.get("created", 0),
                access_decayed=p6.get("access_decayed", 0),
                importance_demoted=p6.get("importance_demoted", 0),
                details=f"mnemos-nyx-cycle, phases={sorted(phases)}",
                phase_details=json.dumps(phase_stats, default=str),
            )
        except Exception as e:
            log(f"Warning: failed to log consolidation run: {e}")

    return phase_stats
