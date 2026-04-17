"""
Nyx Cycle phases for memory consolidation v2.

Six phases modeled after brain sleep consolidation:
  Phase 1: Triage. classify new memories, estimate budget
  Phase 2: Dedup. two-tier merge (tight + topic)
  Phase 3: Weave. cross-category connection detection
  Phase 4: Contradict. temporal evolution detection
  Phase 5: Synthesize. generate novel cross-domain insights
  Phase 6: Bookkeep. decay, cleanup, logging

All phases use whichever LLM is configured via mnemos.consolidation.llm
(any OpenAI-compatible endpoint). Originals are archived, never deleted.
Safety guards preserved.

Dependencies: mnemos.constants, mnemos.embed, mnemos.consolidation.llm,
mnemos.consolidation.prompts
"""

import json
import re
import sqlite3
import struct
import time
from collections import defaultdict

import numpy as np

from .prompts import (
    MERGE_SYSTEM, MERGE_SYSTEM_PROSE,
    WEAVE_SYSTEM, CONTRADICT_SYSTEM,
    SYNTHESIS_SYSTEM, SYNTHESIS_SYSTEM_PROSE, TRIAGE_SYSTEM,
)
from ..embed import embed as fastembed_embed_raw, text_hash, prep_memory_text
from ..constants import (
    SKIP_IMPORTANCE, MAX_CLUSTER_SIZE,
    FASTEMBED_MODEL, FASTEMBED_DIMS, CML_MODE,
)


def _merge_prompt():
    """Pick the prose or CML merge prompt based on MNEMOS_CML_MODE."""
    return MERGE_SYSTEM_PROSE if CML_MODE == "off" else MERGE_SYSTEM


def _synthesis_prompt():
    """Pick the prose or CML synthesis prompt based on MNEMOS_CML_MODE."""
    return SYNTHESIS_SYSTEM_PROSE if CML_MODE == "off" else SYNTHESIS_SYSTEM
from .llm import opus_chat, is_configured as llm_is_configured


def fastembed_embed(texts, prefix="passage"):
    """Backwards-compat adapter for the Nyx phase code."""
    if prefix == "search_query":
        prefix = "query"
    return fastembed_embed_raw(texts, prefix=prefix)


def _vec_join_col(conn):
    """Detect whether embed_vec uses 'id' (explicit PK) or 'rowid' (implicit)."""
    import sqlite3 as _sq
    try:
        conn.execute("SELECT id FROM embed_vec LIMIT 0").fetchone()
        return "id"
    except _sq.OperationalError:
        return "rowid"


def store_embeddings(conn, tuples, model=None):
    """Bulk-insert embeddings via the SQLite vec extension."""
    import struct as _struct
    join_col = _vec_join_col(conn)
    for source_db, source_id, thash, embedding in tuples:
        # Remove existing embedding for this source_id
        existing = conn.execute(
            "SELECT id FROM embed_meta WHERE source_db = ? AND source_id = ?",
            (source_db, source_id),
        ).fetchone()
        if existing:
            conn.execute(f"DELETE FROM embed_vec WHERE {join_col} = ?", (existing[0],))
            conn.execute(
                "DELETE FROM embed_meta WHERE source_db = ? AND source_id = ?",
                (source_db, source_id),
            )
        cur = conn.execute(
            "INSERT INTO embed_vec(embedding) VALUES (?)",
            (_struct.pack(f"{len(embedding)}f", *embedding),),
        )
        vec_id = cur.lastrowid
        conn.execute(
            "INSERT INTO embed_meta (id, source_db, source_id, text_hash, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (vec_id, source_db, source_id, thash, model),
        )


# --- Constants ---

# Phase 2 thresholds
TIGHT_THRESHOLD = 0.88    # Tier 1A: near-duplicate dedup
TOPIC_THRESHOLD = 0.75    # Tier 1B: same-topic merge

# Phase 3 thresholds
WEAVE_MIN_SIMILARITY = 0.55  # Cross-category connection minimum
WEAVE_TOP_K = 3              # Top K cross-category matches per memory

# Phase 4 thresholds
CONTRADICT_MIN_SIM = 0.60
CONTRADICT_MAX_SIM = 0.85

# Phase 5
NYX_PACKET_SIZE = 25

# Budget limits
NORMAL_MAX_CALLS = 30      # Max Opus calls per normal run
SURGE_MAX_CALLS = 80       # Max Opus calls during surge
SURGE_THRESHOLD = 50       # New memories count that triggers surge mode


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Shared helpers
# =============================================================================

def load_embeddings(conn, project=None):
    """Load active memory embeddings. Returns (embeddings_dict, mem_by_id)."""
    where = "status = 'active'"
    params = []
    if project:
        where += " AND project = ?"
        params.append(project)

    memories = conn.execute(
        f"SELECT * FROM memories WHERE {where} ORDER BY id", params
    ).fetchall()
    mem_by_id = {m["id"]: dict(m) for m in memories}
    active_ids = set(mem_by_id.keys())

    # Filter out evergreen, high-importance, and consolidation-locked for merging
    skip_ids = set()
    for mid, m in mem_by_id.items():
        if m.get("importance", 5) >= SKIP_IMPORTANCE:
            skip_ids.add(mid)
        tags = m.get("tags", "") or ""
        if "evergreen" in tags:
            skip_ids.add(mid)
        if m.get("consolidation_lock"):
            skip_ids.add(mid)

    # Load embeddings from vec tables
    meta_rows = conn.execute(
        "SELECT id, source_id FROM embed_meta WHERE source_db = 'memory'"
    ).fetchall()

    embeddings = {}
    for meta_id, source_id in meta_rows:
        if source_id not in active_ids:
            continue
        row = conn.execute(
            "SELECT embedding FROM embed_vec WHERE id = ?", (meta_id,)
        ).fetchone()
        if row and row[0]:
            blob = row[0]
            n_floats = len(blob) // 4
            vec = np.array(struct.unpack(f"{n_floats}f", blob), dtype=np.float32)
            embeddings[source_id] = vec

    # Separate: mergeable (excludes skip_ids) vs all (for weaving/synthesis)
    mergeable = {k: v for k, v in embeddings.items() if k not in skip_ids}

    if skip_ids:
        skipped_with_embeds = len([s for s in skip_ids if s in embeddings])
        log(f"Loaded {len(embeddings)} embeddings ({len(mergeable)} mergeable, "
            f"{skipped_with_embeds} protected)")

    return embeddings, mergeable, mem_by_id


def _is_nyx_generated(mem):
    """Check if a memory was generated by the Nyx cycle."""
    tags = (mem.get("tags") or "").lower()
    return any(t in tags for t in ("synthesized", "nyx-cycle", "bridge"))


def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    ids = sorted(embeddings.keys())
    if len(ids) < 2:
        return ids, np.array([])

    matrix = np.stack([embeddings[i] for i in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = matrix / norms
    sim_matrix = normalized @ normalized.T
    return ids, sim_matrix


def find_clusters(ids, sim_matrix, threshold, mem_by_id, max_clusters=50,
                  enforce_project_boundary=True):
    """Complete-linkage agglomerative clustering.

    Returns list of [memory_id, ...] clusters.
    """
    n = len(ids)
    if n < 2:
        return []

    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                if enforce_project_boundary and mem_by_id:
                    pi = mem_by_id.get(ids[i], {}).get("project")
                    pj = mem_by_id.get(ids[j], {}).get("project")
                    if pi != pj:
                        continue
                adj[i][j] = adj[j][i] = True

    # Greedy complete-linkage clique finding
    used = set()
    clusters = []

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                pairs.append((sim_matrix[i][j], i, j))
    pairs.sort(reverse=True)

    for _, seed_i, seed_j in pairs:
        if seed_i in used or seed_j in used:
            continue

        cluster = {seed_i, seed_j}
        for k in range(n):
            if k in used or k in cluster:
                continue
            if len(cluster) >= MAX_CLUSTER_SIZE:
                break
            if all(adj[k][m] for m in cluster):
                cluster.add(k)

        if len(cluster) >= 2:
            cluster_ids = sorted(ids[i] for i in cluster)
            # Skip clusters that are entirely nyx-generated or consolidated (anti-churn)
            if mem_by_id:
                has_original = any(
                    not _is_nyx_generated(mem_by_id.get(mid, {}))
                    and "consolidated" not in (mem_by_id.get(mid, {}).get("tags", "") or "")
                    for mid in cluster_ids
                )
                if not has_original:
                    continue
            clusters.append(cluster_ids)
            used.update(cluster)

    clusters = sorted(clusters, key=lambda c: c[0])
    return clusters[:max_clusters]


def merge_cluster(cluster_ids, mem_by_id):
    """LLM-merge a cluster into a single CML super-memory.

    Clusters of size 2 run a single merge call. Clusters of size 3+ use
    hierarchical pairwise merging: each step merges two inputs into one
    output, which participates as an input at the next level. The LLM
    never sees more than two memories at once, so the "output roughly the
    size of one input" intuition holds even for deep clusters; compounding
    compression at each level is mitigated by MERGE_SYSTEM's size-scaling
    language (no arbitrary-size truncation, growth with cluster size is
    expected).
    """
    items = []
    for mid in cluster_ids:
        mem = mem_by_id[mid]
        created = mem.get("created_at", "unknown")[:7]
        items.append({
            "id": mid,
            "source_ids": [mid],  # provenance: original ids that fed into this content
            "project": mem["project"],
            "content": mem["content"],
            "header": (f"--- Memory #{mid} [{mem['project']}] ({created}) "
                       f"tags:{mem.get('tags', '')} imp:{mem.get('importance', 5)} ---"),
        })

    if len(items) < 2:
        return items[0]["content"] if items else None

    def _src_summary(srcs):
        # Summarize source ids compactly: "#1, #2, #3" or "#1...#42 (N=8)" when long
        if len(srcs) <= 4:
            return ", ".join(f"#{s}" for s in srcs)
        return f"#{srcs[0]}...#{srcs[-1]} (N={len(srcs)})"

    level = 0
    current = items
    while len(current) > 1:
        next_level = []
        i = 0
        while i + 1 < len(current):
            a, b = current[i], current[i + 1]
            total_chars = len(a["content"]) + len(b["content"])
            target = int(total_chars * 0.8)
            # Synthesize a header that names provenance even at deeper levels
            a_header = a.get("header") or f"--- Merged from {_src_summary(a['source_ids'])} ---"
            b_header = b.get("header") or f"--- Merged from {_src_summary(b['source_ids'])} ---"
            body = (
                f"{a_header}\n{a['content']}\n\n"
                f"{b_header}\n{b['content']}"
            )
            user_msg = (
                f"The two memories below total {total_chars} characters. Your output should be "
                f"roughly {target} characters (±20%): preserve every specific (names, numbers, "
                f"dates, paths, amounts, decisions, preferences). Only remove content that is "
                f"genuine overlap between the two memories (same fact stated twice).\n\n"
                f"Merge these 2 related memories into one {'prose paragraph' if CML_MODE == 'off' else 'CML block'}:\n\n{body}"
            )
            merged = opus_chat(
                [
                    {"role": "system", "content": _merge_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max(1024, total_chars),
                temperature=0.3,
                phase="MERGE",
            )
            if merged:
                merged = re.sub(r"^```\w*\n?", "", merged)
                merged = re.sub(r"\n?```$", "", merged).strip()
            else:
                # LLM call failed (returned None/empty). Fall back to raw
                # concatenation so nothing is lost — but LOG IT so debugging
                # doesn't have to guess why a merged memory is suddenly
                # twice the expected size. Previously this was silent; a
                # persistent LLM outage would inflate merged memories
                # without any operator signal.
                log(f"  Warning: LLM merge call returned empty for L{level} "
                    f"pair (sources={a['source_ids']} + {b['source_ids']}); "
                    f"falling back to raw concatenation (content will be "
                    f"~{len(a['content']) + len(b['content'])} chars)")
                merged = a["content"] + "\n---\n" + b["content"]
            combined_sources = a["source_ids"] + b["source_ids"]
            next_level.append({
                "id": f"L{level}_{i//2}",
                "source_ids": combined_sources,
                "project": a["project"],
                "content": merged,
            })
            i += 2
        if i < len(current):
            next_level.append(current[i])  # odd one out, carry forward
        current = next_level
        level += 1
    return current[0]["content"]


def apply_merge(conn, cluster_ids, merged_content, mem_by_id):
    """Store merged memory, archive originals. Returns new memory ID."""
    projects = [mem_by_id[mid]["project"] for mid in cluster_ids]
    project = max(set(projects), key=projects.count)

    all_tags = set()
    for mid in cluster_ids:
        tags = mem_by_id[mid].get("tags", "") or ""
        for t in tags.split(","):
            t = t.strip()
            if t:
                all_tags.add(t)
    all_tags.add("consolidated")

    max_importance = max(mem_by_id[mid].get("importance", 5) for mid in cluster_ids)
    inherit_lock = 1 if any(mem_by_id[mid].get("consolidation_lock") for mid in cluster_ids) else 0
    inherit_verified = 1 if any(mem_by_id[mid].get("verified") for mid in cluster_ids) else 0

    types = [mem_by_id[mid].get("type") or "fact" for mid in cluster_ids]
    non_fact = [t for t in types if t != "fact"]
    inherit_type = non_fact[0] if non_fact else "fact"

    confirmed_dates = [mem_by_id[mid].get("last_confirmed") for mid in cluster_ids
                       if mem_by_id[mid].get("last_confirmed")]
    inherit_confirmed = max(confirmed_dates) if confirmed_dates else None

    # Atomic merge: INSERT new memory + UPDATE archive sources + INSERT
    # embedding all go in one transaction. If embedding fails we ROLLBACK
    # the whole thing so the DB never contains an unembeddable merged
    # memory without its sources — previously we committed the merge
    # first, then tried to embed, and a failure there left orphans that
    # only `doctor --migrate` could later notice. One-transaction keeps
    # the invariant that every active memory has an embedding in the
    # same write window.
    try:
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO memories (project, content, tags, importance, type, verified, "
            "consolidation_lock, layer, last_confirmed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'semantic', ?)",
            (project, merged_content, ",".join(sorted(all_tags)), max_importance,
             inherit_type, inherit_verified, inherit_lock, inherit_confirmed),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for mid in cluster_ids:
            conn.execute(
                "UPDATE memories SET status = 'archived', "
                "updated_at = datetime('now', 'localtime'), "
                "tags = tags || ',merged-into-' || ? "
                "WHERE id = ?",
                (str(new_id), mid),
            )

        # Embed inside the same transaction. layer="semantic" matches the
        # INSERT above and keeps prep_memory_text consistent with stored
        # metadata. If fastembed_embed fails (network, OOM, etc.) the
        # exception bubbles out and the outer except ROLLBACKs.
        text = prep_memory_text(project, merged_content,
                                ",".join(sorted(all_tags)),
                                mem_type=inherit_type, layer="semantic")
        emb = fastembed_embed([text])
        if not (emb and emb[0]):
            raise RuntimeError("fastembed returned empty embedding")
        thash = text_hash(text)
        store_embeddings(conn, [(
            "memory", new_id, thash, emb[0]
        )], model=FASTEMBED_MODEL)

        conn.commit()
        return new_id
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(f"Warning: merge aborted for cluster {cluster_ids}: {e}. "
            f"No changes committed (transaction rolled back).")
        return None


# =============================================================================
# Phase 1: Triage
# =============================================================================

def phase_triage(conn, mem_by_id, last_run_at):
    """Pre-classify new memories. Returns (new_ids, is_surge)."""
    new_ids = []
    for mid, m in mem_by_id.items():
        if m["status"] != "active":
            continue
        if last_run_at and m["created_at"] > last_run_at:
            new_ids.append(mid)

    if not last_run_at:
        new_ids = list(mem_by_id.keys())

    is_surge = len(new_ids) > SURGE_THRESHOLD
    log(f"Phase 1: {len(new_ids)} new memories since last run"
        f"{' (SURGE MODE)' if is_surge else ''}")

    return new_ids, is_surge


# =============================================================================
# Phase 2: Dedup Merge (two-tier)
# =============================================================================

def phase_dedup(conn, mergeable_embeddings, mem_by_id, is_surge, execute=False):
    """Two-tier dedup: tight (0.88) + topic (0.75). Returns stats dict."""
    stats = {"tight_found": 0, "tight_merged": 0, "topic_found": 0,
             "topic_merged": 0, "archived": 0, "created": 0}

    if len(mergeable_embeddings) < 2:
        log("Phase 2: Not enough mergeable memories")
        return stats

    max_clusters = SURGE_MAX_CALLS if is_surge else NORMAL_MAX_CALLS

    # Tier 1A: Tight dedup
    log(f"Phase 2A: Tight dedup (threshold={TIGHT_THRESHOLD})...")
    ids, sim_matrix = cosine_similarity_matrix(mergeable_embeddings)
    tight_clusters = find_clusters(ids, sim_matrix, TIGHT_THRESHOLD,
                                   mem_by_id, max_clusters=max_clusters)
    stats["tight_found"] = len(tight_clusters)

    if tight_clusters:
        log(f"  Found {len(tight_clusters)} tight clusters")
        for i, cluster in enumerate(tight_clusters):
            _log_cluster(cluster, mem_by_id, i + 1)
            if execute:
                merged = merge_cluster(cluster, mem_by_id)
                if merged:
                    new_id = apply_merge(conn, cluster, merged, mem_by_id)
                    if new_id is not None:
                        log(f"  Merged → #{new_id}, archived {cluster}")
                        stats["tight_merged"] += 1
                        stats["archived"] += len(cluster)
                        stats["created"] += 1
                        # Remove merged IDs from embeddings for tier 1B
                        for mid in cluster:
                            mergeable_embeddings.pop(mid, None)
                    else:
                        stats["merge_aborted"] = stats.get("merge_aborted", 0) + 1
                time.sleep(0.5)

    # Tier 1B: Topic merge (looser threshold)
    if len(mergeable_embeddings) >= 2:
        log(f"Phase 2B: Topic merge (threshold={TOPIC_THRESHOLD})...")
        ids, sim_matrix = cosine_similarity_matrix(mergeable_embeddings)
        remaining = max(0, max_clusters - stats["tight_merged"])
        topic_clusters = find_clusters(ids, sim_matrix, TOPIC_THRESHOLD,
                                       mem_by_id, max_clusters=remaining)
        stats["topic_found"] = len(topic_clusters)

        if topic_clusters:
            log(f"  Found {len(topic_clusters)} topic clusters")
            for i, cluster in enumerate(topic_clusters):
                _log_cluster(cluster, mem_by_id, i + 1)
                if execute:
                    merged = merge_cluster(cluster, mem_by_id)
                    if merged:
                        new_id = apply_merge(conn, cluster, merged, mem_by_id)
                        if new_id is not None:
                            log(f"  Merged → #{new_id}, archived {cluster}")
                            stats["topic_merged"] += 1
                            stats["archived"] += len(cluster)
                            stats["created"] += 1
                        else:
                            stats["merge_aborted"] = stats.get("merge_aborted", 0) + 1
                    time.sleep(0.5)

    log(f"Phase 2 done: {stats['tight_found']}+{stats['topic_found']} clusters found, "
        f"{stats['tight_merged']}+{stats['topic_merged']} merged")
    return stats


def _log_cluster(cluster, mem_by_id, num):
    """Log a cluster's contents."""
    print(f"\n  Cluster {num} ({len(cluster)} memories):")
    for mid in cluster:
        mem = mem_by_id.get(mid, {})
        content = mem.get("content", "?")[:100]
        print(f"    #{mid} [{mem.get('project', '?')}] {content}")


# =============================================================================
# Phase 3: Thematic Weaving
# =============================================================================

def phase_weave(conn, all_embeddings, mem_by_id, is_surge, execute=False):
    """Find cross-category connections. Returns stats dict."""
    stats = {"pairs_evaluated": 0, "links_created": 0, "insights_stored": 0}

    if len(all_embeddings) < 4:
        log("Phase 3: Not enough memories for cross-category weaving")
        return stats

    log("Phase 3: Thematic weaving (cross-category connections)...")

    # Filter out nyx-generated memories. only weave original content
    original_ids = {
        mid for mid in all_embeddings
        if not _is_nyx_generated(mem_by_id.get(mid, {}))
    }
    log(f"  {len(original_ids)} original memories (skipping {len(all_embeddings) - len(original_ids)} nyx-generated)")

    ids = sorted(original_ids)
    id_to_idx = {mid: i for i, mid in enumerate(ids)}

    # Build similarity matrix for all embeddings
    matrix = np.stack([all_embeddings[i] for i in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = matrix / norms
    sim_matrix = normalized @ normalized.T

    # Find cross-category pairs
    cross_pairs = []
    seen = set()

    for idx_a, mid_a in enumerate(ids):
        proj_a = mem_by_id.get(mid_a, {}).get("project")
        if not proj_a:
            continue

        # Find top-K from OTHER categories
        sims = []
        for idx_b, mid_b in enumerate(ids):
            if mid_a == mid_b:
                continue
            proj_b = mem_by_id.get(mid_b, {}).get("project")
            if proj_a == proj_b:
                continue  # same category, skip
            cos = float(sim_matrix[idx_a][idx_b])
            if cos >= WEAVE_MIN_SIMILARITY:
                sims.append((mid_b, cos))

        sims.sort(key=lambda x: x[1], reverse=True)
        for mid_b, cos in sims[:WEAVE_TOP_K]:
            pair_key = tuple(sorted([mid_a, mid_b]))
            if pair_key not in seen:
                seen.add(pair_key)
                cross_pairs.append((mid_a, mid_b, cos))

    # Sort by similarity (most promising first)
    cross_pairs.sort(key=lambda x: x[2], reverse=True)

    # Check existing links to avoid re-evaluating
    existing_links = set()
    try:
        rows = conn.execute(
            "SELECT source_id, target_id FROM memory_links"
        ).fetchall()
        for s, t in rows:
            existing_links.add(tuple(sorted([s, t])))
    except Exception:
        pass

    cross_pairs = [
        (a, b, c) for a, b, c in cross_pairs
        if tuple(sorted([a, b])) not in existing_links
    ]

    max_eval = SURGE_MAX_CALLS if is_surge else NORMAL_MAX_CALLS
    cross_pairs = cross_pairs[:max_eval]

    log(f"  {len(cross_pairs)} candidate cross-category pairs to evaluate")

    for mid_a, mid_b, cos in cross_pairs:
        mem_a = mem_by_id[mid_a]
        mem_b = mem_by_id[mid_b]

        print(f"\n  Pair: #{mid_a} [{mem_a.get('project')}] × "
              f"#{mid_b} [{mem_b.get('project')}] (sim={cos:.3f})")
        print(f"    A: {mem_a.get('content', '')[:80]}")
        print(f"    B: {mem_b.get('content', '')[:80]}")

        if not execute:
            stats["pairs_evaluated"] += 1
            continue

        prompt = (
            f"Memory A (#{mid_a}, category: {mem_a.get('project')}):\n"
            f"{mem_a.get('content', '')}\n\n"
            f"Memory B (#{mid_b}, category: {mem_b.get('project')}):\n"
            f"{mem_b.get('content', '')}"
        )

        result = opus_chat(
            [
                {"role": "system", "content": WEAVE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.3,
            phase="WEAVE",
        )
        stats["pairs_evaluated"] += 1

        if not result or "NO_LINK" in result:
            print(f"    → No link")
            continue

        link_type, strength, insight = _parse_weave_result(result)
        if link_type:
            # Store link
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_links "
                    "(source_id, target_id, relation_type, strength) "
                    "VALUES (?, ?, ?, ?)",
                    (mid_a, mid_b, link_type, strength),
                )
                conn.commit()
                stats["links_created"] += 1
                log(f"    → Link: {link_type} (strength={strength:.2f})")
            except Exception as e:
                log(f"    → Link failed: {e}")

            # Store bridge insight as a new memory
            if insight:
                tag_str = f"synthesized,nyx-cycle,bridge,src:nyx-{time.strftime('%Y-%m-%d')}"
                if CML_MODE == "off":
                    content = f"Bridge between memory #{mid_a} and #{mid_b}: {insight}"
                else:
                    content = f"L: Bridge #{mid_a}↔#{mid_b}: {insight}"
                conn.execute(
                    "INSERT INTO memories (project, content, tags, importance, "
                    "type, layer, consolidation_lock) "
                    "VALUES ('personal', ?, ?, 5, 'learning', 'semantic', 1)",
                    (content, tag_str),
                )
                conn.commit()
                stats["insights_stored"] += 1
                log(f"    → Insight stored: {insight[:80]}")

        time.sleep(0.5)

    log(f"Phase 3 done: {stats['pairs_evaluated']} evaluated, "
        f"{stats['links_created']} links, {stats['insights_stored']} insights")
    return stats


def _parse_weave_result(text):
    """Parse LINK_TYPE/STRENGTH/INSIGHT from LLM response."""
    link_type = None
    strength = 0.5
    insight = None

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("LINK_TYPE:"):
            val = line.split(":", 1)[1].strip().lower()
            valid = {"evolves", "informs", "contradicts", "enables", "reflects"}
            if val in valid:
                link_type = val
        elif line.startswith("STRENGTH:"):
            try:
                strength = float(line.split(":", 1)[1].strip())
                strength = max(0.1, min(1.0, strength))
            except ValueError:
                pass
        elif line.startswith("INSIGHT:"):
            insight = line.split(":", 1)[1].strip()

    return link_type, strength, insight


# =============================================================================
# Phase 4: Contradiction Scan
# =============================================================================

def phase_contradict(conn, mergeable_embeddings, mem_by_id, is_surge, execute=False):
    """Detect decisions that evolved, reversed, or conflict. Returns stats dict."""
    stats = {"candidates": 0, "superseded": 0, "evolved": 0,
             "contradicts": 0, "compatible": 0}

    if len(mergeable_embeddings) < 2:
        log("Phase 4: Not enough memories for contradiction scan")
        return stats

    log("Phase 4: Contradiction scan...")

    # Focus on decisions and facts (most likely to contradict)
    target_types = {"decision", "fact"}
    target_ids = [
        mid for mid in mergeable_embeddings
        if mem_by_id.get(mid, {}).get("type") in target_types
    ]

    if len(target_ids) < 2:
        log("  Not enough decision/fact memories")
        return stats

    # Build subset similarity matrix
    target_embeds = {mid: mergeable_embeddings[mid] for mid in target_ids}
    ids, sim_matrix = cosine_similarity_matrix(target_embeds)
    id_to_idx = {mid: i for i, mid in enumerate(ids)}

    # Find same-category pairs with moderate similarity
    candidates = []
    for i, mid_a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            mid_b = ids[j]
            # Same category only
            if mem_by_id[mid_a]["project"] != mem_by_id[mid_b]["project"]:
                continue
            cos = float(sim_matrix[i][j])
            if CONTRADICT_MIN_SIM <= cos <= CONTRADICT_MAX_SIM:
                candidates.append((mid_a, mid_b, cos))

    candidates.sort(key=lambda x: x[2], reverse=True)
    max_eval = (SURGE_MAX_CALLS // 2) if is_surge else (NORMAL_MAX_CALLS // 2)
    candidates = candidates[:max_eval]
    stats["candidates"] = len(candidates)

    log(f"  {len(candidates)} candidate contradiction pairs")

    for mid_a, mid_b, cos in candidates:
        mem_a = mem_by_id[mid_a]
        mem_b = mem_by_id[mid_b]

        # Determine chronological order
        older_id = mid_a if mem_a["created_at"] <= mem_b["created_at"] else mid_b
        newer_id = mid_b if older_id == mid_a else mid_a
        older = mem_by_id[older_id]
        newer = mem_by_id[newer_id]

        print(f"\n  Pair: #{older_id} ({older['created_at'][:10]}) × "
              f"#{newer_id} ({newer['created_at'][:10]}) [{older.get('project')}] sim={cos:.3f}")
        print(f"    Older: {older.get('content', '')[:80]}")
        print(f"    Newer: {newer.get('content', '')[:80]}")

        if not execute:
            continue

        prompt = (
            f"Older memory (#{older_id}, {older['created_at'][:10]}):\n"
            f"{older.get('content', '')}\n\n"
            f"Newer memory (#{newer_id}, {newer['created_at'][:10]}):\n"
            f"{newer.get('content', '')}"
        )

        result = opus_chat(
            [
                {"role": "system", "content": CONTRADICT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.3,
            phase="CONTRADICT",
        )

        if not result:
            continue

        classification, explanation = _parse_contradict(result)

        if classification == "SUPERSEDED":
            # Archive the older memory, link to newer
            conn.execute(
                "UPDATE memories SET status='archived', "
                "updated_at=datetime('now','localtime'), "
                "tags=tags||',superseded-by-'||? WHERE id=?",
                (str(newer_id), older_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_links "
                "(source_id, target_id, relation_type, strength) "
                "VALUES (?, ?, 'superseded_by', 0.9)",
                (older_id, newer_id),
            )
            conn.commit()
            stats["superseded"] += 1
            log(f"    → SUPERSEDED: #{older_id} archived, linked to #{newer_id}")

        elif classification == "EVOLVED":
            # Mark the older memory's valid_until so `valid_only=True`
            # searches exclude it — the newer memory supersedes it
            # temporally. Without this, both stay active and ranking
            # treats them as equally current. Use today's date as a
            # best-effort approximation; the LLM classification was
            # based on the temporal argument in their texts.
            conn.execute(
                "UPDATE memories SET valid_until = date('now','localtime'), "
                "updated_at = datetime('now','localtime') "
                "WHERE id = ? AND (valid_until IS NULL OR valid_until > date('now','localtime'))",
                (older_id,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_links "
                "(source_id, target_id, relation_type, strength) "
                "VALUES (?, ?, 'evolves', 0.7)",
                (older_id, newer_id),
            )
            conn.commit()
            stats["evolved"] += 1
            log(f"    → EVOLVED: {explanation[:60]}")

        elif classification == "CONTRADICTS":
            conn.execute(
                "INSERT OR IGNORE INTO memory_links "
                "(source_id, target_id, relation_type, strength) "
                "VALUES (?, ?, 'contradicts', 0.8)",
                (older_id, newer_id),
            )
            conn.commit()
            stats["contradicts"] += 1
            log(f"    → CONTRADICTS: {explanation[:60]}")

        else:
            stats["compatible"] += 1
            print(f"    → Compatible")

        time.sleep(0.5)

    log(f"Phase 4 done: {stats['candidates']} evaluated. "
        f"{stats['superseded']} superseded, {stats['evolved']} evolved, "
        f"{stats['contradicts']} contradicts, {stats['compatible']} compatible")
    return stats


def _parse_contradict(text):
    """Parse classification|explanation from LLM response."""
    text = text.strip()
    valid = {"SUPERSEDED", "EVOLVED", "CONTRADICTS", "COMPATIBLE"}

    if "|" in text:
        parts = text.split("|", 1)
        cls = parts[0].strip().upper()
        explanation = parts[1].strip() if len(parts) > 1 else ""
        if cls in valid:
            return cls, explanation

    # Fallback: check first word
    first_word = text.split()[0].strip().upper() if text else ""
    if first_word in valid:
        return first_word, text
    return "COMPATIBLE", text


# =============================================================================
# Phase 5: Insight Synthesis
# =============================================================================

def phase_synthesize(conn, all_embeddings, mem_by_id, execute=False):
    """Generate novel cross-domain insights from memory corpus. Returns stats dict."""
    stats = {"packet_size": 0, "insights_generated": 0}

    active = {mid: m for mid, m in mem_by_id.items() if m["status"] == "active"}
    if len(active) < 10:
        log("Phase 5: Not enough active memories for synthesis")
        return stats

    log("Phase 5: Insight synthesis (Nyx generation)...")

    # Select Nyx packet. diverse, high-signal memories
    packet_ids = _select_nyx_packet(conn, active, all_embeddings)
    stats["packet_size"] = len(packet_ids)
    log(f"  Nyx packet: {len(packet_ids)} memories across "
        f"{len(set(active[m]['project'] for m in packet_ids if m in active))} categories")

    if not execute:
        for mid in packet_ids:
            mem = active.get(mid, {})
            print(f"    #{mid} [{mem.get('project')}] {mem.get('content', '')[:80]}")
        return stats

    # Build prompt
    parts = []
    for mid in packet_ids:
        mem = active[mid]
        parts.append(
            f"--- #{mid} [{mem['project']}] type:{mem.get('type', 'fact')} "
            f"imp:{mem.get('importance', 5)} ---\n"
            f"{mem['content'][:500]}"
        )

    memories_text = "\n\n".join(parts)
    prompt = (
        f"Nyx packet: {len(packet_ids)} memories from "
        f"{len(set(active[m]['project'] for m in packet_ids))} categories.\n\n"
        f"{memories_text}"
    )

    result = opus_chat(
        [
            {"role": "system", "content": _synthesis_prompt()},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.5,  # slightly higher for creative synthesis
        phase="SYNTHESIZE",
    )

    if not result:
        log("  Synthesis returned empty")
        return stats

    # Parse and store individual insights, linking each via nyx_insights
    insights = _parse_insights(result)
    tag_str = f"synthesized,nyx-cycle,src:nyx-{time.strftime('%Y-%m-%d')}"
    source_ids_json = json.dumps(packet_ids)

    for insight in insights:
        cur = conn.execute(
            "INSERT INTO memories (project, content, tags, importance, "
            "type, layer, consolidation_lock) "
            "VALUES ('personal', ?, ?, 5, 'learning', 'semantic', 1)",
            (insight, tag_str),
        )
        new_memory_id = cur.lastrowid
        # Link the new insight memory to the source packet via nyx_insights
        conn.execute(
            "INSERT INTO nyx_insights (memory_id, source_ids, insight_type, "
            "consolidation_type) VALUES (?, ?, 'synthesis', 'synthesis')",
            (new_memory_id, source_ids_json),
        )
        stats["insights_generated"] += 1
        log(f"  Insight: {insight[:100]}")

    if insights:
        conn.commit()

        # Embed new insights
        for row in conn.execute(
            "SELECT id, content, tags FROM memories WHERE tags LIKE ? AND status='active'",
            (f"%src:nyx-{time.strftime('%Y-%m-%d')}%",)
        ).fetchall():
            try:
                text = prep_memory_text(
                    "personal", row[1], row[2],
                    mem_type="learning", layer="semantic",
                )
                emb = fastembed_embed([text])
                if emb and emb[0]:
                    thash = text_hash(text)
                    store_embeddings(conn, [("memory", row[0], thash, emb[0])],
                                     model=FASTEMBED_MODEL)
            except Exception as e:
                log(f"  Warning: embedding failed for insight #{row[0]}: {e}")
        conn.commit()

    log(f"Phase 5 done: {stats['insights_generated']} insights generated "
        f"from {stats['packet_size']} memories")
    return stats


def _select_nyx_packet(conn, active_memories, embeddings):
    """Select diverse, high-signal memories for synthesis."""
    scored = {}

    # Get recent links for link-boost
    link_counts = defaultdict(int)
    try:
        rows = conn.execute(
            "SELECT source_id, target_id FROM memory_links "
            "WHERE created_at > datetime('now', '-7 days', 'localtime')"
        ).fetchall()
        for s, t in rows:
            link_counts[s] += 1
            link_counts[t] += 1
    except Exception:
        pass

    for mid, m in active_memories.items():
        # Skip nyx-generated memories. only synthesize from originals
        if _is_nyx_generated(m):
            continue

        score = 0.0

        # Recency boost: recently accessed = "on the mind"
        if m.get("last_accessed"):
            try:
                days_str = f"julianday('now','localtime') - julianday('{m['last_accessed']}')"
                days = conn.execute(f"SELECT {days_str}").fetchone()[0]
                score += max(0, 5 - days / 7)
            except Exception:
                pass

        # Importance
        score += m.get("importance", 5) * 0.3

        # New links boost
        score += link_counts.get(mid, 0) * 2

        # New memories boost
        try:
            days_str = f"julianday('now','localtime') - julianday('{m['created_at']}')"
            days = conn.execute(f"SELECT {days_str}").fetchone()[0]
            if days < 14:
                score += 3
        except Exception:
            pass

        scored[mid] = score

    # Round-robin across categories for diversity
    by_project = defaultdict(list)
    for mid in sorted(scored, key=scored.get, reverse=True):
        by_project[active_memories[mid]["project"]].append(mid)

    selected = []
    while len(selected) < NYX_PACKET_SIZE:
        added = False
        for proj in sorted(by_project.keys()):
            if by_project[proj] and len(selected) < NYX_PACKET_SIZE:
                selected.append(by_project[proj].pop(0))
                added = True
        if not added:
            break

    return selected


def _parse_insights(text):
    """Parse insights from synthesis output.

    CML mode: insights are L:-prefixed lines (possibly with continuation lines).
    Prose mode: insights are blank-line-separated paragraphs.
    """
    if CML_MODE == "off":
        # Split on blank lines; each non-empty chunk is one insight.
        chunks = [c.strip() for c in text.strip().split("\n\n")]
        return [c for c in chunks if c]

    insights = []
    current = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("L:") or line.startswith("L :"):
            if current:
                insights.append("\n".join(current))
            current = [line]
        elif current and line:
            current.append(line)

    if current:
        insights.append("\n".join(current))
    return insights
