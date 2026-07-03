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
import os
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
from ..splitter import (
    split_content, split_is_lossless, needs_split, split_enabled,
    explode_cml_chain,
)
from ..constants import (
    SKIP_IMPORTANCE, MAX_CLUSTER_SIZE,
    FASTEMBED_MODEL, FASTEMBED_DIMS, CML_MODE,
    DEFAULT_NAMESPACE,
    CONTRADICT_MIN_SIM, CONTRADICT_MAX_SIM,
    NYX_CONTRADICT_FINDER_DEFAULT, NLI_FINDER_THRESHOLD, NLI_FINDER_MAX_PAIRS,
    TIGHT_THRESHOLD, TOPIC_THRESHOLD, WEAVE_MIN_SIMILARITY, WEAVE_TOP_K,
    NYX_PACKET_SIZE, NORMAL_MAX_CALLS, SURGE_MAX_CALLS, SURGE_THRESHOLD,
    MERGE_ENGINE_DEFAULT, CLUSTER_GATE_DEFAULT, CLUSTER_GATE_TAU,
    CLUSTER_GATE_MAX_LINES, CANDIDACY_DEFAULT, CANDIDACY_TOP_K,
    WEAVE_NOVELTY_TAU,
)
from .mechanical import mechanical_merge_cluster
from .. import nli


def _active_namespace():
    """Namespace for memories Nyx creates, same resolution as the MCP server
    and CLI. Without this, consolidation outputs land in 'default' and become
    invisible to every namespace-filtered search: the cycle would archive
    visible memories and replace them with unreachable ones."""
    return os.environ.get("MNEMOS_NAMESPACE", DEFAULT_NAMESPACE)


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


# All Nyx tunables live in ..constants (single settings surface, v10.15.1):
# TIGHT_THRESHOLD, TOPIC_THRESHOLD, WEAVE_MIN_SIMILARITY, WEAVE_TOP_K,
# NYX_PACKET_SIZE, NORMAL_MAX_CALLS, SURGE_MAX_CALLS, SURGE_THRESHOLD,
# CONTRADICT_MIN_SIM, CONTRADICT_MAX_SIM.


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Shared helpers
# =============================================================================

def load_embeddings(conn, project=None):
    """Load active memory embeddings. Returns (all_embeddings, mergeable_embeddings, mem_by_id)."""
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

    # Filter out evergreen, high-importance, decision-type, and consolidation-locked
    # from merging. Decisions are authoritative records; merging blends and compresses
    # them (lossy) then archives the originals, so they are woven (linked) but never
    # merged, per the small-memories-that-link design. They stay in all_embeddings so
    # weave/contradict still reach them.
    skip_ids = set()
    for mid, m in mem_by_id.items():
        if m.get("importance", 5) >= SKIP_IMPORTANCE:
            skip_ids.add(mid)
        tags = m.get("tags", "") or ""
        if "evergreen" in tags:
            skip_ids.add(mid)
        if m.get("consolidation_lock"):
            skip_ids.add(mid)
        if m.get("type") == "decision":
            skip_ids.add(mid)

    # Load embeddings from vec tables
    meta_rows = conn.execute(
        "SELECT id, source_id FROM embed_meta WHERE source_db = 'memory'"
    ).fetchall()

    embeddings = {}
    # embed_vec may be rowid-keyed (fresh sqlite_store schema) or carry an explicit
    # 'id' PK (legacy v7/v8 DBs). Detect which, exactly as store_embeddings does;
    # hardcoding 'id' crashes ("no such column: id") on every fresh install.
    join_col = _vec_join_col(conn)
    for meta_id, source_id in meta_rows:
        if source_id not in active_ids:
            continue
        row = conn.execute(
            f"SELECT embedding FROM embed_vec WHERE {join_col} = ?", (meta_id,)
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


def mutual_topk_adjacency(sim_matrix, k):
    """Rank-based pair candidacy: adj[i][j] iff j is in i's top-k nearest
    neighbors AND i is in j's. Immune to the compressed e5 cosine space,
    where absolute thresholds are noise (measured 2026-07-03: 45% of all
    active-pair similarities cleared 0.78).
    """
    n = len(sim_matrix)
    topk = []
    for i in range(n):
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: sim_matrix[i][j], reverse=True)
        topk.append(set(order[:k]))
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j in topk[i] and i in topk[j]:
                adj[i][j] = adj[j][i] = True
    return adj


def nli_cluster_gate(cluster, mem_by_id, tau=None):
    """Phase-2 admission gate: a member stays only when it shares at least
    one line-level bidirectional-entailment fact with another member.

    Validated on the 2026-07-03 production clusters (weave-bench
    gate_replay): both cosine-built noise clusters dissolve entirely, so
    merges only fire on genuine restatements. Returns the admitted ids;
    fewer than 2 means the cluster dissolves. Gate off or NLI unavailable
    passes the cluster through unchanged (loud log).
    """
    if tau is None:
        tau = CLUSTER_GATE_TAU
    gate_mode = os.environ.get(
        "MNEMOS_CLUSTER_GATE", CLUSTER_GATE_DEFAULT).lower()
    if gate_mode == "off":
        return list(cluster)
    if not nli.is_available():
        log("  Cluster gate: NLI unavailable, passing cluster ungated")
        return list(cluster)

    def head(mid):
        content = mem_by_id.get(mid, {}).get("content", "") or ""
        lines = [ln.strip() for ln in explode_cml_chain(content).split("\n")
                 if ln.strip() and not ln.strip().startswith("---")]
        return "\n".join(lines[:CLUSTER_GATE_MAX_LINES])

    texts = {mid: head(mid) for mid in cluster}
    shares = {mid: False for mid in cluster}
    members = list(cluster)
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if shares[a] and shares[b]:
                continue
            score = nli.line_max_duplicate(texts[a], texts[b])
            if score is not None and score >= tau:
                shares[a] = shares[b] = True
    return [mid for mid in members if shares[mid]]


def find_clusters(ids, sim_matrix, threshold, mem_by_id, max_clusters=50,
                  enforce_project_boundary=True, candidacy=None, top_k=None):
    """Complete-linkage agglomerative clustering.

    candidacy 'mutual-topk' (default) builds pair candidacy from mutual
    top-k neighbor rank instead of the absolute threshold; a 0.5 cosine
    floor only guards degenerate cases. candidacy 'threshold' is the
    legacy absolute-cutoff behavior. Returns list of [memory_id, ...]
    clusters.
    """
    n = len(ids)
    if n < 2:
        return []

    if candidacy is None:
        candidacy = os.environ.get(
            "MNEMOS_CANDIDACY", CANDIDACY_DEFAULT).lower()
    mutual = None
    if candidacy == "mutual-topk":
        mutual = mutual_topk_adjacency(
            sim_matrix, top_k if top_k is not None else CANDIDACY_TOP_K)
        threshold = 0.5  # sanity floor only; rank decides candidacy

    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                if mutual is not None and not mutual[i][j]:
                    continue
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
                # concatenation so nothing is lost, but LOG IT so debugging
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
    # memory without its sources; previously we committed the merge
    # first, then tried to embed, and a failure there left orphans that
    # only `doctor --migrate` could later notice. One-transaction keeps
    # the invariant that every active memory has an embedding in the
    # same write window.
    try:
        conn.execute("BEGIN")
        tag_str = ",".join(sorted(all_tags))

        # Size-guard (v10.8.0): never emit an oversized merged memory. Split
        # losslessly into atomic siblings (no LLM, no fact loss); the primary
        # (first) id is returned so the caller's lineage contract is unchanged,
        # the rest are chained with 'related' links. consolidation_lock
        # clusters are kept whole.
        # The local merge LLM often ignores the one-fact-per-line instruction
        # and chains distinct facts on one line with ';'. Normalize that to one
        # fact per line mechanically (loss-guarded, no LLM) before the size
        # guard, so merged CML stays atomic and splittable.
        merged_content = explode_cml_chain(merged_content)
        chunks = [merged_content]
        if split_enabled() and not inherit_lock and needs_split(merged_content):
            cand = split_content(merged_content)
            if len(cand) > 1 and split_is_lossless(merged_content, cand):
                chunks = cand

        new_ids = []
        n = len(chunks)
        for i, ch in enumerate(chunks):
            ctags = tag_str if n == 1 else f"{tag_str},split-part:{i + 1}/{n}"
            conn.execute(
                "INSERT INTO memories (namespace, project, content, tags, importance, type, verified, "
                "consolidation_lock, layer, last_confirmed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'semantic', ?)",
                (_active_namespace(), project, ch, ctags, max_importance,
                 inherit_type, inherit_verified, inherit_lock, inherit_confirmed),
            )
            cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            new_ids.append(cid)

            # Embed each child inside the same transaction. A failure bubbles
            # out and the outer except ROLLBACKs the whole merge.
            text = prep_memory_text(project, ch, ctags,
                                    mem_type=inherit_type, layer="semantic")
            emb = fastembed_embed([text])
            if not (emb and emb[0]):
                raise RuntimeError("fastembed returned empty embedding")
            store_embeddings(conn, [(
                "memory", cid, text_hash(text), emb[0]
            )], model=FASTEMBED_MODEL)

        new_id = new_ids[0]

        # Chain split siblings so the cluster is walkable (no-op when n == 1).
        for a, b in zip(new_ids, new_ids[1:]):
            conn.execute(
                "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation_type, strength) "
                "VALUES (?, ?, 'related', 0.6)",
                (a, b),
            )

        for mid in cluster_ids:
            conn.execute(
                "UPDATE memories SET status = 'archived', "
                "updated_at = datetime('now', 'localtime'), "
                "tags = tags || ',merged-into-' || ? "
                "WHERE id = ?",
                (str(new_id), mid),
            )

        # Lineage row: get_merged_sources / search(expand_merged=True) read
        # exclusively from nyx_insights, so without this the primary merge
        # path produced super-memories with permanently empty merged_from
        # (tags alone are invisible to the lineage join).
        conn.execute(
            "INSERT INTO nyx_insights (memory_id, source_ids, insight_type, consolidation_type) "
            "VALUES (?, ?, 'merge', 'aggregation')",
            (new_id, ",".join(str(i) for i in cluster_ids)),
        )

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


def load_memory_meta(conn, project=None):
    """Lightweight metadata for Phase 1 triage: id, status and created_at of
    active memories, with no embeddings loaded. Mirrors load_embeddings' active
    (+ optional project) filter so triage sees the same memory set whether or
    not the LLM phases run, which lets Phase 1 run standalone in SQL-only mode.
    """
    where = "status = 'active'"
    params = []
    if project:
        where += " AND project = ?"
        params.append(project)
    rows = conn.execute(
        f"SELECT id, status, created_at FROM memories WHERE {where} ORDER BY id",
        params,
    ).fetchall()
    return {r["id"]: {"status": r["status"], "created_at": r["created_at"]} for r in rows}


# =============================================================================
# Phase 2: Dedup Merge (two-tier)
# =============================================================================

def phase_dedup(conn, mergeable_embeddings, mem_by_id, is_surge, execute=False):
    """Phase 2 dedup. Mechanical engine (v10.17.0 default): mutual-topk
    candidacy, NLI admission gate, line-union merge, no LLM. Legacy llm
    engine keeps the 10.16 two-tier generative path unchanged.
    """
    stats = {"tight_found": 0, "tight_merged": 0, "topic_found": 0,
             "topic_merged": 0, "archived": 0, "created": 0,
             "gate_dissolved": 0, "gate_ejected": 0}

    if len(mergeable_embeddings) < 2:
        log("Phase 2: Not enough mergeable memories")
        return stats

    engine = os.environ.get("MNEMOS_MERGE_ENGINE", MERGE_ENGINE_DEFAULT).lower()
    max_clusters = SURGE_MAX_CALLS if is_surge else NORMAL_MAX_CALLS

    # Tier 1A: Tight dedup
    log(f"Phase 2A: Tight dedup (engine={engine})...")
    ids, sim_matrix = cosine_similarity_matrix(mergeable_embeddings)
    tight_clusters = find_clusters(ids, sim_matrix, TIGHT_THRESHOLD,
                                   mem_by_id, max_clusters=max_clusters)
    stats["tight_found"] = len(tight_clusters)

    if tight_clusters:
        log(f"  Found {len(tight_clusters)} tight clusters")
        for i, cluster in enumerate(tight_clusters):
            _log_cluster(cluster, mem_by_id, i + 1)

            admitted = nli_cluster_gate(cluster, mem_by_id)
            ejected = [mid for mid in cluster if mid not in admitted]
            if ejected:
                stats["gate_ejected"] += len(ejected)
                log(f"  Gate ejected {ejected} (no shared fact at "
                    f">= {CLUSTER_GATE_TAU})")
            if len(admitted) < 2:
                stats["gate_dissolved"] += 1
                log("  Cluster dissolved by gate, no merge")
                continue

            if execute:
                if engine == "mechanical":
                    merged = mechanical_merge_cluster(admitted, mem_by_id)
                    if merged is None:
                        log("  Mechanical merge unavailable (NLI backend "
                            "missing?), cluster skipped")
                        stats["merge_aborted"] = stats.get("merge_aborted", 0) + 1
                        continue
                else:
                    merged = merge_cluster(admitted, mem_by_id)
                if merged:
                    new_id = apply_merge(conn, admitted, merged, mem_by_id)
                    if new_id is not None:
                        log(f"  Merged → #{new_id}, archived {admitted}")
                        stats["tight_merged"] += 1
                        stats["archived"] += len(admitted)
                        stats["created"] += 1
                        # Remove merged IDs from embeddings for tier 1B
                        for mid in admitted:
                            mergeable_embeddings.pop(mid, None)
                    else:
                        stats["merge_aborted"] = stats.get("merge_aborted", 0) + 1
                if engine != "mechanical":
                    time.sleep(0.5)

    # Tier 1B: Topic merge (looser threshold). Generative aggregation of
    # same-topic distinct facts: LLM-tier work by definition (the gate
    # rejects non-restatement clusters), so the mechanical engine retires it.
    if engine == "mechanical":
        log("Phase 2B: topic tier retired under the mechanical engine "
            "(aggregation is LLM-tier work)")
    elif len(mergeable_embeddings) >= 2:
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
    print(f"\n  Cluster {num} ({len(cluster)} memories):", flush=True)
    for mid in cluster:
        mem = mem_by_id.get(mid, {})
        content = mem.get("content", "?")[:100]
        print(f"    #{mid} [{mem.get('project', '?')}] {content}", flush=True)


# =============================================================================
# Phase 3: Thematic Weaving
# =============================================================================

def store_bridge_insight(conn, mid_a, mid_b, insight):
    """Persist a Phase-3 bridge insight as an active memory WITH its vector.

    Bridges used to be inserted content-only: active, FTS-indexed via
    trigger, but never embedded, so they stayed vector-invisible forever
    and showed up as permanently `missing` in embed_status. Embedding is
    best-effort: on embedder failure the bridge still lands (FTS-only) and
    `mnemos embed-fill` picks it up later. Returns the new memory id.
    """
    tag_str = f"synthesized,nyx-cycle,bridge,src:nyx-{time.strftime('%Y-%m-%d')}"
    if CML_MODE == "off":
        content = f"Bridge between memory #{mid_a} and #{mid_b}: {insight}"
    else:
        content = f"L: Bridge #{mid_a}↔#{mid_b}: {insight}"
    # Episodic layer (v10.17.0): bridges are derivative content and earn
    # permanence through retrieval; unretrieved ones decay at the episodic
    # half-life instead of squatting in the semantic tier forever.
    conn.execute(
        "INSERT INTO memories (namespace, project, content, tags, importance, "
        "type, layer, consolidation_lock) "
        "VALUES (?, 'personal', ?, ?, 5, 'learning', 'episodic', 1)",
        (_active_namespace(), content, tag_str),
    )
    cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    try:
        text = prep_memory_text("personal", content, tag_str,
                                mem_type="learning", layer="episodic")
        emb = fastembed_embed([text])
        if emb and emb[0]:
            store_embeddings(conn, [("memory", cid, text_hash(text), emb[0])],
                             model=FASTEMBED_MODEL)
    except Exception as e:
        log(f"    Warning: bridge #{cid} stored without vector ({e}); "
            "run `mnemos embed-fill`")
    conn.commit()
    return cid


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

    # Staleness guard (v10.17.0): a memory with an outgoing superseded_by or
    # evolves link is an outdated statement of its subject; weaving from it
    # bakes stale state into fresh insights (observed 2026-07-03: bridges
    # citing a retired LLM routing decision). The link graph already knows.
    try:
        stale = {
            row[0] for row in conn.execute(
                "SELECT DISTINCT source_id FROM memory_links "
                "WHERE relation_type IN ('superseded_by', 'evolves')"
            )
        }
    except Exception:
        stale = set()
    stale_here = original_ids & stale
    if stale_here:
        original_ids -= stale_here
        log(f"  Staleness guard: excluded {len(stale_here)} superseded/evolved sources")

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
              f"#{mid_b} [{mem_b.get('project')}] (sim={cos:.3f})", flush=True)
        print(f"    A: {mem_a.get('content', '')[:80]}", flush=True)
        print(f"    B: {mem_b.get('content', '')[:80]}", flush=True)

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
            print(f"    → No link", flush=True)
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

            # Store bridge insight as a new memory, unless it is a
            # restatement. Novelty gate (v10.17.0): an insight entailed by
            # either source alone adds no information; keep the link, skip
            # the memory. In-distribution NLI use (literal entailment), not
            # the refuted relation classification (benchmarks/weave-bench).
            if insight:
                redundant = False
                if nli.is_available():
                    for src in (mem_a, mem_b):
                        e = nli.p_entailment(
                            (src.get("content", "") or "")[:700], insight)
                        if e is not None and e >= WEAVE_NOVELTY_TAU:
                            redundant = True
                            break
                if redundant:
                    stats["insights_skipped_redundant"] = (
                        stats.get("insights_skipped_redundant", 0) + 1)
                    log(f"    → Insight redundant (entailed by a source), "
                        "link kept, memory skipped")
                else:
                    store_bridge_insight(conn, mid_a, mid_b, insight)
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

def select_contradict_candidates(ids, sim_matrix, mem_by_id, mode="cosine",
                                 min_sim=None, max_sim=None):
    """Same-project pair selection for the phase-4 contradiction scan.

    mode='cosine' keeps the legacy similarity band [min_sim, max_sim].
    mode='nli' applies the floor only: the band ceiling excluded
    near-identical pairs, which is exactly where real contradictions live
    ("X is 64GB" vs "X is 32GB" sits above 0.85 cosine). In nli mode the
    line-level finder scores the pairs, so the ceiling has no job.
    """
    if min_sim is None:
        min_sim = CONTRADICT_MIN_SIM
    if max_sim is None:
        max_sim = CONTRADICT_MAX_SIM
    pairs = []
    for i, mid_a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            mid_b = ids[j]
            if mem_by_id[mid_a]["project"] != mem_by_id[mid_b]["project"]:
                continue
            cos = float(sim_matrix[i][j])
            if cos < min_sim:
                continue
            if mode != "nli" and cos > max_sim:
                continue
            pairs.append((mid_a, mid_b, cos))
    return pairs


def phase_contradict(conn, mergeable_embeddings, mem_by_id, is_surge,
                     execute=False, judge="llm"):
    """Detect decisions that evolved, reversed, or conflict.

    judge='llm' classifies candidates immediately (10.16 behavior) and
    first consumes any contradiction-candidate links queued by earlier
    keyless runs. judge='queue' (zero-LLM tier) records flagged pairs as
    contradiction-candidate links and stops; a later llm-judged run picks
    them up. Returns stats dict.
    """
    stats = {"candidates": 0, "superseded": 0, "evolved": 0,
             "contradicts": 0, "compatible": 0, "queued": 0}

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

    finder_mode = os.environ.get(
        "MNEMOS_NYX_CONTRADICT_FINDER", NYX_CONTRADICT_FINDER_DEFAULT).lower()
    candidates = select_contradict_candidates(
        ids, sim_matrix, mem_by_id, mode=finder_mode)

    # Scan memory (v10.17.3): pairs already judged or already linked are
    # excluded before any scoring. Without this every COMPATIBLE verdict
    # was forgotten and the same pair re-entered the finder/judge loop
    # every cycle. Clearance is permanent for the pair as stored; a later
    # content update does not re-open it (known ceiling, acceptable while
    # updates re-embed rarely).
    handled = set()
    try:
        rows = conn.execute(
            "SELECT source_id, target_id FROM memory_links "
            "WHERE relation_type IN ('contradicts', 'superseded_by', "
            "'evolves', 'contradiction-cleared', 'contradiction-candidate')"
        ).fetchall()
        for s, t in rows:
            handled.add((s, t))
            handled.add((t, s))
    except Exception:
        pass
    if handled:
        before = len(candidates)
        candidates = [(a, b, c) for a, b, c in candidates
                      if (a, b) not in handled]
        if before != len(candidates):
            log(f"  Scan memory: skipped {before - len(candidates)} "
                "already-judged/linked pairs")
    candidates.sort(key=lambda x: x[2], reverse=True)
    max_eval = (SURGE_MAX_CALLS // 2) if is_surge else (NORMAL_MAX_CALLS // 2)

    if finder_mode == "nli" and nli.is_available():
        # Line-level NLI scores each cosine-gated pair; only pairs the
        # finder flags reach the LLM judge. Recall-first threshold - the
        # judge owns precision. Scores are memoized in nli_scan_cache
        # (v10.18.0), keyed on content hashes: a pair's score is a pure
        # function of the two contents, so unchanged pairs cost nothing
        # on later runs. NLI_FINDER_MAX_PAIRS therefore budgets only NEW
        # scorings; never-scored pairs beyond the budget backfill across
        # subsequent nights (measured before the cache: a 342-fact store
        # re-scored ~185 static pairs for ~31 minutes, nightly).
        def _chash(mid):
            return text_hash(mem_by_id[mid].get("content", "") or "")

        cache = {}
        try:
            for pmin, pmax, ah, bh, p in conn.execute(
                    "SELECT pair_min, pair_max, a_hash, b_hash, p_contra "
                    "FROM nli_scan_cache"):
                cache[(pmin, pmax)] = (ah, bh, p)
        except Exception:
            pass

        scored, upserts = [], []
        fresh_budget = NLI_FINDER_MAX_PAIRS
        hits = deferred = 0
        for mid_a, mid_b, cos in candidates:
            key = (min(mid_a, mid_b), max(mid_a, mid_b))
            ha, hb = _chash(key[0]), _chash(key[1])
            hit = cache.get(key)
            if hit and hit[0] == ha and hit[1] == hb:
                hits += 1
                p = hit[2]
            elif fresh_budget > 0:
                fresh_budget -= 1
                p = nli.line_max_contradiction(
                    mem_by_id[mid_a].get("content", ""),
                    mem_by_id[mid_b].get("content", ""))
                if p is None:
                    continue
                upserts.append((key[0], key[1], ha, hb, p))
            else:
                deferred += 1
                continue
            if p >= NLI_FINDER_THRESHOLD:
                scored.append((mid_a, mid_b, cos, p))
        if upserts and execute:
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO nli_scan_cache "
                    "(pair_min, pair_max, a_hash, b_hash, p_contra) "
                    "VALUES (?, ?, ?, ?, ?)", upserts)
                conn.commit()
            except Exception as e:
                log(f"  Scan cache write failed ({e}); scores not memoized")
        scored.sort(key=lambda x: x[3], reverse=True)
        log(f"  NLI finder: {len(scored)} flagged at P(contra) >= "
            f"{NLI_FINDER_THRESHOLD} ({hits} cached, {len(upserts)} newly "
            f"scored, {deferred} deferred to backfill)")
        candidates = [(a, b, c) for a, b, c, _ in scored]
    elif finder_mode == "nli":
        log("  NLI finder requested but transformers/torch unavailable; "
            "using cosine candidates unscored")

    candidates = candidates[:max_eval]

    if judge == "queue":
        # Zero-LLM tier: record flagged pairs as contradiction-candidate
        # links (idempotent) for a later llm-judged run. No archiving, no
        # valid_until edits; queue entries are inert until judged.
        stats["candidates"] = len(candidates)
        for mid_a, mid_b, cos in candidates:
            log(f"  Queued candidate #{mid_a} x #{mid_b} (sim={cos:.3f})")
            if execute:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_links "
                    "(source_id, target_id, relation_type, strength) "
                    "VALUES (?, ?, 'contradiction-candidate', ?)",
                    (mid_a, mid_b, round(min(cos, 0.99), 3)),
                )
                stats["queued"] += 1
        if execute:
            conn.commit()
        log(f"Phase 4 done (queue mode): {stats['queued']} candidates queued "
            "for the LLM tier")
        return stats

    # judge == 'llm': consume any queued candidates from earlier keyless
    # runs first, then fresh candidates. Queued links are deleted after
    # judging (replaced by the verdict link) or when a member is gone.
    queued_pairs = []
    try:
        rows = conn.execute(
            "SELECT source_id, target_id FROM memory_links "
            "WHERE relation_type = 'contradiction-candidate'"
        ).fetchall()
        for s, t in rows:
            if execute:
                conn.execute(
                    "DELETE FROM memory_links WHERE relation_type = "
                    "'contradiction-candidate' AND source_id = ? AND target_id = ?",
                    (s, t),
                )
            if s in mem_by_id and t in mem_by_id:
                queued_pairs.append((s, t, 1.0))
        if execute and rows:
            conn.commit()
        if queued_pairs:
            log(f"  Consuming {len(queued_pairs)} queued candidates from "
                "earlier zero-LLM runs")
    except Exception as e:
        log(f"  Queue read failed ({e}); continuing with fresh candidates")

    seen_pairs = {tuple(sorted((a, b))) for a, b, _ in queued_pairs}
    candidates = queued_pairs + [
        (a, b, c) for a, b, c in candidates
        if tuple(sorted((a, b))) not in seen_pairs
    ]
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
              f"#{newer_id} ({newer['created_at'][:10]}) [{older.get('project')}] sim={cos:.3f}", flush=True)
        print(f"    Older: {older.get('content', '')[:80]}", flush=True)
        print(f"    Newer: {newer.get('content', '')[:80]}", flush=True)

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
            # Blast-radius guard: the classifier's verdict is derived from
            # memory content interpolated into the prompt, i.e. from data the
            # project's own rules treat as untrusted. A steered verdict must
            # not be able to archive load-bearing memories, so verified and
            # importance>=9 memories get the link recorded but keep their
            # status; a human or the owning agent decides.
            if older.get("verified") or older.get("importance", 5) >= 9:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_links "
                    "(source_id, target_id, relation_type, strength) "
                    "VALUES (?, ?, 'superseded_by', 0.9)",
                    (older_id, newer_id),
                )
                conn.commit()
                stats["superseded_skipped"] = stats.get("superseded_skipped", 0) + 1
                log(f"    → SUPERSEDED verdict on #{older_id} NOT applied "
                    "(verified/high-importance guard); link recorded only")
                continue
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
            # searches exclude it; the newer memory supersedes it
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
            # Tombstone so the pair never re-enters the finder/judge loop
            # (see the scan-memory filter in candidate selection).
            conn.execute(
                "INSERT OR IGNORE INTO memory_links "
                "(source_id, target_id, relation_type, strength) "
                "VALUES (?, ?, 'contradiction-cleared', 0.1)",
                (older_id, newer_id),
            )
            conn.commit()
            print(f"    → Compatible (cleared)", flush=True)

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
            print(f"    #{mid} [{mem.get('project')}] {mem.get('content', '')[:80]}", flush=True)
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
            "INSERT INTO memories (namespace, project, content, tags, importance, "
            "type, layer, consolidation_lock) "
            "VALUES (?, 'personal', ?, ?, 5, 'learning', 'semantic', 1)",
            (_active_namespace(), insight, tag_str),
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
                days = conn.execute(
                    "SELECT julianday('now','localtime') - julianday(?)",
                    (m["last_accessed"],),
                ).fetchone()[0]
                score += max(0, 5 - days / 7)
            except Exception:
                pass

        # Importance
        score += m.get("importance", 5) * 0.3

        # New links boost
        score += link_counts.get(mid, 0) * 2

        # New memories boost
        try:
            days = conn.execute(
                "SELECT julianday('now','localtime') - julianday(?)",
                (m["created_at"],),
            ).fetchone()[0]
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
