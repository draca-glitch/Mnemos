"""
Mnemos core: orchestration for store/search/get/update operations.

This module contains the high-level retrieval pipeline:
  - 3-way dedup on store (FTS + CML + vector → cross-encoder rerank)
  - Hybrid search (FTS + vector → RRF merge → optional rerank)
  - Auto-widen on thin results
  - Contradiction detection
  - Dynamic importance bumping

The actual data persistence is delegated to a MnemosStore implementation,
making Mnemos backend-agnostic.
"""

import math
import re
from typing import Optional

from .storage.base import MnemosStore, Memory
from .storage.sqlite_store import SQLiteStore
from .embed import embed, prep_memory_text
from .rerank import rerank, rrf_merge
from .query import fts_dedup
from .constants import (
    HYBRID_MIN_MEMORIES, IMPORTANCE_THRESHOLDS, VEC_DEDUP_MAX_DISTANCE,
    DEDUP_RERANK_THRESHOLD, CONTRADICTION_VEC_THRESHOLD,
    CONTRADICTION_RERANK_MIN, CONTRADICTION_RERANK_HIGH,
    DEFAULT_NAMESPACE, DEFAULT_ENABLE_RERANK, DEFAULT_RETRIEVAL_LOG,
    DEFAULT_CONTRADICT_MODE,
    VALID_TYPES, VALID_LAYERS, CML_MODE,
)


CML_TYPE_PREFIXES = ("D:", "C:", "F:", "L:", "P:", "W:")


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _extract_cml_subject(content):
    """Extract subject from CML-formatted content (D:foo, F:bar, etc.)"""
    first_line = content.strip().split("\n")[0]
    for prefix in CML_TYPE_PREFIXES:
        if first_line.startswith(prefix):
            rest = first_line[len(prefix):].strip()
            subject = re.split(r'[→∵⚠@△✓✗∅;>\s]', rest)[0].strip()
            if subject:
                return prefix[0], subject
    return None, None


class Mnemos:
    """Main Mnemos interface for store, search, get, update memories.

    By default uses SQLiteStore. Pass any MnemosStore implementation to use
    a different backend (Qdrant, Postgres, etc.).
    """

    def __init__(
        self,
        store: Optional[MnemosStore] = None,
        namespace: str = DEFAULT_NAMESPACE,
        enable_rerank: bool = DEFAULT_ENABLE_RERANK,
        enable_contradiction_detection: bool = True,
        enable_retrieval_log: bool = DEFAULT_RETRIEVAL_LOG,
    ):
        self.store = store or SQLiteStore(namespace=namespace)
        self.namespace = namespace
        self.enable_rerank = enable_rerank
        self.enable_contradiction_detection = enable_contradiction_detection
        self.enable_retrieval_log = enable_retrieval_log

    # --- Store ---

    def store_memory(self, project, content, tags="", importance=5,
                     mem_type="fact", layer="semantic", verified=False,
                     subcategory=None, valid_from=None, valid_until=None,
                     consolidation_lock=False,
                     skip_dedup=False) -> dict:
        """Store a new memory with full pipeline: dedup → embed → store → contradiction check."""
        if not project or not content:
            return {"error": "project and content are required"}
        # Strip NUL bytes from content and tags. SQLite tolerates them fine
        # but downstream consumers (jq, shell scripts, some strict JSON
        # parsers) truncate or reject at NUL. Silent data loss in recipients
        # is worse than a noisy mutation here.
        if content and "\x00" in content:
            content = content.replace("\x00", "")
        if tags and "\x00" in tags:
            tags = tags.replace("\x00", "")
        if mem_type not in VALID_TYPES:
            mem_type = "fact"
        if layer not in VALID_LAYERS:
            layer = "semantic"
        importance = max(1, min(10, int(importance)))

        # Dedup check
        cached_embedding = None
        if not skip_dedup:
            dupe, cached_embedding = self._unified_dedup(
                content, project, tags, mem_type=mem_type, layer=layer
            )
            if dupe:
                return dupe

        # Embed if not already
        if cached_embedding is None:
            text = prep_memory_text(project, content, tags, mem_type=mem_type, layer=layer)
            embeddings = embed([text], prefix="passage")
            if embeddings and embeddings[0]:
                cached_embedding = embeddings[0]

        # Build Memory and persist
        memory = Memory(
            namespace=self.namespace,
            project=project,
            content=content,
            tags=tags or "",
            importance=importance,
            type=mem_type,
            layer=layer,
            verified=1 if verified else 0,
            subcategory=subcategory,
            valid_from=valid_from,
            valid_until=valid_until,
            consolidation_lock=1 if consolidation_lock else 0,
        )
        mid = self.store.store_memory(memory, embedding=cached_embedding)

        result = {
            "id": mid, "status": "stored", "project": project,
            "importance": importance, "type": mem_type, "layer": layer,
            "embedded": cached_embedding is not None,
        }
        if subcategory:
            result["subcategory"] = subcategory
        if valid_from:
            result["valid_from"] = valid_from
        if valid_until:
            result["valid_until"] = valid_until

        # Contradiction detection (Tier 1-3 pipeline, MNEMOS_CONTRADICT_MODE
        # selects rerank|llm|vec|off; see _detect_contradictions). Only
        # returns warnings for classifications that need caller attention:
        # contradicts, refines, evolves. Silent `relates` links are
        # persisted but not surfaced.
        if self.enable_contradiction_detection and cached_embedding:
            contradictions = self._detect_contradictions(mid, content, project, cached_embedding)
            if contradictions:
                result["contradictions"] = contradictions
                # Summarize by classification so callers can tell at a glance
                # whether it's a real contradict vs a refines/evolves hint.
                counts = {}
                for c in contradictions:
                    cls = c.get("classification", "contradicts")
                    counts[cls] = counts.get(cls, 0) + 1
                parts = ", ".join(f"{n} {cls}" for cls, n in counts.items())
                result["contradiction_warning"] = (
                    f"⚠ relationship flag(s) detected: {parts}. Review conflicting memories."
                )

        return result

    def _unified_dedup(self, content, project, tags, mem_type="", layer=""):
        """3-way dedup: FTS + CML + vec → cross-encoder rerank → confidence."""
        candidates = {}

        # 1) FTS keyword overlap
        fts_hits = fts_dedup(self.store, content)
        for d in fts_hits:
            mid = d["id"]
            candidates[mid] = {
                "id": mid, "content": d["content"], "project": d.get("project", ""),
                "methods": {"fts"}, "fts_similarity": d["similarity"],
            }

        # 2) CML subject conflict. Same CML prefix (D:/C:/F:/L:/P:/W:)
        # and same subject token in the same project is a strong dedup
        # signal that FTS can miss when the subject is a short, rare token
        # (FTS5 tokenizers drop 1-2 char words, and BM25 under-weights
        # very short queries). Only runs on SQLite-backed stores since
        # we need a raw conn for the LIKE scan; other backends simply
        # skip this tier and rely on FTS + vector.
        cml_type, subject = _extract_cml_subject(content)
        if cml_type and subject and CML_MODE != "off" and hasattr(self.store, "_get_conn"):
            try:
                conn = self.store._get_conn()
                pattern = f"{cml_type}:{subject}%"
                rows = conn.execute(
                    "SELECT id, content, project FROM memories "
                    "WHERE status='active' AND namespace=? AND project=? "
                    "AND content LIKE ? LIMIT 5",
                    (self.namespace, project, pattern),
                ).fetchall()
                for row in rows:
                    mid = row["id"] if hasattr(row, "keys") else row[0]
                    if mid in candidates:
                        candidates[mid]["methods"].add("cml")
                    else:
                        r_content = row["content"] if hasattr(row, "keys") else row[1]
                        r_project = row["project"] if hasattr(row, "keys") else row[2]
                        candidates[mid] = {
                            "id": mid, "content": r_content,
                            "project": r_project, "methods": {"cml"},
                        }
            except Exception:
                pass

        # 3) Vector similarity
        text = prep_memory_text(project, content, tags or "", mem_type=mem_type, layer=layer)
        embedding = None
        if text.strip():
            embeddings = embed([text], prefix="passage")
            if embeddings and embeddings[0]:
                embedding = embeddings[0]
                vec_results = self.store.search_vec(embedding, limit=5)
                for source_id, distance in vec_results:
                    if distance < VEC_DEDUP_MAX_DISTANCE * 1.5:
                        if source_id in candidates:
                            candidates[source_id]["methods"].add("vec")
                            candidates[source_id]["vec_distance"] = distance
                        else:
                            mems = self.store.get_memories_by_ids([source_id])
                            if source_id in mems:
                                m = mems[source_id]
                                candidates[source_id] = {
                                    "id": source_id, "content": m.content,
                                    "project": m.project, "methods": {"vec"},
                                    "vec_distance": distance,
                                }

        if not candidates:
            return None, embedding

        # Rerank with cross-encoder
        if self.enable_rerank:
            try:
                docs = [{"text": c["content"], "id": c["id"]} for c in candidates.values()]
                ranked = rerank(content, docs)
                best = ranked[0]
                best_id = best["id"]
                raw_score = best.get("_rerank_score", 0)
                score = _sigmoid(raw_score)
            except Exception:
                # Reranker not available, use highest method-count signal
                score = 0.75
                best_id = max(candidates.keys(), key=lambda k: len(candidates[k]["methods"]))
        else:
            score = 0.75
            best_id = max(candidates.keys(), key=lambda k: len(candidates[k]["methods"]))

        if score < DEDUP_RERANK_THRESHOLD:
            return None, embedding

        cand = candidates[best_id]
        methods = sorted(cand["methods"])
        return {
            "existing_id": best_id,
            "confidence": round(score, 4),
            "methods": methods,
            "existing_content": cand["content"][:200],
            "warning": f"Likely duplicate ({', '.join(methods)}, confidence {score:.0%})",
            "hint": f"Consider updating #{best_id} instead",
        }, embedding

    # Valid classifications returned by the Tier-3 LLM classifier, plus
    # their treatment: whether to emit a warning to the caller and which
    # link type to persist. `relates` is the v10.3.0 addition that kills
    # the false-positive noise from same-topic-complementary pairs.
    _CONTRADICTION_CLASSES = {
        # classification  → (link_type, warn)
        "contradicts":      ("contradicts",  True),
        "refines":          ("refines",       True),
        "evolves":          ("evolves",       True),
        "relates":          ("relates",       False),  # silent link
        "unrelated":        (None,            False),  # no link
    }

    def _detect_contradictions(self, new_id, content, project, embedding):
        """Find topically similar memories and classify the relationship.

        Three-tier pipeline (v10.3.0, 2026-04-16). Previous single-threshold
        design flagged any same-topic pair as a contradiction because the
        cross-encoder scores topical overlap, not semantic conflict. The
        new scheme introduces a `relates` link type for moderate-score
        pairs (silent, no warning) and reserves `contradicts` for pairs
        that either score above CONTRADICTION_RERANK_HIGH or are
        explicitly classified as contradictory by an LLM.

        Tiers:
          1. Vec gate (L2 < CONTRADICTION_VEC_THRESHOLD, same-project filter)
          2. Cross-encoder rerank (drops vec-candidates below MIN threshold)
          3. Classification:
             - mode='vec'    → all vec-gated = contradicts (pre-v10.3 behavior)
             - mode='rerank' → score >= HIGH = contradicts, MIN..HIGH = relates
             - mode='llm'    → LLM classifies all rerank survivors into one of
                               {contradicts, refines, evolves, relates, unrelated}

        Mode is set via MNEMOS_CONTRADICT_MODE env (default 'rerank').
        """
        import os
        mode = os.environ.get(
            "MNEMOS_CONTRADICT_MODE", DEFAULT_CONTRADICT_MODE,
        ).lower()
        if mode == "off":
            return []

        # Tier 1: vec gate
        vec_results = self.store.search_vec(embedding, limit=10)
        if not vec_results:
            return []
        candidate_ids = [
            sid for sid, dist in vec_results
            if sid != new_id and dist < CONTRADICTION_VEC_THRESHOLD
        ]
        if not candidate_ids:
            return []
        memories = self.store.get_memories_by_ids(candidate_ids)
        # Same-project filter (contradictions almost always live within a
        # project; cross-project topical overlap is mostly complementary)
        memories = {
            mid: m for mid, m in memories.items() if m.project == project
        }
        if not memories:
            return []

        # Tier 1-only mode: treat every vec-gated candidate as contradicts.
        # This is the honest path for users who explicitly opt out of
        # rerank: pick mode=vec. The `rerank` and `llm` modes require
        # the cross-encoder by design — it's the component that makes
        # the `relates` silent-link refinement possible. If a caller
        # sets mode=rerank with MNEMOS_ENABLE_RERANK=0 we let rerank()
        # fail (it will throw or return empty), and `_detect_contradictions`
        # returns []. That's their choice to cripple the pipeline.
        if mode == "vec":
            warnings = []
            for mid, mem in memories.items():
                self.store.store_link(new_id, mid, "contradicts", 0.5)
                warnings.append(self._contradiction_warning(
                    mid, mem, 0.5, "contradicts",
                ))
            return warnings

        # Tier 2: cross-encoder rerank (canonical for modes rerank and llm)
        try:
            docs = [{"text": m.content, "id": mid} for mid, m in memories.items()]
            ranked = rerank(content, docs)
        except Exception:
            return []

        llm_available = False
        if mode == "llm":
            try:
                from .consolidation.llm import is_configured as _llm_ok
                llm_available = _llm_ok()
            except Exception:
                llm_available = False

        warnings = []
        for hit in ranked:
            score = _sigmoid(hit.get("_rerank_score", 0))
            if score < CONTRADICTION_RERANK_MIN:
                continue  # not even topically similar
            hit_id = hit["id"]
            mem = memories.get(hit_id)
            if not mem:
                continue

            # Tier 3: classify
            classification = self._classify_relationship(
                content, mem.content, score, mode, llm_available,
            )
            link_type, warn = self._CONTRADICTION_CLASSES.get(
                classification, ("relates", False),
            )

            # Persist link (skipped for `unrelated`)
            if link_type is not None:
                self.store.store_link(new_id, hit_id, link_type, round(score, 4))

            # Emit warning only for classifications that need caller attention
            if warn:
                warnings.append(self._contradiction_warning(
                    hit_id, mem, score, classification,
                ))
        return warnings

    def _classify_relationship(self, new_content, existing_content, score,
                               mode, llm_available):
        """Return one of: contradicts / refines / evolves / relates / unrelated.

        Rerank-only mode uses the score as a crude proxy: HIGH score =
        contradicts (same topic + strong match could be disagreement), MIN..HIGH
        score = relates (same topic, complementary). LLM mode replaces the
        rerank-score heuristic at Tier 3 with an actual semantic call.
        """
        if mode == "llm" and llm_available:
            try:
                from .consolidation.llm import chat
                msgs = [
                    {"role": "system", "content": (
                        "Classify the relationship between two memory entries. "
                        "Reply with exactly one word from: contradicts, refines, "
                        "evolves, relates, unrelated.\n\n"
                        "contradicts: explicit conflict (A says X, B says not-X on same subject).\n"
                        "refines: B refines/expands/corrects a fact in A (same subject, added detail, no conflict).\n"
                        "evolves: B is a temporally-later update of A's fact (A was true then, B is true now).\n"
                        "relates: same topic but complementary, no conflict, no temporal progression.\n"
                        "unrelated: different topics despite surface similarity."
                    )},
                    {"role": "user", "content": (
                        f"Entry A:\n{(existing_content or '')[:600]}\n\n"
                        f"Entry B (new):\n{(new_content or '')[:600]}\n\n"
                        "Classification:"
                    )},
                ]
                response = chat(msgs, max_tokens=16, temperature=0.0)
                if response:
                    # Be defensive about whitespace-only or punctuation-only
                    # responses: .split()[0] would IndexError on "" / "\n".
                    words = response.strip().lower().split()
                    word = words[0].strip(".,:;!?") if words else ""
                    if word and word in self._CONTRADICTION_CLASSES:
                        return word
            except Exception:
                pass
            # LLM failed → fall through to rerank heuristic

        # Rerank-score heuristic (also the default when mode='rerank')
        if score >= CONTRADICTION_RERANK_HIGH:
            return "contradicts"
        return "relates"

    @staticmethod
    def _contradiction_warning(hit_id, mem, score, classification):
        link_type = Mnemos._CONTRADICTION_CLASSES.get(classification, ("relates", False))[0]
        return {
            "conflicting_id": hit_id,
            "conflicting_content": (mem.content or "")[:200],
            "conflicting_project": mem.project,
            "similarity": round(score, 4),
            "classification": classification,
            "suggested_action": f"link:{link_type}" if link_type else "no_action",
        }

    # --- Search ---

    def search(self, query, project=None, subcategory=None, layer=None,
               type_filter=None, status="active", valid_only=False,
               search_mode=None, limit=20, auto_widen=True,
               expand_merged=False, snippet_chars=None,
               include_linked=False, linked_depth=1) -> dict:
        # Clamp linked_depth at entry: negative values would make the
        # BFS guard `if dist >= linked_depth` trivially true after the
        # root, silently disabling all link expansion. The MCP tool
        # schema caps at 1..3 but direct Python callers can pass any
        # int. Same defensive posture as snippet_chars / limit.
        if include_linked:
            linked_depth = max(1, min(int(linked_depth or 1), 3))
        """Hybrid retrieval: FTS5 + vector search + RRF merge + optional rerank.

        When `expand_merged=True`, results that were created by the Nyx
        cycle (consolidated super-memories) are enriched with their original
        source memories under a `merged_from` key, filtered to exclude any
        sources marked as superseded via `valid_until`. This is the
        "tier-2 recall" pattern: gist memory in the result, episodic detail
        available on demand.
        """
        if not query:
            return {"error": "query is required"}
        limit = min(limit, 50)

        if not search_mode:
            count = self.store.count_active()
            search_mode = "hybrid" if count >= HYBRID_MIN_MEMORIES else "fts"

        # FTS search
        fts_ids = []
        if search_mode in ("fts", "hybrid"):
            fts_ids = self.store.search_fts(
                query, project=project, subcategory=subcategory,
                layer=layer, type_filter=type_filter, status=status,
                valid_only=valid_only, limit=limit * 2,
            )

        # Vector search
        vec_ids = []
        query_embedding = None
        if search_mode in ("vec", "hybrid") and self.store.supports_vec():
            embeddings = embed([query], prefix="query")
            if embeddings and embeddings[0]:
                query_embedding = embeddings[0]
                vec_results = self.store.search_vec(
                    query_embedding, project=project, subcategory=subcategory,
                    layer=layer, type_filter=type_filter, status=status,
                    valid_only=valid_only, limit=limit * 2,
                )
                vec_ids = [r[0] for r in vec_results]

        # Merge.
        # For hybrid mode we fetch a wider pool from RRF so the cross-encoder
        # reranker (applied below) actually has candidates to reorder.
        rerank_pool = max(limit * 3, 20)
        if search_mode == "hybrid" and fts_ids and vec_ids:
            merged_ids = rrf_merge(fts_ids, vec_ids)[:rerank_pool]
        elif search_mode == "vec":
            merged_ids = vec_ids[:limit]
        else:
            merged_ids = fts_ids[:limit]

        # Auto-widen on thin results
        widened = False
        if auto_widen and project and len(merged_ids) < 3:
            widen_fts = self.store.search_fts(
                query, project=None, subcategory=subcategory,
                layer=layer, type_filter=type_filter, status=status,
                valid_only=valid_only, limit=10,
            )
            extra = [mid for mid in widen_fts if mid not in merged_ids][: 3 - len(merged_ids)]
            if extra:
                merged_ids.extend(extra)
                widened = True

        # Fetch full memories (wider pool if rerank will run)
        memories = self.store.get_memories_by_ids(merged_ids)

        # Cross-encoder reranking (default on for hybrid).
        # Skipped on tiny result sets where reordering cannot help.
        if (search_mode == "hybrid" and self.enable_rerank
                and len(merged_ids) >= 4):
            try:
                docs = [
                    {"id": mid, "text": memories[mid].content}
                    for mid in merged_ids
                    if mid in memories
                ]
                ranked = rerank(query, docs)
                merged_ids = [d["id"] for d in ranked][:limit]
            except Exception:
                merged_ids = merged_ids[:limit]
        else:
            merged_ids = merged_ids[:limit]

        results = [memories[mid].to_dict() for mid in merged_ids if mid in memories]

        # Attach links
        link_map = {}
        if merged_ids:
            link_map = self.store.get_links(merged_ids)
            for r in results:
                if r["id"] in link_map:
                    r["links"] = link_map[r["id"]]

        # Snippet extraction: replace full content with a query-matched window.
        # Caller opts in with snippet_chars; default (None) keeps full content
        # for backward compatibility. Memories that don't match the FTS query
        # (pure vec hits) fall back to a sentence scored by query-word
        # overlap, then to a head slice if no sentence scores above zero.
        # The `snippet` flag on a result indicates its content was
        # actually modified, not merely that snippet_chars was requested.
        if snippet_chars and snippet_chars > 0:
            try:
                snippet_map = self.store.get_snippets(
                    merged_ids, query, chars=snippet_chars
                )
            except Exception:
                snippet_map = {}
            for r in results:
                mid = r["id"]
                if mid in snippet_map and snippet_map[mid]:
                    r["content"] = snippet_map[mid]
                    r["snippet"] = True
                elif r.get("content") and len(r["content"]) > snippet_chars:
                    # Vec-only hit (no FTS match). Pick the best sentence by
                    # query-word overlap, else fall back to head slice.
                    r["content"] = self._vec_fallback_snippet(
                        r["content"], query, snippet_chars,
                    )
                    r["snippet"] = True
                # else: content is already short enough, no modification
                # — `snippet` flag is not set

        # Linked memory expansion: fold linked memories as summaries into
        # each result. BFS traversal up to `linked_depth` hops, with cycle
        # detection (visited set) and a total-nodes cap to prevent graph
        # explosion. Each summary carries `distance` = hops from the
        # originating result so callers can tell immediate links apart
        # from transitive ones.
        if include_linked and link_map:
            # Cap total linked nodes surfaced per result to keep response
            # sizes bounded. Graph blowup at depth=3 in a well-linked
            # store can easily reach hundreds; this cap keeps responses
            # usable while preserving the spirit of "see the graph."
            MAX_LINKED_PER_RESULT = 30
            already_in_results = {r["id"] for r in results}

            for r in results:
                visited = {r["id"]}
                frontier = [(r["id"], 0)]  # (node_id, distance-from-root)
                collected = []  # [{id, project, relation, strength, distance, content}]
                local_link_cache = dict(link_map)  # will expand as we BFS

                while frontier and len(collected) < MAX_LINKED_PER_RESULT:
                    node_id, dist = frontier.pop(0)
                    if dist >= linked_depth:
                        continue
                    # For nodes beyond the initial result set, we haven't
                    # fetched their links yet — fetch lazily on first visit
                    if node_id not in local_link_cache:
                        try:
                            newlinks = self.store.get_links([node_id])
                            local_link_cache.update(newlinks)
                        except Exception:
                            local_link_cache[node_id] = []
                    for link in local_link_cache.get(node_id, []):
                        lid = link["linked_id"]
                        if lid in visited:
                            continue
                        visited.add(lid)
                        # Skip linking to something already in the top-level
                        # result set — callers already have it, don't double
                        if lid in already_in_results:
                            continue
                        collected.append({
                            "id": lid,
                            "relation": link["relation"],
                            "strength": link["strength"],
                            "distance": dist + 1,  # hops from root
                            "_via": node_id,       # which node reached it
                        })
                        frontier.append((lid, dist + 1))
                        if len(collected) >= MAX_LINKED_PER_RESULT:
                            break

                # Bulk-fetch all collected memory bodies in one query
                if collected:
                    linked_ids = [c["id"] for c in collected]
                    fetched = self.store.get_memories_by_ids(linked_ids)
                    summaries = []
                    for c in collected:
                        mem = fetched.get(c["id"]) or memories.get(c["id"])
                        if not mem:
                            continue
                        summary = {
                            "id": c["id"],
                            "project": mem.project,
                            "relation": c["relation"],
                            "strength": c["strength"],
                            "distance": c["distance"],
                            "content": (mem.content or "")[:200],
                        }
                        # Only include `via` for depth > 1 to avoid clutter
                        # on the common depth=1 path
                        if c["distance"] > 1:
                            summary["via"] = c["_via"]
                        summaries.append(summary)
                    if summaries:
                        r["linked_memories"] = summaries

        # Tier-2 recall: expand consolidated memories to their source originals
        if expand_merged:
            for r in results:
                try:
                    sources = self.store.get_merged_sources(r["id"], valid_only=True)
                except Exception:
                    sources = []
                if sources:
                    r["merged_from"] = [s.to_dict() for s in sources]

        # Opt-in retrieval logging: best-effort side channel, never affects
        # the returned result. Logs (query, memory_id) per returned hit so
        # callers can later mine real-query distribution for benchmarks,
        # quality analysis, and autoimprove cycles.
        if self.enable_retrieval_log and results:
            try:
                self.store.log_retrieval(
                    query=query,
                    memory_ids=[r["id"] for r in results if "id" in r],
                )
            except Exception:
                pass

        response = {
            "count": len(results),
            "search_mode": search_mode,
            "results": results,
        }
        if widened:
            response["widened"] = True
        if expand_merged:
            response["expand_merged"] = True
        if snippet_chars:
            response["snippet_chars"] = snippet_chars
        if include_linked:
            response["include_linked"] = True
        return response

    # --- Get / Update / Delete ---

    def get(self, mid: int) -> dict:
        memory = self.store.get_memory(mid, increment_access=True)
        if not memory:
            return {"error": f"Memory #{mid} not found"}

        # Dynamic importance bumping
        new_count = memory.access_count
        new_importance = memory.importance
        for threshold, min_imp in IMPORTANCE_THRESHOLDS:
            if new_count >= threshold and memory.importance < min_imp:
                new_importance = min_imp
                break
        if new_importance != memory.importance:
            self.store.update_memory(mid, {"importance": new_importance})
            memory.importance = new_importance

        return memory.to_dict()

    def update(self, mid: int, **fields) -> dict:
        if not fields:
            return {"error": "no fields to update"}

        # Normalize boolean-ish fields the DB stores as 0/1 integers
        for bool_field in ("consolidation_lock", "verified"):
            if bool_field in fields:
                fields[bool_field] = 1 if fields[bool_field] else 0

        # Re-embed if content/tags/type/layer change
        needs_embed = any(k in fields for k in ("content", "tags", "type", "layer"))
        embedding = None
        if needs_embed:
            current = self.store.get_memory(mid, increment_access=False)
            if not current:
                return {"error": f"Memory #{mid} not found"}
            merged = {**current.to_dict(), **fields}
            text = prep_memory_text(
                merged.get("project", ""),
                merged.get("content", ""),
                merged.get("tags", ""),
                mem_type=merged.get("type", ""),
                layer=merged.get("layer", ""),
            )
            embeddings = embed([text], prefix="passage")
            if embeddings and embeddings[0]:
                embedding = embeddings[0]

        ok = self.store.update_memory(mid, fields, embedding=embedding)
        if not ok:
            return {"error": f"Memory #{mid} not found"}
        return {"id": mid, "status": "updated", "fields": list(fields.keys())}

    def delete(self, mid: int, hard: bool = False) -> dict:
        ok = self.store.delete_memory(mid, hard=hard)
        if not ok:
            return {"error": f"Memory #{mid} not found"}
        return {"id": mid, "status": "deleted" if hard else "archived"}

    def stats(self) -> dict:
        return self.store.stats()

    # Stop words for query-token extraction when scoring vec-only snippet
    # fallback. Tiny list — we want to keep "python", "vec", etc. but drop
    # noise like "the", "is", "and". Not a full stopword lexicon on purpose:
    # this runs inline per-hit, cost matters more than perfect recall.
    _SNIPPET_STOPWORDS = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "if", "of", "in", "on", "to", "for",
        "with", "at", "by", "from", "as", "this", "that", "these",
        "those", "i", "you", "we", "they", "it", "its",
        "how", "what", "when", "where", "why", "who",
        "do", "does", "did", "can", "could", "should", "would",
    ])

    @classmethod
    def _vec_fallback_snippet(cls, content: str, query: str, chars: int) -> str:
        """Pick the best sentence in `content` by query-word overlap.

        Used when a search hit matched via vec similarity but NOT FTS, so
        the FTS `snippet()` call returned nothing useful. Splitting into
        sentences and scoring by substantive-word overlap is cheap (no
        extra embedding calls) and usually beats a head slice: even when
        the query terms aren't lexically identical to the content, common
        informative words often overlap enough to identify the best
        sentence.

        Falls back to head slice when no sentence scores above zero
        (e.g., query terms are semantically present but lexically absent).
        """
        if not content or not content.strip():
            # Defensive: whitespace-only content would produce "" + " …"
            # fragment, which is cosmetic junk. Early return is honest.
            return ""
        if chars <= 0:
            # Negative / zero budgets: return empty rather than Python's
            # content[:negative] semantics that truncate from the end.
            return ""
        # Extract substantive query words (lowercased, stopwords dropped,
        # length >= 3 to skip short tokens that add noise).
        import re
        q_tokens = {
            t for t in re.findall(r"\w+", query.lower())
            if len(t) >= 3 and t not in cls._SNIPPET_STOPWORDS
        }
        if not q_tokens:
            return content[:chars] + (" …" if len(content) > chars else "")

        # Sentence split on . ! ? (preserve sentence content; skip split
        # inside abbreviations via a simple heuristic: require space after
        # the punctuation). Good enough for CML / prose content.
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        if not sentences:
            return content[:chars] + " …"

        best_sentence = sentences[0]
        best_score = 0
        for s in sentences:
            s_tokens = set(re.findall(r"\w+", s.lower()))
            score = len(q_tokens & s_tokens)
            if score > best_score:
                best_score = score
                best_sentence = s

        if best_score == 0:
            # No token overlap anywhere; return head slice
            return content[:chars] + " …"

        # Truncate the winning sentence if it's still too long, or pad
        # with adjacent context up to the budget.
        if len(best_sentence) <= chars:
            return best_sentence
        return best_sentence[:chars] + " …"

    @staticmethod
    def _match_snippet(text: str, anchor: str, chars: int = 120) -> str:
        """Extract a window of approximately `chars` centered on the first
        occurrence of `anchor` in `text`. Falls back to head slice if the
        anchor isn't present (anchor text may have been replaced away).
        """
        if not text:
            return ""
        idx = text.find(anchor) if anchor else -1
        if idx < 0:
            return text[:chars] + (" …" if len(text) > chars else "")
        half = max(chars // 2, 40)
        start = max(0, idx - half)
        end = min(len(text), idx + len(anchor) + half)
        prefix = "… " if start > 0 else ""
        suffix = " …" if end < len(text) else ""
        return prefix + text[start:end] + suffix

    def bulk_rewrite(self, pattern: str, replacement: str,
                     project: Optional[str] = None,
                     tags: Optional[str] = None,
                     dry_run: bool = True,
                     max_affected: int = 50,
                     use_regex: bool = False,
                     preview_chars: int = 120) -> dict:
        """Find-and-replace across memories with preview-commit flow.

        Safety invariants:
          - dry_run=True by default: returns preview without writing
          - max_affected cap: aborts before any write if exceeded
          - namespace-scoped: only matches memories in self.namespace
          - active-only: archived memories excluded
          - re-embeds every modified memory (content changed → embedding
            must too, else vector search drifts from truth)

        Returns dict with keys: matched, affected, changes, dry_run,
        optional error. Each change has {id, before, after, diff_chars}
        where before/after are snippets around the match site.
        """
        if not pattern:
            return {"error": "pattern is required", "matched": 0,
                    "affected": 0, "changes": [], "dry_run": dry_run}
        if not hasattr(self.store, "_get_conn"):
            return {"error": "bulk_rewrite only supported on SQLite-based stores",
                    "matched": 0, "affected": 0, "changes": [],
                    "dry_run": dry_run}

        import re
        regex = None
        if use_regex:
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return {"error": f"invalid regex: {e}", "matched": 0,
                        "affected": 0, "changes": [], "dry_run": dry_run}

        conn = self.store._get_conn()
        where = "status = 'active' AND namespace = ?"
        params: list = [self.namespace]
        if project:
            where += " AND project = ?"
            params.append(project)
        if tags:
            # Word-boundary tag match: wrap both the stored tags CSV and the
            # search pattern with commas so the LIKE query only matches the
            # intended tag atom. Previous `LIKE '%foo%'` matched tags like
            # 'unnamed' when user asked for 'name' — substring leak.
            where += " AND (',' || tags || ',') LIKE ?"
            params.append(f"%,{tags},%")

        if use_regex:
            rows = conn.execute(
                f"SELECT id, content FROM memories WHERE {where}", params,
            ).fetchall()
            matches = [(r["id"], r["content"]) for r in rows
                       if regex.search(r["content"] or "")]
        else:
            rows = conn.execute(
                f"SELECT id, content FROM memories WHERE {where} AND content LIKE ?",
                params + [f"%{pattern}%"],
            ).fetchall()
            matches = [(r["id"], r["content"]) for r in rows]

        # Compute actual changes (filter out no-op rewrites where pattern
        # matched but replacement was identical, e.g. idempotent rerun)
        changes = []
        for mid, content in matches:
            if use_regex:
                new_content = regex.sub(replacement, content or "")
            else:
                new_content = (content or "").replace(pattern, replacement)
            if new_content == content:
                continue
            changes.append({
                "id": mid,
                "_new_content": new_content,
                "before": self._match_snippet(
                    content or "", pattern if not use_regex else "",
                    preview_chars,
                ),
                "after": self._match_snippet(
                    new_content, replacement, preview_chars,
                ),
                "diff_chars": len(new_content) - len(content or ""),
            })

        if len(changes) > max_affected:
            return {
                "error": (
                    f"Would modify {len(changes)} memories, exceeds "
                    f"max_affected={max_affected}. Tighten filters or "
                    f"raise max_affected explicitly."
                ),
                "matched": len(matches),
                "affected": 0,
                "changes": [],
                "dry_run": dry_run,
            }

        affected = 0
        if not dry_run:
            for change in changes:
                ok = self.update(change["id"], content=change["_new_content"])
                if isinstance(ok, dict) and ok.get("status") == "updated":
                    affected += 1

        # Strip internal field before returning
        for change in changes:
            change.pop("_new_content", None)

        return {
            "matched": len(matches),
            "affected": affected if not dry_run else len(changes),
            "changes": changes,
            "dry_run": dry_run,
        }

    def list_tags(self, project=None, min_count=1, order_by="count",
                  limit=500) -> list:
        """List unique tags across active memories with usage counts.

        Useful for discovering existing tagging conventions before
        creating new tags, preventing tag drift ("authoritative" vs
        "canonical" vs "verified" for the same concept).
        """
        return self.store.list_tags(
            namespace=self.namespace, project=project,
            min_count=min_count, order_by=order_by, limit=limit,
        )

    # --- Briefing / digest / map / health ---

    @staticmethod
    def _briefing_line(content: str, max_chars: int = 160) -> str:
        """Truncate a memory's first line for briefing with a natural boundary
        and an ellipsis marker.

        Prefers sentence-ending punctuation ('. ', '! ', '? ') over clause
        boundaries ('; ', ', ') over word boundaries, and always appends
        ' …' when truncation occurred. Previously, raw [:180] slicing left
        fragments like '... we don\\'t' that read as mid-sentence even when
        they happened to land on a word boundary.
        """
        line = (content or "").split("\n")[0].strip()
        if len(line) <= max_chars:
            return line
        cut = line[:max_chars]
        min_cut = max_chars // 3
        for sep in (". ", "! ", "? "):
            pos = cut.rfind(sep)
            if pos > min_cut:
                return cut[:pos + 1] + " …"
        for sep in ("; ", ", "):
            pos = cut.rfind(sep)
            if pos > min_cut:
                return cut[:pos] + " …"
        pos = cut.rfind(" ")
        if pos > min_cut:
            return cut[:pos] + " …"
        return cut + "…"

    def briefing(self, project: Optional[str] = None, budget_chars: int = 1300) -> str:
        """Compact session-start briefing of top memories by importance + recency.

        Returns a markdown string fitting roughly `budget_chars` characters.
        Designed to be injected into AI session context (~370 tokens at default).
        Per-entry truncation uses sentence-aware boundaries with an ellipsis
        marker so lines that got cut off read cleanly rather than dangling
        mid-sentence.
        """
        if hasattr(self.store, "_get_conn"):
            conn = self.store._get_conn()
            where = "WHERE m.status = 'active' AND m.namespace = ?"
            params = [self.namespace]
            if project:
                where += " AND m.project = ?"
                params.append(project)
            try:
                rows = conn.execute(
                    f"SELECT id, project, content, importance, type "
                    f"FROM memories m {where} "
                    f"ORDER BY importance DESC, last_accessed DESC LIMIT 30",
                    params,
                ).fetchall()
            except Exception:
                rows = []
        else:
            rows = []

        lines = []
        used = 0
        header = f"# Mnemos briefing ({self.namespace})\n\n"
        used += len(header)
        for r in rows:
            line = f"- [{r['project']}] {self._briefing_line(r['content'])}\n"
            if used + len(line) > budget_chars:
                break
            lines.append(line)
            used += len(line)
        return header + "".join(lines)

    def digest(self, days: int = 7, project: Optional[str] = None) -> list:
        """Recent memories from the last N days."""
        if not hasattr(self.store, "_get_conn"):
            return []
        conn = self.store._get_conn()
        where = "WHERE status = 'active' AND namespace = ?"
        params = [self.namespace]
        where += " AND julianday('now', 'localtime') - julianday(created_at) <= ?"
        params.append(days)
        if project:
            where += " AND project = ?"
            params.append(project)
        rows = conn.execute(
            f"SELECT id, project, content, created_at, importance "
            f"FROM memories {where} ORDER BY created_at DESC LIMIT 100",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def map(self) -> dict:
        """Compact topic index: count by project/subcategory."""
        if not hasattr(self.store, "_get_conn"):
            return {}
        conn = self.store._get_conn()
        rows = conn.execute(
            "SELECT project, subcategory, COUNT(*) as n "
            "FROM memories WHERE status='active' AND namespace=? "
            "GROUP BY project, subcategory ORDER BY project, n DESC",
            (self.namespace,),
        ).fetchall()
        result = {}
        for r in rows:
            project, sub, n = r["project"], r["subcategory"], r["n"]
            if project not in result:
                result[project] = {"total": 0, "subcategories": {}}
            result[project]["total"] += n
            if sub:
                result[project]["subcategories"][sub] = n
        return result

    def embed_status(self) -> dict:
        """Embedding coverage report."""
        if not hasattr(self.store, "_get_conn"):
            return {"error": "embed status only supported on SQLite-based stores"}
        conn = self.store._get_conn()
        active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active' AND namespace=?",
            (self.namespace,),
        ).fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM embed_meta em "
            "JOIN memories m ON m.id = em.source_id "
            "WHERE em.source_db = 'memory' AND m.status='active' AND m.namespace=?",
            (self.namespace,),
        ).fetchone()[0]
        return {
            "active": active,
            "embedded": embedded,
            "missing": active - embedded,
            "coverage": round(embedded / active, 4) if active else 0.0,
        }

    def doctor(self, migrate: bool = False) -> dict:
        """Health check + optional self-repair. Sanity-checks the store and
        reports issues; when `migrate=True`, safely applies migrations for
        detected schema drift with an automatic pre-migration DB backup.

        Migrations are safe and reversible:
          - Missing columns in `memories` → ALTER TABLE ADD COLUMN with
            documented defaults (idempotent, triggered by init_schema
            backfill in v10.3.4+, doctor --migrate just re-runs it)
          - Missing `namespace` index → CREATE INDEX
          - Missing `retrieval_log` / `tool_usage` / `consolidation_log` /
            `nyx_state` tables → CREATE TABLE from current schema
        Never drops data. Backup path is returned in the report so callers
        can roll back.
        """
        if not hasattr(self.store, "_get_conn"):
            return {"error": "doctor only supported on SQLite-based stores"}
        conn = self.store._get_conn()
        report = {
            "namespace": self.namespace,
            "checks": [],
            "issues": [],
            "migrations_applied": [],
        }

        # --- Schema drift detection ---
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        required = {"id", "project", "content", "type", "layer", "subcategory",
                    "valid_from", "valid_until", "namespace"}
        missing_cols = required - cols
        if missing_cols:
            report["issues"].append(f"Missing columns in memories: {sorted(missing_cols)}")
        else:
            report["checks"].append("Schema is up to date (v10)")

        # Detect missing auxiliary tables that v10.2+ expects
        existing_tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected_aux = {"retrieval_log", "tool_usage", "consolidation_log", "nyx_state"}
        missing_tables = expected_aux - existing_tables
        if missing_tables:
            report["issues"].append(f"Missing aux tables: {sorted(missing_tables)}")

        # Backup before any migrations
        if migrate and (missing_cols or missing_tables) and hasattr(self.store, "db_path"):
            import shutil, time
            src = self.store.db_path
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = f"{src}.bak-pre-doctor-migrate-{ts}"
            try:
                shutil.copy2(src, backup)
                report["backup"] = backup
            except Exception as e:
                report["issues"].append(f"Backup failed; aborting migrations: {e}")
                migrate = False  # bail out safely

        if migrate and missing_cols:
            # init_schema already knows how to backfill; invoking it is the
            # cleanest migration path since it has the authoritative column
            # defaults baked in.
            try:
                self.store.init_schema()
                post_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
                backfilled = sorted(missing_cols & post_cols)
                if backfilled:
                    report["migrations_applied"].append(
                        f"Backfilled columns: {backfilled}"
                    )
            except Exception as e:
                report["issues"].append(f"Column migration failed: {e}")

        if migrate and missing_tables:
            try:
                # init_schema also creates aux tables if absent
                self.store.init_schema()
                post_tables = {
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                created = sorted(missing_tables & post_tables)
                if created:
                    report["migrations_applied"].append(
                        f"Created aux tables: {created}"
                    )
            except Exception as e:
                report["issues"].append(f"Table migration failed: {e}")

        # --- FTS sanity ---
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            if mem_count != fts_count:
                report["issues"].append(
                    f"FTS index out of sync: {mem_count} memories vs {fts_count} FTS rows"
                )
                if migrate:
                    try:
                        # Rebuild FTS from scratch (idempotent, cheap at <50K memories)
                        conn.execute(
                            "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
                        )
                        conn.commit()
                        new_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                        report["migrations_applied"].append(
                            f"Rebuilt FTS index ({new_count} rows)"
                        )
                    except Exception as e:
                        report["issues"].append(f"FTS rebuild failed: {e}")
            else:
                report["checks"].append(f"FTS index synced ({fts_count} rows)")
        except Exception as e:
            report["issues"].append(f"FTS check failed: {e}")

        # --- Embedding coverage ---
        cov = self.embed_status()
        if cov.get("missing", 0) > 0:
            report["issues"].append(
                f"{cov['missing']} active memories without embeddings ({cov['coverage']:.0%} coverage)"
            )
            # Note: doctor --migrate does NOT auto-embed missing memories;
            # embedding cost/time is caller-controlled. Document as pending.
            if migrate:
                report["checks"].append(
                    "Embedding backfill skipped (runtime cost; use `mnemos embed-fill` "
                    "or call store_memory on affected rows to trigger re-embed)"
                )
        else:
            report["checks"].append(f"Embeddings: 100% coverage ({cov.get('embedded', 0)} vectors)")

        # Re-check issues post-migration so callers see the clean state
        if migrate and report["migrations_applied"]:
            # Filter out issues that were resolved
            post_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
            post_tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            resolved_issues = []
            for issue in report["issues"]:
                if "Missing columns" in issue and not (required - post_cols):
                    continue  # resolved
                if "Missing aux tables" in issue and not (expected_aux - post_tables):
                    continue
                resolved_issues.append(issue)
            report["issues"] = resolved_issues

        report["status"] = "healthy" if not report["issues"] else "issues_detected"
        return report

    # CWD → (project, subcategory) heuristic used by prime(). Empty by
    # default; applications with known repo layouts should subclass and
    # override, or pass `cwd_map` at call time. Without this, a bare
    # CWD signal produces a vec query that matches random memories
    # across all projects.
    CWD_PROJECT_MAP: list = []

    @classmethod
    def _resolve_cwd_context(cls, cwd: Optional[str],
                             cwd_map: "Optional[list]" = None):
        """Map a working directory to (project, subcategory, tokens).

        `cwd_map` is a list of (path_prefix, project, subcategory) tuples.
        First matching prefix wins, so more-specific paths should come
        first. Returns (None, None, []) if no mapping matches.
        """
        if not cwd:
            return None, None, []
        mapping = cwd_map if cwd_map is not None else cls.CWD_PROJECT_MAP
        for prefix, proj, subcat in mapping:
            if cwd.startswith(prefix):
                tokens = [proj]
                if subcat:
                    tokens.append(subcat)
                return proj, subcat, tokens
        return None, None, []

    def prime(self, context: str, project: Optional[str] = None,
              limit: int = 5, cwd: Optional[str] = None,
              cwd_map: "Optional[list]" = None) -> list:
        """Surface memories likely relevant to a context string.

        Used by session hooks to inject relevant memories at startup or
        on first user prompt. Lightweight wrapper around hybrid search.

        When `cwd` is provided, the CWD is resolved via `cwd_map` (or
        the class-level `CWD_PROJECT_MAP`) to an inferred project and
        subcategory. The inferred project becomes the exclusive filter
        for results (unless the caller passed an explicit `project`
        that takes precedence), and the project/subcat tokens are
        prepended to the vec query so the semantic anchor is stronger
        than path-token bleed alone.
        """
        # CWD heuristic: infer project/subcategory from the working directory
        cwd_project, _cwd_subcat, cwd_tokens = self._resolve_cwd_context(cwd, cwd_map)
        effective_project = project or cwd_project

        # Augment the context string with CWD-derived tokens if any
        if cwd_tokens:
            augmented = " ".join(cwd_tokens) + (" " + context if context else "")
        else:
            augmented = context

        if not augmented:
            return []
        result = self.search(
            augmented, project=effective_project, limit=limit, search_mode="hybrid",
        )
        return result.get("results", [])

    def consolidate(self, execute: bool = False,
                    phases: "set[int] | None" = None,
                    surge: bool = False,
                    project: "str | None" = None) -> dict:
        """Run the Nyx cycle consolidation. Requires LLM for phases 1-4.

        Phase 5 (Bookkeeping) always runs without LLM. See
        mnemos.consolidation.llm for environment variable configuration.
        """
        from .consolidation import run_nyx_cycle
        return run_nyx_cycle(
            self.store, execute=execute, phases=phases,
            surge=surge, project=project,
        )

    def close(self):
        self.store.close()
