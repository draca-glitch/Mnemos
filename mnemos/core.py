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
    CONTRADICTION_RERANK_THRESHOLD, DEFAULT_NAMESPACE,
    DEFAULT_ENABLE_RERANK,
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
    ):
        self.store = store or SQLiteStore(namespace=namespace)
        self.namespace = namespace
        self.enable_rerank = enable_rerank
        self.enable_contradiction_detection = enable_contradiction_detection

    # --- Store ---

    def store_memory(self, project, content, tags="", importance=5,
                     mem_type="fact", layer="semantic", verified=False,
                     subcategory=None, valid_from=None, valid_until=None,
                     consolidation_lock=False,
                     skip_dedup=False) -> dict:
        """Store a new memory with full pipeline: dedup → embed → store → contradiction check."""
        if not project or not content:
            return {"error": "project and content are required"}
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

        # Contradiction detection
        if self.enable_contradiction_detection and cached_embedding:
            contradictions = self._detect_contradictions(mid, content, project, cached_embedding)
            if contradictions:
                result["contradictions"] = contradictions
                result["contradiction_warning"] = (
                    f"⚠ {len(contradictions)} potential contradiction(s) detected. "
                    f"Review conflicting memories."
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

    def _detect_contradictions(self, new_id, content, project, embedding):
        """Find topically similar memories that might contradict the new one.

        Precision-first filter chain (tightened 2026-04-09 after an
        autoimprove iteration eliminated 4 false positives observed in a
        store batch):

          1. Vector similarity gate  (L2 < CONTRADICTION_VEC_THRESHOLD)
          2. Same-project filter     (cross-project topical overlap is
                                      almost always complementary, not
                                      contradictory, this kills the bulk
                                      of observed FPs)
          3. Cross-encoder rerank    (score >= CONTRADICTION_RERANK_THRESHOLD)

        The cross-encoder answers "are these about the same topic?" rather
        than "do these say opposite things?", so it cannot be trusted alone
        as a contradiction signal, vec gate + project filter carry most of
        the precision load.
        """
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
        if not memories:
            return []

        # Same-project filter: drop candidates from other projects before
        # running the (expensive) reranker. Contradictions almost always
        # live inside the same project; cross-project topical matches are
        # the dominant false-positive source.
        memories = {
            mid: m for mid, m in memories.items()
            if m.project == project
        }
        if not memories:
            return []

        try:
            docs = [{"text": m.content, "id": mid} for mid, m in memories.items()]
            ranked = rerank(content, docs)
        except Exception:
            return []

        contradictions = []
        for hit in ranked:
            score = _sigmoid(hit.get("_rerank_score", 0))
            if score < CONTRADICTION_RERANK_THRESHOLD:
                continue
            hit_id = hit["id"]
            mem = memories.get(hit_id)
            if not mem:
                continue
            self.store.store_link(new_id, hit_id, "contradicts", round(score, 4))
            contradictions.append({
                "conflicting_id": hit_id,
                "conflicting_content": mem.content[:200],
                "conflicting_project": mem.project,
                "similarity": round(score, 4),
            })
        return contradictions

    # --- Search ---

    def search(self, query, project=None, subcategory=None, layer=None,
               type_filter=None, status="active", valid_only=False,
               search_mode=None, limit=20, auto_widen=True,
               expand_merged=False, snippet_chars=None,
               include_linked=False, linked_depth=1) -> dict:
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
        # (pure vec hits) fall back to a head slice of the content. The
        # `snippet` flag on a result indicates its content was actually
        # modified, not merely that snippet_chars was requested.
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
                    # Fallback for vec-only hits: head slice
                    r["content"] = r["content"][:snippet_chars] + " …"
                    r["snippet"] = True
                # else: content is already short enough, no modification
                # — `snippet` flag is not set

        # Linked memory expansion: fold first-hop linked memories as summaries
        # into each result. Saves round-trips for callers that want to see
        # the relationship graph without a follow-up memory_get per link.
        if include_linked and link_map:
            # Collect unique linked IDs across all results (depth=1 only for now)
            linked_ids = set()
            for r in results:
                for link in link_map.get(r["id"], []):
                    linked_ids.add(link["linked_id"])
            # Don't refetch memories that are already in the result set
            already = {r["id"] for r in results}
            to_fetch = list(linked_ids - already)
            fetched = self.store.get_memories_by_ids(to_fetch) if to_fetch else {}
            for r in results:
                linked_summaries = []
                for link in link_map.get(r["id"], []):
                    lid = link["linked_id"]
                    m = fetched.get(lid) or memories.get(lid)
                    if not m:
                        continue
                    linked_summaries.append({
                        "id": lid,
                        "project": m.project,
                        "relation": link["relation"],
                        "strength": link["strength"],
                        "content": (m.content or "")[:200],
                    })
                if linked_summaries:
                    r["linked_memories"] = linked_summaries

        # Tier-2 recall: expand consolidated memories to their source originals
        if expand_merged:
            for r in results:
                try:
                    sources = self.store.get_merged_sources(r["id"], valid_only=True)
                except Exception:
                    sources = []
                if sources:
                    r["merged_from"] = [s.to_dict() for s in sources]

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

    def briefing(self, project: Optional[str] = None, budget_chars: int = 1300) -> str:
        """Compact session-start briefing of top memories by importance + recency.

        Returns a markdown string fitting roughly `budget_chars` characters.
        Designed to be injected into AI session context (~370 tokens at default).
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
            line = f"- [{r['project']}] {r['content'][:180]}\n"
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

    def doctor(self) -> dict:
        """Health check, sanity-check the store and report issues."""
        if not hasattr(self.store, "_get_conn"):
            return {"error": "doctor only supported on SQLite-based stores"}
        conn = self.store._get_conn()
        report = {"namespace": self.namespace, "checks": [], "issues": []}

        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        required = {"id", "project", "content", "type", "layer", "subcategory",
                    "valid_from", "valid_until", "namespace"}
        missing = required - cols
        if missing:
            report["issues"].append(f"Missing columns in memories: {sorted(missing)}")
        else:
            report["checks"].append("Schema is up to date (v10)")

        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            if mem_count != fts_count:
                report["issues"].append(
                    f"FTS index out of sync: {mem_count} memories vs {fts_count} FTS rows"
                )
            else:
                report["checks"].append(f"FTS index synced ({fts_count} rows)")
        except Exception as e:
            report["issues"].append(f"FTS check failed: {e}")

        cov = self.embed_status()
        if cov.get("missing", 0) > 0:
            report["issues"].append(
                f"{cov['missing']} active memories without embeddings ({cov['coverage']:.0%} coverage)"
            )
        else:
            report["checks"].append(f"Embeddings: 100% coverage ({cov.get('embedded', 0)} vectors)")

        report["status"] = "healthy" if not report["issues"] else "issues_detected"
        return report

    def prime(self, context: str, project: Optional[str] = None, limit: int = 5) -> list:
        """Surface memories likely relevant to a context string.

        Used by session hooks to inject relevant memories at startup.
        Lightweight wrapper around hybrid search.
        """
        if not context:
            return []
        result = self.search(context, project=project, limit=limit, search_mode="hybrid")
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
