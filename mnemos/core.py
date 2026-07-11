"""
Mnemos core: orchestration for store/search/get/update hot-path plus
bulk_rewrite/list_tags maintenance operations.

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
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional

from .storage.base import MnemosStore, Memory
from .storage.sqlite_store import SQLiteStore
from .embed import embed, prep_memory_text, text_hash as embed_text_hash
from .rerank import rerank, rrf_merge
from .query import fts_dedup
from . import nli
from .constants import (
    HYBRID_MIN_MEMORIES, IMPORTANCE_THRESHOLDS, VEC_DEDUP_MAX_DISTANCE,
    DEDUP_RERANK_THRESHOLD, CONTRADICTION_VEC_THRESHOLD,
    CONTRADICTION_RERANK_MIN, CONTRADICTION_RERANK_HIGH,
    DEFAULT_NAMESPACE, DEFAULT_ENABLE_RERANK, DEFAULT_RETRIEVAL_LOG,
    DEFAULT_CONTRADICT_MODE, DEDUP_CONFIRM_DEFAULT,
    NLI_CONTRA_THRESHOLD, NLI_DEDUP_THRESHOLD, NLI_DEDUP_MAX_CANDIDATES,
    VALID_TYPES, VALID_LAYERS, CML_MODE,
)


CML_TYPE_PREFIXES = ("D:", "C:", "F:", "L:", "P:", "W:", "R:")


@contextmanager
def _regex_time_limit():
    """SIGALRM-based bound on caller-supplied regex scans.

    Python's re has no timeout, so a catastrophic-backtracking pattern from
    the MCP-exposed bulk_rewrite would hang the single-threaded server loop
    forever, dry_run included. Unix main-thread only; degrades to a no-op
    elsewhere (signal.signal outside the main thread raises ValueError).
    Override seconds via MNEMOS_BULK_REWRITE_TIMEOUT, 0 disables."""
    try:
        seconds = int(os.environ.get("MNEMOS_BULK_REWRITE_TIMEOUT", "30") or 0)
    except ValueError:
        seconds = 30
    if (seconds <= 0 or os.name == "nt"
            or threading.current_thread() is not threading.main_thread()):
        yield
        return
    import signal

    def _raise(signum, frame):
        raise TimeoutError("regex scan timed out")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _summarize_quick_check(rows):
    """Reduce PRAGMA quick_check output to (ok, summary). quick_check yields a
    single 'ok' row when clean, or one row per problem when corrupt; report the
    first few plus a count so the extent of corruption isn't hidden."""
    msgs = [r[0] for r in rows] or ["ok"]
    if msgs == ["ok"]:
        return True, "ok"
    shown = "; ".join(msgs[:3])
    if len(msgs) > 3:
        shown += f"; (+{len(msgs) - 3} more)"
    return False, shown


def _corruption_hint(exc):
    """If exc looks like SQLite on-disk corruption, return an actionable hint,
    else None. Turns a raw 'database disk image is malformed' into guidance."""
    msg = str(exc).lower()
    if "malformed" in msg or "disk image" in msg or "not a database" in msg:
        return ("the database may be corrupted; run `mnemos doctor` to check "
                "and restore the latest `mnemos backup` snapshot if needed")
    return None


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
                     skip_dedup=False, _no_split=False) -> dict:
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

        # v10.4.0: opt-in cemelify-on-import. When MNEMOS_CEMELIFY_ON_IMPORT=1,
        # rewrite raw content into CML before persistence. Falls back silently
        # to the original content on any LLM failure (this flag must never
        # turn LLM into a hard dependency). Skipped when the caller opts into
        # consolidation_lock (prose-protection convention).
        import os as _os
        if _os.environ.get("MNEMOS_CEMELIFY_ON_IMPORT") == "1" and not consolidation_lock:
            try:
                from .cemelify import cemelify as _cemelify
                content = _cemelify(content) or content
            except Exception:
                pass

        if mem_type not in VALID_TYPES:
            mem_type = "fact"
        if layer not in VALID_LAYERS:
            layer = "semantic"
        importance = max(1, min(10, int(importance)))

        # Size-guard (v10.8.0): split oversized content into atomic sibling
        # memories, losslessly and without an LLM, so no single memory balloons
        # past the threshold. Skipped for consolidation_lock (intentional whole
        # prose) and when the caller already split (an un-splittable single line).
        # Normalize single-line prefix-chained CML (';'-joined facts) to one
        # fact per line before the size guard. Mechanical, loss-guarded, no LLM;
        # a blob it cannot safely explode is returned unchanged.
        from .splitter import split_enabled, needs_split, explode_cml_chain
        if not _no_split:
            content = explode_cml_chain(content)
        if (not _no_split and not consolidation_lock and split_enabled()
                and needs_split(content)):
            return self._store_split(
                project, content, tags, importance, mem_type, layer,
                verified, subcategory, valid_from, valid_until,
            )

        # Dedup check
        cached_embedding = None
        if not skip_dedup:
            dupe, cached_embedding = self._unified_dedup(
                content, project, tags, mem_type=mem_type, layer=layer
            )
            if dupe:
                return dupe

        # Embed if not already. The hash is of the canonical embed text and
        # travels with the vector so staleness stays detectable; it applies
        # equally to dedup-cached embeddings, which embed the same text.
        canonical_text = prep_memory_text(project, content, tags, mem_type=mem_type, layer=layer)
        if cached_embedding is None:
            embeddings = embed([canonical_text], prefix="passage")
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
        mid = self.store.store_memory(
            memory, embedding=cached_embedding,
            text_hash=embed_text_hash(canonical_text) if cached_embedding else None,
        )

        result = {
            "id": mid, "status": "stored", "project": project,
            "importance": importance, "type": mem_type, "layer": layer,
            "embedded": cached_embedding is not None,
        }
        if cached_embedding is None and getattr(self.store, "supports_vec", lambda: False)():
            result["warning"] = (
                "embedding failed; memory stored FTS-only and invisible to "
                "vector search until backfilled, run `mnemos embed-fill`"
            )
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

    def _store_split(self, project, content, tags, importance, mem_type, layer,
                     verified, subcategory, valid_from, valid_until):
        """Store oversized content as atomic sibling memories, losslessly.

        Splits content with the mechanical line splitter (no LLM, no fact
        loss), stores each chunk through the normal single-store path so each
        child is embedded and FTS-indexed correctly, then chains the children
        with 'related' links into a traceable cluster. Returns a summary with
        all child ids. Phase 1 layers a hub plus parent/child links on top.
        """
        from .splitter import split_content, split_is_lossless, explode_cml_lines
        chunks = split_content(content)
        # Never risk losing a fact: if the mechanical split is not provably
        # lossless (or did not actually split), store the original whole.
        if len(chunks) <= 1 or not split_is_lossless(content, chunks):
            return self.store_memory(
                project, content, tags=tags, importance=importance,
                mem_type=mem_type, layer=layer, verified=verified,
                subcategory=subcategory, valid_from=valid_from,
                valid_until=valid_until, skip_dedup=True, _no_split=True,
            )
        base = ",".join(
            t for t in (tags or "").split(",")
            if t and not t.startswith(("split-from:#", "split-part:"))
        )
        n = len(chunks)
        ids = []
        for i, ch in enumerate(chunks):
            # The size splitter is line-preserving, so a packed
            # multi-statement line lands verbatim in its child; explode it
            # here because _no_split below skips the store-path exploder.
            ch = explode_cml_lines(ch)
            extra = f"split-part:{i + 1}/{n},nyx-split"
            child_tags = f"{base},{extra}" if base else extra
            res = self.store_memory(
                project, ch, tags=child_tags, importance=importance,
                mem_type=mem_type, layer=layer, verified=verified,
                subcategory=subcategory, valid_from=valid_from,
                valid_until=valid_until, skip_dedup=True, _no_split=True,
            )
            if res.get("id"):
                ids.append(res["id"])
        # Chain siblings with 'related' links so the cluster is walkable without
        # an O(n^2) mesh. Phase 1 replaces this with a hub plus parent/child.
        for a, b in zip(ids, ids[1:]):
            self.store.store_link(a, b, "related", strength=0.6)
        return {
            "status": "stored-split", "ids": ids, "parts": len(ids),
            "project": project, "importance": importance, "type": mem_type,
            "layer": layer,
        }

    def remediate_oversized(self, min_size=4000, max_size=None, dry_run=False, limit=None,
                            include_archived=False, hard=False):
        """Split existing oversized active memories into atomic sibling memories.

        For each active, non-consolidation_lock memory in this namespace whose
        content exceeds min_size (and is at most max_size when given): split it
        losslessly with the mechanical splitter, store the chunks as atomic
        siblings inheriting project/type/layer/importance/subcategory/valid_*/
        verified, tag each child split-from:#<orig>, chain them with 'related'
        links, re-point the original's links onto the first child, then archive
        the original and move its vector to the tier-2 archived index. All
        mutations go through store APIs so FTS and vec stay consistent. Returns
        a summary dict. Reuses the same splitter as the live store path.
        """
        from .splitter import (split_content, split_is_lossless,
                               split_preserves_all_sentences, explode_cml_lines)

        ids = self.store.find_oversized_memories(
            self.namespace, min_size, max_size=max_size,
            include_archived=include_archived, limit=limit,
        )
        if not ids:
            return {
                "scanned": 0, "split": 0, "children_created": 0,
                "archived": 0, "skipped_unsplittable": 0, "errors": 0,
                "dry_run": dry_run, "min_size": min_size, "max_size": max_size,
            }

        summary = {
            "scanned": len(ids), "split": 0, "children_created": 0,
            "archived": 0, "skipped_unsplittable": 0, "errors": 0,
            "dry_run": dry_run, "min_size": min_size, "max_size": max_size,
        }

        prev_contra = self.enable_contradiction_detection
        self.enable_contradiction_detection = False  # bulk: skip per-child LLM checks
        try:
            for oid in ids:
                try:
                    mem = self.store.get_memory(oid, increment_access=False)
                    if not mem:
                        continue
                    chunks = split_content(mem.content, hard=hard)
                    gate = split_preserves_all_sentences if hard else split_is_lossless
                    if len(chunks) <= 1 or not gate(mem.content, chunks):
                        summary["skipped_unsplittable"] += 1
                        continue
                    if dry_run:
                        summary["split"] += 1
                        summary["children_created"] += len(chunks)
                        continue
                    base = ",".join(
                        t for t in (mem.tags or "").split(",")
                        if t and not t.startswith(("split-from:#", "split-part:"))
                    )
                    marker = f"split-from:#{oid}"
                    n = len(chunks)
                    child_ids = []
                    for i, ch in enumerate(chunks):
                        # Same per-line explode as _store_split: children
                        # inherit packed lines verbatim and are stored with
                        # _no_split, so this is their only atomization pass.
                        ch = explode_cml_lines(ch)
                        parts = [base, f"split-part:{i + 1}/{n}", marker, "nyx-split"]
                        ctags = ",".join(t for t in parts if t)
                        res = self.store_memory(
                            mem.project, ch, tags=ctags, importance=mem.importance,
                            mem_type=mem.type, layer=mem.layer,
                            verified=bool(mem.verified), subcategory=mem.subcategory,
                            valid_from=mem.valid_from, valid_until=mem.valid_until,
                            skip_dedup=True, _no_split=True,
                        )
                        if res.get("id"):
                            child_ids.append(res["id"])
                    if not child_ids:
                        summary["errors"] += 1
                        continue
                    for a, b in zip(child_ids, child_ids[1:]):
                        self.store.store_link(a, b, "related", strength=0.6)
                    # Re-point the original's links onto the first child so the
                    # graph stays connected after the parent is archived.
                    for l in self.store.get_links([oid]).get(oid, []):
                        other = l.get("linked_id")
                        if other and other != oid and other not in child_ids:
                            self.store.store_link(
                                child_ids[0], other,
                                l.get("relation", "related"),
                                l.get("strength", 0.5),
                            )
                    if mem.status == "active":
                        self.store.delete_memory(oid, hard=False)
                        if hasattr(self.store, "move_embedding_to_archive"):
                            try:
                                self.store.move_embedding_to_archive(oid)
                            except Exception:
                                pass
                    else:
                        # Archived original (tier-2): keep it as the lineage
                        # anchor; archive the children too so active search is
                        # unaffected and tier-2 gains sharp per-topic vectors.
                        for cid in child_ids:
                            self.store.delete_memory(cid, hard=False)
                            if hasattr(self.store, "move_embedding_to_archive"):
                                try:
                                    self.store.move_embedding_to_archive(cid)
                                except Exception:
                                    pass
                    summary["split"] += 1
                    summary["children_created"] += len(child_ids)
                    summary["archived"] += 1
                except Exception:
                    summary["errors"] += 1
        finally:
            self.enable_contradiction_detection = prev_contra
        return summary

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

        # 2) CML subject conflict. Same CML prefix (D:/C:/F:/L:/P:/W:/R:)
        # and same subject token in the same project is a strong dedup
        # signal that FTS can miss when the subject is a short, rare token
        # (FTS5 tokenizers drop 1-2 char words, and BM25 under-weights
        # very short queries). Only runs on SQLite-backed stores since
        # we need a content LIKE scan; other backends simply
        # skip this tier and rely on FTS + vector.
        cml_type, subject = _extract_cml_subject(content)
        if cml_type and subject and CML_MODE != "off":
            try:
                pattern = f"{cml_type}:{subject}%"
                rows = self.store.find_cml_subject_matches(
                    self.namespace, project, pattern, limit=5)
                for row in rows:
                    mid = row["id"]
                    if mid in candidates:
                        candidates[mid]["methods"].add("cml")
                    else:
                        candidates[mid] = {
                            "id": mid, "content": row["content"],
                            "project": row["project"], "methods": {"cml"},
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

        # NLI confirm tier (MNEMOS_DEDUP_CONFIRM=nli): bidirectional
        # entailment is the duplicate question asked directly - each side
        # must entail the other. Replaces the cross-encoder confidence
        # score, which measures topicality and over-blocks structurally
        # similar but distinct facts. When NLI says "not a duplicate" that
        # verdict is final (no fall-through to the coarser scorers); when
        # the runtime is unavailable the legacy path below takes over.
        import os as _os
        confirm_mode = _os.environ.get(
            "MNEMOS_DEDUP_CONFIRM", DEDUP_CONFIRM_DEFAULT).lower()
        if confirm_mode == "nli" and nli.is_available():
            ranked_cands = sorted(
                candidates.values(),
                key=lambda c: c.get("vec_distance", float("inf")))
            best_cand, best_score = None, None
            for cand in ranked_cands[:NLI_DEDUP_MAX_CANDIDATES]:
                s = nli.bidirectional_entailment(content, cand["content"])
                if s is None:
                    continue
                if best_score is None or s > best_score:
                    best_cand, best_score = cand, s
            if best_score is not None:
                if best_score < NLI_DEDUP_THRESHOLD:
                    return None, embedding
                best_cand["methods"].add("nli")
                methods = sorted(best_cand["methods"])
                return {
                    "existing_id": best_cand["id"],
                    "confidence": round(best_score, 4),
                    "methods": methods,
                    "existing_content": best_cand["content"][:200],
                    "warning": (f"Likely duplicate ({', '.join(methods)}, "
                                f"confidence {best_score:.0%})"),
                    "hint": f"Consider updating #{best_cand['id']} instead",
                }, embedding

        # Score the best candidate. Prefer the cross-encoder; without it, fall
        # back to the actual vector distance rather than a blanket score. The old
        # fallback assigned a flat 0.75 to any coarse (FTS/CML/vec) match, which
        # is above the dedup threshold, so EVERY coarse match was flagged a
        # duplicate and the store was blocked, distinct or not (silent data loss).
        score = None
        best_id = None
        if self.enable_rerank:
            try:
                docs = [{"text": c["content"], "id": c["id"]} for c in candidates.values()]
                ranked = rerank(content, docs)
                best_id = ranked[0]["id"]
                score = _sigmoid(ranked[0].get("_rerank_score", 0))
            except Exception:
                score = None  # reranker unavailable; use the vec-distance fallback
        if score is None:
            # No usable reranker: block only on a strong vector signal, scaled so
            # the dedup threshold is reached exactly at VEC_DEDUP_MAX_DISTANCE.
            # Candidates matched only by FTS/CML (no vec distance) are not a
            # confident duplicate without the cross-encoder, so they do not block.
            best_id = min(
                candidates, key=lambda k: candidates[k].get("vec_distance", float("inf"))
            )
            dist = candidates[best_id].get("vec_distance")
            if dist is None or dist >= VEC_DEDUP_MAX_DISTANCE:
                return None, embedding
            score = 1.0 - dist * (1.0 - DEDUP_RERANK_THRESHOLD) / VEC_DEDUP_MAX_DISTANCE

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

        # NLI mode: ask the contradiction question directly. Max-direction
        # P(contradiction) over both premise/hypothesis orders (real
        # contradictions score asymmetrically), warn + link only at the
        # high-precision threshold. No relates band: the WEAVE phase owns
        # topical linking, and the reranker's topicality signal is exactly
        # what this mode exists to remove.
        if mode == "nli":
            if not nli.is_available():
                import sys as _sys
                print("mnemos: MNEMOS_CONTRADICT_MODE=nli but transformers/"
                      "torch unavailable; skipping contradiction check "
                      "(install mnemos[nli])", file=_sys.stderr)
                return []
            warnings = []
            for mid, mem in memories.items():
                p = nli.p_contradiction(content, mem.content)
                if p is None or p < NLI_CONTRA_THRESHOLD:
                    continue
                self.store.store_link(new_id, mid, "contradicts", round(p, 4))
                warnings.append(self._contradiction_warning(
                    mid, mem, p, "contradicts",
                ))
            return warnings

        # Tier 1-only mode: treat every vec-gated candidate as contradicts.
        # This is the honest path for users who explicitly opt out of
        # rerank: pick mode=vec. The `rerank` and `llm` modes require
        # the cross-encoder by design - it's the component that makes
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
        # rerank() degrades by returning the docs unscored. Scoring those
        # with the sigmoid default would put EVERY candidate at exactly 0.5,
        # inside the `relates` band, mass-writing spurious links on any
        # reranker hiccup. No scores = no classification basis = bail.
        if not any("_rerank_score" in d for d in ranked):
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

    def search(self, *args, **kwargs):
        """Public search entry: delegates to _search_impl and converts a raw
        SQLite corruption error into an actionable `mnemos doctor` hint (the
        failure mode that surfaced opaquely at search time on 2026-06-27)."""
        try:
            return self._search_impl(*args, **kwargs)
        except sqlite3.DatabaseError as e:
            hint = _corruption_hint(e)
            if hint:
                raise sqlite3.DatabaseError(f"{e}; {hint}") from e
            raise

    def _search_impl(self, query, project=None, subcategory=None, layer=None,
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
        elif search_mode == "hybrid" and vec_ids:
            # FTS found nothing but the vector side did (typical for
            # cross-lingual queries where no token overlaps). Without this
            # branch the vec hits were computed and then thrown away.
            merged_ids = vec_ids[:rerank_pool]
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
                # - `snippet` flag is not set

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
                    # fetched their links yet - fetch lazily on first visit
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
                        # result set - callers already have it, don't double
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
                        # Search results are status-filtered; the link graph
                        # is not. Without this guard, archived content
                        # resurfaces in summaries of active results.
                        if mem.status != "active":
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
        tier2_recall = None
        if expand_merged:
            surfaced = {r["id"] for r in results if "id" in r}
            for r in results:
                try:
                    sources = self.store.get_merged_sources(r["id"], valid_only=True)
                except Exception:
                    sources = []
                if sources:
                    r["merged_from"] = [s.to_dict() for s in sources]
                    surfaced.update(s.id for s in sources)
            # v10.7.0: also KNN the archived index directly so an archived
            # original is reachable even when its consolidated parent did not
            # rank in primary search. The lineage join above can only surface
            # originals whose parent already ranked; this can't.
            if query_embedding is not None:
                try:
                    arch_hits = self.store.search_vec_archived(
                        query_embedding, project=project, subcategory=subcategory,
                        layer=layer, type_filter=type_filter, valid_only=valid_only,
                        limit=limit,
                    )
                except Exception:
                    arch_hits = []
                dist_by_id = {sid: dist for sid, dist in arch_hits}
                fresh = [sid for sid, _ in arch_hits if sid not in surfaced]
                if fresh:
                    mems = self.store.get_memories_by_ids(fresh)
                    tier2_recall = []
                    for sid in fresh:  # preserve distance order
                        mem = mems.get(sid)
                        if mem is None:
                            continue
                        d = mem.to_dict()
                        d["vec_distance"] = round(float(dist_by_id[sid]), 4)
                        tier2_recall.append(d)

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
        if tier2_recall:
            response["tier2_recall"] = tier2_recall
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

        # Useful-loop (v10.17.0): drilling into a memory shortly after a
        # search surfaced it is the zero-friction usefulness signal. Marks
        # only rows the retrieval logger wrote, so it is inert when
        # retrieval logging is off.
        if self.enable_retrieval_log:
            self.store.mark_retrieval_useful(mid)

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
        thash = None
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
                thash = embed_text_hash(text)

        ok = self.store.update_memory(mid, fields, embedding=embedding, text_hash=thash)
        if not ok:
            return {"error": f"Memory #{mid} not found"}
        result = {"id": mid, "status": "updated", "fields": list(fields.keys())}
        if needs_embed:
            # A failed re-embed leaves the OLD vector in place; the caller
            # must be able to see that, and the stale text_hash makes it
            # repairable by embed-status/sync tooling.
            result["embedded"] = embedding is not None
            if embedding is None:
                result["warning"] = (
                    "content changed but re-embedding failed; vector is stale "
                    "until the next embed sync"
                )
        return result

    def delete(self, mid: int, hard: bool = False) -> dict:
        ok = self.store.delete_memory(mid, hard=hard)
        if not ok:
            return {"error": f"Memory #{mid} not found"}
        return {"id": mid, "status": "deleted" if hard else "archived"}

    def stats(self) -> dict:
        return self.store.stats()

    # Stop words for query-token extraction when scoring vec-only snippet
    # fallback. Tiny list - we want to keep "python", "vec", etc. but drop
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

        import re
        regex = None
        if use_regex:
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return {"error": f"invalid regex: {e}", "matched": 0,
                        "affected": 0, "changes": [], "dry_run": dry_run}

        try:
            from contextlib import nullcontext
            guard = _regex_time_limit() if use_regex else nullcontext()
            with guard:
                content_pattern = None if use_regex else f"%{pattern}%"
                rows = self.store.find_memories_by_content(
                    self.namespace, project=project, tags=tags,
                    content_pattern=content_pattern,
                )
                if use_regex:
                    matches = [(r["id"], r["content"]) for r in rows
                               if regex.search(r["content"] or "")]
                else:
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
        except TimeoutError:
            return {"error": ("regex scan timed out (catastrophic "
                              "backtracking?); simplify the pattern"),
                    "matched": 0, "affected": 0, "changes": [],
                    "dry_run": dry_run}

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
        rows = self.store.get_briefing_memories(
            self.namespace, project=project, limit=30)

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
        return self.store.get_digest_memories(
            self.namespace, days=days, project=project, limit=100)

    def map(self) -> dict:
        """Compact topic index: count by project/subcategory."""
        rows = self.store.get_project_map(self.namespace)
        result = {}
        for r in rows:
            project, sub, n = r["project"], r["subcategory"], r["n"]
            if project not in result:
                result[project] = {"total": 0, "subcategories": {}}
            result[project]["total"] += n
            if sub:
                result[project]["subcategories"][sub] = n
        return result

    def reindex_archived(self, batch_size: int = 128) -> dict:
        """Backfill the tier-2 archived vector index (v10.7.0).

        Embeds every archived memory that has no vector in embed_vec_arch yet
        and stores it in the archived index, so tier-2 recall (expand_merged)
        can reach originals that consolidation archived and stripped of their
        primary-index vectors. Idempotent: re-running only embeds what is still
        missing.
        """
        if not hasattr(self.store, "archived_missing_embeddings"):
            return {"error": "tier-2 index only supported on SQLite-based stores"}
        missing = self.store.archived_missing_embeddings(namespace=self.namespace)
        # Legacy pass (v10.24.0): rows whose tier-2 meta carries a pre-v10.6
        # truncated hash were also embedded under a pre-canonical formula, so
        # they get re-embedded, not just re-hashed.
        legacy = (self.store.archived_legacy_hash_rows(namespace=self.namespace)
                  if hasattr(self.store, "archived_legacy_hash_rows") else [])
        pending = len(missing)
        done = legacy_done = 0
        for batch_src, is_legacy in ((missing, False), (legacy, True)):
            for i in range(0, len(batch_src), batch_size):
                batch = batch_src[i:i + batch_size]
                texts = [
                    prep_memory_text(m.project, m.content, m.tags or "",
                                     mem_type=m.type, layer=m.layer)
                    for m in batch
                ]
                vecs = embed(texts, prefix="passage")
                for m, text, v in zip(batch, texts, vecs):
                    if v:
                        self.store._store_archived_embedding(m.id, v, embed_text_hash(text))
                        if is_legacy:
                            legacy_done += 1
                        else:
                            done += 1
        return {
            "archived_pending_before": pending,
            "embedded_now": done,
            "legacy_pending_before": len(legacy),
            "legacy_reembedded": legacy_done,
            "archived_index_size": self.store.archived_embed_count(),
        }

    def embed_fill(self, limit: Optional[int] = None, dry_run: bool = False) -> dict:
        """Backfill vectors for active memories that have none.

        The store path deliberately keeps a memory when embedding fails
        (availability over consistency), which used to be a one-way door:
        reindex-archived only covers the tier-2 index and doctor pointed at
        this command before it existed. Missing vectors here are exactly the
        rows embed_status reports as `missing`.
        """
        rows = self.store.get_unembedded_memories(
            self.namespace, limit=limit)
        filled = failed = 0
        for r in rows:
            if dry_run:
                continue
            text = prep_memory_text(
                r["project"], r["content"], r["tags"] or "",
                mem_type=r["type"] or "", layer=r["layer"] or "",
            )
            vecs = embed([text], prefix="passage")
            if vecs and vecs[0]:
                self.store._store_embedding(
                    r["id"], vecs[0], text_hash=embed_text_hash(text))
                filled += 1
            else:
                failed += 1
        return {"missing": len(rows), "filled": filled, "failed": failed,
                "dry_run": dry_run}

    def embed_status(self) -> dict:
        """Embedding coverage report."""
        return self.store.get_embed_coverage(self.namespace)

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
        report = self.store.health_check(self.namespace, migrate=migrate)

        # --- Embedding coverage ---
        # Guarded: embed_status queries embed_meta columns, and schema drift
        # there (the exact condition doctor exists to diagnose) must surface
        # as a reported issue, not kill the whole health check.
        try:
            cov = self.embed_status()
        except Exception as e:
            report["issues"].append(f"Embedding coverage check failed: {e}")
            cov = {"missing": 0, "embedded": "unknown"}
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

        # --- Content/vector coherence (v10.19.0) ---
        # _store_embedding records text_hash "so staleness is detectable
        # later"; this is the later. Re-deriving each active memory's
        # embed-text hash from current content catches silent divergence:
        # content mutated without a re-embed (direct SQL writes bypassing
        # the API, or a write-path bug), which retrieval cannot see and
        # which no other check covers. Near-total mismatch means the
        # embed-text FORMAT changed between versions, not row corruption,
        # and is reported as such.
        try:
            coherence = self.store.get_coherence_mismatches(self.namespace)
            checkable = coherence["checkable"]
            mismatched = coherence["mismatched_ids"]
            if not mismatched:
                report["checks"].append(
                    f"Content/vector coherence: {checkable} active memories verified")
            elif checkable >= 20 and len(mismatched) > checkable * 0.9:
                # Ratio semantics need a population; on tiny stores a single
                # tampered row is 100% and must be treated as tampering.
                report["issues"].append(
                    f"Content/vector coherence: {len(mismatched)}/{checkable} "
                    "mismatched. Near-total mismatch means the embed-text "
                    "format changed across versions; re-embed the store "
                    "rather than treating rows as corrupted")
            else:
                report["issues"].append(
                    f"Content/vector coherence: {len(mismatched)} active "
                    f"memories whose content no longer matches their stored "
                    f"embedding (ids {mismatched[:10]}"
                    f"{'...' if len(mismatched) > 10 else ''}): content was "
                    "changed without a re-embed")
                if migrate:
                    repaired = self.store.reembed_mismatched(
                        self.namespace, mismatched,
                        embed_fn=embed, text_hash_fn=embed_text_hash,
                        prep_fn=prep_memory_text)
                    report["migrations_applied"].append(
                        f"re-embedded {repaired}/{len(mismatched)} "
                        "hash-mismatched memories")
        except Exception as e:
            report["issues"].append(f"Coherence check failed: {e}")

        # --- Vector model provenance ---
        # A different-dims model fails loudly at insert; a same-dims swap is
        # the dangerous case: old and new vectors coexist and every KNN
        # comparison is silently meaningless. NULL model rows predate
        # provenance tracking and are reported as a check, not an issue.
        try:
            from .constants import FASTEMBED_MODEL
            rows = self.store.get_vector_provenance()
            foreign = [(r["model"], r["count"]) for r in rows
                       if r["model"] and r["model"] != FASTEMBED_MODEL]
            untracked = sum(r["count"] for r in rows if not r["model"])
            if foreign:
                report["issues"].append(
                    "Mixed vector provenance: "
                    + ", ".join(f"{n} vectors from '{m}'" for m, n in foreign)
                    + f" alongside current model '{FASTEMBED_MODEL}'. "
                    "Same-index vectors from different models make KNN "
                    "results meaningless; re-embed with `mnemos embed-fill`."
                )
            elif untracked:
                report["checks"].append(
                    f"Vector provenance: {untracked} vectors predate model "
                    "tracking (assumed current model)"
                )
            else:
                report["checks"].append("Vector provenance: consistent")
        except Exception as e:
            report["issues"].append(f"Provenance check failed: {e}")

        # --- Archive-side embedding lifecycle (v10.24.0) ---
        # The 2026-07-07 forensic audit found three archive-side drifts the
        # active-side checks never see: embed_meta tables created before the
        # localtime DDL keep a UTC embedded_at default (synchronous embeds
        # then look hours older than their memory's creation), hard deletes
        # used to leave tier-2 rows behind, and consolidation used to archive
        # memories without moving their vectors into the tier-2 index.
        try:
            self.store.check_archive_lifecycle(self.namespace, migrate, report)
        except Exception as e:
            report["issues"].append(f"Archive-lifecycle check failed: {e}")

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
