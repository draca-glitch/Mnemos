#!/usr/bin/env python3
"""
LoCoMo benchmark runner for Mnemos.

Tests the hybrid retrieval pipeline (FTS5 BM25 + FastEmbed/e5-large vector
+ RRF + optional Jina cross-encoder rerank, optional per-session CML
compression) against the LoCoMo dataset (10 conversations, 19-32 sessions
each, 1,986 QA pairs across 5 categories).

Honest-methodology guardrails:
  * Category 5 questions are adversarial (answer NOT in conversation),
    should be evaluated as abstention and are mathematically undefined
    for R@K. They are EXCLUDED from retrieval recall (the 446 skipped
    questions are reported separately, same convention as LongMemEval
    abstention).
  * Top-K is capped at 10. The smallest conversation has 19 sessions,
    so K=10 is safely below every conversation's session count. This
    avoids the "top-K exceeds session count -> retrieval returns every
    session -> LLM reranker is the actual matcher" bypass that top-50
    runs against 19-32-session conversations produce.
  * Four Mnemos modes are runnable for apples-to-apples with LongMemEval:
      hybrid          BM25 + vector, no rerank, no CML
      hybrid+rerank   + Jina cross-encoder
      hybrid --cml    + per-session CML compression at index time
      hybrid+rerank --cml  canonical Mnemos
    No generative LLM sits in any retrieval path; the cross-encoder is a
    local ONNX discriminative scorer.

Usage:
    python locomo_bench.py                          # default hybrid
    python locomo_bench.py --mode hybrid+rerank
    python locomo_bench.py --mode hybrid --cml
    python locomo_bench.py --mode hybrid+rerank --cml

Dependencies: fastembed, sqlite-vec, numpy, mnemos package on sys.path.
CML modes require `cml_session.py` (in this directory) and whatever LLM
env vars it reads for compression.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from mnemos.query import clean_fts_query  # noqa: E402
from mnemos.constants import RRF_K, BM25_WEIGHTS, FASTEMBED_MODEL, FASTEMBED_DIMS  # noqa: E402
from mnemos.embed import embed as fastembed_embed_raw, text_hash  # noqa: E402
from mnemos.rerank import rerank, rrf_merge  # noqa: E402
from mnemos.storage.sqlite_store import _ensure_vec_db as ensure_vec_db, _serialize_vec  # noqa: E402


DATA_PATH = os.path.join(os.path.dirname(__file__), "locomo10.json")
TEMP_DB = "/tmp/locomo_bench.db"
SOURCE_KEY = "locomo_bench"

# Maximum K we evaluate. The smallest LoCoMo conversation has 19 sessions;
# we stay safely below that so retrieval is doing real work.
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


CATEGORY_LABELS = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal-reasoning",
    4: "open-domain",
    5: "adversarial (unanswerable)",
}


def fastembed_embed(texts, prefix="passage"):
    if prefix == "search_query":
        prefix = "query"
    return fastembed_embed_raw(texts, prefix=prefix)


def store_embeddings(conn, tuples, model=None):
    for source_db, source_id, thash, embedding in tuples:
        cur = conn.execute(
            "INSERT INTO embed_vec(embedding) VALUES (?)",
            (_serialize_vec(embedding),),
        )
        vec_id = cur.lastrowid
        conn.execute(
            "INSERT INTO embed_meta (id, source_db, source_id, text_hash, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (vec_id, source_db, source_id, thash, model),
        )


def vec_search(conn, embedding, source_db, limit=20):
    rows = conn.execute(
        """SELECT em.source_db, em.source_id, ev.distance
           FROM embed_vec ev
           JOIN embed_meta em ON em.id = ev.rowid
           WHERE em.source_db = ? AND ev.embedding MATCH ? AND k = ?
           ORDER BY ev.distance""",
        (source_db, _serialize_vec(embedding), limit),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def create_temp_db():
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)
    conn = ensure_vec_db(TEMP_DB, dims=FASTEMBED_DIMS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("""
        CREATE TABLE docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE docs_fts USING fts5(
            content, session_id,
            content=docs,
            content_rowid=id,
            tokenize='porter unicode61 remove_diacritics 2'
        )
    """)
    conn.execute("""
        CREATE TRIGGER docs_ai AFTER INSERT ON docs BEGIN
            INSERT INTO docs_fts(rowid, content, session_id)
            VALUES (new.id, new.content, new.session_id);
        END
    """)
    conn.commit()
    return conn


def reset_db(conn):
    conn.execute("DELETE FROM docs")
    conn.execute("DELETE FROM docs_fts")
    conn.execute("DELETE FROM embed_meta")
    conn.execute("DELETE FROM embed_vec")
    conn.commit()


def build_corpus(conversation, use_cml=False):
    """Turn a LoCoMo conversation into a corpus of session documents.

    Each session_N in `conversation` becomes one document. The session_id
    is the session number as a string ("1", "2", ...) so it matches the
    number in the QA evidence strings (D{session}:{turn}).
    """
    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )

    corpus = []
    for sk in session_keys:
        session_num = sk.split("_")[1]
        turns = conversation[sk]
        if not turns:
            continue

        if use_cml:
            from cml_session import compress_session
            # LoCoMo turns have (speaker, text, dia_id). cml_session expects
            # role/content. Map: speaker -> role, text -> content. CML compressor
            # extracts user-relevant facts; for LoCoMo's back-and-forth we feed
            # the whole session and let the compressor do its job.
            mapped = [{"role": t.get("speaker", "user"), "content": t.get("text", "")}
                      for t in turns if t.get("text")]
            if not mapped:
                continue
            try:
                text = compress_session(mapped)
            except Exception as e:
                print(f"    ! CML compress failed for session {session_num}: {e}")
                text = " ".join(t.get("text", "") for t in turns)
        else:
            text = " ".join(f"{t.get('speaker', '')}: {t.get('text', '')}" for t in turns)

        if text.strip():
            corpus.append({"session_id": session_num, "text": text})

    return corpus


def ingest_corpus(conn, corpus, batch_embeddings):
    for doc in corpus:
        conn.execute(
            "INSERT INTO docs (session_id, content) VALUES (?, ?)",
            (doc["session_id"], doc["text"]),
        )
    conn.commit()

    rows = conn.execute("SELECT id, session_id FROM docs ORDER BY id").fetchall()
    tuples = []
    for row, emb in zip(rows, batch_embeddings):
        thash = text_hash(row["session_id"])
        tuples.append((SOURCE_KEY, row["id"], thash, emb))
    store_embeddings(conn, tuples, model=FASTEMBED_MODEL)
    conn.commit()


def search_fts(conn, query, limit=50):
    for fts_mode in ("AND", "OR"):
        fts_query = clean_fts_query(query, mode=fts_mode)
        if not fts_query:
            continue
        try:
            rows = conn.execute(f"""
                SELECT d.id, d.session_id,
                       bm25(docs_fts, {BM25_WEIGHTS[0]}, {BM25_WEIGHTS[1]}) AS rank
                FROM docs_fts fts
                JOIN docs d ON d.id = fts.rowid
                WHERE docs_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, limit)).fetchall()
            if rows:
                return [r["id"] for r in rows]
        except sqlite3.OperationalError:
            continue
    return []


def search_vec(conn, query_embedding, limit=50):
    results = vec_search(conn, query_embedding, source_db=SOURCE_KEY, limit=limit)
    return [r[1] for r in results]


def search_hybrid(conn, query, query_embedding, limit=50):
    fts_ids = search_fts(conn, query, limit=limit)
    vec_ids = search_vec(conn, query_embedding, limit=limit) if query_embedding is not None else []
    if fts_ids and vec_ids:
        return rrf_merge(fts_ids, vec_ids)[:limit]
    if vec_ids:
        return vec_ids[:limit]
    return fts_ids[:limit]


def search_hybrid_rerank(conn, query, query_embedding, limit=50, rerank_top=20,
                         rerank_text_cap=2000):
    """Hybrid + Jina cross-encoder rerank.

    LoCoMo sessions are long (~8-9 turns each, often multi-sentence), so we cap
    each candidate to `rerank_text_cap` characters before scoring to keep the
    cross-encoder tractable on CPU. The embedder still sees the full
    first-4000-char session at index time; this cap only affects the pair-
    scoring step, same pattern the standard retrieval literature uses for
    cross-encoder reranking against long documents.
    """
    candidate_ids = search_hybrid(conn, query, query_embedding, limit=rerank_top)
    if not candidate_ids:
        return []
    ph = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT id, content FROM docs WHERE id IN ({ph})", candidate_ids
    ).fetchall()
    id_to_content = {r["id"]: r["content"] for r in rows}
    docs = [{"text": id_to_content.get(cid, "")[:rerank_text_cap], "id": cid}
            for cid in candidate_ids if cid in id_to_content]
    ranked = rerank(query, docs)
    return [r["id"] for r in ranked[:limit]]


def id_to_session(conn, doc_ids):
    if not doc_ids:
        return []
    ph = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"SELECT id, session_id FROM docs WHERE id IN ({ph})", doc_ids
    ).fetchall()
    id_map = {r["id"]: r["session_id"] for r in rows}
    seen = set()
    result = []
    for did in doc_ids:
        sid = id_map.get(did, "")
        if sid and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


# --- Evidence parsing ---

EVIDENCE_RE = re.compile(r"D(\d+):(\d+)")


def parse_evidence(evidence):
    """Parse LoCoMo evidence list -> set of session numbers (as strings).

    Input: ["D1:3", "D2:5"] -> {"1", "2"}
    """
    sessions = set()
    for ev in evidence or []:
        m = EVIDENCE_RE.search(ev)
        if m:
            sessions.add(m.group(1))
    return sessions


# --- Metrics ---

def recall_any_at_k(retrieved, correct, k):
    """Did at least one correct session appear in top K?"""
    top_k = set(retrieved[:k])
    return float(any(c in top_k for c in correct))


def recall_all_at_k(retrieved, correct, k):
    """Did ALL correct sessions appear in top K?"""
    if not correct:
        return 0.0
    top_k = set(retrieved[:k])
    return float(all(c in top_k for c in correct))


def ndcg_at_k(retrieved, correct, k):
    correct_set = set(correct)
    relevances = np.array([1.0 if r in correct_set else 0.0 for r in retrieved[:k]])
    if len(relevances) == 0 or relevances.sum() == 0:
        return 0.0
    dcg = relevances[0]
    if len(relevances) > 1:
        dcg += np.sum(relevances[1:] / np.log2(np.arange(2, len(relevances) + 1)))
    ideal = np.sort(relevances)[::-1]
    idcg = ideal[0]
    if len(ideal) > 1:
        idcg += np.sum(ideal[1:] / np.log2(np.arange(2, len(ideal) + 1)))
    return dcg / idcg if idcg > 0 else 0.0


# --- Main runner ---

def run_benchmark(args):
    print(f"Loading LoCoMo data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    # Summary
    total_qa = sum(len(c["qa"]) for c in data)
    adv_qa = sum(1 for c in data for q in c["qa"] if q.get("category") == 5)
    evaluable_qa = total_qa - adv_qa
    print(f"Conversations: {len(data)}")
    print(f"QA pairs total: {total_qa}  ({adv_qa} adversarial excluded -> {evaluable_qa} evaluable for R@K)")
    print(f"Per-conversation session counts: "
          + ", ".join(str(len([k for k in c["conversation"] if k.startswith("session_") and not k.endswith("_date_time")])) for c in data))
    print(f"Mode: {args.mode}  CML: {args.cml}  Max-K: {MAX_K}")
    print(f"{'='*70}")

    print("Loading FastEmbed model...", end=" ", flush=True)
    fastembed_embed(["warmup"], prefix="search_query")
    print("done")

    conn = create_temp_db()

    recall_any = {k: [] for k in K_VALUES}
    recall_all = {k: [] for k in K_VALUES}
    ndcg = {k: [] for k in K_VALUES}
    cat_recall_any = defaultdict(lambda: {k: [] for k in K_VALUES})
    q_times = []
    failures = []
    adv_counted = 0

    for ci, conv in enumerate(data):
        sid = conv["sample_id"]
        conv_t0 = time.time()

        corpus = build_corpus(conv["conversation"], use_cml=args.cml)
        if not corpus:
            print(f"  [{ci+1}/{len(data)}] {sid}: empty corpus, skipping")
            continue

        reset_db(conn)
        texts = [doc["text"][:4000] for doc in corpus]
        try:
            embeddings = fastembed_embed(texts, prefix="passage")
        except Exception as e:
            print(f"  [{ci+1}/{len(data)}] {sid}: embed failed ({e}), skipping")
            continue
        if not embeddings or len(embeddings) != len(corpus):
            print(f"  [{ci+1}/{len(data)}] {sid}: embed size mismatch, skipping")
            continue
        ingest_corpus(conn, corpus, embeddings)

        n_sess = len(corpus)
        n_q = len(conv["qa"])
        eval_q = 0

        for qa in conv["qa"]:
            cat = qa.get("category", 0)
            if cat == 5:
                adv_counted += 1
                continue

            question = qa["question"]
            gold = parse_evidence(qa.get("evidence"))
            if not gold:
                failures.append((sid, question[:50]))
                continue

            t0 = time.time()
            q_embs = fastembed_embed([question], prefix="search_query")
            q_emb = q_embs[0] if q_embs else None

            if args.mode == "hybrid":
                doc_ids = (search_hybrid(conn, question, q_emb, limit=50)
                           if q_emb is not None else search_fts(conn, question, limit=50))
            elif args.mode == "hybrid+rerank":
                doc_ids = (search_hybrid_rerank(conn, question, q_emb, limit=50)
                           if q_emb is not None else search_fts(conn, question, limit=50))
            elif args.mode == "fts":
                doc_ids = search_fts(conn, question, limit=50)
            elif args.mode == "vec":
                doc_ids = search_vec(conn, q_emb, limit=50) if q_emb is not None else []
            else:
                doc_ids = search_hybrid(conn, question, q_emb, limit=50)

            retrieved = id_to_session(conn, doc_ids)
            q_times.append(time.time() - t0)
            eval_q += 1

            for k in K_VALUES:
                r_any = recall_any_at_k(retrieved, gold, k)
                r_all = recall_all_at_k(retrieved, gold, k)
                recall_any[k].append(r_any)
                recall_all[k].append(r_all)
                ndcg[k].append(ndcg_at_k(retrieved, gold, k))
                cat_recall_any[cat][k].append(r_any)

        conv_elapsed = time.time() - conv_t0
        r5 = np.mean(recall_any[5]) if recall_any[5] else 0
        print(f"  [{ci+1}/{len(data)}] {sid}: sessions={n_sess} qa={n_q} "
              f"eval={eval_q}  running R@5 (any)={r5:.1%}  conv_time={conv_elapsed:.1f}s")

    conn.close()
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    # --- Report ---
    n_eval = len(recall_any[1]) if recall_any[1] else 0
    print(f"\n{'='*70}")
    print("LOCOMO BENCHMARK RESULTS - Mnemos")
    print(f"Mode: {args.mode}  CML: {args.cml}")
    print(f"Questions evaluated: {n_eval}  (adversarial skipped: {adv_counted}; failures: {len(failures)})")
    print(f"{'='*70}\n")

    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for k in K_VALUES:
        r_any = np.mean(recall_any[k]) if recall_any[k] else 0
        print(f"R@{k} (recall_any){'':<4} {r_any:>9.1%}")
    print()
    for k in K_VALUES:
        r_all = np.mean(recall_all[k]) if recall_all[k] else 0
        print(f"R@{k} (recall_all){'':<4} {r_all:>9.1%}")
    print()
    for k in [5, 10]:
        n = np.mean(ndcg[k]) if ndcg[k] else 0
        print(f"NDCG@{k:<14} {n:>9.1%}")

    if q_times:
        print(f"\nAvg query time: {np.mean(q_times)*1000:.1f} ms  (total: {sum(q_times):.0f}s)")

    print(f"\n{'Category':<32} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'N':>5}")
    print("-" * 66)
    for cat in sorted(cat_recall_any.keys()):
        scores = cat_recall_any[cat]
        label = CATEGORY_LABELS.get(cat, f"cat-{cat}")
        n = len(scores[1])
        r1 = np.mean(scores[1]) if scores[1] else 0
        r3 = np.mean(scores[3]) if scores[3] else 0
        r5 = np.mean(scores[5]) if scores[5] else 0
        r10 = np.mean(scores[10]) if scores[10] else 0
        print(f"{label:<32} {r1:>5.1%} {r3:>5.1%} {r5:>5.1%} {r10:>5.1%} {n:>5}")

    # Save
    cml_suffix = "_cml" if args.cml else ""
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"results_locomo_{args.mode}{cml_suffix}.json",
    )
    results = {
        "benchmark": "LoCoMo",
        "dataset_path": DATA_PATH,
        "mode": args.mode,
        "cml": args.cml,
        "max_k": MAX_K,
        "questions_evaluated": n_eval,
        "adversarial_skipped": adv_counted,
        "failures": len(failures),
        "recall_any": {str(k): float(np.mean(recall_any[k])) if recall_any[k] else 0
                       for k in K_VALUES},
        "recall_all": {str(k): float(np.mean(recall_all[k])) if recall_all[k] else 0
                       for k in K_VALUES},
        "ndcg": {str(k): float(np.mean(ndcg[k])) if ndcg[k] else 0
                 for k in K_VALUES},
        "per_category": {
            str(cat): {
                "label": CATEGORY_LABELS.get(cat, f"cat-{cat}"),
                "n": len(scores[1]),
                **{f"recall_any@{k}": float(np.mean(scores[k])) if scores[k] else 0
                   for k in K_VALUES},
            }
            for cat, scores in cat_recall_any.items()
        },
        "avg_query_ms": float(np.mean(q_times) * 1000) if q_times else 0,
        "total_query_s": float(sum(q_times)) if q_times else 0,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo benchmark runner for Mnemos")
    parser.add_argument("--limit", type=int, help="Limit to first N conversations")
    parser.add_argument("--mode", choices=["fts", "vec", "hybrid", "hybrid+rerank"],
                        default="hybrid",
                        help="Search mode (default: hybrid)")
    parser.add_argument("--cml", action="store_true",
                        help="Pre-compress each session into CML before indexing "
                             "(uses cml_session.compress_session, requires LLM env vars)")
    args = parser.parse_args()
    run_benchmark(args)
