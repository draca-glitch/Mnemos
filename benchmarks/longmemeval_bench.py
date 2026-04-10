#!/usr/bin/env python3
"""
LongMemEval benchmark runner for the agent memory system (v10.0).

Tests our hybrid retrieval pipeline (FTS5 BM25 + FastEmbed/e5-large vector
search + RRF fusion + optional Jina cross-encoder reranking) against the
LongMemEval benchmark dataset (500 questions, ~40 sessions each).

Metrics: Recall@K (1,3,5,10), NDCG@K (5,10), per question-type breakdown.
Comparable to MemPalace's 96.6% R@5 (raw) and published baselines.

Usage:
    python longmemeval_bench.py [--limit N] [--mode fts|vec|hybrid|hybrid+rerank]
                                [--granularity session|turn] [--k 5]

Dependencies: fastembed, sqlite-vec, numpy.
"""

import json
import math
import os
import sqlite3
import struct
import sys
import time
import argparse
import numpy as np
from collections import defaultdict

# Make the mnemos package importable from the repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from mnemos.query import clean_fts_query
from mnemos.constants import RRF_K, BM25_WEIGHTS, FASTEMBED_MODEL, FASTEMBED_DIMS
from mnemos.embed import embed as fastembed_embed_raw, text_hash
from mnemos.rerank import rerank, rrf_merge
from mnemos.storage.sqlite_store import _ensure_vec_db as ensure_vec_db, _serialize_vec


def fastembed_embed(texts, prefix="passage"):
    """Adapter that matches the prefix convention used by the benchmark."""
    # Mnemos's embed() uses "passage"/"query"; strip "search_" prefix from old callers
    if prefix == "search_query":
        prefix = "query"
    return fastembed_embed_raw(texts, prefix=prefix)


def store_embeddings(conn, tuples, model=None):
    """Bulk-store embeddings into the embed_vec table."""
    for source_db, source_id, thash, embedding in tuples:
        cur = conn.execute(
            "INSERT INTO embed_vec(embedding) VALUES (?)",
            (_serialize_vec(embedding),),
        )
        vec_id = cur.lastrowid
        conn.execute(
            "INSERT INTO embed_meta (id, source_db, source_id, text_hash, model) VALUES (?, ?, ?, ?, ?)",
            (vec_id, source_db, source_id, thash, model),
        )


def vec_search(conn, embedding, source_db, limit=20):
    """Brute-force vector search filtered by source_db."""
    rows = conn.execute(
        """SELECT em.source_db, em.source_id, ev.distance
           FROM embed_vec ev
           JOIN embed_meta em ON em.id = ev.rowid
           WHERE em.source_db = ? AND ev.embedding MATCH ? AND k = ?
           ORDER BY ev.distance""",
        (source_db, _serialize_vec(embedding), limit),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]

DATA_PATH = os.path.join(os.path.dirname(__file__), "longmemeval_s.json")
TEMP_DB = "/tmp/longmemeval_bench.db"
SOURCE_KEY = "bench"


def create_temp_db():
    """Create a fresh temporary database with FTS5 + vec tables."""
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
            INSERT INTO docs_fts(rowid, content, session_id) VALUES (new.id, new.content, new.session_id);
        END
    """)
    conn.commit()
    return conn


def reset_db(conn):
    """Clear all data for next question (faster than recreating)."""
    conn.execute("DELETE FROM docs")
    conn.execute("DELETE FROM docs_fts")
    conn.execute("DELETE FROM embed_meta")
    conn.execute("DELETE FROM embed_vec")
    conn.commit()


def build_corpus(entry, granularity="session"):
    """Extract documents from haystack_sessions."""
    corpus = []
    for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        if granularity == "session":
            # Concatenate all user turns into one document (standard approach)
            text = " ".join(t["content"] for t in session if t["role"] == "user")
            if text.strip():
                corpus.append({"session_id": sid, "text": text})
        else:  # turn-level
            for i, turn in enumerate(session):
                if turn["role"] == "user" and turn["content"].strip():
                    corpus.append({"session_id": f"{sid}_turn{i}", "text": turn["content"]})
    return corpus


def ingest_corpus(conn, corpus, batch_embeddings):
    """Insert corpus into temp DB with FTS5 + vector embeddings."""
    for doc in corpus:
        conn.execute(
            "INSERT INTO docs (session_id, content) VALUES (?, ?)",
            (doc["session_id"], doc["text"]),
        )
    conn.commit()

    # Store embeddings
    rows = conn.execute("SELECT id, session_id FROM docs ORDER BY id").fetchall()
    embed_tuples = []
    for row, emb in zip(rows, batch_embeddings):
        thash = text_hash(row["session_id"])
        embed_tuples.append((SOURCE_KEY, row["id"], thash, emb))
    store_embeddings(conn, embed_tuples, model=FASTEMBED_MODEL)
    conn.commit()


def search_fts(conn, query, limit=50):
    """FTS5 BM25 search with AND/OR fallback."""
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
    """Vector similarity search."""
    results = vec_search(conn, query_embedding, source_db=SOURCE_KEY, limit=limit)
    return [r[1] for r in results]  # source_id = docs.id


def search_hybrid(conn, query, query_embedding, limit=50):
    """Hybrid FTS + vector search with RRF merge."""
    fts_ids = search_fts(conn, query, limit=limit)
    vec_ids = search_vec(conn, query_embedding, limit=limit)

    if fts_ids and vec_ids:
        return rrf_merge(fts_ids, vec_ids)[:limit]
    elif vec_ids:
        return vec_ids[:limit]
    else:
        return fts_ids[:limit]


def search_hybrid_rerank(conn, query, query_embedding, limit=50, rerank_top=20):
    """Hybrid + Jina cross-encoder reranking."""
    candidate_ids = search_hybrid(conn, query, query_embedding, limit=rerank_top)
    if not candidate_ids:
        return []

    # Fetch content for reranking
    ph = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT id, content FROM docs WHERE id IN ({ph})", candidate_ids
    ).fetchall()
    id_to_content = {r["id"]: r["content"] for r in rows}

    docs = [{"text": id_to_content.get(cid, ""), "id": cid} for cid in candidate_ids if cid in id_to_content]
    ranked = rerank(query, docs)
    return [r["id"] for r in ranked[:limit]]


def id_to_session(conn, doc_ids):
    """Map doc IDs back to session IDs."""
    if not doc_ids:
        return []
    ph = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"SELECT id, session_id FROM docs WHERE id IN ({ph})", doc_ids
    ).fetchall()
    id_map = {r["id"]: r["session_id"] for r in rows}
    # Preserve order, deduplicate session IDs
    seen = set()
    result = []
    for did in doc_ids:
        sid = id_map.get(did, "")
        # Strip turn suffix for turn-level granularity
        base_sid = sid.split("_turn")[0]
        if base_sid and base_sid not in seen:
            seen.add(base_sid)
            result.append(base_sid)
    return result


# --- Metrics ---

def recall_at_k(retrieved, correct, k):
    """Recall@K: did at least one correct session appear in top K?"""
    top_k = set(retrieved[:k])
    return float(any(c in top_k for c in correct))


def ndcg_at_k(retrieved, correct, k):
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    correct_set = set(correct)
    relevances = np.array([1.0 if r in correct_set else 0.0 for r in retrieved[:k]])
    if len(relevances) == 0 or relevances.sum() == 0:
        return 0.0
    # DCG
    dcg = relevances[0]
    if len(relevances) > 1:
        dcg += np.sum(relevances[1:] / np.log2(np.arange(2, len(relevances) + 1)))
    # Ideal DCG
    ideal = np.sort(relevances)[::-1]
    idcg = ideal[0]
    if len(ideal) > 1:
        idcg += np.sum(ideal[1:] / np.log2(np.arange(2, len(ideal) + 1)))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark(args):
    print(f"Loading LongMemEval data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    total = len(data)
    non_abs = [d for d in data if "_abs" not in d.get("question_id", "")]
    print(f"Total questions: {total} ({len(non_abs)} non-abstention)")
    print(f"Mode: {args.mode} | Granularity: {args.granularity}")
    print(f"{'='*70}")

    # Warm up FastEmbed model
    print("Loading FastEmbed model...", end=" ", flush=True)
    fastembed_embed(["warmup"], prefix="search_query")
    print("done")

    # Create temp DB
    conn = create_temp_db()

    # Results tracking
    k_values = [1, 3, 5, 10]
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    type_recall = defaultdict(lambda: {k: [] for k in k_values})
    times = []
    failures = []

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        correct_ids = entry["answer_session_ids"]

        # Skip abstention for retrieval eval
        if "_abs" in qid:
            continue

        t0 = time.time()

        # Build corpus
        corpus = build_corpus(entry, granularity=args.granularity)
        if not corpus:
            failures.append(qid)
            continue

        # Reset DB
        reset_db(conn)

        # Batch embed all documents
        texts = [doc["text"][:2000] for doc in corpus]  # cap length for embedding
        try:
            embeddings = fastembed_embed(texts, prefix="passage")
        except Exception as e:
            failures.append(qid)
            continue

        if not embeddings or len(embeddings) != len(corpus):
            failures.append(qid)
            continue

        # Ingest
        ingest_corpus(conn, corpus, embeddings)

        # Embed query
        query_embs = fastembed_embed([question], prefix="search_query")
        query_emb = query_embs[0] if query_embs else None

        # Retrieve
        if args.mode == "fts":
            doc_ids = search_fts(conn, question, limit=50)
        elif args.mode == "vec":
            doc_ids = search_vec(conn, query_emb, limit=50) if query_emb else []
        elif args.mode == "hybrid":
            doc_ids = search_hybrid(conn, question, query_emb, limit=50) if query_emb else search_fts(conn, question, limit=50)
        elif args.mode == "hybrid+rerank":
            doc_ids = search_hybrid_rerank(conn, question, query_emb, limit=50) if query_emb else search_fts(conn, question, limit=50)
        else:
            doc_ids = search_hybrid(conn, question, query_emb, limit=50)

        # Map to session IDs
        retrieved_sessions = id_to_session(conn, doc_ids)

        elapsed = time.time() - t0
        times.append(elapsed)

        # Score
        for k in k_values:
            r = recall_at_k(retrieved_sessions, correct_ids, k)
            n = ndcg_at_k(retrieved_sessions, correct_ids, k)
            recall_scores[k].append(r)
            ndcg_scores[k].append(n)
            type_recall[qtype][k].append(r)

        # Progress
        if (i + 1) % 25 == 0 or i == 0:
            r5_so_far = np.mean(recall_scores[5]) if recall_scores[5] else 0
            print(f"  [{i+1:3d}/{total}] R@5={r5_so_far:.3f} | last={elapsed:.2f}s | {qtype}")

    conn.close()
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"LONGMEMEVAL BENCHMARK RESULTS - Memory System v10.0")
    print(f"Mode: {args.mode} | Granularity: {args.granularity}")
    print(f"Questions evaluated: {sum(len(v) for v in recall_scores.values()) // len(k_values)}")
    if failures:
        print(f"Failures: {len(failures)}")
    print(f"{'='*70}")

    print(f"\n{'Metric':<15} {'Value':>10}")
    print(f"{'-'*30}")
    for k in k_values:
        r = np.mean(recall_scores[k]) if recall_scores[k] else 0
        print(f"Recall@{k:<8} {r:>9.1%}")
    print()
    for k in [5, 10]:
        n = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0
        print(f"NDCG@{k:<9} {n:>9.1%}")

    print(f"\nAvg time/question: {np.mean(times):.2f}s (total: {sum(times):.0f}s)")

    # Per-type breakdown
    print(f"\n{'Question Type':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'N':>4}")
    print(f"{'-'*58}")
    for qtype in sorted(type_recall.keys()):
        scores = type_recall[qtype]
        n = len(scores[1])
        r1 = np.mean(scores[1]) if scores[1] else 0
        r3 = np.mean(scores[3]) if scores[3] else 0
        r5 = np.mean(scores[5]) if scores[5] else 0
        r10 = np.mean(scores[10]) if scores[10] else 0
        print(f"{qtype:<30} {r1:>5.1%} {r3:>5.1%} {r5:>5.1%} {r10:>5.1%} {n:>4}")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON (R@5):")
    r5 = np.mean(recall_scores[5]) if recall_scores[5] else 0
    print(f"  This system ({args.mode}):  {r5:.1%}")
    print(f"  MemPalace raw ChromaDB:    96.6%")
    print(f"  MemPalace hybrid v4:       98.4% (held-out)")
    print(f"  BM25 baseline (paper):     ~85%")
    print(f"  Dense retrieval (paper):   ~90%")

    # Save results
    results = {
        "mode": args.mode,
        "granularity": args.granularity,
        "recall": {str(k): float(np.mean(recall_scores[k])) for k in k_values},
        "ndcg": {str(k): float(np.mean(ndcg_scores[k])) for k in [5, 10]},
        "per_type": {
            qtype: {str(k): float(np.mean(scores[k])) for k in k_values}
            for qtype, scores in type_recall.items()
        },
        "avg_time": float(np.mean(times)),
        "total_time": float(sum(times)),
        "questions_evaluated": sum(len(v) for v in recall_scores.values()) // len(k_values),
        "failures": len(failures),
    }
    out_path = os.path.join(os.path.dirname(__file__), f"results_{args.mode}_{args.granularity}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for memory system v10")
    parser.add_argument("--limit", type=int, help="Limit to first N questions (for quick testing)")
    parser.add_argument("--mode", choices=["fts", "vec", "hybrid", "hybrid+rerank"], default="hybrid",
                        help="Search mode (default: hybrid)")
    parser.add_argument("--granularity", choices=["session", "turn"], default="session",
                        help="Document granularity (default: session)")
    args = parser.parse_args()
    run_benchmark(args)
