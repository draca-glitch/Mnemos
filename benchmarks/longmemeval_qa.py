#!/usr/bin/env python3
"""
LongMemEval end-to-end QA accuracy benchmark for Mnemos v10.

Extends longmemeval_bench.py (retrieval-only) with an answerer + LLM-judge
scoring stage, matching the methodology of the LongMemEval paper:

    retrieve top-K sessions -> answer with LLM -> judge vs reference

Unlike the retrieval benchmark, this includes the 30 abstention questions,
which test whether the answerer correctly refuses when the retrieved context
does not contain the answer (distractors-only).

Defaults:
  - Retrieval: hybrid+rerank, session granularity, K=5
  - Answerer:  Claude Sonnet 4.6 (via DO Gradient)
  - Judge:     Claude Opus 4.6 (strongest model, matches paper convention)

Usage:
    python longmemeval_qa.py [--limit N] [--mode hybrid+rerank]
                             [--k 5] [--answerer sonnet|opus|qwen3]
                             [--judge opus|sonnet] [--out results.json]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, "/root/scripts")

import numpy as np

import longmemeval_bench as _bench
# Use a separate temp DB so we don't clobber a concurrently running retrieval bench.
# LME_TEMP_DB env var allows multiple concurrent QA runs on the same host.
_bench.TEMP_DB = os.environ.get("LME_TEMP_DB", "/tmp/longmemeval_qa.db")
from longmemeval_bench import (
    DATA_PATH, fastembed_embed, create_temp_db, reset_db,
    build_corpus, ingest_corpus,
    search_fts, search_vec, search_hybrid, search_hybrid_rerank,
    id_to_session,
)
TEMP_DB = _bench.TEMP_DB
from llm_utils import sonnet_chat, opus_chat, qwen3_chat, haiku_chat, gpt4o_chat
from cml_session import compress_session

ANSWERERS = {"haiku": haiku_chat, "sonnet": sonnet_chat, "opus": opus_chat,
             "qwen3": qwen3_chat, "gpt4o": gpt4o_chat}
JUDGES = {"sonnet": sonnet_chat, "opus": opus_chat, "gpt4o": gpt4o_chat}

SESSION_CHAR_CAP = 20000
ANSWER_PROMPT = """You are answering a question about the user based on their past conversation sessions. Use ONLY the provided sessions as evidence.

If the provided sessions do NOT contain the information needed to answer, reply exactly: I don't have enough information to answer that.

Today's date: {question_date}

Retrieved sessions:
{sessions_text}

Question: {question}

Answer concisely. Do not speculate beyond what the sessions state."""

JUDGE_GUIDANCE = {
    "_abs": "The reference indicates the system SHOULD decline / state the info is unavailable. CORRECT if the system refuses or says it does not have the information. INCORRECT if the system provides a specific answer.",
    "temporal-reasoning": "Be strict about numerics. Accept numerically equivalent phrasings (e.g. '7 days' and 'one week'). Accept answers the reference explicitly marks acceptable.",
    "multi-session": "Accept factually / numerically equivalent answers. The answer may need to aggregate across sessions.",
    "knowledge-update": "When information was updated across sessions, ONLY the most recent value is correct.",
    "single-session-preference": "Accept if the system's answer captures the user's preference as described in the reference. Minor wording differences are fine.",
}
JUDGE_DEFAULT = "Be strict on factual content, lenient on phrasing. Accept clear semantic equivalents."

JUDGE_PROMPT = """You are grading an automated QA system.

Question: {question}
Reference answer: {reference}
System answer: {system}

Guidance: {guidance}

Reply with exactly CORRECT or INCORRECT on the first line. Nothing else."""


def format_session_text(session, max_chars=SESSION_CHAR_CAP):
    """Render a list of turns as 'User: ...\\nAssistant: ...' text, capped."""
    buf = []
    total = 0
    for turn in session:
        role = "User" if turn["role"] == "user" else "Assistant"
        line = f"{role}: {turn['content'].strip()}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                buf.append(line[:remaining] + "...")
            break
        buf.append(line)
        total += len(line) + 1
    return "\n".join(buf)


def build_answer_context(entry, top_session_ids, k, cml_context=False):
    """Build the context block feeding the answerer: top-K sessions as text.

    When cml_context=True, each session is Haiku-compressed to CML facts
    (cached by SHA256). Falls back to raw session text on compression failure.
    """
    sid_to_session = dict(zip(entry["haystack_session_ids"], entry["haystack_sessions"]))
    sid_to_date = dict(zip(entry["haystack_session_ids"], entry["haystack_dates"]))
    blocks = []
    for i, sid in enumerate(top_session_ids[:k], start=1):
        session = sid_to_session.get(sid)
        if not session:
            continue
        date = sid_to_date.get(sid, "unknown date")
        if cml_context:
            cml = compress_session(session)
            body = cml if cml else format_session_text(session)
            label = "CML" if cml else "Session"
            blocks.append(f"[{label} {i} | {date}]\n{body}")
        else:
            body = format_session_text(session)
            blocks.append(f"[Session {i} | {date}]\n{body}")
    return "\n\n".join(blocks) if blocks else "(no sessions retrieved)"


def retrieve(conn, entry, mode, granularity, k_retrieve=10):
    """Run retrieval for one entry and return ordered session IDs."""
    corpus = build_corpus(entry, granularity=granularity)
    if not corpus:
        return []
    reset_db(conn)
    texts = [doc["text"][:2000] for doc in corpus]
    try:
        embeddings = fastembed_embed(texts, prefix="passage")
    except Exception:
        return []
    if not embeddings or len(embeddings) != len(corpus):
        return []
    ingest_corpus(conn, corpus, embeddings)
    query = entry["question"]
    q_embs = fastembed_embed([query], prefix="search_query")
    q_emb = q_embs[0] if q_embs else None
    if mode == "fts":
        doc_ids = search_fts(conn, query, limit=50)
    elif mode == "vec":
        doc_ids = search_vec(conn, q_emb, limit=50) if q_emb else []
    elif mode == "hybrid":
        doc_ids = search_hybrid(conn, query, q_emb, limit=50) if q_emb else search_fts(conn, query, limit=50)
    else:  # hybrid+rerank
        doc_ids = search_hybrid_rerank(conn, query, q_emb, limit=50) if q_emb else search_fts(conn, query, limit=50)
    return id_to_session(conn, doc_ids)[:k_retrieve]


def judge_verdict(judge_fn, question, reference, system_answer, qtype, is_abstention):
    """Call the judge and parse CORRECT / INCORRECT. Returns (correct: bool, raw: str)."""
    guidance = JUDGE_GUIDANCE["_abs"] if is_abstention else JUDGE_GUIDANCE.get(qtype, JUDGE_DEFAULT)
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, system=system_answer, guidance=guidance)
    raw = judge_fn([{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0)
    if not raw:
        return None, ""
    first = raw.strip().splitlines()[0].upper()
    if "CORRECT" in first and "INCORRECT" not in first:
        return True, raw
    if "INCORRECT" in first:
        return False, raw
    # Fallback: look anywhere in response
    up = raw.upper()
    if "INCORRECT" in up:
        return False, raw
    if "CORRECT" in up:
        return True, raw
    return None, raw


def run(args):
    print(f"Loading LongMemEval data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    print(f"Questions: {len(data)} | abstention: {sum(1 for x in data if '_abs' in x['question_id'])}")
    print(f"Mode: {args.mode} | K={args.k} | answerer={args.answerer} | judge={args.judge}")
    print("=" * 72)

    answer_fn = ANSWERERS[args.answerer]
    judge_fn = JUDGES[args.judge]

    print("Loading FastEmbed model...", end=" ", flush=True)
    fastembed_embed(["warmup"], prefix="search_query")
    print("done")

    conn = create_temp_db()

    correct_total = 0
    evaluated_total = 0
    type_correct = defaultdict(int)
    type_count = defaultdict(int)
    per_q = []
    answerer_fails = 0
    judge_fails = 0

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        reference = entry["answer"]
        is_abs = "_abs" in qid
        type_key = "abstention" if is_abs else qtype
        t0 = time.time()

        top_sids = retrieve(conn, entry, args.mode, args.granularity, k_retrieve=max(args.k, 10))
        context = build_answer_context(entry, top_sids, args.k, cml_context=args.cml_context)
        prompt = ANSWER_PROMPT.format(
            question_date=entry.get("question_date", "unknown"),
            sessions_text=context,
            question=question,
        )
        sys_answer = answer_fn([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.0)
        if sys_answer is None:
            answerer_fails += 1
            per_q.append({"qid": qid, "type": type_key, "correct": None, "reason": "answerer_fail"})
            continue

        verdict, raw = judge_verdict(judge_fn, question, reference, sys_answer, qtype, is_abs)
        if verdict is None:
            judge_fails += 1
            per_q.append({"qid": qid, "type": type_key, "correct": None, "reason": "judge_fail",
                          "system_answer": sys_answer, "judge_raw": raw})
            continue

        evaluated_total += 1
        type_count[type_key] += 1
        if verdict:
            correct_total += 1
            type_correct[type_key] += 1
        per_q.append({
            "qid": qid, "type": type_key, "correct": verdict,
            "system_answer": sys_answer, "reference": reference,
            "retrieved": top_sids[: args.k],
        })
        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            acc = correct_total / max(1, evaluated_total)
            print(f"  [{i+1:3d}/{len(data)}] acc={acc:.3f} ({correct_total}/{evaluated_total}) | last={elapsed:.1f}s | {type_key}")

    conn.close()
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    print("\n" + "=" * 72)
    print("LONGMEMEVAL QA BENCHMARK RESULTS")
    print(f"Mode: {args.mode} | K={args.k} | answerer={args.answerer} | judge={args.judge}")
    print(f"Evaluated: {evaluated_total}/{len(data)} | answerer_fails={answerer_fails} | judge_fails={judge_fails}")
    print("=" * 72)

    overall = correct_total / max(1, evaluated_total)
    print(f"\nOverall QA accuracy: {overall:.1%}  ({correct_total}/{evaluated_total})")

    print(f"\n{'Type':<28} {'Acc':>7} {'N':>5}")
    print("-" * 42)
    for t in sorted(type_count.keys()):
        n = type_count[t]
        a = type_correct[t] / n if n else 0.0
        print(f"{t:<28} {a:>6.1%} {n:>5}")

    out = {
        "mode": args.mode,
        "k": args.k,
        "answerer": args.answerer,
        "judge": args.judge,
        "cml_context": args.cml_context,
        "overall_accuracy": overall,
        "evaluated": evaluated_total,
        "total": len(data),
        "per_type": {t: {"accuracy": type_correct[t] / type_count[t] if type_count[t] else 0.0,
                         "n": type_count[t]} for t in type_count},
        "answerer_fails": answerer_fails,
        "judge_fails": judge_fails,
        "per_question": per_q,
    }
    cml_tag = "_cml" if args.cml_context else ""
    out_path = args.out or os.path.join(
        os.path.dirname(__file__),
        f"qa_results_{args.mode}_k{args.k}_{args.answerer}_{args.judge}{cml_tag}.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--mode", choices=["fts", "vec", "hybrid", "hybrid+rerank"], default="hybrid+rerank")
    p.add_argument("--granularity", choices=["session", "turn"], default="session")
    p.add_argument("--k", type=int, default=5, help="top-K retrieved sessions fed to answerer")
    p.add_argument("--answerer", choices=list(ANSWERERS), default="sonnet")
    p.add_argument("--judge", choices=list(JUDGES), default="opus")
    p.add_argument("--cml-context", action="store_true",
                   help="Compress retrieved sessions to CML (Haiku) before feeding to answerer")
    p.add_argument("--out", help="output JSON path")
    args = p.parse_args()
    run(args)
