#!/usr/bin/env python3
"""
LongMemEval QA with consolidation (Option B).

Simulates the dedup/merge phase of Mnemos's Nyx consolidation pipeline on
each question's haystack before answering. Tests the hypothesis that
consolidation closes the multi-session aggregation gap that per-session
CML compression alone cannot.

Pipeline per question:
  1. For each of ~40 haystack sessions: compress to CML (cached).
  2. Parse each CML into individual facts: (prefix, subject, body, session_id, date).
  3. Group facts by (prefix, subject). For groups with >= 2 facts, merge via
     Qwen3 into one consolidated fact that preserves all specifics
     (counts, dates, names, amounts).
  4. Retrieve top-K consolidated facts via hybrid (FTS5 + embed + RRF).
  5. Feed top-K facts to the answerer; judge with the configured judge.
  6. Report per-type accuracy including abstention.

This is NOT the full Nyx cycle (no weave, no contradict, no synthesis),
it is the dedup/merge phase in isolation, which is the one we hypothesise
matters most for multi-session LongMemEval questions.

Usage:
    python longmemeval_consolidated.py [--limit N] [--k 10]
                                       [--answerer gpt4o|sonnet|opus]
                                       [--judge gpt4o|opus]
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/root/scripts")

import numpy as np

import longmemeval_bench as _bench
_bench.TEMP_DB = os.environ.get("LME_TEMP_DB", "/tmp/longmemeval_consolidated.db")
from longmemeval_bench import (
    DATA_PATH, fastembed_embed, create_temp_db, reset_db,
    search_fts, search_vec,
)
from longmemeval_bench import SOURCE_KEY
from mnemos.rerank import rrf_merge
from mnemos.storage.sqlite_store import _serialize_vec
from mnemos.embed import text_hash
from mnemos.constants import FASTEMBED_MODEL

from cml_session import compress_session
from longmemeval_qa import (
    ANSWERERS, JUDGES, ANSWER_PROMPT, judge_verdict,
)
from llm_utils import qwen3_chat

TEMP_DB = _bench.TEMP_DB
CML_PREFIXES = ("D", "C", "F", "L", "P", "W")
CML_FACT_SPLIT_RE = re.compile(r";\s*")

MERGE_PROMPT = """Merge these {n} related facts about '{subject}' (type {prefix}) from different sessions into one consolidated CML fact.

CRITICAL:
- Preserve ALL specifics verbatim: names, numbers, dates, amounts, counts.
- If items are distinct (e.g. separate purchases), list them all separated by commas.
- If one later session updates or corrects earlier ones, keep the MOST RECENT value and note the update.
- Include the aggregate count when the items are countable ("3 items: X, Y, Z").
- Preserve the session date(s) if the order matters for the facts.

FACTS:
{items}

Output ONE CML line starting with {prefix}:{subject}→ and nothing else. No fencing, no explanation."""


def parse_cml_facts(cml_text, session_id, session_date):
    """Split a CML string into individual fact tuples."""
    facts = []
    if not cml_text:
        return facts
    for raw in CML_FACT_SPLIT_RE.split(cml_text):
        fact = raw.strip()
        if not fact or len(fact) < 3 or fact[1] != ":":
            continue
        prefix = fact[0]
        if prefix not in CML_PREFIXES:
            continue
        body = fact[2:].strip()
        # Extract subject (part before →), lowercased for grouping.
        if "→" in body:
            subject, rest = body.split("→", 1)
            subject = subject.strip()
            rest = rest.strip()
        else:
            subject = None
            rest = body
        facts.append({
            "prefix": prefix,
            "subject": subject,
            "body": rest,
            "session_id": session_id,
            "date": session_date,
            "raw": fact,
        })
    return facts


def _fact_to_text(fact):
    """Render a fact back to CML string form."""
    if fact["subject"]:
        return f"{fact['prefix']}:{fact['subject']}→{fact['body']}"
    return f"{fact['prefix']}:{fact['body']}"


def merge_group(prefix, subject, group):
    """Ask Qwen3 to merge a group of facts with the same (prefix, subject)."""
    items_text = "\n".join(
        f"- {f['body']} [session {f['session_id']} @ {f['date']}]" for f in group
    )
    prompt = MERGE_PROMPT.format(
        n=len(group), subject=subject, prefix=prefix, items=items_text
    )
    merged = qwen3_chat([{"role": "user", "content": prompt}], max_tokens=300, temperature=0.1)
    if not merged:
        return None
    merged = merged.strip()
    # Normalise: ensure it starts with the expected prefix:subject→
    expected = f"{prefix}:{subject}→"
    if not merged.startswith(expected):
        # Try salvaging: look for a line that starts with the expected prefix
        for line in merged.splitlines():
            line = line.strip()
            if line.startswith(expected):
                merged = line
                break
        else:
            # Couldn't salvage , fall back to None so caller keeps originals
            return None
    return merged


def consolidate_haystack(entry):
    """Build consolidated fact corpus for one LongMemEval entry.

    Returns a list of dicts, each representing one fact (consolidated or singleton).
    """
    all_facts = []
    for sid, sess, date in zip(
        entry["haystack_session_ids"], entry["haystack_sessions"], entry["haystack_dates"]
    ):
        cml = compress_session(sess)
        if cml:
            all_facts.extend(parse_cml_facts(cml, sid, date))

    # Group by (prefix, lowercase(subject))
    groups = defaultdict(list)
    no_subject = []
    for f in all_facts:
        if f["subject"]:
            key = (f["prefix"], f["subject"].lower())
            groups[key].append(f)
        else:
            no_subject.append(f)

    consolidated = []
    for (prefix, subject_lc), group in groups.items():
        if len(group) == 1:
            consolidated.append(group[0])
            continue
        # Use the most-capitalised form of the subject for readability
        subject = max((f["subject"] for f in group), key=lambda s: sum(1 for c in s if c.isupper()))
        merged_text = merge_group(prefix, subject, group)
        if merged_text:
            # Extract body from merged text
            body = merged_text[len(prefix) + 1 + len(subject) + 1:].strip()
            session_ids = sorted({f["session_id"] for f in group})
            dates = sorted({f["date"] for f in group})
            consolidated.append({
                "prefix": prefix,
                "subject": subject,
                "body": body,
                "session_id": ",".join(session_ids),
                "date": ",".join(dates),
                "raw": merged_text,
                "merged_from": len(group),
            })
        else:
            consolidated.extend(group)

    consolidated.extend(no_subject)
    return consolidated, len(all_facts)


def build_fact_db(conn, facts):
    """Ingest consolidated facts into the temp DB for hybrid retrieval."""
    # Clear
    reset_db(conn)
    # Insert
    for fact in facts:
        text = _fact_to_text(fact)
        conn.execute(
            "INSERT INTO docs (session_id, content) VALUES (?, ?)",
            (fact["session_id"], text),
        )
    conn.commit()
    # Embed
    rows = conn.execute("SELECT id, content FROM docs ORDER BY id").fetchall()
    texts = [r["content"][:2000] for r in rows]
    embeddings = fastembed_embed(texts, prefix="passage")
    for row, emb in zip(rows, embeddings):
        cur = conn.execute(
            "INSERT INTO embed_vec(embedding) VALUES (?)",
            (_serialize_vec(emb),),
        )
        vec_id = cur.lastrowid
        thash = text_hash(str(row["id"]))
        conn.execute(
            "INSERT INTO embed_meta (id, source_db, source_id, text_hash, model) VALUES (?, ?, ?, ?, ?)",
            (vec_id, SOURCE_KEY, row["id"], thash, FASTEMBED_MODEL),
        )
    conn.commit()


def retrieve_facts(conn, question, k):
    """Hybrid retrieval over consolidated facts. Returns list of fact dicts."""
    q_embs = fastembed_embed([question], prefix="search_query")
    q_emb = q_embs[0] if q_embs else None
    fts_ids = search_fts(conn, question, limit=50)
    vec_ids = search_vec(conn, q_emb, limit=50) if q_emb is not None else []
    if fts_ids and vec_ids:
        ranked = rrf_merge(fts_ids, vec_ids)
    elif vec_ids:
        ranked = vec_ids
    else:
        ranked = fts_ids
    ranked = ranked[:k]
    if not ranked:
        return []
    ph = ",".join("?" for _ in ranked)
    rows = conn.execute(
        f"SELECT id, session_id, content FROM docs WHERE id IN ({ph})", ranked
    ).fetchall()
    id_to_row = {r["id"]: r for r in rows}
    return [id_to_row[i] for i in ranked if i in id_to_row]


def build_consolidated_context(retrieved_rows):
    """Format retrieved consolidated facts for the answerer prompt."""
    if not retrieved_rows:
        return "(no facts retrieved)"
    blocks = []
    for i, row in enumerate(retrieved_rows, start=1):
        blocks.append(f"[Fact {i} | from session(s) {row['session_id']}]\n{row['content']}")
    return "\n\n".join(blocks)


def run(args):
    print(f"Loading LongMemEval data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    n_abs = sum(1 for x in data if "_abs" in x["question_id"])
    print(f"Questions: {len(data)} | abstention: {n_abs}")
    print(f"K={args.k} | answerer={args.answerer} | judge={args.judge}")
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
    consolidation_stats = {"total_raw_facts": 0, "total_consolidated_facts": 0, "merges_performed": 0}

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        reference = entry["answer"]
        is_abs = "_abs" in qid
        type_key = "abstention" if is_abs else qtype
        t0 = time.time()

        consolidated, raw_count = consolidate_haystack(entry)
        merges = sum(1 for f in consolidated if f.get("merged_from"))
        consolidation_stats["total_raw_facts"] += raw_count
        consolidation_stats["total_consolidated_facts"] += len(consolidated)
        consolidation_stats["merges_performed"] += merges

        if not consolidated:
            per_q.append({"qid": qid, "type": type_key, "correct": None, "reason": "empty_corpus"})
            continue

        build_fact_db(conn, consolidated)
        retrieved = retrieve_facts(conn, question, k=args.k)
        context = build_consolidated_context(retrieved)
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
            per_q.append({
                "qid": qid, "type": type_key, "correct": None, "reason": "judge_fail",
                "system_answer": sys_answer, "judge_raw": raw,
            })
            continue

        evaluated_total += 1
        type_count[type_key] += 1
        if verdict:
            correct_total += 1
            type_correct[type_key] += 1
        per_q.append({
            "qid": qid, "type": type_key, "correct": verdict,
            "system_answer": sys_answer, "reference": reference,
            "raw_fact_count": raw_count, "consolidated_fact_count": len(consolidated), "merges": merges,
            "retrieved_facts": [r["content"] for r in retrieved],
        })

        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            acc = correct_total / max(1, evaluated_total)
            avg_merge = consolidation_stats["merges_performed"] / (i + 1)
            print(f"  [{i+1:3d}/{len(data)}] acc={acc:.3f} ({correct_total}/{evaluated_total}) | "
                  f"raw={raw_count}->cons={len(consolidated)} | merges/q={avg_merge:.1f} | "
                  f"last={elapsed:.1f}s | {type_key}")

    conn.close()
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    print("\n" + "=" * 72)
    print("LONGMEMEVAL CONSOLIDATED QA RESULTS")
    print(f"K={args.k} | answerer={args.answerer} | judge={args.judge}")
    print(f"Evaluated: {evaluated_total}/{len(data)} | answerer_fails={answerer_fails} | judge_fails={judge_fails}")
    print("=" * 72)

    overall = correct_total / max(1, evaluated_total)
    print(f"\nOverall QA accuracy: {overall:.1%} ({correct_total}/{evaluated_total})")
    print(f"Raw -> consolidated facts ratio: {consolidation_stats['total_consolidated_facts']}/{consolidation_stats['total_raw_facts']} "
          f"({100 * consolidation_stats['total_consolidated_facts'] / max(1, consolidation_stats['total_raw_facts']):.0f}%)")
    print(f"Merges performed total: {consolidation_stats['merges_performed']}")

    print(f"\n{'Type':<28} {'Acc':>7} {'N':>5}")
    print("-" * 42)
    for t in sorted(type_count):
        n = type_count[t]
        a = type_correct[t] / n if n else 0.0
        print(f"{t:<28} {a:>6.1%} {n:>5}")

    out = {
        "pipeline": "consolidated",
        "k": args.k,
        "answerer": args.answerer,
        "judge": args.judge,
        "overall_accuracy": overall,
        "evaluated": evaluated_total,
        "total": len(data),
        "per_type": {t: {"accuracy": type_correct[t] / type_count[t] if type_count[t] else 0.0,
                         "n": type_count[t]} for t in type_count},
        "consolidation_stats": consolidation_stats,
        "answerer_fails": answerer_fails,
        "judge_fails": judge_fails,
        "per_question": per_q,
    }
    out_path = args.out or os.path.join(
        os.path.dirname(__file__),
        f"qa_consolidated_k{args.k}_{args.answerer}_{args.judge}.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--k", type=int, default=10, help="top-K consolidated facts to feed to answerer")
    p.add_argument("--answerer", choices=list(ANSWERERS), default="gpt4o")
    p.add_argument("--judge", choices=list(JUDGES), default="gpt4o")
    p.add_argument("--out", help="output JSON path")
    args = p.parse_args()
    run(args)
