#!/usr/bin/env python3
"""
Session → CML compressor for LongMemEval benchmarks.

Compresses a LongMemEval chat session (list of user/assistant turns) into
dense Compressed Memory Language (CML) facts using Claude Haiku via the
DO Gradient API. Results are SHA256-cached in a small SQLite DB so repeat
runs and ablations are effectively free after the first pass.

Intended flow: retrieve top-K sessions normally (bi-encoder is OOD on CML
so keep retrieval on natural language), then CML-compress the top-K before
feeding to the answerer. Tests whether context compression narrows the
R@5→QA gap.

Usage (as a module):
    from cml_session import compress_session
    cml = compress_session(session_turns)
"""

import hashlib
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, "/root/scripts")
from llm_utils import haiku_chat

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cml_session_cache.db")

CML_SESSION_SYSTEM = """You extract user-relevant facts from a chat session into dense CML (Compressed Memory Language).

CML NOTATION:
Prefixes: D:(decision) C:(contact) F:(fact/preference) L:(learning/event) P:(procedure) W:(warning)
Symbols: →(uses/chose/is) ∵(because) >(preferred over) @(location/when) △(was/changed from) ∅(declined/rejected) ✓(confirmed) ✗(rejected) ;(separator)

WHAT TO CAPTURE:
1. Everything the USER said about themselves: facts, preferences, opinions, plans, events, relationships, purchases, decisions.
2. Everything the user AGREED TO or CONFIRMED from the assistant's reply.
3. Specific details from user statements: names, numbers, dates, places, amounts, durations.
4. The session's date/time context if stated.

WHAT TO DROP:
- Generic assistant advice, tutorials, code blocks, recipes, explanations the user didn't act on.
- Pleasantries, filler, transitions.
- Meta-comments about the conversation itself.

FORMATTING:
- One prefix-tagged fact per ; chain.
- Pack related facts onto one line.
- Preserve all specifics verbatim (names, numbers, dates).
- Target: 10-25% of original token count. Be ruthless but don't drop facts.

OUTPUT: only the compressed CML facts. No headers, fences, explanations, or preambles."""


def _ensure_cache():
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cml_cache (
            session_hash TEXT PRIMARY KEY,
            cml_text TEXT NOT NULL,
            original_len INTEGER,
            cml_len INTEGER,
            created_at REAL
        )
    """)
    conn.commit()
    return conn


def _hash_session(session):
    """Deterministic hash of the session content."""
    blob = json.dumps(
        [[t.get("role", ""), t.get("content", "")] for t in session],
        separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _serialize_session(session):
    """Render session as a plain transcript for the Haiku prompt."""
    lines = []
    for turn in session:
        role = "USER" if turn.get("role") == "user" else "ASSISTANT"
        lines.append(f"{role}: {turn.get('content', '').strip()}")
    return "\n\n".join(lines)


def compress_session(session, cache=True, max_tokens=600, temperature=0.2):
    """Compress one LongMemEval session to CML. Returns the CML string.

    Uses a SHA256-keyed SQLite cache so repeat runs skip the API call.
    """
    conn = _ensure_cache() if cache else None
    h = _hash_session(session)
    if conn is not None:
        row = conn.execute("SELECT cml_text FROM cml_cache WHERE session_hash = ?", (h,)).fetchone()
        if row:
            conn.close()
            return row[0]

    transcript = _serialize_session(session)
    messages = [
        {"role": "system", "content": CML_SESSION_SYSTEM},
        {"role": "user", "content": transcript},
    ]
    cml = haiku_chat(messages, max_tokens=max_tokens, temperature=temperature)
    if cml is None:
        if conn is not None:
            conn.close()
        return None

    if conn is not None:
        conn.execute(
            "INSERT OR REPLACE INTO cml_cache (session_hash, cml_text, original_len, cml_len, created_at) VALUES (?, ?, ?, ?, ?)",
            (h, cml, len(transcript), len(cml), time.time()),
        )
        conn.commit()
        conn.close()
    return cml


def cache_stats():
    conn = _ensure_cache()
    row = conn.execute("SELECT COUNT(*), AVG(cml_len * 1.0 / original_len), AVG(original_len), AVG(cml_len) FROM cml_cache").fetchone()
    conn.close()
    if not row or row[0] == 0:
        return {"count": 0}
    return {
        "count": row[0],
        "avg_compression_ratio": row[1],
        "avg_original_len": row[2],
        "avg_cml_len": row[3],
    }


if __name__ == "__main__":
    # Smoke test on the first LongMemEval session.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=0, help="question index in the dataset")
    p.add_argument("--session", type=int, default=0, help="session index within that question's haystack")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()

    if args.stats:
        print(json.dumps(cache_stats(), indent=2))
        sys.exit(0)

    with open("/root/work/benchmarks/longmemeval_s.json") as f:
        data = json.load(f)
    entry = data[args.index]
    sess = entry["haystack_sessions"][args.session]
    print(f"Question: {entry['question']}")
    print(f"Answer: {entry['answer']}")
    print(f"Session turns: {len(sess)}")
    original = _serialize_session(sess)
    print(f"Original length: {len(original)} chars")
    print()
    cml = compress_session(sess)
    print(f"--- CML ({len(cml)} chars, {len(cml)*100/len(original):.1f}%) ---")
    print(cml)
