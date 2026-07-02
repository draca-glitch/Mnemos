#!/usr/bin/env python3
"""One-time migration of a Mnemos store to English-primary content.

Finds active, non-locked memories whose prose is not English (using the same
heuristic the NLI decision layer routes by, mnemos.nli.is_english) and
rewrites them in English via the configured consolidation LLM. Verbatim
quotes stay in the original language; terms of art, legal identifiers, and
institution names stay untranslated when the original word is the canonical
retrieval key. Locked memories (consolidation_lock=1) are skipped: they are
document-shaped and must not be machine-rewritten.

Safety rails:
  - line-count guard: a translation that changes the line structure is
    retried once, then the original is kept
  - per-line digit guard: any line whose digit sequences differ from the
    original keeps the original line (exact values must survive verbatim)
  - dry-run mode translates a sample and writes a report, touching nothing

Writes go through the package API (content + re-embed + text hash in one
transaction; FTS stays in sync via triggers). BACK UP THE DB FIRST:
    sqlite3 "$MNEMOS_DB" ".backup '$MNEMOS_DB.bak-pre-english'"

Config (env): MNEMOS_DB, MNEMOS_NAMESPACE, and the standard MNEMOS_LLM_*
settings (API URL, key, model). MNEMOS_LLM_MODEL_TRANSLATE pins a stronger
model for this job only; translation quality becomes the store, so use the
strongest model you are willing to pay for.

Usage:
    python scripts/translate_store_english.py --dry-run [--sample 0.12]
    python scripts/translate_store_english.py --execute
"""

import argparse
import json
import os
import re
import sys

from mnemos import nli
from mnemos.consolidation.llm import chat, is_configured
from mnemos.embed import embed, prep_memory_text, text_hash
from mnemos.storage.sqlite_store import SQLiteStore

PROMPT = """Translate this memory record to English.

Rules:
- Preserve the line structure exactly: same number of lines, same order.
- Do NOT translate or alter: CML prefixes (F:, D:, C:, L:, P:, W:, R:), symbols, names of people/products/places, file paths, code, commands, numbers, numeric dates and timestamps, version strings, technical identifiers, tags.
- DO translate month and weekday names and label words.
- Direct quotes (text inside quotation marks of any kind) stay VERBATIM in the original language.
- Terms of art, legal identifiers, and institution names stay in the original language when they are the canonical term (optionally add a one-time English gloss in parentheses).
- Translate ONLY the natural-language prose. Text already in English stays untouched.
- Output ONLY the translated record, nothing else.

Record:
{text}"""


def digit_seqs(text):
    return sorted(re.findall(r"\d+", text))


def apply_digit_guard(orig, trans):
    out, guarded = [], 0
    for o, t in zip(orig.split("\n"), trans.split("\n")):
        if digit_seqs(o) != digit_seqs(t):
            out.append(o)
            guarded += 1
        else:
            out.append(t)
    return "\n".join(out), guarded


def translate(content):
    """Returns (translation, flags). Falls back to None after two
    line-structure failures."""
    flags = []
    for attempt in (1, 2):
        result = chat(
            [{"role": "user", "content": PROMPT.format(text=content)}],
            max_tokens=8000, temperature=0.0, phase="TRANSLATE")
        if result is None:
            flags.append(f"llm-error-{attempt}")
            continue
        result = result.strip()
        if len(result.split("\n")) == len(content.split("\n")):
            return result, flags
        flags.append(f"line-mismatch-{attempt}")
    return None, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="write changes")
    ap.add_argument("--dry-run", action="store_true",
                    help="translate a sample, report, write nothing")
    ap.add_argument("--sample", type=float, default=0.12,
                    help="dry-run sample fraction (default 0.12)")
    args = ap.parse_args()
    if args.execute == args.dry_run:
        ap.error("pass exactly one of --execute / --dry-run")
    if not is_configured():
        sys.exit("no LLM configured; set MNEMOS_LLM_API_URL / _API_KEY / _MODEL")

    db = os.environ.get("MNEMOS_DB")
    if not db:
        sys.exit("MNEMOS_DB not set")
    namespace = os.environ.get("MNEMOS_NAMESPACE", "default")
    store = SQLiteStore(db_path=db, namespace=namespace)
    conn = store._get_conn()

    rows = [dict(r) for r in conn.execute(
        "SELECT id, project, content, tags, type, layer FROM memories "
        "WHERE status='active' AND namespace=? AND consolidation_lock=0",
        (namespace,)).fetchall()]
    candidates = [r for r in rows if not nli.is_english(r["content"])]
    print(f"active non-locked: {len(rows)}; non-English candidates: "
          f"{len(candidates)}", flush=True)

    if args.dry_run:
        step = max(1, round(1 / args.sample))
        candidates = candidates[::step]
        print(f"dry-run sample: {len(candidates)} rows", flush=True)

    stats = {"updated": 0, "fallback": 0, "unchanged": 0, "guarded_lines": 0}
    report = []
    for row in candidates:
        trans, flags = translate(row["content"])
        if trans is None:
            stats["fallback"] += 1
            report.append({"id": row["id"], "action": "fallback", "flags": flags})
            continue
        final, guarded = apply_digit_guard(row["content"], trans)
        stats["guarded_lines"] += guarded
        if final == row["content"]:
            stats["unchanged"] += 1
            report.append({"id": row["id"], "action": "unchanged"})
            continue
        if args.dry_run:
            stats["updated"] += 1
            report.append({"id": row["id"], "action": "would-update",
                           "before": row["content"], "after": final})
            continue
        prepped = prep_memory_text(row["project"], final, row["tags"] or "",
                                   mem_type=row["type"] or "",
                                   layer=row["layer"] or "")
        vecs = embed(prepped)
        if not vecs:
            report.append({"id": row["id"], "action": "skipped-embed-failed"})
            continue
        store.update_memory(row["id"], {"content": final},
                            embedding=vecs[0], text_hash=text_hash(prepped))
        stats["updated"] += 1
        report.append({"id": row["id"], "action": "updated",
                       "guarded_lines": guarded, "flags": flags})

    out = "translate_dryrun_report.json" if args.dry_run else "translate_report.json"
    json.dump({"stats": stats, "rows": report}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print(f"stats: {stats}; report: {out}", flush=True)


if __name__ == "__main__":
    main()
