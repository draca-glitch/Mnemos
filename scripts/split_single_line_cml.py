#!/usr/bin/env python3
"""
split_single_line_cml.py - repair single-line CML memories in a Mnemos DB.

WHAT / WHY
  Older cemelify/merge prompts packed a whole memory onto one physical line
  ("F:a; D:b; W:c"). The line-based splitter cannot cut those, so they persist
  as unsplittable single-line blobs. This tool finds active memories that are a
  single physical line packing two or more CML statements and rewrites each as
  one fact per line, so they become atomic and splittable.

HOW (safe by construction)
  - The split is mechanical: stdlib only, NO LLM. It breaks before each
    canonical prefix (F:/D:/C:/L:/P:/W:/R:) that starts a new statement; a ';'
    NOT followed by a prefix stays intra-fact, so a single fact is never
    shredded. (Same logic as mnemos.splitter.explode_cml_chain, inlined here so
    this runs on boxes whose mnemos predates that function.)
  - Loss-guarded: a memory is rewritten only when its separator-free content is
    byte-identical before and after (nothing dropped, added, or reordered);
    otherwise it is skipped and reported.
  - Writes go THROUGH the Mnemos update API (Mnemos.update), which re-embeds and
    re-syncs FTS5 + the vector index. It never issues a raw UPDATE (that would
    desync FTS+vec, per the Mnemos rule).

DEPS
  The installed `mnemos` package (any 10.x) - used only on --apply for the
  re-embedding update path. The scan and split need only the stdlib.

USAGE
  # dry run (default): show every memory that would change, write nothing
  MNEMOS_DB=/path/to/memory.db python scripts/split_single_line_cml.py

  # apply (back up the DB first!)
  sqlite3 /path/to/memory.db ".backup '/path/to/memory.db.bak'"
  MNEMOS_DB=/path/to/memory.db python scripts/split_single_line_cml.py --apply

  # explicit db / throttle
  python scripts/split_single_line_cml.py --db /path/memory.db --min-len 200 --limit 20 --apply
"""
import argparse
import os
import re
import sqlite3
import sys

# Mechanical CML statement boundary: a canonical prefix that is neither glued to
# a preceding alphanumeric (so "F:" inside "PDF:" is not a boundary) nor followed
# by a path separator (so a Windows drive letter "C:\" or a "D:/path" URL is not
# a boundary). The trailing (?![\\/]) is what keeps file paths from false-splitting.
_BOUNDARY = re.compile(r"(?<![A-Za-z0-9])(?=[FDCLPWR]:(?![\\/]))")


def _sep_free(text):
    return re.sub(r"[\s;.]+", "", text or "")


def explode(content):
    """Single physical line of prefix-chained CML -> one statement per line.
    Returns content unchanged if it is empty, already multi-line, resolves to
    fewer than two statements, or the loss guard fails."""
    if not content or "\n" in content:
        return content
    parts = []
    for seg in _BOUNDARY.split(content):
        seg = re.sub(r";+\s*$", "", seg.strip()).strip()
        if seg:
            parts.append(seg)
    if len(parts) < 2:
        return content
    out = "\n".join(parts)
    return out if _sep_free(out) == _sep_free(content) else content


def is_candidate(content, min_len):
    if not content or "\n" in content or len(content) < min_len:
        return False
    return explode(content) != content


def main():
    ap = argparse.ArgumentParser(
        description="Split single-line multi-statement CML memories into one fact per line, "
        "re-embedding through Mnemos."
    )
    ap.add_argument("--db", help="path to memory.db (falls back to MNEMOS_DB env)")
    ap.add_argument("--min-len", type=int, default=0,
                    help="skip single-line memories shorter than this (default 0; the real "
                    "gate is having 2+ CML statements on one line)")
    ap.add_argument("--limit", type=int, default=0, help="cap number processed (0 = all)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    db = args.db or os.environ.get("MNEMOS_DB")
    if not db or not os.path.exists(db):
        sys.exit(f"error: DB not found. Pass --db or set MNEMOS_DB (got: {db!r})")
    # Bind the Mnemos update path to this same DB before mnemos is imported.
    os.environ["MNEMOS_DB"] = db

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, namespace, content FROM memories WHERE status='active'"
        ).fetchall()
    finally:
        conn.close()

    cands = [(mid, ns, c, explode(c)) for mid, ns, c in rows if is_candidate(c, args.min_len)]
    if args.limit:
        cands = cands[:args.limit]

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"scanned {len(rows)} active memories; {len(cands)} single-line CML blobs to split "
          f"(min-len={args.min_len}). Mode: {mode}\n")

    for mid, ns, before, after in cands:
        nlines = after.count("\n") + 1
        print(f"--- #{mid} [ns={ns}] {len(before)}c -> {nlines} lines")
        if not args.apply:
            shown = after if len(after) <= 600 else after[:600] + " ...[truncated]"
            print(shown + "\n")

    if not cands:
        print("nothing to do: no single-line multi-statement CML memories found.")
        return
    if not args.apply:
        print(f"dry run: {len(cands)} would change. Back up the DB, then re-run with --apply.")
        return

    # Apply: re-embed via the Mnemos update path, one instance per namespace.
    from mnemos.core import Mnemos
    instances = {}
    ok = fail = 0
    for mid, ns, before, after in cands:
        m = instances.get(ns)
        if m is None:
            m = instances[ns] = (Mnemos(namespace=ns) if ns else Mnemos())
        res = m.update(mid, content=after)
        if res.get("status") == "updated":
            ok += 1
            warn = "" if res.get("embedded", True) else "  (WARN: re-embed failed, vector stale)"
            print(f"  #{mid} updated{warn}")
        else:
            fail += 1
            print(f"  #{mid} FAILED: {res}")
    print(f"\napplied: {ok} updated, {fail} failed.")


if __name__ == "__main__":
    main()
