#!/usr/bin/env python3
"""
mnemos_sortkit: one-off Opus-4.8-routed lossless topic-sort of oversized memories.

The Opus agent supplies ONLY a routing (a topic id per CML block). This kit does
all content handling mechanically, so a fact is never paraphrased, dropped, or
duplicated. Losslessness is gated by splitter.split_preserves_all_lines.

Subcommands:
  dump <id>            Print JSON: memory metadata + numbered CML blocks. The
                       router agent reads this to decide the topic grouping.
  place <id> <af>      Apply an assignment file (JSON: {"assign":[int per block],
                       "labels":{"0":"medication",...}}) via splitter.topic_sort,
                       verify multiset-losslessness, write a result file under
                       /tmp/mnemos-sort/<id>.result.json. NO database write.
  apply [--commit]     Serially store every result file: each topic group becomes
                       an atomic child (subcategory path = base/label), tagged
                       split-from:#<orig>+nyx-sort, sibling-linked; the original is
                       archived (vector moved to tier-2) and its links re-pointed.
                       Without --commit it only reports what it would do.

Usage:
  MNEMOS_DB=/root/work/memory.db MNEMOS_NAMESPACE=mikael \\
    /root/venvs/ai/bin/python scripts/mnemos_sortkit.py <cmd> ...
"""
import json
import os
import re
import sys

OUT_DIR = "/tmp/mnemos-sort"
_COLS = ("id", "namespace", "project", "type", "importance", "subcategory",
         "valid_from", "valid_until", "status", "content")


def _mnemos():
    from mnemos.core import Mnemos
    return Mnemos(namespace=os.environ.get("MNEMOS_NAMESPACE", "mikael"),
                  enable_rerank=False, enable_contradiction_detection=False)


def _fetch(m, mid):
    cols = ", ".join(_COLS)
    return m.store._get_conn().execute(
        f"SELECT {cols} FROM memories WHERE id=?", (mid,)).fetchone()


def cmd_dump(mid):
    from mnemos.splitter import _blocks
    m = _mnemos()
    row = _fetch(m, mid)
    if not row:
        print(json.dumps({"error": "not found", "id": mid}))
        return
    blocks = ["\n".join(b) for b in _blocks(row["content"])]
    print(json.dumps({
        "id": row["id"], "project": row["project"], "type": row["type"],
        "importance": row["importance"], "subcategory": row["subcategory"],
        "status": row["status"], "chars": len(row["content"]),
        "n_blocks": len(blocks), "blocks": blocks,
    }, ensure_ascii=False))
    m.close()


def cmd_place(mid, assign_file):
    from mnemos.splitter import topic_sort, split_preserves_all_lines
    m = _mnemos()
    row = _fetch(m, mid)
    if not row:
        print(json.dumps({"error": "not found"}))
        sys.exit(2)
    from mnemos.splitter import _blocks
    n_blocks = len(_blocks(row["content"]))
    spec = json.load(open(assign_file))
    if "groups" in spec:
        # Agent-friendly format: list which block indices belong to each topic.
        assign = [None] * n_blocks
        labels = {}
        for tid, g in enumerate(spec["groups"]):
            labels[tid] = g.get("label", f"part{tid}")
            for bi in g.get("blocks", []):
                if isinstance(bi, int) and 0 <= bi < n_blocks:
                    assign[bi] = tid
        if any(a is None for a in assign):  # uncovered blocks go to a catch-all
            misc = len(spec["groups"])
            labels[misc] = "misc"
            assign = [misc if a is None else a for a in assign]
    else:
        assign = spec.get("assign")
        labels = {int(k): v for k, v in (spec.get("labels") or {}).items()}
    groups = topic_sort(row["content"], lambda blocks: assign)
    chunks = [c for _, c in groups]
    lossless = split_preserves_all_lines(row["content"], chunks)
    sorted_ok = len({t for t, _ in groups}) > 1
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "orig_id": row["id"], "namespace": row["namespace"], "project": row["project"],
        "type": row["type"], "importance": row["importance"],
        "subcategory": row["subcategory"], "valid_from": row["valid_from"],
        "valid_until": row["valid_until"], "status": row["status"],
        "lossless": lossless, "sorted": sorted_ok,
        "groups": [{"topic_id": t, "label": labels.get(t, f"part{t}"), "content": c}
                   for t, c in groups],
    }
    json.dump(out, open(f"{OUT_DIR}/{mid}.result.json", "w"), ensure_ascii=False)
    print(json.dumps({"id": mid, "lossless": lossless, "sorted": sorted_ok,
                      "n_groups": len(groups)}))
    if not lossless:
        sys.exit(3)
    m.close()


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:40] or "misc"


def cmd_apply(commit=False):
    m = _mnemos()
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".result.json")) \
        if os.path.isdir(OUT_DIR) else []
    summ = {"results": len(files), "stored_children": 0, "archived_originals": 0,
            "archived_children": 0, "skipped_not_lossless": 0, "errors": 0,
            "commit": commit}
    for fn in files:
        try:
            r = json.load(open(f"{OUT_DIR}/{fn}"))
            if not r.get("lossless"):
                summ["skipped_not_lossless"] += 1
                continue
            oid = r["orig_id"]
            base = r.get("subcategory") or r["project"]
            child_ids = []
            for g in r["groups"]:
                if not commit:
                    summ["stored_children"] += 1
                    continue
                res = m.store_memory(
                    r["project"], g["content"],
                    tags=f"split-from:#{oid},nyx-sort,topic:{_slug(g['label'])}",
                    importance=r["importance"], mem_type=r["type"], layer="semantic",
                    subcategory=f"{base}/{_slug(g['label'])}",
                    valid_from=r.get("valid_from"), valid_until=r.get("valid_until"),
                    skip_dedup=True, _no_split=True)
                if res.get("id"):
                    child_ids.append(res["id"])
                    summ["stored_children"] += 1
            if not commit:
                continue
            for a, b in zip(child_ids, child_ids[1:]):
                m.store.store_link(a, b, "related", 0.6)
            for link in m.store.get_links([oid]).get(oid, []):
                o = link.get("linked_id")
                if o and o != oid and o not in child_ids:
                    m.store.store_link(child_ids[0], o, link.get("relation", "related"),
                                       link.get("strength", 0.5))
            if r["status"] == "active":
                m.store.delete_memory(oid, hard=False)
                if hasattr(m.store, "move_embedding_to_archive"):
                    try:
                        m.store.move_embedding_to_archive(oid)
                    except Exception:
                        pass
                summ["archived_originals"] += 1
            else:
                # Original was already archived (tier-2). Keep its children in the
                # archived tier too, so active search is unaffected and tier-2
                # recall gains sharp per-topic vectors. The original stays as the
                # lineage anchor.
                for cid in child_ids:
                    m.store.delete_memory(cid, hard=False)
                    if hasattr(m.store, "move_embedding_to_archive"):
                        try:
                            m.store.move_embedding_to_archive(cid)
                        except Exception:
                            pass
                    summ["archived_children"] += 1
        except Exception:
            summ["errors"] += 1
    print(json.dumps(summ, indent=2))
    m.close()


def main(argv):
    if not argv:
        print(__doc__)
        sys.exit(1)
    cmd = argv[0]
    if cmd == "dump":
        cmd_dump(int(argv[1]))
    elif cmd == "place":
        cmd_place(int(argv[1]), argv[2])
    elif cmd == "apply":
        cmd_apply(commit="--commit" in argv[1:])
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
