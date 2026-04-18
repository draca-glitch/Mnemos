"""
Mnemos CLI: store/search/get/update/stats plus bulk_rewrite/list_tags
maintenance from the command line.

Usage:
    mnemos add --project dev "F:Mnemos uses sqlite-vec"
    mnemos search "vector storage" --project dev
    mnemos get 42
    mnemos stats
    mnemos serve  # start MCP server on stdio
"""

import argparse
import json
import os
import sys

from .core import Mnemos
from .constants import VALID_TYPES, VALID_LAYERS, DEFAULT_NAMESPACE


def cmd_add(mnemos, args):
    result = mnemos.store_memory(
        project=args.project,
        content=args.content,
        tags=args.tags or "",
        importance=args.importance or 5,
        mem_type=args.type or "fact",
        layer=args.layer or "semantic",
        verified=args.verified,
        subcategory=args.subcategory,
        valid_from=args.valid_from,
        valid_until=args.valid_until,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_search(mnemos, args):
    result = mnemos.search(
        query=args.query,
        project=args.project,
        subcategory=args.subcategory,
        type_filter=args.type,
        layer=args.layer,
        valid_only=args.valid_only,
        search_mode=args.mode,
        limit=args.limit,
        expand_merged=args.expand_merged,
        snippet_chars=getattr(args, "snippet_chars", None),
        include_linked=getattr(args, "include_linked", False),
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Found {result.get('count', 0)} results (mode: {result.get('search_mode')})")
        for r in result.get("results", []):
            mid = r.get("id")
            project = r.get("project", "")
            content = (r.get("content") or "")[:120]
            print(f"  #{mid} [{project}] {content}")


def cmd_get(mnemos, args):
    result = mnemos.get(args.id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_update(mnemos, args):
    fields = {}
    for k in ("content", "project", "tags", "importance", "status", "type",
              "layer", "subcategory", "valid_from", "valid_until"):
        v = getattr(args, k, None)
        if v is not None:
            fields[k] = v
    result = mnemos.update(args.id, **fields)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_delete(mnemos, args):
    result = mnemos.delete(args.id, hard=args.hard)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_stats(mnemos, args):
    print(json.dumps(mnemos.stats(), indent=2, ensure_ascii=False))


def cmd_tags(mnemos, args):
    tags = mnemos.list_tags(
        project=args.project, min_count=args.min_count,
        order_by=args.order_by, limit=args.limit,
    )
    if args.json:
        print(json.dumps(tags, indent=2, ensure_ascii=False))
    else:
        if not tags:
            print("(no tags)")
            return
        for t in tags:
            print(f"  {t['count']:5d}  {t['tag']}  (example #{t['example_id']})")


def cmd_briefing(mnemos, args):
    print(mnemos.briefing(project=args.project, budget_chars=args.budget))


def cmd_digest(mnemos, args):
    rows = mnemos.digest(days=args.days, project=args.project)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(f"# Digest, last {args.days} day(s) ({len(rows)} memories)\n")
        for r in rows:
            print(f"  #{r['id']} [{r['project']}] ({r['created_at']}) {r['content'][:120]}")


def cmd_map(mnemos, args):
    m = mnemos.map()
    if args.json:
        print(json.dumps(m, indent=2, ensure_ascii=False))
        return
    print("# Mnemos memory map\n")
    for project, info in sorted(m.items()):
        print(f"## {project} ({info['total']})")
        for sub, n in info.get("subcategories", {}).items():
            print(f"  {sub}: {n}")
        print()


def cmd_embed_status(mnemos, args):
    print(json.dumps(mnemos.embed_status(), indent=2, ensure_ascii=False))


def cmd_doctor(mnemos, args):
    report = mnemos.doctor(migrate=getattr(args, "migrate", False))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"# Mnemos doctor (namespace: {report['namespace']})")
    print(f"\nStatus: {report['status']}\n")
    if report.get("backup"):
        print(f"Pre-migration backup: {report['backup']}\n")
    if report.get("checks"):
        print("Checks:")
        for c in report["checks"]:
            print(f"  ok  {c}")
    if report.get("migrations_applied"):
        print("\nMigrations applied:")
        for m in report["migrations_applied"]:
            print(f"  +   {m}")
    if report.get("issues"):
        print("\nIssues:")
        for i in report["issues"]:
            print(f"  !!  {i}")
        if not getattr(args, "migrate", False):
            print("\nRun `mnemos doctor --migrate` to apply safe fixes.")


def cmd_prime(mnemos, args):
    rows = mnemos.prime(args.context, project=args.project, limit=args.limit)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for r in rows:
            print(f"  #{r['id']} [{r['project']}] {r['content'][:120]}")


def cmd_consolidate(mnemos, args):
    phases = None
    if args.phases:
        phases = {int(p) for p in args.phases.split(",")}
    elif args.nyx:
        phases = {1, 2, 3, 4, 5, 6}
    try:
        stats = mnemos.consolidate(
            execute=args.execute,
            phases=phases,
            surge=args.surge,
            project=args.project,
        )
    except RuntimeError as e:
        # v10.4.0: loud-fail on missing LLM config. One-line stderr message
        # (no traceback) and exit code 2 so cron / shell hooks can detect.
        print(f"mnemos consolidate: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(stats, indent=2, default=str, ensure_ascii=False))


def cmd_ingest(mnemos, args):
    from .ingest import ingest_path
    stats = ingest_path(
        mnemos,
        args.path,
        project=args.project or "ingested",
        subcategory=args.subcategory,
        pattern=args.pattern or "*",
        recursive=args.recursive,
        chunk_chars=args.chunk,
        importance=args.importance or 4,
        skip_dedup=not args.dedup,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    label = "DRY RUN" if args.dry_run else "EXECUTED"
    print(f"# Ingest report ({label})")
    print(f"  Files scanned:  {stats.get('files', 0)}")
    print(f"  Chunks created: {stats.get('chunks', 0)}")
    print(f"  Stored:         {stats.get('stored', 0)}")
    print(f"  Skipped:        {stats.get('skipped', 0)}")
    print(f"  Errors:         {stats.get('errors', 0)}")
    if stats.get("by_extension"):
        print(f"  By extension:")
        for ext, n in sorted(stats["by_extension"].items(), key=lambda x: -x[1]):
            print(f"    {ext}: {n}")


def cmd_serve(mnemos, args):
    from . import mcp_server
    mcp_server.main()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mnemos", description="Mnemos memory system CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # add
    p = sub.add_parser("add", help="Store a new memory")
    p.add_argument("--project", "-p", required=True)
    p.add_argument("content")
    p.add_argument("--tags", "-t")
    p.add_argument("--importance", "-i", type=int, choices=range(1, 11))
    p.add_argument("--type", choices=sorted(VALID_TYPES))
    p.add_argument("--layer", choices=sorted(VALID_LAYERS))
    p.add_argument("--verified", action="store_true")
    p.add_argument("--subcategory", "--sub")
    p.add_argument("--valid-from")
    p.add_argument("--valid-until")
    p.set_defaults(fn=cmd_add)

    # search
    p = sub.add_parser("search", help="Search memories")
    p.add_argument("query")
    p.add_argument("--project", "-p")
    p.add_argument("--subcategory", "--sub")
    p.add_argument("--type", choices=sorted(VALID_TYPES))
    p.add_argument("--layer", choices=sorted(VALID_LAYERS))
    p.add_argument("--valid-only", action="store_true")
    p.add_argument("--mode", choices=["fts", "vec", "hybrid"])
    p.add_argument("--limit", "-l", type=int, default=20)
    p.add_argument("--expand-merged", action="store_true",
                   help="Tier-2 recall: enrich consolidated memories with their source originals")
    p.add_argument("--snippet-chars", type=int, default=None,
                   help="Replace result content with a query-matched window of ~N chars")
    p.add_argument("--include-linked", action="store_true",
                   help="Fold first-hop linked memories into each result as summaries")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    # get
    p = sub.add_parser("get", help="Get a memory by ID")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_get)

    # update
    p = sub.add_parser("update", help="Update a memory")
    p.add_argument("id", type=int)
    p.add_argument("--content")
    p.add_argument("--project", "-p")
    p.add_argument("--tags", "-t")
    p.add_argument("--importance", "-i", type=int, choices=range(1, 11))
    p.add_argument("--status", choices=["active", "archived"])
    p.add_argument("--type", choices=sorted(VALID_TYPES))
    p.add_argument("--layer", choices=sorted(VALID_LAYERS))
    p.add_argument("--subcategory", "--sub")
    p.add_argument("--valid-from")
    p.add_argument("--valid-until")
    p.set_defaults(fn=cmd_update)

    # delete
    p = sub.add_parser("delete", help="Archive (default) or hard-delete a memory")
    p.add_argument("id", type=int)
    p.add_argument("--hard", action="store_true")
    p.set_defaults(fn=cmd_delete)

    # stats
    p = sub.add_parser("stats", help="Show memory statistics")
    p.set_defaults(fn=cmd_stats)

    # tags
    p = sub.add_parser("tags", help="List unique tags with usage counts (discover tag conventions)")
    p.add_argument("--project", "-p")
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--order-by", choices=["count", "alpha"], default="count")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_tags)

    # briefing
    p = sub.add_parser("briefing", help="Compact session-start briefing (~370 tokens)")
    p.add_argument("--project", "-p")
    p.add_argument("--budget", type=int, default=1300, help="Char budget (default 1300)")
    p.set_defaults(fn=cmd_briefing)

    # digest
    p = sub.add_parser("digest", help="Recent memories from last N days")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--project", "-p")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_digest)

    # map
    p = sub.add_parser("map", help="Topic index by project/subcategory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_map)

    # embed-status
    p = sub.add_parser("embed-status", help="Embedding coverage report")
    p.set_defaults(fn=cmd_embed_status)

    # doctor
    p = sub.add_parser("doctor", help="Health check (and optional self-repair of schema drift)")
    p.add_argument("--migrate", action="store_true",
                   help="Apply safe fixes for detected schema drift (backfills missing columns, creates missing aux tables, rebuilds out-of-sync FTS index). Creates a pre-migration DB backup before any changes.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    # prime
    p = sub.add_parser("prime", help="Predictive priming, find context-relevant memories")
    p.add_argument("context", help="Context string (e.g., current task description)")
    p.add_argument("--project", "-p")
    p.add_argument("--limit", "-l", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_prime)

    # consolidate (Nyx cycle)
    p = sub.add_parser("consolidate", help="Run Nyx cycle (requires LLM for phases 2-5; phase 6 is SQL-only bookkeeping)")
    p.add_argument("--execute", action="store_true", help="Apply changes (default: dry run)")
    p.add_argument("--phases", help="Comma-separated phases 1..6 (e.g., 2,3,6 = dedup+weave+bookkeeping)")
    p.add_argument("--nyx", action="store_true", help="Include synthesis (phase 5)")
    p.add_argument("--surge", action="store_true", help="Force surge mode (higher LLM call limits)")
    p.add_argument("--project", "-p")
    p.set_defaults(fn=cmd_consolidate)

    # ingest
    p = sub.add_parser("ingest", help="Ingest text files into Mnemos as memories")
    p.add_argument("path", help="File or directory to ingest")
    p.add_argument("--project", "-p", help="Project to store under (default: ingested)")
    p.add_argument("--subcategory", "--sub", help="Subcategory (default: source folder name)")
    p.add_argument("--pattern", help='Glob pattern, e.g., "*.md" (default: *)')
    p.add_argument("--recursive", "-r", action="store_true", help="Walk subdirectories")
    p.add_argument("--chunk", type=int, default=2000, help="Max chars per chunk (0 = no chunking)")
    p.add_argument("--importance", "-i", type=int, choices=range(1, 11), help="Default importance (default 4)")
    p.add_argument("--dedup", action="store_true", help="Run dedup check (slower; default off for bulk)")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen without storing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    # serve
    p = sub.add_parser("serve", help="Start MCP server on stdio")
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    # v10.4.2: honor MNEMOS_NAMESPACE env (the MCP server already does;
    # without this the CLI silently used DEFAULT_NAMESPACE and reported 0
    # for every command on a non-default-namespace database).
    namespace = os.environ.get("MNEMOS_NAMESPACE", DEFAULT_NAMESPACE)
    mnemos = Mnemos(namespace=namespace)
    try:
        args.fn(mnemos, args)
    finally:
        mnemos.close()


if __name__ == "__main__":
    main()
