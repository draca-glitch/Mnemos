"""
Mnemos MCP server: exposes 4 tools (store, search, get, update) over JSON-RPC.

Protocol: newline-delimited JSON-RPC 2.0 over stdin/stdout.
Methods: initialize, notifications/initialized, tools/list, tools/call.

Designed to work with any MCP-compatible AI client: Claude Code, Cursor,
ChatGPT Desktop, Gemini, etc. CPU-only, no GPU required.

Storage backend is configurable via environment:
  MNEMOS_BACKEND=sqlite (default) | qdrant | postgres
  MNEMOS_DB=/path/to/memory.db    (SQLite path)
  MNEMOS_NAMESPACE=default        (multi-user namespace)
"""

import json
import os
import sys

from .core import Mnemos
from .constants import (
    DEFAULT_PROJECTS, VALID_TYPES, VALID_LAYERS, DEFAULT_NAMESPACE,
    CML_MODE, DEFAULT_TOOL_USAGE_LOG,
)


_STORE_DESC_CML = (
    "Store a new memory. Auto-detects duplicates and contradictions.\n\n"
    "FORMAT GUIDANCE, when to use CML vs prose:\n"
    "  Use CML (Compressed Memory Language) for: facts, decisions, contacts, configs, preferences, warnings. "
    "CML prefixes: D:(decision) C:(contact) F:(fact) L:(learning) P:(preference) W:(warning). "
    "Symbols: → ↔ ← ∵ ∴ △ ⚠ @ ✓ ✗ ~ ∅ … ; > #N. Dense, one-line-per-fact chains with ;\n"
    "  Use plain prose for: runbooks with ordered steps, long-form reference documents, code blocks, "
    "creative writing, multi-paragraph narrative. These suffer from CML compression. "
    "For prose-format memories, set consolidation_lock=true (via memory_update after store) "
    "to prevent the Nyx cycle from cemelifying them later.\n"
    "  When in doubt: if the content is primarily a set of atomic facts → CML. "
    "If it has essential ordering, structure, or code → prose."
)

_STORE_DESC_PROSE = (
    "Store a new memory. Auto-detects duplicates and contradictions. "
    "Write the content as clear natural prose; keep it concise (one or two "
    "sentences for a single fact, a short paragraph for a cluster of related "
    "facts). No special formatting required; Mnemos indexes plain text."
)

_STORE_DESCRIPTION = _STORE_DESC_PROSE if CML_MODE == "off" else _STORE_DESC_CML

_CONTENT_DESC_CML = "The memory content. Use CML for facts/decisions/configs; use prose for runbooks/docs/code (see description)."
_CONTENT_DESC_PROSE = "The memory content as clear natural prose."
_CONTENT_DESCRIPTION = _CONTENT_DESC_PROSE if CML_MODE == "off" else _CONTENT_DESC_CML

_LOCK_DESC_STORE_CML = "Set true for prose-format content (runbooks, long docs, code blocks) to prevent the Nyx cycle from cemelifying it."
_LOCK_DESC_STORE_PROSE = "Set true to prevent the Nyx cycle from merging this memory with others during consolidation."
_LOCK_DESCRIPTION_STORE = _LOCK_DESC_STORE_PROSE if CML_MODE == "off" else _LOCK_DESC_STORE_CML

_LOCK_DESC_UPDATE_CML = "Set true to prevent the Nyx cycle from cemelifying or merging this memory."
_LOCK_DESC_UPDATE_PROSE = "Set true to prevent the Nyx cycle from merging this memory with others during consolidation."
_LOCK_DESCRIPTION_UPDATE = _LOCK_DESC_UPDATE_PROSE if CML_MODE == "off" else _LOCK_DESC_UPDATE_CML


def build_mnemos():
    """Construct a Mnemos instance based on environment configuration.

    The reranker enable flag is read from MNEMOS_ENABLE_RERANK in
    constants.DEFAULT_ENABLE_RERANK and applied via the Mnemos constructor
    default; we do not re-read it here to keep the env var read in exactly
    one place.
    """
    backend = os.environ.get("MNEMOS_BACKEND", "sqlite").lower()
    namespace = os.environ.get("MNEMOS_NAMESPACE", DEFAULT_NAMESPACE)

    if backend == "sqlite":
        from .storage.sqlite_store import SQLiteStore
        # db_path defaults to constants.DEFAULT_DB_PATH (which reads MNEMOS_DB
        # in a single place), so we do not re-read the env var here.
        store = SQLiteStore(namespace=namespace)
    elif backend == "qdrant":
        from .storage.qdrant_store import QdrantStore
        # sqlite_path inherits DEFAULT_DB_PATH from constants via SQLiteStore.
        store = QdrantStore(
            qdrant_url=os.environ.get("MNEMOS_QDRANT_URL", "http://localhost:6333"),
            collection=os.environ.get("MNEMOS_QDRANT_COLLECTION", "mnemos_memories"),
            api_key=os.environ.get("MNEMOS_QDRANT_API_KEY"),
            namespace=namespace,
        )
    elif backend == "postgres":
        from .storage.postgres_store import PostgresStore
        store = PostgresStore(namespace=namespace)
    else:
        raise ValueError(f"Unknown MNEMOS_BACKEND: {backend}")

    return Mnemos(store=store, namespace=namespace)


# --- Tool definitions ---

TOOL_DEFINITIONS = [
    {
        "name": "memory_store",
        "description": _STORE_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Top-level category (e.g., dev, finance, personal)"},
                "content": {"type": "string", "description": _CONTENT_DESCRIPTION},
                "tags": {"type": "string", "description": "Comma-separated tags"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "type": {"type": "string", "enum": list(sorted(VALID_TYPES)), "default": "fact"},
                "layer": {"type": "string", "enum": list(sorted(VALID_LAYERS)), "default": "semantic"},
                "verified": {"type": "boolean", "default": False},
                "subcategory": {"type": "string", "description": "Hierarchical sub-category (e.g., 'crypto' under finance)"},
                "valid_from": {"type": "string", "description": "ISO date when fact becomes valid"},
                "valid_until": {"type": "string", "description": "ISO date when fact expires"},
                "consolidation_lock": {"type": "boolean", "default": False, "description": _LOCK_DESCRIPTION_STORE},
            },
            "required": ["project", "content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Hybrid search: FTS5 + vector + RRF + optional rerank. Auto-widens on thin results. Supports snippet extraction and linked-memory expansion for bandwidth-aware callers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "project": {"type": "string"},
                "subcategory": {"type": "string"},
                "type": {"type": "string", "enum": list(sorted(VALID_TYPES))},
                "layer": {"type": "string", "enum": list(sorted(VALID_LAYERS))},
                "status": {"type": "string", "default": "active"},
                "valid_only": {"type": "boolean", "default": False, "description": "Exclude memories past their valid_until"},
                "search_mode": {"type": "string", "enum": ["fts", "vec", "hybrid"]},
                "limit": {"type": "integer", "default": 20, "maximum": 50},
                "expand_merged": {"type": "boolean", "default": False, "description": "Tier-2 recall: enrich consolidated memories with their source originals (filtered to currently valid ones)"},
                "snippet_chars": {"type": "integer", "minimum": 50, "maximum": 2000, "description": "If set, replace result content with a query-matched window of ~this many characters (FTS5 snippet for FTS hits, head slice for vec-only hits). Major token-budget saver when hits are inside large consolidated memories."},
                "include_linked": {"type": "boolean", "default": False, "description": "Fold first-hop linked memories into each result as summaries. Saves round-trips when tracing relationships."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_bulk_rewrite",
        "description": (
            "Find-and-replace across memories in one call. Default dry_run=true returns a preview "
            "(matched count, affected count, per-memory before/after snippets) without touching the DB. "
            "Set dry_run=false to commit. max_affected caps the operation: if more memories would "
            "change than the cap, the call errors out without writing anything (prevents runaway "
            "rewrites). Re-embeds every modified memory. Namespace-scoped, active-only. "
            "Use_regex=true switches from plain substring to Python regex syntax."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Substring (default) or regex (if use_regex=true) to find"},
                "replacement": {"type": "string", "description": "Text to replace with. Empty string allowed for deletion."},
                "project": {"type": "string", "description": "Optional: only rewrite memories in this project"},
                "tags": {"type": "string", "description": "Optional: only rewrite memories whose tags contain this substring"},
                "dry_run": {"type": "boolean", "default": True, "description": "If true (default), return preview without writing. If false, commit."},
                "max_affected": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500, "description": "Abort without writing if more memories would change than this"},
                "use_regex": {"type": "boolean", "default": False, "description": "Treat pattern as Python regex instead of plain substring"},
                "preview_chars": {"type": "integer", "default": 120, "minimum": 40, "maximum": 500, "description": "Chars of context around each match in the preview"},
            },
            "required": ["pattern", "replacement"],
        },
    },
    {
        "name": "memory_list_tags",
        "description": "Discover existing tag conventions. Returns unique tags with usage counts and an example memory ID per tag, so callers can reuse established tag names instead of creating drifted duplicates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Filter to tags used within this project only"},
                "min_count": {"type": "integer", "default": 1, "minimum": 1, "description": "Only return tags used at least this many times"},
                "order_by": {"type": "string", "enum": ["count", "alpha"], "default": "count"},
                "limit": {"type": "integer", "default": 500, "maximum": 2000},
            },
        },
    },
    {
        "name": "memory_get",
        "description": "Get a memory by ID. Bumps access count and importance at thresholds.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "memory_update",
        "description": "Update fields of an existing memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "content": {"type": "string"},
                "project": {"type": "string"},
                "tags": {"type": "string"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 10},
                "status": {"type": "string", "enum": ["active", "archived"]},
                "type": {"type": "string", "enum": list(sorted(VALID_TYPES))},
                "layer": {"type": "string", "enum": list(sorted(VALID_LAYERS))},
                "subcategory": {"type": "string"},
                "valid_from": {"type": "string"},
                "valid_until": {"type": "string"},
                "consolidation_lock": {"type": "boolean", "description": _LOCK_DESCRIPTION_UPDATE},
            },
            "required": ["id"],
        },
    },
]


def tool_store(mnemos, params):
    return mnemos.store_memory(
        project=params.get("project", ""),
        content=params.get("content", ""),
        tags=params.get("tags", ""),
        importance=params.get("importance", 5),
        mem_type=params.get("type", "fact"),
        layer=params.get("layer", "semantic"),
        verified=params.get("verified", False),
        subcategory=params.get("subcategory"),
        valid_from=params.get("valid_from"),
        valid_until=params.get("valid_until"),
        consolidation_lock=params.get("consolidation_lock", False),
    )


def tool_search(mnemos, params):
    return mnemos.search(
        query=params.get("query", ""),
        project=params.get("project"),
        subcategory=params.get("subcategory"),
        type_filter=params.get("type"),
        layer=params.get("layer"),
        status=params.get("status", "active"),
        valid_only=params.get("valid_only", False),
        search_mode=params.get("search_mode"),
        limit=params.get("limit", 20),
        expand_merged=params.get("expand_merged", False),
        snippet_chars=params.get("snippet_chars"),
        include_linked=params.get("include_linked", False),
    )


def tool_bulk_rewrite(mnemos, params):
    return mnemos.bulk_rewrite(
        pattern=params.get("pattern", ""),
        replacement=params.get("replacement", ""),
        project=params.get("project"),
        tags=params.get("tags"),
        dry_run=params.get("dry_run", True),
        max_affected=params.get("max_affected", 50),
        use_regex=params.get("use_regex", False),
        preview_chars=params.get("preview_chars", 120),
    )


def tool_list_tags(mnemos, params):
    return {
        "tags": mnemos.list_tags(
            project=params.get("project"),
            min_count=params.get("min_count", 1),
            order_by=params.get("order_by", "count"),
            limit=params.get("limit", 500),
        ),
    }


def tool_get(mnemos, params):
    mid = params.get("id")
    if mid is None:
        return {"error": "id is required"}
    return mnemos.get(mid)


def tool_update(mnemos, params):
    mid = params.get("id")
    if mid is None:
        return {"error": "id is required"}
    fields = {k: v for k, v in params.items() if k != "id" and v is not None}
    return mnemos.update(mid, **fields)


TOOL_DISPATCH = {
    "memory_store": tool_store,
    "memory_search": tool_search,
    "memory_get": tool_get,
    "memory_update": tool_update,
    "memory_list_tags": tool_list_tags,
    "memory_bulk_rewrite": tool_bulk_rewrite,
}


def read_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def send_msg(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    sys.stderr.write("Mnemos MCP server v10.2 starting (CPU-only, no GPU required)\n")
    sys.stderr.flush()

    mnemos = build_mnemos()

    while True:
        msg = read_msg()
        if msg is None:
            break

        method = msg.get("method", "")
        id_ = msg.get("id")
        params = msg.get("params", {})

        if id_ is None:
            continue

        if method == "initialize":
            send_msg({
                "jsonrpc": "2.0", "id": id_,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mnemos", "version": "10.3.1"},
                },
            })
            # Warm up the embedder so first search is instant
            try:
                from .embed import embed
                embed(["warmup"], prefix="query")
                sys.stderr.write("Mnemos: e5-large model loaded\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"Mnemos: embedder warmup failed: {e}\n")
            # Warm up the reranker only if rerank is enabled
            if mnemos.enable_rerank:
                try:
                    from .rerank import rerank
                    rerank("warmup", [{"id": 0, "text": "warmup document"}])
                    sys.stderr.write("Mnemos: jina reranker loaded\n")
                    sys.stderr.flush()
                except Exception as e:
                    sys.stderr.write(f"Mnemos: reranker warmup failed: {e}\n")

        elif method == "tools/list":
            send_msg({"jsonrpc": "2.0", "id": id_, "result": {"tools": TOOL_DEFINITIONS}})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            # Opt-in tool-usage logging: if enabled, record tool_name +
            # timestamp. No arguments, no content. Useful for health-check
            # tooling that wants to answer "was the server responsive?"
            # without parsing MCP stdin/stdout logs.
            if DEFAULT_TOOL_USAGE_LOG:
                try:
                    mnemos.store.log_tool_usage(tool_name)
                except Exception:
                    pass
            handler = TOOL_DISPATCH.get(tool_name)
            if not handler:
                send_msg({
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
                        "isError": True,
                    },
                })
                continue
            try:
                result = handler(mnemos, tool_args)
                send_msg({
                    "jsonrpc": "2.0", "id": id_,
                    "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
                })
            except Exception as e:
                send_msg({
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                        "isError": True,
                    },
                })
        else:
            send_msg({
                "jsonrpc": "2.0", "id": id_,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    main()
