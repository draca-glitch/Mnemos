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
from .constants import DEFAULT_PROJECTS, VALID_TYPES, VALID_LAYERS, DEFAULT_NAMESPACE


def build_mnemos():
    """Construct a Mnemos instance based on environment configuration."""
    backend = os.environ.get("MNEMOS_BACKEND", "sqlite").lower()
    namespace = os.environ.get("MNEMOS_NAMESPACE", DEFAULT_NAMESPACE)
    enable_rerank = os.environ.get("MNEMOS_ENABLE_RERANK", "0").lower() in ("1", "true", "yes", "on")

    if backend == "sqlite":
        from .storage.sqlite_store import SQLiteStore
        store = SQLiteStore(
            db_path=os.environ.get("MNEMOS_DB"),
            namespace=namespace,
        )
    elif backend == "qdrant":
        from .storage.qdrant_store import QdrantStore
        store = QdrantStore(
            sqlite_path=os.environ.get("MNEMOS_DB"),
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

    return Mnemos(store=store, namespace=namespace, enable_rerank=enable_rerank)


# --- Tool definitions ---

TOOL_DEFINITIONS = [
    {
        "name": "memory_store",
        "description": "Store a new memory. Auto-detects duplicates and contradictions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Top-level category (e.g., dev, finance, personal)"},
                "content": {"type": "string", "description": "The memory content"},
                "tags": {"type": "string", "description": "Comma-separated tags"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "type": {"type": "string", "enum": list(sorted(VALID_TYPES)), "default": "fact"},
                "layer": {"type": "string", "enum": list(sorted(VALID_LAYERS)), "default": "semantic"},
                "verified": {"type": "boolean", "default": False},
                "subcategory": {"type": "string", "description": "Hierarchical sub-category (e.g., 'crypto' under finance)"},
                "valid_from": {"type": "string", "description": "ISO date when fact becomes valid"},
                "valid_until": {"type": "string", "description": "ISO date when fact expires"},
            },
            "required": ["project", "content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Hybrid search: FTS5 + vector + RRF + optional rerank. Auto-widens on thin results.",
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
            },
            "required": ["query"],
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
    )


def tool_get(mnemos, params):
    return mnemos.get(params.get("id"))


def tool_update(mnemos, params):
    fields = {k: v for k, v in params.items() if k != "id" and v is not None}
    return mnemos.update(params.get("id"), **fields)


TOOL_DISPATCH = {
    "memory_store": tool_store,
    "memory_search": tool_search,
    "memory_get": tool_get,
    "memory_update": tool_update,
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
    sys.stderr.write("Mnemos MCP server v10.0 starting (CPU-only, no GPU required)\n")
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
                    "serverInfo": {"name": "mnemos", "version": "10.0.0"},
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
