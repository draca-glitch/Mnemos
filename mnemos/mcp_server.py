"""
Mnemos MCP server: exposes 6 tools over JSON-RPC
(hot: store, search, get, update; maintenance: bulk_rewrite, list_tags).

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
import threading
import time
from collections import OrderedDict

from .core import Mnemos
from . import _resource, __version__
from .constants import DEFAULT_SELF, DEFAULT_AGENT
from .constants import (
    DEFAULT_PROJECTS, VALID_TYPES, VALID_LAYERS, DEFAULT_NAMESPACE,
    CML_MODE, DEFAULT_TOOL_USAGE_LOG,
)


_STORE_DESC_CML = (
    "Store a new memory. Auto-detects duplicates and contradictions.\n\n"
    "FORMAT GUIDANCE, when to use CML vs prose:\n"
    "  Use CML (Compressed Memory Language) for: facts, decisions, contacts, configs, preferences, warnings. "
    "CML prefixes: D:(decision) C:(contact) F:(fact) L:(learning) P:(preference) W:(warning) R:(restriction). "
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
                "valid_only": {"type": "boolean", "default": True, "description": "Only currently valid memories (valid_from <= today < valid_until), including linked expansion. Pass false to retrieve history: expired or not-yet-valid memories."},
                "search_mode": {"type": "string", "enum": ["fts", "vec", "hybrid"]},
                "limit": {"type": "integer", "default": 20, "maximum": 50},
                "expand_merged": {"type": "boolean", "default": False, "description": "Tier-2 recall: enrich consolidated memories with their source originals (filtered to currently valid ones)"},
                "snippet_chars": {"type": "integer", "minimum": 50, "maximum": 2000, "description": "If set, replace result content with a query-matched window of ~this many characters (FTS5 snippet for FTS hits, head slice for vec-only hits). Major token-budget saver when hits are inside large consolidated memories."},
                "include_linked": {"type": "boolean", "default": False, "description": "Fold linked memories into each result as summaries. BFS traversal up to linked_depth hops. Saves round-trips when tracing relationships."},
                "linked_depth": {"type": "integer", "default": 1, "minimum": 1, "maximum": 3, "description": "When include_linked=true, how many hops to traverse. 1 = direct links only (default). 2-3 = transitive links; capped at 30 total linked nodes per result to prevent graph explosion. Each linked entry carries `distance` (hops from root) and optional `via` for depth>1 transitive links."},
                "include_audit_links": {"type": "boolean", "default": False, "description": "Also return Nyx audit links (contradiction-cleared). Off by default: they record that a pair was judged, not a relationship, and dominate link counts on mature stores."},
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
        "description": "Get a memory by ID in the current namespace. Bumps access count and importance at thresholds; does not confirm that its claims remain true.",
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
                "subcategory": {"type": ["string", "null"], "description": "null or empty string clears it"},
                "valid_from": {"type": ["string", "null"], "description": "ISO date; null or empty string clears it"},
                "valid_until": {"type": ["string", "null"], "description": "ISO date; null or empty string clears it, which brings an expired memory back into default (valid_only) search"},
                "verified": {"type": "boolean", "description": "Mark the memory as verified against a source or by the user. Also records last_confirmed."},
                "confirmed": {"type": "boolean", "description": "Record last_confirmed now. Implied by a content change or verified=true; pass false to suppress that for mechanical edits, true to confirm without changing anything. Ordinary reads never confirm."},
                "consolidation_lock": {"type": "boolean", "description": _LOCK_DESCRIPTION_UPDATE},
            },
            "required": ["id"],
        },
    },
]


# --- Self memory (v10.42.0) ---
# `self: true` on a tool call points it at the agent's own namespace
# (self:<agent>) instead of the user's store. The flag is only advertised,
# and only honoured, when MNEMOS_SELF is on. bulk_rewrite stays out: an
# unattended mass rewrite of an agent's self-model is not a thing to offer.
SELF_TOOLS = frozenset({
    "memory_store", "memory_search", "memory_get", "memory_update", "memory_list_tags",
})
_SELF_PROPERTY = {
    "type": "boolean",
    "default": False,
    "description": (
        "Operate on the agent's own namespace (self:<agent>) instead of the user's "
        "store: memories the agent keeps about itself, traits and observed habits, "
        "commitments about its own behaviour, each with the episode behind it. Same "
        "database, separate namespace. The agent is the name the client declared in "
        "initialize (clientInfo.name), or MNEMOS_AGENT when it declared none."
    ),
}
if DEFAULT_SELF:
    for _tool in TOOL_DEFINITIONS:
        if _tool["name"] in SELF_TOOLS:
            _tool["inputSchema"]["properties"]["self"] = _SELF_PROPERTY

# An MCP client names itself once, in initialize. The shared HTTP server sees
# many clients, so the name is kept per Mcp-Session-Id (bounded, oldest out);
# a stdio server has exactly one client for its lifetime.
_SESSION_AGENTS: "OrderedDict[str, str]" = OrderedDict()
_SESSION_AGENTS_MAX = 512
_STDIO_AGENT: list = []
_AGENT_LOCK = threading.Lock()


def _remember_agent(session_id, client_info):
    name = client_info.get("name") if isinstance(client_info, dict) else None
    if not isinstance(name, str) or not name.strip():
        return
    with _AGENT_LOCK:
        if session_id:
            _SESSION_AGENTS[session_id] = name
            _SESSION_AGENTS.move_to_end(session_id)
            while len(_SESSION_AGENTS) > _SESSION_AGENTS_MAX:
                _SESSION_AGENTS.popitem(last=False)
        else:
            _STDIO_AGENT[:] = [name]


def client_hint_from_user_agent(user_agent):
    """Product token of a User-Agent header: `claude-code/2.1.292 (cli)` ->
    `claude-code`. A client that never initialises against this server
    process (Claude Code keeps calling across a server restart and sends no
    Mcp-Session-Id, observed 2026-10-09) still names itself this way on
    every request."""
    if not isinstance(user_agent, str):
        return None
    token = user_agent.strip().split(" ", 1)[0].split("/", 1)[0].strip()
    return token or None


def _agent_for(session_id, client_hint=None):
    with _AGENT_LOCK:
        if session_id and session_id in _SESSION_AGENTS:
            return _SESSION_AGENTS[session_id]
        if _STDIO_AGENT:
            return _STDIO_AGENT[0]
    return client_hint or DEFAULT_AGENT


def _self_target(mnemos, session_id, client_hint=None):
    """The Mnemos view a self=true call runs against, or (None, reason)."""
    if not DEFAULT_SELF:
        return None, "self memory is off on this server: set MNEMOS_SELF=1 to enable it"
    agent = _agent_for(session_id, client_hint)
    if not agent:
        return None, ("no agent identity: the client declared no clientInfo.name on "
                      "initialize, sent no User-Agent, and MNEMOS_AGENT is unset")
    return mnemos.self_view(agent), None


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
        valid_only=params.get("valid_only", True),
        search_mode=params.get("search_mode"),
        limit=params.get("limit", 20),
        expand_merged=params.get("expand_merged", False),
        snippet_chars=params.get("snippet_chars"),
        include_linked=params.get("include_linked", False),
        linked_depth=params.get("linked_depth", 1),
        include_audit_links=params.get("include_audit_links", False),
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


# Fields a caller may want to unset. null or "" clears them; omitting the key
# leaves them alone. Every other field treats None as "not provided".
NULLABLE_FIELDS = frozenset({"valid_from", "valid_until", "subcategory"})


def tool_update(mnemos, params):
    mid = params.get("id")
    if mid is None:
        return {"error": "id is required"}
    fields = {}
    for key, value in params.items():
        if key == "id":
            continue
        if key in NULLABLE_FIELDS and (value is None or value == ""):
            fields[key] = None
        elif value is not None:
            fields[key] = value
    return mnemos.update(mid, **fields)


TOOL_DISPATCH = {
    "memory_store": tool_store,
    "memory_search": tool_search,
    "memory_get": tool_get,
    "memory_update": tool_update,
    "memory_list_tags": tool_list_tags,
    "memory_bulk_rewrite": tool_bulk_rewrite,
}


# Sentinel for a stdin line that could not be parsed: distinct from None
# (EOF, terminate) so one malformed line skips instead of killing the loop.
SKIP_MSG = object()

# Model warmup must run at most once per process: the stdio transport sees one
# initialize per lifetime, but the HTTP transport sees one per attached client.
_WARMUP_DONE = threading.Event()


def _maybe_warmup(mnemos):
    """Eagerly load the models so the first search is instant. Eager by
    default; set MNEMOS_EAGER_WARMUP=0 on a memory-constrained host to load
    lazily on first use instead. Idempotent across MCP sessions."""
    if os.environ.get("MNEMOS_EAGER_WARMUP", "1") != "1":
        return
    if _WARMUP_DONE.is_set():
        return
    _WARMUP_DONE.set()
    try:
        from .embed import embed
        embed(["warmup"], prefix="query")
        from .embed import embed_model_id
        sys.stderr.write(f"Mnemos: embedder loaded: {embed_model_id()}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"Mnemos: embedder warmup failed: {e}\n")
    # Warm up the reranker only if rerank is enabled
    if mnemos.enable_rerank:
        try:
            from .rerank import rerank
            rerank("warmup", [{"id": 0, "text": "warmup document"}])
            from .constants import RERANKER_MODEL
            sys.stderr.write(f"Mnemos: reranker loaded: {RERANKER_MODEL}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Mnemos: reranker warmup failed: {e}\n")


# --- MCP dual-era protocol support (2026-07-28 + legacy 2024-11-05) --------
# 2026-07-28 removed the initialize handshake: modern clients declare their
# version per-request in _meta and probe with server/discover; legacy clients
# still open with initialize, so absence of _meta is never a mismatch.
PROTOCOL_MODERN = "2026-07-28"
PROTOCOL_LEGACY = "2024-11-05"
# Revisions we know by name, newest first. This list is what server/discover
# advertises -- it is documentation, not the gate. Using it AS the gate meant
# every revision we had not heard of was refused, which broke working clients
# twice: once for 2025-03-26/2025-06-18, again for 2025-11-25.
KNOWN_VERSIONS = [
    PROTOCOL_MODERN,
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    PROTOCOL_LEGACY,
]
SUPPORTED_VERSIONS = KNOWN_VERSIONS


def _is_dated_revision(version):
    parts = version.split("-")
    return (
        len(version) == 10
        and len(parts) == 3
        and [len(p) for p in parts] == [4, 2, 2]
        and all(p.isdigit() for p in parts)
    )


def protocol_supported(version):
    """Serve any dated revision between the oldest we support and the newest we
    know, named or not.

    Every revision in that range shares one tools/list and tools/call wire
    format, so the legacy path answers correctly for the ones we have no name
    for. Anything newer than PROTOCOL_MODERN is still refused: we cannot know
    what it changed, and -32022 with the known list lets the client downgrade
    to something we do understand. ISO dates sort chronologically as strings.
    """
    return bool(version) and _is_dated_revision(version) and (
        PROTOCOL_LEGACY <= version <= PROTOCOL_MODERN
    )


SERVER_INSTRUCTIONS = "Persistent memory for AI agents: store, search, get, update memories plus bulk rewrite and tag discovery."
META_PROTOCOL = "io.modelcontextprotocol/protocolVersion"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
CACHE_TTL_MS = 3600000
ERR_UNSUPPORTED_PROTOCOL = -32022  # renumbered from -32004 in 2026-07-28


def _server_info():
    return {"name": "mnemos", "version": __version__}


def handle_message(mnemos, msg, session_id=None, client_hint=None):
    """Transport-agnostic JSON-RPC dispatch.

    Returns the response dict for a request, or None for notifications and
    id-less messages (nothing to send). Transports own framing and I/O;
    everything protocol-shaped lives here. `session_id` is the transport's
    client handle (Mcp-Session-Id on HTTP, None on stdio) and `client_hint`
    the client's self-description outside the protocol (the User-Agent
    product on HTTP); both only matter for the self namespace, which needs
    to know who is asking.
    """
    response = _dispatch(mnemos, msg, session_id, client_hint)
    # 2026-07-28 additive result envelope; legacy clients ignore the extra
    # keys. An initialize result (has protocolVersion) stays legacy-shaped.
    if response is not None:
        res = response.get("result")
        if isinstance(res, dict) and "protocolVersion" not in res:
            response = dict(response)
            res = dict(res)
            res.setdefault("resultType", "complete")
            meta = dict(res.get("_meta") or {})
            meta.setdefault(META_SERVER_INFO, _server_info())
            res["_meta"] = meta
            response["result"] = res
    return response


def _dispatch(mnemos, msg, session_id=None, client_hint=None):
    method = msg.get("method", "")
    id_ = msg.get("id")
    params = msg.get("params", {})

    if id_ is None:
        return None

    requested = ((params or {}).get("_meta") or {}).get(META_PROTOCOL)
    if requested is not None and not protocol_supported(requested):
        return {
            "jsonrpc": "2.0", "id": id_,
            "error": {
                "code": ERR_UNSUPPORTED_PROTOCOL,
                "message": "Unsupported protocol version",
                "data": {"supported": SUPPORTED_VERSIONS, "requested": requested},
            },
        }

    if method == "server/discover":
        return {"jsonrpc": "2.0", "id": id_, "result": {
            "supportedVersions": SUPPORTED_VERSIONS,
            "capabilities": {"tools": {}},
            "instructions": SERVER_INSTRUCTIONS,
            "ttlMs": CACHE_TTL_MS,
            "cacheScope": "public",
        }}

    if method == "initialize":
        response = {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "protocolVersion": (
                    params.get("protocolVersion")
                    if protocol_supported(params.get("protocolVersion"))
                    else PROTOCOL_LEGACY
                ),
                "capabilities": {"tools": {}},
                "serverInfo": _server_info(),
            },
        }
        _remember_agent(session_id, params.get("clientInfo"))
        _maybe_warmup(mnemos)
        return response

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {
            "tools": sorted(TOOL_DEFINITIONS, key=lambda t: t.get("name", "")),
            "ttlMs": CACHE_TTL_MS,
            "cacheScope": "public",
        }}

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        # Opt-in tool-usage logging: if enabled, record tool_name +
        # timestamp. No arguments, no content. Useful for health-check
        # tooling that wants to answer "was the server responsive?"
        # without parsing MCP transport logs.
        if DEFAULT_TOOL_USAGE_LOG:
            try:
                mnemos.store.log_tool_usage(tool_name)
            except Exception:
                pass
        handler = TOOL_DISPATCH.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0", "id": id_,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
                    "isError": True,
                },
            }
        target = mnemos
        use_self = False
        if isinstance(tool_args, dict) and "self" in tool_args:
            tool_args = dict(tool_args)
            use_self = bool(tool_args.pop("self"))
        if use_self:
            target, reason = _self_target(mnemos, session_id, client_hint)
            if target is None:
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": reason})}],
                        "isError": True,
                    },
                }
        try:
            result = handler(target, tool_args)
            if use_self and isinstance(result, dict):
                result = dict(result)
                result["namespace"] = target.namespace
            return {
                "jsonrpc": "2.0", "id": id_,
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
            }
        except Exception as e:
            # Full detail to stderr for the operator; the caller gets the
            # class plus a truncated message (raw str(e) can carry DB
            # paths and whole schema fragments).
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            brief = f"{type(e).__name__}: {str(e)[:300]}"
            return {
                "jsonrpc": "2.0", "id": id_,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": brief})}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0", "id": id_,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def read_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Mnemos: skipping malformed stdin line: {e}\n")
        sys.stderr.flush()
        return SKIP_MSG


def send_msg(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    sys.stderr.write(f"Mnemos MCP server v{__version__} starting (CPU-only, no GPU required)\n")
    sys.stderr.flush()

    mnemos = build_mnemos()

    # Optional idle-unload reaper. Only starts when MNEMOS_MODEL_IDLE_TTL > 0;
    # on a memory-constrained host it returns embedder/reranker RSS to the OS
    # after they sit idle. No thread, and no effect, by default.
    _resource.start_idle_reaper(
        log=lambda m: (sys.stderr.write(m + "\n"), sys.stderr.flush()))

    while True:
        msg = read_msg()
        if msg is None:
            break
        if msg is SKIP_MSG:
            continue

        response = handle_message(mnemos, msg)
        if response is not None:
            send_msg(response)


if __name__ == "__main__":
    main()
