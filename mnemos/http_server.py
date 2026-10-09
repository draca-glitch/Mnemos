"""
Streamable-HTTP transport for the Mnemos MCP server.

Purpose: let several AI harnesses (Claude Code, Codex, Grok, ...) attach to ONE
long-lived Mnemos process so the ONNX models (e5-large, Jina reranker, NLI)
load exactly once, instead of every harness stdio-spawning a private copy that
duplicates gigabytes of anonymous heap. Measured on a 31G host (2026-08-18):
three stdio copies held 16.4G + 5.5G + warmup pending; the shared file-backed
RSS of the .onnx files was ~12MB, everything else was per-process weight copies
plus ONNX Runtime arena growth.

Protocol: MCP streamable HTTP, request/response subset. Each POST body is one
JSON-RPC message (or a batch array); the response is application/json. There is
no server-initiated stream, so GET returns 405. Notifications get 202 with an
empty body. An Mcp-Session-Id header is issued on initialize and accepted (but
not required) on subsequent requests: tool state is process-global by design,
that is the whole point.

Concurrency: HTTP sessions are served by threads, but dispatch is serialized
through one process-wide lock. Two harnesses searching at once queue for a few
hundred milliseconds instead of running concurrent ONNX sessions that grow the
ORT arena without bound. SQLite writes already take BEGIN IMMEDIATE underneath.

Security: binds 127.0.0.1 (or a 0600 user-only unix socket) and refuses
non-local addresses unless MNEMOS_HTTP_ALLOW_NONLOCAL=1 is set explicitly.
No auth on localhost; do not expose this beyond the machine.

Usage:
  mnemos serve --http 127.0.0.1:8377
  mnemos serve --unix /run/user/1000/mnemos.sock

stdio remains the default transport (`mnemos-mcp`, `mnemos serve`) for
single-client installs; nothing about it changes.
"""

import json
import os
import socket
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .mcp_server import (
    client_hint_from_user_agent,
    ERR_UNSUPPORTED_PROTOCOL,
    SUPPORTED_VERSIONS,
    build_mnemos,
    handle_message,
    protocol_supported,
)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# One ONNX inference at a time across every attached harness. Coarse on
# purpose: the lock covers the whole dispatch so embed/rerank/NLI can never
# run concurrently, which is what kept per-process ORT arenas growing.
_DISPATCH_LOCK = threading.Lock()


class MnemosHTTPHandler(BaseHTTPRequestHandler):
    server_version = f"mnemos-mcp/{__version__}"
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self._session_id:
            self.send_header("Mcp-Session-Id", self._session_id)
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        self._session_id = ""
        # Read the body before anything can return early. Leaving it unread
        # desyncs a keep-alive connection: the next request line is parsed as a
        # continuation of the body still sitting in the socket buffer, so one
        # refused request takes the following one down with it.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""

        # MCP-Protocol-Version is required on Streamable HTTP since 2025-06-18;
        # absent means a legacy client and is accepted, present-but-unsupported
        # is refused up front (400, same error shape as the JSON-RPC layer).
        proto = self.headers.get("MCP-Protocol-Version")
        if proto and not protocol_supported(proto):
            self._send_json(400, {
                "jsonrpc": "2.0", "id": None,
                "error": {
                    "code": ERR_UNSUPPORTED_PROTOCOL,
                    "message": "Unsupported protocol version",
                    "data": {"supported": SUPPORTED_VERSIONS, "requested": proto},
                },
            })
            return
        try:
            msg = json.loads(raw)
        except Exception:
            self._send_json(400, {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            })
            return

        batch = isinstance(msg, list)
        messages = msg if batch else [msg]
        responses = []
        session_id = (self.headers.get("Mcp-Session-Id") or "").strip()
        client_hint = client_hint_from_user_agent(self.headers.get("User-Agent"))
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("method") == "initialize" and m.get("id") is not None:
                self._session_id = uuid.uuid4().hex
                session_id = self._session_id
            with _DISPATCH_LOCK:
                response = handle_message(
                    self.server.mnemos, m, session_id=session_id or None,
                    client_hint=client_hint)
            if response is not None:
                responses.append(response)

        if not responses:
            self._send_empty(202)
            return
        self._send_json(200, responses if batch else responses[0])

    def do_GET(self) -> None:
        # No server-initiated stream in this subset.
        self._session_id = ""
        self._send_empty(405)

    def do_DELETE(self) -> None:
        # Session teardown is a no-op: state is process-global by design.
        self._session_id = ""
        self._send_empty(202)

    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        sys.stderr.write("Mnemos http: %s\n" % (format % args))


class MnemosHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, mnemos):
        super().__init__(address, MnemosHTTPHandler)
        self.mnemos = mnemos


class MnemosUnixHTTPServer(MnemosHTTPServer):
    address_family = socket.AF_UNIX

    def __init__(self, socket_path, mnemos):
        self._socket_path = socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        super().__init__(socket_path, mnemos)
        os.chmod(socket_path, 0o600)

    def server_bind(self):
        super().server_bind()
        # BaseHTTPRequestHandler formats client_address[0]; AF_UNIX peers
        # report an empty string, give it a stable shape instead.
        self.server_name = "localhost"
        self.server_port = 0

    def get_request(self):
        request, _ = super().get_request()
        return request, ("unix", 0)


def _parse_hostport(value: str) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"expected HOST:PORT, got {value!r}")
    return host or "127.0.0.1", int(port)


def serve(http: str | None = None, unix: str | None = None, mnemos=None) -> None:
    """Blocking entry point for `mnemos serve --http/--unix`."""
    if not http and not unix:
        raise ValueError("serve() needs --http HOST:PORT or --unix PATH")

    if mnemos is None:
        mnemos = build_mnemos()

    # The shared server never exits, so it is the deployment that most needs
    # the idle reaper: a stdio harness reclaims its arena by dying.
    from . import _resource
    _resource.start_idle_reaper(
        log=lambda m: (sys.stderr.write(m + "\n"), sys.stderr.flush()))

    # One-line language-coverage notice at startup. doctor carries the full
    # version; this is for the operator who never runs doctor.
    try:
        from .language import scan_contents, is_english_only
        from .embed import effective_model
        from . import constants as _c
        # Touch the store FIRST: per-store pinning only runs on first store
        # access, and judging effective_model() before that judges the raw
        # default, not what this store will actually use. Epsilon's first
        # 10.33.0 boot warned "embedder is English-only" about a store pinned
        # to e5 for exactly this reason.
        rows = mnemos.store.sample_contents(mnemos.namespace)
        enc_en = is_english_only(effective_model())
        rr_en = _c.DEFAULT_ENABLE_RERANK and is_english_only(_c.RERANKER_MODEL)
        if enc_en or rr_en:
            affected, total = scan_contents(rows)
            if total and affected / total > 0.10:
                which = ("models are" if (enc_en and rr_en) else
                         "reranker is" if rr_en else "embedder is")
                sys.stderr.write(
                    f"Mnemos: {affected}/{total} sampled memories contain "
                    f"non-English content but the configured {which} "
                    "English-only; see `mnemos doctor` for options\n")
                sys.stderr.flush()
    except Exception:
        pass

    servers = []
    if http:
        host, port = _parse_hostport(http)
        if host not in LOCAL_HOSTS and os.environ.get("MNEMOS_HTTP_ALLOW_NONLOCAL") != "1":
            raise ValueError(
                f"refusing to bind non-local address {host!r}; "
                "set MNEMOS_HTTP_ALLOW_NONLOCAL=1 if you really mean it"
            )
        servers.append(MnemosHTTPServer((host, port), mnemos))
    if unix:
        servers.append(MnemosUnixHTTPServer(unix, mnemos))

    for srv in servers:
        if isinstance(srv, MnemosUnixHTTPServer):
            where = srv._socket_path
        else:
            where = "%s:%d" % srv.server_address[:2]
        sys.stderr.write(f"Mnemos MCP server v{__version__} listening on {where} (shared, models load once)\n")
    sys.stderr.flush()

    threads = []
    for srv in servers[1:]:
        t = threading.Thread(target=srv.serve_forever, daemon=True, name="mnemos-http-extra")
        t.start()
        threads.append(t)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.shutdown()
            srv.server_close()
