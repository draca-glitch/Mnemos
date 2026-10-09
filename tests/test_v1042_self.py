"""
v10.42.0: self memory. An agent's memories about itself live in their own
namespace (self:<agent>) beside the user's store, behind MNEMOS_SELF.

Import-time constants are reloaded per test so the switch can be flipped;
search uses search_mode=fts so no embedding model is required.
"""

import importlib
import json
import os
import tempfile
import threading
import urllib.request

import pytest


def _reload(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    # Import-time constants (DEFAULT_SELF, DEFAULT_DB_PATH, embed dims) are
    # baked into several modules; reload them in dependency order so every
    # module sees the same environment, including the store's default path.
    import mnemos.constants as c
    import mnemos.embed as embed
    import mnemos.storage.sqlite_store as sqlite_store
    import mnemos.core as core
    import mnemos.cli as cli
    import mnemos.mcp_server as srv
    import mnemos.http_server as http
    for mod in (c, embed, sqlite_store, core, cli, srv, http):
        importlib.reload(mod)
    return srv


@pytest.fixture
def db_path(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("MNEMOS_DB", path)
    monkeypatch.setenv("MNEMOS_EAGER_WARMUP", "0")
    monkeypatch.setenv("MNEMOS_ENABLE_RERANK", "0")
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def _restore_modules(monkeypatch):
    yield
    _reload(monkeypatch, MNEMOS_SELF=None, MNEMOS_AGENT=None)


def _rpc(method, id_=1, params=None):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _result(response):
    return json.loads(response["result"]["content"][0]["text"])


def _stdio_mnemos(db_path):
    from mnemos.core import Mnemos
    from mnemos.storage.sqlite_store import SQLiteStore
    return Mnemos(store=SQLiteStore(db_path=db_path), namespace="user")


class TestConstants:
    def test_agent_slug_is_namespace_safe(self):
        from mnemos.constants import agent_slug, self_namespace
        assert agent_slug("Claude Code") == "claude-code"
        assert agent_slug("codex-mcp-client") == "codex-mcp-client"
        assert agent_slug("  Grok CLI/1.0.46  ") == "grok-cli-1.0.46"
        assert agent_slug("x" * 80) == "x" * 40
        assert self_namespace("Claude Code") == "self:claude-code"
        with pytest.raises(ValueError):
            self_namespace("***")

    def test_off_by_default(self, monkeypatch):
        srv = _reload(monkeypatch, MNEMOS_SELF=None)
        from mnemos.constants import DEFAULT_SELF
        assert DEFAULT_SELF is False
        for tool in srv.TOOL_DEFINITIONS:
            assert "self" not in tool["inputSchema"]["properties"]


class TestStdio:
    def test_self_flag_refused_when_off(self, monkeypatch, db_path):
        srv = _reload(monkeypatch, MNEMOS_SELF=None)
        mnemos = _stdio_mnemos(db_path)
        try:
            srv.handle_message(mnemos, _rpc("initialize", params={"clientInfo": {"name": "bot"}}))
            response = srv.handle_message(mnemos, _rpc("tools/call", id_=2, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "P:blunt by default", "self": True},
            }))
            assert response["result"]["isError"] is True
            assert "MNEMOS_SELF" in _result(response)["error"]
            assert mnemos.store.count_active("user") == 0
        finally:
            mnemos.close()

    def test_self_namespace_from_client_info(self, monkeypatch, db_path):
        srv = _reload(monkeypatch, MNEMOS_SELF="1")
        assert all("self" in t["inputSchema"]["properties"]
                   for t in srv.TOOL_DEFINITIONS if t["name"] in srv.SELF_TOOLS)
        assert "self" not in next(
            t for t in srv.TOOL_DEFINITIONS if t["name"] == "memory_bulk_rewrite"
        )["inputSchema"]["properties"]
        mnemos = _stdio_mnemos(db_path)
        try:
            srv.handle_message(mnemos, _rpc("initialize", params={"clientInfo": {"name": "Test Bot"}}))
            stored = _result(srv.handle_message(mnemos, _rpc("tools/call", id_=2, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "P:blunt by default, verify first zqx42",
                              "self": True},
            })))
            assert stored["namespace"] == "self:test-bot"
            assert stored["id"]

            # Invisible from the user's store, visible through the self flag.
            plain = _result(srv.handle_message(mnemos, _rpc("tools/call", id_=3, params={
                "name": "memory_search",
                "arguments": {"query": "zqx42", "search_mode": "fts"},
            })))
            assert plain["count"] == 0
            own = _result(srv.handle_message(mnemos, _rpc("tools/call", id_=4, params={
                "name": "memory_search",
                "arguments": {"query": "zqx42", "search_mode": "fts", "self": True},
            })))
            assert own["count"] == 1 and own["namespace"] == "self:test-bot"

            # get/update are namespace-scoped: the id resolves only with the flag.
            missing = _result(srv.handle_message(mnemos, _rpc("tools/call", id_=5, params={
                "name": "memory_get", "arguments": {"id": stored["id"]},
            })))
            assert not missing or missing.get("error") or missing.get("id") != stored["id"]
            got = _result(srv.handle_message(mnemos, _rpc("tools/call", id_=6, params={
                "name": "memory_get", "arguments": {"id": stored["id"], "self": True},
            })))
            assert got["id"] == stored["id"]
            assert mnemos.store.count_active("user") == 0
            assert mnemos.store.count_active("self:test-bot") == 1
        finally:
            mnemos.close()

    def test_env_agent_is_the_fallback(self, monkeypatch, db_path):
        srv = _reload(monkeypatch, MNEMOS_SELF="1", MNEMOS_AGENT="envbot")
        mnemos = _stdio_mnemos(db_path)
        try:
            srv.handle_message(mnemos, _rpc("initialize", params={}))
            stored = _result(srv.handle_message(mnemos, _rpc("tools/call", id_=2, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "L:fallback identity", "self": True},
            })))
            assert stored["namespace"] == "self:envbot"
        finally:
            mnemos.close()

    def test_no_identity_is_an_error(self, monkeypatch, db_path):
        srv = _reload(monkeypatch, MNEMOS_SELF="1", MNEMOS_AGENT=None)
        mnemos = _stdio_mnemos(db_path)
        try:
            srv.handle_message(mnemos, _rpc("initialize", params={}))
            response = srv.handle_message(mnemos, _rpc("tools/call", id_=2, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "L:nobody", "self": True},
            }))
            assert response["result"]["isError"] is True
            assert "identity" in _result(response)["error"]
        finally:
            mnemos.close()


def _post(url, payload, session=None, user_agent=None):
    headers = {"Content-Type": "application/json"}
    if session:
        headers["Mcp-Session-Id"] = session
    if user_agent is not None:
        headers["User-Agent"] = user_agent
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
        return dict(resp.headers), json.loads(body) if body else None


class TestSharedHttp:
    def test_each_client_gets_its_own_self(self, monkeypatch, db_path):
        _reload(monkeypatch, MNEMOS_SELF="1")
        from mnemos.core import Mnemos
        from mnemos.storage.sqlite_store import SQLiteStore
        from mnemos.http_server import MnemosHTTPServer

        mnemos = Mnemos(store=SQLiteStore(db_path=db_path), namespace="user")
        server = MnemosHTTPServer(("127.0.0.1", 0), mnemos)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/"
        try:
            hdr_a, _ = _post(url, _rpc("initialize", params={"clientInfo": {"name": "claude-code"}}))
            hdr_b, _ = _post(url, _rpc("initialize", params={"clientInfo": {"name": "codex"}}))
            sid_a, sid_b = hdr_a["Mcp-Session-Id"], hdr_b["Mcp-Session-Id"]

            _, body = _post(url, _rpc("tools/call", id_=2, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "P:only mine zqx77", "self": True},
            }), session=sid_a)
            assert _result(body)["namespace"] == "self:claude-code"

            _, body = _post(url, _rpc("tools/call", id_=3, params={
                "name": "memory_search",
                "arguments": {"query": "zqx77", "search_mode": "fts", "self": True},
            }), session=sid_b)
            found = _result(body)
            assert found["namespace"] == "self:codex" and found["count"] == 0

            _, body = _post(url, _rpc("tools/call", id_=4, params={
                "name": "memory_search",
                "arguments": {"query": "zqx77", "search_mode": "fts", "self": True},
            }), session=sid_a)
            assert _result(body)["count"] == 1

            # The user's store never sees it.
            _, body = _post(url, _rpc("tools/call", id_=5, params={
                "name": "memory_search",
                "arguments": {"query": "zqx77", "search_mode": "fts"},
            }), session=sid_b)
            assert _result(body)["count"] == 0
        finally:
            server.shutdown()
            server.server_close()
            mnemos.close()

    def test_user_agent_names_a_client_that_never_initialised(self, monkeypatch, db_path):
        """Claude Code keeps calling a restarted server without a new
        initialize and sends no Mcp-Session-Id (captured 2026-10-09); its
        User-Agent product token is the identity then. A session that did
        initialise keeps the name it declared."""
        srv = _reload(monkeypatch, MNEMOS_SELF="1")
        from mnemos.core import Mnemos
        from mnemos.storage.sqlite_store import SQLiteStore
        from mnemos.http_server import MnemosHTTPServer

        assert srv.client_hint_from_user_agent("claude-code/2.1.292 (cli)") == "claude-code"
        assert srv.client_hint_from_user_agent("node") == "node"
        assert srv.client_hint_from_user_agent("") is None
        assert srv.client_hint_from_user_agent(None) is None

        mnemos = Mnemos(store=SQLiteStore(db_path=db_path), namespace="user")
        server = MnemosHTTPServer(("127.0.0.1", 0), mnemos)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/"
        try:
            _, body = _post(url, _rpc("tools/call", id_=2, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "P:named by my user agent zqx88",
                              "self": True},
            }), user_agent="claude-code/2.1.292 (cli)")
            assert _result(body)["namespace"] == "self:claude-code"

            hdr, _ = _post(url, _rpc("initialize", params={"clientInfo": {"name": "codex"}}),
                           user_agent="node")
            _, body = _post(url, _rpc("tools/call", id_=3, params={
                "name": "memory_search",
                "arguments": {"query": "zqx88", "search_mode": "fts", "self": True},
            }), session=hdr["Mcp-Session-Id"], user_agent="node")
            assert _result(body)["namespace"] == "self:codex"

            # urllib would otherwise send Python-urllib/3.x and name a namespace.
            _, body = _post(url, _rpc("tools/call", id_=4, params={
                "name": "memory_store",
                "arguments": {"project": "self", "content": "L:nobody home", "self": True},
            }), user_agent="")
            assert body["result"]["isError"] is True
            assert "identity" in _result(body)["error"]
        finally:
            server.shutdown()
            server.server_close()
            mnemos.close()


class TestCli:
    def test_self_switch_needs_the_feature_and_an_agent(self, monkeypatch, db_path, capsys):
        from mnemos import cli
        _reload(monkeypatch, MNEMOS_SELF=None, MNEMOS_AGENT="cli-bot")
        with pytest.raises(SystemExit):
            cli.main(["--self", "briefing"])
        assert "MNEMOS_SELF" in capsys.readouterr().err

        _reload(monkeypatch, MNEMOS_SELF="1", MNEMOS_AGENT=None)
        with pytest.raises(SystemExit):
            cli.main(["--self", "briefing"])
        assert "MNEMOS_AGENT" in capsys.readouterr().err

    def test_self_briefing_reads_the_self_namespace(self, monkeypatch, db_path, capsys):
        from mnemos import cli
        _reload(monkeypatch, MNEMOS_SELF="1", MNEMOS_AGENT="CLI Bot", MNEMOS_NAMESPACE="user")
        cli.main(["--self", "add", "-p", "self", "P:teasing is the pressure valve zqx99", "-i", "7"])
        capsys.readouterr()
        cli.main(["--self", "briefing"])
        out = capsys.readouterr().out
        assert "self:cli-bot" in out and "zqx99" in out
        cli.main(["briefing"])
        out = capsys.readouterr().out
        assert "zqx99" not in out and "(user)" in out
