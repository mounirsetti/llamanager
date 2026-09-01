"""The stdio proxy: credential resolution and faithful forwarding.

The proxy exists so a host that launches a child process reaches the one
daemon that owns the hardware. Its whole contract is that it adds nothing
of its own — so these tests pin that tools, schemas, annotations and
error flags come back exactly as the daemon sent them.
"""
from __future__ import annotations

import mcp.types as types
import pytest

from llamanager.admin_client import resolve_admin_key, resolve_base_url


def test_base_url_prefers_explicit_then_env_then_config(cfg, monkeypatch):
    monkeypatch.delenv("LLAMANAGER_URL", raising=False)
    cfg.bind, cfg.port = "127.0.0.1", 7200
    assert resolve_base_url(cfg) == "http://127.0.0.1:7200"

    monkeypatch.setenv("LLAMANAGER_URL", "http://box.local:9000/")
    assert resolve_base_url(cfg) == "http://box.local:9000"
    # An explicit flag still wins over the environment.
    assert resolve_base_url(cfg, "http://other:1/") == "http://other:1"


def test_admin_key_falls_back_to_the_local_control_key(cfg, monkeypatch):
    """Same-box stdio must work with no configuration at all."""
    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    (cfg.data_dir / ".local-control-key").write_text("lm_localkey")
    assert resolve_admin_key(cfg) == "lm_localkey"

    monkeypatch.setenv("LLAMANAGER_ADMIN_KEY", "lm_fromenv")
    assert resolve_admin_key(cfg) == "lm_fromenv"
    assert resolve_admin_key(cfg, "lm_explicit") == "lm_explicit"


def test_missing_key_is_an_error_not_a_guess(cfg, monkeypatch):
    from llamanager.admin_client import AdminClientError

    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    key_file = cfg.data_dir / ".local-control-key"
    if key_file.exists():
        key_file.unlink()
    with pytest.raises(AdminClientError) as e:
        resolve_admin_key(cfg)
    # The message has to name the ways out, since this is what a user sees
    # when their MCP host says the server failed to start.
    assert "LLAMANAGER_ADMIN_KEY" in str(e.value)


class _FakeUpstream:
    """Stands in for the daemon's MCP session."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return types.ListToolsResult(tools=[
            types.Tool(
                name="server_status",
                description="status",
                input_schema={"type": "object", "properties": {}},
                annotations=types.ToolAnnotations(read_only_hint=True),
            )
        ])

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "boom":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="it failed")],
                is_error=True)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="ok")],
            structured_content={"echo": args})


def _handlers(upstream):
    """The two forwarding handlers, built the way _serve builds them."""
    async def list_tools(_ctx, _params):
        return await upstream.list_tools()

    async def call_tool(_ctx, params):
        return await upstream.call_tool(params.name, params.arguments or {})

    return list_tools, call_tool


@pytest.mark.asyncio
async def test_tools_are_forwarded_with_schema_and_annotations():
    upstream = _FakeUpstream()
    list_tools, _ = _handlers(upstream)
    result = await list_tools(None, types.PaginatedRequestParams())
    assert [t.name for t in result.tools] == ["server_status"]
    # Annotations must survive: they drive approval prompts in the host.
    assert result.tools[0].annotations.read_only_hint is True
    assert result.tools[0].input_schema == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_arguments_and_results_pass_through_unchanged():
    upstream = _FakeUpstream()
    _, call_tool = _handlers(upstream)
    params = types.CallToolRequestParams(name="load_model",
                                         arguments={"model": "a/b.gguf"})
    result = await call_tool(None, params)
    assert upstream.calls == [("load_model", {"model": "a/b.gguf"})]
    assert result.structured_content == {"echo": {"model": "a/b.gguf"}}


@pytest.mark.asyncio
async def test_tool_errors_stay_errors():
    """A failed tool must not be forwarded as a success."""
    upstream = _FakeUpstream()
    _, call_tool = _handlers(upstream)
    result = await call_tool(None, types.CallToolRequestParams(name="boom"))
    assert result.is_error is True
    assert "it failed" in result.content[0].text


def test_unreachable_daemon_exits_with_a_pointed_message(capsys):
    """No daemon, no silent retry loop — say what to start."""
    from llamanager.mcp_stdio import run_stdio_proxy

    # Port 1 is not a llamanager; the connect attempt fails immediately.
    code = run_stdio_proxy("http://127.0.0.1:1", "lm_whatever")
    assert code == 1
    err = capsys.readouterr().err
    assert "cannot reach llamanager" in err
    assert "llamanager serve" in err
