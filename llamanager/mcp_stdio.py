"""Serve llamanager's MCP tools over stdio by proxying the running daemon.

Some hosts (Claude Desktop, and anything installed from an ``.mcpb``
bundle) launch an MCP server as a child process and talk to it on stdin
and stdout. They cannot dial an HTTP endpoint themselves.

This module bridges that gap without becoming a second llamanager: it
opens one Streamable HTTP connection to the daemon that is already
running — the process that owns the GPU, the queue, the model slots and
the database — and forwards ``tools/list`` and ``tools/call`` verbatim in
both directions. Building a second app here instead would load a second
copy of that state and fight the first one for the hardware.

Forwarding is deliberately dumb. Tool schemas, descriptions, annotations
and errors are whatever the daemon says they are, so a tool added to
``mcp_server.py`` needs no change here.
"""

from __future__ import annotations

import logging
import sys

import anyio
import httpx2
import mcp.types as types
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

log = logging.getLogger("llamanager.mcp.stdio")


class ProxyUnavailable(RuntimeError):
    """The daemon could not be reached — reported, never retried silently."""


async def _serve(base_url: str, key: str) -> None:
    # stdout is the protocol channel and stderr is the host's log pane; a
    # per-request INFO line for every tool call helps nobody there.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    endpoint = f"{base_url.rstrip('/')}/mcp/"
    http = httpx2.AsyncClient(
        base_url=base_url,
        headers={"authorization": f"Bearer {key}"},
        # A generation tool call legitimately blocks for many minutes; the
        # daemon, not this proxy, decides when a call has taken too long.
        timeout=httpx2.Timeout(None),
    )
    async with http:
        try:
            transport = streamable_http_client(endpoint, http_client=http)
            client_cm = Client(transport)
            upstream = await client_cm.__aenter__()
        except Exception as e:  # noqa: BLE001 — turned into a clear message
            raise ProxyUnavailable(
                f"cannot reach llamanager at {endpoint}: {e}\n"
                f"Start the daemon first (`llamanager serve`, or the "
                f"llamanager user service), and check the key is valid."
            ) from e

        try:
            from . import __version__
            proxy: Server = Server("llamanager", version=__version__)

            # ``ServerResult`` is a union alias in the 2.x types, so the
            # upstream result object is returned as-is — which is also what
            # keeps this proxy faithful: schemas, annotations and error
            # flags are whatever the daemon produced.
            async def _list_tools(_ctx, _params: types.PaginatedRequestParams
                                  ) -> types.ListToolsResult:
                return await upstream.list_tools()

            async def _call_tool(_ctx, params: types.CallToolRequestParams
                                 ) -> types.CallToolResult:
                return await upstream.call_tool(
                    params.name, params.arguments or {})

            proxy.add_request_handler(
                "tools/list", types.PaginatedRequestParams, _list_tools)
            proxy.add_request_handler(
                "tools/call", types.CallToolRequestParams, _call_tool)

            async with stdio_server() as (read_stream, write_stream):
                await proxy.run(
                    read_stream,
                    write_stream,
                    proxy.create_initialization_options(),
                )
        finally:
            await client_cm.__aexit__(None, None, None)


def run_stdio_proxy(base_url: str, key: str) -> int:
    """Blocking entry point. Returns a process exit code."""
    try:
        anyio.run(_serve, base_url, key)
    except ProxyUnavailable as e:
        print(str(e), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0
