"""Best-effort identification of *who* sent a request, beyond the origin key.

An origin (API key) is a coarse identity: several apps or machines can share
one key, in which case the activity feed can't tell them apart. This module
extracts the extra signal that's already present on every inbound HTTP request
— peer address, User-Agent — and, for loopback callers, resolves the source
port back to the local process that owns it (PID + name).

Everything here is strictly best-effort: any failure (no client info, a proxy
that hides the peer, psutil without permission to read another user's sockets)
degrades to "we just don't know" rather than raising. It is intentionally
read-only and never blocks the request beyond a short psutil scan, which the
caller runs off the event loop via ``asyncio.to_thread``.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Addresses we treat as "this machine" and therefore worth resolving to a PID.
_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
_UA_MAX = 200  # trim pathological User-Agent headers before they hit the log.


def _peer_addr(request: Any) -> tuple[str | None, int | None]:
    """Return (address, port) of the caller.

    A reverse proxy in front of llamanager hides the real peer behind the
    socket, so honour the standard forwarding headers when present — but in
    that case we deliberately drop the port (it's the proxy's, not the
    client's) so we don't later resolve it to the proxy's PID.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first, None
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip(), None
    client = request.client
    if client:
        return client.host, client.port
    return None, None


def _resolve_local_process(addr: str | None, port: int | None) -> str | None:
    """For a loopback caller, map its source port to "name (pid N)".

    Returns None for non-loopback callers, when psutil lacks permission to
    inspect the owning process (e.g. it belongs to another user), or when no
    matching socket is found (the connection may have already closed).
    """
    if not port or addr not in _LOOPBACK:
        return None
    try:
        import psutil

        # The client's socket has laddr == its own ephemeral source port; that
        # port is unique to the owning process, so a laddr-port match pins it.
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                try:
                    return f"{psutil.Process(conn.pid).name()} (pid {conn.pid})"
                except (psutil.Error, OSError):
                    return f"pid {conn.pid}"
    except (psutil.Error, OSError, ImportError):
        return None
    except Exception:  # pragma: no cover - psutil edge cases vary by platform
        log.debug("local process resolution failed", exc_info=True)
        return None
    return None


async def describe_caller(request: Any) -> dict[str, Any]:
    """Build a compact, JSON-serialisable description of the caller.

    Keys are only present when known: ``addr``, ``port``, ``user_agent``,
    ``process``. Safe to call on every request; the (potentially tens-of-ms)
    psutil scan is offloaded so it never stalls the event loop.
    """
    import asyncio

    addr, port = _peer_addr(request)
    info: dict[str, Any] = {}
    if addr:
        info["addr"] = addr
    if port is not None:
        info["port"] = port
    ua = request.headers.get("user-agent")
    if ua:
        info["user_agent"] = ua[:_UA_MAX]
    # Only loopback callers can be mapped to a local PID; skip the (threaded)
    # psutil scan entirely for everyone else so remote requests pay nothing.
    if port is not None and addr in _LOOPBACK:
        proc = await asyncio.to_thread(_resolve_local_process, addr, port)
        if proc:
            info["process"] = proc
    return info


def format_caller(info: dict[str, Any] | None) -> str:
    """Render a caller dict as a compact parenthetical, e.g.
    ``(127.0.0.1, python3 (pid 4821), openai-python/1.40)``. Returns "" when
    there's nothing useful to show."""
    if not info:
        return ""
    parts: list[str] = []
    addr = info.get("addr")
    if addr:
        parts.append(str(addr))
    proc = info.get("process")
    if proc:
        parts.append(str(proc))
    ua = info.get("user_agent")
    if ua:
        parts.append(str(ua))
    return f"({', '.join(parts)})" if parts else ""
