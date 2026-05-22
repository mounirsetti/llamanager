"""Tests for the SSE streaming proxy in :mod:`llamanager.api_v1`.

The interesting failure mode covered here: the readiness gate releases
the request, but then the upstream inference engine is unreachable when
the proxy actually tries to POST. Previously this leaked an
``httpx.ConnectError`` up through ``StreamingResponse`` after the 200 OK
headers had already been flushed — the client saw a half-open SSE stream
and no usable error message. The proxy must catch this and emit a
structured SSE error event plus a final ``[DONE]`` so clients close
cleanly.
"""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

from llamanager.api_v1 import _stream_with_keepalives
from llamanager.auth import Origin
from llamanager.config import Config
from llamanager.db import DB
from llamanager.queue_mgr import QueueManager, QueuedRequest
from llamanager.server_manager import ServerManager


def _free_port() -> int:
    """Grab a port that's free *now*. The brief race window between this
    call and the test using it is acceptable — we want the connect to
    fail with ECONNREFUSED, and binding a fresh ephemeral port reliably
    produces that on every platform we care about."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _make_cfg(tmp_path: Path, *, llama_server_port: int) -> Config:
    data = tmp_path / "lm"
    data.mkdir()
    (data / "logs").mkdir()
    (data / "models").mkdir()
    return Config(data_dir=data, llama_server_port=llama_server_port)


def _make_origin() -> Origin:
    return Origin(id=1, name="t", priority=50,
                  allowed_models=["*"], is_admin=False, created_at=0.0)


def _make_qr() -> QueuedRequest:
    qr = QueuedRequest(
        request_id="test-rid",
        origin=_make_origin(),
        priority=50,
        model_required=None,
        enqueued_at=0.0,
        seq=0,
    )
    # Skip Phase 1: pretend the dispatcher already gave us our slot.
    qr.ready.set()
    return qr


def _collect_sse(chunks: list[bytes]) -> tuple[list[dict], list[str]]:
    """Parse SSE bytes into (data-event JSON payloads, comment lines)."""
    text = b"".join(chunks).decode("utf-8")
    events: list[dict] = []
    comments: list[str] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                events.append({"__done__": True})
            else:
                events.append(json.loads(payload))
        elif line.startswith(":"):
            comments.append(line)
    return events, comments


def test_stream_with_keepalives_emits_clean_error_when_upstream_unreachable(tmp_path):
    """Engine is dead at proxy time → client receives a structured SSE
    error event and a final [DONE] marker, no unhandled exception."""
    cfg = _make_cfg(tmp_path, llama_server_port=_free_port())
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            qr = _make_qr()
            disconnected = asyncio.Event()
            chunks: list[bytes] = []
            async for chunk in _stream_with_keepalives(
                qm, qr, sm,
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}],
                 "stream": True},
                disconnected,
            ):
                chunks.append(chunk)

            events, _comments = _collect_sse(chunks)
            assert events, "expected at least one SSE event"
            # First (or only) non-[DONE] event should be our structured error.
            err_events = [e for e in events if not e.get("__done__")]
            assert len(err_events) == 1, f"expected 1 error event, got: {events}"
            err = err_events[0]["error"]
            assert err["type"] == "llamanager_upstream_unreachable"
            assert "could not reach inference engine" in err["message"]
            # And the stream must terminate cleanly with [DONE].
            assert events[-1] == {"__done__": True}

        asyncio.run(go())
    finally:
        db.close()
