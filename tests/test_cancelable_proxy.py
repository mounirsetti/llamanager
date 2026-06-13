"""Tests for cancellable upstream proxying in :mod:`llamanager.api_v1`.

Regression coverage for the bug where in-flight text requests ignored
cancellation: a bare ``await client.post(..., timeout=None)`` blocked until
the engine returned the *whole* completion, and the streaming path only
checked the cancel flag between received chunks — so a long prompt prefill
(no bytes flowing yet) was completely uninterruptible. Both now race the
upstream call against ``qr.cancel`` and abort promptly.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from llamanager.api_v1 import _cancelable_post, _iter_until_cancel
from llamanager.queue_mgr import Cancelled


class _BlockingClient:
    """Stand-in for ``httpx.AsyncClient`` whose ``post`` never returns.

    Models an engine that has accepted the request and is grinding away
    (prefill/generation) without producing a response yet. ``aborted`` flips
    when the in-flight POST task is cancelled, proving the connection would
    be torn down.
    """

    def __init__(self) -> None:
        self.aborted = False

    async def post(self, url, json=None):  # noqa: A002 - mirror httpx signature
        try:
            await asyncio.Event().wait()  # block forever
        except asyncio.CancelledError:
            self.aborted = True
            raise
        raise AssertionError("unreachable")


class _FakeQR:
    def __init__(self) -> None:
        self.cancel = asyncio.Event()


def test_cancelable_post_aborts_when_cancel_fires():
    async def go():
        client = _BlockingClient()
        qr = _FakeQR()

        async def fire_cancel():
            await asyncio.sleep(0.01)
            qr.cancel.set()

        with pytest.raises(Cancelled):
            await asyncio.gather(
                _cancelable_post(client, "http://x/v1/chat/completions",
                                 {"messages": []}, qr),
                fire_cancel(),
            )
        # The blocked POST task was cancelled → connection would close and
        # the engine would stop.
        assert client.aborted is True

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_cancelable_post_returns_response_when_not_cancelled():
    async def go():
        sent = {}

        class _OKClient:
            async def post(self, url, json=None):  # noqa: A002
                sent["url"] = url
                return httpx.Response(200, json={"ok": True})

        qr = _FakeQR()
        r = await _cancelable_post(_OKClient(), "http://x/v1/c", {"a": 1}, qr)
        assert r.status_code == 200
        assert sent["url"] == "http://x/v1/c"

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_cancelable_post_propagates_upstream_error():
    async def go():
        class _BoomClient:
            async def post(self, url, json=None):  # noqa: A002
                raise httpx.ConnectError("refused")

        with pytest.raises(httpx.ConnectError):
            await _cancelable_post(_BoomClient(), "http://x", {}, _FakeQR())

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_iter_until_cancel_interrupts_prefill_stall():
    """The iterator blocks (no bytes yet) — cancel must still stop it."""
    async def go():
        cancel = asyncio.Event()
        reached_anext = asyncio.Event()
        aborted = {"value": False}

        async def stalling_stream():
            reached_anext.set()
            try:
                await asyncio.Event().wait()  # prefill: never yields
            except asyncio.CancelledError:
                aborted["value"] = True
                raise
            yield b"unreachable"

        async def consume():
            chunks = []
            async for chunk in _iter_until_cancel(stalling_stream(), cancel):
                chunks.append(chunk)
            return chunks

        task = asyncio.ensure_future(consume())
        await reached_anext.wait()
        cancel.set()
        chunks = await task
        assert chunks == []          # nothing was yielded
        assert aborted["value"] is True  # the blocked read was aborted

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_iter_until_cancel_passes_chunks_through_until_done():
    async def go():
        cancel = asyncio.Event()

        async def stream():
            for b in (b"a", b"b", b"c"):
                yield b

        got = [c async for c in _iter_until_cancel(stream(), cancel)]
        assert got == [b"a", b"b", b"c"]

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_iter_until_cancel_stops_mid_stream_when_cancel_fires():
    async def go():
        cancel = asyncio.Event()

        async def stream():
            yield b"a"
            cancel.set()          # cancel arrives after the first chunk
            yield b"b"
            yield b"c"

        got = [c async for c in _iter_until_cancel(stream(), cancel)]
        # First chunk delivered; cancel observed before the next fetch
        # completes, so iteration stops early.
        assert got == [b"a"]

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))
