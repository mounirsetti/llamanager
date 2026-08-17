"""The image queue slot is owned by the engine run, not by the response.

Regression cover for the orphan bug:

    A browser dropped its SSE connection mid-generation (Cancel button,
    page reload, tab close). ``_images_stream``'s ``finally`` ran on
    GeneratorExit and called ``mark_in_flight_done``, which refunded the
    image slot and stamped the DB row ``done`` — while the diffusion
    engine kept running and holding ~26 GB of VRAM. With the counter back
    at zero the dispatcher started llama-server on top of it and filled
    the card; the job was invisible in the queue the whole time.

The invariant these tests pin: the slot follows the *engine*, never the
HTTP connection. A dropped client changes nothing — the job keeps its
slot and runs to completion (so a page refresh doesn't throw away a
half-finished generation; the page reattaches via /ui/images/status).
Stopping early is an explicit act, and only an explicit cancel —
``qr.cancel``, as set by POST /admin/queue/{id}/cancel — stops the
engine.

No subprocesses or HTTP servers here — the runner is stubbed and drives
the cancel handshake through ``cancel_event``, exactly as the real one
does via ImageTaskRunner._watch_cancel.
"""
from __future__ import annotations

import asyncio

from llamanager import api_v1
from llamanager.api_v1 import _images_stream
from llamanager.engines._base import ImageRequest
from llamanager.image_runner import ImageError


def _image_req() -> ImageRequest:
    return ImageRequest(prompt="a cat", width=64, height=64, steps=4,
                        seed=1, n=1, ref_images=[])


class _FakeOrigin:
    name = "test-origin"


class _FakeQR:
    """The slice of QueuedRequest that _images_stream touches."""

    def __init__(self) -> None:
        self.request_id = "req-1"
        self.origin = _FakeOrigin()
        self.status = "running"
        self.error = None
        self.ready = asyncio.Event()
        self.cancel = asyncio.Event()
        self.incognito = False


class _FakeQM:
    def __init__(self) -> None:
        self.done_calls: list[dict] = []

    def position_for(self, qr) -> int:
        return -1

    def mark_in_flight_done(self, req, **kw) -> None:
        self.done_calls.append(kw)


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


class _FakeRunner:
    """A diffusion engine mid-denoise: finishes when told to.

    ``release`` stands in for the engine completing normally; the
    ``cancel_event`` path mirrors ImageTaskRunner._watch_cancel turning
    an explicit cancel into a SIGTERM.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = False
        self.cancelled = False

    async def run(self, **kw):
        self.started.set()
        cancel = kw["cancel_event"]
        waits = [asyncio.create_task(cancel.wait()),
                 asyncio.create_task(self.release.wait())]
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waits:
                w.cancel()
        self.stopped = True
        if cancel.is_set():
            self.cancelled = True
            raise ImageError("cancelled")
        raise ImageError("done-but-stubbed")


def _start(qm, qr, runner):
    return _images_stream(
        qm, qr, runner, _FakeRequest(), _image_req(),
        model_required="img-engine", engine="fake", profile_obj=None,
        response_format="b64_json",
    )


async def _drain_until_refunded(qm, limit: int = 300) -> None:
    for _ in range(limit):
        if qm.done_calls:
            return
        await asyncio.sleep(0.01)


def test_client_disconnect_keeps_the_job_and_its_slot():
    """A refresh/close must not cancel the job *or* free the GPU slot."""

    async def go():
        qm, qr, runner = _FakeQM(), _FakeQR(), _FakeRunner()
        qr.ready.set()                      # slot already granted
        resp = await _start(qm, qr, runner)
        body = resp.body_iterator

        # Pull the first event; the engine task is now live.
        chunk = await body.__anext__()
        assert b"status=loading" in chunk
        await asyncio.wait_for(runner.started.wait(), timeout=2)
        assert qm.done_calls == []

        # The client goes away: Starlette closes the response generator.
        await body.aclose()
        await asyncio.sleep(0.05)

        assert not qr.cancel.is_set(), \
            "a dropped connection is not a cancel — the page can reattach"
        assert not runner.stopped, "the engine must keep running"
        assert qm.done_calls == [], \
            "the slot stays taken while the engine holds the GPU — this is " \
            "the orphan bug: refunding here let llama-server start on top"

        # It finishes on its own; only then is the slot refunded.
        runner.release.set()
        await _drain_until_refunded(qm)
        assert len(qm.done_calls) == 1, \
            "the slot is refunded exactly once, by the engine run"
        assert qm.done_calls[0]["cancelled"] is False

    asyncio.run(go())


def test_explicit_cancel_stops_the_engine_and_then_refunds():
    """qr.cancel (POST /admin/queue/{id}/cancel) is the way to stop."""

    async def go():
        qm, qr, runner = _FakeQM(), _FakeQR(), _FakeRunner()
        qr.ready.set()
        resp = await _start(qm, qr, runner)
        body = resp.body_iterator
        await body.__anext__()
        await asyncio.wait_for(runner.started.wait(), timeout=2)

        qr.cancel.set()                     # what the admin route does
        await _drain_until_refunded(qm)

        assert runner.cancelled, "explicit cancel must reach the engine"
        assert len(qm.done_calls) == 1
        assert qm.done_calls[0]["cancelled"] is True

        await body.aclose()

    asyncio.run(go())


def test_slot_refunded_once_on_the_normal_path():
    """Happy path still finalises exactly once, from the run task."""

    class _OkRunner:
        async def run(self, **kw):
            raise ImageError("boom")       # terminal, but not a cancel

    async def go():
        qm, qr = _FakeQM(), _FakeQR()
        qr.ready.set()
        resp = await _start(qm, qr, _OkRunner())
        chunks = [c async for c in resp.body_iterator]

        assert any(b"boom" in c for c in chunks)
        assert len(qm.done_calls) == 1
        assert qm.done_calls[0]["cancelled"] is False
        assert qm.done_calls[0]["error"] == "boom"

    asyncio.run(go())


def test_abandoned_while_still_queued_refunds_immediately():
    """Dying before the engine starts must still release the slot —
    nothing else is running that could do it."""

    class _NeverRuns:
        async def run(self, **kw):        # pragma: no cover - not reached
            raise AssertionError("engine must not start")

    async def go():
        qm, qr = _FakeQM(), _FakeQR()     # ready never set: still queued
        resp = await _start(qm, qr, _NeverRuns())
        body = resp.body_iterator
        chunk = await body.__anext__()    # keepalive / status while queued
        assert chunk.startswith(b":")
        await body.aclose()

        assert len(qm.done_calls) == 1, \
            "a queued request that unwinds must refund its own slot"

    # The queued branch only emits on the keepalive tick; don't sit
    # through the production 10 s for it.
    original = api_v1.KEEPALIVE_INTERVAL_S
    api_v1.KEEPALIVE_INTERVAL_S = 0.05
    try:
        asyncio.run(go())
    finally:
        api_v1.KEEPALIVE_INTERVAL_S = original
