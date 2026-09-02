"""Retry a transcription only when the worker process vanished.

ROCm ships no MIOpen kernel database for gfx1201, so the first inference
ever run for a given model's tensor shapes compiles kernels on the spot,
and that path can take the worker down natively — the connection closes
with no reply and no traceback. A fresh worker then finds the compiled
kernels in the MIOpen cache and succeeds. Measured once the shapes are
cached: 8 cold starts, 8 successes.

That makes one retry the right shape of fix. It also makes it dangerous:
a retry that catches too much turns a repeatable failure into a silent
one. These tests exist mostly to pin how *narrow* it is.
"""
from __future__ import annotations

import httpx
import pytest

from llamanager.audio_runner import AudioError, AudioWorkerDied


class _Runner:
    """The retry loop, lifted onto a stub so it can be driven directly."""

    def __init__(self, proxy_results):
        from llamanager.audio_runner import AudioTaskRunner
        import asyncio

        self._results = list(proxy_results)
        self.ensured = 0
        self.stopped = 0
        self._start_lock = asyncio.Lock()
        self._ensure_and_proxy = AudioTaskRunner._ensure_and_proxy.__get__(self)

    async def _ensure_worker(self, model_id, model_path, engine):
        self.ensured += 1

    async def _stop_worker_locked(self):
        self.stopped += 1

    async def _proxy(self, *a, **kw):
        outcome = self._results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def run(self):
        return await self._ensure_and_proxy(
            "m", "/path", "asr", None, None, "req-1", False)


def _died():
    return AudioWorkerDied("the ASR worker died while transcribing (ReadError)")


@pytest.mark.asyncio
async def test_a_dead_worker_is_retried_once_with_a_fresh_worker():
    r = _Runner([_died(), "transcript"])
    assert await r.run() == "transcript"
    assert r.ensured == 2, "the retry must start a new worker"
    assert r.stopped == 1, "the dead worker must be torn down first"


@pytest.mark.asyncio
async def test_a_second_death_is_raised_not_retried_forever():
    r = _Runner([_died(), _died()])
    with pytest.raises(AudioWorkerDied):
        await r.run()
    assert r.ensured == 2, "exactly one retry, not a loop"


@pytest.mark.asyncio
async def test_a_worker_that_answers_with_an_error_is_not_retried():
    """The engine told us something; repeating it just fails again."""
    r = _Runner([AudioError("model failed to load: no such file"), "unused"])
    with pytest.raises(AudioError) as e:
        await r.run()
    assert "no such file" in str(e.value)
    assert r.ensured == 1
    assert r.stopped == 0


@pytest.mark.asyncio
async def test_an_unexpected_exception_is_not_swallowed():
    r = _Runner([RuntimeError("something else entirely"), "unused"])
    with pytest.raises(RuntimeError):
        await r.run()
    assert r.ensured == 1


@pytest.mark.asyncio
async def test_the_happy_path_starts_exactly_one_worker():
    r = _Runner(["transcript"])
    assert await r.run() == "transcript"
    assert (r.ensured, r.stopped) == (1, 0)


def test_transport_errors_become_a_named_worker_death():
    """httpx transport errors stringify to "", which is how this first
    surfaced as a bare "502:" with no cause. The class name has to survive."""
    for exc in (httpx.ReadError(""), httpx.RemoteProtocolError(""),
                httpx.ConnectError("")):
        assert str(exc) == "", "precondition: these carry no message"
        assert isinstance(exc, httpx.TransportError)

    err = AudioWorkerDied(f"the ASR worker died while transcribing "
                          f"({type(httpx.ReadError('')).__name__})")
    assert "ReadError" in str(err)
    # It must still read as an AudioError to every existing handler.
    assert isinstance(err, AudioError)
