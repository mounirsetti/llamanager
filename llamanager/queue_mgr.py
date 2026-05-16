"""Origin-aware priority queue + single-concurrency dispatcher.

Spec §4.2 + §6.1. Holds an in-memory priority queue keyed on
(-priority, enqueued_at, seq) so heapq sorts in the right order.
The dispatcher coroutine pulls items, ensures llama-server is running
the right model (triggering a swap if needed), then yields the spec back
to the request handler so it can stream the upstream response itself.

Why dispatcher-yields-to-handler instead of dispatcher-streams-itself:
streaming through three coroutines (client connection, dispatcher,
upstream client) is awkward; instead the dispatcher just *gates* who
gets to use the single upstream slot. The handler does the streaming.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .auth import Origin
from .config import Config
from .db import DB
from .server_manager import ServerManager, StartSpec, resolve_spec

log = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class Cancelled(Exception):
    pass


@dataclass(order=False)
class QueuedRequest:
    request_id: str
    origin: Origin
    priority: int
    model_required: str | None  # bare model id (e.g. "repo/file.gguf") or None
    enqueued_at: float
    seq: int
    profile_required: str | None = None  # optional per-request profile selection
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    # signalled by dispatcher when it's this request's turn:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    # set by dispatcher to indicate model is ready (or to error out)
    error: str | None = None
    status: str = "queued"  # queued|swapping_model|running|done|cancelled|failed
    started_at: float | None = None
    # optional: populated by handler when finished
    finished_at: float | None = None

    def heap_key(self) -> tuple[int, float, int]:
        # higher priority first → negate. Then FIFO by enqueued_at, then seq.
        return (-self.priority, self.enqueued_at, self.seq)


class QueueManager:
    def __init__(self, cfg: Config, db: DB, sm: ServerManager):
        self.cfg = cfg
        self.db = db
        self.sm = sm
        self._heap: list[tuple[tuple[int, float, int], QueuedRequest]] = []
        self._by_id: dict[str, QueuedRequest] = {}
        self._in_flight: dict[str, QueuedRequest] = {}
        self._cancelled_in_heap: int = 0
        self._cv = asyncio.Condition()
        self._seq = itertools.count()
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._paused = False
        self._slot = asyncio.Semaphore(max(1, cfg.max_concurrent))

    # ---- lifecycle ----
    def start(self) -> None:
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        self._dispatcher_task = None

    # ---- enqueue / cancel ----
    async def enqueue(self, *, origin: Origin, model_required: str | None,
                      profile_required: str | None = None,
                      ) -> QueuedRequest:
        active_pending = len(self._heap) - self._cancelled_in_heap
        if active_pending + len(self._in_flight) >= self.cfg.max_queue_depth:
            raise QueueFull("queue is full")
        req = QueuedRequest(
            request_id=str(uuid.uuid4()),
            origin=origin,
            priority=origin.priority,
            model_required=model_required,
            profile_required=profile_required,
            enqueued_at=time.time(),
            seq=next(self._seq),
        )
        self.db.insert_request(
            request_id=req.request_id,
            origin_id=origin.id,
            model=model_required,
            priority=req.priority,
        )
        async with self._cv:
            heapq.heappush(self._heap, (req.heap_key(), req))
            self._by_id[req.request_id] = req
            self._cv.notify_all()
        return req

    def cancel(self, request_id: str) -> bool:
        req = self._by_id.get(request_id)
        if not req:
            return False
        if req.cancel.is_set():
            return True
        req.cancel.set()
        if req.status in ("queued", "swapping_model"):
            if req.status == "queued":
                self._cancelled_in_heap += 1
            req.status = "cancelled"
        if not req.ready.is_set():
            req.ready.set()  # wake the waiter so it can observe cancel
        self.db.update_request_status(req.request_id, "cancelled",
                                      finished_at=time.time())
        self.db.log_event("request_cancelled", {"id": req.request_id})
        return True

    def cancel_all_for_origin(self, origin_id: int) -> int:
        n = 0
        for req in list(self._by_id.values()):
            if req.origin.id == origin_id and req.status in ("queued",):
                if self.cancel(req.request_id):
                    n += 1
        return n

    # ---- pause / resume ----
    def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False
        async with self._cv:
            self._cv.notify_all()

    # ---- introspection ----
    def snapshot(self) -> dict[str, Any]:
        pending: list[dict[str, Any]] = []
        for _, req in sorted(self._heap):
            if req.status == "cancelled":
                continue
            pending.append(_request_public(req))
        in_flight = [_request_public(r) for r in self._in_flight.values()]
        return {
            "depth": len(pending),
            "in_flight": in_flight,
            "pending": pending,
            "paused": self._paused,
        }

    def get(self, request_id: str) -> QueuedRequest | None:
        return self._by_id.get(request_id)

    # ---- handler-side API ----
    async def wait_for_slot(self, req: QueuedRequest) -> None:
        """Called by the request handler. Returns when it's safe to proxy.

        Raises Cancelled if the request was cancelled before its turn.
        Raises RuntimeError if the dispatcher errored out (e.g. swap failed).
        Raises asyncio.TimeoutError if queue_timeout_s is exceeded."""
        timeout = self.cfg.queue_timeout_s if self.cfg.queue_timeout_s > 0 else None
        await asyncio.wait_for(req.ready.wait(), timeout=timeout)
        if req.cancel.is_set():
            raise Cancelled()
        if req.error:
            raise RuntimeError(req.error)

    def mark_in_flight_done(self, req: QueuedRequest, *, error: str | None,
                            cancelled: bool, prompt_tokens: int | None,
                            completion_tokens: int | None) -> None:
        req.finished_at = time.time()
        if cancelled or req.cancel.is_set():
            req.status = "cancelled"
            self.db.update_request_status(req.request_id, "cancelled",
                                          finished_at=req.finished_at,
                                          prompt_tokens=prompt_tokens,
                                          completion_tokens=completion_tokens)
        elif error:
            req.status = "failed"
            self.db.update_request_status(req.request_id, "failed",
                                          finished_at=req.finished_at,
                                          error=error)
        else:
            req.status = "done"
            self.db.update_request_status(req.request_id, "done",
                                          finished_at=req.finished_at,
                                          prompt_tokens=prompt_tokens,
                                          completion_tokens=completion_tokens)
        # Release dispatcher slot only if one was actually acquired (i.e.,
        # the request progressed past "queued" — it was dispatched).
        if req.request_id in self._in_flight or req.status in (
            "running", "swapping_model", "done", "failed",
        ):
            self._slot.release()
        # remove from registry
        self._by_id.pop(req.request_id, None)
        self._in_flight.pop(req.request_id, None)

    # ---- dispatcher ----
    def _pop_next(self) -> QueuedRequest | None:
        """Pop the next request, applying starvation protection.

        If max_wait_s > 0, any request waiting longer than that threshold
        is promoted ahead of higher-priority items (oldest-starved first).
        Cancelled entries are drained as encountered.
        """
        now = time.time()
        max_wait = self.cfg.max_wait_s

        # Check for starved requests when anti-starvation is enabled.
        if max_wait > 0:
            starved_idx: int | None = None
            starved_at: float = now  # track the oldest starved entry
            for i, (_, r) in enumerate(self._heap):
                if r.cancel.is_set() or r.status == "cancelled":
                    continue
                wait = now - r.enqueued_at
                if wait > max_wait and r.enqueued_at < starved_at:
                    starved_at = r.enqueued_at
                    starved_idx = i
            if starved_idx is not None:
                _, req = self._heap[starved_idx]
                # Remove from heap and re-heapify.
                self._heap[starved_idx] = self._heap[-1]
                self._heap.pop()
                if self._heap:
                    heapq.heapify(self._heap)
                return req

        # Normal priority-ordered pop, skipping cancelled entries.
        while self._heap:
            _, req = heapq.heappop(self._heap)
            if req.cancel.is_set() or req.status == "cancelled":
                self._cancelled_in_heap = max(0, self._cancelled_in_heap - 1)
                continue
            return req
        return None

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                async with self._cv:
                    while self._paused or not self._heap:
                        await self._cv.wait()
                    req = self._pop_next()
                    if req is None:
                        continue
                # Acquire slot before deciding on swap.
                await self._slot.acquire()
                try:
                    await self._prepare_and_release(req)
                except Exception as e:
                    log.exception("dispatcher: prepare failed for %s", req.request_id)
                    req.error = str(e)
                    req.status = "failed"
                    req.ready.set()
                    # Don't release slot here — the handler owns slot
                    # release via mark_in_flight_done in its finally block.
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("dispatcher loop crashed; sleeping 1s and continuing")
                await asyncio.sleep(1.0)

    async def _prepare_and_release(self, req: QueuedRequest) -> None:
        """Ensure the right model is loaded, then signal the handler.

        ``req.model_required`` is a bare model id or None. The legacy
        ``profile:foo`` shorthand is no longer accepted; per-request profile
        selection arrives via the X-Llamanager-Profile header path instead.
        """
        wanted = req.model_required
        wanted_profile = getattr(req, "profile_required", None)
        loaded = self.sm.runtime.current_model

        need_swap = False
        target_spec: StartSpec | None = None

        if wanted is None:
            if loaded is None:
                # cold start with configured defaults.
                target_spec = resolve_spec(self.cfg)
                need_swap = True
        else:
            target_spec = resolve_spec(self.cfg, model=wanted, profile=wanted_profile)
            if (loaded != target_spec.model_id
                    or self.sm.runtime.current_profile != target_spec.profile_name):
                need_swap = True

        if need_swap and target_spec is not None:
            req.status = "swapping_model"
            self.db.update_request_status(req.request_id, "swapping_model")
            self.db.log_event("dispatch_swap_required",
                              {"req": req.request_id, "from": loaded,
                               "to": target_spec.model_id})
            t0 = time.time()
            try:
                if self.sm.is_running:
                    await self.sm.swap(target_spec)
                else:
                    await self.sm.start(target_spec)
            except Exception as e:
                req.error = f"swap failed: {e}"
                req.status = "failed"
                self.db.update_request_status(req.request_id, "failed",
                                              error=req.error,
                                              finished_at=time.time())
                req.ready.set()
                return
            self.db.log_event("dispatch_swap_done",
                              {"req": req.request_id,
                               "to": target_spec.model_id,
                               "duration_s": round(time.time() - t0, 2)})

        # Mark in-flight and signal handler.
        self._in_flight[req.request_id] = req
        req.status = "running"
        req.started_at = time.time()
        self.db.update_request_status(req.request_id, "running",
                                      started_at=req.started_at)
        req.ready.set()


def _request_public(req: QueuedRequest | None) -> dict[str, Any] | None:
    if req is None:
        return None
    return {
        "id": req.request_id,
        "origin": req.origin.name,
        "priority": req.priority,
        "model_required": req.model_required,
        "status": req.status,
        "enqueued_at": req.enqueued_at,
        "started_at": req.started_at,
        "finished_at": req.finished_at,
    }
