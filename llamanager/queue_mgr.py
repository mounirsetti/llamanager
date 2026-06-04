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
from .caller import format_caller
from .config import Config, ENGINE_FAMILY, detect_engine_for_id
from .db import DB
from .server_manager import ServerManager, StartSpec, resolve_spec

log = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class Cancelled(Exception):
    pass


# Upper bound on persisted prompt/response text. Generous enough that no
# real chat turn is ever cut, but a guard against a pathological context
# (e.g. a megabyte-long paste) bloating the requests table.
_MAX_STORED_TEXT = 200_000


def _clip_text(s: str | None, limit: int = _MAX_STORED_TEXT) -> str | None:
    if s is None:
        return None
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n… [truncated {len(s) - limit} more chars]"


@dataclass(order=False)
class QueuedRequest:
    request_id: str
    origin: Origin
    priority: int
    model_required: str | None  # bare model id (e.g. "repo/file.gguf") or None
    enqueued_at: float
    seq: int
    profile_required: str | None = None  # optional per-request profile selection
    # "text" or "image" — set at enqueue time based on the model's engine
    # family. Text requests proxy through llama-server; image requests
    # are dispatched to ImageTaskRunner.
    task_type: str = "text"
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    # signalled by dispatcher when it's this request's turn:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    # set by dispatcher to indicate model is ready (or to error out)
    error: str | None = None
    status: str = "queued"  # queued|swapping_model|running|done|cancelled|failed
    started_at: float | None = None
    # optional: populated by handler when finished
    finished_at: float | None = None
    # True once the dispatcher has counted this request against the
    # family in-flight quota. ``mark_in_flight_done`` decrements only
    # when this is set, so early-abort paths (failures before dispatch)
    # don't leak capacity.
    dispatched: bool = False
    # Multi-slot routing: set by the dispatcher to the slot id that
    # owns this request's upstream. None means "use the pool default
    # (slot 0)" — which is also the single-slot behaviour.
    slot_id: int | None = None

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
        # The condition is signalled when:
        #   * a new request enters the heap (enqueue, resume)
        #   * an in-flight request completes (mark_in_flight_done)
        #   * a request is cancelled (cancel)
        # The dispatcher waits on it while no eligible work exists.
        self._cv = asyncio.Condition()
        self._seq = itertools.count()
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._paused = False
        # Family-aware in-flight counters. Text has cfg.max_concurrent
        # capacity (typically 1); image has a fixed 1-slot ceiling because
        # diffusion generations are heavy and the runner already serializes
        # them internally via its own lock. When cfg.allow_concurrent is
        # False (the default), the two families are mutually exclusive:
        # no image starts while text is in flight and vice versa.
        self._in_flight_count: dict[str, int] = {"text": 0, "image": 0}
        # Monotonic timestamp of the last time the queue had work — bumped on
        # enqueue and on each in-flight completion. Read by ``idle_seconds()``
        # for the auto-update-when-idle scheduler (see auto_update.AutoUpdater).
        self._last_busy_monotonic: float = time.monotonic()
        # Set by the memory watchdog (request_reclaim) when host memory gets
        # tight. Consumed at the next task boundary in _prepare_and_release to
        # reset the engine to baseline — a between-task reclaim clears the
        # accumulated prompt cache / KV checkpoints without interrupting any
        # live generation. See mem_guard.MemoryWatchdog.
        self._reclaim_pending: bool = False

    def request_reclaim(self) -> None:
        """Ask the dispatcher to reset the engine before the next task runs."""
        self._reclaim_pending = True

    # ---- idle detection ----
    def idle_seconds(self) -> float:
        """Seconds the queue has had no work, or ``0.0`` if anything is active.

        "Active" means any in-flight request (text or image) or any
        non-cancelled pending entry in the heap. While active, returns 0.0.
        Once fully drained, returns how long it's been since the last enqueue
        or completion. Used to gate idle-only background work.
        """
        if self._in_flight:
            return 0.0
        if (len(self._heap) - self._cancelled_in_heap) > 0:
            return 0.0
        return max(0.0, time.monotonic() - self._last_busy_monotonic)

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
                      task_type: str | None = None,
                      caller: dict[str, Any] | None = None,
                      ) -> QueuedRequest:
        active_pending = len(self._heap) - self._cancelled_in_heap
        if active_pending + len(self._in_flight) >= self.cfg.max_queue_depth:
            raise QueueFull("queue is full")
        if task_type is None:
            task_type = _infer_task_type(self.cfg, model_required)
        req = QueuedRequest(
            request_id=str(uuid.uuid4()),
            origin=origin,
            priority=origin.priority,
            model_required=model_required,
            profile_required=profile_required,
            task_type=task_type,
            enqueued_at=time.time(),
            seq=next(self._seq),
        )
        self.db.insert_request(
            request_id=req.request_id,
            origin_id=origin.id,
            model=model_required,
            priority=req.priority,
        )
        # Surface the inbound request on the activity feed *and* in the
        # raw log file. The text family carries actual user prompts; the
        # image family carries diffusion engine requests. Either way we
        # want a "received" beat plus the matching "done" / "failed" beat
        # from mark_in_flight_done.
        self.db.log_event("request_received", {
            "id": req.request_id,
            "task_type": req.task_type,
            "model": model_required,
            "profile": profile_required,
            "origin": origin.name,
            # Best-effort caller identity (peer addr, User-Agent, local PID).
            # Absent/empty when the source can't be determined.
            "caller": caller or None,
        })
        label = "chat" if req.task_type == "text" else req.task_type
        target = f"`{model_required}`" if model_required else "default model"
        caller_tail = f" {format_caller(caller)}" if caller else ""
        log.info("%s: request from %s%s for %s (id=%s)",
                 label, origin.name, caller_tail, target, req.request_id)
        self._last_busy_monotonic = time.monotonic()
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
        was_queued = req.status == "queued"
        req.cancel.set()
        if req.status in ("queued", "swapping_model"):
            if was_queued:
                self._cancelled_in_heap += 1
            req.status = "cancelled"
        if not req.ready.is_set():
            req.ready.set()  # wake the waiter so it can observe cancel
        self.db.update_request_status(req.request_id, "cancelled",
                                      finished_at=time.time())
        self.db.log_event("request_cancelled", {"id": req.request_id})
        # Pre-dispatch cancellations have no handler waiting in the
        # ``mark_in_flight_done`` finally block to clean up, so drop the
        # bookkeeping here. The heap tombstone is removed lazily by
        # _pop_next the next time it reaches the top.
        if was_queued and not req.dispatched:
            self._by_id.pop(req.request_id, None)
        # Wake the dispatcher: a cancelled high-priority entry no longer
        # blocks any starvation-window calculations, and another family
        # may now be eligible.
        self._wake_dispatcher_soon()
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

    def position_for(self, req: QueuedRequest) -> int:
        """Position of ``req`` in the pending queue, 0-indexed.

        ``0`` means ``req`` is next up; ``2`` means two other pending
        requests will be dispatched before it; ``-1`` means ``req`` is
        no longer in pending (already dispatched, in flight, or done).

        We count across all families — the dashboard cares about how
        many requests sit ahead, not which family they belong to. With
        ``allow_concurrent=False`` (the default) text and image
        contend for the same hardware slot anyway, so a single
        ordering reflects reality.
        """
        if req.request_id not in self._by_id:
            return -1
        # _heap stores (heap_key, qr) tuples; sort by key to mirror the
        # order the dispatcher will pop them in.
        ahead = 0
        for _, other in sorted(self._heap, key=lambda kv: kv[0]):
            if other.request_id == req.request_id:
                return ahead
            if other.status == "cancelled":
                continue
            ahead += 1
        return -1

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
                            completion_tokens: int | None,
                            prompt_text: str | None = None,
                            response_text: str | None = None) -> None:
        req.finished_at = time.time()
        # Reset the idle clock so a freshly-drained queue starts counting its
        # quiet window from this completion (see idle_seconds()).
        self._last_busy_monotonic = time.monotonic()
        duration_s = req.finished_at - req.enqueued_at
        # Persist the prompt/response so the UI's request-detail view can
        # show them. Only write columns we actually captured — image jobs
        # and the Anthropic streaming path leave them None, which must not
        # clobber anything (the row had NULL there anyway). Even a partial
        # response on a cancel/error is worth keeping. Skip entirely when the
        # operator set conversation retention to 0 (no content on disk).
        text_fields: dict[str, Any] = {}
        if getattr(self.cfg, "conversation_retention_days", 0) > 0:
            pt = _clip_text(prompt_text)
            if pt is not None:
                text_fields["prompt_text"] = pt
            rt = _clip_text(response_text)
            if rt is not None:
                text_fields["response_text"] = rt
        if cancelled or req.cancel.is_set():
            req.status = "cancelled"
            self.db.update_request_status(req.request_id, "cancelled",
                                          finished_at=req.finished_at,
                                          prompt_tokens=prompt_tokens,
                                          completion_tokens=completion_tokens,
                                          **text_fields)
        elif error:
            req.status = "failed"
            self.db.update_request_status(req.request_id, "failed",
                                          finished_at=req.finished_at,
                                          error=error,
                                          **text_fields)
        else:
            req.status = "done"
            self.db.update_request_status(req.request_id, "done",
                                          finished_at=req.finished_at,
                                          prompt_tokens=prompt_tokens,
                                          completion_tokens=completion_tokens,
                                          **text_fields)
        # Mirror the terminal status onto the activity feed.
        self.db.log_event("request_done", {
            "id": req.request_id,
            "task_type": req.task_type,
            "model": req.model_required,
            "origin": req.origin.name,
            "status": req.status,
            "duration_s": duration_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
        })
        # Same beat, but written through Python logging so it lands in
        # the raw llamanager.log file (which doesn't tail the DB events
        # table). Activity-view rendering still happens from the DB row.
        label = "chat" if req.task_type == "text" else req.task_type
        if req.status == "done":
            if completion_tokens and duration_s > 0:
                log.info("%s: response sent — %d tokens in %.1fs (%.1f tok/s) [id=%s]",
                         label, completion_tokens, duration_s,
                         completion_tokens / duration_s, req.request_id)
            elif completion_tokens:
                log.info("%s: response sent — %d tokens [id=%s]",
                         label, completion_tokens, req.request_id)
            else:
                log.info("%s: response sent in %.1fs [id=%s]",
                         label, duration_s, req.request_id)
        elif req.status == "cancelled":
            log.info("%s: cancelled after %.1fs [id=%s]",
                     label, duration_s, req.request_id)
        else:  # failed
            log.warning("%s: failed — %s [id=%s]",
                        label, error or "unknown", req.request_id)
        # Family-aware capacity release. We only refund the slot when this
        # request was actually counted as in-flight; the early-abort
        # paths (ref-image decode failure, profile not found, etc.) call
        # mark_in_flight_done before dispatch and must not free a slot
        # they never took.
        if req.dispatched:
            family = req.task_type if req.task_type in self._in_flight_count else "text"
            self._in_flight_count[family] = max(
                0, self._in_flight_count[family] - 1,
            )
            req.dispatched = False
            # Wake the dispatcher in case freeing this slot makes another
            # waiting request eligible (especially across families when
            # allow_concurrent=false).
            self._wake_dispatcher_soon()
        # remove from registry
        self._by_id.pop(req.request_id, None)
        self._in_flight.pop(req.request_id, None)

    def _wake_dispatcher_soon(self) -> None:
        """Fire-and-forget notify on the dispatcher condition. Called from
        sync contexts (mark_in_flight_done) where we can't ``async with``
        the underlying lock directly."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # not in an event loop (test setup, shutdown)
        loop.create_task(self._notify_cv())

    async def _notify_cv(self) -> None:
        async with self._cv:
            self._cv.notify_all()

    # ---- dispatcher ----
    def _can_dispatch(self, req: QueuedRequest) -> bool:
        """Family-aware eligibility check. Honours:

          * ``cfg.max_concurrent`` for the text family
          * a hard 1-slot ceiling for image (heavy generations; the runner
            also has its own internal lock as a backstop)
          * ``cfg.allow_concurrent`` — when False, text and image are
            mutually exclusive (single-slot dashboard invariant); when
            True, they can run in parallel up to their per-family caps.
        """
        text_in = self._in_flight_count.get("text", 0)
        image_in = self._in_flight_count.get("image", 0)
        if req.task_type == "image":
            if image_in >= 1:
                return False
            if not self.cfg.allow_concurrent and text_in > 0:
                return False
            return True
        # text
        if text_in >= max(1, self.cfg.max_concurrent):
            return False
        if not self.cfg.allow_concurrent and image_in > 0:
            return False
        return True

    def _pop_next(self) -> QueuedRequest | None:
        """Pick the next dispatchable request, applying starvation
        protection AND family-aware eligibility.

        Returns ``None`` when nothing currently in the heap can be
        dispatched (either the heap is empty or every remaining entry's
        family is blocked by the in-flight count). The caller (the
        dispatch loop) then waits on the condition variable until either
        a new request arrives or an in-flight one finishes.

        Cancelled entries are drained as encountered.
        """
        now = time.time()
        max_wait = self.cfg.max_wait_s

        # First pass: drain any cancelled entries from the top of the heap
        # so they don't keep us awake. (Mid-heap cancellations are removed
        # lazily when we eventually pop them.)
        while self._heap and (
            self._heap[0][1].cancel.is_set()
            or self._heap[0][1].status == "cancelled"
        ):
            heapq.heappop(self._heap)
            self._cancelled_in_heap = max(0, self._cancelled_in_heap - 1)

        # Starvation: any request waiting longer than max_wait gets
        # promoted, but only if its family is currently eligible. (A
        # starved image task behind a long-running text task in
        # non-concurrent mode still has to wait for text to finish.)
        if max_wait > 0:
            starved_idx: int | None = None
            starved_at: float = now
            for i, (_, r) in enumerate(self._heap):
                if r.cancel.is_set() or r.status == "cancelled":
                    continue
                if not self._can_dispatch(r):
                    continue
                wait = now - r.enqueued_at
                if wait > max_wait and r.enqueued_at < starved_at:
                    starved_at = r.enqueued_at
                    starved_idx = i
            if starved_idx is not None:
                _, req = self._heap[starved_idx]
                self._heap[starved_idx] = self._heap[-1]
                self._heap.pop()
                if self._heap:
                    heapq.heapify(self._heap)
                return req

        # Normal priority-ordered scan. We can't just heappop() and put
        # back if ineligible (heapq has no peek-then-skip primitive), so
        # walk in sorted order. For typical depths (a few in flight, a
        # handful pending) this is fine; heap operations elsewhere
        # dominate.
        for i, (_, r) in enumerate(sorted(self._heap)):
            if r.cancel.is_set() or r.status == "cancelled":
                continue
            if self._can_dispatch(r):
                # Remove this entry from the heap (linear scan to find it
                # since sorted() produced a copy).
                for j, (_, hr) in enumerate(self._heap):
                    if hr is r:
                        self._heap[j] = self._heap[-1]
                        self._heap.pop()
                        if self._heap:
                            heapq.heapify(self._heap)
                        break
                return r
        return None

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                async with self._cv:
                    while True:
                        if self._paused:
                            await self._cv.wait()
                            continue
                        req = self._pop_next()
                        if req is not None:
                            # Count the in-flight slot here, while we
                            # still hold the cv lock, so the next
                            # eligibility check sees the updated count.
                            family = req.task_type if req.task_type in self._in_flight_count else "text"
                            self._in_flight_count[family] += 1
                            req.dispatched = True
                            break
                        await self._cv.wait()
                # Outside the cv: prepare the upstream engine for this
                # request (model swap for text; for image we just hand
                # over to the runner — it coordinates GPU yield itself).
                try:
                    await self._prepare_and_release(req)
                except Exception as e:
                    log.exception("dispatcher: prepare failed for %s",
                                  req.request_id)
                    req.error = str(e)
                    req.status = "failed"
                    req.ready.set()
                    # Slot release happens in mark_in_flight_done via
                    # the handler's finally block. req.dispatched stays
                    # True so the refund actually occurs.
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

        For image-family tasks (``req.task_type == "image"``), no swap is
        done here — the image runner owns the cross-family coordination
        via ServerManager.yield_to_image() at dispatch time. We just mark
        the slot in-flight and let the handler invoke the runner.
        """
        wanted = req.model_required
        wanted_profile = getattr(req, "profile_required", None)
        loaded = self.sm.runtime.current_model

        # Image-family tasks bypass the text server swap entirely.
        if req.task_type == "image":
            self._in_flight[req.request_id] = req
            req.status = "running"
            req.started_at = time.time()
            # Record the resolved model_id on the request row even when the
            # client didn't pin one — otherwise the dashboard's "Recent"
            # table renders "(loaded)" forever instead of the actual model.
            resolved_model = req.model_required or self.sm.runtime.current_model
            self.db.update_request_status(req.request_id, "running",
                                          started_at=req.started_at,
                                          model=resolved_model)
            req.ready.set()
            return

        # ---- Multi-slot routing branch ----
        # When the beta is on, we route by model id across N parallel
        # ServerManagers instead of swapping a single slot. The cold-cache
        # decision is "reject" per plan: the operator owns slot assignments.
        if getattr(self.cfg, "multi_slot_enabled", False):
            if wanted is None:
                req.error = (
                    "multi-slot is on; this request must include a model "
                    "(via the request body or X-Llamanager-Model header)."
                )
                req.status = "failed"
                self.db.update_request_status(req.request_id, "failed",
                                              error=req.error,
                                              finished_at=time.time())
                req.ready.set()
                return
            slot_sm = None
            if hasattr(self.sm, "find_for"):
                slot_sm = self.sm.find_for(wanted)
            if slot_sm is None:
                req.error = (
                    f"model {wanted!r} is not loaded in any slot. "
                    "Load it via /ui/slots or `llamanager slots load`."
                )
                req.status = "failed"
                self.db.update_request_status(req.request_id, "failed",
                                              error=req.error,
                                              finished_at=time.time())
                req.ready.set()
                return
            # Found a slot. If the profile differs, swap in-place within
            # that slot (cheap when the model file is the same — llama-server
            # is restarted with new args).
            target_spec = resolve_spec(self.cfg, model=wanted,
                                       profile=wanted_profile)
            if slot_sm.runtime.current_profile != target_spec.profile_name:
                req.status = "swapping_model"
                self.db.update_request_status(req.request_id, "swapping_model")
                self.db.log_event("dispatch_slot_swap",
                                  {"req": req.request_id,
                                   "slot": slot_sm.slot_id,
                                   "to_profile": target_spec.profile_name})
                try:
                    await slot_sm.swap(target_spec)
                except Exception as e:
                    req.error = f"slot {slot_sm.slot_id} swap failed: {e}"
                    req.status = "failed"
                    self.db.update_request_status(req.request_id, "failed",
                                                  error=req.error,
                                                  finished_at=time.time())
                    req.ready.set()
                    return
            req.slot_id = slot_sm.slot_id
            self._in_flight[req.request_id] = req
            req.status = "running"
            req.started_at = time.time()
            self.db.update_request_status(req.request_id, "running",
                                          started_at=req.started_at,
                                          model=target_spec.model_id)
            req.ready.set()
            return

        # ---- Legacy single-slot path (multi-slot off) ----
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

        # Memory reclaim at the task boundary (requested by the watchdog under
        # memory pressure). A swap already cold-restarts the engine, so it
        # resets memory for free — we only need an explicit restart when the
        # SAME model continues, where the accumulated prompt cache / KV
        # checkpoints are what's eating RAM. Doing it here (between tasks)
        # means no live generation is interrupted.
        if self._reclaim_pending:
            self._reclaim_pending = False
            if not need_swap and self.sm.is_running:
                self.db.log_event("dispatch_mem_reclaim",
                                  {"req": req.request_id, "model": loaded})
                log.info("memory reclaim: restarting engine to reset cache "
                         "before request %s", req.request_id)
                try:
                    await self.sm.restart()
                except Exception as e:  # noqa: BLE001 — reclaim is best-effort
                    log.warning("memory reclaim restart failed: %s", e)

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

        # Mark in-flight and signal handler. The ``model`` column is
        # snapshotted to the resolved model id (either what the client
        # asked for, or the now-loaded one after any swap) so the
        # dashboard's Recent table can attribute each request to the
        # real model that served it.
        self._in_flight[req.request_id] = req
        req.status = "running"
        req.started_at = time.time()
        resolved_model = (
            (target_spec.model_id if target_spec else None)
            or req.model_required
            or self.sm.runtime.current_model
        )
        self.db.update_request_status(req.request_id, "running",
                                      started_at=req.started_at,
                                      model=resolved_model)
        req.ready.set()


def _request_public(req: QueuedRequest | None) -> dict[str, Any] | None:
    if req is None:
        return None
    return {
        "id": req.request_id,
        "origin": req.origin.name,
        "priority": req.priority,
        "model_required": req.model_required,
        "task_type": req.task_type,
        "status": req.status,
        "enqueued_at": req.enqueued_at,
        "started_at": req.started_at,
        "finished_at": req.finished_at,
    }


def _infer_task_type(cfg: Config, model_id: str | None) -> str:
    """Map a request's model id to a task family. Falls back to ``text``
    when the model is unknown (preserves legacy behaviour for callers
    that don't specify a model)."""
    if not model_id:
        # Inherit from the configured default model, if any. Otherwise
        # we'd treat "no model header" as text, which matches the
        # historical assumption.
        model_id = cfg.default_model or None
    if not model_id:
        return "text"
    engine = detect_engine_for_id(model_id, cfg.models_dir)
    return "image" if ENGINE_FAMILY.get(engine, "text") == "image" else "text"
