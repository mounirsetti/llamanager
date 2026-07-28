"""Queue-level tests for text + image dispatch.

The dispatcher is the only place where the per-family ``in_flight_count``
state is mutated, so these tests directly exercise that path with stubbed
ServerManager / image-runner side effects. The point is to catch the
specific failure modes the queue refactor was supposed to fix:

    * slot leak when an early-abort path calls mark_in_flight_done before
      dispatch (Bug 1 in the audit)
    * dead heap entries left behind by early-abort failures (Bug 2)
    * allow_concurrent=true not actually allowing text + image to run
      in parallel (Bug 3)

We never spawn real subprocesses or HTTP servers here — pure asyncio.
The project doesn't enable pytest-asyncio mode, so each test is a sync
function that wraps its body in ``asyncio.run(...)``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from llamanager.auth import Origin
from llamanager.config import Config
from llamanager.db import DB
from llamanager.queue_mgr import QueueManager
from llamanager.server_manager import ServerManager


def _make_cfg(tmp_path: Path, *, max_concurrent: int = 1,
              allow_concurrent: bool = False) -> Config:
    data = tmp_path / "lm"
    data.mkdir()
    (data / "logs").mkdir()
    (data / "models").mkdir()
    return Config(
        data_dir=data,
        max_concurrent=max_concurrent,
        allow_concurrent=allow_concurrent,
    )


def _make_origin(*, name: str = "t", priority: int = 50,
                 origin_id: int = 1) -> Origin:
    return Origin(id=origin_id, name=name, priority=priority,
                  allowed_models=["*"], is_admin=False, created_at=0.0)


def _plant_image_model(cfg: Config, name: str = "img-engine") -> None:
    """Make ``detect_engine_for_id`` see ``name`` as an image-family model."""
    d = cfg.models_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenizer_config.json").write_text("{}")
    (d / "preprocessor_config.json").write_text("{}")
    (d / "shard.safetensors").write_bytes(b"")


def _plant_text_model(cfg: Config, name: str = "tiny.gguf") -> None:
    (cfg.models_dir / name).write_bytes(b"")


def test_mark_in_flight_done_does_not_leak_slot_on_predispatch_abort(tmp_path):
    """Bug 1: early-abort (failure between enqueue and dispatch) used to
    free a slot the dispatcher never acquired, permanently inflating
    concurrency. Now: ``req.dispatched`` gates the refund."""
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            req = await qm.enqueue(origin=_make_origin(), model_required=None)
            assert req.dispatched is False
            before = dict(qm._in_flight_count)
            qm.mark_in_flight_done(
                req, error="bad input",
                cancelled=False, prompt_tokens=None, completion_tokens=None,
            )
            # In-flight counters untouched because no slot was acquired.
            assert qm._in_flight_count == before
            # Bookkeeping is dropped so the request can't be reused.
            assert req.request_id not in qm._by_id

        asyncio.run(go())
    finally:
        db.close()


def test_cancel_predispatch_drops_bookkeeping(tmp_path):
    """Bug 2: cancelling a queued (pre-dispatch) request removes it from
    ``_by_id`` so the dispatcher won't trip over a dead entry, and the
    in-flight counters stay untouched."""
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            req = await qm.enqueue(origin=_make_origin(), model_required=None)
            rid = req.request_id
            assert rid in qm._by_id

            before = dict(qm._in_flight_count)
            assert qm.cancel(rid) is True
            assert req.cancel.is_set()
            assert req.status == "cancelled"
            assert rid not in qm._by_id
            assert qm._in_flight_count == before

        asyncio.run(go())
    finally:
        db.close()


def test_can_dispatch_serialises_text_and_image_by_default(tmp_path):
    """When ``allow_concurrent=False`` (default), text and image are
    mutually exclusive: an image request can't run while a text request
    is in flight, and vice versa."""
    cfg = _make_cfg(tmp_path, allow_concurrent=False)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_text_model(cfg)
            _plant_image_model(cfg)
            text_req = await qm.enqueue(origin=_make_origin(),
                                        model_required="tiny.gguf")
            image_req = await qm.enqueue(origin=_make_origin(),
                                         model_required="img-engine")
            assert text_req.task_type == "text"
            assert image_req.task_type == "image"

            qm._in_flight_count["text"] = 1
            assert qm._can_dispatch(text_req) is False, \
                "second text task should respect max_concurrent=1"
            assert qm._can_dispatch(image_req) is False, \
                "image must not start while text is in flight"

            qm._in_flight_count["text"] = 0
            qm._in_flight_count["image"] = 1
            assert qm._can_dispatch(text_req) is False, \
                "text must not start while image is in flight"
            assert qm._can_dispatch(image_req) is False, \
                "second image must respect the 1-slot ceiling"

        asyncio.run(go())
    finally:
        db.close()


def test_allow_concurrent_lets_text_and_image_run_in_parallel(tmp_path):
    """Bug 3: with ``allow_concurrent=True`` the queue actually allows
    a text and an image task to be dispatched at the same time."""
    cfg = _make_cfg(tmp_path, allow_concurrent=True)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_text_model(cfg)
            _plant_image_model(cfg)
            text_req = await qm.enqueue(origin=_make_origin(),
                                        model_required="tiny.gguf")
            image_req = await qm.enqueue(origin=_make_origin(),
                                         model_required="img-engine")

            qm._in_flight_count["text"] = 1
            assert qm._can_dispatch(image_req) is True, \
                "image can run alongside text in concurrent mode"
            qm._in_flight_count["image"] = 1
            assert qm._can_dispatch(image_req) is False, \
                "still capped at one image at a time"
            assert qm._can_dispatch(text_req) is False, \
                "still capped at max_concurrent text tasks"

        asyncio.run(go())
    finally:
        db.close()


def test_dispatcher_picks_eligible_lower_priority_when_higher_blocked(tmp_path):
    """When the highest-priority pending request can't be dispatched
    (its family is blocked), the dispatcher must look further down the
    heap rather than head-of-line-block. This is the whole reason
    ``_pop_next`` was reworked to scan instead of heappop."""
    cfg = _make_cfg(tmp_path, allow_concurrent=True)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            high = _make_origin(name="hi", priority=90, origin_id=1)
            low = _make_origin(name="lo", priority=10, origin_id=2)
            _plant_text_model(cfg)
            _plant_image_model(cfg)
            await qm.enqueue(origin=high, model_required="img-engine")
            await qm.enqueue(origin=low, model_required="tiny.gguf")

            # Image quota is already taken — so the dispatcher should
            # skip the high-priority image and pick the low-priority text.
            qm._in_flight_count["image"] = 1
            picked = qm._pop_next()
            assert picked is not None
            assert picked.task_type == "text"
            assert picked.origin.name == "lo"

        asyncio.run(go())
    finally:
        db.close()


def test_dispatcher_full_cycle_marks_inflight_and_releases(tmp_path):
    """End-to-end with the dispatcher loop running: enqueue, observe the
    in-flight count rise, then mark_in_flight_done and watch it fall.

    Image-family requests skip the swap entirely in _prepare_and_release,
    so this test uses an image request to avoid needing a real
    text-server start path.
    """
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_image_model(cfg)
            qm.start()
            try:
                req = await qm.enqueue(origin=_make_origin(),
                                       model_required="img-engine")
                await asyncio.wait_for(req.ready.wait(), timeout=2.0)
                assert req.dispatched is True
                assert qm._in_flight_count["image"] == 1

                qm.mark_in_flight_done(
                    req, error=None, cancelled=False,
                    prompt_tokens=None, completion_tokens=None,
                )
                # Give the wake-task one tick to run.
                await asyncio.sleep(0.05)
                assert req.dispatched is False
                assert qm._in_flight_count["image"] == 0
            finally:
                await qm.stop()

        asyncio.run(go())
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Regression: a busy image engine blocks dispatch even when the queue's
# own in-flight counter says otherwise.
#
# The failure this pins down: a client that dropped its SSE connection
# mid-generation used to refund the image slot from the *response's*
# finally block, while the diffusion engine kept running and holding its
# VRAM. With image_in back at 0 the dispatcher happily started
# llama-server on top of a live 26 GB image engine, filling the card.
#
# The queue therefore takes two independent readings — its counter and
# the runner's run lock — and blocks if either says "busy".
# ---------------------------------------------------------------------------

class _FakeImageRunner:
    """Minimal stand-in exposing the ``is_busy`` contract."""

    def __init__(self) -> None:
        self.busy = False

    @property
    def is_busy(self) -> bool:
        return self.busy


def test_busy_image_engine_blocks_text_even_when_counter_is_zero(tmp_path):
    cfg = _make_cfg(tmp_path, allow_concurrent=False)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)
        runner = _FakeImageRunner()
        qm.image_runner = runner

        async def go():
            _plant_text_model(cfg)
            _plant_image_model(cfg)
            text_req = await qm.enqueue(origin=_make_origin(),
                                        model_required="tiny.gguf")
            image_req = await qm.enqueue(origin=_make_origin(),
                                         model_required="img-engine")

            # Counter says the GPU is free; the engine says otherwise.
            # This is exactly the post-disconnect state.
            qm._in_flight_count["image"] = 0
            runner.busy = True
            assert qm._can_dispatch(text_req) is False, \
                "text must not start while a diffusion engine holds the GPU"
            assert qm._can_dispatch(image_req) is False, \
                "a second image must not stack onto a running engine"

            # Engine releases the GPU -> both become dispatchable again.
            runner.busy = False
            assert qm._can_dispatch(text_req) is True
            assert qm._can_dispatch(image_req) is True

        asyncio.run(go())
    finally:
        db.close()


def test_image_runner_missing_or_broken_never_blocks_dispatch(tmp_path):
    """The extra check is a safety net, not a new way to wedge the queue."""
    cfg = _make_cfg(tmp_path, allow_concurrent=False)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_text_model(cfg)
            text_req = await qm.enqueue(origin=_make_origin(),
                                        model_required="tiny.gguf")

            # Never wired up.
            assert qm._image_engine_busy() is False
            assert qm._can_dispatch(text_req) is True

            class _Broken:
                @property
                def is_busy(self):
                    raise RuntimeError("runtime state unreadable")

            qm.image_runner = _Broken()
            assert qm._image_engine_busy() is False, \
                "a raising is_busy must degrade to 'not busy', not propagate"
            assert qm._can_dispatch(text_req) is True

        asyncio.run(go())
    finally:
        db.close()
