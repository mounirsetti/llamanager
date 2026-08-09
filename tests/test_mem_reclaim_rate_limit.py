"""The memory-reclaim restart must be rate-limited at the dispatcher.

Background (2026-08-09): the watchdog sat latched at CRITICAL — a full swap
file with 35 GB of RAM free — and re-armed ``request_reclaim()`` every 15s.
The dispatcher honoured every one of them, so ``llama-server`` was restarted
before 4581 of 5004 requests over a week, adding ~12s to each. The
classification bug is fixed in mem_guard; this is the independent floor that
bounds the damage no matter how often the watchdog asks.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llamanager.auth import Origin
from llamanager.config import Config
from llamanager.db import DB
from llamanager.queue_mgr import QueueManager


class _StubRuntime:
    def __init__(self) -> None:
        self.current_model = "tiny.gguf"
        self.current_profile = None


class _StubSM:
    """Just enough ServerManager for the reclaim branch of _prepare_and_release."""

    def __init__(self) -> None:
        self.runtime = _StubRuntime()
        self.is_running = True
        self.restarts = 0

    async def restart(self, spec=None):
        self.restarts += 1
        return 1234


def _make_cfg(tmp_path: Path, **kw) -> Config:
    data = tmp_path / "lm"
    data.mkdir()
    (data / "logs").mkdir()
    (data / "models").mkdir()
    return Config(data_dir=data, **kw)


def _origin() -> Origin:
    return Origin(id=1, name="t", priority=50, allowed_models=["*"],
                  is_admin=False, created_at=0.0)


def _reclaim_events(db: DB) -> list[dict]:
    rows = db.conn.execute(
        "select payload_json from events where kind='dispatch_mem_reclaim' "
        "order by id").fetchall()
    return [json.loads(r[0]) for r in rows]


def test_reclaim_restart_is_rate_limited(tmp_path):
    cfg = _make_cfg(tmp_path, mem_reclaim_min_interval_s=300.0)
    db = DB(cfg.db_path)
    try:
        sm = _StubSM()
        qm = QueueManager(cfg, db, sm)

        async def one_request():
            req = await qm.enqueue(origin=_origin(), model_required=None)
            await qm._prepare_and_release(req)
            return req

        async def go():
            # 1) First ask: the engine really is restarted.
            qm.request_reclaim()
            await one_request()
            assert sm.restarts == 1

            # 2) Three more asks inside the window: all suppressed, so the
            #    requests start immediately instead of paying a reload each.
            for _ in range(3):
                qm.request_reclaim()
                await one_request()
            assert sm.restarts == 1, "reclaim restarts must not run back-to-back"
            assert qm._reclaim_skipped == 3

            # 3) Once the window has passed, reclaim works again — and the
            #    event records how many were suppressed in between.
            qm._last_reclaim_ts -= cfg.mem_reclaim_min_interval_s + 1
            qm.request_reclaim()
            await one_request()
            assert sm.restarts == 2
            assert qm._reclaim_skipped == 0

            events = _reclaim_events(db)
            assert len(events) == 2
            assert events[0]["skipped_since_last"] == 0
            assert events[1]["skipped_since_last"] == 3

        asyncio.run(go())
    finally:
        db.close()


def test_reclaim_floor_can_be_disabled(tmp_path):
    """``reclaim_min_interval_s = 0`` restores the old every-time behaviour
    for operators who want it."""
    cfg = _make_cfg(tmp_path, mem_reclaim_min_interval_s=0.0)
    db = DB(cfg.db_path)
    try:
        sm = _StubSM()
        qm = QueueManager(cfg, db, sm)

        async def go():
            for _ in range(3):
                qm.request_reclaim()
                req = await qm.enqueue(origin=_origin(), model_required=None)
                await qm._prepare_and_release(req)
            assert sm.restarts == 3

        asyncio.run(go())
    finally:
        db.close()


def test_no_reclaim_means_no_restart(tmp_path):
    """The dispatcher must not touch a warm engine when nothing asked it to."""
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = _StubSM()
        qm = QueueManager(cfg, db, sm)

        async def go():
            for _ in range(3):
                req = await qm.enqueue(origin=_origin(), model_required=None)
                await qm._prepare_and_release(req)
                assert req.status == "running"
            assert sm.restarts == 0

        asyncio.run(go())
    finally:
        db.close()
