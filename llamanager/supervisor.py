"""Crash supervisor — implements the 3-in-N restart policy from spec §6.5.

When the crash limit is hit, the supervisor enters a cooldown period
(equal to the window duration) then resets and will try again. This
prevents permanent degradation from transient issues.

Multi-slot aware: crash bookkeeping is per-slot, so a flapping slot 2
doesn't poison slot 0's recovery window. Each slot keeps its own
exit-time deque, consecutive-failure count, cooldown exponent, and
last-success timestamp.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque

from .config import Config

log = logging.getLogger(__name__)


class _SlotCrashState:
    """Per-slot crash counters. Each slot's recovery is independent."""

    __slots__ = ("exits", "consecutive_failures",
                 "last_success_start", "cooldown_count")

    def __init__(self) -> None:
        self.exits: deque[float] = deque()
        self.consecutive_failures: int = 0
        self.last_success_start: float | None = None
        self.cooldown_count: int = 0


class Supervisor:
    """Watches all slots in the pool for unexpected exits.

    ``sm`` is a ``ServerPool`` (or, when wired in legacy tests, any
    object that exposes ``add_exit_listener``, ``mark_degraded``,
    ``restart``, and ``slot(id)``). The supervisor only ever calls
    methods the pool exposes; it does not reach into individual
    ServerManagers except through the pool's accessors.
    """

    def __init__(self, cfg: Config, sm) -> None:
        self.cfg = cfg
        self.sm = sm
        self.enabled: bool = True
        self._state: dict[int, _SlotCrashState] = defaultdict(_SlotCrashState)
        self._task: asyncio.Task[None] | None = None
        # Queue carries (slot_id, returncode). The pool subscribes every
        # known slot to this single queue on construction.
        self._queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue(maxsize=100)
        sm.add_exit_listener(self._queue)

    # ----- subscription -----
    def subscribe_slot(self, slot_sm) -> None:
        """Wire a newly-added slot's exit events into our queue.

        Called by ``ServerPool.add_slot()`` at runtime so a slot added
        after boot still gets supervised.
        """
        slot_sm.add_exit_listener(self._queue)

    # ----- lifecycle -----
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                slot_id, rc = await self._queue.get()
            except asyncio.CancelledError:
                return
            await self._handle_exit(slot_id, rc)

    # ----- per-slot helpers -----
    def _slot_manager(self, slot_id: int):
        """Resolve the per-slot ServerManager to call restart() on.

        For slot 0 we use the pool's legacy single-instance entry points
        (``restart``, ``mark_degraded``) so behaviour with multi-slot
        off is identical to the pre-pool design. For slot > 0 we route
        through ``sm.slot(slot_id)``.
        """
        if slot_id == 0:
            return self.sm
        if hasattr(self.sm, "slot"):
            return self.sm.slot(slot_id) or self.sm
        return self.sm

    async def _restart_slot(self, slot_id: int) -> None:
        """Restart the named slot, falling back to a pool-level restart
        if no per-slot accessor exists (e.g. legacy single-instance
        passthrough)."""
        target = self._slot_manager(slot_id)
        # Each ServerManager and ServerPool both expose ``restart``.
        await target.restart()

    def _mark_degraded(self, slot_id: int, reason: str) -> None:
        target = self._slot_manager(slot_id)
        if hasattr(target, "mark_degraded"):
            target.mark_degraded(reason)

    async def _await_mem_recovery(self, slot_id: int, *,
                                  timeout_s: float = 180.0,
                                  poll_s: float = 5.0) -> None:
        """After an OOM kill (rc -9), block the restart until host memory is no
        longer under pressure — so we don't reload straight back into an OOM.
        Bounded by ``timeout_s`` so a persistently-tight box still eventually
        retries (and hits the normal crash-window cooldown)."""
        try:
            from .mem_guard import (read_mem_state, classify_pressure,
                                    MemThresholds, Pressure)
        except Exception:  # noqa: BLE001 — never block restart on an import error
            return
        th = MemThresholds.from_cfg(self.cfg)
        if classify_pressure(read_mem_state(), th) < Pressure.WARN:
            return  # memory already fine — restart normally
        self._mark_degraded(slot_id, "OOM-killed; waiting for memory to recover")
        log.error("supervisor: slot %d was OOM-killed (rc=-9) and host memory "
                  "is still tight — pausing restart up to %.0fs to avoid an "
                  "OOM restart loop", slot_id, timeout_s)
        waited = 0.0
        while waited < timeout_s:
            await asyncio.sleep(poll_s)
            waited += poll_s
            if classify_pressure(read_mem_state(), th) < Pressure.WARN:
                log.info("supervisor: slot %d — memory recovered after %.0fs, "
                         "proceeding with restart", slot_id, waited)
                return
        log.warning("supervisor: slot %d — memory still tight after %.0fs; "
                    "proceeding (crash-window cooldown will bound retries)",
                    slot_id, timeout_s)

    # ----- core logic -----
    async def _handle_exit(self, slot_id: int, rc: int) -> None:
        if not self.enabled:
            log.info(
                "supervisor: auto-restart disabled, skipping restart "
                "(slot=%d, rc=%d). Toggle on with "
                "`llamanager setup autorestart on`.",
                slot_id, rc,
            )
            return

        # OOM restart-loop guard. rc == -9 is a SIGKILL — almost always the
        # kernel OOM-killer reaping llama-server. Restarting immediately reloads
        # the same oversized footprint and re-OOMs: a doom loop that pins the
        # box (this is exactly what drove a machine to a hard reboot). If host
        # memory is still tight, wait for it to recover before restarting.
        if rc == -9:
            await self._await_mem_recovery(slot_id)

        st = self._state[slot_id]
        now = time.time()

        # Did we just have a successful long-enough run? Reset counters.
        if (st.last_success_start
                and now - st.last_success_start >= self.cfg.success_run_seconds):
            st.consecutive_failures = 0
            st.exits.clear()

        # Slide the window.
        cutoff = now - self.cfg.window_seconds
        st.exits.append(now)
        while st.exits and st.exits[0] < cutoff:
            st.exits.popleft()

        if len(st.exits) >= self.cfg.max_restarts_in_window:
            # Exponential cooldown: window * 1, window * 2, ... capped at 1h.
            cooldown = min(
                self.cfg.window_seconds * (2 ** st.cooldown_count),
                3600,
            )
            st.cooldown_count += 1
            log.error(
                "supervisor: slot %d had %d exits in %ds, entering cooldown "
                "for %ds before retrying (attempt #%d)",
                slot_id, len(st.exits), self.cfg.window_seconds,
                cooldown, st.cooldown_count,
            )
            self._mark_degraded(
                slot_id,
                f"{len(st.exits)} crashes in {self.cfg.window_seconds}s",
            )
            await asyncio.sleep(cooldown)
            st.exits.clear()
            st.consecutive_failures = 0
            log.info("supervisor: slot %d cooldown over, attempting recovery "
                     "restart", slot_id)
            try:
                await self._restart_slot(slot_id)
                st.last_success_start = time.time()
                st.cooldown_count = 0  # reset on success
                log.info("supervisor: slot %d recovery restart succeeded",
                         slot_id)
            except Exception as e:
                log.error("supervisor: slot %d recovery restart failed: %s",
                          slot_id, e)
                # Don't give up — the next crash event will trigger another
                # cooldown cycle with a longer wait
            return

        st.consecutive_failures += 1
        backoff = min(2 ** st.consecutive_failures, 30)
        log.info("supervisor: slot %d backing off %ss before restart "
                 "(failure #%d)", slot_id, backoff, st.consecutive_failures)
        await asyncio.sleep(backoff)

        try:
            await self._restart_slot(slot_id)
            st.last_success_start = time.time()
        except Exception as e:
            log.error("supervisor: slot %d restart failed: %s", slot_id, e)
            st.exits.append(time.time())
            if len(st.exits) >= self.cfg.max_restarts_in_window:
                self._mark_degraded(slot_id, f"restart kept failing: {e}")
