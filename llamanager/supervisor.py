"""Crash supervisor — implements the 3-in-N restart policy from spec §6.5.

When the crash limit is hit, the supervisor enters a cooldown period
(equal to the window duration) then resets and will try again. This
prevents permanent degradation from transient issues.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from .config import Config
from .server_manager import ServerManager

log = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, cfg: Config, sm: ServerManager):
        self.cfg = cfg
        self.sm = sm
        self.enabled: bool = True
        self.exits: deque[float] = deque()
        self.consecutive_failures = 0
        self.last_success_start: float | None = None
        self._cooldown_count: int = 0
        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[int] = asyncio.Queue(maxsize=100)
        sm.add_exit_listener(self._queue)

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
                rc = await self._queue.get()
            except asyncio.CancelledError:
                return
            await self._handle_exit(rc)

    async def _handle_exit(self, rc: int) -> None:
        if not self.enabled:
            log.debug("supervisor: auto-restart disabled, skipping restart (rc=%d)", rc)
            return

        now = time.time()

        # Did we just have a successful long-enough run? Reset counters.
        if (self.last_success_start
                and now - self.last_success_start >= self.cfg.success_run_seconds):
            self.consecutive_failures = 0
            self.exits.clear()

        # Slide the window.
        cutoff = now - self.cfg.window_seconds
        self.exits.append(now)
        while self.exits and self.exits[0] < cutoff:
            self.exits.popleft()

        if len(self.exits) >= self.cfg.max_restarts_in_window:
            # Exponential cooldown: window * 1, window * 2, window * 4, ... capped at 1 hour
            cooldown = min(
                self.cfg.window_seconds * (2 ** self._cooldown_count),
                3600,
            )
            self._cooldown_count += 1
            log.error(
                "supervisor: %d exits in %ds, entering cooldown for %ds "
                "before retrying (attempt #%d)",
                len(self.exits), self.cfg.window_seconds,
                cooldown, self._cooldown_count,
            )
            self.sm.mark_degraded(
                f"{len(self.exits)} crashes in {self.cfg.window_seconds}s"
            )
            await asyncio.sleep(cooldown)
            self.exits.clear()
            self.consecutive_failures = 0
            log.info("supervisor: cooldown over, attempting recovery restart")
            try:
                await self.sm.restart()
                self.last_success_start = time.time()
                self._cooldown_count = 0  # reset on success
                log.info("supervisor: recovery restart succeeded")
            except Exception as e:
                log.error("supervisor: recovery restart failed: %s", e)
                # Don't give up — the next crash event will trigger another
                # cooldown cycle with a longer wait
            return

        self.consecutive_failures += 1
        backoff = min(2 ** self.consecutive_failures, 30)
        log.info("supervisor: backing off %ss before restart (failure #%d)",
                 backoff, self.consecutive_failures)
        await asyncio.sleep(backoff)

        try:
            await self.sm.restart()
            self.last_success_start = time.time()
        except Exception as e:
            log.error("supervisor: restart failed: %s", e)
            self.exits.append(time.time())
            if len(self.exits) >= self.cfg.max_restarts_in_window:
                self.sm.mark_degraded(f"restart kept failing: {e}")
