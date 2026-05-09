"""Crash supervisor — implements the 3-in-5 restart policy from spec §6.5."""
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
        self.exits: deque[float] = deque()
        self.consecutive_failures = 0
        self.last_success_start: float | None = None
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
        now = time.time()

        # Did we just have a successful long-enough run? Reset counter.
        if (self.last_success_start
                and now - self.last_success_start >= self.cfg.success_run_seconds):
            self.consecutive_failures = 0

        # Slide the window.
        cutoff = now - self.cfg.window_seconds
        self.exits.append(now)
        while self.exits and self.exits[0] < cutoff:
            self.exits.popleft()

        if len(self.exits) >= self.cfg.max_restarts_in_window:
            log.error("supervisor: %d exits in %ss — giving up",
                      len(self.exits), self.cfg.window_seconds)
            self.sm.mark_degraded(
                f"{len(self.exits)} crashes in {self.cfg.window_seconds}s"
            )
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
            # if it failed to even start, that counts as another exit
            self.exits.append(time.time())
            if len(self.exits) >= self.cfg.max_restarts_in_window:
                self.sm.mark_degraded(f"restart kept failing: {e}")
