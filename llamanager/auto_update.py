"""Auto-update-when-idle scheduler.

A single in-process background loop that, for each engine the operator has
opted in (``cfg.auto_update_engines``), checks upstream for a newer build on a
fixed cadence and — once the daemon has been idle (no in-flight or pending
requests) for ``cfg.auto_update_idle_seconds`` — applies the update.

Three engine classes share one string keyspace (see config.py):

* **llama.cpp / Atomic / MLX variants** — keyed by variant id
  (``"llama.cpp-cuda"``). A *real* upstream version check via
  ``llama_installer.check_for_update``; the update reuses
  ``llama_installer.install_variant`` (the same code the Setup page's
  "Update to X" button runs). When the variant is the active engine and a
  model is loaded, the install is wrapped in ``ServerPool.yield_to_image``
  so the running model is unloaded first (a running ``.exe`` is locked on
  Windows) and restored afterwards.
* **diffusion engines** — keyed by engine name (``"hidream"``, ``"z_image"``).
  A diffusion engine's version is the version of its pinned ``diffusers``
  dependency. The *target* is the diffusers release this llamanager build
  ships (``engine_installer.DIFFUSERS_PIN``); the *installed* version is read
  live from the engine's venv. Auto-update fires only when the installed
  diffusers is strictly **older** than the shipped pin — a real, known-good
  signal that moves when the operator updates llamanager to a build pinning a
  newer diffusers. It never chases git ``main`` or jumps ahead of the tested
  pin, and only fires for an already-installed engine. The update itself
  re-runs the installer (``pip install --upgrade`` via ``EngineInstaller``),
  which brings the venv to the pinned set.
* **llamanager itself** — the reserved key ``"llamanager"``. Reuses the
  ``/ui/about`` self-update path; skipped for editable installs.

The loop reads ``app.state.cfg`` on every tick so UI/CLI toggles and SIGHUP
config reloads take effect without a restart.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .engine_installer import ENGINE_PLANS
from .llama_installer import (
    check_for_update,
    detect_binary,
    detect_variant_for_binary,
    install_variant,
    parse_variant_id,
    variant_id,
)

log = logging.getLogger(__name__)

# How often the loop wakes to evaluate idleness + due checks. The upstream
# version check itself is throttled per engine to ``check_interval_seconds``;
# this is just the polling granularity for the idle window.
TICK_SECONDS = 60.0

# Don't spam the activity feed with "skipped, daemon busy" beats — at most one
# per engine per this many seconds while an update stays pending.
_SKIP_LOG_THROTTLE_SECONDS = 1800.0

# Reserved key for the daemon's own self-update.
SELF_KEY = "llamanager"


class AutoUpdater:
    """Background idle-gated updater. One instance per daemon, started from
    the app lifespan (see ``app.create_app``)."""

    def __init__(self, app, *, tick_seconds: float = TICK_SECONDS) -> None:
        self.app = app
        self.tick_seconds = tick_seconds
        # Monotonic timestamps, keyed by engine key.
        self._last_check: dict[str, float] = {}
        self._last_update: dict[str, float] = {}
        self._last_skip_log: dict[str, float] = {}
        # Result of the most recent upstream check: True = an update is
        # available / due to apply.
        self._update_available: dict[str, bool] = {}
        # Set while an inline update (llama variant / self) is running so a
        # later tick doesn't start a second heavy job.
        self._busy = False

    # ----- lifecycle -----
    async def run(self, stop_event: asyncio.Event) -> None:
        """Tick until ``stop_event`` is set. Never propagates exceptions —
        a failed tick is logged and the loop continues."""
        while not stop_event.is_set():
            try:
                await self._tick()
            except Exception:  # noqa: BLE001 — a background loop must not die
                log.exception("auto_update: tick failed")
            try:
                await asyncio.wait_for(stop_event.wait(),
                                       timeout=self.tick_seconds)
            except asyncio.TimeoutError:
                pass

    # ----- classification -----
    @staticmethod
    def classify(key: str) -> tuple[str, ...] | None:
        """Map an engine key to its update mechanism.

        Returns ``("self",)`` | ``("llama", source, backend)`` |
        ``("diffusion", engine)`` | ``None`` (unknown key).
        """
        if key == SELF_KEY:
            return ("self",)
        parsed = parse_variant_id(key)
        if parsed is not None:
            return ("llama", parsed[0], parsed[1])
        if key in ENGINE_PLANS:
            return ("diffusion", key)
        return None

    # ----- one evaluation pass -----
    async def _tick(self) -> None:
        cfg = self.app.state.cfg
        enabled = [k for k, v in (cfg.auto_update_engines or {}).items() if v]
        if not enabled:
            return

        now = time.monotonic()
        check_interval = max(60, int(cfg.auto_update_check_interval_seconds))
        idle_required = max(0, int(cfg.auto_update_idle_seconds))

        # 1. Refresh upstream checks that are due (or never run yet).
        for key in enabled:
            if self.classify(key) is None:
                continue
            last = self._last_check.get(key)
            if last is not None and (now - last) < check_interval:
                continue
            self._update_available[key] = await self._check_engine(key)
            self._last_check[key] = time.monotonic()

        # 2. Apply at most one update per tick to avoid concurrent heavy jobs.
        if self._busy or self._anything_installing():
            return
        queue = self.app.state.queue
        for key in enabled:
            if not self._update_available.get(key):
                continue
            # Don't re-attempt the same engine more often than the check
            # cadence (covers diffusion's "always due" re-install path).
            last_upd = self._last_update.get(key)
            if last_upd is not None and (now - last_upd) < check_interval:
                continue
            idle = queue.idle_seconds()
            if idle < idle_required:
                self._maybe_log_skip(key, idle, idle_required)
                continue
            await self._perform(key)
            self._last_update[key] = time.monotonic()
            self._last_skip_log.pop(key, None)
            # Force a fresh upstream check before the next attempt.
            self._update_available[key] = False
            self._last_check[key] = 0.0
            break

    def _maybe_log_skip(self, key: str, idle: float, required: int) -> None:
        now = time.monotonic()
        last = self._last_skip_log.get(key)
        if last is not None and (now - last) < _SKIP_LOG_THROTTLE_SECONDS:
            return
        self._last_skip_log[key] = now
        self.app.state.db.log_event("auto_update_skipped_busy", {
            "engine": key,
            "idle_seconds": round(idle, 1),
            "required_seconds": required,
        })
        log.info("auto_update: %s has an update but daemon not idle "
                 "(%.0fs < %ds) — deferring", key, idle, required)

    def _anything_installing(self) -> bool:
        """True if any install is already in flight (engine installer task,
        a variant install, or our own inline job)."""
        installer = self.app.state.engine_installer
        for eng in ENGINE_PLANS:
            if installer.active_for_engine(eng):
                return True
        for st in self.app.state.install_states.values():
            if getattr(st, "status", "") == "running":
                return True
        return False

    # ----- upstream checks -----
    async def _check_engine(self, key: str) -> bool:
        kind = self.classify(key)
        loop = asyncio.get_running_loop()
        try:
            if kind[0] == "self":
                from .api_ui import _check_latest_release
                info = await loop.run_in_executor(None, _check_latest_release)
                available = bool(info.get("is_newer"))
                if available:
                    self.app.state.db.log_event("auto_update_check", {
                        "engine": key, "available": True,
                        "latest": info.get("latest"),
                        "current": info.get("current"),
                    })
                return available
            if kind[0] == "llama":
                _, source, backend = kind
                info = await loop.run_in_executor(
                    None, check_for_update, source, backend)
                if info.has_update:
                    self.app.state.db.log_event("auto_update_check", {
                        "engine": key, "available": True,
                        "latest": info.latest, "current": info.installed,
                    })
                return bool(info.has_update)
            if kind[0] == "diffusion":
                # Real version check: is the installed diffusers older than
                # the release this llamanager build pins? Never first-installs
                # (not-installed engines report no update) and never jumps
                # past the tested pin.
                engine = kind[1]
                from .engine_installer import diffusion_update_info
                info = await loop.run_in_executor(
                    None, diffusion_update_info, self.app.state.cfg, engine)
                if info["has_update"]:
                    self.app.state.db.log_event("auto_update_check", {
                        "engine": key, "available": True,
                        "latest": info["target"], "current": info["installed"],
                    })
                return bool(info["has_update"])
        except Exception as e:  # noqa: BLE001 — checks never raise into the loop
            log.warning("auto_update: check for %s failed: %s", key, e)
            return False
        return False

    # ----- apply -----
    async def _perform(self, key: str) -> None:
        kind = self.classify(key)
        self._busy = True
        self.app.state.db.log_event("auto_update_started", {"engine": key})
        log.info("auto_update: applying update for %s", key)
        try:
            if kind[0] == "self":
                await self._update_self(key)
            elif kind[0] == "llama":
                await self._update_llama(key, kind[1], kind[2])
            elif kind[0] == "diffusion":
                await self._update_diffusion(key, kind[1])
        except Exception as e:  # noqa: BLE001
            log.exception("auto_update: update for %s failed", key)
            self.app.state.db.log_event("auto_update_failed",
                                        {"engine": key, "error": str(e)})
        finally:
            self._busy = False

    async def _update_llama(self, key: str, source: str, backend: str) -> None:
        cfg = self.app.state.cfg
        sm = self.app.state.sm
        states = self.app.state.install_states
        state = states.get(key)
        if state is None:
            from .llama_installer import InstallState
            state = InstallState()
            states[key] = state
        # Reset the per-variant install state so the Setup page reflects the
        # auto-update run identically to a manual click.
        state.status = "running"
        state.lines = []
        state.error = None
        state.installed_path = None

        active = detect_binary(cfg.llama_server_binary)
        active_variant = detect_variant_for_binary(active) if active else None
        is_active = active_variant == (source, backend)

        if is_active and sm.is_running:
            # Stop the running model, swap the binary, restore the model.
            async with sm.yield_to_image(reason=f"auto-update:{key}"):
                await install_variant(state, source, backend)
        else:
            await install_variant(state, source, backend)

        if state.status == "done":
            self.app.state.db.log_event("auto_update_done", {
                "engine": key, "path": state.installed_path,
            })
        else:
            self.app.state.db.log_event("auto_update_failed", {
                "engine": key, "error": state.error or "install did not complete",
            })

    async def _update_diffusion(self, key: str, engine: str) -> None:
        installer = self.app.state.engine_installer
        if installer.active_for_engine(engine):
            return
        # Fire-and-forget: EngineInstaller streams its own
        # install_started / install_done / install_failed events, and
        # ``_anything_installing`` keeps the next tick from overlapping.
        installer.start(engine)
        from .engine_installer import diffusion_target_version
        target = diffusion_target_version(engine, self.app.state.cfg)
        self.app.state.db.log_event("auto_update_done", {
            "engine": key,
            "note": f"updating diffusers to {target}" if target else "re-installing deps",
        })

    async def _update_self(self, key: str) -> None:
        from .api_ui import _run_self_update, _schedule_self_restart
        loop = asyncio.get_running_loop()
        res: dict[str, Any] = await loop.run_in_executor(None, _run_self_update)
        if res.get("mode") == "editable":
            log.info("auto_update: skipping llamanager self-update "
                     "(editable install — update the checkout manually)")
            self.app.state.db.log_event("auto_update_failed", {
                "engine": key,
                "error": "editable install; update the checkout manually",
            })
            return
        if not res.get("ok"):
            self.app.state.db.log_event("auto_update_failed", {
                "engine": key, "error": res.get("error", "self-update failed"),
            })
            return
        self.app.state.db.log_event("auto_update_done",
                                    {"engine": key, "note": "restarting"})
        sm = self.app.state.sm
        if sm.is_running:
            await sm.stop()
        await _schedule_self_restart()
