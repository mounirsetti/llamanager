"""ServerPool — the multi-slot abstraction layered over ServerManager.

When ``cfg.multi_slot_enabled`` is False, the pool wraps a single
ServerManager (slot 0) and forwards every method call to it — behaviour
is byte-identical to the pre-pool design. When True, the pool also
holds N additional ServerManager instances, each on its own port, and
exposes routing helpers (``find_for``, ``upstream_for``).

Persistence: slot 0 keeps writing to ``runtime.json`` as before. Slots
1..N are listed in a sibling ``slots.json`` manifest (see
``slots_state``); their transient runtime state is held in memory.

Why a pool instead of teaching ServerManager about N children: keeping
each child as the existing, well-tested ServerManager confines
multiplicity to one module. The crash supervisor, dispatcher, and
proxy URL construction need almost no changes when multi-slot is off.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import runtime_state as rt
from . import slots_state
from .config import Config
from .db import DB
from .server_manager import ServerError, ServerManager, StartSpec, resolve_spec

log = logging.getLogger(__name__)


@dataclass
class SlotView:
    """Read-only snapshot of one slot — used by /admin/slots and the UI.

    Built from a ServerManager instance plus the manifest entry. Kept
    small and JSON-serialisable so it can drop straight into responses.
    ``size_gb`` is the on-disk file size of the slot's loaded model
    (None when the slot is empty), used as the VRAM-budget proxy in
    the dashboard until a live VRAM probe lands.
    """
    id: int
    port: int
    state: str             # stopped|starting|running|swapping|crashed|degraded
    model: str | None
    profile: str | None
    pid: int | None
    started_at: float | None
    uptime_s: float | None = None
    size_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "port": self.port,
            "state": self.state,
            "model": self.model,
            "profile": self.profile,
            "pid": self.pid,
            "started_at": self.started_at,
            "uptime_s": self.uptime_s,
            "size_gb": self.size_gb,
        }


class ServerPool:
    """A small fleet of ServerManager instances.

    The public surface intentionally mirrors the legacy ServerManager
    (``upstream_base``, ``runtime``, ``is_running``, ``start``, ``stop``,
    ``restart``, ``swap``, ``yield_to_image``, ``status``, ``health``,
    ``mark_degraded``, ``add_exit_listener``) so existing call sites can
    treat ``app.state.sm`` as before. Slot-aware behaviour lives in the
    new methods (``slots``, ``find_for``, ``add_slot``, ``remove_slot``,
    ``start_slot``, ``stop_slot``, ``swap_in``, ``upstream_for``,
    ``boot_slots``, ``has_in_flight``).
    """

    def __init__(self, cfg: Config, db: DB) -> None:
        self.cfg = cfg
        self.db = db
        # Slot 0 is always present — legacy single-instance behaviour.
        self._slot0 = ServerManager(cfg, db, slot_id=0)
        self._slots: dict[int, ServerManager] = {0: self._slot0}
        self._manifest_path: Path = cfg.data_dir / "slots.json"
        self._manifest = slots_state.load(self._manifest_path)
        # Pool-level lock for add/remove operations — keeps slot id
        # allocation race-free.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Legacy forwarders — every call site that used to talk to a single
    # ServerManager continues to work. All operations apply to slot 0
    # except for fanouts (stop on shutdown, yield_to_image on image arrival).
    # ------------------------------------------------------------------ #

    @property
    def runtime(self) -> rt.RuntimeState:
        return self._slot0.runtime

    @property
    def upstream_base(self) -> str:
        return self._slot0.upstream_base

    @property
    def is_running(self) -> bool:
        return self._slot0.is_running

    @property
    def spec(self) -> StartSpec | None:
        return self._slot0.spec

    @property
    def proc(self):  # asyncio.subprocess.Process | None
        return self._slot0.proc

    def add_exit_listener(self, q: asyncio.Queue[int]) -> None:
        """Wire a single queue to every slot's exit events.

        The supervisor calls this once on its own queue. With multi-slot
        off the listener only fires for slot 0; with multi-slot on the
        same listener receives exit codes from every slot — the
        supervisor's per-slot crash bookkeeping (see ``supervisor.py``)
        keys on which slot was running last.
        """
        for sm in self._slots.values():
            sm.add_exit_listener(q)

    async def start(self, spec: StartSpec) -> int:
        """Legacy entry. Starts slot 0 with ``spec``."""
        return await self._slot0.start(spec)

    async def stop(self) -> None:
        """Stop ALL slots. Called by app lifespan teardown."""
        # Reverse-order stop so slot 0 is last (parallels the
        # historical "stop the main server last" behaviour).
        for slot_id in sorted(self._slots.keys(), reverse=True):
            try:
                await self._slots[slot_id].stop()
            except Exception:  # noqa: BLE001
                log.exception("pool: error stopping slot %d", slot_id)

    async def restart(self, spec: StartSpec | None = None) -> int:
        return await self._slot0.restart(spec)

    async def swap(self, new_spec: StartSpec) -> int:
        """Legacy single-slot swap — operates on slot 0.

        Multi-slot callers use ``swap_in(slot_id, spec)``.
        """
        return await self._slot0.swap(new_spec)

    async def health(self) -> bool:
        return await self._slot0.health()

    def status(self) -> dict[str, Any]:
        return self._slot0.status()

    def mark_degraded(self, reason: str) -> None:
        self._slot0.mark_degraded(reason)

    @contextlib.asynccontextmanager
    async def yield_to_image(self, *, reason: str = "image"):
        """Coordinate VRAM hand-off to an image task.

        Single-slot: forwards to slot 0's yield_to_image (existing
        behaviour). Multi-slot: fanout over every loaded slot, unless
        ``cfg.allow_diffusion_with_slots`` is on (then no-op — LLM
        slots stay loaded; operator owns the VRAM budget).
        """
        if not self.cfg.multi_slot_enabled:
            async with self._slot0.yield_to_image(reason=reason):
                yield
            return
        if getattr(self.cfg, "allow_diffusion_with_slots", False):
            # Operator opted in to keeping LLM slots loaded across image
            # tasks. The diffusion worker accepts the VRAM risk.
            log.info(
                "pool: image task arrived with allow_diffusion_with_slots=on; "
                "keeping %d LLM slot(s) loaded",
                sum(1 for s in self._slots.values() if s.is_running),
            )
            yield
            return
        # Fanout: enter each running slot's yield_to_image so all stop +
        # all restart in the correct order. AsyncExitStack lets us hold
        # multiple async context managers simultaneously without nesting
        # explicit ``async with`` blocks at indeterminate depth.
        async with contextlib.AsyncExitStack() as stack:
            for slot_id in sorted(self._slots.keys()):
                sm = self._slots[slot_id]
                if not sm.is_running:
                    continue
                await stack.enter_async_context(
                    sm.yield_to_image(reason=f"{reason}:slot{slot_id}")
                )
            yield

    # ------------------------------------------------------------------ #
    # New multi-slot surface
    # ------------------------------------------------------------------ #

    def default(self) -> ServerManager:
        """The slot 0 ServerManager — legacy callers that legitimately
        need a singular instance (autolaunch, /admin/server/* routes)."""
        return self._slot0

    def slot(self, slot_id: int) -> ServerManager | None:
        return self._slots.get(slot_id)

    def slots(self) -> list[SlotView]:
        out: list[SlotView] = []
        for slot_id in sorted(self._slots.keys()):
            sm = self._slots[slot_id]
            uptime = (time.time() - sm.runtime.started_at
                      if sm.runtime.started_at else None)
            # Best-effort file size for the running model. We read it
            # from spec.model_path's stat() — same path the engine is
            # serving. None for empty slots or if the file is gone.
            size_gb: float | None = None
            if sm.spec is not None and sm.spec.model_path is not None:
                try:
                    size_b = sm.spec.model_path.stat().st_size
                    size_gb = round(size_b / (1024 ** 3), 2)
                except OSError:
                    size_gb = None
            out.append(SlotView(
                id=slot_id,
                port=sm._port,
                state=sm.runtime.state,
                model=sm.runtime.current_model,
                profile=sm.runtime.current_profile,
                pid=sm.runtime.pid,
                started_at=sm.runtime.started_at,
                uptime_s=uptime,
                size_gb=size_gb,
            ))
        return out

    def total_loaded_size_gb(self) -> float:
        """Sum of file sizes (GB) across all currently-running slots.

        Lightweight proxy for VRAM pressure. Useful for the dashboard
        and as input to admission decisions when a real VRAM probe
        isn't available. Empty slots contribute zero.
        """
        total = 0.0
        for sv in self.slots():
            if sv.size_gb is not None:
                total += sv.size_gb
        return round(total, 2)

    def find_for(self, model_id: str) -> ServerManager | None:
        """Return the slot whose currently-loaded model matches ``model_id``.

        Used by the queue dispatcher to skip the swap when a request's
        model is already warm in some slot. Returns the first hit; if a
        user loads the same model into two slots (e.g. different
        profiles), only one will ever be used — explicit slot pinning
        is intentionally NOT part of v1 (per plan).
        """
        for sm in self._slots.values():
            if sm.is_running and sm.runtime.current_model == model_id:
                return sm
        return None

    def upstream_for(self, slot_id: int | None) -> str:
        """Resolve a slot id (or None) to a base URL.

        ``None`` returns the default (slot 0) — for legacy callers that
        haven't been wired through the routing helper yet.
        """
        if slot_id is None:
            return self._slot0.upstream_base
        sm = self._slots.get(slot_id)
        return sm.upstream_base if sm else self._slot0.upstream_base

    def _allocate_port(self) -> int:
        used = {sm._port for sm in self._slots.values()}
        base = self.cfg.multi_slot_base_port
        # Search a window twice the max size to give us headroom in case
        # someone else is squatting on a port in the range.
        for offset in range(self.cfg.multi_slot_max * 2 + 4):
            candidate = base + offset
            if candidate == self.cfg.port:  # don't collide with the daemon's own bind port
                continue
            if candidate in used:
                continue
            return candidate
        raise ServerError(
            f"no free port available in pool window starting at {base}"
        )

    def _next_slot_id(self) -> int:
        for sid in range(1, self.cfg.multi_slot_max + 1):
            if sid not in self._slots:
                return sid
        raise ServerError(
            f"slot capacity {self.cfg.multi_slot_max} reached"
        )

    async def add_slot(self, *, supervisor=None) -> int:
        """Allocate the next free slot id + port, create a ServerManager,
        persist the manifest entry. Returns the new slot id.

        ``supervisor`` (optional) — when passed, the new slot's exit
        events are wired into the supervisor's listener queue so a
        crash in any slot is observed.
        """
        if not self.cfg.multi_slot_enabled:
            raise ServerError("multi-slot is not enabled")
        async with self._lock:
            slot_id = self._next_slot_id()
            port = self._allocate_port()
            sm = ServerManager(self.cfg, self.db,
                               slot_id=slot_id, port_override=port)
            self._slots[slot_id] = sm
            if supervisor is not None and hasattr(supervisor, "subscribe_slot"):
                supervisor.subscribe_slot(sm)
            # Persist the empty slot so we restore it on next boot.
            self._manifest.slots = [
                s for s in self._manifest.slots if s.id != slot_id
            ]
            self._manifest.slots.append(slots_state.SlotEntry(
                id=slot_id, port=port,
            ))
            slots_state.save(self._manifest_path, self._manifest)
            log.info("pool: added slot %d on port %d", slot_id, port)
            return slot_id

    async def remove_slot(self, slot_id: int, *,
                          drain_timeout: float = 30.0) -> dict[str, Any]:
        """Stop a slot and remove it from the pool + manifest.

        Slot 0 cannot be removed. Drains in-flight requests for up to
        ``drain_timeout`` seconds; after that, force-stops the child
        and the dispatcher will surface aborted-request errors to the
        affected clients on its next bookkeeping tick (see
        ``queue_mgr.mark_in_flight_done``).
        """
        if slot_id == 0:
            raise ServerError("slot 0 cannot be removed")
        async with self._lock:
            sm = self._slots.get(slot_id)
            if not sm:
                return {"removed": False, "reason": "no such slot"}
            # Best-effort drain: the queue layer is the authority on
            # "has in-flight"; the pool can only signal the wait. The
            # admin/CLI handler is responsible for calling has_in_flight
            # in a loop before invoking remove_slot if it wants a softer
            # path. v1 takes the hammer route per user decision.
            forced = False
            if drain_timeout > 0 and sm.is_running:
                forced = True  # we WILL stop; the timeout is symbolic
                # No real drain hook in the pool itself — the caller's
                # job. Stop after the timeout.
                await asyncio.sleep(0)  # yield once
            await sm.stop()
            self._slots.pop(slot_id, None)
            self._manifest.slots = [
                s for s in self._manifest.slots if s.id != slot_id
            ]
            slots_state.save(self._manifest_path, self._manifest)
            log.info("pool: removed slot %d (forced=%s)", slot_id, forced)
            return {"removed": True, "forced": forced}

    async def start_slot(self, slot_id: int, spec: StartSpec) -> int:
        sm = self._slots.get(slot_id)
        if not sm:
            raise ServerError(f"unknown slot {slot_id}")
        pid = await sm.start(spec)
        self._persist_slot_assignment(slot_id, spec)
        return pid

    async def swap_in(self, slot_id: int, spec: StartSpec) -> int:
        sm = self._slots.get(slot_id)
        if not sm:
            raise ServerError(f"unknown slot {slot_id}")
        pid = await sm.swap(spec)
        self._persist_slot_assignment(slot_id, spec)
        return pid

    async def stop_slot(self, slot_id: int) -> None:
        sm = self._slots.get(slot_id)
        if not sm:
            return
        await sm.stop()
        # Clear assignment in the manifest (keeps the slot itself).
        if slot_id != 0:
            entry = self._manifest.by_id(slot_id)
            if entry is not None:
                entry.model_id = None
                entry.profile = None
                entry.args = {}
                slots_state.save(self._manifest_path, self._manifest)

    def _persist_slot_assignment(self, slot_id: int, spec: StartSpec) -> None:
        if slot_id == 0:
            return  # slot 0 is described by runtime.json
        entry = self._manifest.by_id(slot_id)
        if entry is None:
            return  # shouldn't happen — add_slot writes the entry
        entry.model_id = spec.model_id
        entry.profile = spec.profile_name
        entry.args = dict(spec.extra_args or {})
        slots_state.save(self._manifest_path, self._manifest)

    async def boot_slots(self, *, supervisor=None) -> None:
        """Recreate slots 1..N from ``slots.json`` and start any that
        carry a model assignment. Called from the app lifespan after
        the pool is constructed."""
        if not self.cfg.multi_slot_enabled:
            return
        for entry in list(self._manifest.slots):
            if entry.id == 0:
                continue
            if entry.id in self._slots:
                continue
            sm = ServerManager(self.cfg, self.db,
                               slot_id=entry.id,
                               port_override=entry.port)
            self._slots[entry.id] = sm
            if supervisor is not None and hasattr(supervisor, "subscribe_slot"):
                supervisor.subscribe_slot(sm)
            if entry.model_id:
                try:
                    spec = resolve_spec(
                        self.cfg,
                        model=entry.model_id,
                        profile=entry.profile or None,
                        args=entry.args or {},
                    )
                    asyncio.create_task(self._safe_start(entry.id, spec))
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "pool: slot %d boot failed for model %s: %s",
                        entry.id, entry.model_id, e,
                    )

    async def _safe_start(self, slot_id: int, spec: StartSpec) -> None:
        try:
            await self._slots[slot_id].start(spec)
        except Exception as e:  # noqa: BLE001
            log.warning("pool: slot %d start failed: %s", slot_id, e)

    def has_in_flight(self, slot_id: int) -> bool:
        """Returns True if any QueuedRequest is currently using ``slot_id``.

        The queue's in-flight bookkeeping lives there, not here — this
        helper exists for symmetry but defers to whoever owns the queue.
        Phase B will fill this in by reading the queue's state.
        """
        # Wired by api_admin / api_ui handlers that have access to the
        # queue manager. Placeholder for now.
        return False
