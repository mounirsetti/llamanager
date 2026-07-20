"""Memory guardrails — keep an inference launch from making the box unusable.

Background
----------
A model that *loads* comfortably can still exhaust memory *during use*. Two
mechanisms drove a real incident (see the 2026-06-04 logs):

* **KV cache.** llama.cpp reserves KV for the full ``--ctx-size``. For a model
  with little or no GQA the KV cache can be enormous — far past a 32 GB GPU
  *and* 61 GB of system RAM. It is very model-specific, so it must be read
  from the GGUF geometry rather than assumed: gemma-4-31B was long mis-sized
  here as ``head_count_kv = 32`` (a scalar fallback taken because the real
  value is a *per-layer array*), giving a phantom 15 GB at ctx 4096. Its true
  KV is ~1.1 GB — it interleaves 50 sliding-window layers capped at a
  1024-token window with only 10 global layers at 4 KV heads. See
  ``gguf_meta._kv_cache_gb``.
* **Host prompt cache.** llama-server keeps past prompts' KV checkpoints in
  host RAM (``--cache-ram``, default 8192 MiB) so long prompts can be reused.
  With huge contexts a single checkpoint set ran to 25 GB, and several
  coexisted — pushing the host into swap thrash.

This module is the math + monitoring behind the guardrails. It is deliberately
free of app/HTTP/asyncio-wiring concerns beyond the small ``MemoryWatchdog``
helper, whose *actions* are injected so policy lives in the caller:

* sizing helpers (``kv_cache_gb`` / ``safe_max_ctx`` / ``ctx_safety``) reuse
  the GGUF metadata reader and are pure — unit-testable without a GPU;
* ``read_mem_state`` / ``classify_pressure`` read host RAM + swap via psutil;
* ``MemoryWatchdog`` samples pressure on an interval and calls back when it
  crosses a soft/hard threshold — the caller decides what to do.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import psutil

from .gguf_meta import GgufMeta, _kv_cache_gb, kv_bytes_per_elem, read_gguf_meta

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live GPU memory (amdgpu sysfs)
# ---------------------------------------------------------------------------
#
# Why this exists: on 2026-07-16 a wedged llama-server was SIGKILLed and the
# amdgpu/KFD driver never reclaimed its context — 26 GB stayed pinned (device
# VRAM first, then host RAM via GTT) attributed to a PID that no longer
# existed. Every later engine launch saw ~7.5 GB free of 32 GB, silently built
# a mostly-CPU config, and ran at 0.5 tok/s until every request timed out. The
# box looked "fine" the whole time. These probes let us *notice*.
#
# All files below are world-readable, so no root/rocm-smi dependency.

_DRM_CLASS = Path("/sys/class/drm")
_KFD_PROC = Path("/sys/class/kfd/kfd/proc")
_AMD_VENDOR = "0x1002"


@dataclass(frozen=True)
class GpuMem:
    """Live memory readings for one AMD GPU, in GB.

    ``gtt_used`` is *host* RAM pinned by the GPU driver for the card to
    address. It is not VRAM, it cannot be swapped, and a leak there is
    invisible to VRAM-only tools like ``rocm-smi --showmeminfo vram`` — which
    is exactly why the 2026-07-16 leak was mistaken for "the GPU is fine".
    """
    card: str
    vram_used: float
    vram_total: float
    gtt_used: float
    gtt_total: float

    @property
    def vram_free(self) -> float:
        return max(0.0, self.vram_total - self.vram_used)


def _read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def amd_card_dirs() -> list[Path]:
    """Device dirs for AMD GPUs, e.g. ``/sys/class/drm/card1/device``."""
    out: list[Path] = []
    try:
        candidates = sorted(_DRM_CLASS.glob("card[0-9]*/device"))
    except OSError:
        return out
    for d in candidates:
        try:
            if (d / "vendor").read_text().strip() == _AMD_VENDOR:
                out.append(d)
        except OSError:
            continue
    return out


def read_gpu_mem() -> list[GpuMem]:
    """Live VRAM + GTT usage per AMD card. Empty list if unreadable."""
    gb = float(1024 ** 3)
    out: list[GpuMem] = []
    for d in amd_card_dirs():
        vu = _read_int_file(d / "mem_info_vram_used")
        vt = _read_int_file(d / "mem_info_vram_total")
        gu = _read_int_file(d / "mem_info_gtt_used")
        gt = _read_int_file(d / "mem_info_gtt_total")
        if vu is None or vt is None:
            continue
        out.append(GpuMem(card=d.parent.name,
                          vram_used=vu / gb, vram_total=vt / gb,
                          gtt_used=(gu or 0) / gb, gtt_total=(gt or 0) / gb))
    return out


def kfd_proc_pids() -> set[int]:
    """PIDs the KFD driver currently holds a compute context for."""
    try:
        return {int(p.name) for p in _KFD_PROC.iterdir() if p.name.isdigit()}
    except OSError:
        return set()


def stale_kfd_pids() -> set[int]:
    """KFD contexts whose owning process no longer exists.

    A non-empty result means the driver leaked a GPU context: the memory is
    unreachable from userspace and only a GPU reset or reboot frees it. Both
    are disruptive, so callers should *report* this, never act on it.
    """
    return {pid for pid in kfd_proc_pids() if not Path(f"/proc/{pid}").exists()}


# ---------------------------------------------------------------------------
# Sizing math (pure — reuses the GGUF metadata reader)
# ---------------------------------------------------------------------------

# VRAM the engine needs ON TOP of weights + KV + projector. This is NOT the
# context — the KV cache (which *is* the context) and the mmproj are each
# counted explicitly elsewhere. What's left is llama's compute (activation)
# buffer for a forward pass, the backend context, and allocator
# fragmentation. Unlike KV, this does not grow with context length — it's
# roughly fixed per model/batch — so we model it as a small base plus a
# fragmentation fraction, NOT a flat percentage of the card. (The old flat
# 12% over-reserved big cards: ~3.8 GB on a 32 GB GPU vs the ~1-2 GB really
# used, which wrongly flagged models that comfortably fit.)
_GPU_OVERHEAD_BASE_GB = 1.0          # compute/activation buffer, per slot
_GPU_FRAGMENTATION_FRACTION = 0.04   # allocator slack; scales mildly with size
# llama-server picks --parallel automatically (typically 4) when the operator
# doesn't pin it. Each slot reserves its own compute buffer, so for a fit
# estimate we assume this many slots unless a profile sets --parallel. Kept in
# sync with the live JS estimate in templates/base.html.
DEFAULT_PARALLEL_SLOTS = 4
# RAM kept free for the OS when judging whether a model fits VRAM+RAM at all
# (below this it would thrash swap = "danger").
_HOST_RESERVE_GB = 4.0
# Floor so a safe-ctx suggestion never returns something absurdly tiny when the
# weights already nearly fill the budget.
_MIN_SUGGESTED_CTX = 2048


def _gpu_overhead_gb(vram_gb: float, n_parallel: int = 1) -> float:
    """Non-weight, non-KV VRAM the engine reserves (compute buffer + slack).

    The compute/activation buffer is roughly one per request slot, so it scales
    with ``--parallel``; the fragmentation slack scales mildly with card size.
    This is a conservative approximation, not an exact llama.cpp accounting.
    """
    slots = max(1, int(n_parallel or 1))
    return (_GPU_OVERHEAD_BASE_GB * slots
            + max(0.0, vram_gb) * _GPU_FRAGMENTATION_FRACTION)


def usable_vram_gb(vram_gb: float, n_parallel: int = 1) -> float:
    """VRAM available for weights + KV + projector, after engine overhead."""
    return max(0.0, vram_gb - _gpu_overhead_gb(vram_gb, n_parallel))


def _load_meta(model_path: Path) -> GgufMeta | None:
    try:
        if Path(model_path).is_file():
            return read_gguf_meta(Path(model_path))
    except Exception as e:  # noqa: BLE001 — sizing is best-effort, never fatal
        log.debug("gguf meta read failed for %s: %s", model_path, e)
    return None


def kv_cache_gb(model_path: Path, ctx_size: int,
                meta: GgufMeta | None = None,
                kv_cache_type: str = "",
                include_recurrent: bool = True) -> float | None:
    """Estimated KV-cache size (GB) for ``ctx_size`` tokens, or None.

    ``kv_cache_type`` ("" / "f16" / "q8_0" / "q5_1" / "q4_0") scales the
    per-element size; blank means the f16 default. ``include_recurrent=False``
    drops the constant SSM-state term so the result is purely ctx-proportional
    (for callers that derive a per-token rate — see ``_kv_cache_gb``).
    """
    meta = meta or _load_meta(model_path)
    if meta is None:
        return None
    return _kv_cache_gb(meta, ctx_size,
                        bytes_per_elem=kv_bytes_per_elem(kv_cache_type),
                        include_recurrent=include_recurrent)


def weights_gb(model_path: Path, meta: GgufMeta | None = None) -> float:
    """On-disk weight size in GB (what full GPU offload must hold)."""
    meta = meta or _load_meta(model_path)
    if meta and meta.file_size:
        return meta.file_size / (1024 ** 3)
    try:
        return Path(model_path).stat().st_size / (1024 ** 3)
    except OSError:
        return 0.0


def file_gb(path: Path | str) -> float:
    """Size of a file on disk in GB (0 if missing). Used to estimate the VRAM
    a vision projector (mmproj) adds — it loads roughly its own file size."""
    try:
        return Path(path).stat().st_size / (1024 ** 3)
    except OSError:
        return 0.0


def safe_max_ctx(model_path: Path, budget_gb: float,
                 meta: GgufMeta | None = None,
                 extra_gb: float = 0.0,
                 kv_cache_type: str = "",
                 n_parallel: int = 1) -> int | None:
    """Largest ctx whose weights+KV (+``extra_gb``) fit in ``budget_gb``.

    ``extra_gb`` is any fixed footprint beyond the weights that also lives in
    the budget — most often a vision projector (mmproj). ``kv_cache_type``
    scales the KV per token; ``n_parallel`` scales the engine compute-buffer
    reservation. Returns None when we lack the metadata to compute KV per
    token. The result is rounded down to a multiple of 256 (llama's ctx
    granularity) and floored at ``_MIN_SUGGESTED_CTX``.
    """
    meta = meta or _load_meta(model_path)
    if meta is None:
        return None
    probe = kv_cache_gb(model_path, 1, meta=meta, kv_cache_type=kv_cache_type)
    if not probe or probe <= 0:
        return None
    budget = (usable_vram_gb(budget_gb, n_parallel)
              - weights_gb(model_path, meta=meta) - max(0.0, extra_gb))
    if budget <= 0:
        return _MIN_SUGGESTED_CTX

    # Invert the KV curve by search rather than dividing the budget by a
    # per-token rate. KV is monotonic in ctx but NOT proportional to it:
    # sliding-window layers stop growing once ctx passes the window, and
    # hybrid SSM layers never grow at all. Extrapolating a rate measured at
    # ctx=1 (where every layer still counts) understated gemma-4's usable
    # context by ~6x — suggesting 11k on a card that comfortably holds 64k.
    def kv_at(ctx: int) -> float:
        return kv_cache_gb(model_path, ctx, meta=meta,
                           kv_cache_type=kv_cache_type) or 0.0

    hi = meta.context_length or 1_048_576
    if kv_at(hi) <= budget:
        return max(_MIN_SUGGESTED_CTX, (hi // 256) * 256)
    lo = 0
    while hi - lo > 256:
        mid = (lo + hi) // 2
        if kv_at(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return max(_MIN_SUGGESTED_CTX, (lo // 256) * 256)


@dataclass(frozen=True)
class CtxSafety:
    """Verdict on whether a (model, ctx) pairing is safe for the hardware."""
    level: str                 # "ok" | "warn" | "danger"
    kv_gb: float | None        # estimated KV at the requested ctx
    weights_gb: float          # weight footprint
    budget_gb: float           # the memory budget we compared against
    safe_ctx: int | None       # largest ctx that fits the budget
    message: str               # human-readable, ready for a log line or toast

    @property
    def is_unsafe(self) -> bool:
        return self.level in ("warn", "danger")


def ctx_safety(model_path: Path, ctx_size: int | None,
               vram_gb: float | None, ram_gb: float | None,
               extra_gb: float = 0.0,
               kv_cache_type: str = "",
               n_parallel: int = 1) -> CtxSafety | None:
    """Classify a ctx choice against the GPU (or, if larger, GPU+RAM) budget.

    ``danger`` — weights+KV won't even fit GPU+RAM (will spill to swap/OOM).
    ``warn``   — fits RAM but not VRAM alone (forces a big CPU spill / slow).
    ``ok``     — fits comfortably in VRAM.

    ``extra_gb`` is added to the weight footprint — pass the vision projector
    (mmproj) size here when the profile uses one, since it also occupies VRAM.
    ``kv_cache_type`` scales the KV footprint; ``n_parallel`` scales the engine
    compute-buffer reservation.

    Returns None when ctx or metadata is missing (nothing to say).
    """
    if not ctx_size:
        return None
    meta = _load_meta(model_path)
    if meta is None:
        return None
    kv = kv_cache_gb(model_path, ctx_size, meta=meta, kv_cache_type=kv_cache_type)
    if kv is None:
        return None
    extra = max(0.0, extra_gb)
    w = weights_gb(model_path, meta=meta)
    base = w + extra                 # weights + projector
    vram = vram_gb or 0.0
    ram = ram_gb or 0.0
    need = base + kv
    vram_budget = usable_vram_gb(vram, n_parallel)          # after engine overhead
    total_budget = usable_vram_gb(vram, n_parallel) + max(0.0, ram - _HOST_RESERVE_GB)
    safe = (safe_max_ctx(model_path, vram, meta=meta, extra_gb=extra,
                         kv_cache_type=kv_cache_type, n_parallel=n_parallel)
            if vram else None)
    # Spell out the projector and KV quant in the breakdown when present.
    kv_desc = f"KV {kv:.1f}" + (f" ({kv_cache_type})"
                               if kv_cache_type and kv_cache_type != "f16" else "")
    base_desc = (f"weights {w:.1f} + mmproj {extra:.1f} + {kv_desc}"
                 if extra > 0.05 else f"weights {w:.1f} + {kv_desc}")

    if vram and need <= vram_budget:
        level = "ok"
        msg = (f"ctx {ctx_size:,} needs ~{need:.1f} GB ({base_desc}); "
               f"fits {vram:.0f} GB VRAM.")
    elif vram and need <= vram:
        # Fits the raw card but eats into the compute-buffer headroom — tight.
        level = "warn"
        msg = (f"ctx {ctx_size:,} needs ~{need:.1f} GB ({base_desc}) — tight for "
               f"{vram:.0f} GB VRAM once compute buffers are added; may spill "
               f"to RAM.")
        if safe:
            msg += f" A ctx around {safe:,} keeps comfortable headroom."
    elif need <= total_budget and total_budget > 0:
        level = "warn"
        over = (f"over {vram:.0f} GB VRAM" if vram
                else "may not fit VRAM")          # vram unknown (not detected)
        msg = (f"ctx {ctx_size:,} needs ~{need:.1f} GB ({base_desc}) — {over}, "
               f"will spill to RAM and run slowly.")
        if safe:
            msg += f" A ctx around {safe:,} would stay on the GPU."
    else:
        level = "danger"
        budget_desc = (f"VRAM+RAM ({vram:.0f}+{ram:.0f} GB)" if vram
                       else f"available memory ({ram:.0f} GB RAM)")
        msg = (f"ctx {ctx_size:,} needs ~{need:.1f} GB ({base_desc}) — more "
               f"than {budget_desc}. This will exhaust memory and thrash swap.")
        if safe:
            msg += f" Reduce ctx to about {safe:,} or lower."
    return CtxSafety(level=level, kv_gb=kv, weights_gb=w,
                     budget_gb=vram_budget, safe_ctx=safe, message=msg)


# ---------------------------------------------------------------------------
# Host prompt-cache sizing
# ---------------------------------------------------------------------------

# llama-server's --cache-ram default. We never raise above this; we only lower
# it toward what the host can actually spare.
_LLAMA_DEFAULT_CACHE_RAM_MIB = 8192
_MIN_CACHE_RAM_MIB = 512


def derive_cache_ram_mib(available_ram_gb: float,
                         fraction: float = 0.25,
                         ceiling_mib: int = _LLAMA_DEFAULT_CACHE_RAM_MIB) -> int:
    """A host prompt-cache cap (MiB) derived from currently-available RAM.

    The default 8 GB cache is a lot on a memory-tight box already holding a
    big model. We bound it to ``fraction`` of available RAM, never above the
    engine default and never below a small floor.
    """
    derived = int(available_ram_gb * 1024 * fraction)
    return max(_MIN_CACHE_RAM_MIB, min(ceiling_mib, derived))


# ---------------------------------------------------------------------------
# Live memory state + pressure
# ---------------------------------------------------------------------------

class Pressure(IntEnum):
    OK = 0
    WARN = 1      # soft threshold — getting tight, act gently
    CRITICAL = 2  # hard threshold — act now to avoid thrash


@dataclass(frozen=True)
class MemThresholds:
    """Pressure thresholds. Defaults are conservative for a desktop box."""
    warn_avail_frac: float = 0.15     # WARN when available RAM < 15% of total
    crit_avail_frac: float = 0.07     # CRITICAL when < 7%
    warn_swap_frac: float = 0.25      # WARN when swap used > 25% of swap
    crit_swap_frac: float = 0.60      # CRITICAL when > 60%

    @classmethod
    def from_cfg(cls, cfg: object) -> "MemThresholds":
        g = lambda name, d: float(getattr(cfg, name, d))  # noqa: E731
        return cls(
            warn_avail_frac=g("mem_warn_avail_frac", cls.warn_avail_frac),
            crit_avail_frac=g("mem_crit_avail_frac", cls.crit_avail_frac),
            warn_swap_frac=g("mem_warn_swap_frac", cls.warn_swap_frac),
            crit_swap_frac=g("mem_crit_swap_frac", cls.crit_swap_frac),
        )


@dataclass(frozen=True)
class MemState:
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    swap_used_gb: float

    @property
    def ram_available_frac(self) -> float:
        return (self.ram_available_gb / self.ram_total_gb
                if self.ram_total_gb else 1.0)

    @property
    def swap_used_frac(self) -> float:
        return (self.swap_used_gb / self.swap_total_gb
                if self.swap_total_gb else 0.0)

    def summary(self) -> str:
        return (f"RAM {self.ram_available_gb:.1f}/{self.ram_total_gb:.1f} GB free, "
                f"swap {self.swap_used_gb:.1f}/{self.swap_total_gb:.1f} GB used")


def read_mem_state() -> MemState:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    gb = lambda b: b / (1024 ** 3)  # noqa: E731
    return MemState(
        ram_total_gb=gb(vm.total),
        ram_available_gb=gb(vm.available),
        swap_total_gb=gb(sw.total),
        swap_used_gb=gb(sw.used),
    )


def classify_pressure(state: MemState, th: MemThresholds) -> Pressure:
    """Map a memory snapshot to a pressure level (the worse of RAM/swap)."""
    level = Pressure.OK
    if state.ram_available_frac < th.crit_avail_frac:
        level = Pressure.CRITICAL
    elif state.ram_available_frac < th.warn_avail_frac:
        level = max(level, Pressure.WARN)
    # Swap is only meaningful pressure if swap actually exists.
    if state.swap_total_gb > 0:
        if state.swap_used_frac > th.crit_swap_frac:
            level = Pressure.CRITICAL
        elif state.swap_used_frac > th.warn_swap_frac:
            level = max(level, Pressure.WARN)
    return Pressure(level)


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

class MemoryWatchdog:
    """Sample host memory on an interval and report pressure transitions.

    Policy is injected. ``on_pressure(level, state)`` is awaited whenever the
    pressure level *rises* (OK→WARN, WARN→CRITICAL, OK→CRITICAL) and once when
    it falls back to OK (so the caller can clear any "act before next task"
    flag). It is intentionally edge-triggered, not fired every tick, so the
    caller isn't spammed while memory sits in the WARN band.
    """

    def __init__(self, cfg: object, *, on_pressure, interval_s: float = 5.0):
        self.cfg = cfg
        self._on_pressure = on_pressure
        self.interval_s = float(getattr(cfg, "mem_guard_interval_s", interval_s))
        self.enabled = bool(getattr(cfg, "mem_guard_enabled", True))
        self._task: asyncio.Task | None = None
        self._last = Pressure.OK
        # While memory stays CRITICAL, re-fire the callback on this cadence so
        # the caller can *escalate* reclaim if the first action didn't relieve
        # pressure (the box is actively thrashing — one edge-triggered callback
        # isn't enough). 0 disables the repeat (pure edge-triggered).
        self._crit_repeat_s = float(getattr(cfg, "mem_crit_repeat_s", 15.0))
        self._crit_elapsed = 0.0

    @property
    def last_level(self) -> Pressure:
        return self._last

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        t, self._task = self._task, None
        if t is not None:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _run(self) -> None:
        log.info("memory watchdog: started (interval %.1fs)", self.interval_s)
        th = MemThresholds.from_cfg(self.cfg)
        while True:
            try:
                state = read_mem_state()
                level = classify_pressure(state, th)
                if level != self._last:
                    prev, self._last = self._last, level
                    self._crit_elapsed = 0.0
                    # Report on any rise, and on the return to OK.
                    if level > prev or level == Pressure.OK:
                        await self._emit(level, state)
                elif level == Pressure.CRITICAL and self._crit_repeat_s > 0:
                    # Sustained CRITICAL — re-fire on the repeat cadence so the
                    # caller keeps (and escalates) reclaim while the box thrashes.
                    self._crit_elapsed += self.interval_s
                    if self._crit_elapsed >= self._crit_repeat_s:
                        self._crit_elapsed = 0.0
                        await self._emit(level, state)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — never let the loop die
                log.debug("memory watchdog tick failed: %s", e)
            await asyncio.sleep(self.interval_s)

    async def _emit(self, level: Pressure, state: MemState) -> None:
        if level == Pressure.CRITICAL:
            log.warning("memory CRITICAL: %s", state.summary())
        elif level == Pressure.WARN:
            log.warning("memory tight: %s", state.summary())
        else:
            log.info("memory recovered: %s", state.summary())
        try:
            res = self._on_pressure(level, state)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:  # noqa: BLE001 — a bad callback must not kill us
            log.error("memory watchdog callback failed: %s", e)
