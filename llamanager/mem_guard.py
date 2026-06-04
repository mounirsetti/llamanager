"""Memory guardrails — keep an inference launch from making the box unusable.

Background
----------
A model that *loads* comfortably can still exhaust memory *during use*. Two
mechanisms drove a real incident (see the 2026-06-04 logs):

* **KV cache.** llama.cpp reserves KV for the full ``--ctx-size``. For a model
  with little or no GQA (e.g. gemma-4-31B has ``head_count_kv = 32``) the KV
  cache is enormous — ~77 GB at ctx 64k, ~315 GB at ctx 262k — far past a
  32 GB GPU *and* 61 GB of system RAM.
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

from .gguf_meta import GgufMeta, _kv_cache_gb, read_gguf_meta

log = logging.getLogger(__name__)


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
_GPU_OVERHEAD_BASE_GB = 1.0          # compute/activation buffer (≈ fixed)
_GPU_FRAGMENTATION_FRACTION = 0.04   # allocator slack; scales mildly with size
# RAM kept free for the OS when judging whether a model fits VRAM+RAM at all
# (below this it would thrash swap = "danger").
_HOST_RESERVE_GB = 4.0
# Floor so a safe-ctx suggestion never returns something absurdly tiny when the
# weights already nearly fill the budget.
_MIN_SUGGESTED_CTX = 2048


def _gpu_overhead_gb(vram_gb: float) -> float:
    """Non-weight, non-KV VRAM the engine reserves (compute buffer + slack)."""
    return _GPU_OVERHEAD_BASE_GB + max(0.0, vram_gb) * _GPU_FRAGMENTATION_FRACTION


def usable_vram_gb(vram_gb: float) -> float:
    """VRAM available for weights + KV + projector, after engine overhead."""
    return max(0.0, vram_gb - _gpu_overhead_gb(vram_gb))


def _load_meta(model_path: Path) -> GgufMeta | None:
    try:
        if Path(model_path).is_file():
            return read_gguf_meta(Path(model_path))
    except Exception as e:  # noqa: BLE001 — sizing is best-effort, never fatal
        log.debug("gguf meta read failed for %s: %s", model_path, e)
    return None


def kv_cache_gb(model_path: Path, ctx_size: int,
                meta: GgufMeta | None = None) -> float | None:
    """Estimated fp16 KV-cache size (GB) for ``ctx_size`` tokens, or None."""
    meta = meta or _load_meta(model_path)
    if meta is None:
        return None
    return _kv_cache_gb(meta, ctx_size)


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
                 extra_gb: float = 0.0) -> int | None:
    """Largest ctx whose weights+KV (+``extra_gb``) fit in ``budget_gb``.

    ``extra_gb`` is any fixed footprint beyond the weights that also lives in
    the budget — most often a vision projector (mmproj). Returns None when we
    lack the metadata to compute KV per token. The result is rounded down to a
    multiple of 256 (llama's ctx granularity) and floored at
    ``_MIN_SUGGESTED_CTX``.
    """
    meta = meta or _load_meta(model_path)
    if meta is None:
        return None
    per_tok_gb = kv_cache_gb(model_path, 1, meta=meta)
    if not per_tok_gb or per_tok_gb <= 0:
        return None
    budget = (usable_vram_gb(budget_gb)
              - weights_gb(model_path, meta=meta) - max(0.0, extra_gb))
    if budget <= 0:
        return _MIN_SUGGESTED_CTX
    raw = int(budget / per_tok_gb)
    return max(_MIN_SUGGESTED_CTX, (raw // 256) * 256)


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
               extra_gb: float = 0.0) -> CtxSafety | None:
    """Classify a ctx choice against the GPU (or, if larger, GPU+RAM) budget.

    ``danger`` — weights+KV won't even fit GPU+RAM (will spill to swap/OOM).
    ``warn``   — fits RAM but not VRAM alone (forces a big CPU spill / slow).
    ``ok``     — fits comfortably in VRAM.

    ``extra_gb`` is added to the weight footprint — pass the vision projector
    (mmproj) size here when the profile uses one, since it also occupies VRAM.

    Returns None when ctx or metadata is missing (nothing to say).
    """
    if not ctx_size:
        return None
    meta = _load_meta(model_path)
    if meta is None:
        return None
    kv = kv_cache_gb(model_path, ctx_size, meta=meta)
    if kv is None:
        return None
    extra = max(0.0, extra_gb)
    w = weights_gb(model_path, meta=meta)
    base = w + extra                 # weights + projector
    vram = vram_gb or 0.0
    ram = ram_gb or 0.0
    need = base + kv
    vram_budget = usable_vram_gb(vram)                      # after engine overhead
    total_budget = usable_vram_gb(vram) + max(0.0, ram - _HOST_RESERVE_GB)
    safe = safe_max_ctx(model_path, vram, meta=meta, extra_gb=extra) if vram else None
    # Spell out the projector in the breakdown only when there is one.
    base_desc = (f"weights {w:.1f} + mmproj {extra:.1f} + KV {kv:.1f}"
                 if extra > 0.05 else f"weights {w:.1f} + KV {kv:.1f}")

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
        msg = (f"ctx {ctx_size:,} needs ~{need:.1f} GB ({base_desc}) — over "
               f"{vram:.0f} GB VRAM, will spill to RAM and run slowly.")
        if safe:
            msg += f" A ctx around {safe:,} would stay on the GPU."
    else:
        level = "danger"
        msg = (f"ctx {ctx_size:,} needs ~{need:.1f} GB ({base_desc}) — more "
               f"than VRAM+RAM ({vram:.0f}+{ram:.0f} GB). This will exhaust "
               f"memory and thrash swap.")
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
                    # Report on any rise, and on the return to OK.
                    if level > prev or level == Pressure.OK:
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
