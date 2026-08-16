"""Tests for the memory guardrails (mem_guard).

Covers the pure sizing math (KV estimate, safe-ctx, ctx-safety verdict), the
host prompt-cache derivation, the RAM/swap pressure classification, and the
watchdog's edge-triggered callback. None of it touches a real GPU or model —
GGUF metadata and memory readings are synthesised.
"""
from __future__ import annotations

import asyncio

import dataclasses

import pytest

from llamanager import mem_guard as mg
from llamanager.gguf_meta import GgufMeta


# A synthetic no-GQA 31B-ish model (head_count_kv == head_count): KV is huge.
# NOTE: this is NOT gemma-4 — that model was long mis-modelled here as no-GQA
# because its head_count_kv is a per-layer *array* the parser skipped, leaving
# a scalar fallback. Real gemma-4 geometry is in SWA_GEMMA4 below.
NO_GQA = GgufMeta(architecture="synthetic-mha", block_count=60,
                  embedding_length=5376,
                  head_count=42, head_count_kv=42, context_length=131072,
                  file_size=int(20.4 * 1024**3))
# A GQA MoE: head_count_kv much smaller than head_count -> tiny KV.
GQA = GgufMeta(architecture="qwen", block_count=40, embedding_length=4096,
               head_count=32, head_count_kv=2, context_length=262144,
               file_size=int(19.7 * 1024**3))


# ---------- sizing math ----------

def test_kv_grows_linearly_with_ctx():
    a = mg.kv_cache_gb("x", 8192, meta=NO_GQA)
    b = mg.kv_cache_gb("x", 16384, meta=NO_GQA)
    assert a and b and abs(b - 2 * a) < 1e-6


def test_no_gqa_kv_dwarfs_gqa():
    """Same ctx, the no-GQA model's KV must be far larger — this is the whole
    'be considerate of the model, not just the number' point."""
    big = mg.kv_cache_gb("x", 32768, meta=NO_GQA)
    small = mg.kv_cache_gb("x", 32768, meta=GQA)
    assert big > small * 5


# A model that declares an explicit head_dim (key_length/value_length) decoupled
# from embedding_length // head_count — like Qwen3.6-27B (head_dim 256, not
# 5120//24=213). KV sizing must read it rather than derive the wrong 213.
EXPLICIT_HEAD_DIM = GgufMeta(architecture="qwen35", block_count=64,
                             embedding_length=5120, head_count=24,
                             head_count_kv=4, key_length=256, value_length=256,
                             context_length=262144, file_size=int(20.6 * 1024**3))


def test_kv_uses_explicit_head_dim_not_embed_ratio():
    """When the model declares key_length/value_length, KV must use it, not
    embedding_length // head_count (which would be 213 vs the real 256)."""
    derived = GgufMeta(architecture="qwen35", block_count=64,
                       embedding_length=5120, head_count=24, head_count_kv=4,
                       context_length=262144, file_size=int(20.6 * 1024**3))
    explicit = mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM)
    fallback = mg.kv_cache_gb("x", 110336, meta=derived)
    # 256/213 larger than the embed//head_count fallback.
    assert explicit and fallback and explicit > fallback
    assert abs(explicit / fallback - 256 / 213) < 0.02
    # Sanity: ~26.9 GiB f16 at 110k for this model.
    assert 26.0 < explicit < 28.0


# Real gemma-4-31B geometry, read from the shipped GGUF: 60 layers in a
# repeating 5-local + 1-global pattern; local (sliding-window) layers use
# 16 KV heads x 256 dim capped at a 1024-token window, global layers 4 KV
# heads x 512 dim over the full context.
SWA_GEMMA4 = GgufMeta(
    architecture="gemma4", block_count=60, embedding_length=5376,
    head_count=32, head_count_kv=16, key_length=512, value_length=512,
    key_length_swa=256, value_length_swa=256, sliding_window=1024,
    sliding_window_pattern=[(i % 6) != 5 for i in range(60)],
    head_count_kv_per_layer=[4 if (i % 6) == 5 else 16 for i in range(60)],
    context_length=262144, file_size=int(20.4 * 1024**3))


def test_swa_kv_is_sublinear_and_small():
    """Sliding-window layers cap their KV at the window, so KV must grow far
    slower than linearly — doubling ctx must NOT double KV."""
    a = mg.kv_cache_gb("x", 32768, meta=SWA_GEMMA4)
    b = mg.kv_cache_gb("x", 65536, meta=SWA_GEMMA4)
    assert a and b
    assert b < 2 * a          # sublinear: only the 10 global layers scale
    # ~1.1 GB at 4k and ~5.8 GB at 64k for this model, f16.
    small = mg.kv_cache_gb("x", 4096, meta=SWA_GEMMA4)
    assert small and 0.8 < small < 1.5
    assert 5.0 < b < 6.5


def test_swa_model_fits_card_that_naive_sizing_rejects():
    """Regression for the phantom that made the guardrail reject gemma-4 at
    ctx 4096: modelling every layer as full-context attention over-counted KV
    by ~13x and claimed 35 GB on a model that really needs ~21.5 GB."""
    naive = GgufMeta(architecture="gemma4", block_count=60,
                     embedding_length=5376, head_count=32, head_count_kv=32,
                     key_length=512, value_length=512,
                     context_length=262144, file_size=int(20.4 * 1024**3))
    real = mg.kv_cache_gb("x", 4096, meta=SWA_GEMMA4)
    phantom = mg.kv_cache_gb("x", 4096, meta=naive)
    assert real and phantom
    assert phantom > real * 10
    weights = 20.4
    assert weights + real < 31.0          # comfortably fits a 32 GB card
    assert weights + phantom > 32.0       # what the old math wrongly claimed


def test_safe_max_ctx_inverts_the_real_kv_curve(monkeypatch):
    """safe_max_ctx must search the actual KV curve, not divide the budget by
    a per-token rate sampled at ctx=1. For an SWA model every layer counts at
    ctx=1 but only the sparse global layers keep growing, so the rate-based
    version understated gemma-4's usable context by ~6x (11k vs ~112k)."""
    monkeypatch.setattr(mg, "_load_meta", lambda p: SWA_GEMMA4)
    monkeypatch.setattr(mg, "weights_gb", lambda p, meta=None: 20.4)
    got = mg.safe_max_ctx("x", 32.0, meta=SWA_GEMMA4)
    assert got is not None
    assert got > 60000, f"suggested only {got}; SWA ctx headroom lost again"
    # And the suggestion must genuinely fit the budget it was derived from.
    kv = mg.kv_cache_gb("x", got, meta=SWA_GEMMA4)
    assert kv and 20.4 + kv <= mg.usable_vram_gb(32.0, 1) + 0.01


def test_kv_cache_type_scales_footprint():
    """Quantizing the KV cache shrinks it by the dtype's bit-width vs f16."""
    f16 = mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM)
    q8 = mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM, kv_cache_type="q8_0")
    q5 = mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM, kv_cache_type="q5_1")
    q4 = mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM, kv_cache_type="q4_0")
    assert f16 > q8 > q5 > q4
    assert abs(q8 / f16 - 8.5 / 16) < 0.01     # q8_0 = 8.5 bpw
    assert abs(q5 / f16 - 6.0 / 16) < 0.01     # q5_1 = 6.0 bpw
    assert abs(q4 / f16 - 4.5 / 16) < 0.01     # q4_0 = 4.5 bpw
    # Blank, "f16", and unknown types all fall back to the full f16 size.
    assert mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM, kv_cache_type="f16") == f16
    assert mg.kv_cache_gb("x", 110336, meta=EXPLICIT_HEAD_DIM, kv_cache_type="bogus") == f16


def test_parallel_raises_overhead_and_lowers_safe_ctx(monkeypatch):
    """More request slots reserve more compute buffer, so usable VRAM and the
    safe ctx both drop as --parallel grows."""
    monkeypatch.setattr(mg, "_load_meta", lambda p: EXPLICIT_HEAD_DIM)
    assert mg.usable_vram_gb(32.0, 1) > mg.usable_vram_gb(32.0, 4)
    one = mg.safe_max_ctx("x", 31.7, kv_cache_type="q5_1", n_parallel=1)
    four = mg.safe_max_ctx("x", 31.7, kv_cache_type="q5_1", n_parallel=4)
    assert one and four and one > four
    # The real case: 110k at q5_1 + 4 slots is unsafe on a 32 GB card, and the
    # suggested safe ctx is well below it.
    v = mg.ctx_safety("x", 110336, vram_gb=31.7, ram_gb=64.0,
                      kv_cache_type="q5_1", n_parallel=4)
    assert v.is_unsafe and v.safe_ctx and v.safe_ctx < 110336
    assert "q5_1" in v.message


def test_safe_max_ctx_respects_budget(tmp_path, monkeypatch):
    # Patch the loader so we don't need a real file on disk.
    monkeypatch.setattr(mg, "_load_meta", lambda p: NO_GQA)
    safe = mg.safe_max_ctx("ignored", 32.0)
    assert safe is not None
    # weights+KV at the suggested ctx must fit the 88% budget.
    need = mg.weights_gb("ignored", meta=NO_GQA) + mg.kv_cache_gb("x", safe, meta=NO_GQA)
    assert need <= mg.usable_vram_gb(32.0) + 0.1
    assert safe % 256 == 0


def test_ctx_safety_levels(monkeypatch):
    monkeypatch.setattr(mg, "_load_meta", lambda p: NO_GQA)
    danger = mg.ctx_safety("x", 64384, vram_gb=32.0, ram_gb=61.0)
    assert danger.level == "danger" and danger.safe_ctx and danger.is_unsafe
    ok = mg.ctx_safety("x", danger.safe_ctx, vram_gb=32.0, ram_gb=61.0)
    assert ok.level == "ok" and not ok.is_unsafe

    # The GQA MoE has a tiny KV, so even 262k is never 'danger' (no swap
    # thrash) — unlike the no-GQA model above. That's the model-aware point.
    monkeypatch.setattr(mg, "_load_meta", lambda p: GQA)
    verdict = mg.ctx_safety("x", 262144, vram_gb=32.0, ram_gb=61.0)
    assert verdict.level != "danger"


def test_ctx_safety_without_vram_still_flags_danger(monkeypatch):
    """The RAM-only danger check must work even when VRAM is unknown."""
    monkeypatch.setattr(mg, "_load_meta", lambda p: NO_GQA)
    v = mg.ctx_safety("x", 64384, vram_gb=None, ram_gb=61.0)
    assert v.level == "danger"


def test_mmproj_size_counts_against_the_budget(monkeypatch):
    """A vision projector occupies VRAM, so passing extra_gb must lower the
    safe ctx and show up in the breakdown."""
    monkeypatch.setattr(mg, "_load_meta", lambda p: GQA)  # tiny KV, big weights
    no_proj = mg.safe_max_ctx("x", 32.0)
    with_proj = mg.safe_max_ctx("x", 32.0, extra_gb=1.3)
    assert with_proj is not None and no_proj is not None
    assert with_proj < no_proj                     # projector eats headroom
    v = mg.ctx_safety("x", 8192, vram_gb=32.0, ram_gb=61.0, extra_gb=1.3)
    assert "mmproj 1.3" in v.message               # broken out for the operator


def test_file_gb_missing_is_zero(tmp_path):
    assert mg.file_gb(tmp_path / "nope.gguf") == 0.0
    p = tmp_path / "proj.gguf"
    p.write_bytes(b"x" * (1024 * 1024))            # 1 MiB
    assert 0.0 < mg.file_gb(p) < 0.01


# ---------- cache-ram derivation ----------

def test_cache_ram_scales_down_with_pressure():
    assert mg.derive_cache_ram_mib(40) == 8192      # plentiful -> engine default
    assert mg.derive_cache_ram_mib(12) == 3072      # 25% of 12 GB
    assert mg.derive_cache_ram_mib(0.1) == 512      # floor


# ---------- pressure classification ----------

def _state(avail_frac, swap_frac, ram=61.0, swap=8.0, io=None, io_out=None):
    """``io`` is total traffic (display); ``io_out`` is the eviction rate the
    classifier reads. Passing only ``io`` models traffic of unknown direction
    — which must not escalate."""
    return mg.MemState(ram_total_gb=ram, ram_available_gb=ram * avail_frac,
                       swap_total_gb=swap, swap_used_gb=swap * swap_frac,
                       swap_io_mb_s=io, swap_out_mb_s=io_out)


@pytest.mark.parametrize("avail,swap,expected", [
    (0.50, 0.00, mg.Pressure.OK),
    (0.12, 0.10, mg.Pressure.WARN),     # RAM tight
    (0.05, 0.10, mg.Pressure.CRITICAL), # RAM critical
    # Swap occupancy is gated on RAM: with plenty free it is history, not
    # pressure (see test_full_swap_with_free_ram_is_not_pressure).
    (0.40, 0.30, mg.Pressure.OK),
    (0.40, 0.70, mg.Pressure.OK),
    # Under the gate (< 25% free) occupancy escalates as before.
    (0.20, 0.30, mg.Pressure.WARN),
    (0.20, 0.70, mg.Pressure.CRITICAL),
    (0.008, 1.0, mg.Pressure.CRITICAL), # the incident: 0.5/61 GB free, full swap
])
def test_classify_pressure(avail, swap, expected):
    th = mg.MemThresholds()
    assert mg.classify_pressure(_state(avail, swap), th) is expected


def test_full_swap_with_free_ram_is_not_pressure():
    """The 2026-08-09 regression: swap 7.9/8.0 GB used, 35/60 GB RAM free and
    zero swap traffic. Occupancy alone latched the guard at CRITICAL, which
    restarted the engine before 92% of requests."""
    th = mg.MemThresholds()
    s = mg.MemState(ram_total_gb=60.0, ram_available_gb=35.0,
                    swap_total_gb=8.0, swap_used_gb=7.9, swap_io_mb_s=0.0)
    assert mg.classify_pressure(s, th) is mg.Pressure.OK


def test_swap_out_escalates_regardless_of_headroom():
    """Eviction is live thrash — it counts even with RAM to spare."""
    th = mg.MemThresholds()
    assert mg.classify_pressure(
        _state(0.50, 0.2, io=6.0, io_out=5.0), th) is mg.Pressure.WARN
    assert mg.classify_pressure(
        _state(0.50, 0.2, io=60.0, io_out=50.0), th) is mg.Pressure.CRITICAL
    # Unknown traffic (None) must not be read as zero *or* as pressure.
    assert mg.classify_pressure(_state(0.50, 0.2, io=None), th) is mg.Pressure.OK


def test_swap_in_alone_is_recovery_not_pressure():
    """2026-08-16: after a 14-minute MiniMax-H3 render released its RAM, the
    box paged 19 GB of desktop back in. The guard summed both directions and
    saw 37-55 MB/s with 46/61 GB free, logged CRITICAL four times and armed an
    engine restart each time. Page-ins are the recovery, not the squeeze."""
    th = mg.MemThresholds()
    s = mg.MemState(ram_total_gb=60.9, ram_available_gb=46.2,
                    swap_total_gb=72.0, swap_used_gb=28.2,
                    swap_io_mb_s=55.4, swap_out_mb_s=0.0)
    assert mg.classify_pressure(s, th) is mg.Pressure.OK
    # The same box while it is actually being squeezed still escalates.
    squeezed = dataclasses.replace(s, swap_out_mb_s=55.4)
    assert mg.classify_pressure(squeezed, th) is mg.Pressure.CRITICAL


def test_sampler_splits_the_two_directions(monkeypatch):
    """The rate pair is (total, out): a burst of pure page-ins reports its
    size in the total and zero in the out-rate."""
    sampler = mg.SwapIoSampler()
    pages = [(0, 0), (256 * 1024, 0)]        # 1 GiB in, nothing out
    monkeypatch.setattr(mg, "read_swap_io_pages", lambda: pages.pop(0))
    t = [1000.0, 1001.0]
    monkeypatch.setattr(mg.time, "monotonic", lambda: t.pop(0))
    assert sampler.sample() is None          # first reading has no baseline
    total, out = sampler.sample()
    assert round(total) == 1024 and out == 0.0


def test_no_swap_box_ignores_swap_signal():
    th = mg.MemThresholds()
    s = mg.MemState(ram_total_gb=32, ram_available_gb=20, swap_total_gb=0, swap_used_gb=0)
    assert mg.classify_pressure(s, th) is mg.Pressure.OK


def test_thresholds_from_cfg_override_defaults():
    class Cfg:
        mem_crit_swap_io_mb_s = 3.5
        mem_swap_gate_avail_frac = 0.9

    th = mg.MemThresholds.from_cfg(Cfg())
    assert th.crit_swap_io_mb_s == 3.5
    assert th.swap_gate_avail_frac == 0.9
    assert th.warn_avail_frac == mg.MemThresholds.warn_avail_frac  # untouched
    # With the gate raised, the same full-swap/free-RAM box is CRITICAL again.
    s = mg.MemState(ram_total_gb=60.0, ram_available_gb=35.0,
                    swap_total_gb=8.0, swap_used_gb=7.9)
    assert mg.classify_pressure(s, th) is mg.Pressure.CRITICAL


# ---------- swap I/O sampling ----------

def test_swap_io_sampler_rate(monkeypatch):
    counters = [(1000, 2000)]
    clock = [100.0]
    monkeypatch.setattr(mg, "read_swap_io_pages", lambda: counters[-1])
    monkeypatch.setattr(mg.time, "monotonic", lambda: clock[0])
    s = mg.SwapIoSampler()
    assert s.sample() is None                 # first reading: no baseline yet
    # 512 pages in + 512 out = 1024 pages x 4 KiB = 4 MiB over 2s = 2 MB/s
    # total, of which the out half is 1 MB/s. The pair is (total, out).
    counters.append((1512, 2512))
    clock[0] = 102.0
    total, out = s.sample()
    assert total == pytest.approx(1024 * mg._PAGE_BYTES / 2 / (1024 ** 2), rel=0.01)
    assert out == pytest.approx(512 * mg._PAGE_BYTES / 2 / (1024 ** 2), rel=0.01)
    # A quiet interval reads as zero, not as the previous rate.
    clock[0] = 104.0
    assert s.sample() == (0.0, 0.0)
    # A gap longer than MAX_DT re-baselines rather than averaging over it.
    clock[0] = 104.0 + mg.SwapIoSampler.MAX_DT + 1
    counters.append((99999, 99999))
    assert s.sample() is None


def test_swap_io_sampler_unreadable_is_none(monkeypatch):
    monkeypatch.setattr(mg, "read_swap_io_pages", lambda: None)
    assert mg.SwapIoSampler().sample() is None


# ---------- watchdog ----------

def test_watchdog_edge_triggered(monkeypatch):
    seq = [_state(0.50, 0.0), _state(0.10, 0.1),
           _state(0.03, 0.8), _state(0.50, 0.0)]
    i = {"n": 0}

    def fake_read():
        s = seq[min(i["n"], len(seq) - 1)]
        i["n"] += 1
        return s

    monkeypatch.setattr(mg, "read_mem_state", fake_read)
    calls = []

    async def cb(level, state):
        calls.append(level)

    class Cfg:
        mem_guard_enabled = True
        mem_guard_interval_s = 0.02

    async def run():
        wd = mg.MemoryWatchdog(Cfg(), on_pressure=cb, interval_s=0.02)
        wd.start()
        await asyncio.sleep(0.2)
        await wd.stop()

    asyncio.run(run())
    # Rising edges then the return to OK — not one per tick.
    assert calls == [mg.Pressure.WARN, mg.Pressure.CRITICAL, mg.Pressure.OK]


def test_watchdog_disabled_does_not_start(monkeypatch):
    class Cfg:
        mem_guard_enabled = False

    wd = mg.MemoryWatchdog(Cfg(), on_pressure=lambda *a: None)
    wd.start()
    assert wd._task is None


def _run_watchdog(cfg, states, seconds=0.3):
    """Drive the watchdog over a state sequence; return (watchdog, levels)."""
    seq = list(states)
    i = {"n": 0}

    def fake_read():
        s = seq[min(i["n"], len(seq) - 1)]
        i["n"] += 1
        return s

    calls = []

    async def cb(level, state):
        calls.append(level)

    holder = {}

    async def run():
        wd = mg.MemoryWatchdog(cfg, on_pressure=cb,
                               interval_s=cfg.mem_guard_interval_s)
        holder["wd"] = wd
        wd.start()
        await asyncio.sleep(seconds)
        await wd.stop()

    import unittest.mock as _m
    with _m.patch.object(mg, "read_mem_state", fake_read):
        asyncio.run(run())
    return holder["wd"], calls


def test_watchdog_repeats_while_critical():
    """Sustained CRITICAL re-fires the callback on the repeat cadence so the
    caller can keep/escalate reclaim — not just once on the rising edge."""
    class Cfg:
        mem_guard_enabled = True
        mem_guard_interval_s = 0.02
        mem_crit_repeat_s = 0.05
        mem_crit_repeat_max_s = 0.05   # cap == base, i.e. back-off disabled

    _wd, calls = _run_watchdog(Cfg(), [_state(0.03, 0.8)])
    # One edge-triggered CRITICAL plus several repeats over ~0.3s at 0.05s.
    assert calls and all(c == mg.Pressure.CRITICAL for c in calls)
    assert len(calls) >= 3, calls


def test_watchdog_backs_off_when_reclaim_achieves_nothing():
    """Reclaim isn't free — it restarts the engine between tasks. When memory
    doesn't improve, the callback cadence must widen instead of hammering."""
    class Cfg:
        mem_guard_enabled = True
        mem_guard_interval_s = 0.02
        mem_crit_repeat_s = 0.05
        mem_crit_repeat_max_s = 10.0

    wd, calls = _run_watchdog(Cfg(), [_state(0.03, 0.8)])
    assert calls and all(c == mg.Pressure.CRITICAL for c in calls)
    # Doubling from 0.05s means at most a handful of emits in 0.3s.
    assert len(calls) <= 4, calls
    assert wd._crit_repeat_cur > Cfg.mem_crit_repeat_s


def test_backoff_decision():
    """The back-off decision itself, without timing: no relief widens the
    cadence (capped), any real relief snaps it back to the base."""
    class Cfg:
        mem_guard_enabled = True
        mem_crit_repeat_s = 10.0
        mem_crit_repeat_max_s = 40.0
        mem_relief_gb = 1.0

    wd = mg.MemoryWatchdog(Cfg(), on_pressure=lambda *a: None)
    wd._last_emit_avail = 2.0

    wd._apply_backoff(mg.MemState(61.0, 2.5, 8.0, 8.0))   # +0.5 GB — no relief
    assert wd._crit_repeat_cur == 20.0
    wd._apply_backoff(mg.MemState(61.0, 2.5, 8.0, 8.0))
    assert wd._crit_repeat_cur == 40.0
    wd._apply_backoff(mg.MemState(61.0, 2.5, 8.0, 8.0))
    assert wd._crit_repeat_cur == 40.0                    # capped

    wd._last_emit_avail = 2.0
    wd._apply_backoff(mg.MemState(61.0, 6.0, 8.0, 8.0))   # +4 GB — it worked
    assert wd._crit_repeat_cur == 10.0

    # No baseline yet (first repeat after a rising edge) → no back-off.
    wd._last_emit_avail = None
    wd._crit_repeat_cur = 10.0
    wd._apply_backoff(mg.MemState(61.0, 2.0, 8.0, 8.0))
    assert wd._crit_repeat_cur == 10.0


def test_coexist_feasibility_gate(tmp_path, monkeypatch):
    """ASR refuses to coexist with the LLM when host memory is already tight."""
    from llamanager.audio_runner import AudioTaskRunner
    from llamanager.config import Config
    from llamanager.db import DB
    r = AudioTaskRunner(Config(data_dir=tmp_path, asr_coexist=True),
                        DB(tmp_path / "s.db"))
    monkeypatch.setattr(mg, "read_mem_state", lambda: _state(0.50, 0.0))  # OK
    assert r._coexist_feasible() is True
    monkeypatch.setattr(mg, "read_mem_state", lambda: _state(0.03, 0.8))  # CRITICAL
    assert r._coexist_feasible() is False


def test_supervisor_oom_wait_returns_when_memory_ok(tmp_path, monkeypatch):
    """The OOM restart-guard returns immediately when memory is healthy (so a
    normal crash still restarts promptly), and is only invoked for rc == -9."""
    import asyncio
    from llamanager.supervisor import Supervisor
    from llamanager.config import Config

    class _StubSM:
        def add_exit_listener(self, q): pass
        def mark_degraded(self, reason): pass

    monkeypatch.setattr(mg, "read_mem_state", lambda: _state(0.50, 0.0))  # OK
    sup = Supervisor(Config(data_dir=tmp_path), _StubSM())
    # Should return promptly (no wait) since memory is fine.
    asyncio.run(asyncio.wait_for(sup._await_mem_recovery(0), timeout=1.0))
