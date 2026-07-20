"""Tests for the GPU-memory leak guards around the engine lifecycle.

Background: a wedged llama-server was SIGKILLed and the amdgpu/KFD driver never
reclaimed its context. 26 GB stayed pinned against a dead PID — first as device
VRAM, then migrating into host RAM as GTT — so every later launch saw ~7.5 GB
free of 32 GB, silently built a mostly-CPU config via ``--fit``, and ran at
~0.5 tok/s until every request hit the queue timeout. Nothing noticed for 12
hours. These tests pin the three behaviours that make that loud instead:

1. don't SIGKILL a process the driver still holds a GPU context for (that is
   what creates the leak) without a long grace first;
2. after exit, verify the memory actually came back — counting GTT, since a
   leak migrating VRAM->host RAM reads as "released" if you watch VRAM alone;
3. before launching, compare *live* free VRAM against what the run needs, and
   refuse outright when the profile demands a full-GPU run.

No real GPU is touched — the sysfs probes are monkeypatched.
"""
from __future__ import annotations

import asyncio

import pytest

from llamanager import mem_guard as mg
from llamanager import server_manager as sm


def _gpu(vram_used: float, gtt_used: float = 0.0, total: float = 32.0):
    return mg.GpuMem(card="card1", vram_used=vram_used, vram_total=total,
                     gtt_used=gtt_used, gtt_total=30.0)


# ---------- _gpu_freed: the VRAM->GTT migration blind spot ----------

def test_freed_when_memory_returns():
    before, after = [_gpu(26.0)], [_gpu(0.3)]
    assert sm.ServerManager._gpu_freed(before, after) is True


def test_not_freed_when_vram_migrates_into_host_gtt():
    """The leak's signature: VRAM drains but the same memory reappears pinned
    as GTT. Watching VRAM alone would call this released."""
    before, after = [_gpu(26.0, gtt_used=0.5)], [_gpu(1.0, gtt_used=24.0)]
    assert sm.ServerManager._gpu_freed(before, after) is False


def test_not_freed_when_memory_simply_stays():
    before, after = [_gpu(26.0)], [_gpu(26.0)]
    assert sm.ServerManager._gpu_freed(before, after) is False


# ---------- stale KFD contexts ----------

def test_stale_kfd_pids_flags_dead_owner(monkeypatch):
    """A KFD context whose PID no longer exists is a definitive leak."""
    monkeypatch.setattr(mg, "kfd_proc_pids", lambda: {999999, 1})
    stale = mg.stale_kfd_pids()
    assert 999999 in stale      # certainly dead
    assert 1 not in stale       # init is alive


# ---------- preflight ----------

class _Spec:
    def __init__(self, ctx=4096, model_gb=20.4):
        self.model_id = "unsloth/gemma-4-31B-it"
        self.model_path = "/nonexistent/model.gguf"
        self.mmproj_path = None
        self.extra_args = {"ctx-size": ctx}
        self._model_gb = model_gb


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(mg, "file_gb", lambda p: 20.4)
    monkeypatch.setattr(mg, "kv_cache_gb",
                        lambda *a, **k: 1.1)   # real gemma-4 KV at ctx 4096
    monkeypatch.setattr(mg, "stale_kfd_pids", lambda: set())
    return monkeypatch


def test_preflight_passes_on_healthy_card(patched):
    """21.5 GB needed, 30.5 GB free — must not raise or complain."""
    patched.setattr(mg, "read_gpu_mem", lambda: [_gpu(1.4)])
    sm._preflight_gpu_memory(_Spec(), None, "llama", gpu_only=True)


def test_preflight_refuses_gpu_only_when_card_is_occupied(patched):
    """The exact 2026-07-16 condition: 7.5 GB free of 32 GB. A GPU-only
    profile must fail fast rather than start and silently run on CPU."""
    patched.setattr(mg, "read_gpu_mem", lambda: [_gpu(24.5)])
    with pytest.raises(sm.ServerError) as e:
        sm._preflight_gpu_memory(_Spec(), None, "llama", gpu_only=True)
    msg = str(e.value)
    assert "refusing to start" in msg
    assert "7.5 GB" in msg or "7.4 GB" in msg     # reports what's actually free


def test_preflight_only_warns_when_spill_is_allowed(patched, caplog):
    """Same shortage, but a profile that permits RAM spill should proceed —
    loudly. Slow is a valid choice; silent is not."""
    patched.setattr(mg, "read_gpu_mem", lambda: [_gpu(24.5)])
    with caplog.at_level("WARNING"):
        sm._preflight_gpu_memory(_Spec(), None, "llama", gpu_only=False)
    assert any("spill to system RAM" in r.getMessage() for r in caplog.records)


def test_preflight_reports_leak_but_still_starts_when_vram_is_free(patched, caplog):
    """A leak that has migrated entirely into host RAM (GTT) leaves VRAM
    genuinely free, so the model can still load onto the GPU. Report it
    loudly — the remedy is a reset/reboot, not stopping a model — but do NOT
    block: refusing to run while VRAM is available would make the box useless
    for the sake of a condition we deliberately don't auto-clear."""
    patched.setattr(mg, "read_gpu_mem", lambda: [_gpu(2.0, gtt_used=24.0)])
    patched.setattr(mg, "stale_kfd_pids", lambda: {807245})
    with caplog.at_level("ERROR"):
        sm._preflight_gpu_memory(_Spec(), None, "llama", gpu_only=True)
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "807245" in logged
    assert "reset or reboot" in logged


def test_preflight_names_leak_as_the_cause_when_it_blocks(patched):
    """When the leak is still holding VRAM, the refusal must name it — that
    changes the operator's remedy from 'stop a model' to 'needs a reset'."""
    patched.setattr(mg, "read_gpu_mem", lambda: [_gpu(24.5, gtt_used=4.0)])
    patched.setattr(mg, "stale_kfd_pids", lambda: {807245})
    with pytest.raises(sm.ServerError) as e:
        sm._preflight_gpu_memory(_Spec(), None, "llama", gpu_only=True)
    assert "807245" in str(e.value)


def test_preflight_ignores_non_llama_engines(patched):
    patched.setattr(mg, "read_gpu_mem", lambda: [_gpu(31.0)])
    sm._preflight_gpu_memory(_Spec(), None, "mlx", gpu_only=True)


def test_preflight_noop_without_readable_gpu(patched):
    """No AMD sysfs (other vendor / container) must not block launches."""
    patched.setattr(mg, "read_gpu_mem", lambda: [])
    sm._preflight_gpu_memory(_Spec(), None, "llama", gpu_only=True)
