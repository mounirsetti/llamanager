"""Tests for per-server GPU pinning.

Covers the three moving parts of the "GPU for text inference" setting:
the ``--list-devices`` parser, name→env resolution, the config round-trip,
and the env injection in the launch path. None of it touches a real GPU —
subprocess enumeration is faked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


# Real captured output from `llama-server --list-devices` on a box with an
# AMD dGPU next to an Intel iGPU (the case this feature exists for).
SAMPLE_LIST = """\
WARNING: radv is not a conformant Vulkan implementation, testing use only.
Available devices:
  Vulkan0: Intel(R) Graphics (ARL) (46789 MiB, 42110 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 11445 MiB free)
"""


# ---------- parser ----------

def test_parse_device_list_basic():
    from llamanager.gpu_detect import parse_device_list
    devs = parse_device_list(SAMPLE_LIST)
    assert len(devs) == 2
    intel, amd = devs
    assert intel.backend == "Vulkan" and intel.index == 0
    assert intel.name == "Intel(R) Graphics (ARL)"
    assert intel.mem_total_mib == 46789 and intel.mem_free_mib == 42110
    # The "(RADV GFX1201)" parens are part of the name, not the mem group.
    assert amd.index == 1
    assert amd.name == "AMD Radeon AI PRO R9700 (RADV GFX1201)"
    assert amd.mem_free_mib == 11445


def test_parse_device_list_no_mem_and_noise():
    from llamanager.gpu_detect import parse_device_list
    devs = parse_device_list("ggml noise\n  CUDA0: NVIDIA Thing\nmore noise\n")
    assert len(devs) == 1
    assert devs[0].backend == "CUDA" and devs[0].name == "NVIDIA Thing"
    assert devs[0].mem_free_mib is None


def test_parse_device_list_empty():
    from llamanager.gpu_detect import parse_device_list
    assert parse_device_list("no devices here") == []


# ---------- enumeration strips pre-existing filters ----------

def test_list_llama_devices_strips_visible_filter(monkeypatch):
    from llamanager import gpu_detect

    seen_env = {}

    def fake_run(cmd, **kw):
        seen_env.update(kw.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, stdout=SAMPLE_LIST, stderr="")

    monkeypatch.setattr(gpu_detect.subprocess, "run", fake_run)
    env = {"GGML_VK_VISIBLE_DEVICES": "0", "PATH": "/usr/bin"}
    devs = gpu_detect.list_llama_devices("llama-server", env=env)
    assert len(devs) == 2  # full set, not filtered to the one in env
    # The pre-existing pin must not leak into enumeration.
    assert "GGML_VK_VISIBLE_DEVICES" not in seen_env
    assert seen_env.get("PATH") == "/usr/bin"


def test_list_llama_devices_tolerates_failure(monkeypatch):
    from llamanager import gpu_detect

    def boom(cmd, **kw):
        raise OSError("no such binary")

    monkeypatch.setattr(gpu_detect.subprocess, "run", boom)
    assert gpu_detect.list_llama_devices("nope") == []


# ---------- name → env resolution ----------

def _patch_devices(monkeypatch):
    from llamanager import gpu_detect
    monkeypatch.setattr(
        gpu_detect, "list_llama_devices",
        lambda binary, env=None, timeout=15.0: gpu_detect.parse_device_list(SAMPLE_LIST),
    )


def test_visible_devices_env_resolves_name_to_index(monkeypatch):
    from llamanager import gpu_detect
    _patch_devices(monkeypatch)
    # AMD is Vulkan1 in this enumeration → that's the index we must pin.
    env = gpu_detect.visible_devices_env(
        "llama-server", "AMD Radeon AI PRO R9700 (RADV GFX1201)")
    assert env == {"GGML_VK_VISIBLE_DEVICES": "1"}
    # Intel is Vulkan0.
    env2 = gpu_detect.visible_devices_env("llama-server", "Intel(R) Graphics (ARL)")
    assert env2 == {"GGML_VK_VISIBLE_DEVICES": "0"}


def test_visible_devices_env_empty_name_is_none(monkeypatch):
    from llamanager import gpu_detect
    _patch_devices(monkeypatch)
    assert gpu_detect.visible_devices_env("llama-server", "") is None
    assert gpu_detect.visible_devices_env("llama-server", "   ") is None


def test_visible_devices_env_missing_device_is_none(monkeypatch):
    from llamanager import gpu_detect
    _patch_devices(monkeypatch)
    # A name that isn't present → run on all devices rather than refuse.
    assert gpu_detect.visible_devices_env("llama-server", "Some Other GPU") is None


# ---------- robust (backend-agnostic) name matching ----------

def test_clean_gpu_name_strips_driver_and_memory_annotations():
    from llamanager.gpu_detect import clean_gpu_name
    assert clean_gpu_name("AMD Radeon AI PRO R9700 (RADV GFX1201)") == \
        "AMD Radeon AI PRO R9700"
    assert clean_gpu_name("Intel(R) Graphics (ARL)") == "Intel Graphics"


def test_match_device_across_backend_naming():
    """The real bug: a card pinned under its ROCm name (no codename) must
    still match the Vulkan/RADV enumeration that appends '(RADV GFXxxxx)'."""
    from llamanager import gpu_detect
    devs = gpu_detect.parse_device_list(SAMPLE_LIST)  # Vulkan names w/ codename
    m = gpu_detect.match_device(devs, "AMD Radeon AI PRO R9700")  # ROCm-style
    assert m is not None and m.backend == "Vulkan" and m.index == 1
    # Case / whitespace insensitive too.
    assert gpu_detect.match_device(devs, "amd  radeon ai pro r9700").index == 1
    # A genuinely different card does not match.
    assert gpu_detect.match_device(devs, "NVIDIA GeForce RTX 4090") is None


def test_visible_devices_env_matches_suffixless_name(monkeypatch):
    """End-to-end: the exact pin string that broke in the field now resolves."""
    from llamanager import gpu_detect
    _patch_devices(monkeypatch)
    env = gpu_detect.visible_devices_env("llama-server", "AMD Radeon AI PRO R9700")
    assert env == {"GGML_VK_VISIBLE_DEVICES": "1"}


def test_match_device_vram_tiebreaker_for_identical_names():
    """Two cards whose names normalise the same are disambiguated by VRAM."""
    from llamanager import gpu_detect
    from llamanager.gpu_detect import LlamaDevice
    devs = [
        LlamaDevice("CUDA", 0, "NVIDIA RTX 6000", 24564, 24000),
        LlamaDevice("CUDA", 1, "NVIDIA RTX 6000", 49140, 49000),
    ]
    m = gpu_detect.match_device(devs, "NVIDIA RTX 6000", stored_total_mib=49140)
    assert m is not None and m.index == 1


def test_visible_devices_env_backend_mapping(monkeypatch):
    from llamanager import gpu_detect
    monkeypatch.setattr(
        gpu_detect, "list_llama_devices",
        lambda binary, env=None, timeout=15.0:
            gpu_detect.parse_device_list("  ROCm0: AMD Radeon (gfx1201)\n"),
    )
    env = gpu_detect.visible_devices_env("llama-server", "AMD Radeon (gfx1201)")
    assert env == {"ROCR_VISIBLE_DEVICES": "0"}


# ---------- config round-trip ----------

def _write_min_config(tmp_path: Path) -> Path:
    from llamanager.config import DEFAULT_CONFIG_TOML
    import tomlkit
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))
    return cfg_path


def test_gpu_device_config_roundtrip(tmp_path: Path):
    from llamanager.config import load_config, update_server_gpu
    cfg_path = _write_min_config(tmp_path)

    # Default: empty.
    assert load_config(cfg_path).llama_gpu_device == ""

    update_server_gpu(cfg_path, device_name="AMD Radeon AI PRO R9700 (RADV GFX1201)")
    reloaded = load_config(cfg_path)
    assert reloaded.llama_gpu_device == "AMD Radeon AI PRO R9700 (RADV GFX1201)"

    # Clearing removes the pin.
    update_server_gpu(cfg_path, device_name="")
    assert load_config(cfg_path).llama_gpu_device == ""


# ---------- launch-path injection ----------

def test_engine_env_injects_pin(monkeypatch):
    from llamanager import server_manager, gpu_detect
    _patch_devices(monkeypatch)
    # Vulkan build (no rocm lib path) + a pinned AMD card.
    env = server_manager._engine_env(
        "/x/llama.cpp-vulkan/llama-server",
        gpu_device="AMD Radeon AI PRO R9700 (RADV GFX1201)",
    )
    assert env is not None
    assert env["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_engine_env_no_pin_is_inherit(monkeypatch):
    from llamanager import server_manager
    # No device + vulkan build → inherit (None), unchanged behaviour.
    assert server_manager._engine_env("/x/llama.cpp-vulkan/llama-server") is None


# ---------- recurrent-model CUDA-graph workaround ----------

def test_is_recurrent_detects_ssm_and_hybrid():
    from llamanager.gguf_meta import GgufMeta, is_recurrent
    plain = GgufMeta(architecture="gemma4", block_count=48)
    assert is_recurrent(plain) is False
    # Hybrid SSM/attention (Qwen3.6 "qwen35"): full_attention_interval + ssm.
    hybrid = GgufMeta(architecture="qwen35", block_count=64,
                      full_attention_interval=4, ssm_state_size=128,
                      ssm_inner_size=4096, ssm_conv_kernel=4)
    assert is_recurrent(hybrid) is True
    # Pure recurrent (Mamba-style): ssm geometry, no full_attention_interval.
    mamba = GgufMeta(architecture="mamba2", block_count=64, ssm_state_size=128)
    assert is_recurrent(mamba) is True


def test_model_needs_graph_disable_only_rocm_recurrent(monkeypatch, tmp_path):
    from llamanager import server_manager
    from llamanager.gguf_meta import GgufMeta
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0")  # is_file() True; read is monkeypatched below
    hybrid = GgufMeta(architecture="qwen35", block_count=64,
                      full_attention_interval=4, ssm_state_size=128)
    plain = GgufMeta(architecture="gemma4", block_count=48)
    HIP = "/x/llama.cpp-hip/llama-server"
    VK = "/x/llama.cpp-vulkan/llama-server"

    monkeypatch.setattr(server_manager, "read_gguf_meta", lambda p: hybrid)
    assert server_manager.model_needs_graph_disable(model, HIP) is True
    # Same recurrent model on a non-ROCm build → no workaround.
    assert server_manager.model_needs_graph_disable(model, VK) is False

    monkeypatch.setattr(server_manager, "read_gguf_meta", lambda p: plain)
    # Plain transformer on ROCm → keep graphs (and their decode speedup).
    assert server_manager.model_needs_graph_disable(model, HIP) is False


def test_model_needs_graph_disable_read_error_is_safe(monkeypatch, tmp_path):
    from llamanager import server_manager
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0")

    def boom(_):
        raise ValueError("malformed header")
    monkeypatch.setattr(server_manager, "read_gguf_meta", boom)
    # Best-effort: any read failure leaves graphs on rather than crashing launch.
    assert server_manager.model_needs_graph_disable(
        model, "/x/llama.cpp-hip/llama-server") is False


def test_engine_env_disable_graphs_and_operator_override(monkeypatch):
    from llamanager import server_manager
    HIP = "/x/llama.cpp-hip/llama-server"
    # No rocm lib dir resolved in this test env, but disable_cuda_graphs still
    # applies the var onto a copied environment.
    monkeypatch.delenv("GGML_CUDA_DISABLE_GRAPHS", raising=False)
    env = server_manager._engine_env(HIP, disable_cuda_graphs=True)
    assert env is not None and env["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    # An explicit operator value wins (setdefault) — e.g. forcing graphs on.
    monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", "0")
    env = server_manager._engine_env(HIP, disable_cuda_graphs=True)
    assert env["GGML_CUDA_DISABLE_GRAPHS"] == "0"
    # Not requested → untouched (plain inherit for a vulkan build).
    monkeypatch.delenv("GGML_CUDA_DISABLE_GRAPHS", raising=False)
    assert server_manager._engine_env(
        "/x/llama.cpp-vulkan/llama-server", disable_cuda_graphs=False) is None
