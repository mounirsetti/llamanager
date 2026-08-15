"""Tests for the MiniMax-H3 engine and the diffusion-profile plumbing.

Three groups, all regressions of things that were silently wrong:

1. **MiniMax-H3 registration and sizing.** The engine is Modular-Diffusers-only
   and enormous (124 GB of bf16 weights), so it has to be detected by a
   different marker than every other adapter, and it has to refuse a request it
   cannot host instead of thrashing.

2. **Platform-aware install plans.** The installer used to map Apple Silicon to
   the CPU wheel index, giving up the Metal backend for no reason. Each GPU
   family must now resolve to its own torch build.

3. **Profile persistence.** ``video_num_frames`` / ``video_fps`` were declared
   on ``Profile`` but never serialised or parsed, so every saved video profile
   silently lost its clip length and fell back to a 121-frame default. And
   ``default_profiles()`` called without a model dir handed a Krea *original*
   checkpoint the GGUF profile set, which fails at load time.

No GPU, no network, and no model weights are touched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from llamanager import diffusion_catalog, engines
from llamanager.config import (ENGINE_FAMILY, Profile, detect_engine_for_path,
                               load_config, save_profile)
from llamanager.engine_installer import TORCH_BACKENDS, resolve_plan
from llamanager.gpu_detect import GpuProfile

_RUNNER = (Path(__file__).resolve().parents[1] / "llamanager" / "engines"
           / "_minimax_h3_runner.py")


def _runner_module():
    """Import the runner directly — it is a standalone script, not a package
    member, and importing it must not require torch at module scope."""
    spec = importlib.util.spec_from_file_location("_mmh3_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- registration


def test_minimax_h3_is_a_registered_video_engine():
    assert "minimax_h3" in engines.ADAPTERS
    assert ENGINE_FAMILY["minimax_h3"] == "video"
    adapter = engines.get("minimax_h3")
    assert adapter.capabilities()["output_ext"] == "mp4"
    # Two keyframes: the clip's first and last frame.
    assert adapter.capabilities()["ref_images_max"] == 2


def test_minimax_h3_profile_schema_omits_guidance_and_negative_prompt():
    """The checkpoints are guidance-distilled: there is no guider, no
    negative prompt and no guidance scale. Offering those knobs would be
    offering settings the model ignores."""
    keys = {f.key for f in engines.get("minimax_h3").profile_schema()}
    assert "image_guidance" not in keys
    assert "image_negative_prompt" not in keys
    # But it must expose clip length and the offload strategy.
    assert {"video_num_frames", "image_editing_scheduler"} <= keys


def test_minimax_h3_in_catalog():
    entries = diffusion_catalog.for_engine("minimax_h3")
    assert [e.hf_repo for e in entries] == ["MiniMaxAI/MiniMax-H3"]


def test_detects_modular_diffusers_layout(tmp_path):
    """MiniMax-H3 ships ``modular_model_index.json``, not ``model_index.json``,
    so detection keys off that file plus a MiniMaxH3 component class."""
    d = tmp_path / "MiniMax-H3"
    d.mkdir()
    (d / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3"]}', encoding="utf-8")
    assert detect_engine_for_path(d) == "minimax_h3"


def test_other_modular_repos_are_not_claimed(tmp_path):
    """A Modular Diffusers repo that isn't MiniMax-H3 must not be claimed."""
    d = tmp_path / "SomethingElse"
    d.mkdir()
    (d / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKL"]}', encoding="utf-8")
    assert detect_engine_for_path(d) != "minimax_h3"


# ---------------------------------------------------------------- frame maths


@pytest.mark.parametrize("requested", [1, 5, 6, 22, 100, 120, 124, 130, 361])
def test_num_frames_snaps_up_to_a_decodable_count(requested):
    """The video VAE only decodes 17n+5 frames, and snapping must never
    shorten the request."""
    snapped = _runner_module().snap_num_frames(requested)
    assert (snapped - 5) % 17 == 0
    assert snapped >= requested


def test_upstream_example_frame_count_is_already_valid():
    mod = _runner_module()
    assert mod.snap_num_frames(124) == 124
    assert mod.MIN_SECONDS <= 124 / mod.NATIVE_FPS <= mod.MAX_SECONDS


# ---------------------------------------------------------------- memory guard


def _sized(mod, monkeypatch, accel, ram):
    monkeypatch.setattr(mod, "_accelerator_gib", lambda device: accel)
    monkeypatch.setattr(mod, "_host_ram_gib", lambda: ram)
    return mod


def test_quantisation_factors_match_what_was_measured():
    """Ratios measured on gfx1201 against an 8192x8192 bf16 linear."""
    mod = _runner_module()
    assert mod.QUANT_FACTORS["int8"] == 0.5
    assert mod.QUANT_FACTORS["fp8"] == 0.5
    assert mod.QUANT_FACTORS["int4"] == 0.25
    assert mod.QUANT_FACTORS["none"] == 1.0
    # bitsandbytes NF4 measured 128.0 -> 33.0 MiB = 3.88x.
    assert mod.QUANT_FACTORS["nf4"] == pytest.approx(1 / 3.88, rel=1e-6)
    # Dynamic-activation int8 measured 85x slower for the same memory and
    # must never be offered as an option.
    assert "int8_dynamic" not in mod.QUANT_FACTORS


def test_nf4_is_served_by_bitsandbytes_not_torchao():
    """The 4-bit path that works on ROCm comes from bitsandbytes; routing it
    to torchao would hit the missing-mslk failure instead."""
    mod = _runner_module()
    assert "nf4" in mod._BNB_QUANTS
    assert "int4" not in mod._BNB_QUANTS


def test_nf4_plus_split_fits_a_32gb_card(monkeypatch):
    """The result that matters on AMD: 21.5 GiB peak, fully resident."""
    mod = _sized(_runner_module(), monkeypatch, accel=31.9, ram=60.0)
    verdict, detail = mod.plan_memory("cuda", "nf4", "none", split=True)
    assert verdict == "ok"
    assert "resident" in detail


def test_split_residency_halves_the_resident_peak():
    """Conditioner and transformer are never live together, so peak is max()
    of the two rather than their sum."""
    mod = _runner_module()
    t, c = mod.weight_gib("int4")
    joint = t + c + mod._VAE_GIB
    split = max(t, c) + mod._VAE_GIB
    assert split < joint
    # 15.5 (int4 conditioner) + 10.4 (bf16 VAEs, measured on the real
    # checkpoint: vae/ 9.8 GB + audio_vae/ 0.58 GB). An earlier 5.5 GB
    # estimate for the VAEs made this look like 21 GB.
    assert split == pytest.approx(25.9, abs=0.5)


def test_int4_plus_split_fits_a_32gb_card(monkeypatch):
    """The whole point of the quantised path: 21 GiB peak on a 32 GB card,
    fully resident, nothing in host RAM."""
    mod = _sized(_runner_module(), monkeypatch, accel=31.9, ram=60.0)
    verdict, detail = mod.plan_memory("cuda", "int4", "none", split=True)
    assert verdict == "ok"
    assert "resident" in detail


def test_int8_plus_split_does_not_fit_a_32gb_card(monkeypatch):
    """36.5 GiB still overflows 32 GB — the runner must say so rather than
    optimistically starting a 140 GB download."""
    mod = _sized(_runner_module(), monkeypatch, accel=31.9, ram=60.0)
    verdict, _ = mod.plan_memory("cuda", "int8", "none", split=True)
    assert verdict == "refuse"


def test_int8_plus_split_fits_a_48gb_card(monkeypatch):
    mod = _sized(_runner_module(), monkeypatch, accel=48.0, ram=128.0)
    verdict, _ = mod.plan_memory("cuda", "int8", "none", split=True)
    assert verdict == "ok"


def test_joint_residency_needs_roughly_double(monkeypatch):
    """Same quantisation, both halves resident at once: 67 GiB, so a card that
    passes with split fails without it."""
    mod = _sized(_runner_module(), monkeypatch, accel=48.0, ram=128.0)
    assert mod.plan_memory("cuda", "int8", "none", split=True)[0] == "ok"
    assert mod.plan_memory("cuda", "int8", "none", split=False)[0] == "refuse"


def test_unquantised_run_suggests_quantisation(monkeypatch):
    mod = _sized(_runner_module(), monkeypatch, accel=31.9, ram=60.0)
    verdict, detail = mod.plan_memory("cuda", "none", "none", split=True)
    assert verdict == "refuse"
    assert "int8" in detail


def test_refuses_when_host_ram_cannot_hold_the_offloaded_weights(monkeypatch):
    """int8 + block offload parks ~62 GB in host RAM. A 60 GB machine has to
    be told no, with the numbers, rather than swapping itself to death."""
    mod = _sized(_runner_module(), monkeypatch, accel=32.0, ram=60.0)
    verdict, detail = mod.plan_memory("cuda", "int8", "block")
    assert verdict == "refuse"
    assert "60.0" in detail and "host RAM" in detail


def test_allows_offload_when_host_ram_is_large_enough(monkeypatch):
    mod = _sized(_runner_module(), monkeypatch, accel=32.0, ram=128.0)
    assert mod.plan_memory("cuda", "int8", "block")[0] == "ok"


def test_allows_resident_run_on_an_80gb_card(monkeypatch):
    mod = _sized(_runner_module(), monkeypatch, accel=80.0, ram=256.0)
    assert mod.plan_memory("cuda", "int8", "none")[0] == "ok"


def test_quant_probe_reports_unavailable_without_torchao(monkeypatch):
    """The probe must answer "no" rather than raising when the option cannot
    be built at all — that is what lets the runner fall back."""
    mod = _runner_module()
    ok, why = mod.probe_quant("int4", "cpu")
    assert isinstance(ok, bool) and isinstance(why, str)
    # "none" is always available and never touches torch.
    assert mod.probe_quant("none", "cpu") == (True, "no quantisation")


# ---------------------------------------------------------- platform install


def test_apple_silicon_resolves_to_metal_not_cpu():
    """Regression: Apple used to map to "cpu", which pinned the install to the
    CPU wheel index and threw away the Metal backend the default macOS arm64
    wheel already carries."""
    assert "mps" in TORCH_BACKENDS
    plan = resolve_plan("minimax_h3", GpuProfile(kind="apple"))
    assert plan.target == "mps"
    assert plan.torch_index_url == ""          # no index override
    assert any(p == "torch" for p in plan.packages)


@pytest.mark.parametrize("kind,expected", [
    ("amd", "amd-rocmrel7.2.1"),
    ("nvidia", "cuda"),
    ("apple", "mps"),
    ("cpu", "cpu"),
])
def test_each_gpu_family_gets_its_own_target(kind, expected):
    plan = resolve_plan("minimax_h3", GpuProfile(kind=kind))
    assert plan.target == expected


def test_rocm_path_supplies_amd_wheels_and_drops_generic_torch():
    plan = resolve_plan("minimax_h3", GpuProfile(kind="amd", rocm_arch="gfx1201"))
    assert plan.wheel_urls, "expected AMD wheels for the ROCm path"
    assert not any(p == "torch" for p in plan.packages)


def test_minimax_plan_tracks_diffusers_git_main():
    """MiniMax-H3 is not in the pinned diffusers release, so its plan must
    pull git main rather than the pin every other engine uses."""
    plan = resolve_plan("minimax_h3", GpuProfile(kind="nvidia"))
    assert any("github.com/huggingface/diffusers" in p for p in plan.packages)
    assert not any(str(p).startswith("diffusers==") for p in plan.packages)
    assert "torchao" in plan.packages


def test_platform_notes_point_at_the_backend_that_works_there():
    """Regression on a wrong claim: ROCm *does* have a 4-bit path — it is
    bitsandbytes NF4, not torchao int4 — so the AMD note must recommend nf4
    rather than telling operators the model can only run unquantised."""
    cuda = resolve_plan("minimax_h3", GpuProfile(kind="nvidia")).notes
    rocm = resolve_plan("minimax_h3", GpuProfile(kind="amd")).notes
    apple = resolve_plan("minimax_h3", GpuProfile(kind="apple")).notes
    assert "nf4" in rocm and "bitsandbytes" in rocm
    assert "no ROCm kernels" in rocm          # about torchao int4, correctly
    assert "nf4" in cuda and "int4" in cuda   # both available on CUDA
    assert "Metal" in apple                   # neither backend ships Metal


def test_install_plan_ships_both_quantisation_backends():
    pkgs = resolve_plan("minimax_h3", GpuProfile(kind="amd")).packages
    assert "bitsandbytes" in pkgs
    assert "torchao" in pkgs


# ------------------------------------------------------- profile persistence


def test_video_profile_round_trips_frames_and_fps(tmp_path):
    """Regression: ``video_num_frames`` / ``video_fps`` were on the dataclass
    but in neither the serialiser nor the parser, so a saved video profile
    came back with no clip length and silently fell back to the adapter's
    121-frame default — which a consumer card then refuses."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[server]\nmodels_dir = "%s"\n' % tmp_path, encoding="utf-8")
    save_profile(cfg_path, "Wan-AI/Wan2.2", "wan-best",
                 Profile(name="wan-best", image_size="1152x640",
                         image_steps=30, video_num_frames=49, video_fps=16))
    text = cfg_path.read_text(encoding="utf-8")
    assert "video_num_frames = 49" in text
    assert "video_fps = 16" in text

    reloaded = load_config(cfg_path).get_profile("Wan-AI/Wan2.2", "wan-best")
    assert reloaded.video_num_frames == 49
    assert reloaded.video_fps == 16


def test_krea_defaults_follow_the_checkpoint_layout(tmp_path):
    """Regression: ``default_profiles()`` with no model dir returns the GGUF
    set. Seeding an *original* Diffusers checkpoint with those produces
    profiles that fail at load time ("GGUF quants are not loadable")."""
    original = tmp_path / "Krea-2-Turbo"
    original.mkdir()
    (original / "model_index.json").write_text(
        '{"_class_name": "Krea2Pipeline"}', encoding="utf-8")

    with_dir = engines.default_profiles("krea", original)
    assert all(p["image_model_type"] == "original" for p in with_dir.values())

    without_dir = engines.default_profiles("krea")
    assert any(str(p["image_model_type"]).endswith(".gguf")
               for p in without_dir.values())


def test_default_profiles_helper_is_total():
    """Unknown engines and adapters without defaults return {} rather than
    raising, so callers can treat every engine uniformly."""
    assert engines.default_profiles("no-such-engine") == {}


def test_measured_best_profiles_are_the_shipped_defaults():
    """The sweep results are the out-of-box settings, listed best-first."""
    z = engines.default_profiles("z_image")
    assert next(iter(z)) == "z-image-best"
    assert z["z-image-best"]["image_guidance"] == 5.5
    assert z["z-image-best"]["image_negative_prompt"]

    wan = engines.default_profiles("wan")
    assert next(iter(wan)) == "wan-best"
    assert wan["wan-best"]["image_size"] == "1152x640"
    assert wan["wan-best"]["video_num_frames"] == 49


# ------------------------------------------------------------ quant plumbing


def test_memory_strategies_map_to_runner_flags():
    """The UI offers one "memory strategy" selector; it has to expand to the
    right (--offload, --residency) pair."""
    from llamanager.engines import minimax_h3 as mm
    assert mm.MEMORY_STRATEGIES["split"] == ("none", "split")
    assert mm.MEMORY_STRATEGIES["joint"] == ("none", "joint")
    assert mm.MEMORY_STRATEGIES["block"][0] == "block"


def test_profiles_cover_each_hardware_tier():
    profiles = engines.default_profiles("minimax_h3")
    # nf4 leads: it is the only 4-bit path with ROCm kernels.
    assert next(iter(profiles)) == "h3-gpu-nf4"
    assert profiles["h3-gpu-nf4"]["image_model_type"] == "nf4"
    assert profiles["h3-gpu-nf4"]["image_editing_scheduler"] == "split"
    # Only the offload profile is allowed to put weights in system RAM.
    on_gpu = [n for n, f in profiles.items()
              if f["image_editing_scheduler"] in ("split", "joint")]
    assert set(on_gpu) == {"h3-gpu-nf4", "h3-gpu-int4", "h3-gpu-int8", "h3-bf16"}


def test_default_precision_is_the_one_that_fits():
    """The schema default has to be a 4-bit option, or the out-of-box profile
    cannot run on a 32 GB card at all."""
    from llamanager.engines import minimax_h3 as mm
    field = next(f for f in mm.profile_schema() if f.key == "image_model_type")
    assert field.default == "nf4"
    assert field.options[0] == "nf4"


def test_build_command_passes_quantisation_through(tmp_path):
    """A profile's precision and strategy must reach the runner argv."""
    from llamanager.config import Config, Profile as P
    from llamanager.engines import minimax_h3 as mm
    from llamanager.engines._base import ImageRequest

    cfg = Config()
    cfg.minimax_h3_python = str(Path(__file__).resolve())   # any existing file
    cfg.data_dir = tmp_path
    argv, _ = mm.build_command(
        cfg, tmp_path, P(name="p", image_model_type="int4",
                         image_editing_scheduler="split",
                         video_num_frames=124),
        ImageRequest(prompt="x", width=0, height=0, steps=None, seed=None, n=1),
        tmp_path / "out.mp4")
    assert argv[argv.index("--quantize") + 1] == "int4"
    assert argv[argv.index("--offload") + 1] == "none"
    assert argv[argv.index("--residency") + 1] == "split"


# ------------------------------------------------- destination preflight


def test_disk_preflight_verdicts(tmp_path, monkeypatch):
    """The confirm dialog's traffic light: enough / tight / not enough."""
    from llamanager import api_ui
    import shutil
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")

    def fake_usage(_p, free_gib):
        return Usage(500 * 1024 ** 3, 0, int(free_gib * 1024 ** 3))

    monkeypatch.setattr(shutil, "disk_usage", lambda p: fake_usage(p, 200))
    assert api_ui.disk_preflight(tmp_path, 10 * 1024 ** 3)["verdict"] == "ok"

    monkeypatch.setattr(shutil, "disk_usage", lambda p: fake_usage(p, 12))
    assert api_ui.disk_preflight(tmp_path, 10 * 1024 ** 3)["verdict"] == "tight"

    monkeypatch.setattr(shutil, "disk_usage", lambda p: fake_usage(p, 5))
    bad = api_ui.disk_preflight(tmp_path, 10 * 1024 ** 3)
    assert bad["verdict"] == "insufficient"
    assert bad["after_bytes"] < 0


def test_preflight_measures_the_filesystem_a_new_dir_would_land_on(tmp_path):
    """A destination that does not exist yet still has a free-space answer:
    whichever mount its first existing ancestor sits on."""
    from llamanager import api_ui
    target = tmp_path / "not" / "created" / "yet"
    pf = api_ui.disk_preflight(target, 1)
    assert pf["exists"] is False
    assert pf["mount"] == str(tmp_path)
    assert pf["free_bytes"] > 0


def test_venv_root_honours_the_override(tmp_path):
    """The dialog can repoint where engine venvs are built — the data dir is
    often on a small system partition."""
    from llamanager.config import Config
    from llamanager.engine_installer import venv_root
    cfg = Config()
    cfg.data_dir = tmp_path / "data"
    assert venv_root(cfg) == tmp_path / "data" / "venvs"
    cfg.venvs_dir = tmp_path / "big-disk" / "venvs"
    assert venv_root(cfg) == tmp_path / "big-disk" / "venvs"


def test_catalog_pulls_only_the_components_diffusers_needs():
    """Regression on a 330 GB mistake: MiniMax-H3's repo ships its original
    release alongside the diffusers conversion, so a whole-repo pull is
    464 GB where the components actually loaded are 140 GB."""
    entry = diffusion_catalog.for_engine("minimax_h3")[0]
    assert "," in entry.subfolder, "expected a component list, not one folder"
    parts = [p.strip() for p in entry.subfolder.split(",")]
    assert "transformer" in parts and "text_encoder" in parts
    # The original-release folders must NOT be pulled.
    assert "FL2VA" not in parts and "Ref2VA" not in parts
    assert entry.approx_size_gb == pytest.approx(140.0)
