"""Tests for the image-engine plumbing.

Smoke-tests the additive structure (engine families, image profile
round-trip, adapter detection, queue family routing, API auth gating,
yield_to_image swap-and-restore semantics) without actually running an
image engine — neither hidream nor flux2 are available in CI.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


# ---------- config + profile ----------

def test_engine_family_lookup():
    from llamanager.config import ENGINE_FAMILY, engine_family
    assert ENGINE_FAMILY["llama"] == "text"
    assert ENGINE_FAMILY["mlx"] == "text"
    assert ENGINE_FAMILY["hidream"] == "image"
    assert ENGINE_FAMILY["flux2"] == "image"
    assert ENGINE_FAMILY["ideogram4"] == "image"
    assert engine_family("llama") == "text"
    assert engine_family("hidream") == "image"
    # Unknown engines fall back to text so legacy configs keep working.
    assert engine_family("foobar") == "text"


def test_detect_engine_for_hidream_dir(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "HiDream-O1-Image"
    d.mkdir()
    (d / "tokenizer_config.json").write_text("{}")
    (d / "preprocessor_config.json").write_text("{}")
    (d / "model-00001-of-00008.safetensors").write_bytes(b"")
    assert detect_engine_for_path(d) == "hidream"


def test_detect_engine_for_flux2_dir(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "flux2-dev"
    d.mkdir()
    (d / "flux2-dev-Q6_K.gguf").write_bytes(b"")
    (d / "ae.safetensors").write_bytes(b"")
    assert detect_engine_for_path(d) == "flux2"


def test_detect_engine_for_ideogram4_dir(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "ideogram-4-fp8"
    d.mkdir()
    (d / "model_index.json").write_text('{"_class_name":"Ideogram4Pipeline"}')
    assert detect_engine_for_path(d) == "ideogram4"


def test_detect_engine_for_comfy_ideogram4_dir(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "Comfy-Org" / "Ideogram-4"
    (d / "diffusion_models").mkdir(parents=True)
    (d / "diffusion_models" / "ideogram4_fp8_scaled.safetensors").write_bytes(b"")
    assert detect_engine_for_path(d) == "ideogram4"


def test_detect_engine_for_mlx_dir_still_works(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "mlx-model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "weights.safetensors").write_bytes(b"")
    assert detect_engine_for_path(d) == "mlx"


def test_image_profile_roundtrips_through_toml(tmp_path: Path):
    """Save + reload an image profile via the on-disk TOML."""
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, Profile, load_config, save_profile,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    # Override the data_dir so we don't touch ~/.llamanager.
    import tomlkit
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))

    cfg = load_config(cfg_path)
    prof = Profile(
        name="hidream-dev",
        image_model_type="dev",
        image_steps=28,
        image_size="2048x2048",
        image_seed=42,
        image_guidance=None,
    )
    save_profile(cfg.config_path, "HiDream-ai/HiDream-O1-Image", "hidream-dev", prof)

    reloaded = load_config(cfg_path)
    m = reloaded.get_model("HiDream-ai/HiDream-O1-Image")
    assert m is not None
    p = m.profiles["hidream-dev"]
    assert p.image_model_type == "dev"
    assert p.image_steps == 28
    assert p.image_size == "2048x2048"
    assert p.image_seed == 42


def test_image_config_section_roundtrip(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, load_config, update_image_config,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    import tomlkit
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))

    update_image_config(
        cfg_path,
        hidream_python="/opt/venv/bin/python",
        hidream_repo="/opt/HiDream-O1-Image",
        flux2_sd_cli="/opt/sd-cli",
        flux2_device_index=1,
    )
    cfg = load_config(cfg_path)
    assert cfg.hidream_python == "/opt/venv/bin/python"
    assert cfg.hidream_repo == "/opt/HiDream-O1-Image"
    assert cfg.flux2_sd_cli == "/opt/sd-cli"
    assert cfg.flux2_device_index == 1


def test_coexistence_policy_defaults_preserve_single_slot_invariant(tmp_path: Path):
    from llamanager.config import load_config
    cfg = load_config(tmp_path / "missing.toml")
    # Default: swap to image, restore text after. Not concurrent.
    assert cfg.unload_text_on_arrival is True
    assert cfg.restart_text_after_image is True
    assert cfg.allow_concurrent is False


# ---------- adapter ----------

def test_hidream_adapter_builds_argv(tmp_path: Path):
    from llamanager.engines import hidream
    from llamanager.engines._base import ImageRequest
    from llamanager.config import Config, Profile

    repo = tmp_path / "HiDream-O1-Image"
    repo.mkdir()
    (repo / "inference.py").write_text("print('ok')")
    venv = tmp_path / ".venv-hidream" / "bin"
    venv.mkdir(parents=True)
    py = venv / "python"
    py.write_text("")
    py.chmod(0o755)

    cfg = Config(
        data_dir=tmp_path,
        hidream_python=str(py),
        hidream_repo=str(repo),
    )
    model = tmp_path / "models" / "HiDream-O1-Image"
    model.mkdir(parents=True)
    (model / "tokenizer_config.json").write_text("{}")
    (model / "preprocessor_config.json").write_text("{}")
    (model / "shard-0.safetensors").write_bytes(b"")

    prof = Profile(
        name="hidream-dev",
        image_model_type="dev",
        image_steps=28,
        image_size="2048x2048",
        image_seed=42,
    )
    req = ImageRequest(prompt="a dog", width=0, height=0, steps=None, seed=None, n=1)
    out = tmp_path / "out.png"
    argv, env = hidream.build_command(cfg, model, prof, req, out)
    assert argv[0] == str(py)
    assert "--model_path" in argv
    assert str(model) in argv
    assert "--model_type" in argv
    assert "dev" in argv
    assert "--prompt" in argv
    assert "a dog" in argv
    # 2048 — adapter snaps to bucket from profile.
    assert "--width" in argv
    assert "2048" in argv
    # Per-request seed wins over profile seed when set.


def test_flux2_adapter_builds_argv_with_env(tmp_path: Path):
    from llamanager.engines import flux2
    from llamanager.engines._base import ImageRequest
    from llamanager.config import Config, Profile

    sd_cli = tmp_path / "sd-cli"
    sd_cli.write_text("")
    sd_cli.chmod(0o755)
    model = tmp_path / "models" / "flux2-dev"
    model.mkdir(parents=True)
    (model / "flux2-dev-Q6_K.gguf").write_bytes(b"")
    (model / "ae.safetensors").write_bytes(b"")
    (model / "Mistral-Small-3.2.gguf").write_bytes(b"")

    cfg = Config(
        data_dir=tmp_path,
        flux2_sd_cli=str(sd_cli),
        flux2_device_index=1,
    )
    prof = Profile(
        name="flux2-fast",
        image_size="1024x1024",
        image_steps=8,
        image_guidance=1.0,
    )
    req = ImageRequest(prompt="oil painting of pears", width=0, height=0,
                       steps=None, seed=None, n=1)
    out = tmp_path / "out.png"
    argv, env = flux2.build_command(cfg, model, prof, req, out)
    assert argv[0] == str(sd_cli)
    assert "--diffusion-model" in argv
    assert "--vae" in argv
    assert "--cfg-scale" in argv
    assert "1.0" in argv
    assert env.get("GGML_VK_VISIBLE_DEVICES") == "1"


def test_hidream_progress_parser():
    from llamanager.engines import hidream
    ev = hidream.parse_progress("step 14/28 [00:18<00:18, 0.78it/s]")
    assert ev is not None
    assert ev.step == 14
    assert ev.total == 28
    # Garbage lines return None.
    assert hidream.parse_progress("[INFO] cuda is available") is None


def test_flux2_progress_parser():
    from llamanager.engines import flux2
    ev = flux2.parse_progress("  3/28  [ 18.15s/it]")
    assert ev is not None
    assert ev.step == 3
    assert ev.total == 28


def test_krea_lora_profile_fields_roundtrip(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, Profile, load_config, save_profile,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    import tomlkit
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))

    prof = Profile(
        name="krea-realism",
        image_model_type="original",
        image_lora_weights="gokaygokay/Krea-2-Realism-LoRA",
        image_lora_scale=1.0,
    )
    save_profile(cfg_path, "krea/Krea-2-Turbo", "krea-realism", prof)
    cfg = load_config(cfg_path)
    p = cfg.models["krea/Krea-2-Turbo"].profiles["krea-realism"]
    assert p.image_lora_weights == "gokaygokay/Krea-2-Realism-LoRA"
    assert p.image_lora_scale == 1.0


def test_ideogram4_adapter_builds_argv(tmp_path: Path):
    from llamanager.engines import ideogram4
    from llamanager.config import Config, Profile
    from llamanager.engines._base import ImageRequest

    fake_py = tmp_path / "python"
    fake_py.write_bytes(b"")
    cfg = Config()
    cfg.ideogram4_python = str(fake_py)
    model = tmp_path / "ideogram-ai" / "ideogram-4-fp8"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text('{"_class_name":"Ideogram4Pipeline"}')
    prof = Profile(
        name="ideogram4-fp8",
        image_model_type="fp8",
        image_size="1024x1024",
        image_editing_scheduler="V4_QUALITY_48",
        image_seed=7,
        args={"magic_prompt": False, "warn_on_caption_issues": True},
    )
    req = ImageRequest(
        prompt='{"high_level_description":"poster"}',
        width=0, height=0, steps=None, seed=None, n=1,
    )
    argv, env = ideogram4.build_command(cfg, model, prof, req, tmp_path / "out.png")
    assert "_ideogram4_runner.py" in argv[2]
    assert "--weights-repo" in argv
    assert argv[argv.index("--weights-repo") + 1] == str(model)
    assert "--quantization" in argv
    assert argv[argv.index("--quantization") + 1] == "fp8"
    assert "--sampler-preset" in argv
    assert argv[argv.index("--sampler-preset") + 1] == "V4_QUALITY_48"
    assert "--no-magic-prompt" in argv
    assert "--warn-on-caption-issues" in argv
    assert env["PYTHONIOENCODING"] == "utf-8"


# ---------- queue routing ----------

def test_queue_infers_image_task_type(tmp_path: Path):
    """Verify that enqueuing a request for an image-family model routes
    it to ``task_type='image'`` so the dispatcher skips the text-swap path.
    """
    from llamanager.auth import AuthManager, Origin, load_or_create_lookup_secret
    from llamanager.config import Config
    from llamanager.db import DB
    from llamanager.queue_mgr import QueueManager
    from llamanager.server_manager import ServerManager

    data = tmp_path / "llamanager"
    data.mkdir()
    (data / "logs").mkdir()
    models_dir = data / "models"
    models_dir.mkdir()
    cfg = Config(data_dir=data)

    # Plant a hidream-shaped model on disk so detect_engine_for_id sees it.
    hidream = models_dir / "HiDream-O1-Image"
    hidream.mkdir()
    (hidream / "tokenizer_config.json").write_text("{}")
    (hidream / "preprocessor_config.json").write_text("{}")
    (hidream / "shard.safetensors").write_bytes(b"")

    db = DB(data / "state.db")
    sm = ServerManager(cfg, db)
    qm = QueueManager(cfg, db, sm)
    # Hand-build an Origin (skip AuthManager).
    origin = Origin(id=1, name="test", priority=50,
                    allowed_models=["*"], is_admin=False,
                    created_at=0.0)

    async def go():
        # Image-family model → task_type=image.
        req = await qm.enqueue(origin=origin,
                                model_required="HiDream-O1-Image")
        assert req.task_type == "image"
        # GGUF (text-family) → task_type=text.
        (models_dir / "tiny.gguf").write_bytes(b"")
        req2 = await qm.enqueue(origin=origin,
                                 model_required="tiny.gguf")
        assert req2.task_type == "text"

    asyncio.run(go())
    db.close()


# ---------- API auth ----------

def test_images_endpoint_requires_bearer(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "test", "model": "hidream"},
    )
    assert resp.status_code == 401


def test_images_endpoint_rejects_disallowed_model(app):
    """A bearer token without the image model in its allowed_models gets a 403."""
    from fastapi.testclient import TestClient
    am = app.state.auth
    am.ensure_bootstrap()
    # Create an origin allowed to talk to one specific text model only.
    _, key = am.create_origin(name="restricted",
                              allowed_models=["tiny.gguf"])
    client = TestClient(app)
    resp = client.post(
        "/v1/images/generations",
        headers={"Authorization": f"Bearer {key}"},
        json={"prompt": "test", "model": "HiDream-O1-Image"},
    )
    assert resp.status_code == 403


# ---------- yield_to_image ----------

def test_yield_to_image_when_text_not_running_is_noop(cfg, tmp_path):
    """When no text engine is running, yield_to_image is a no-op."""
    from llamanager.db import DB
    from llamanager.server_manager import ServerManager

    db = DB(tmp_path / "state.db")
    sm = ServerManager(cfg, db)

    async def go():
        entered = False
        async with sm.yield_to_image():
            entered = True
            assert not sm.is_running
        assert entered

    asyncio.run(go())
    db.close()


def test_yield_to_image_skips_when_concurrent_mode(cfg, tmp_path):
    """allow_concurrent=True bypasses unload/restart entirely."""
    from llamanager.db import DB
    from llamanager.server_manager import ServerManager, StartSpec
    from pathlib import Path as _P

    cfg.allow_concurrent = True
    db = DB(tmp_path / "state.db")
    sm = ServerManager(cfg, db)
    # Pretend the server is running with a known spec.
    fake_spec = StartSpec(
        model_path=_P("/tmp/fake.gguf"),
        mmproj_path=None,
        extra_args={},
        profile_name=None,
        model_id="fake.gguf",
    )
    sm.spec = fake_spec
    # Force is_running True.
    class _P_:
        returncode = None
        pid = 12345
    sm.proc = _P_()  # type: ignore[assignment]

    async def go():
        async with sm.yield_to_image():
            # Concurrent mode: server was *not* stopped — spec preserved.
            assert sm.spec is fake_spec
            assert sm.is_running

    asyncio.run(go())
    db.close()


# ---------- reference-image helpers ----------

def test_decode_ref_image_accepts_raw_base64_png():
    """Bare base64 (no data URL) decodes when bytes start with a PNG header."""
    import base64
    from llamanager.api_v1 import _decode_ref_image
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    payload = base64.b64encode(png_bytes).decode("ascii")
    blob, ext = _decode_ref_image(payload, 0)
    assert blob == png_bytes
    assert ext == "png"


def test_decode_ref_image_accepts_data_url_jpeg():
    """data:image/jpeg;base64,... is parsed and sniffed by magic bytes."""
    import base64
    from llamanager.api_v1 import _decode_ref_image
    jpg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 32
    payload = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode("ascii")
    blob, ext = _decode_ref_image(payload, 0)
    assert blob == jpg_bytes
    assert ext == "jpg"


def test_decode_ref_image_rejects_non_image_bytes():
    """Bytes that don't match PNG/JPEG/WebP magic raise a 400."""
    import base64
    from fastapi import HTTPException
    from llamanager.api_v1 import _decode_ref_image
    payload = base64.b64encode(b"this is plain text, not an image").decode("ascii")
    try:
        _decode_ref_image(payload, 3)
    except HTTPException as e:
        assert e.status_code == 400
        assert "image[3]" in e.detail
    else:
        raise AssertionError("expected HTTPException")


def test_image_request_carries_ref_fields():
    """ImageRequest stores ref-image fields and they survive copy."""
    from llamanager.engines._base import ImageRequest
    from pathlib import Path as _P
    req = ImageRequest(
        prompt="x", width=0, height=0, steps=None, seed=None, n=3,
        ref_images=[_P("/tmp/a.png"), _P("/tmp/b.png")],
        keep_original_aspect=True,
        layout_bboxes="[[0.1,0.4,0.2,0.6]]",
        strength=0.65,
    )
    assert req.ref_images == [_P("/tmp/a.png"), _P("/tmp/b.png")]
    assert req.keep_original_aspect is True
    assert req.layout_bboxes == "[[0.1,0.4,0.2,0.6]]"
    assert req.strength == 0.65


def test_hidream_adapter_emits_ref_flags(tmp_path: Path):
    """HiDream's build_command forwards --ref_images, --keep_original_aspect,
    and --editing_scheduler from request + profile."""
    from llamanager.engines import hidream
    from llamanager.config import Config, Profile
    from llamanager.engines._base import ImageRequest
    cfg = Config()
    fake_py = tmp_path / "python.exe"
    fake_py.write_text("")
    cfg.hidream_python = str(fake_py)
    cfg.hidream_repo = str(tmp_path)
    (tmp_path / "inference.py").write_text("")
    model_dir = tmp_path / "HiDream-O1-Image"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text("{}")
    (model_dir / "preprocessor_config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"")
    prof = Profile(name="hidream-dev", image_model_type="dev",
                   image_editing_scheduler="flow_match")
    refs = [tmp_path / "ref0.png"]
    refs[0].write_bytes(b"\x89PNG\r\n\x1a\n")
    req = ImageRequest(
        prompt="edit me", width=2048, height=2048,
        steps=None, seed=42, n=1,
        ref_images=refs, keep_original_aspect=True,
    )
    argv, env = hidream.build_command(cfg, model_dir, prof, req,
                                       tmp_path / "out.png")
    assert "--ref_images" in argv
    assert str(refs[0]) in argv
    assert "--keep_original_aspect" in argv
    assert "--editing_scheduler" in argv
    sched_idx = argv.index("--editing_scheduler")
    assert argv[sched_idx + 1] == "flow_match"
    # UTF-8 env survives.
    assert env.get("PYTHONIOENCODING") == "utf-8"


def test_flux2_adapter_rejects_multiple_refs(tmp_path: Path):
    """Flux2 only supports one reference (img2img); two refs => RuntimeError."""
    import pytest as _pytest
    from llamanager.engines import flux2
    from llamanager.config import Config, Profile
    from llamanager.engines._base import ImageRequest
    cfg = Config()
    sd_cli = tmp_path / "sd-cli.exe"
    sd_cli.write_bytes(b"")
    cfg.flux2_sd_cli = str(sd_cli)
    model_dir = tmp_path / "flux2-dev"
    model_dir.mkdir()
    (model_dir / "flux2-dev-Q6_K.gguf").write_bytes(b"")
    (model_dir / "ae.safetensors").write_bytes(b"")
    refs = [tmp_path / "a.png", tmp_path / "b.png"]
    for r in refs:
        r.write_bytes(b"\x89PNG\r\n\x1a\n")
    req = ImageRequest(
        prompt="img2img", width=1024, height=1024,
        steps=None, seed=None, n=1, ref_images=refs, strength=0.6,
    )
    with _pytest.raises(RuntimeError, match="at most one reference"):
        flux2.build_command(cfg, model_dir, Profile(name="x"), req,
                            tmp_path / "out.png")


def test_flux2_adapter_emits_init_img_and_strength(tmp_path: Path):
    """Single ref + strength => -i <path> --strength <s>."""
    from llamanager.engines import flux2
    from llamanager.config import Config, Profile
    from llamanager.engines._base import ImageRequest
    cfg = Config()
    sd_cli = tmp_path / "sd-cli.exe"
    sd_cli.write_bytes(b"")
    cfg.flux2_sd_cli = str(sd_cli)
    model_dir = tmp_path / "flux2-dev"
    model_dir.mkdir()
    (model_dir / "flux2-dev-Q6_K.gguf").write_bytes(b"")
    (model_dir / "ae.safetensors").write_bytes(b"")
    ref = tmp_path / "init.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    req = ImageRequest(
        prompt="vary me", width=1024, height=1024,
        steps=None, seed=None, n=1, ref_images=[ref], strength=0.65,
    )
    argv, _env = flux2.build_command(cfg, model_dir, Profile(name="x"), req,
                                      tmp_path / "out.png")
    assert "-i" in argv
    assert str(ref) in argv
    assert "--strength" in argv
    s_idx = argv.index("--strength")
    assert float(argv[s_idx + 1]) == 0.65


def test_new_image_filename_uses_engine_prefix_and_hhmm(tmp_path: Path):
    """Filenames follow <eng><hhmm>[-NN].png with engine-prefix + wall-clock
    time, and collide-safely append -2, -3 within the same minute."""
    import re
    from llamanager.image_runner import _new_image_filename
    fn1 = _new_image_filename("hidream", tmp_path)
    assert re.fullmatch(r"hid\d{4}\.png", fn1), fn1
    # Pre-create that exact name so the next call has to disambiguate.
    (tmp_path / fn1).write_bytes(b"")
    fn2 = _new_image_filename("hidream", tmp_path)
    assert re.fullmatch(r"hid\d{4}-2\.png", fn2), fn2
    (tmp_path / fn2).write_bytes(b"")
    fn3 = _new_image_filename("hidream", tmp_path)
    assert re.fullmatch(r"hid\d{4}-3\.png", fn3), fn3
    # Different engine → different prefix.
    fn_flux = _new_image_filename("flux2", tmp_path)
    assert re.fullmatch(r"flu\d{4}\.png", fn_flux), fn_flux
    # Unknown engine name still produces a 3-letter prefix.
    fn_other = _new_image_filename("z", tmp_path)
    assert re.fullmatch(r"z\d{4}\.png", fn_other), fn_other


def test_profile_roundtrips_new_image_ref_fields(tmp_path: Path):
    """image_editing_scheduler and image_strength survive save+reload."""
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, Profile, load_config, save_profile,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    import tomlkit
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))

    prof = Profile(
        name="hidream-edit",
        image_model_type="dev",
        image_editing_scheduler="flow_match",
        image_strength=0.55,
    )
    save_profile(cfg_path, "HiDream-O1-Image", "hidream-edit", prof)
    reloaded = load_config(cfg_path)
    m = reloaded.get_model("HiDream-O1-Image")
    assert m is not None
    p = m.profiles["hidream-edit"]
    assert p.image_editing_scheduler == "flow_match"
    assert p.image_strength == 0.55


def test_krea_comfy_lora_field_is_restricted_to_the_loras_folder(tmp_path):
    """ComfyUI can only load a LoRA that is already in the model's folder,
    so the field is a strict picker — no free-typed repo ids."""
    from llamanager.engines import krea_comfy

    cf = next(f for f in krea_comfy.profile_schema()
              if f.key == "image_lora_weights")
    assert cf.options_dir == "loras" and cf.options_free is False


def test_images_endpoint_forwards_a_request_step_count():
    """Regression: /v1/images/generations parsed `steps` and then built the
    ImageRequest with steps=None, so the composer's per-request override (and
    any API caller's) was accepted and silently ignored. Every adapter's
    _resolved_steps() prefers req.steps over the profile, so the value has to
    survive the handler."""
    import inspect
    from llamanager import api_v1

    src = inspect.getsource(api_v1.images_generations)
    assert "steps=steps_override" in src, (
        "images_generations must pass the parsed step override into "
        "ImageRequest, not None")
    assert 'body.get("steps")' in src


def test_hidream_probe_and_spawn_share_the_rocm_env(monkeypatch):
    """The --num_inference_steps probe runs `inference.py --help`, which
    imports torch. Without the ROCm lib dirs that import dies and the probe
    reads the failure as "flag unsupported", silently dropping every step
    override — so the probe must use the same env as the spawn."""
    from llamanager.engines import hidream

    monkeypatch.setattr("llamanager.gpu_detect.rocm_lib_dirs",
                        lambda: ["/opt/rocm/lib"])
    env = hidream._rocm_env()
    assert env["LD_LIBRARY_PATH"].startswith("/opt/rocm/lib")

    seen = {}

    class _R:
        stdout = "--num_inference_steps"
        stderr = ""

    def _fake_run(argv, **kw):
        seen.update(kw)
        return _R()

    monkeypatch.setattr(hidream.subprocess, "run", _fake_run)
    hidream._HELP_PROBE_CACHE.clear()
    assert hidream._supports_steps_flag(Path("/py"), Path(__file__)) is True
    assert "/opt/rocm/lib" in seen["env"]["LD_LIBRARY_PATH"]


# ---- request-level overrides (model_type / guidance / editing_scheduler) ----

def _img_req(**kw):
    from llamanager.engines._base import ImageRequest
    base = dict(prompt="p", width=0, height=0, steps=None, seed=None, n=1)
    base.update(kw)
    return ImageRequest(**base)


def test_request_overrides_beat_the_profile_and_the_engine_default():
    """Precedence is request > profile > engine default, for all three knobs."""
    from llamanager.config import Profile
    from llamanager.engines import _base

    empty, prof = Profile(name="e"), Profile(
        name="p", image_model_type="dev", image_guidance=5.0,
        image_editing_scheduler="flash")

    # Nothing set anywhere.
    assert _base.pick_model_type(_img_req(), empty) == ""
    assert _base.pick_guidance(_img_req(), empty) is None
    assert _base.pick_scheduler(_img_req(), empty) == ""
    # Profile only.
    assert _base.pick_model_type(_img_req(), prof) == "dev"
    assert _base.pick_guidance(_img_req(), prof) == 5.0
    assert _base.pick_scheduler(_img_req(), prof) == "flash"
    # Request wins, including over a profile that set the same knob.
    req = _img_req(model_type="full", guidance=1.5, editing_scheduler="flow_match")
    assert _base.pick_model_type(req, prof) == "full"
    assert _base.pick_guidance(req, prof) == 1.5
    assert _base.pick_scheduler(req, prof) == "flow_match"
    # And over an empty profile — the reported bug: "Recipe: full" with the
    # profile picker left on "(use engine defaults)" used to fall through to
    # the engine default and silently run dev.
    assert _base.pick_model_type(req, empty) == "full"
    # guidance 0.0 is a value, not "unset".
    assert _base.pick_guidance(_img_req(guidance=0.0), prof) == 0.0


def test_images_endpoint_forwards_the_three_dropped_overrides():
    """Regression: the composer sends model_type/guidance/editing_scheduler
    and the handler discarded all three."""
    import inspect
    from llamanager import api_v1

    src = inspect.getsource(api_v1.images_generations)
    for field in ("model_type", "guidance", "editing_scheduler"):
        assert f"{field}={field}_override" in src, field


def test_hidream_sidecar_reports_what_the_recipe_actually_runs(tmp_path):
    """The dev recipe hardwires 28 steps and guidance 0.0 whatever was asked,
    so effective_params must correct the request rather than echo it."""
    from llamanager.config import Config, Profile
    from llamanager.engines import hidream

    cfg = Config()
    dev = hidream.effective_params(cfg, tmp_path, Profile(name="p"),
                                   _img_req(steps=100))
    assert dev["model_type"] == "dev"
    assert dev["steps"] == 28 and dev["guidance"] == 0.0

    full = hidream.effective_params(
        cfg, tmp_path, Profile(name="p"),
        _img_req(model_type="full", steps=12, guidance=4.0))
    assert full["model_type"] == "full"
    assert full["steps"] == 12 and full["guidance"] == 4.0


def test_per_image_request_carries_every_field():
    """The runner splits an n>1 request into one ImageRequest per sample. It
    used to list the fields by hand, so anything added to ImageRequest was
    dropped before the adapter saw it — the bug that made model_type,
    guidance and editing_scheduler inert. Only per-sample values may differ."""
    import dataclasses
    import inspect
    from llamanager import image_runner

    src = inspect.getsource(image_runner.ImageTaskRunner.run)
    assert "per_req = replace(" in src, (
        "build the per-sample request with dataclasses.replace so new fields "
        "are carried automatically")

    req = _img_req(model_type="full", guidance=4.0,
                   editing_scheduler="flow_match", steps=17, seed=None)
    per = dataclasses.replace(req, seed=99, n=1, ref_images=[])
    for f in dataclasses.fields(req):
        if f.name in ("seed", "n", "ref_images"):
            continue
        assert getattr(per, f.name) == getattr(req, f.name), f.name


# ---- reference provenance in the sidecar ----------------------------------

def test_ref_thumbnails_are_small_self_contained_data_uris(tmp_path):
    """The staged reference files are deleted when a run finishes, so the
    sidecar embeds thumbnails instead of naming paths that will not exist.
    They have to stay small: eight of them (the per-request cap) ride along
    in every sidecar."""
    from PIL import Image
    from llamanager.image_runner import _ref_thumbnails

    big = tmp_path / "ref.png"
    Image.new("RGB", (2048, 2048), (180, 60, 40)).save(big)
    out = _ref_thumbnails([big])
    assert len(out) == 1
    assert out[0].startswith("data:image/jpeg;base64,")
    assert len(out[0]) < 120_000, "thumbnail should be tens of KB, not MB"

    # A missing or corrupt reference must not cost the operator an image.
    assert _ref_thumbnails([tmp_path / "nope.png"]) == []


def test_runner_records_references_before_deleting_them():
    """Regression: the sidecar has to be built while the staged refs still
    exist — the runner removes their directory once the run completes."""
    import inspect
    from llamanager import image_runner

    src = inspect.getsource(image_runner.ImageTaskRunner.run)
    assert "_ref_thumbnails(per_req.ref_images)" in src
    assert src.index("_ref_thumbnails") < src.index("rmtree"), (
        "thumbnails must be taken before the ref directory is removed")


def test_favicon_keeps_a_clear_pixel_between_the_wordmark_and_the_dot():
    """The tab icon is drawn into a 16px box: the whole 512-unit tile maps to
    16 device pixels, so 1px = 32 units. The tray icon's 15-unit gap is 0.47px
    there — the dot and the "m" land in the same pixel and fuse, which is what
    "the dot prints on top of the m" looked like. Guard the two numbers that
    keep them apart."""
    import re
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    svg = (root / "assets" / "favicon.svg").read_text(encoding="utf-8")
    circle = re.search(r'<circle cx="([\d.]+)" cy="[\d.]+" r="([\d.]+)"', svg)
    assert circle, "favicon should carry a single accent dot"
    cx, r = float(circle.group(1)), float(circle.group(2))

    # Where the outlined glyphs actually end: the path is placed by a
    # translate()/scale() pair, so read the transform and the path's extent.
    m = re.search(r'transform="translate\(([\d.-]+) [\d.-]+\) scale\(([\d.]+)\)"', svg)
    assert m, "glyphs should be outlined and placed with translate/scale"
    tx, s = float(m.group(1)), float(m.group(2))
    xs = [float(v) for v in re.findall(r'[ML](-?[\d.]+) -?[\d.]+', svg)]
    ink_end = tx + max(xs) * s

    units_per_px_at_16 = 512 / 16
    gap_px = (cx - r - ink_end) / units_per_px_at_16
    assert gap_px >= 1.0, f"only {gap_px:.2f}px between the m and the dot at 16px"
    assert (2 * r) / units_per_px_at_16 >= 2.0, "dot renders smaller than 2px at 16px"


# ---- per-request profile-field overrides (the composer's own controls) ----

def test_overrides_land_on_a_copy_of_the_profile():
    """The composer renders one control per profile field but the API only
    ever read a hand-picked few, so the LoRA picker was collected, sent and
    dropped — generating without the LoRA the operator chose."""
    from llamanager.api_v1 import _profile_with_overrides
    from llamanager.config import Profile

    base = Profile(name="kreac-best", image_steps=8,
                   image_lora_weights="old.safetensors")
    out = _profile_with_overrides(
        base, "krea_comfy",
        {"image_lora_weights": "Krea2-realism-V2.safetensors"})

    assert out.image_lora_weights == "Krea2-realism-V2.safetensors"
    assert out.image_steps == 8, "untouched fields must survive"
    assert base.image_lora_weights == "old.safetensors", "must not mutate"


def test_overrides_coerce_to_the_field_kind():
    from llamanager.api_v1 import _profile_with_overrides
    from llamanager.config import Profile

    out = _profile_with_overrides(
        Profile(name="p"), "krea_comfy",
        {"image_steps": "12", "image_lora_scale": "0.8"})
    assert out.image_steps == 12 and isinstance(out.image_steps, int)
    assert out.image_lora_scale == 0.8


def test_overrides_work_without_a_saved_profile():
    """Choosing controls without saving a profile is the common case."""
    from llamanager.api_v1 import _profile_with_overrides

    out = _profile_with_overrides(None, "krea_comfy", {"image_steps": 4})
    assert out is not None and out.image_steps == 4


def test_an_unknown_override_is_a_400_not_a_shrug():
    """Silently ignoring a knob the caller set is the bug being removed."""
    import pytest
    from fastapi import HTTPException
    from llamanager.api_v1 import _profile_with_overrides
    from llamanager.config import Profile

    with pytest.raises(HTTPException) as ei:
        _profile_with_overrides(Profile(name="p"), "krea_comfy",
                                {"image_lora_wieght": "x"})
    assert ei.value.status_code == 400
    assert "image_lora_wieght" in str(ei.value.detail)


def test_an_uncoercible_override_is_a_400():
    import pytest
    from fastapi import HTTPException
    from llamanager.api_v1 import _profile_with_overrides
    from llamanager.config import Profile

    with pytest.raises(HTTPException) as ei:
        _profile_with_overrides(Profile(name="p"), "krea_comfy",
                                {"image_steps": "eight"})
    assert ei.value.status_code == 400
    assert "image_steps" in str(ei.value.detail)


def test_blank_overrides_leave_the_profile_alone():
    """An untouched control sends "" — that means inherit, not clear."""
    from llamanager.api_v1 import _profile_with_overrides
    from llamanager.config import Profile

    base = Profile(name="p", image_lora_weights="keep.safetensors")
    out = _profile_with_overrides(base, "krea_comfy",
                                  {"image_lora_weights": ""})
    assert out.image_lora_weights == "keep.safetensors"


def test_both_generation_endpoints_apply_overrides():
    import inspect
    from llamanager import api_v1

    for fn in (api_v1.images_generations, api_v1.videos_generations):
        src = inspect.getsource(fn)
        assert "_profile_with_overrides" in src, fn.__name__
        assert 'body.get("overrides")' in src, fn.__name__


def test_a_schema_key_profile_cannot_carry_is_named_not_a_500():
    """An adapter declaring a field config.Profile has no slot for is a
    mismatch to report, not a TypeError inside dataclasses.replace."""
    import pytest
    from fastapi import HTTPException
    from llamanager.api_v1 import _profile_with_overrides
    from llamanager.config import Profile

    class _Field:
        key, kind = "image_not_a_real_field", "text"

    class _Adapter:
        @staticmethod
        def profile_schema():
            return [_Field()]

    import llamanager.engines as engines
    real = engines.get
    engines.get = lambda name: _Adapter if name == "fake" else real(name)
    try:
        with pytest.raises(HTTPException) as ei:
            _profile_with_overrides(Profile(name="p"), "fake",
                                    {"image_not_a_real_field": "x"})
    finally:
        engines.get = real
    assert ei.value.status_code == 400
    assert "image_not_a_real_field" in str(ei.value.detail)


# ------------------------------------------------ profile-aware arity guard


def _arity(engine="krea_comfy", model=None, profile=None, n=0):
    from llamanager.api_v1 import _check_ref_arity
    from llamanager.config import Config, Profile

    class _Cfg(Config):
        def __init__(self, prof):
            super().__init__()
            self._prof = prof

        def get_profile(self, model_id, name):
            return self._prof

    return _check_ref_arity(_Cfg(profile), engine, model, "p" if profile else None, n)


def test_arity_guard_enforces_the_profiles_maximum():
    import pytest
    from fastapi import HTTPException
    from llamanager.config import Profile

    prof = Profile(name="p",
                   image_lora_weights="krea2_identity_edit_v1_2.safetensors")
    with pytest.raises(HTTPException) as ei:
        _arity(model="Krea-2-Turbo-Comfy", profile=prof, n=3)
    assert ei.value.status_code == 400
    assert "at most 2" in str(ei.value.detail)


def test_arity_guard_enforces_the_minimum_even_with_zero_refs():
    """An image-to-video profile with no reference used to burn a queue slot
    and fail at dispatch."""
    import pytest
    from fastapi import HTTPException
    from llamanager.config import Profile

    prof = Profile(name="p", image_model_type="Q4_K_M-Turbo")
    with pytest.raises(HTTPException) as ei:
        _arity(engine="minimax_h3_comfy", model="MiniMax-H3-Comfy",
               profile=prof, n=0)
    assert ei.value.status_code == 400
    assert "opening frame" in str(ei.value.detail)


def test_arity_guard_refuses_a_ref_on_a_profile_that_cannot_read_one():
    import pytest
    from fastapi import HTTPException
    from llamanager.config import Profile

    with pytest.raises(HTTPException) as ei:
        _arity(model="Krea-2-Turbo-Comfy", profile=Profile(name="p"), n=2)
    assert "at most 1" in str(ei.value.detail)


def test_arity_guard_accepts_what_the_profile_accepts():
    from llamanager.config import Profile

    prof = Profile(name="p", image_model_type="Q4_K_M-Ref-Turbo")
    for n in (1, 5, 9):
        assert _arity(engine="minimax_h3_comfy", model="MiniMax-H3-Comfy",
                      profile=prof, n=n) is None


def test_the_videos_route_accepts_a_list_of_images():
    """REF2VA composes from up to nine references, which the singular
    ``image`` field made unreachable."""
    import inspect
    from llamanager import api_v1

    src = inspect.getsource(api_v1.videos_generations)
    assert 'body.get("images")' in src
    assert "_check_ref_arity" in src
    # And the images route uses the same guard.
    assert "_check_ref_arity" in inspect.getsource(api_v1.images_generations)
