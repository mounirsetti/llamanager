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
    assert ENGINE_FAMILY["krea"] == "image"
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


def test_detect_engine_for_krea_dir(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "Krea-2-Turbo-GGUF"
    d.mkdir()
    (d / "krea2_turbo-Q6_K.gguf").write_bytes(b"")
    assert detect_engine_for_path(d) == "krea"


def test_detect_engine_for_original_krea_dir(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "Krea-2-Turbo"
    d.mkdir()
    (d / "model_index.json").write_text('{"_class_name":"Krea2Pipeline"}')
    assert detect_engine_for_path(d) == "krea"


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


def test_krea_adapter_builds_argv_for_selected_quant(tmp_path: Path):
    from llamanager.engines import krea
    from llamanager.config import Config, Profile
    from llamanager.engines._base import ImageRequest

    fake_py = tmp_path / "python"
    fake_py.write_bytes(b"")
    cfg = Config()
    cfg.z_image_python = str(fake_py)
    model = tmp_path / "vantagewithai" / "Krea-2-Turbo-GGUF"
    model.mkdir(parents=True)
    (model / "krea2_turbo-Q6_K.gguf").write_bytes(b"")
    (model / "krea2_turbo-Q8_0.gguf").write_bytes(b"")
    prof = Profile(
        name="krea",
        image_model_type="krea2_turbo-Q8_0.gguf",
        image_steps=8,
        image_guidance=1.0,
        image_negative_prompt="blur",
        image_lora_weights="gokaygokay/Krea-2-Realism-LoRA",
        image_lora_scale=0.8,
    )
    req = ImageRequest(
        prompt="test", width=1024, height=1024,
        steps=None, seed=123, n=1,
    )
    argv, env = krea.build_command(cfg, model, prof, req, tmp_path / "out.png")
    assert "_krea_runner.py" in argv[2]
    assert "--gguf" in argv
    assert argv[argv.index("--gguf") + 1].endswith("krea2_turbo-Q8_0.gguf")
    assert "--negative_prompt" in argv
    assert "blur" in argv
    assert "--lora" in argv
    assert argv[argv.index("--lora") + 1] == "gokaygokay/Krea-2-Realism-LoRA"
    assert "--lora-scale" in argv
    assert float(argv[argv.index("--lora-scale") + 1]) == 0.8
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_krea_adapter_builds_argv_for_original_repo(tmp_path: Path):
    from llamanager.engines import krea
    from llamanager.config import Config, Profile
    from llamanager.engines._base import ImageRequest

    fake_py = tmp_path / "python"
    fake_py.write_bytes(b"")
    cfg = Config()
    cfg.z_image_python = str(fake_py)
    model = tmp_path / "krea" / "Krea-2-Turbo"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text('{"_class_name":"Krea2Pipeline"}')
    prof = Profile(name="krea-original", image_model_type="original")
    req = ImageRequest(
        prompt="test", width=1024, height=1024,
        steps=None, seed=None, n=1,
    )
    argv, _env = krea.build_command(cfg, model, prof, req, tmp_path / "out.png")
    assert "--model_path" in argv
    assert argv[argv.index("--model_path") + 1] == str(model)
    assert "--gguf" not in argv


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
