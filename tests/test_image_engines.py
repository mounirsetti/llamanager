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
