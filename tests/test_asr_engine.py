"""Tests for the ASR (speech-to-text / Whisper) engine plumbing.

Smoke-tests the additive structure (audio engine family, detection ordering,
adapter argv/env, profile round-trip, queue family routing, installer reuse
plan) without actually running Whisper — torch/transformers aren't assumed in
CI. The real model run is covered by the manual end-to-end verification.
"""
from __future__ import annotations

import json
from pathlib import Path


# ---------- engine family ----------

def test_asr_engine_family():
    from llamanager.config import ENGINE_FAMILY, AUDIO_ENGINES, engine_family
    assert ENGINE_FAMILY["asr"] == "audio"
    assert "asr" in AUDIO_ENGINES
    assert engine_family("asr") == "audio"
    # Unknown still falls back to text.
    assert engine_family("foobar") == "text"


# ---------- detection ----------

def _make_whisper_dir(d: Path) -> None:
    """A realistic HF Whisper folder: it also has the tokenizer/preprocessor
    config + safetensors that would otherwise match hidream/mlx."""
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
    }))
    (d / "tokenizer_config.json").write_text("{}")
    (d / "preprocessor_config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"")


def test_detect_whisper_dir_is_asr_not_hidream(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "whisper-large-v3-turbo-ar-quran"
    _make_whisper_dir(d)
    # Despite having tokenizer_config + preprocessor_config + safetensors
    # (the hidream shape) and config.json + safetensors (the mlx shape),
    # ASR detection must win because it runs first.
    assert detect_engine_for_path(d) == "asr"


def test_detect_whisper_via_architectures_only(tmp_path: Path):
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "whisper-x"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "architectures": ["WhisperForConditionalGeneration"],
    }))
    (d / "model.safetensors").write_bytes(b"")
    assert detect_engine_for_path(d) == "asr"


def test_plain_mlx_dir_still_detects_mlx(tmp_path: Path):
    """Guard: the ASR check must not steal generic config.json+safetensors dirs."""
    from llamanager.config import detect_engine_for_path
    d = tmp_path / "mlx-model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "weights.safetensors").write_bytes(b"")
    assert detect_engine_for_path(d) == "mlx"


# ---------- adapter ----------

def test_asr_adapter_builds_argv_and_env(tmp_path: Path):
    from llamanager.engines import asr
    from llamanager.engines._base import AudioRequest
    from llamanager.config import Config, Profile

    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    py = venv / "python"
    py.write_text("")
    py.chmod(0o755)

    model = tmp_path / "models" / "whisper"
    _make_whisper_dir(model)

    cfg = Config(data_dir=tmp_path, asr_python=str(py))
    prof = Profile(name="quran-ar", audio_language="ar", audio_task="transcribe")
    req = AudioRequest(audio_path=tmp_path / "in.wav", language=None,
                       task="transcribe")
    out = tmp_path / "out.json"
    argv, env = asr.build_command(cfg, model, prof, req, out)
    assert argv[0] == str(py)
    assert "--model_path" in argv and str(model) in argv
    assert "--audio" in argv and str(req.audio_path) in argv
    assert "--output" in argv and str(out) in argv
    # Profile language flows through when the request doesn't override it.
    assert "--language" in argv and "ar" in argv
    assert "--task" in argv and "transcribe" in argv
    assert env["PYTHONUTF8"] == "1"


def test_asr_adapter_request_language_overrides_profile(tmp_path: Path):
    from llamanager.engines import asr
    from llamanager.engines._base import AudioRequest
    from llamanager.config import Config, Profile

    py = tmp_path / "python"
    py.write_text("")
    py.chmod(0o755)
    model = tmp_path / "whisper"
    _make_whisper_dir(model)
    cfg = Config(data_dir=tmp_path, asr_python=str(py))
    # "auto" request language disables forcing even if the profile sets one.
    prof = Profile(name="p", audio_language="ar")
    req = AudioRequest(audio_path=tmp_path / "a.wav", language="auto")
    argv, _ = asr.build_command(cfg, model, prof, req, tmp_path / "o.json")
    assert "--language" not in argv


def test_asr_adapter_unconfigured_raises(tmp_path: Path):
    import pytest
    from llamanager.engines import asr
    from llamanager.engines._base import AudioRequest
    from llamanager.config import Config, Profile
    cfg = Config(data_dir=tmp_path, asr_python="")
    req = AudioRequest(audio_path=tmp_path / "a.wav")
    with pytest.raises(RuntimeError, match="asr_python"):
        asr.build_command(cfg, tmp_path / "m", Profile(name="p"), req,
                          tmp_path / "o.json")


def test_asr_parse_progress():
    from llamanager.engines import asr
    assert asr.parse_progress("[asr] chunk 3/10").step == 3
    assert asr.parse_progress("[asr] chunk 3/10").total == 10
    assert asr.parse_progress("loading weights") is None
    assert asr.parse_progress("") is None


def test_asr_schema_keys_are_profile_attrs():
    """profile_schema keys must be real Profile attributes (the UI does
    Profile(**{key: value})), or audio profile creation would crash."""
    from llamanager.engines import asr
    from llamanager.config import Profile
    valid = {f.name for f in __import__("dataclasses").fields(Profile)}
    for f in asr.profile_schema():
        assert f.key in valid, f"schema key {f.key!r} is not a Profile field"


# ---------- audio profile round-trip ----------

def test_audio_profile_roundtrips_through_toml(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, Profile, load_config, save_profile,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    import tomlkit
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))

    prof = Profile(name="quran-ar", audio_language="ar", audio_task="transcribe")
    save_profile(cfg_path, "whisper-large-v3-turbo-ar-quran", "quran-ar", prof)

    reloaded = load_config(cfg_path)
    m = reloaded.get_model("whisper-large-v3-turbo-ar-quran")
    assert m is not None
    p = m.profiles["quran-ar"]
    assert p.audio_language == "ar"
    assert p.audio_task == "transcribe"


def test_asr_streaming_profile_fields_roundtrip(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, Profile, load_config, save_profile)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    import tomlkit
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))
    prof = Profile(name="live", audio_word_timestamps="on",
                   audio_decode_interval_s=0.75, audio_transport="websocket")
    save_profile(cfg_path, "whisper-x", "live", prof)
    p = load_config(cfg_path).get_model("whisper-x").profiles["live"]
    assert p.audio_word_timestamps == "on"
    assert p.audio_decode_interval_s == 0.75
    assert p.audio_transport == "websocket"


def test_asr_service_settings_roundtrip(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, load_config, update_image_config)
    import tomlkit
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    doc = tomlkit.load(cfg_path.open("rb"))
    doc["server"]["data_dir"] = tmp_path.as_posix()
    cfg_path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))
    update_image_config(cfg_path, asr_vram_budget_gb=10.0, asr_coexist=False,
                        asr_idle_timeout_s=90)
    cfg = load_config(cfg_path)
    assert cfg.asr_vram_budget_gb == 10.0
    assert cfg.asr_coexist is False
    assert cfg.asr_idle_timeout_s == 90


def test_asr_decode_interval_env_wins(tmp_path: Path, monkeypatch):
    from llamanager.config import load_config
    monkeypatch.setenv("ASR_DECODE_INTERVAL_S", "0.25")
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.asr_decode_interval_s == 0.25


def test_asr_worker_command_built(tmp_path: Path):
    from llamanager.engines import asr
    from llamanager.config import Config
    py = tmp_path / "python"
    py.write_text(""); py.chmod(0o755)
    model = tmp_path / "whisper"
    _make_whisper_dir(model)
    cfg = Config(data_dir=tmp_path, asr_python=str(py))
    argv, env = asr.build_worker_command(cfg, model, 8123, 4)
    assert argv[0] == str(py)
    assert "_asr_worker.py" in argv[2]
    assert "--port" in argv and "8123" in argv
    assert "--max-concurrent" in argv and "4" in argv
    assert env["PYTHONUTF8"] == "1"


def test_asr_python_config_roundtrip(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, load_config, update_image_config,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    update_image_config(cfg_path, asr_python="/opt/venv/bin/python")
    cfg = load_config(cfg_path)
    assert cfg.asr_python == "/opt/venv/bin/python"


# ---------- queue family routing ----------

def test_infer_task_type_audio(tmp_path: Path):
    from llamanager.config import Config
    from llamanager.queue_mgr import _infer_task_type
    model = tmp_path / "models" / "whisper"
    _make_whisper_dir(model)
    cfg = Config(data_dir=tmp_path)
    # models_dir defaults to data_dir/models, where we put the whisper folder.
    assert _infer_task_type(cfg, "whisper") == "audio"


def _audio_req():
    from llamanager.queue_mgr import QueuedRequest
    return QueuedRequest(request_id="r1", origin=None, priority=0,
                         model_required="whisper", enqueued_at=0.0, seq=0,
                         task_type="audio")


def test_queue_audio_concurrent_by_vram_budget():
    """coexist=on: audio admits up to the budget-derived concurrency and runs
    alongside text; the cap blocks beyond it."""
    from llamanager.config import Config, asr_max_concurrent
    from llamanager.queue_mgr import QueueManager
    cfg = Config(asr_coexist=True, asr_vram_budget_gb=6.0)  # → ~4 concurrent
    cap = asr_max_concurrent(cfg)
    assert cap >= 2
    qm = QueueManager.__new__(QueueManager)
    qm.cfg = cfg
    qm._in_flight_count = {"text": 3, "image": 0, "audio": 0}  # text loaded
    req = _audio_req()
    # Runs alongside text (coexist), up to the cap.
    qm._in_flight_count["audio"] = cap - 1
    assert qm._can_dispatch(req) is True
    qm._in_flight_count["audio"] = cap
    assert qm._can_dispatch(req) is False


def test_queue_audio_coexist_off_is_exclusive_single():
    """coexist=off: cap 1 and mutually exclusive with text."""
    from llamanager.config import Config
    from llamanager.queue_mgr import QueueManager
    cfg = Config(asr_coexist=False, asr_vram_budget_gb=8.0)
    qm = QueueManager.__new__(QueueManager)
    qm.cfg = cfg
    qm._in_flight_count = {"text": 0, "image": 0, "audio": 0}
    req = _audio_req()
    assert qm._can_dispatch(req) is True
    # a second audio is blocked (cap 1 in exclusive mode)
    qm._in_flight_count["audio"] = 1
    assert qm._can_dispatch(req) is False
    # text in flight blocks audio (exclusive)
    qm._in_flight_count = {"text": 1, "image": 0, "audio": 0}
    assert qm._can_dispatch(req) is False


def test_asr_max_concurrent_from_budget():
    from llamanager.config import Config, asr_max_concurrent
    assert asr_max_concurrent(Config(asr_vram_budget_gb=6.0)) == 4   # (6-2)/1
    assert asr_max_concurrent(Config(asr_vram_budget_gb=3.0)) == 1
    assert asr_max_concurrent(Config(asr_vram_budget_gb=0.0)) >= 1   # uncapped


# ---------- installer reuse plan ----------

def test_asr_install_plan_reuses_diffusion_venv():
    from llamanager.engine_installer import ENGINE_PLANS, AMD_WHEEL_ENGINES
    plan = ENGINE_PLANS["asr"]
    assert plan.reuse_from == ("z_image", "hidream")
    assert "torch" in plan.reuse_probe
    # torch is NOT in the extras list — it comes from the reused venv.
    assert "torch" not in plan.packages
    # but the dedicated-venv fallback can still get AMD torch wheels.
    assert "asr" in AMD_WHEEL_ENGINES


# ---------- runner resolve ----------

def test_asr_models_dir_defaults_to_models_dir(tmp_path: Path):
    from llamanager.config import Config
    cfg = Config(data_dir=tmp_path)
    # No override → asr_models_dir tracks models_dir.
    assert cfg.asr_models_dir == cfg.models_dir
    other = tmp_path / "whisper-store"
    cfg.asr_models_dir_override = other
    assert cfg.asr_models_dir == other
    assert cfg.models_dir != other  # LLM models dir is unaffected


def test_asr_models_dir_config_roundtrip(tmp_path: Path):
    from llamanager.config import (
        DEFAULT_CONFIG_TOML, load_config, update_image_config,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    d = tmp_path / "asr-models"
    update_image_config(cfg_path, asr_models_dir=str(d))
    cfg = load_config(cfg_path)
    assert cfg.asr_models_dir_override == d
    assert cfg.asr_models_dir == d
    # Blank clears the override → back to the shared models dir.
    update_image_config(cfg_path, asr_models_dir="")
    cfg2 = load_config(cfg_path)
    assert cfg2.asr_models_dir_override is None


def test_scan_asr_models_finds_whisper_dirs(tmp_path: Path):
    from llamanager.config import Config
    from llamanager.audio_runner import scan_asr_models
    store = tmp_path / "asr-store"
    _make_whisper_dir(store / "whisper-ar")
    (store / "whisper-ar" / "model.safetensors").write_bytes(b"x" * 10)
    # A non-ASR dir in the same folder must be ignored.
    (store / "notes").mkdir(parents=True)
    (store / "notes" / "config.json").write_text('{"model_type":"llama"}')
    cfg = Config(data_dir=tmp_path, asr_models_dir_override=store)
    found = scan_asr_models(cfg)
    assert [m["model_id"] for m in found] == ["whisper-ar"]
    assert found[0]["size_bytes"] > 0


def test_resolve_audio_engine_uses_asr_models_dir(tmp_path: Path):
    from llamanager.config import Config
    from llamanager.audio_runner import resolve_audio_engine
    store = tmp_path / "asr-store"
    _make_whisper_dir(store / "w")
    cfg = Config(data_dir=tmp_path, asr_models_dir_override=store)
    # Model lives only in the dedicated ASR dir, not models_dir.
    assert resolve_audio_engine(cfg, "w") == "asr"


def test_resolve_audio_engine_rejects_non_audio(tmp_path: Path):
    import pytest
    from llamanager.config import Config
    from llamanager.audio_runner import resolve_audio_engine, AudioError
    # A GGUF (text) model id is not audio.
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.gguf").write_bytes(b"")
    cfg = Config(data_dir=tmp_path)
    with pytest.raises(AudioError):
        resolve_audio_engine(cfg, "x.gguf")
