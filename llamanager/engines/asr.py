"""ASR (Whisper / speech-to-text) adapter.

Whisper-class checkpoints in Hugging Face ``transformers`` format
(``WhisperForConditionalGeneration`` + safetensors) are served via a small
runner script (``_asr_runner.py``) that ships inside this package, invoked
with the user's configured Python interpreter — exactly like the Z-Image
adapter. The interpreter only needs ``torch`` + ``transformers`` installed;
the engine installer typically points ``image.asr_python`` at the shared
z_image venv rather than rebuilding torch.

Audio is decoded to 16 kHz mono in the runner via the system ``ffmpeg``, so
no heavy audio Python deps are pulled into the shared venv.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import AudioRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "asr"
LABEL = "Whisper (transformers)"

# Common Whisper language codes surfaced in the profile editor. "auto" lets
# the model detect the language; this is not an exhaustive list (the runner
# accepts any code the tokenizer knows).
_LANGUAGES = ["auto", "ar", "en", "fr", "de", "es", "tr", "ur", "id"]
_TASKS = ["transcribe", "translate"]
_DEFAULT_TASK = "transcribe"

# The runner emits "chunk i/N" while walking the 30 s windows of a long file.
_STEP_RE = re.compile(r"chunk\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` look like a Whisper checkpoint?

    Mirrors ``config._looks_like_asr``: a Hugging Face ``config.json`` whose
    ``model_type`` is ``whisper`` (or whose ``architectures`` names
    ``WhisperForConditionalGeneration``).
    """
    if not model_dir.is_dir():
        return False
    cfg_file = model_dir / "config.json"
    if not cfg_file.is_file():
        return False
    try:
        import json
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (data.get("model_type") or "").strip().lower() == "whisper":
        return True
    archs = data.get("architectures") or []
    return any("WhisperForConditionalGeneration" in str(a) for a in archs)


def _resolved_language(profile: Profile, req: AudioRequest) -> str | None:
    """Resolve the language hint, or None to let the model auto-detect."""
    lang = req.language if req.language is not None else (profile.audio_language or "")
    lang = (lang or "").strip().lower()
    if not lang or lang == "auto":
        return None
    return lang


def _resolved_task(profile: Profile, req: AudioRequest) -> str:
    task = (req.task or profile.audio_task or _DEFAULT_TASK).strip().lower()
    return task if task in _TASKS else _DEFAULT_TASK


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: AudioRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one Whisper transcription."""
    if not cfg.asr_python:
        raise RuntimeError(
            "image.asr_python is not configured — install the ASR engine "
            "dependencies on the engines page (it reuses the diffusion venv)."
        )
    python = Path(cfg.asr_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"asr python not found: {python}")
    runner = Path(__file__).with_name("_asr_runner.py")
    if not runner.exists():
        raise RuntimeError(f"asr runner missing: {runner}")

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--model_path", str(model_path),
        "--audio", str(req.audio_path),
        "--output", str(out_path),
        "--task", _resolved_task(profile, req),
    ]
    lang = _resolved_language(profile, req)
    if lang:
        argv += ["--language", lang]

    # Honour profile.args as raw passthrough (snake_case → --kebab-case) for
    # power-user overrides (e.g. --device cpu, --beam-size 5).
    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    env: dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    # AMD ROCm: the lightweight torch wheels dlopen system ROCm libs that live
    # under /opt/rocm/core-*/lib, which isn't on the default linker path.
    # Prepend the ROCm lib dirs so ``import torch`` succeeds and sees the GPU.
    # No-op on non-AMD hosts (rocm_lib_dirs() returns []). GPU *ordering* is
    # left to the user's llamanager device settings — we don't pick a device.
    from ..gpu_detect import rocm_lib_dirs
    rocm_dirs = rocm_lib_dirs()
    if rocm_dirs:
        prior = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            rocm_dirs + ([prior] if prior else []))
    return argv, env


def _rocm_env() -> dict[str, str]:
    """Base env for a Python worker/runner: UTF-8 + the system ROCm libs on
    LD_LIBRARY_PATH so torch imports and sees the GPU (no-op off AMD)."""
    env: dict[str, str] = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    from ..gpu_detect import rocm_lib_dirs
    rocm_dirs = rocm_lib_dirs()
    if rocm_dirs:
        prior = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            rocm_dirs + ([prior] if prior else []))
    return env


def build_worker_command(
    cfg: Config, model_path: Path, port: int, max_concurrent: int,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) to launch the persistent warm ASR worker
    (``_asr_worker.py``) for ``model_path`` on ``port``. Managed by
    ``audio_runner.AudioTaskRunner``."""
    if not cfg.asr_python:
        raise RuntimeError(
            "image.asr_python is not configured — install the ASR engine "
            "dependencies on the ASR models page.")
    python = Path(cfg.asr_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"asr python not found: {python}")
    worker = Path(__file__).with_name("_asr_worker.py")
    if not worker.exists():
        raise RuntimeError(f"asr worker missing: {worker}")
    argv = [
        str(python), "-u", str(worker),
        "--model_path", str(model_path),
        "--host", "127.0.0.1", "--port", str(int(port)),
        "--max-concurrent", str(max(1, int(max_concurrent))),
    ]
    return argv, _rocm_env()


def parse_progress(line: str) -> ProgressEvent | None:
    if not line:
        return None
    m = _STEP_RE.search(line)
    if not m:
        return None
    try:
        step = int(m.group(1))
        total = int(m.group(2))
    except ValueError:
        return None
    if total <= 0 or total > 100000 or step < 0 or step > total:
        return None
    return ProgressEvent(step=step, total=total)


def profile_schema() -> list[ProfileField]:
    return [
        ProfileField(
            key="audio_language", label="Language", kind="select",
            default="auto", options=_LANGUAGES,
            help="ISO code to force, or 'auto' to let the model detect it.",
        ),
        ProfileField(
            key="audio_task", label="Task", kind="select",
            default=_DEFAULT_TASK, options=_TASKS,
            help="transcribe = same language · translate = into English.",
        ),
        ProfileField(
            key="audio_word_timestamps", label="Word timestamps", kind="select",
            default="off", options=["off", "on"],
            help="Return per-word timing + confidence ({w,t0,t1,p}). Heavier.",
        ),
        ProfileField(
            key="audio_decode_interval_s", label="Decode interval (s)",
            kind="float", default=None,
            help="Streaming partial cadence (live WebSocket mode).",
        ),
        ProfileField(
            key="audio_transport", label="Transport", kind="select",
            default="http", options=["http", "websocket"],
            help="Preferred transport; websocket enables live streaming.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    """ASR consumes audio, not reference images."""
    return {"ref_images_max": 0}


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "quran-ar": {"audio_language": "ar", "audio_task": "transcribe"},
        "auto": {"audio_language": "auto", "audio_task": "transcribe"},
    }
