"""whisper.cpp / GGML (Vulkan) ASR adapter.

Serves whisper.cpp GGML checkpoints (``ggml-*.bin``) through a small warm
worker (``_whispercpp_worker.py``) that shells out to the ``whisper-cli``
binary built from source with Vulkan (``-DGGML_VULKAN=1``). Unlike the
transformers ``asr`` engine, this needs **no Python venv** — the worker shim is
stdlib-only and runs under llamanager's own interpreter, and all the heavy
lifting (model load, GPU inference) happens in the native binary. That makes it
the reliable GPU path on this AMD R9700 / Intel-iGPU host, where Vulkan works
but ROCm-torch is fragile.

The worker exposes the same loopback protocol as ``_asr_worker.py``
(``/healthz``, ``/transcribe``, ``/transcribe_pcm``) and the same transcript
envelope, so ``audio_runner`` / ``asr_stream`` treat all audio engines
uniformly.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "whispercpp"
LABEL = "whisper.cpp (GGML / Vulkan)"

_LANGUAGES = ["auto", "ar", "en", "fr", "de", "es", "tr", "ur", "id"]
_TASKS = ["transcribe", "translate"]

# whisper-cli prints "[00:00:30.000 --> 00:01:00.000]" per segment; the worker
# re-emits "chunk i/N" progress lines so this mirrors asr.parse_progress.
_STEP_RE = re.compile(r"chunk\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` hold a whisper.cpp GGML model? Mirrors
    ``config._looks_like_whispercpp``: a ``ggml-*.bin`` (or whisper ``.gguf``)."""
    from ..config import _looks_like_whispercpp
    return _looks_like_whispercpp(model_dir)


def _model_file(model_path: Path) -> Path:
    """Resolve the actual GGML checkpoint inside ``model_path`` (a folder). The
    worker also does this; kept here so ``build_worker_command`` can fail fast."""
    if model_path.is_file():
        return model_path
    candidates = sorted(
        p for p in model_path.iterdir()
        if p.is_file() and (
            (p.name.lower().startswith("ggml-") and p.suffix.lower() == ".bin")
            or (p.suffix.lower() == ".gguf" and "whisper" in p.name.lower())
        )
    )
    if not candidates:
        raise RuntimeError(f"no ggml-*.bin whisper model found in {model_path}")
    return candidates[0]


def build_worker_command(
    cfg: Config, model_path: Path, port: int, max_concurrent: int,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) to launch the warm whisper.cpp worker. The shim runs
    under llamanager's own interpreter (stdlib only); the native ``whisper-cli``
    does the inference."""
    if not cfg.whispercpp_binary:
        raise RuntimeError(
            "image.whispercpp_binary is not configured — build whisper.cpp "
            "(Vulkan) on the ASR models page first.")
    binary = Path(cfg.whispercpp_binary).expanduser()
    if not binary.exists():
        raise RuntimeError(f"whisper-cli not found: {binary}")
    worker = Path(__file__).with_name("_whispercpp_worker.py")
    if not worker.exists():
        raise RuntimeError(f"whispercpp worker missing: {worker}")
    model_file = _model_file(model_path)
    argv = [
        sys.executable, "-u", str(worker),
        "--whisper-cli", str(binary),
        "--model", str(model_file),
        "--host", "127.0.0.1", "--port", str(int(port)),
        "--max-concurrent", str(max(1, int(max_concurrent))),
    ]
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return argv, env


def parse_progress(line: str) -> ProgressEvent | None:
    if not line:
        return None
    m = _STEP_RE.search(line)
    if not m:
        return None
    try:
        step, total = int(m.group(1)), int(m.group(2))
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
            default="transcribe", options=_TASKS,
            help="transcribe = same language · translate = into English.",
        ),
        ProfileField(
            key="audio_word_timestamps", label="Word timestamps", kind="select",
            default="off", options=["off", "on"],
            help="Per-word timing via whisper.cpp --max-len 1 token splitting.",
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
    """whisper.cpp consumes audio; live streaming is pseudo-streaming (the
    daemon re-decodes a rolling window), so no native_streaming flag."""
    return {"ref_images_max": 0}


def configured(cfg: Config) -> bool:
    """Ready when a built whisper-cli binary is configured and present."""
    b = cfg.whispercpp_binary
    return bool(b) and Path(b).expanduser().exists()


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "auto": {"audio_language": "auto", "audio_task": "transcribe"},
        "quran-ar": {"audio_language": "ar", "audio_task": "transcribe"},
    }
