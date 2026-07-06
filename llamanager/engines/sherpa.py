"""sherpa-onnx ASR adapter (streaming transducer).

Serves sherpa-onnx models (Zipformer transducer, Paraformer, CTC — a folder
with ``tokens.txt`` + ``*.onnx`` graphs) through a warm worker
(``_sherpa_worker.py``) running in a small pip venv (``sherpa-onnx`` +
onnxruntime, **no torch**). Its distinguishing feature is **true low-latency
streaming**: an online recognizer consumes audio chunks and emits incremental
hypotheses, so the live-mic path can feed PCM into a stateful stream instead of
re-decoding a rolling window. The worker therefore advertises
``native_streaming`` via ``capabilities()``, which ``asr_stream`` uses to pick
the stateful ``/stream_pcm`` path.

For batch/file transcription the worker still answers the shared loopback
protocol (``/healthz``, ``/transcribe``, ``/transcribe_pcm``) with the same
transcript envelope, so ``audio_runner`` treats it like any other audio engine.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config
from ._base import ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "sherpa"
LABEL = "sherpa-onnx (streaming transducer)"

# sherpa transducers are typically language-specific; the model itself fixes the
# language, so we surface a minimal set. "auto" = let the model decide.
_LANGUAGES = ["auto", "en", "ar", "fr", "de", "es", "zh"]

_STEP_RE = re.compile(r"chunk\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` hold a sherpa-onnx model? Mirrors
    ``config._looks_like_sherpa``: a ``tokens.txt`` + at least one ``*.onnx``."""
    from ..config import _looks_like_sherpa
    return _looks_like_sherpa(model_dir)


def build_worker_command(
    cfg: Config, model_path: Path, port: int, max_concurrent: int,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) to launch the warm sherpa-onnx worker in its venv."""
    if not cfg.sherpa_python:
        raise RuntimeError(
            "image.sherpa_python is not configured — install the sherpa-onnx "
            "engine dependencies on the ASR models page first.")
    python = Path(cfg.sherpa_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"sherpa python not found: {python}")
    worker = Path(__file__).with_name("_sherpa_worker.py")
    if not worker.exists():
        raise RuntimeError(f"sherpa worker missing: {worker}")
    argv = [
        str(python), "-u", str(worker),
        "--model_path", str(model_path),
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
            help="Most sherpa models are single-language; 'auto' uses the model.",
        ),
        ProfileField(
            key="audio_word_timestamps", label="Word timestamps", kind="select",
            default="off", options=["off", "on"],
            help="Emit per-token timings from the transducer alignment.",
        ),
        ProfileField(
            key="audio_decode_interval_s", label="Decode interval (s)",
            kind="float", default=None,
            help="Streaming partial cadence (live WebSocket mode).",
        ),
        ProfileField(
            key="audio_transport", label="Transport", kind="select",
            default="websocket", options=["http", "websocket"],
            help="websocket enables the native low-latency streaming path.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    """sherpa-onnx online recognizers stream natively — the daemon feeds audio
    chunks into a stateful stream rather than re-decoding a window."""
    return {"ref_images_max": 0, "native_streaming": True}


def configured(cfg: Config) -> bool:
    """Ready when a Python interpreter with sherpa-onnx is configured."""
    return bool(cfg.sherpa_python)


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "live": {"audio_language": "auto", "audio_transport": "websocket",
                 "audio_decode_interval_s": 0.4},
    }
