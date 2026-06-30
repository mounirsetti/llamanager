"""Atomic JSON file describing the current run.

Spec §4.7: runtime.json holds (PID, current_model, current_profile,
started_at). Used to recover state if llamanager itself restarts.

The ``image`` sub-state tracks transient image-engine generations. Image
engines are one-shot CLIs, so there's no persistent ``running`` state —
``status`` is ``idle`` between requests and ``generating`` during one.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageRuntimeState:
    """Transient state for the active image task.

    Single-task (mutual exclusion within the image family). Fields are
    reset to defaults when no task is running.
    """
    status: str = "idle"              # idle|generating|failed
    engine: str | None = None         # "hidream" | "flux2"
    model_id: str | None = None
    profile: str | None = None
    request_id: str | None = None
    step: int | None = None
    total_steps: int | None = None
    started_at: float | None = None
    last_event_at: float | None = None


@dataclass
class AudioRuntimeState:
    """Transient state for the active audio (ASR / Whisper) task.

    Single-task (mutual exclusion within the audio family), one-shot like the
    image family. Fields reset to defaults when no task is running.
    """
    status: str = "idle"              # idle|transcribing|failed
    engine: str | None = None         # "asr"
    model_id: str | None = None
    profile: str | None = None
    request_id: str | None = None
    step: int | None = None
    total_steps: int | None = None
    started_at: float | None = None
    last_event_at: float | None = None


@dataclass
class RuntimeState:
    state: str = "stopped"            # stopped|starting|running|swapping|crashed|degraded
    pid: int | None = None
    current_model: str | None = None
    current_profile: str | None = None
    current_args: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    last_event_at: float | None = None
    image: ImageRuntimeState = field(default_factory=ImageRuntimeState)
    audio: AudioRuntimeState = field(default_factory=AudioRuntimeState)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load(path: Path) -> RuntimeState:
    if not path.exists():
        return RuntimeState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return RuntimeState()
    current_model = data.get("current_model")
    current_profile = data.get("current_profile")
    # Defensive: a profile without a model is meaningless under the new
    # parent/child invariant. Drop the dangling profile rather than crashing
    # later when something tries to validate the pair. The next start/swap
    # will repopulate from a fresh resolve_spec call.
    if current_profile and not current_model:
        current_profile = None
    image_raw = data.get("image") or {}
    image = ImageRuntimeState(
        status=image_raw.get("status", "idle"),
        engine=image_raw.get("engine"),
        model_id=image_raw.get("model_id"),
        profile=image_raw.get("profile"),
        request_id=image_raw.get("request_id"),
        step=image_raw.get("step"),
        total_steps=image_raw.get("total_steps"),
        started_at=image_raw.get("started_at"),
        last_event_at=image_raw.get("last_event_at"),
    )
    audio_raw = data.get("audio") or {}
    audio = AudioRuntimeState(
        status=audio_raw.get("status", "idle"),
        engine=audio_raw.get("engine"),
        model_id=audio_raw.get("model_id"),
        profile=audio_raw.get("profile"),
        request_id=audio_raw.get("request_id"),
        step=audio_raw.get("step"),
        total_steps=audio_raw.get("total_steps"),
        started_at=audio_raw.get("started_at"),
        last_event_at=audio_raw.get("last_event_at"),
    )
    return RuntimeState(
        state=data.get("state", "stopped"),
        pid=data.get("pid"),
        current_model=current_model,
        current_profile=current_profile,
        current_args=data.get("current_args", {}) or {},
        started_at=data.get("started_at"),
        last_event_at=data.get("last_event_at"),
        image=image,
        audio=audio,
    )


def save(path: Path, state: RuntimeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rt-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        finally:
            raise
