"""Shared types used by every image-engine adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageRequest:
    """One image-generation request, resolved and ready to dispatch.

    The adapter consumes this object to produce a subprocess argv. Most
    fields are engine-agnostic; engine-specific knobs live on
    ``profile.args`` (raw passthrough) or the typed ``profile.image_*``
    fields on ``Profile``.

    Reference images, when present, are absolute paths on disk that the
    API layer has already validated and persisted (see
    ``api_v1._stage_ref_images``). The runner owns the lifecycle of the
    parent directory and removes it after the run completes.
    """
    prompt: str
    width: int
    height: int
    steps: int | None
    seed: int | None
    n: int
    # Reference images (paths on disk). Empty list means a pure text-to-image
    # request. One path triggers editing semantics on HiDream and img2img on
    # Flux2; multiple paths are HiDream-only (composition / multi-subject).
    ref_images: list[Path] = field(default_factory=list)
    # When a single reference image is provided, preserve its aspect ratio
    # for the output (HiDream: resizes ref to max 2048 on the long side and
    # uses those dimensions; ignored by Flux2).
    keep_original_aspect: bool = False
    # Optional layout-box JSON forwarded verbatim to HiDream's
    # --layout_bboxes flag. Ignored by other engines.
    layout_bboxes: str | None = None
    # Flux2 img2img denoise strength (0.0 = keep init exactly,
    # 1.0 = full regeneration). Ignored by HiDream — its editing recipe is
    # scheduler-controlled, not strength-controlled.
    strength: float | None = None


@dataclass
class AudioRequest:
    """One speech-to-text request, resolved and ready to dispatch.

    The adapter consumes this object to produce a subprocess argv. The audio
    file is an absolute path on disk that the API layer has already staged
    (see ``api_v1`` transcription handler); the runner owns the lifecycle of
    its parent directory and removes it after the run completes.
    """
    audio_path: Path
    # ISO language hint (e.g. ``"ar"``) or ``None`` to let the model decide.
    language: str | None = None
    # ``"transcribe"`` (same language) or ``"translate"`` (to English).
    task: str = "transcribe"
    # When True, return word-level output ({w,t0,t1,p}) + audio_ms in the
    # ``{type:"transcript", …}`` envelope. Heavier (cross-attentions).
    word_timestamps: bool = False


@dataclass
class ProgressEvent:
    """One progress tick parsed from an adapter's stderr/stdout.

    The runner forwards ``step`` / ``total`` to ``runtime_state`` so the
    dashboard can show "step 14/28". Both fields are optional; an event
    with neither is still useful (it marks "still alive").
    """
    step: int | None = None
    total: int | None = None
    note: str | None = None


@dataclass
class ProfileField:
    """One UI field for the engine-aware profile editor."""
    key: str                            # matches a Profile attribute
    label: str
    kind: str                           # "text" | "int" | "float" | "select"
    default: Any = None
    options: list[str] | None = None    # for kind="select"
    help: str = ""                      # short caption shown under the field
