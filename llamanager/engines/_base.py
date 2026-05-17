"""Shared types used by every image-engine adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ImageRequest:
    """One image-generation request, resolved and ready to dispatch.

    The adapter consumes this object to produce a subprocess argv. None
    of the fields are engine-specific — engine-specific knobs live on
    ``profile.args`` (raw passthrough) or the typed ``profile.image_*``
    fields on ``Profile``.
    """
    prompt: str
    width: int
    height: int
    steps: int | None
    seed: int | None
    n: int


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
