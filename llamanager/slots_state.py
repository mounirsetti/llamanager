"""Atomic JSON persistence for additional LLM slots.

``runtime.json`` (see ``runtime_state.py``) continues to describe slot 0
— the legacy single instance. This sibling file describes slots 1..N
when the multi-slot beta is enabled. A separate file keeps the legacy
shape pristine for users who never enable the feature.

Schema (``~/.llamanager/slots.json``)::

    {
      "version": 1,
      "slots": [
        {
          "id": 1,
          "port": 7202,
          "model_id": "org/repo/Q4_K_M.gguf" | null,
          "profile":  "fast" | null,
          "args":     { ... }
        },
        ...
      ]
    }

When ``model_id`` is null the slot is idle. The file is rewritten on
every add / remove / load / unload via the atomic-rename pattern from
``runtime_state.save``.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CURRENT_VERSION = 1


@dataclass
class SlotEntry:
    """Persisted state for one non-default slot."""
    id: int
    port: int
    model_id: str | None = None
    profile: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlotsManifest:
    """The whole slots.json document."""
    version: int = CURRENT_VERSION
    slots: list[SlotEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "slots": [s.to_dict() for s in self.slots],
        }

    def by_id(self, slot_id: int) -> SlotEntry | None:
        for s in self.slots:
            if s.id == slot_id:
                return s
        return None


def load(path: Path) -> SlotsManifest:
    """Read ``slots.json`` or return an empty manifest if missing /
    unreadable. Never raises.
    """
    if not path.exists():
        return SlotsManifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SlotsManifest()
    raw_slots = data.get("slots") or []
    slots: list[SlotEntry] = []
    for raw in raw_slots:
        if not isinstance(raw, dict):
            continue
        try:
            slots.append(SlotEntry(
                id=int(raw["id"]),
                port=int(raw["port"]),
                model_id=raw.get("model_id") or None,
                profile=raw.get("profile") or None,
                args=dict(raw.get("args") or {}),
            ))
        except (KeyError, TypeError, ValueError):
            # Defensive: a corrupt entry shouldn't take down the daemon.
            continue
    return SlotsManifest(
        version=int(data.get("version") or CURRENT_VERSION),
        slots=slots,
    )


def save(path: Path, manifest: SlotsManifest) -> None:
    """Atomic-rename writer. Same pattern as runtime_state.save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".slots-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        finally:
            raise
