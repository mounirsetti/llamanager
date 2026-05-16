"""Atomic JSON file describing the current run.

Spec §4.7: runtime.json holds (PID, current_model, current_profile,
started_at). Used to recover state if llamanager itself restarts.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuntimeState:
    state: str = "stopped"            # stopped|starting|running|swapping|crashed|degraded
    pid: int | None = None
    current_model: str | None = None
    current_profile: str | None = None
    current_args: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    last_event_at: float | None = None

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
    return RuntimeState(
        state=data.get("state", "stopped"),
        pid=data.get("pid"),
        current_model=current_model,
        current_profile=current_profile,
        current_args=data.get("current_args", {}) or {},
        started_at=data.get("started_at"),
        last_event_at=data.get("last_event_at"),
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
