"""Image-engine adapters.

Each adapter describes one image generation engine as a small, declarative
module: how to detect its on-disk layout, how to build a subprocess
invocation, how to surface its progress to the UI, and which profile
fields are meaningful.

Adapters are intentionally *not* classes — keeping them as modules with
top-level functions makes adding a third engine a single-file change with
no inheritance to chase. Two concrete adapters ship today: ``hidream``
and ``flux2``.
"""
from __future__ import annotations

from typing import Protocol

from . import flux2, hidream

# Public registry. Keys match ``engine_type`` strings used in config.py.
ADAPTERS = {
    "hidream": hidream,
    "flux2": flux2,
}


def get(engine: str):
    """Return the adapter module for ``engine`` or raise ``KeyError``."""
    return ADAPTERS[engine]


__all__ = ["ADAPTERS", "get", "hidream", "flux2"]
