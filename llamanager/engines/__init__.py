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

from . import flux2, hidream, z_image

# Public registry. Keys match ``engine_type`` strings used in config.py.
ADAPTERS = {
    "hidream": hidream,
    "flux2": flux2,
    "z_image": z_image,
}


def get(engine: str):
    """Return the adapter module for ``engine`` or raise ``KeyError``."""
    return ADAPTERS[engine]


# Defaults for the per-engine capability map the image UI consumes. An
# adapter declares only what differs by defining ``capabilities()``.
_CAP_DEFAULTS = {
    "ref_images_max": 0,            # 0 = no reference-image support
    "ref_label": "Reference images",
    "ref_help": "",
    "strength": False,             # img2img denoise-strength control
    "keep_original_aspect": False,  # lock output to a single ref's aspect
}


def capabilities(engine: str) -> dict:
    """Return the (defaults-merged) capability map for ``engine``.

    Unknown engines / adapters without a ``capabilities()`` get the
    no-reference-image defaults, so callers can treat every engine
    uniformly."""
    caps = dict(_CAP_DEFAULTS)
    mod = ADAPTERS.get(engine)
    fn = getattr(mod, "capabilities", None) if mod else None
    if fn:
        try:
            caps.update(fn() or {})
        except Exception:  # noqa: BLE001 — a bad adapter shouldn't break the page
            pass
    return caps


__all__ = ["ADAPTERS", "get", "capabilities", "hidream", "flux2", "z_image"]
