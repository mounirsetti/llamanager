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

from . import (asr, flux2, hidream, ideogram4, krea, minimax_h3, sherpa, wan,
               whispercpp, z_image)

# Public registry. Keys match ``engine_type`` strings used in config.py.
ADAPTERS = {
    "hidream": hidream,
    "flux2": flux2,
    "z_image": z_image,
    "krea": krea,
    "ideogram4": ideogram4,
    "wan": wan,
    "minimax_h3": minimax_h3,
    "asr": asr,
    "whispercpp": whispercpp,
    "sherpa": sherpa,
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
    "output_ext": "png",           # produced file extension (video → "mp4")
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


def default_profiles(engine: str, model_dir=None) -> dict:
    """Built-in starting profiles for ``engine``, checkpoint-aware.

    Some adapters key their defaults off the on-disk layout — Krea returns a
    different set for the original Diffusers checkpoint than for the GGUF
    one, and seeding an original checkpoint with GGUF profiles produces
    profiles that fail at load time with "GGUF quants are not loadable".
    Pass ``model_dir`` and it reaches the adapters that accept it; adapters
    that don't take it are called unchanged.

    Returns ``{}`` rather than raising when an adapter has no defaults, so
    callers can treat every engine uniformly.
    """
    import inspect
    mod = ADAPTERS.get(engine)
    fn = getattr(mod, "default_profiles", None) if mod else None
    if fn is None:
        return {}
    try:
        takes_dir = "model_dir" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes_dir = False
    try:
        return fn(model_dir=model_dir) if (takes_dir and model_dir is not None) else fn()
    except Exception:  # noqa: BLE001 — a bad adapter shouldn't break the page
        return {}


__all__ = [
    "ADAPTERS", "get", "capabilities", "default_profiles",
    "hidream", "flux2", "z_image", "krea", "ideogram4", "wan", "minimax_h3", "asr",
    "whispercpp", "sherpa",
]
