"""Z-Image (Tongyi-MAI) adapter — also handles Z-Anime fine-tunes.

Z-Image is a Single-Stream Diffusion Transformer served via Hugging Face
``diffusers``. Unlike HiDream (where we wrap the upstream's hand-rolled
inference.py) or FLUX 2 (which uses the sd.cpp CLI), Z-Image's reference
runtime IS the diffusers library — so we ship a small runner script
(``_z_image_runner.py``) inside this package and invoke it with the
user's configured Python interpreter.

The user only needs to point ``image.z_image_python`` at a Python
executable from an environment that has ``diffusers`` / ``transformers``
/ ``accelerate`` / ``torch`` installed. There's no separate "source
folder" to clone — the runner lives with llamanager.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "z_image"
LABEL = "Z-Image (Tongyi-MAI)"

# Inference defaults from the Tongyi-MAI README.
_DEFAULT_STEPS_FAST = 28
_DEFAULT_STEPS_QUALITY = 50
_DEFAULT_GUIDANCE = 4.0
_DEFAULT_SIZE = "1024x1024"

# A few aspect-ratio buckets that work well per the README ("any aspect
# ratio between 512 and 2048"). Listing common ones gives the operator a
# starting point without locking them out of typing custom dims.
SIZE_BUCKETS = [
    "1024x1024",
    "1280x720", "720x1280",
    "1280x1280",
    "1536x1024", "1024x1536",
    "2048x1152", "1152x2048",
    "2048x2048",
]

# tqdm progress lines from diffusers look like "100%|██████| 50/50 [00:42<00:00, ...]"
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` look like a Z-Image (or Z-Anime) checkpoint?

    Z-Image and Z-Anime both ship a Diffusers ``model_index.json`` that
    names ``ZImagePipeline`` as the pipeline class. We only consider a
    folder a match when that field is present — generic Diffusers
    pipelines (FLUX, SD3, etc.) have the same on-disk shape but a
    different ``_class_name`` and shouldn't get this engine assigned.
    """
    if not model_dir.is_dir():
        return False
    mi = model_dir / "model_index.json"
    if not mi.is_file():
        return False
    try:
        data = json.loads(mi.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    cls = (data.get("_class_name") or "").strip()
    return cls == "ZImagePipeline"


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        return req.width, req.height
    if profile.image_size and "x" in profile.image_size:
        w, h = profile.image_size.lower().split("x", 1)
        try:
            return int(w), int(h)
        except ValueError:
            pass
    return 1024, 1024


def _resolved_steps(profile: Profile, req: ImageRequest) -> int:
    if req.steps is not None:
        return int(req.steps)
    if profile.image_steps is not None:
        return int(profile.image_steps)
    return _DEFAULT_STEPS_QUALITY


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one Z-Image invocation."""
    if not cfg.z_image_python:
        raise RuntimeError(
            "image.z_image_python is not configured — set it on the "
            "Diffusion engines page."
        )
    python = Path(cfg.z_image_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"z_image python not found: {python}")
    runner = Path(__file__).with_name("_z_image_runner.py")
    if not runner.exists():
        raise RuntimeError(f"z_image runner missing: {runner}")

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    guidance = profile.image_guidance if profile.image_guidance is not None else _DEFAULT_GUIDANCE

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--model_path", str(model_path),
        "--output", str(out_path),
        "--prompt", req.prompt,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(int(steps)),
        "--guidance", str(float(guidance)),
    ]
    if seed is not None:
        argv += ["--seed", str(int(seed))]

    # Honour profile.args as raw passthrough (snake_case → --kebab-case),
    # so power users can flip --device or --dtype overrides without code
    # changes. Reference-image inputs are not supported by Z-Image's
    # base pipeline today; ignore ``req.ref_images`` silently rather
    # than failing the whole request.
    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    env: dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    return argv, env


def parse_progress(line: str) -> ProgressEvent | None:
    if not line:
        return None
    m = _STEP_RE.search(line)
    if not m:
        return None
    try:
        step = int(m.group(1))
        total = int(m.group(2))
    except ValueError:
        return None
    if total <= 0 or total > 5000 or step < 0 or step > total:
        return None
    return ProgressEvent(step=step, total=total)


def profile_schema() -> list[ProfileField]:
    return [
        ProfileField(
            key="image_size", label="Resolution", kind="select",
            default=_DEFAULT_SIZE, options=SIZE_BUCKETS,
            help="Any aspect ratio in 512–2048; longer dims = slower.",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS_QUALITY,
            help="28 (fast) to 50 (quality). Distill variants run at 4 or 8.",
        ),
        ProfileField(
            key="image_guidance", label="Guidance scale", kind="float",
            default=_DEFAULT_GUIDANCE,
            help="3.0–5.0 per the upstream recommendation.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
    ]


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "z-image-fast": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS_FAST,
            "image_guidance": _DEFAULT_GUIDANCE,
        },
        "z-image-quality": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS_QUALITY,
            "image_guidance": _DEFAULT_GUIDANCE,
        },
    }
