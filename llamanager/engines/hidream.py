"""HiDream-O1-Image adapter (Linux, ROCm).

Wraps the upstream ``inference.py`` script. One-shot: the subprocess
loads the 8-shard 8.80B-param checkpoint, generates one image, and exits.

Reference: docs/hidream.md (also reachable in the operator's notes at
``/media/.../Soulthread/TestAI/docs/hidream.md``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "hidream"
LABEL = "HiDream-O1-Image"

# Resolution buckets — see HiDream's models/utils.py. The smallest is
# 2048x2048; passing anything smaller silently snaps up. We surface the
# buckets in the UI so the operator can pick deliberately.
RESOLUTION_BUCKETS = [
    "2048x2048",
    "2304x1728", "1728x2304",
    "2560x1440", "1440x2560",
    "2496x1664", "1664x2496",
    "3104x1312", "1312x3104",
    "2304x1792", "1792x2304",
]

_DEFAULT_MODEL_TYPE = "dev"   # 28 steps, cfg 0.0, flash scheduler
_DEFAULT_STEPS_DEV = 28
_DEFAULT_STEPS_FULL = 50

# Match step lines emitted by the upstream sampler. The exact format
# depends on tqdm's bar; we accept the common "N/M [elapsed<...]" tail
# as well as "step N/M" prefixes used by some forks.
_STEP_RE = re.compile(r"(?:step\s+)?(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` look like a HiDream-O1-Image checkpoint?"""
    if not model_dir.is_dir():
        return False
    return (
        (model_dir / "tokenizer_config.json").exists()
        and (model_dir / "preprocessor_config.json").exists()
        and any(model_dir.glob("*.safetensors"))
    )


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        return req.width, req.height
    if profile.image_size and "x" in profile.image_size:
        w, h = profile.image_size.lower().split("x", 1)
        try:
            return int(w), int(h)
        except ValueError:
            pass
    return 2048, 2048


def _resolved_steps(profile: Profile, req: ImageRequest, model_type: str) -> int:
    if req.steps is not None:
        return int(req.steps)
    if profile.image_steps is not None:
        return int(profile.image_steps)
    return _DEFAULT_STEPS_FULL if model_type == "full" else _DEFAULT_STEPS_DEV


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one HiDream invocation."""
    if not cfg.hidream_python:
        raise RuntimeError(
            "image.hidream_python is not configured — set it on the Setup page."
        )
    if not cfg.hidream_repo:
        raise RuntimeError(
            "image.hidream_repo is not configured — set it on the Setup page."
        )
    python = Path(cfg.hidream_python).expanduser()
    inference_py = Path(cfg.hidream_repo).expanduser() / "inference.py"
    if not python.exists():
        raise RuntimeError(f"hidream python not found: {python}")
    if not inference_py.exists():
        raise RuntimeError(f"hidream inference.py not found: {inference_py}")

    model_type = (profile.image_model_type or _DEFAULT_MODEL_TYPE).lower()
    if model_type not in ("dev", "full"):
        model_type = _DEFAULT_MODEL_TYPE
    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req, model_type)
    seed = req.seed if req.seed is not None else profile.image_seed

    argv: list[str] = [
        str(python), "-u", str(inference_py),
        "--model_path", str(model_path),
        "--model_type", model_type,
        "--prompt", req.prompt,
        "--output_image", str(out_path),
        "--width", str(width),
        "--height", str(height),
    ]
    # The dev recipe is fixed-step in the upstream script; we still pass
    # --num_inference_steps where supported, and full ignores it.
    if steps is not None:
        argv += ["--num_inference_steps", str(steps)]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if profile.image_guidance is not None and model_type == "full":
        argv += ["--guidance_scale", str(float(profile.image_guidance))]

    # Honour profile.args as raw passthrough (snake_case → --kebab-case).
    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    env: dict[str, str] = {}
    return argv, env


def parse_progress(line: str) -> ProgressEvent | None:
    """Best-effort step extraction. Returns None for lines that don't
    look like step counts."""
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
    # Filter out obviously-not-a-step matches (e.g. dates, byte counts).
    if total <= 0 or total > 5000 or step < 0 or step > total:
        return None
    return ProgressEvent(step=step, total=total)


def profile_schema() -> list[ProfileField]:
    """Fields the Images / Profiles UI should render for this engine."""
    return [
        ProfileField(
            key="image_model_type", label="Recipe", kind="select",
            default="dev", options=["dev", "full"],
            help="dev: 28 steps · full: 50 steps + cfg.",
        ),
        ProfileField(
            key="image_size", label="Resolution", kind="select",
            default="2048x2048", options=RESOLUTION_BUCKETS,
            help="Smallest bucket is 2048×2048 — smaller requests snap up.",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=None, help="Leave blank to use the recipe default.",
        ),
        ProfileField(
            key="image_guidance", label="Guidance scale", kind="float",
            default=None, help="Only effective with the full recipe.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
    ]


def default_profiles() -> dict[str, dict[str, Any]]:
    """Two opinionated profiles auto-created on detection."""
    return {
        "hidream-dev": {
            "image_model_type": "dev",
            "image_size": "2048x2048",
            "image_steps": _DEFAULT_STEPS_DEV,
        },
        "hidream-full": {
            "image_model_type": "full",
            "image_size": "2048x2048",
            "image_steps": _DEFAULT_STEPS_FULL,
            "image_guidance": 5.0,
        },
    }
