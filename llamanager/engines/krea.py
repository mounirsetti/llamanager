"""Krea 2 Turbo adapter.

Supports both the original Diffusers repo (``krea/Krea-2-Turbo``) and the
community GGUF repo (``vantagewithai/Krea-2-Turbo-GGUF``), which contains
one transformer file per quantization. GGUF runs borrow the remaining
pipeline components from the original repo.
It intentionally reuses ``image.z_image_python`` so operators can share the
same torch/diffusers venv across Z-Image, Z-Anime, and Krea.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "krea"
LABEL = "Krea 2 Turbo"

KREA_GGUF_REPO = "vantagewithai/Krea-2-Turbo-GGUF"
KREA_ORIGINAL_REPO = "krea/Krea-2-Turbo"
BASE_REPO = KREA_ORIGINAL_REPO

QUANT_FILES = [
    "krea2_turbo-Q2_K.gguf",
    "krea2_turbo-Q3_K_S.gguf",
    "krea2_turbo-Q3_K_M.gguf",
    "krea2_turbo-Q4_0.gguf",
    "krea2_turbo-Q4_1.gguf",
    "krea2_turbo-Q4_K_S.gguf",
    "krea2_turbo-Q4_K_M.gguf",
    "krea2_turbo-Q5_0.gguf",
    "krea2_turbo-Q5_1.gguf",
    "krea2_turbo-Q5_K_S.gguf",
    "krea2_turbo-Q5_K_M.gguf",
    "krea2_turbo-Q6_K.gguf",
    "krea2_turbo-Q8_0.gguf",
]

QUANT_SIZE_GB = {
    "krea2_turbo-Q2_K.gguf": 4.89,
    "krea2_turbo-Q3_K_S.gguf": 6.01,
    "krea2_turbo-Q3_K_M.gguf": 6.01,
    "krea2_turbo-Q4_0.gguf": 7.49,
    "krea2_turbo-Q4_1.gguf": 8.18,
    "krea2_turbo-Q4_K_S.gguf": 7.49,
    "krea2_turbo-Q4_K_M.gguf": 7.49,
    "krea2_turbo-Q5_0.gguf": 8.87,
    "krea2_turbo-Q5_1.gguf": 9.67,
    "krea2_turbo-Q5_K_S.gguf": 8.87,
    "krea2_turbo-Q5_K_M.gguf": 8.87,
    "krea2_turbo-Q6_K.gguf": 10.58,
    "krea2_turbo-Q8_0.gguf": 13.71,
}

_DEFAULT_SIZE = "1024x1024"
_DEFAULT_STEPS_FAST = 4
_DEFAULT_STEPS_QUALITY = 8
_DEFAULT_GUIDANCE = 1.0
_DEFAULT_QUANT = "krea2_turbo-Q6_K.gguf"
_ORIGINAL_SENTINEL = "original"

SIZE_BUCKETS = [
    "768x768",
    "1024x1024",
    "1152x896", "896x1152",
    "1280x720", "720x1280",
    "1344x768", "768x1344",
    "1536x1024", "1024x1536",
]

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _find_quant_files(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []
    return sorted(
        p for p in model_dir.iterdir()
        if p.is_file()
        and p.name.lower().startswith("krea2_turbo-")
        and p.suffix.lower() == ".gguf"
    )


def _is_original_layout(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    mi = model_dir / "model_index.json"
    if not mi.is_file():
        return False
    try:
        import json
        data = json.loads(mi.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (data.get("_class_name") or "").strip() == "Krea2Pipeline"


def detect(model_dir: Path) -> bool:
    return _is_original_layout(model_dir) or bool(_find_quant_files(model_dir))


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        return req.width, req.height
    if profile.image_size and "x" in profile.image_size:
        try:
            w, h = profile.image_size.lower().split("x", 1)
            return int(w), int(h)
        except ValueError:
            pass
    return 1024, 1024


def _resolved_steps(profile: Profile, req: ImageRequest) -> int:
    if req.steps is not None:
        return int(req.steps)
    if profile.image_steps is not None:
        return int(profile.image_steps)
    return _DEFAULT_STEPS_FAST


def _selected_quant(model_dir: Path, profile: Profile) -> Path:
    quants = _find_quant_files(model_dir)
    if not quants:
        raise RuntimeError(f"Krea model dir has no krea2_turbo-*.gguf files: {model_dir}")
    wanted = profile.image_model_type or ""
    if wanted:
        p = model_dir / wanted
        if p.is_file():
            return p
        raise RuntimeError(
            f"Krea quant {wanted!r} is not installed in {model_dir}. "
            "Download it from the Diffusion engines page or pick an installed quant."
        )
    preferred = model_dir / _DEFAULT_QUANT
    if preferred.is_file():
        return preferred
    return quants[-1]


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    if not cfg.z_image_python:
        raise RuntimeError(
            "image.z_image_python is not configured. Install the shared "
            "Z-Image/Krea dependencies on the Diffusion engines page."
        )
    python = Path(cfg.z_image_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"Krea python not found: {python}")
    runner = Path(__file__).with_name("_krea_runner.py")
    if not runner.exists():
        raise RuntimeError(f"Krea runner missing: {runner}")

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    guidance = (
        float(profile.image_guidance)
        if profile.image_guidance is not None
        else _DEFAULT_GUIDANCE
    )
    original = _is_original_layout(model_path)
    quant = None if original else _selected_quant(model_path, profile)

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
    if quant is not None:
        argv += ["--gguf", str(quant)]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if profile.image_negative_prompt:
        argv += ["--negative_prompt", profile.image_negative_prompt]
    if profile.image_editing_scheduler:
        argv += ["--device", profile.image_editing_scheduler]
    if profile.image_strength is not None:
        argv += ["--true-cfg", str(float(profile.image_strength))]

    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    return argv, {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


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
            key="image_model_type", label="Weights", kind="select",
            default=_DEFAULT_QUANT,
            options=[_ORIGINAL_SENTINEL, *QUANT_FILES],
            help="Original uses krea/Krea-2-Turbo; GGUF options use the selected quant file.",
        ),
        ProfileField(
            key="image_size", label="Resolution", kind="select",
            default=_DEFAULT_SIZE, options=SIZE_BUCKETS,
            help="1024² is the balanced default; larger buckets need more VRAM.",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS_FAST,
            help="Krea 2 Turbo is a fast model: 4 for draft, 8 for quality.",
        ),
        ProfileField(
            key="image_guidance", label="Guidance scale", kind="float",
            default=_DEFAULT_GUIDANCE,
            help="Default 1.0. Raise carefully; Turbo models usually need little CFG.",
        ),
        ProfileField(
            key="image_strength", label="True CFG", kind="float",
            default=None,
            help="Advanced Qwen-Image true_cfg_scale override; blank uses diffusers default.",
        ),
        ProfileField(
            key="image_negative_prompt", label="Negative prompt", kind="text",
            default="", help="Optional negative prompt.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
        ProfileField(
            key="image_editing_scheduler", label="Device", kind="select",
            default="", options=["", "cuda", "cpu", "mps"],
            help="Advanced override. Blank auto-selects CUDA, MPS, then CPU.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    return {"ref_images_max": 0}


def default_profiles(model_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    if model_dir is not None and _is_original_layout(model_dir):
        return {
            "krea-original": {
                "image_model_type": _ORIGINAL_SENTINEL,
                "image_size": _DEFAULT_SIZE,
                "image_steps": _DEFAULT_STEPS_QUALITY,
                "image_guidance": 0.0,
            },
        }
    return {
        "krea-fast-q6": {
            "image_model_type": _DEFAULT_QUANT,
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS_FAST,
            "image_guidance": _DEFAULT_GUIDANCE,
        },
        "krea-quality-q8": {
            "image_model_type": "krea2_turbo-Q8_0.gguf",
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS_QUALITY,
            "image_guidance": _DEFAULT_GUIDANCE,
        },
    }
