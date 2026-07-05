"""Ideogram 4 adapter using the official ideogram-oss/ideogram4 package."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "ideogram4"
LABEL = "Ideogram 4"

_DEFAULT_SIZE = "1024x1024"
_DEFAULT_QUANT = "fp8"
_DEFAULT_PRESET = "V4_QUALITY_48"

SIZE_BUCKETS = [
    "1024x1024",
    "1280x768", "768x1280",
    "1536x1024", "1024x1536",
    "2048x2048",
]
PRESETS = ["V4_FAST_12", "V4_BALANCED_20", "V4_QUALITY_48"]
QUANTIZATIONS = ["fp8", "nf4"]

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def detect(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    mi = model_dir / "model_index.json"
    if mi.is_file():
        try:
            data = json.loads(mi.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if (data.get("_class_name") or "").strip() == "Ideogram4Pipeline":
            return True
    comfy_markers = (
        model_dir / "diffusion_models" / "ideogram4_fp8_scaled.safetensors",
        model_dir / "diffusion_models" / "ideogram4_unconditional_fp8_scaled.safetensors",
        model_dir / "text_encoders" / "qwen3vl_8b_fp8_scaled.safetensors",
        model_dir / "vae" / "ae.safetensors",
    )
    return any(p.is_file() for p in comfy_markers)


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


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    if not cfg.ideogram4_python:
        raise RuntimeError(
            "image.ideogram4_python is not configured. Install Ideogram 4 "
            "dependencies on the Diffusion engines page."
        )
    python = Path(cfg.ideogram4_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"Ideogram 4 python not found: {python}")
    runner = Path(__file__).with_name("_ideogram4_runner.py")
    if not runner.exists():
        raise RuntimeError(f"Ideogram 4 runner missing: {runner}")

    width, height = _resolved_size(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    quant = profile.image_model_type or _DEFAULT_QUANT
    if quant not in QUANTIZATIONS:
        quant = _DEFAULT_QUANT
    preset = profile.image_editing_scheduler or _DEFAULT_PRESET
    if preset not in PRESETS:
        preset = _DEFAULT_PRESET

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--weights-repo", str(model_path),
        "--output", str(out_path),
        "--prompt", req.prompt,
        "--width", str(width),
        "--height", str(height),
        "--quantization", quant,
        "--sampler-preset", preset,
    ]
    if seed is not None:
        argv += ["--seed", str(int(seed))]

    args = dict(profile.args or {})
    magic_prompt = args.pop("magic_prompt", None)
    if isinstance(magic_prompt, bool):
        argv.append("--magic-prompt" if magic_prompt else "--no-magic-prompt")

    # Profile.args remains the escape hatch for official runner flags:
    # magic_prompt_key/model, hive_text_key, hive_visual_key,
    # warn_on_caption_issues, device, dtype, etc.
    for k, v in args.items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    from ..gpu_detect import rocm_lib_dirs
    rocm_dirs = rocm_lib_dirs()
    if rocm_dirs:
        import os as _os
        prior = _os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = _os.pathsep.join(
            rocm_dirs + ([prior] if prior else []))
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
            key="image_model_type", label="Quantization", kind="select",
            default=_DEFAULT_QUANT, options=QUANTIZATIONS,
            help="Use fp8 on AMD/R9700. nf4 is CUDA-oriented per Ideogram.",
        ),
        ProfileField(
            key="image_size", label="Resolution", kind="select",
            default=_DEFAULT_SIZE, options=SIZE_BUCKETS,
            help="Ideogram supports multiples of 16 up to 2048.",
        ),
        ProfileField(
            key="image_editing_scheduler", label="Sampler preset", kind="select",
            default=_DEFAULT_PRESET, options=PRESETS,
            help="Quality preset bundles steps, CFG schedule, mu, and std.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=0, help="Leave blank for the runner default.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    return {"ref_images_max": 0}


def default_profiles(model_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    # Profile defaults are intentionally AMD-friendly. Users on NVIDIA can
    # clone/switch to nf4 once their environment is CUDA-backed.
    return {
        "ideogram4-fp8-quality": {
            "image_model_type": "fp8",
            "image_size": _DEFAULT_SIZE,
            "image_editing_scheduler": _DEFAULT_PRESET,
            "image_seed": 0,
            "args": {"magic_prompt": True},
        },
        "ideogram4-fp8-no-magic": {
            "image_model_type": "fp8",
            "image_size": _DEFAULT_SIZE,
            "image_editing_scheduler": _DEFAULT_PRESET,
            "image_seed": 0,
            "args": {"magic_prompt": False},
        },
    }
