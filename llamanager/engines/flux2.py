"""FLUX 2 Dev / stable-diffusion.cpp adapter.

Wraps the upstream ``sd-cli`` binary (Vulkan or CUDA build). One-shot:
the subprocess loads the diffusion model + VAE + text-encoder, writes
one PNG, and exits.

The FLUX 2 stack uses three model files in one directory:
  * ``flux2-dev-Q*.gguf``         — the diffusion transformer (~26 GB on Q6_K)
  * ``ae.safetensors``            — the VAE (~95 MB)
  * ``Mistral-Small-3.2-*.gguf``  — the text encoder (~16 GB on Q5_K_M)

Reference: docs/flux2.md.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "flux2"
LABEL = "FLUX 2 Dev (sd.cpp)"

_DEFAULT_STEPS_FAST = 8
_DEFAULT_STEPS_QUALITY = 28
_DEFAULT_CFG = 1.0
_DEFAULT_SAMPLER = "euler"

# sd.cpp emits progress lines like "  3/28  [ 18.15s/it]" — match the
# leading "N/M" portion regardless of whitespace.
_STEP_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def _find_files(model_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Locate (diffusion.gguf, ae.safetensors, text_encoder.gguf) inside
    ``model_dir``. Returns None for any file not present."""
    if not model_dir.is_dir():
        return None, None, None
    diffusion: Path | None = None
    vae: Path | None = None
    text_encoder: Path | None = None
    for p in sorted(model_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".gguf"):
            if "flux" in name and diffusion is None:
                diffusion = p
            elif diffusion is not None and text_encoder is None:
                # Heuristic: the second GGUF is the text encoder.
                text_encoder = p
            elif text_encoder is None:
                text_encoder = p
        elif name.endswith(".safetensors") and (
            name == "ae.safetensors"
            or name.endswith("vae.safetensors")
            or name.startswith("ae")
        ):
            if vae is None:
                vae = p
    return diffusion, vae, text_encoder


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` look like a FLUX 2 / sd.cpp checkpoint folder?"""
    diff, vae, _te = _find_files(model_dir)
    return diff is not None and vae is not None


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


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one sd-cli invocation."""
    if not cfg.flux2_sd_cli:
        raise RuntimeError(
            "image.flux2_sd_cli is not configured — set it on the Setup page."
        )
    sd_cli = Path(cfg.flux2_sd_cli).expanduser()
    if not sd_cli.exists():
        raise RuntimeError(f"sd-cli binary not found: {sd_cli}")

    diff, vae, text_encoder = _find_files(model_path)
    if not diff or not vae:
        raise RuntimeError(
            f"FLUX 2 model dir missing required files: {model_path} "
            f"(need a flux*.gguf + ae*.safetensors)"
        )

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    cfg_scale = float(profile.image_guidance) if profile.image_guidance is not None else _DEFAULT_CFG

    argv: list[str] = [
        str(sd_cli),
        "--diffusion-model", str(diff),
        "--vae", str(vae),
        "-p", req.prompt,
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", _DEFAULT_SAMPLER,
        "--steps", str(steps),
        "-W", str(width),
        "-H", str(height),
        "-o", str(out_path),
        "--diffusion-fa",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "-v",
    ]
    if text_encoder is not None:
        argv += ["--llm", str(text_encoder)]
    if seed is not None:
        argv += ["-s", str(int(seed))]

    # Reference image: sd-cli's --init-img (img2img). FLUX 2 Dev is not a
    # Flux Kontext model, so the --ref-image (-r) flag does not apply —
    # we use the classic init-image path. Strength defaults to sd.cpp's
    # built-in 0.75 unless the request or profile overrides it.
    if req.ref_images:
        if len(req.ref_images) != 1:
            raise RuntimeError(
                f"flux2 supports at most one reference image (img2img); "
                f"got {len(req.ref_images)}"
            )
        argv += ["-i", str(req.ref_images[0])]
        strength = req.strength
        if strength is None and profile.image_strength is not None:
            strength = float(profile.image_strength)
        if strength is not None:
            argv += ["--strength", f"{float(strength):.4f}"]

    # Profile.args is raw passthrough (snake_case → --kebab-case).
    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    env: dict[str, str] = {}
    if cfg.flux2_device_index is not None:
        env["GGML_VK_VISIBLE_DEVICES"] = str(int(cfg.flux2_device_index))
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
            default="1024x1024",
            options=["768x768", "1024x1024", "1152x896", "896x1152",
                     "1216x832", "832x1216", "1344x768", "768x1344"],
            help="FLUX 2 trains on multiple aspect ratios; 1024² is a balanced default.",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS_FAST,
            help="8 steps for fast iteration · 20-28 for quality.",
        ),
        ProfileField(
            key="image_guidance", label="CFG scale", kind="float",
            default=_DEFAULT_CFG,
            help="FLUX 2 is flow-matching; values above 1.0 typically degrade output.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
    ]


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "flux2-fast": {
            "image_size": "1024x1024",
            "image_steps": _DEFAULT_STEPS_FAST,
            "image_guidance": _DEFAULT_CFG,
        },
        "flux2-quality": {
            "image_size": "1024x1024",
            "image_steps": _DEFAULT_STEPS_QUALITY,
            "image_guidance": _DEFAULT_CFG,
        },
    }
