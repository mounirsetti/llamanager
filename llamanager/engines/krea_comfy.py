"""Krea 2 Turbo via ComfyUI — text-to-image from GGUF weights.

The ComfyUI-backed sibling of ``krea.py`` (the diffusers one). Same model,
different loader, and the loader is the point: Krea 2 Turbo's community GGUF
quants are unusable from diffusers, which has no single-file or GGUF path for
this architecture, so the diffusers adapter can only run the full
24.5 GB bf16 checkpoint. The Q6_K quant is 9.9 GB and loads directly here,
which leaves a 32 GB card enough headroom to hold the whole pipeline resident
instead of streaming weights.

Krea 2 is guidance-distilled: cfg is 1.0 and the negative branch is a zeroed
copy of the positive conditioning (what the reference workflow does), so there
is no negative prompt and no guidance scale to tune. Its conditioner is a
Qwen3-VL 4B of its own — loader type ``krea2``, not ``qwen_image`` — shipped
as fp8_scaled safetensors rather than GGUF.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "krea_comfy"
LABEL = "Krea 2 Turbo (ComfyUI)"

_DEFAULT_STEPS = 8             # the Turbo distill's design point
_DEFAULT_CFG = 1.0             # guidance-distilled: raising this degrades output
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"
_DEFAULT_SIZE = "1024x1024"

CLIP_FILE = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"

# Transformer quants. Q6_K is the default: it is the largest quant that still
# leaves comfortable headroom next to the conditioner on a 32 GB card.
QUANT_FILES: dict[str, tuple[str, float]] = {
    "Q8_0":   ("krea2_turbo-Q8_0.gguf", 12.76),
    "Q6_K":   ("krea2_turbo-Q6_K.gguf", 9.86),
    "Q5_K_M": ("krea2_turbo-Q5_K_M.gguf", 8.26),
    "Q4_K_M": ("krea2_turbo-Q4_K_M.gguf", 6.97),
    "Q3_K_M": ("krea2_turbo-Q3_K_M.gguf", 5.60),
}
DEFAULT_QUANT = "Q6_K"

SIZE_BUCKETS = [
    "1024x1024", "1152x896", "896x1152",
    "1216x832", "832x1216", "1344x768", "768x1344",
    "1536x640", "640x1536",
]

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` hold a ComfyUI-format Krea 2 Turbo?

    Keyed on a Krea transformer plus the matching conditioner: a bare GGUF
    with no text encoder cannot generate anything, so it should not be
    claimed as a working model.
    """
    if not model_dir.is_dir():
        return False
    unets = model_dir / "diffusion_models"
    if not unets.is_dir():
        return False
    has_unet = any(
        "krea" in f.name.lower() and f.suffix in (".gguf", ".safetensors")
        for f in unets.iterdir() if f.is_file()
    )
    tes = model_dir / "text_encoders"
    has_te = tes.is_dir() and any(
        f.suffix == ".safetensors" for f in tes.iterdir() if f.is_file())
    return has_unet and has_te


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        return req.width, req.height
    if profile.image_size and "x" in profile.image_size:
        try:
            a, b = profile.image_size.lower().split("x", 1)
            return int(a), int(b)
        except ValueError:
            pass
    return 1024, 1024


def _resolved_steps(profile: Profile, req: ImageRequest) -> int:
    if req.steps is not None:
        return int(req.steps)
    if profile.image_steps is not None:
        return int(profile.image_steps)
    return _DEFAULT_STEPS


def _unet_for(profile: Profile) -> str:
    quant = (profile.image_model_type or "").strip().upper()
    entry = QUANT_FILES.get(quant) or QUANT_FILES[DEFAULT_QUANT]
    return entry[0]


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one Krea 2 Turbo ComfyUI invocation."""
    from . import comfy_backend as cb
    from .minimax_h3_comfy import _runner_env

    if not cfg.comfyui_python or not cfg.comfyui_repo:
        raise RuntimeError(
            "ComfyUI is not installed — install the 'ComfyUI' engine on the "
            "Diffusion engines page. It provides the shared backend this "
            "model runs on.")
    python = Path(cfg.comfyui_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"comfyui python not found: {python}")
    repo = Path(cfg.comfyui_repo).expanduser()
    if not (repo / "main.py").is_file():
        raise RuntimeError(f"comfyui repo has no main.py: {repo}")
    runner = Path(__file__).with_name("_comfy_runner.py")
    if not runner.exists():
        raise RuntimeError(f"comfy runner missing: {runner}")
    if req.ref_images:
        raise RuntimeError(
            "Krea 2 Turbo (ComfyUI) is text-to-image only in this build; "
            "use the diffusers Krea engine for img2img.")

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    unet = _unet_for(profile)
    cfg_scale = (profile.image_guidance
                 if profile.image_guidance is not None else _DEFAULT_CFG)

    missing = cb.missing_files(model_path, {
        "diffusion_models": unet, "text_encoders": CLIP_FILE, "vae": VAE_FILE})
    if missing:
        raise RuntimeError(
            "Krea 2 Turbo model directory is incomplete — missing "
            + ", ".join(missing)
            + f". Download the remaining components into {model_path}.")

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--comfy-repo", str(repo),
        "--model-path", str(model_path),
        "--workflow", str(cb.workflow_path("krea2_t2i_gguf")),
        "--output", str(out_path),
        "--set", f"UNET={unet}",
        "--set", f"CLIP={CLIP_FILE}",
        "--set", f"VAE={VAE_FILE}",
        # --set-str, not --set: a prompt of "2024" is valid JSON and would
        # otherwise reach the graph as a number.
        "--set-str", f"PROMPT={req.prompt}",
        "--set", f"WIDTH={width}",
        "--set", f"HEIGHT={height}",
        "--set", f"STEPS={steps}",
        "--set", f"CFG={float(cfg_scale)}",
        "--set", f"SAMPLER={profile.image_editing_scheduler or _DEFAULT_SAMPLER}",
        "--set", f"SCHEDULER={_DEFAULT_SCHEDULER}",
        "--set", f"SEED={int(seed) if seed is not None else 0}",
    ]

    lora_name = profile.image_lora_weights or ""
    if lora_name and (model_path / "loras" / lora_name).is_file():
        strength = (profile.image_lora_scale
                    if profile.image_lora_scale is not None else 1.0)
        argv += ["--set", f"LORA={lora_name}",
                 "--set", f"LORA_STRENGTH={float(strength)}"]
    else:
        if lora_name:
            log.warning("krea_comfy: LoRA %s not found in %s; sampling "
                        "without it", lora_name, model_path / "loras")
        # Dropped rather than zeroed so no LoRA file is read at all.
        argv += ["--set", "LORA=", "--set", "LORA_STRENGTH=0.0",
                 "--bypass", "2:model"]

    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    return argv, _runner_env(cfg)


def parse_progress(line: str) -> ProgressEvent | None:
    if not line:
        return None
    m = _STEP_RE.search(line)
    if not m:
        return None
    try:
        step, total = int(m.group(1)), int(m.group(2))
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
            help="1024x1024 is the model's native canvas.",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS,
            help="8 is the Turbo distill's design point; 4 for quick drafts.",
        ),
        ProfileField(
            key="image_model_type", label="Transformer quant", kind="select",
            default=DEFAULT_QUANT, options=list(QUANT_FILES),
            help="Q6_K (9.9 GB) keeps the whole pipeline resident on a 32 GB "
                 "card. Q8_0 is closer to bf16; Q4_K_M frees the most memory.",
        ),
        ProfileField(
            key="image_guidance", label="CFG", kind="float",
            default=_DEFAULT_CFG,
            help="Leave at 1.0 — Krea 2 Turbo is guidance-distilled, so "
                 "raising this degrades the image rather than sharpening it.",
        ),
        ProfileField(
            key="image_lora_weights", label="LoRA", kind="text",
            default="",
            help="Optional filename in the model's loras/ folder "
                 "(e.g. krea2_darkbrush.safetensors).",
        ),
        ProfileField(
            key="image_lora_scale", label="LoRA strength", kind="float",
            default=1.0, help="0.8 is a good starting point for style LoRAs.",
        ),
        ProfileField(
            key="image_editing_scheduler", label="Sampler", kind="select",
            default=_DEFAULT_SAMPLER,
            options=["euler", "res_multistep", "dpmpp_2m", "euler_ancestral"],
            help="euler is what the reference workflow uses.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    """Text-to-image only in this build; img2img stays on the diffusers path."""
    return {"output_ext": "png", "ref_images_max": 0}


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "kreac-best": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS,
            "image_guidance": _DEFAULT_CFG,
            "image_model_type": "Q6_K",
        },
        "kreac-draft": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": 4,
            "image_guidance": _DEFAULT_CFG,
            "image_model_type": "Q4_K_M",
        },
        "kreac-wide": {
            "image_size": "1344x768",
            "image_steps": _DEFAULT_STEPS,
            "image_guidance": _DEFAULT_CFG,
            "image_model_type": "Q6_K",
        },
    }
