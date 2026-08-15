"""MiniMax-H3 via ComfyUI — image → video WITH a synchronised soundtrack.

This is the ComfyUI-backed sibling of ``minimax_h3.py`` (the diffusers one).
Both drive the same model; they differ in what they can load, and on a 32 GB
card that difference decides whether the model runs at all.

WHY BOTH EXIST. ``MiniMaxH3Transformer3DModel`` has no ``from_single_file``
and diffusers has no GGUF loader for it, so the diffusers adapter can only
read the released bfloat16 checkpoint — 61.7 GB of transformer plus a 47.9 GB
Qwen3-VL conditioner — and quantise it on the way to the GPU. Measured on this
box (gfx1201, ROCm 7.2, 60 GB RAM) that cost 5.8 s/tensor and ~50 GB of host
RAM, and never completed a generation. ComfyUI reads the community GGUF quants
directly: Q4_K_M is 18.5 GB for the transformer and 13.6 GB for the text
encoder, already quantised on disk, so the card does no conversion work and
the host does not thrash. The diffusers adapter remains the right choice on a
machine with enough memory to hold the bf16 weights.

AUDIO IS THE POINT. H3 samples one latent that carries both picture and
sound; the workflow decodes it twice, through the video VAE and the audio
VAE, and muxes the two into a single mp4. A silent clip from this engine
means the audio branch did not run.

MODEL CONSTRAINTS (fixed by the architecture, not preferences): 24 fps;
frame counts snap to 17n+5; width and height must be multiples of 32; the
native canvas is 768p.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "minimax_h3_comfy"
LABEL = "MiniMax-H3 (ComfyUI, video + audio)"

# The clip's frame rate is a property of the model, not a display choice.
FPS = 24
# Frame counts the video VAE can decode: 17n + 5.
_FRAME_BASE, _FRAME_OFFSET = 17, 5

_DEFAULT_LENGTH = 124          # 17*7 + 5 -> ~5.2 s at 24 fps
_DEFAULT_SIZE = "1344x768"     # the model's native canvas
_DEFAULT_STEPS = 4             # the Turbo distill's design point
_DEFAULT_SAMPLER = "res_multistep"
_DEFAULT_SCHEDULER = "simple"

# Component filenames. A model directory is assembled from four uploaders, so
# these are the names the download buttons write and the workflow expects.
UNET_FILE = "MiniMax-H3-FL2VA-Q4_K_M.gguf"
CLIP_FILE = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
VIDEO_VAE_FILE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_FILE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA_FILE = (
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")

# Transformer quants, largest-first. The text encoder and VAEs are shared.
QUANT_FILES: dict[str, tuple[str, float]] = {
    "Q4_K_M": (UNET_FILE, 18.50),
    "Q3_K_M": ("MiniMax-H3-FL2VA-Q3_K_M.gguf", 14.51),
}

SIZE_BUCKETS = [
    "1344x768", "768x1344",     # native canvas, landscape / portrait
    "1152x640", "640x1152",
    "960x544", "544x960",
    "832x480", "480x832",
]

# ComfyUI's websocket progress, relayed by the runner as "value/max".
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def snap_length(frames: int) -> int:
    """Snap a frame count up to the next 17n+5 the video VAE can decode.

    Snapping up rather than down keeps a requested duration from silently
    becoming shorter than asked for.
    """
    n = max(0, (int(frames) - _FRAME_OFFSET + _FRAME_BASE - 1) // _FRAME_BASE)
    return n * _FRAME_BASE + _FRAME_OFFSET


def snap_dimension(value: int) -> int:
    """Round a width/height to a multiple of 32 (a patch-size requirement)."""
    return max(32, int(round(value / 32)) * 32)


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` hold a ComfyUI-format MiniMax-H3?

    Keyed on the two files nothing else would have together: an H3 transformer
    in ``diffusion_models/`` and the audio VAE. The audio VAE matters — a tree
    without it can produce pictures but not the soundtrack this engine exists
    for, so it should not be claimed as a working H3.
    """
    if not model_dir.is_dir():
        return False
    unets = model_dir / "diffusion_models"
    if not unets.is_dir():
        return False
    has_unet = any(
        "minimax" in f.name.lower() and "h3" in f.name.lower()
        and f.suffix in (".gguf", ".safetensors")
        for f in unets.iterdir() if f.is_file()
    )
    return has_unet and (model_dir / "vae" / AUDIO_VAE_FILE).is_file()


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        w, h = req.width, req.height
    elif profile.image_size and "x" in profile.image_size:
        try:
            a, b = profile.image_size.lower().split("x", 1)
            w, h = int(a), int(b)
        except ValueError:
            w, h = 1344, 768
    else:
        w, h = 1344, 768
    return snap_dimension(w), snap_dimension(h)


def _resolved_steps(profile: Profile, req: ImageRequest) -> int:
    if req.steps is not None:
        return int(req.steps)
    if profile.image_steps is not None:
        return int(profile.image_steps)
    return _DEFAULT_STEPS


def _unet_for(profile: Profile) -> str:
    """Transformer file for the profile's quant, defaulting to Q4_K_M."""
    quant = (profile.image_model_type or "").strip().upper()
    entry = QUANT_FILES.get(quant)
    return entry[0] if entry else UNET_FILE


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one MiniMax-H3 ComfyUI invocation."""
    from . import comfy_backend as cb

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

    if not req.ref_images:
        raise RuntimeError(
            "MiniMax-H3 (ComfyUI) is an image-to-video model: supply one "
            "reference image to use as the opening frame.")
    if len(req.ref_images) != 1:
        raise RuntimeError(
            "MiniMax-H3 image-to-video accepts exactly one reference image; "
            f"got {len(req.ref_images)}")

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    length = snap_length(profile.video_num_frames or _DEFAULT_LENGTH)
    seed = req.seed if req.seed is not None else profile.image_seed
    unet = _unet_for(profile)

    # Fail before starting a server if a component is missing: "vae/... not
    # found" is a far better message than ComfyUI's combo-validation error.
    required = {"diffusion_models": unet, "text_encoders": CLIP_FILE,
                "vae": VIDEO_VAE_FILE}
    missing = cb.missing_files(model_path, required)
    if not (model_path / "vae" / AUDIO_VAE_FILE).is_file():
        missing.append(f"vae/{AUDIO_VAE_FILE}")
    if missing:
        raise RuntimeError(
            "MiniMax-H3 model directory is incomplete — missing "
            + ", ".join(missing)
            + f". Download the remaining components into {model_path}.")

    use_lora = bool(profile.lora_weights) or profile.lora_weights is None
    lora_name = profile.lora_weights or TURBO_LORA_FILE
    lora_present = (model_path / "loras" / lora_name).is_file()
    if use_lora and not lora_present:
        # Not fatal: without the distill the model still samples, it just
        # needs many more steps. Say so rather than failing the request.
        log.warning("minimax_h3_comfy: LoRA %s not found in %s; sampling "
                    "without it (expect to need ~50 steps)",
                    lora_name, model_path / "loras")
        use_lora = False

    workflow = cb.workflow_path("minimax_h3_i2v_gguf")
    argv: list[str] = [
        str(python), "-u", str(runner),
        "--comfy-repo", str(repo),
        "--model-path", str(model_path),
        "--workflow", str(workflow),
        "--output", str(out_path),
        "--init-image", str(req.ref_images[0]),
        "--set", f"UNET={unet}",
        "--set", f"CLIP={CLIP_FILE}",
        "--set", f"VIDEO_VAE={VIDEO_VAE_FILE}",
        "--set", f"AUDIO_VAE={AUDIO_VAE_FILE}",
        "--set", f"PROMPT={req.prompt}",
        "--set", f"WIDTH={width}",
        "--set", f"HEIGHT={height}",
        "--set", f"LENGTH={length}",
        "--set", f"STEPS={steps}",
        "--set", f"FPS={float(FPS)}",
        "--set", f"SAMPLER={profile.image_editing_scheduler or _DEFAULT_SAMPLER}",
        "--set", f"SCHEDULER={_DEFAULT_SCHEDULER}",
        "--set", f"SEED={int(seed) if seed is not None else 0}",
    ]
    if use_lora:
        strength = (profile.lora_scale
                    if profile.lora_scale is not None else 1.0)
        argv += ["--set", f"LORA={lora_name}",
                 "--set", f"LORA_STRENGTH={float(strength)}"]
    else:
        # The LoRA node is dropped from the graph rather than zeroed: a
        # zero-strength LoRA would still read 1.8 GB off disk and load it.
        argv += ["--set", "LORA=", "--set", "LORA_STRENGTH=0.0",
                 "--bypass", "2:model"]

    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    env = _runner_env(cfg)
    return argv, env


def _runner_env(cfg: Config) -> dict[str, str]:
    """Environment for the ComfyUI child (ROCm library paths and MIOpen)."""
    import os

    env: dict[str, str] = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    from ..gpu_detect import rocm_lib_dirs
    rocm_dirs = rocm_lib_dirs()
    if rocm_dirs:
        prior = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            rocm_dirs + ([prior] if prior else []))
        env.setdefault("MIOPEN_FIND_MODE", "FAST")
        # HIP CUDA-graph replay has crashed on this hardware with models that
        # carry state across steps (see the Qwen3.6 MTP gated-delta-net
        # failure); video sampling is the same shape, so graphs stay off.
        env.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
        cache = cfg.data_dir / "cache" / "miopen"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            env.setdefault("MIOPEN_USER_DB_PATH", str(cache))
            env.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(cache))
        except OSError:
            pass
    return env


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
            help="1344x768 is the model's native canvas. Dimensions are "
                 "rounded to a multiple of 32.",
        ),
        ProfileField(
            key="video_num_frames", label="Frames", kind="int",
            default=_DEFAULT_LENGTH,
            help="At 24 fps: 124 ≈ 5s, 209 ≈ 8.7s, 362 ≈ 15s (the maximum). "
                 "Rounded up to the next 17n+5 the video decoder accepts.",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS,
            help="4 with the Turbo distill LoRA (its design point), 8 for the "
                 "8-step variant, ~50 without a LoRA.",
        ),
        ProfileField(
            key="image_model_type", label="Transformer quant", kind="select",
            default="Q4_K_M", options=sorted(QUANT_FILES),
            help="Q4_K_M (18.5 GB) is the quality/VRAM sweet spot on a 32 GB "
                 "card. Q3_K_M (14.5 GB) trades detail for headroom.",
        ),
        ProfileField(
            key="lora_weights", label="Turbo LoRA", kind="text",
            default=TURBO_LORA_FILE,
            help="Filename in the model's loras/ folder. Clear it to sample "
                 "without a distill (raise steps to ~50).",
        ),
        ProfileField(
            key="lora_scale", label="LoRA strength", kind="float",
            default=1.0, help="1.0 is the distill's trained strength.",
        ),
        ProfileField(
            key="image_editing_scheduler", label="Sampler", kind="select",
            default=_DEFAULT_SAMPLER,
            options=["res_multistep", "euler", "euler_ancestral", "dpmpp_2m"],
            help="res_multistep is what the reference workflow uses.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    """One reference image is required; the output is an mp4 with audio."""
    return {
        "output_ext": "mp4",
        "ref_images_max": 1,
        "ref_images_required": True,
        "ref_label": "Opening frame (image→video)",
        "ref_help": "One image to animate as the first frame of the clip. "
                    "The clip is generated with a matching soundtrack.",
    }


def default_profiles() -> dict[str, dict[str, Any]]:
    """Starting profiles, ordered best-first.

    ``h3-turbo-4step`` is the intended everyday recipe: the lightx2v Turbo
    distill collapses 50 sampling steps to 4, which is what makes a video
    model practical on one consumer card.

    ``h3-full-50step`` drops the LoRA. It is the reference quality bar and is
    roughly an order of magnitude slower; keep it for final renders.
    """
    return {
        "h3-turbo-4step": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": 4,
            "video_num_frames": _DEFAULT_LENGTH,
            "video_fps": FPS,
            "image_model_type": "Q4_K_M",
            "lora_weights": TURBO_LORA_FILE,
            "lora_scale": 1.0,
        },
        "h3-turbo-4step-720p": {
            "image_size": "1152x640",
            "image_steps": 4,
            "video_num_frames": _DEFAULT_LENGTH,
            "video_fps": FPS,
            "image_model_type": "Q4_K_M",
            "lora_weights": TURBO_LORA_FILE,
            "lora_scale": 1.0,
        },
        "h3-turbo-8step": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": 8,
            "video_num_frames": _DEFAULT_LENGTH,
            "video_fps": FPS,
            "image_model_type": "Q4_K_M",
            "lora_weights": TURBO_LORA_FILE,
            "lora_scale": 1.0,
        },
        "h3-full-50step": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": 50,
            "video_num_frames": _DEFAULT_LENGTH,
            "video_fps": FPS,
            "image_model_type": "Q4_K_M",
            "lora_weights": "",
        },
    }
