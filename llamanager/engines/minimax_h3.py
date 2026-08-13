"""MiniMax-H3 adapter — joint video + audio generation.

MiniMax-H3 is an omni-modal generator: one 33B-class transformer denoises a
single packed sequence carrying the text conditioning, the keyframe latents,
the video latents and the audio latents together, so a clip arrives with its
soundtrack already in it. Like Z-Image and Wan its reference runtime is
``diffusers``, so we ship a runner (``_minimax_h3_runner.py``) and invoke it
with the operator's configured Python.

Two things separate it from the other video engine (Wan):

* **It is Modular-Diffusers-only.** There is no ``DiffusionPipeline`` half,
  so the runner builds it via ``ModularPipeline.from_pretrained(workflow=…)``.

* **It is guidance-distilled.** No negative prompt, no guidance scale — the
  profile schema deliberately omits both rather than offering knobs the
  model ignores.

Size is the headline constraint: 61.7 GB of transformer plus 62.1 GB of
Qwen3-VL conditioner in bfloat16. The runner sizes each request against the
actual hardware and refuses with the reason instead of thrashing.
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

ENGINE = "minimax_h3"
LABEL = "MiniMax-H3 (video + audio)"

HF_REPO = "MiniMaxAI/MiniMax-H3"

# MiniMax-H3 runs at a fixed 24 fps and generates 5-15 second clips; frames
# snap up to the next 17n+5 the video VAE can decode. 124 frames is upstream's
# own example and lands just under 5.2 s.
NATIVE_FPS = 24
_DEFAULT_NUM_FRAMES = 124
_DEFAULT_STEPS = 50

# The canvas is a 768-pixel short edge by default; both dims must be
# multiples of 32. 960x544 runs ~2.3x faster per step than the trained
# 1344x768, which makes it the practical draft tier.
_DEFAULT_SIZE = "1344x768"
SIZE_BUCKETS = [
    "1344x768", "768x1344",     # the trained canvas
    "1280x768", "768x1280",
    "1024x576", "576x1024",
    "960x544", "544x960",       # ~2.3x faster per step
]

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Quantisation options, smallest-first. The runner probes the chosen option on
# the real backend before loading anything and falls back when kernels are
# missing. Measured on gfx1201/ROCm against an 8192x8192 bf16 linear:
#   bitsandbytes nf4  3.88x smaller, and FASTER than bf16 (0.30 vs 0.35 ms)
#   bitsandbytes int8 2.00x, 0.20 ms
#   torchao int8/fp8  2.00x, but 1.12/2.74 ms (no cpp extensions on ROCm)
#   torchao int4      unavailable on ROCm ("Requires mslk >= 1.0.0")
# So nf4 is the default: it is the only option that fits this model on a
# 32 GB card, and on AMD it is also the fastest.
QUANT_OPTIONS = ("nf4", "int4", "bnb-int8", "int8", "fp8", "none")

# One "memory strategy" selector instead of two overlapping knobs. Maps to
# (--offload, --residency). Only the first two keep every byte on the GPU:
#
#   weights (GiB)      transformer  conditioner   joint peak   split peak
#   bf16                      61.7         62.1        129.3         67.6
#   int8 / fp8                30.9         31.1         67.4         36.5
#   int4                      15.4         15.5         36.5         21.0
#
# So int4 + split is the only combination that fits a 32 GB card, and int8 +
# split needs roughly a 40 GB one.
MEMORY_STRATEGIES: dict[str, tuple[str, str]] = {
    "split": ("none", "split"),   # GPU-only; peak = max(transformer, encoder)
    "joint": ("none", "joint"),   # GPU-only; peak = their sum
    "block": ("block", "split"),  # streams transformer blocks from host RAM
    "leaf":  ("leaf", "split"),   # streams more, incl. the video VAE
}


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` look like a MiniMax-H3 checkpoint?

    The conversion ships a ``modular_model_index.json`` naming every
    component — that file is the marker, and it is what distinguishes a
    Modular Diffusers repo from the ordinary ``model_index.json`` trees the
    other adapters detect.
    """
    if not model_dir.is_dir():
        return False
    idx = model_dir / "modular_model_index.json"
    if not idx.is_file():
        return False
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # Both partitions share every other component, so keying off the
    # MiniMax-specific VAE class is the reliable signal.
    blob = json.dumps(data)
    return "MiniMaxH3" in blob


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        return req.width, req.height
    if profile.image_size and "x" in profile.image_size:
        w, h = profile.image_size.lower().split("x", 1)
        try:
            return int(w), int(h)
        except ValueError:
            pass
    return 1344, 768


def _resolved_steps(profile: Profile, req: ImageRequest) -> int:
    if req.steps is not None:
        return int(req.steps)
    if profile.image_steps is not None:
        return int(profile.image_steps)
    return _DEFAULT_STEPS


def build_command(
    cfg: Config,
    model_path: Path,
    profile: Profile,
    req: ImageRequest,
    out_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for one MiniMax-H3 invocation."""
    python_path = getattr(cfg, "minimax_h3_python", "") or cfg.wan_python
    if not python_path:
        raise RuntimeError(
            "image.minimax_h3_python is not configured — install the engine "
            "on the Diffusion engines page, or point it at a Python with a "
            "diffusers build that ships MiniMax-H3."
        )
    python = Path(python_path).expanduser()
    if not python.exists():
        raise RuntimeError(f"minimax_h3 python not found: {python}")
    runner = Path(__file__).with_name("_minimax_h3_runner.py")
    if not runner.exists():
        raise RuntimeError(f"minimax_h3 runner missing: {runner}")

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    num_frames = (profile.video_num_frames
                  if profile.video_num_frames is not None else _DEFAULT_NUM_FRAMES)
    # MiniMax-H3's clock is fixed at 24 fps; video_fps stays the export rate.
    fps = profile.video_fps if profile.video_fps is not None else NATIVE_FPS

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--model_path", str(model_path),
        "--output", str(out_path),
        "--prompt", req.prompt,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--num-frames", str(num_frames),
        "--fps", str(fps),
        "--repo", HF_REPO,
    ]
    if seed is not None:
        argv += ["--seed", str(seed)]
    # ``image_model_type`` doubles as the precision selector here, and
    # ``image_editing_scheduler`` as the memory strategy — reusing the generic
    # profile columns keeps this engine inside the existing schema.
    quantize = (profile.image_model_type or "int8").strip().lower()
    if quantize in QUANT_OPTIONS:
        argv += ["--quantize", quantize]
    strategy = (profile.image_editing_scheduler or "split").strip().lower()
    offload, residency = MEMORY_STRATEGIES.get(strategy, ("none", "split"))
    argv += ["--offload", offload, "--residency", residency]
    if req.ref_images:
        # First reference is the opening keyframe; a second is the closing one.
        argv += ["--init-image", str(req.ref_images[0])]
        if len(req.ref_images) > 1:
            argv += ["--last-image", str(req.ref_images[1])]

    env: dict[str, str] = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    from ..gpu_detect import rocm_lib_dirs
    rocm_dirs = rocm_lib_dirs()
    if rocm_dirs:
        import os as _os
        prior = _os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = _os.pathsep.join(
            rocm_dirs + ([prior] if prior else []))
        env.setdefault("MIOPEN_FIND_MODE", "FAST")
        cache = cfg.data_dir / "cache" / "miopen"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            env.setdefault("MIOPEN_USER_DB_PATH", str(cache))
            env.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(cache))
        except OSError:
            pass
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
            help="Both dimensions must be multiples of 32. 960x544 runs about "
                 "2.3x faster per step than the trained 1344x768.",
        ),
        ProfileField(
            key="video_num_frames", label="Frames", kind="int",
            default=_DEFAULT_NUM_FRAMES,
            help="Fixed 24fps. Snapped up to the next 17n+5, and the clip must "
                 "land between 5 and 15 seconds (124 frames ≈ 5.2s).",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS,
            help="Counts sigma grid points including the terminal 0, so it "
                 "runs one model evaluation fewer than the number shown.",
        ),
        ProfileField(
            key="image_model_type", label="Weight precision", kind="select",
            default="nf4", options=list(QUANT_OPTIONS),
            help="nf4 (bitsandbytes) is 3.88x smaller — 21.5 GiB peak with "
                 "split residency, the only option that fits a 32 GB card — "
                 "and on ROCm it is faster than bf16. torchao int4 is smaller "
                 "still but CUDA-only. int8/fp8 merely halve the weights "
                 "(36 GiB). The runner probes the backend and falls back.",
        ),
        ProfileField(
            key="image_editing_scheduler", label="Memory strategy", kind="select",
            default="split", options=list(MEMORY_STRATEGIES),
            help="'split' encodes the prompt, frees the conditioner, then "
                 "loads the transformer — peak is max() of the two instead of "
                 "their sum, and everything stays on the GPU. 'joint' loads "
                 "both at once. 'block' and 'leaf' stream weights from system "
                 "RAM, which is slower and keeps the model off the GPU.",
        ),
        ProfileField(
            key="video_fps", label="Export FPS", kind="int",
            default=NATIVE_FPS,
            help="Export rate of the muxed mp4. MiniMax-H3 always generates "
                 "at 24fps; changing this retimes playback.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    """Up to two keyframes (first and last); output is an mp4 with audio."""
    return {
        "output_ext": "mp4",
        "ref_images_max": 2,
        "ref_label": "Keyframes (first, last)",
        "ref_help": "One image starts the clip; a second ends it. Either may "
                    "be given alone. Guidance is baked into the weights, so "
                    "there is no negative prompt or guidance scale.",
    }


def default_profiles() -> dict[str, dict[str, Any]]:
    """Starting profiles, ordered by how much hardware they need.

    ``h3-offload`` is the only one a consumer card can run, and even then the
    weights live in host RAM — roughly 75 GB of it at int8. ``h3-resident``
    is the 80 GB-accelerator recipe and the only fully on-GPU path.
    """
    return {
        # nf4 + split is the recommended path everywhere: 21.5 GiB peak fits a
        # 32 GB card with nothing in system RAM, and bitsandbytes has kernels
        # on both CUDA and ROCm. h3-gpu-int4 is the torchao equivalent, a
        # shade smaller but CUDA-only.
        "h3-gpu-nf4": {
            "image_size": "960x544",
            "image_steps": _DEFAULT_STEPS,
            "video_num_frames": _DEFAULT_NUM_FRAMES,
            "video_fps": NATIVE_FPS,
            "image_model_type": "nf4",
            "image_editing_scheduler": "split",
        },
        "h3-gpu-int4": {
            "image_size": "960x544",
            "image_steps": _DEFAULT_STEPS,
            "video_num_frames": _DEFAULT_NUM_FRAMES,
            "video_fps": NATIVE_FPS,
            "image_model_type": "int4",
            "image_editing_scheduler": "split",
        },
        # ~40 GB and up: halved weights, still entirely on the GPU.
        "h3-gpu-int8": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS,
            "video_num_frames": _DEFAULT_NUM_FRAMES,
            "video_fps": NATIVE_FPS,
            "image_model_type": "int8",
            "image_editing_scheduler": "split",
        },
        # Last resort: streams from system RAM. Slower, and the weights are
        # no longer on the GPU — kept for cards that cannot hold the model.
        "h3-offload": {
            "image_size": "960x544",
            "image_steps": _DEFAULT_STEPS,
            "video_num_frames": _DEFAULT_NUM_FRAMES,
            "video_fps": NATIVE_FPS,
            "image_model_type": "int8",
            "image_editing_scheduler": "block",
        },
        # 80 GB accelerators: no quantisation at all.
        "h3-bf16": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS,
            "video_num_frames": _DEFAULT_NUM_FRAMES,
            "video_fps": NATIVE_FPS,
            "image_model_type": "none",
            "image_editing_scheduler": "split",
        },
    }
