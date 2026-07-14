"""Wan 2.2 (Alibaba Tongyi) adapter — text-and-image → video.

Wan is a diffusion-transformer video model served via Hugging Face
``diffusers`` (``WanPipeline`` / ``WanImageToVideoPipeline`` +
``AutoencoderKLWan``). Like Z-Image, its reference runtime IS the
diffusers library, so we ship a small runner script
(``_wan_runner.py``) inside this package and invoke it with the
user's configured Python interpreter (``image.wan_python``).

The flagship target is ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``: a single
dense 5B model that handles BOTH text-to-video and image-to-video.
A text-only request runs the text-to-video pipeline; supplying one
reference image switches the runner to image-to-video (the image
becomes the first frame).

Video engines produce an ``.mp4`` rather than a single PNG — the
``output_ext`` capability tells ``ImageTaskRunner`` to allocate a
``.mp4`` output path and the gallery to render a ``<video>`` player.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

ENGINE = "wan"
LABEL = "Wan 2.2 (video)"

# Recommended settings from the Wan2.2-TI2V-5B model card (720p).
_DEFAULT_STEPS = 50
_DEFAULT_GUIDANCE = 5.0
_DEFAULT_NUM_FRAMES = 121      # ~5s at 24fps
_DEFAULT_FPS = 24
_DEFAULT_SIZE = "1280x704"

# Resolution buckets the 5B model handles well (720p and a lighter draft
# tier). The operator can still type custom dims via the profile.
SIZE_BUCKETS = [
    "1280x704", "704x1280",     # 720p landscape / portrait
    "960x544", "544x960",       # lighter draft tier
    "832x480", "480x832",
]

# diffusers/tqdm progress lines look like "100%|██████| 50/50 [00:42<00:00]".
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def detect(model_dir: Path) -> bool:
    """Does ``model_dir`` look like a Wan diffusers checkpoint?

    Wan ships a Diffusers ``model_index.json`` naming ``WanPipeline`` (or
    ``WanImageToVideoPipeline``). We also accept a checkpoint whose
    ``transformer/config.json`` declares a ``Wan`` architecture, so a
    tree assembled component-by-component still matches.
    """
    if not model_dir.is_dir():
        return False
    import json
    mi = model_dir / "model_index.json"
    if mi.is_file():
        try:
            data = json.loads(mi.read_text(encoding="utf-8"))
            cls = (data.get("_class_name") or "").strip()
            if cls in ("WanPipeline", "WanImageToVideoPipeline"):
                return True
        except (OSError, json.JSONDecodeError):
            pass
    tcfg = model_dir / "transformer" / "config.json"
    if tcfg.is_file():
        try:
            data = json.loads(tcfg.read_text(encoding="utf-8"))
            cls = (data.get("_class_name") or "")
            arch = " ".join(data.get("architectures") or [])
            if "Wan" in cls or "Wan" in arch:
                return True
        except (OSError, json.JSONDecodeError):
            pass
    return False


def _resolved_size(profile: Profile, req: ImageRequest) -> tuple[int, int]:
    if req.width and req.height:
        return req.width, req.height
    if profile.image_size and "x" in profile.image_size:
        w, h = profile.image_size.lower().split("x", 1)
        try:
            return int(w), int(h)
        except ValueError:
            pass
    return 1280, 704


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
    """Return (argv, env) for one Wan invocation."""
    if not cfg.wan_python:
        raise RuntimeError(
            "image.wan_python is not configured — set it on the "
            "Diffusion engines page."
        )
    python = Path(cfg.wan_python).expanduser()
    if not python.exists():
        raise RuntimeError(f"wan python not found: {python}")
    runner = Path(__file__).with_name("_wan_runner.py")
    if not runner.exists():
        raise RuntimeError(f"wan runner missing: {runner}")

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    guidance = (profile.image_guidance
                if profile.image_guidance is not None else _DEFAULT_GUIDANCE)
    num_frames = (profile.video_num_frames
                  if profile.video_num_frames is not None else _DEFAULT_NUM_FRAMES)
    fps = profile.video_fps if profile.video_fps is not None else _DEFAULT_FPS

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--model_path", str(model_path),
        "--output", str(out_path),
        "--prompt", req.prompt,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(int(steps)),
        "--guidance", str(float(guidance)),
        "--num-frames", str(int(num_frames)),
        "--fps", str(int(fps)),
    ]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if profile.image_negative_prompt:
        argv += ["--negative_prompt", profile.image_negative_prompt]
    if profile.image_editing_scheduler:
        argv += ["--vae-device", profile.image_editing_scheduler]
    if profile.image_model_type:
        argv += ["--dtype", profile.image_model_type]

    # One reference image → image-to-video (the image is the first frame).
    if req.ref_images:
        if len(req.ref_images) != 1:
            raise RuntimeError(
                "wan image-to-video accepts exactly one reference image; "
                f"got {len(req.ref_images)}"
            )
        argv += ["--init-image", str(req.ref_images[0])]

    # Honour profile.args as raw passthrough (snake_case → --kebab-case).
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
    # AMD ROCm: the lightweight torch wheels dlopen system ROCm libs that
    # live under /opt/rocm/core-*/lib, which isn't on the default linker
    # path. Prepend the ROCm lib dirs so ``import torch`` succeeds and sees
    # the GPU. No-op on non-AMD hosts (rocm_lib_dirs() returns []).
    from ..gpu_detect import rocm_lib_dirs
    rocm_dirs = rocm_lib_dirs()
    if rocm_dirs:
        import os as _os
        prior = _os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = _os.pathsep.join(
            rocm_dirs + ([prior] if prior else []))
        # MIOpen safety on AMD: heuristic ("FAST") kernel finder + a
        # writable cache, matching the Z-Image runner. The VAE conv decode
        # is the known gfx1201 hazard; the runner defaults it to CPU.
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
            help="720p (1280x704) is the native quality bucket; smaller = faster.",
        ),
        ProfileField(
            key="video_num_frames", label="Frames", kind="int",
            default=_DEFAULT_NUM_FRAMES,
            help="121 frames ≈ 5s at 24fps. Fewer frames = much faster / less VRAM.",
        ),
        ProfileField(
            key="video_fps", label="Playback FPS", kind="int",
            default=_DEFAULT_FPS,
            help="Frames-per-second of the exported mp4 (native 24).",
        ),
        ProfileField(
            key="image_steps", label="Steps", kind="int",
            default=_DEFAULT_STEPS,
            help="50 is the recommended quality setting; 20–30 for drafts.",
        ),
        ProfileField(
            key="image_guidance", label="Guidance scale", kind="float",
            default=_DEFAULT_GUIDANCE,
            help="5.0 per the upstream recommendation.",
        ),
        ProfileField(
            key="image_seed", label="Seed", kind="int",
            default=None, help="Leave blank for a fresh seed each run.",
        ),
        ProfileField(
            key="image_editing_scheduler", label="VAE device", kind="select",
            default="auto", options=["auto", "cuda", "cpu", "mps"],
            help="Advanced: decode on CPU/cuda/mps. Auto keeps ROCm VAE decode on CPU for stability.",
        ),
        ProfileField(
            key="image_negative_prompt", label="Negative prompt", kind="text",
            default="", help="Optional negative prompt forwarded to the pipeline.",
        ),
        ProfileField(
            key="image_model_type", label="Torch dtype", kind="select",
            default="", options=["", "bfloat16", "float16", "float32"],
            help="Advanced override. Blank auto-selects bfloat16 on CUDA, float16 on MPS, float32 on CPU.",
        ),
    ]


def capabilities() -> dict[str, Any]:
    """One reference image → image-to-video; output is an mp4."""
    return {
        "output_ext": "mp4",
        "ref_images_max": 1,
        "ref_label": "First frame (image→video)",
        "ref_help": "One image to animate as the opening frame of the clip.",
    }


def default_profiles() -> dict[str, dict[str, Any]]:
    return {
        "wan-720p": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS,
            "image_guidance": _DEFAULT_GUIDANCE,
            "video_num_frames": _DEFAULT_NUM_FRAMES,
            "video_fps": _DEFAULT_FPS,
        },
        "wan-draft": {
            "image_size": "960x544",
            "image_steps": 30,
            "image_guidance": _DEFAULT_GUIDANCE,
            "video_num_frames": 49,
            "video_fps": _DEFAULT_FPS,
        },
    }
