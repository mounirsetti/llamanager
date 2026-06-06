"""HiDream-O1-Image adapter (Linux, ROCm).

Wraps the upstream ``inference.py`` script. One-shot: the subprocess
loads the 8-shard 8.80B-param checkpoint, generates one image, and exits.

Reference: docs/hidream.md (also reachable in the operator's notes at
``/media/.../Soulthread/TestAI/docs/hidream.md``).
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import ImageRequest, ProfileField, ProgressEvent

log = logging.getLogger(__name__)

# Cache for `_supports_steps_flag` keyed by (python_path, inference_py_path,
# inference_py_mtime) so we don't pay the ~1-3 s `python inference.py --help`
# probe on every image request. Invalidates automatically when the upstream
# script is edited (e.g. by our install-time patcher or a manual git pull).
_HELP_PROBE_CACHE: dict[tuple[str, str, float], bool] = {}

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


def _supports_steps_flag(python: Path, inference_py: Path) -> bool:
    """Does the local ``inference.py`` accept ``--num_inference_steps``?

    Probes by running ``python inference.py --help`` once and scanning
    the output. The installer applies a patch to add the flag when it
    clones hidream-source, but operators with hand-rolled checkouts
    won't have it — and passing the flag to an unpatched script makes
    argparse reject the whole invocation with rc=2 before the model
    even loads. Cheaper to ask the script what it understands than to
    require the patch to have run.
    """
    try:
        mtime = inference_py.stat().st_mtime
    except OSError:
        return False
    key = (str(python), str(inference_py), mtime)
    if key in _HELP_PROBE_CACHE:
        return _HELP_PROBE_CACHE[key]
    try:
        r = subprocess.run(
            [str(python), str(inference_py), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        text = (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        _HELP_PROBE_CACHE[key] = False
        return False
    supported = ("--num_inference_steps" in text
                 or "--num-inference-steps" in text)
    _HELP_PROBE_CACHE[key] = supported
    if not supported:
        log.info(
            "hidream: inference.py at %s does not advertise "
            "--num_inference_steps; profile/request step counts will be "
            "ignored. Re-run the engine install to apply the patch, or "
            "patch the file manually.",
            inference_py,
        )
    return supported


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
    # Upstream's dev path is hardwired to 28 timesteps via DEFAULT_TIMESTEPS
    # and ignores --num_inference_steps. The full path can accept the flag,
    # but only if our install-time patcher (engine_installer._patch_hidream
    # _inference_steps) has run — vanilla hidream-source rejects the flag
    # and exits 2 before loading the model. Probe the help output so we
    # never emit a flag the script doesn't understand.
    if (model_type == "full" and steps is not None
            and _supports_steps_flag(python, inference_py)):
        argv += ["--num_inference_steps", str(int(steps))]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if profile.image_guidance is not None and model_type == "full":
        argv += ["--guidance_scale", str(float(profile.image_guidance))]

    # Reference images: forwarded as one or more positional values to
    # --ref_images. Upstream treats one ref + --model_type dev as editing
    # mode; multiple refs as composition / multi-subject. The width/height
    # bucket-snap still applies unless --keep_original_aspect is set with
    # exactly one ref.
    if req.ref_images:
        argv.append("--ref_images")
        argv += [str(p) for p in req.ref_images]
        if req.keep_original_aspect and len(req.ref_images) == 1:
            argv.append("--keep_original_aspect")
        # Editing scheduler is only consulted by upstream when the editing
        # branch is taken (exactly one ref + dev model). Sending it in
        # other configs is harmless — argparse accepts and the pipeline
        # discards.
        if profile.image_editing_scheduler:
            sched = profile.image_editing_scheduler.lower()
            if sched in ("flow_match", "flash"):
                argv += ["--editing_scheduler", sched]
    if req.layout_bboxes:
        argv += ["--layout_bboxes", req.layout_bboxes]

    # Honour profile.args as raw passthrough (snake_case → --kebab-case).
    for k, v in (profile.args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    # Force UTF-8 for the subprocess pipes. Transformers' @auto_docstring
    # prints a 🚨 character at import time; with no console attached
    # (asyncio PIPE) Python defaults to the active legacy ANSI codepage
    # on Windows — cp1252 — and that import raises UnicodeEncodeError,
    # killing the model load before generation can start.
    env: dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
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


def capabilities() -> dict[str, Any]:
    """HiDream accepts reference images for editing and composition.

    One ref triggers editing semantics; multiple refs drive multi-subject
    composition (up to 8). ``keep_original_aspect`` is meaningful only with
    exactly one ref (it locks the output to the ref's aspect ratio).
    """
    return {
        "ref_images_max": 8,
        "ref_label": "Reference images",
        "ref_help": "1 image = edit · multiple = composition / multi-subject (up to 8).",
        "keep_original_aspect": True,
    }


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
