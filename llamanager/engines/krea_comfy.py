"""Krea 2 Turbo via ComfyUI — text-to-image and instruction editing.

The only Krea 2 route llamanager ships, and the loader is the reason: Krea 2
Turbo's community GGUF quants are unusable from diffusers, which has no
single-file or GGUF path for this architecture, so a diffusers adapter could
only run the full 24.5 GB bf16 checkpoint. The Q6_K quant is 9.9 GB and loads
directly here, which leaves a 32 GB card enough headroom to hold the whole
pipeline resident instead of streaming weights.

Krea 2 is guidance-distilled: cfg is 1.0 and the negative branch carries no
guidance to tune. Its conditioner is a Qwen3-VL 4B of its own — loader type
``krea2``, not ``qwen_image``.

THREE GRAPHS, CHOSEN BY THE LoRA. Stock Krea 2 cannot read a reference image
at all; two community node packs add that, in incompatible ways, and each
community edit LoRA was trained against exactly one of them. So the profile's
LoRA selects the workflow, the legal number of references, and (pack A) the
reference geometry — see LORA_RECIPES. A LoRA we have no entry for is a plain
style LoRA and stays on the text-to-image graph; asking it to edit raises,
because a LoRA run through the wrong pack still produces an image, just not a
correct one.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config, Profile
from ._base import (ImageRequest, ProfileField, ProgressEvent, pick_guidance,
                    pick_model_type, pick_scheduler)

log = logging.getLogger(__name__)

ENGINE = "krea_comfy"
LABEL = "Krea 2 Turbo (ComfyUI)"

_DEFAULT_STEPS = 8             # the Turbo distill's design point
_DEFAULT_CFG = 1.0             # guidance-distilled: raising this degrades output
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"
_DEFAULT_SIZE = "1024x1024"

VAE_FILE = "qwen_image_vae.safetensors"

# Text encoder. Two forms are supported and they differ by 700 seconds.
#
# The Comfy-Org safetensors (fp8_scaled or bf16) build the encoder through
# ComfyUI's ordinary CLIPLoader, which on gfx1201 spends ~719 s of
# single-threaded CPU per request constructing it — 97% of the whole
# generation. A llama.cpp-style GGUF of the same Qwen3-VL-4B, with its mmproj
# companion beside it, loads through CLIPLoaderGGUF in about one second and
# produces a near-identical image (compared at equal seed). So the GGUF is
# the default and the safetensors are the fallback for a directory that only
# has those.
CLIP_GGUF = "Qwen3-VL-4B-Instruct-Q8_0.gguf"
CLIP_GGUF_MMPROJ = "mmproj-F16.gguf"
CLIP_SAFETENSORS = "qwen3vl_4b_fp8_scaled.safetensors"
CLIP_FILE = CLIP_SAFETENSORS   # kept for callers/tests that reference the name


def resolve_text_encoder(model_dir: Path) -> tuple[str, str]:
    """(filename, template suffix) for the encoder present in ``model_dir``.

    The suffix picks the loader half of a workflow pair: ``_gguf_te`` builds
    the conditioner through CLIPLoaderGGUF, ``_gguf`` through ComfyUI's own
    CLIPLoader.

    A GGUF without its mmproj companion is NOT usable — ComfyUI cannot
    recognise the architecture without the vision tower keys, and both edit
    node packs need that tower to feed images through the encoder at all.
    Having the GGUF but not the mmproj therefore raises: silently serving the
    safetensors instead would turn a 22-second request into a 12-minute one
    and report nothing.
    """
    te = model_dir / "text_encoders"
    if (te / CLIP_GGUF).is_file():
        if not (te / CLIP_GGUF_MMPROJ).is_file():
            raise RuntimeError(
                f"text_encoders/{CLIP_GGUF} is present but its companion "
                f"{CLIP_GGUF_MMPROJ} is not. Without the vision tower ComfyUI "
                "cannot recognise the encoder, and image grounding has "
                f"nothing to look through. Download {CLIP_GGUF_MMPROJ} into "
                f"{te}, or delete the GGUF to use the (much slower) "
                f"{CLIP_SAFETENSORS}.")
        return CLIP_GGUF, "_gguf_te"
    return CLIP_SAFETENSORS, "_gguf"

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

# --------------------------------------------------------------- edit LoRAs
#
# Krea 2 has two community edit node packs, and they are not interchangeable.
# Pack A (ComfyUI-Krea2Edit, lbouaraba) prepends the source latent as clean
# tokens at RoPE frame 1 and shows Qwen3-VL the image at up to 1024 px.
# Pack B (ComfyUI-Krea2-Ostris-Edit) appends references to the image token
# sequence conditioned at timestep 0, and shows Qwen3-VL a deliberately
# coarse 384x384. A LoRA trained against one is out of distribution in the
# other — it still renders, which is exactly why this cannot be a mode toggle
# the operator sets independently of the weights.
#
# So the LoRA IS the recipe: it selects the graph, how many references are
# legal, and (pack A) the reference geometry its training used. An adapter we
# have no entry for is a plain text-to-image LoRA; asking it to edit raises
# rather than guessing a pack.

@dataclass(frozen=True)
class EditRecipe:
    """How one edit LoRA must be run."""
    graph: str          # workflow template stem, without the encoder suffix
    refs_min: int
    refs_max: int
    fit_mode: str = ""  # pack A only: the geometry this version trained on
    note: str = ""


def _identity_edit(version: str, fit_mode: str) -> dict[str, EditRecipe]:
    """The identity-edit LoRA ships full, r128 and r64 cuts of each version.

    They are the same weights at different ranks, so they share a recipe;
    listing them explicitly keeps an unrecognised filename unrecognised.
    """
    r = EditRecipe(graph="krea2_edit_a", refs_min=1, refs_max=2,
                   fit_mode=fit_mode,
                   note="scene first, subject second for two-image edits")
    return {f"krea2_identity_edit_{version}{cut}.safetensors": r
            for cut in ("", "_r128", "_r64")}


LORA_RECIPES: dict[str, EditRecipe] = {
    # Pack A. v1.2 trained on 'fit' geometry; v1/v1.1 on the legacy crop.
    **_identity_edit("v1_2", "fit"),
    **_identity_edit("v1_1", "crop (legacy)"),
    **_identity_edit("v1", "crop (legacy)"),
    # Pack B.
    "krea2_style_reference.safetensors": EditRecipe(
        graph="krea2_edit_b", refs_min=1, refs_max=3,
        note="style transfer: the reference supplies look, the prompt content"),
    "krea2_turbo_openpose_controlnet.safetensors": EditRecipe(
        graph="krea2_edit_b", refs_min=1, refs_max=1,
        note="the reference must be a DWPose/OpenPose skeleton, not a photo"),
}

# Reference slots, in order. A graph names one LoadImage per slot; the engine
# fills the ones it has and drops the rest.
REF_TOKENS = ("REF_IMAGE", "REF_IMAGE_B", "REF_IMAGE_C")

# Nodes that exist only to serve an optional reference slot, by slot index.
# These ids are a property of the frozen templates (comfy_workflows/<graph>_*)
# and are dropped from the graph when the slot is unfilled — a LoadImage has
# no upstream to bypass to.
OPTIONAL_REF_NODES: dict[str, dict[int, tuple[str, ...]]] = {
    "krea2_edit_a": {1: ("12", "14")},   # LoadImage + the VAEEncode behind it
    "krea2_edit_b": {1: ("12",), 2: ("13",)},
}

# Pack A dials. 4.0 is what the pack's own reference workflow ships (the node
# default is 1.0, i.e. off); 768 is the middle of the trained grounding range.
_DEFAULT_REF_BOOST = 4.0
_DEFAULT_GROUNDING_PX = 768


def resolve_recipe(lora_name: str, n_refs: int) -> EditRecipe | None:
    """The recipe for ``lora_name``, or None for a plain text-to-image run.

    Raises when the combination cannot be honoured, rather than picking a
    graph for the operator: running an edit LoRA without its reference, or a
    reference through a graph that was never trained to read one, produces a
    confident wrong image instead of an error.
    """
    known = ", ".join(sorted(LORA_RECIPES))
    if not lora_name:
        if n_refs:
            raise RuntimeError(
                "Krea 2 editing needs an edit LoRA — reference images alone "
                "do nothing, because stock Krea 2 has no path to read them. "
                f"Set a profile LoRA to one of: {known}")
        return None
    recipe = LORA_RECIPES.get(lora_name)
    if recipe is None:
        if n_refs:
            raise RuntimeError(
                f"{lora_name} is not a LoRA llamanager knows how to edit "
                "with, and the two Krea 2 edit node packs need different "
                "graphs — there is no safe guess. Use it without reference "
                f"images for text-to-image, or pick one of: {known}")
        return None
    if n_refs < recipe.refs_min:
        raise RuntimeError(
            f"{lora_name} is an edit LoRA: it needs at least "
            f"{recipe.refs_min} reference image"
            f"{'s' if recipe.refs_min > 1 else ''}"
            + (f" ({recipe.note})" if recipe.note else "")
            + ". Attach one in the composer, or switch the profile's LoRA.")
    if n_refs > recipe.refs_max:
        raise RuntimeError(
            f"{lora_name} takes at most {recipe.refs_max} reference image"
            f"{'s' if recipe.refs_max > 1 else ''}; got {n_refs}.")
    return recipe


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


def _unet_for(profile: Profile, req: ImageRequest) -> str:
    """Transformer filename for the requested quant.

    An unrecognised quant raises: substituting the default would run weights
    the operator did not ask for and report success, and a typo in a saved
    profile is exactly how that happens.
    """
    quant = pick_model_type(req, profile).upper()
    if not quant:
        quant = DEFAULT_QUANT
    entry = QUANT_FILES.get(quant)
    if entry is None:
        raise RuntimeError(
            f"unknown Krea 2 transformer quant {quant!r}; "
            f"expected one of: {', '.join(QUANT_FILES)}")
    return entry[0]


def _reference_args(recipe: EditRecipe, refs: list[Path]) -> list[str]:
    """Upload flags for the filled reference slots, drops for the rest.

    Every slot's token is still substituted even when its node is about to be
    dropped: the graph is rendered before any node is removed, and an
    unsubstituted token is (correctly) a hard error in render_workflow.
    """
    argv: list[str] = []
    optional = OPTIONAL_REF_NODES.get(recipe.graph, {})
    for slot in range(recipe.refs_max):
        token = REF_TOKENS[slot]
        if slot < len(refs):
            argv += ["--image", f"{token}={refs[slot]}"]
            continue
        argv += ["--set-str", f"{token}="]
        for node_id in optional[slot]:
            argv += ["--drop-node", node_id]
    return argv


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
    lora_name = profile.image_lora_weights or ""
    recipe = resolve_recipe(lora_name, len(req.ref_images))

    width, height = _resolved_size(profile, req)
    steps = _resolved_steps(profile, req)
    seed = req.seed if req.seed is not None else profile.image_seed
    unet = _unet_for(profile, req)
    _g = pick_guidance(req, profile)
    cfg_scale = _g if _g is not None else _DEFAULT_CFG

    clip_file, te_suffix = resolve_text_encoder(model_path)
    graph = (recipe.graph if recipe is not None else "krea2_t2i") + te_suffix
    required = {"diffusion_models": unet, "text_encoders": clip_file,
                "vae": VAE_FILE}
    if recipe is not None:
        # The edit LoRA is load-bearing, not decorative: check it here with
        # the rest of the model rather than warning and sampling without it.
        required["loras"] = lora_name
    missing = cb.missing_files(model_path, required)
    if missing:
        raise RuntimeError(
            "Krea 2 Turbo model directory is incomplete — missing "
            + ", ".join(missing)
            + f". Download the remaining components into {model_path}.")

    argv: list[str] = [
        str(python), "-u", str(runner),
        "--comfy-repo", str(repo),
        "--model-path", str(model_path),
        "--workflow", str(cb.workflow_path(graph)),
        "--output", str(out_path),
        "--set", f"UNET={unet}",
        "--set", f"CLIP={clip_file}",
        "--set", f"VAE={VAE_FILE}",
        # --set-str, not --set: a prompt of "2024" is valid JSON and would
        # otherwise reach the graph as a number.
        "--set-str", f"PROMPT={req.prompt}",
        "--set", f"WIDTH={width}",
        "--set", f"HEIGHT={height}",
        "--set", f"STEPS={steps}",
        "--set", f"CFG={float(cfg_scale)}",
        "--set", f"SAMPLER={pick_scheduler(req, profile) or _DEFAULT_SAMPLER}",
        "--set", f"SCHEDULER={_DEFAULT_SCHEDULER}",
        "--set", f"SEED={int(seed) if seed is not None else 0}",
    ]

    if lora_name and (model_path / "loras" / lora_name).is_file():
        strength = (profile.image_lora_scale
                    if profile.image_lora_scale is not None else 1.0)
        argv += ["--set", f"LORA={lora_name}",
                 "--set", f"LORA_STRENGTH={float(strength)}",
                 # Makes the runner check afterwards that ComfyUI actually
                 # bound the thing. ComfyUI only warns when a LoRA's keys
                 # match nothing, so without this a LoRA for the wrong
                 # architecture renders a normal image and reports success.
                 "--lora-file", str(model_path / "loras" / lora_name)]
    else:
        if lora_name:
            # Only reachable for a plain t2i LoRA: an edit recipe already
            # required the file through missing_files() above.
            log.warning("krea_comfy: LoRA %s not found in %s; sampling "
                        "without it", lora_name, model_path / "loras")
        # Dropped rather than zeroed so no LoRA file is read at all.
        argv += ["--set", "LORA=", "--set", "LORA_STRENGTH=0.0",
                 "--bypass", "2:model"]

    if recipe is not None:
        argv += _reference_args(recipe, req.ref_images)
        if recipe.graph == "krea2_edit_a":
            boost = (profile.image_ref_boost
                     if profile.image_ref_boost is not None
                     else _DEFAULT_REF_BOOST)
            px = (profile.image_grounding_px
                  if profile.image_grounding_px is not None
                  else _DEFAULT_GROUNDING_PX)
            argv += ["--set", f"REF_BOOST={float(boost)}",
                     "--set", f"GROUNDING_PX={int(px)}",
                     # Not an operator choice: the LoRA version determines it.
                     "--set-str", f"FIT_MODE={recipe.fit_mode}"]

    keep_warm = int(getattr(cfg, "comfy_keep_warm_s", 0) or 0)
    if keep_warm > 0:
        argv += ["--keep-warm", str(keep_warm)]

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
            default="", options_dir="loras",
            help="A filename from the model's loras/ folder. Style and "
                 "realism LoRAs change text-to-image; the edit LoRAs "
                 "(identity, style-reference, openpose) turn this profile "
                 "into an editor that requires a reference image.",
        ),
        ProfileField(
            key="image_lora_scale", label="LoRA strength", kind="float",
            default=1.0, help="0.8 is a good starting point for style LoRAs; "
                              "the edit LoRAs want 1.0.",
        ),
        ProfileField(
            key="image_ref_boost", label="Reference fidelity", kind="float",
            default=_DEFAULT_REF_BOOST,
            help="Identity-edit only. Multiplies how hard the model looks at "
                 "the reference: 1.0 is off, 4.0 is the reference workflow's "
                 "setting, above ~10 it over-copies and removals stop "
                 "working.",
        ),
        ProfileField(
            key="image_grounding_px", label="Grounding size", kind="int",
            default=_DEFAULT_GROUNDING_PX,
            help="Identity-edit only. How large the text encoder sees the "
                 "reference while reading the instruction: 384-768 is the "
                 "trained range, 1024 favours face likeness, 512 helps a "
                 "stubborn scene change.",
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
    """Reference slots come from the most permissive edit LoRA we support.

    Caps are per engine, but whether references are *usable* depends on the
    profile's LoRA — so the composer offers the slots and build_command
    enforces the pairing. ``ref_images_required`` would be a lie here: the
    same engine serves plain text-to-image.
    """
    return {
        "output_ext": "png",
        "ref_images_max": max(r.refs_max for r in LORA_RECIPES.values()),
    }


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
        # Editing profiles. Each names the LoRA that defines its graph, so
        # selecting the profile is the whole act of switching modes.
        "kreac-edit": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": 10,          # the pack's own workflow ships 10
            "image_guidance": _DEFAULT_CFG,
            "image_model_type": "Q6_K",
            "image_lora_weights": "krea2_identity_edit_v1_2.safetensors",
            "image_lora_scale": 1.0,
            "image_ref_boost": _DEFAULT_REF_BOOST,
            "image_grounding_px": _DEFAULT_GROUNDING_PX,
        },
        "kreac-style-ref": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": _DEFAULT_STEPS,
            "image_guidance": _DEFAULT_CFG,
            "image_model_type": "Q6_K",
            "image_lora_weights": "krea2_style_reference.safetensors",
            "image_lora_scale": 1.0,
        },
        "kreac-pose": {
            "image_size": _DEFAULT_SIZE,
            "image_steps": 10,          # the LoRA card's setting
            "image_guidance": _DEFAULT_CFG,
            "image_model_type": "Q6_K",
            "image_lora_weights": "krea2_turbo_openpose_controlnet.safetensors",
            "image_lora_scale": 0.9,
        },
    }
