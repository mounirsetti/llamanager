"""Curated catalog of supported diffusion models.

Each entry says "this is a model we know how to run" — the on-disk
``model_id`` it produces after a HF snapshot download, which engine it
binds to, and the canonical HF repo / subfolder to pull from. The
Diffusion-models page joins this catalog against what's actually on
disk so it can show "Installed (activate / edit profiles)" or "Not
installed (install on the Diffusion engines page)" for each row.

Keeping the catalog in a single Python module (vs in config.toml) means
we can ship updates with code releases — new model support lands as
one new entry here, no operator action required.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    """One known diffusion model.

    ``canonical_id`` is the directory name produced by a default HF
    snapshot download into ``models_dir``. The registry's model_id
    matches it (except for sub-folder pulls, where the operator's
    target name overrides this — see ``Z-Anime`` for the subfolder
    case).

    ``hf_repo`` and ``subfolder`` populate the existing download form
    on the Diffusion engines page; the install link prefills both so
    one click takes the operator from "I want this model" to
    "downloading".
    """
    canonical_id: str
    engine: str            # 'hidream' | 'z_image' | 'krea' | 'ideogram4' | 'flux2' | 'wan'
    label: str             # human-readable name
    hf_repo: str           # 'org/name'
    subfolder: str = ""    # optional HF subfolder
    approx_size_gb: float = 0.0
    description: str = ""  # 1-3 sentences for the catalog row
    homepage: str = ""     # canonical model URL


CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        canonical_id="HiDream-O1-Image",
        engine="hidream",
        label="HiDream-O1-Image",
        hf_repo="HiDream-ai/HiDream-O1-Image",
        approx_size_gb=18.0,
        description=(
            "HiDream's flagship text-to-image model. Two recipes ship in "
            "one checkpoint: a 28-step 'dev' path and a 50-step 'full' "
            "path with classifier-free guidance. Native resolution buckets "
            "from 2048x2048 up to 3104x1312."
        ),
        homepage="https://huggingface.co/HiDream-ai/HiDream-O1-Image",
    ),
    CatalogEntry(
        canonical_id="Z-Image",
        engine="z_image",
        label="Z-Image (Tongyi-MAI)",
        hf_repo="Tongyi-MAI/Z-Image",
        approx_size_gb=20.0,
        description=(
            "Alibaba Tongyi-MAI's DiT-based text-to-image model. Diffusers "
            "layout, runs via the bundled z_image runner. Solid all-rounder."
        ),
        homepage="https://huggingface.co/Tongyi-MAI/Z-Image",
    ),
    CatalogEntry(
        canonical_id="Z-Anime",
        engine="z_image",
        label="Z-Anime (Z-Image fine-tune)",
        hf_repo="SeeSee21/Z-Anime",
        subfolder="diffusers",
        approx_size_gb=15.0,
        description=(
            "Anime / stylised fine-tune of Z-Image. The full repo is 203 GB "
            "(checkpoint variants for many pipelines); the 'diffusers/' "
            "subfolder is the runnable variant for llamanager (~12-20 GB)."
        ),
        homepage="https://huggingface.co/SeeSee21/Z-Anime",
    ),
    CatalogEntry(
        canonical_id="vantagewithai/Krea-2-Turbo-GGUF",
        engine="krea",
        label="Krea 2 Turbo GGUF",
        hf_repo="vantagewithai/Krea-2-Turbo-GGUF",
        approx_size_gb=13.7,
        description=(
            "Krea 2 Turbo as GGUF-quantized Qwen-Image transformer weights. "
            "Use the quant picker on the Diffusion engines page to download "
            "one file at a time instead of the full 108 GB repo."
        ),
        homepage="https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF",
    ),
    CatalogEntry(
        canonical_id="krea/Krea-2-Turbo",
        engine="krea",
        label="Krea 2 Turbo (original)",
        hf_repo="krea/Krea-2-Turbo",
        approx_size_gb=26.3,
        description=(
            "The original open-weight Krea 2 Turbo Diffusers checkpoint. "
            "Higher quality ceiling than smaller quants, but it needs much "
            "more VRAM and disk than the GGUF downloads."
        ),
        homepage="https://huggingface.co/krea/Krea-2-Turbo",
    ),
    CatalogEntry(
        canonical_id="ideogram-ai/ideogram-4-fp8",
        engine="ideogram4",
        label="Ideogram 4 fp8 (official)",
        hf_repo="ideogram-ai/ideogram-4-fp8",
        approx_size_gb=27.5,
        description=(
            "Official Ideogram 4 fp8 weights in the diffusers layout the "
            "official runner expects. Gated repo: accept the license on "
            "Hugging Face and configure an HF token before downloading. "
            "Non-commercial license."
        ),
        homepage="https://huggingface.co/ideogram-ai/ideogram-4-fp8",
    ),
    CatalogEntry(
        canonical_id="Comfy-Org/Ideogram-4",
        engine="ideogram4",
        label="Ideogram 4 (Comfy-Org)",
        hf_repo="Comfy-Org/Ideogram-4",
        approx_size_gb=65.9,
        description=(
            "ComfyUI-style repack of the Ideogram 4 fp8 weights. Detected "
            "by the engine, but the official runner cannot load this "
            "layout yet — prefer the official fp8 repo."
        ),
        homepage="https://huggingface.co/Comfy-Org/Ideogram-4",
    ),
    CatalogEntry(
        canonical_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        engine="wan",
        label="Wan 2.2 TI2V-5B (text+image→video)",
        hf_repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        approx_size_gb=28.0,
        description=(
            "Alibaba Wan 2.2 — a single dense 5B model that does BOTH "
            "text-to-video and image-to-video. 720p (1280x704), 24fps, ~5s "
            "clips. Full bf16 diffusers weights (no GGUF — the reliable path "
            "on ROCm). Runs via the bundled wan runner; supply one reference "
            "image to animate it as the opening frame."
        ),
        homepage="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ),
    CatalogEntry(
        canonical_id="FLUX.2-dev",
        engine="flux2",
        label="FLUX 2 Dev",
        hf_repo="black-forest-labs/FLUX.2-dev",
        approx_size_gb=24.0,
        description=(
            "Black Forest Labs' second-generation flow-matching model. "
            "Run via sd-cli (stable-diffusion.cpp); the HF repo holds the "
            "canonical fp16 weights — for runnable GGUF quants, search for "
            "a community re-host."
        ),
        homepage="https://huggingface.co/black-forest-labs/FLUX.2-dev",
    ),
]


def for_engine(engine: str) -> list[CatalogEntry]:
    """Catalog entries that target one engine."""
    return [e for e in CATALOG if e.engine == engine]


def by_canonical_id(model_id: str) -> CatalogEntry | None:
    """Look up a catalog entry by its canonical on-disk model id."""
    for e in CATALOG:
        if e.canonical_id == model_id:
            return e
    return None
