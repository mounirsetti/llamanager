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
    engine: str            # 'hidream' | 'z_image' | 'ideogram4' | 'flux2' | 'wan' | 'minimax_h3'
    label: str             # human-readable name
    hf_repo: str           # 'org/name'
    subfolder: str = ""    # optional HF subfolder
    approx_size_gb: float = 0.0
    description: str = ""  # 1-3 sentences for the catalog row
    homepage: str = ""     # canonical model URL
    # Models assembled from several uploaders (the ComfyUI family: transformer,
    # text encoder, VAEs and LoRA each ship from a different repo) list their
    # parts here as (repo, filename, subdir, size_gb, note). The setup page
    # renders one download button per part, each targeting
    # ``canonical_id/subdir`` via the registry's target_dir. ``hf_repo`` above
    # then names only the primary repo, for the homepage link and search.
    components: tuple[tuple[str, str, str, float, str], ...] = ()


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
        canonical_id="MiniMaxAI/MiniMax-H3",
        engine="minimax_h3",
        label="MiniMax-H3 (video + audio)",
        hf_repo="MiniMaxAI/MiniMax-H3",
        # The repo carries BOTH the original FL2VA/Ref2VA release (268 GB)
        # and the diffusers conversion, so a whole-repo pull is 464 GB. Naming
        # the components diffusers actually loads brings it to 140 GB.
        subfolder=("transformer,text_encoder,vae,audio_vae,"
                   "scheduler,audio_scheduler,tokenizer,processor"),
        approx_size_gb=140.0,
        description=(
            "Generates video and its soundtrack together — one transformer "
            "denoises the video and audio latents in the same loop, so there "
            "is no vocoder and no separate audio pass. 24fps, 5-15s clips, "
            "text-to-video or first/last keyframe. Guidance is distilled into "
            "the weights, so there is no negative prompt or guidance scale. "
            "VERY LARGE: 61.7 GB transformer + 62.1 GB Qwen3-VL conditioner "
            "in bf16 — 140 GB on disk for the diffusers components (the full "
            "repo is 464 GB because it also ships the original release "
            "format; the prefilled subfolder list skips that). Quantised to "
            "4-bit NF4 and loaded one component at a time it peaks around "
            "21.5 GB of VRAM, which fits a 32 GB card."
        ),
        homepage="https://huggingface.co/MiniMaxAI/MiniMax-H3",
    ),
    CatalogEntry(
        canonical_id="MiniMax-H3-Comfy",
        engine="minimax_h3_comfy",
        label="MiniMax-H3 (ComfyUI, video + audio)",
        hf_repo="realrebelai/MiniMax-H3_GGUFs",
        # The default recipe only: Q4_K_M transformer + encoder + both VAEs +
        # LoRA. The Q3_K_M component below is an alternative to the Q4_K_M
        # transformer, not an addition, so the component sizes sum higher.
        approx_size_gb=39.3,
        description=(
            "The same model as the entry above, in ComfyUI's pre-quantised "
            "GGUF format instead of bf16 diffusers — 39 GB of downloads "
            "rather than 140 GB, and no quantise-on-load step. That step is "
            "the whole reason this variant exists: diffusers has no GGUF or "
            "single-file loader for MiniMax-H3, so it must convert bf16 "
            "weights on every load, measured here at 5.8 s/tensor and ~50 GB "
            "of host RAM. Generates video and its soundtrack together at "
            "24fps; supply one image as the opening frame. Ships with the "
            "lightx2v Turbo distill LoRA, which cuts sampling from 50 steps "
            "to 4. Requires the ComfyUI engine to be installed."
        ),
        homepage="https://huggingface.co/MiniMaxAI/MiniMax-H3",
        components=(
            ("realrebelai/MiniMax-H3_GGUFs", "MiniMax-H3-FL2VA-Q4_K_M.gguf",
             "diffusion_models", 18.50,
             "Transformer, 4-bit. The quality/VRAM sweet spot on a 32 GB card."),
            ("realrebelai/MiniMax-H3_GGUFs", "MiniMax-H3-FL2VA-Q3_K_M.gguf",
             "diffusion_models", 14.51,
             "Transformer, 3-bit. Optional: more headroom, less detail."),
            ("ChrisColeTech/minimax-h3-turbo-GGUF",
             "split/diffusion_models/minimax_h3_fl2va_turbo_Q4_K_M.gguf",
             "diffusion_models", 10.61,
             "Transformer, 4-bit, with the 8-step Turbo distill fused in and "
             "the adaln projection stored against a curve basis. Same "
             "sampling speed as the Q4_K_M above, but it loads in 16 s "
             "instead of 76 s and holds 11.2 GB instead of 19.4 GB. Needs "
             "the LoRA field cleared and 8 steps. Optional alternative to "
             "the Q4_K_M transformer, not an addition."),
            ("realrebelai/MiniMax-H3_GGUFs",
             "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", "text_encoders", 13.58,
             "Qwen3-VL 32B conditioner, 4-bit."),
            ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors",
             "vae", 4.85, "Video VAE (fp16)."),
            ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors",
             "vae", 0.56,
             "Audio VAE (fp32). Required — without it the clip is silent."),
            ("lightx2v/Minimax-h3-Turbo",
             "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
             "loras", 1.82,
             "Turbo distill LoRA: 50 sampling steps down to 4."),
        ),
    ),
    CatalogEntry(
        canonical_id="Krea-2-Turbo-Comfy",
        engine="krea_comfy",
        label="Krea 2 Turbo (ComfyUI)",
        hf_repo="vantagewithai/Krea-2-Turbo-GGUF",
        # The Q6_K recipe; the other transformer quants below are alternatives
        # to it, not additions.
        approx_size_gb=15.0,
        description=(
            "Krea 2 Turbo loaded from a GGUF quant instead of the 24.5 GB "
            "bf16 checkpoint. diffusers has no single-file or GGUF path for "
            "this architecture, so the quants are only reachable through "
            "ComfyUI — this is the only Krea 2 route llamanager ships. "
            "Q6_K is 9.9 GB, which leaves "
            "a 32 GB card enough headroom to keep the whole pipeline resident. "
            "Guidance-distilled: 8 steps, cfg 1.0, no negative prompt. "
            "Requires the ComfyUI engine to be installed."
        ),
        homepage="https://huggingface.co/krea/Krea-2-Turbo",
        components=(
            ("vantagewithai/Krea-2-Turbo-GGUF", "krea2_turbo-Q6_K.gguf",
             "diffusion_models", 9.86,
             "Transformer, 6-bit. The default: closest to bf16 that still "
             "leaves headroom beside the conditioner."),
            ("vantagewithai/Krea-2-Turbo-GGUF", "krea2_turbo-Q8_0.gguf",
             "diffusion_models", 12.76,
             "Transformer, 8-bit. Optional: highest fidelity, least headroom."),
            ("vantagewithai/Krea-2-Turbo-GGUF", "krea2_turbo-Q4_K_M.gguf",
             "diffusion_models", 6.97,
             "Transformer, 4-bit. Optional: fastest and smallest."),
            ("unsloth/Qwen3-VL-4B-Instruct-GGUF",
             "Qwen3-VL-4B-Instruct-Q8_0.gguf", "text_encoders", 3.99,
             "Qwen3-VL 4B conditioner as GGUF (Q8_0). Loads in ~1 s where the "
             "safetensors form takes ~12 min on ROCm; needs the mmproj below."),
            ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf",
             "text_encoders", 0.78,
             "Vision tower companion for the GGUF conditioner. Required with "
             "it: without these keys ComfyUI cannot recognise the model."),
            ("Comfy-Org/Krea-2",
             "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
             "text_encoders", 4.88,
             "Alternative conditioner (fp8 safetensors). Works, but ~12 min "
             "per request on ROCm; the GGUF pair above is the fast path."),
            ("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors", "vae", 0.24,
             "VAE."),
            ("Comfy-Org/Krea-2", "loras/krea2_darkbrush.safetensors",
             "loras", 0.44, "Optional style LoRA (dark brush)."),
            # Edit LoRAs. Each one selects its own inference graph (see
            # krea_comfy.LORA_RECIPES): the two community node packs place
            # and condition the reference differently, so the adapter is what
            # says which pack a request runs through.
            ("conradlocke/krea2-identity-edit",
             "krea2_identity_edit_v1_2.safetensors", "loras", 1.83,
             "Instruction editing with identity preservation: restage a "
             "person, recolour, insert or restyle, from 1-2 reference "
             "images. Krea 2 Community Licence (commercial use under $1M "
             "revenue). Needs a reference image to run."),
            ("conradlocke/krea2-identity-edit",
             "krea2_identity_edit_v1_2_r64.safetensors", "loras", 0.46,
             "The same edit LoRA at rank 64 — a quarter of the size for "
             "minimal quality loss. Take this one OR the full-rank file "
             "above, not both."),
            ("ostris/krea2_turbo_style_reference",
             "krea2_style_reference.safetensors", "loras", 0.46,
             "Style transfer from up to 3 reference images: the references "
             "supply the look, the prompt the content. Krea 2 Community "
             "Licence. Needs a reference image to run."),
            ("thedeoxen/Krea-2-pose-controlnet",
             "krea2_turbo_openpose_controlnet.safetensors", "loras", 0.23,
             "Pose control (Apache-2.0). Despite the name it is a LoRA, not "
             "a ControlNet: feed it a DWPose/OpenPose skeleton on black as "
             "the reference image and the prompt does the rest. llamanager "
             "does not extract skeletons — supply one."),
            ("gokaygokay/Krea-2-Realism-LoRA",
             "krea2_realism_lora.safetensors", "loras", 0.47,
             "Photorealism style LoRA for plain text-to-image. No reference "
             "image, no extra nodes."),
            ("RudySen/Krea2-realism-V2", "Krea2-realism-V2.safetensors",
             "loras", 1.56,
             "A second photorealism LoRA (MIT), heavier than the one above "
             "and prompt-hungry: it wants 4-5 descriptive sentences."),
        ),
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
