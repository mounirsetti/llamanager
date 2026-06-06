"""Z-Image inference runner — invoked by ``z_image.py`` as a subprocess.

This script ships with llamanager and runs inside the user-configured
Python environment (the one with ``diffusers`` / ``transformers`` /
``accelerate`` installed). It loads a ``ZImagePipeline`` (or, when the
model lives in a ``diffusers/`` subfolder, a generic ``DiffusionPipeline``)
and writes a single PNG to disk.

Progress: diffusers' samplers use tqdm, which writes "N/M [elapsed<eta]"
lines to stderr. The parent adapter's ``parse_progress`` keys off the
"N/M" portion, so we don't need to emit our own progress channel.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# Default Hugging Face repo we pull the *small* scaffold from (component
# configs + tokenizer + scheduler) when a model ships as loose single-file
# weights instead of a full diffusers tree. Only the non-weight files are
# fetched (~16 MB), never the multi-GB safetensors.
DEFAULT_SCAFFOLD_REPO = "Tongyi-MAI/Z-Image"

# Glob patterns for the scaffold download — everything diffusers needs to
# instantiate the pipeline *except* the big component weights, which come
# from the user's own single-file checkpoints.
_SCAFFOLD_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "transformer/config.json",
    "vae/config.json",
]


def _select_device() -> tuple[str, str]:
    """Pick the best available device + dtype.

    Returns (device, dtype_name) where dtype_name is one of "bfloat16",
    "float16", or "float32". Apple MPS doesn't support bfloat16, so we
    fall back to float16 there. CPU stays in float32 to avoid extreme
    slowness from bf16 emulation.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


def _torch_dtype(dtype_name: str):
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


# ---- single-file (ComfyUI / fp8) support --------------------------------
#
# Z-Image is also distributed as loose, single-file safetensors (the
# layout ComfyUI uses): a transformer, a Qwen3 text encoder, and a VAE,
# each as its own fp8/bf16 file, with no per-component config subfolders
# and no tokenizer. ``ZImagePipeline.from_pretrained`` can't read that.
# We instead classify each file by its tensor keys, borrow the tiny
# config/tokenizer/scheduler scaffold from the canonical HF repo, and
# assemble the pipeline by hand.

def _safetensors_keys(path: Path) -> list[str]:
    """Read just the safetensors header and return its tensor names."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return [k for k in hdr if k != "__metadata__"]


def _classify(path: Path) -> str | None:
    """Guess which pipeline component a loose safetensors file holds.

    Returns 'transformer' | 'vae' | 'text_encoder' | None. Detection is
    by signature keys and is prefix-tolerant (ComfyUI checkpoints often
    carry a ``model.diffusion_model.`` prefix on the transformer)."""
    try:
        keys = _safetensors_keys(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    joined = "\n".join(keys)
    # Z-Image DiT signature blocks.
    if any(s in joined for s in (
            "cap_embedder", "context_refiner", "noise_refiner")):
        return "transformer"
    # AutoencoderKL signature.
    if "decoder.conv_in" in joined or "post_quant_conv" in joined:
        return "vae"
    # Qwen text encoder signature.
    if "embed_tokens" in joined:
        return "text_encoder"
    return None


def _is_diffusers_layout(model_path: Path) -> bool:
    """True when the dir is a real diffusers tree (component subfolders
    with their own configs), vs a loose single-file pack."""
    return (model_path / "transformer" / "config.json").is_file() or \
           (model_path / "diffusers" / "model_index.json").is_file()


def _discover_components(model_path: Path, *,
                         transformer_file: str | None,
                         text_encoder_file: str | None,
                         vae_file: str | None) -> dict[str, Path]:
    """Map {component: file} for a single-file pack.

    Operator overrides (``--transformer-file`` etc.) win; otherwise we
    classify every ``*.safetensors`` in the dir. When several transformer
    files are present (e.g. a quality checkpoint plus an N-step distill),
    we default to the one *without* 'distill' in its name and log the
    choice so the operator can pin the other via the profile."""
    found: dict[str, list[Path]] = {"transformer": [], "vae": [], "text_encoder": []}
    for p in sorted(model_path.glob("*.safetensors")):
        kind = _classify(p)
        if kind:
            found[kind].append(p)

    overrides = {
        "transformer": transformer_file,
        "text_encoder": text_encoder_file,
        "vae": vae_file,
    }
    out: dict[str, Path] = {}
    for comp, candidates in found.items():
        ov = overrides[comp]
        if ov:
            chosen = (model_path / ov) if not os.path.isabs(ov) else Path(ov)
            if not chosen.exists():
                raise FileNotFoundError(f"{comp} file not found: {chosen}")
            out[comp] = chosen
            continue
        if not candidates:
            raise RuntimeError(
                f"could not find a {comp} weight file in {model_path} "
                f"(looked at every *.safetensors). Pass --{comp.replace('_','-')}-file."
            )
        if comp == "transformer" and len(candidates) > 1:
            non_distill = [c for c in candidates if "distill" not in c.name.lower()]
            chosen = (non_distill or candidates)[0]
            others = [c.name for c in candidates if c != chosen]
            print(f"[z-image] multiple transformers found; using {chosen.name} "
                  f"(others: {', '.join(others)} — pin via profile "
                  f"transformer_file=...)", file=sys.stderr)
            out[comp] = chosen
        else:
            out[comp] = candidates[0]
    return out


def _ensure_scaffold(model_path: Path, scaffold: str | None,
                     scaffold_repo: str) -> Path:
    """Return a dir containing the component configs + tokenizer + scheduler.

    Resolution order: explicit ``--scaffold`` dir, then a cached
    ``<model>/.zimage-scaffold``, then a fresh HF download of just the
    small files. The cache means we only hit the network once per model."""
    if scaffold:
        sp = Path(scaffold).expanduser()
        if not (sp / "tokenizer").is_dir():
            raise RuntimeError(f"--scaffold {sp} has no tokenizer/ subfolder")
        return sp
    cache = model_path / ".zimage-scaffold"
    if (cache / "tokenizer" / "tokenizer.json").is_file() and \
       (cache / "vae" / "config.json").is_file():
        return cache
    print(f"[z-image] fetching config/tokenizer scaffold from "
          f"{scaffold_repo} (~16 MB, one-time)", file=sys.stderr)
    from huggingface_hub import snapshot_download
    snapshot_download(
        scaffold_repo,
        local_dir=str(cache),
        allow_patterns=_SCAFFOLD_PATTERNS,
    )
    return cache


def _assemble_single_file_pipeline(model_path: Path, dtype_name: str, *,
                                   transformer_file: str | None,
                                   text_encoder_file: str | None,
                                   vae_file: str | None,
                                   scaffold: str | None,
                                   scaffold_repo: str):
    """Build a ZImagePipeline from loose single-file weights + scaffold."""
    import torch
    from safetensors.torch import load_file
    from diffusers import (ZImageTransformer2DModel, AutoencoderKL,
                           FlowMatchEulerDiscreteScheduler, ZImagePipeline)
    from transformers import AutoModel, AutoConfig, AutoTokenizer

    torch_dtype = _torch_dtype(dtype_name)
    comps = _discover_components(
        model_path, transformer_file=transformer_file,
        text_encoder_file=text_encoder_file, vae_file=vae_file)
    scaf = _ensure_scaffold(model_path, scaffold, scaffold_repo)
    print(f"[z-image] single-file pack: transformer={comps['transformer'].name} "
          f"text_encoder={comps['text_encoder'].name} vae={comps['vae'].name}",
          file=sys.stderr)

    def _cast_sd(path: Path, strip_prefix: str | None = None) -> dict:
        sd = load_file(str(path))
        out = {}
        for k, v in sd.items():
            if strip_prefix and k.startswith(strip_prefix):
                k = k[len(strip_prefix):]
            out[k] = v.to(torch_dtype)
        return out

    # Transformer — diffusers' single-file loader understands the Z-Image
    # checkpoint format (incl. fp8) and the diffusers-format config.
    transformer = ZImageTransformer2DModel.from_single_file(
        str(comps["transformer"]),
        config=str(scaf / "transformer"),
        torch_dtype=torch_dtype,
    ).to(torch_dtype)

    # VAE — build from the scaffold config (Z-Image's is 16-latent-channel,
    # which from_single_file can't infer) then load the loose weights.
    vae = AutoencoderKL.from_config(
        json.loads((scaf / "vae" / "config.json").read_text())
    ).to(torch_dtype)
    miss, unexp = vae.load_state_dict(_cast_sd(comps["vae"]), strict=False)
    if miss or unexp:
        print(f"[z-image] vae load: missing={len(miss)} unexpected={len(unexp)}",
              file=sys.stderr)

    # Text encoder — Qwen3Model. ComfyUI files carry a CausalLM-style
    # 'model.' prefix; strip it to match the base-model keys.
    te_cfg = AutoConfig.from_pretrained(str(scaf / "text_encoder"))
    text_encoder = AutoModel.from_config(te_cfg).to(torch_dtype)
    te_sd = _cast_sd(comps["text_encoder"])
    miss, unexp = text_encoder.load_state_dict(te_sd, strict=False)
    if len(miss) > 50:  # prefix mismatch — retry with 'model.' stripped
        miss, unexp = text_encoder.load_state_dict(
            _cast_sd(comps["text_encoder"], strip_prefix="model."), strict=False)
    if miss or unexp:
        print(f"[z-image] text_encoder load: missing={len(miss)} "
              f"unexpected={len(unexp)}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(str(scaf / "tokenizer"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scaf / "scheduler"))

    return ZImagePipeline(
        transformer=transformer, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, scheduler=scheduler,
    )


def _load_pipeline(model_path: Path, dtype_name: str, *,
                   transformer_file: str | None = None,
                   text_encoder_file: str | None = None,
                   vae_file: str | None = None,
                   scaffold: str | None = None,
                   scaffold_repo: str = DEFAULT_SCAFFOLD_REPO):
    """Load a Z-Image pipeline from either a diffusers tree or a loose
    single-file (ComfyUI / fp8) pack.

    A real diffusers layout (component subfolders with configs) goes
    straight through ``from_pretrained``. Otherwise we treat the dir as a
    single-file pack and assemble the pipeline component-by-component."""
    torch_dtype = _torch_dtype(dtype_name)

    if _is_diffusers_layout(model_path):
        sub = model_path / "diffusers"
        target = sub if (sub / "model_index.json").is_file() else model_path
        try:
            from diffusers import ZImagePipeline
            return ZImagePipeline.from_pretrained(
                str(target), torch_dtype=torch_dtype, low_cpu_mem_usage=False)
        except (ImportError, AttributeError):
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(
                str(target), torch_dtype=torch_dtype, low_cpu_mem_usage=False)

    return _assemble_single_file_pipeline(
        model_path, dtype_name,
        transformer_file=transformer_file,
        text_encoder_file=text_encoder_file,
        vae_file=vae_file,
        scaffold=scaffold,
        scaffold_repo=scaffold_repo,
    )


def _resolve_vae_device(requested: str | None, main_device: str) -> str:
    """Pick the device for the VAE decode.

    'auto' keeps the VAE on the main device EXCEPT on AMD ROCm, where the
    full-image conv decode hits MIOpen kernel paths that are unstable on
    newer archs (gfx1201/RDNA4) and can fault the GPU — hard enough to
    drop the desktop session. There we default the (cheap) VAE decode to
    CPU; the expensive transformer still runs on the GPU."""
    import torch
    req = (requested or "auto").lower()
    if req != "auto":
        return req
    if main_device == "cuda" and getattr(torch.version, "hip", None):
        return "cpu"
    return main_device


def _decode_latents(pipe, latents, vae_device: str):
    """Decode latents → PIL with the VAE on ``vae_device``.

    Mirrors ZImagePipeline's in-pipeline decode (unscale → vae.decode →
    postprocess) so the decode can run on a different device than the
    transformer. Used when the VAE is pinned off the main device."""
    import torch
    vae = pipe.vae.to(vae_device)
    latents = latents.to(vae_device, dtype=vae.dtype)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def main() -> int:
    p = argparse.ArgumentParser(description="Z-Image inference runner")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="Override auto-detected device: cuda | mps | cpu.")
    p.add_argument("--dtype", default=None,
                   help="Override auto-detected dtype: bfloat16 | float16 | float32.")
    # Single-file (ComfyUI / fp8) pack controls. Ignored for diffusers-tree
    # models. ``--transformer-file`` lets the operator pin one of several
    # transformers (e.g. an N-step distill) via the profile.
    p.add_argument("--transformer-file", default=None,
                   help="Single-file pack: pin the transformer weights "
                        "(filename in the model dir, or absolute path).")
    p.add_argument("--text-encoder-file", default=None,
                   help="Single-file pack: pin the text-encoder weights.")
    p.add_argument("--vae-file", default=None,
                   help="Single-file pack: pin the VAE weights.")
    p.add_argument("--scaffold", default=None,
                   help="Dir with component configs + tokenizer + scheduler "
                        "(defaults to a cached download from --scaffold-repo).")
    p.add_argument("--scaffold-repo", default=DEFAULT_SCAFFOLD_REPO,
                   help="HF repo to pull the small config/tokenizer scaffold from.")
    p.add_argument("--vae-device", default="auto",
                   help="Device for the VAE decode: auto | cpu | cuda | mps. "
                        "'auto' keeps it on the main device except on AMD ROCm, "
                        "where it defaults to cpu (the conv decode hits unstable "
                        "MIOpen kernel paths on new archs like gfx1201).")
    args = p.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"[z-image] model path does not exist: {model_path}", file=sys.stderr)
        return 2

    device, default_dtype = _select_device()
    if args.device:
        device = args.device
    dtype_name = args.dtype or default_dtype
    print(f"[z-image] device={device} dtype={dtype_name}", file=sys.stderr)

    import torch
    pipe = _load_pipeline(
        model_path, dtype_name,
        transformer_file=args.transformer_file,
        text_encoder_file=args.text_encoder_file,
        vae_file=args.vae_file,
        scaffold=args.scaffold,
        scaffold_repo=args.scaffold_repo,
    )
    pipe.to(device)

    # VAE memory/stability hardening: tile + slice the decode so conv
    # workloads (and MIOpen kernels on AMD) stay small instead of
    # allocating one giant activation for the full image.
    for fn in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, fn)()
        except Exception:  # noqa: BLE001 — best-effort; older diffusers vary
            pass

    # Decide where the VAE decode runs. On ROCm this defaults to CPU to
    # dodge the gfx1201 MIOpen conv crash; the transformer stays on GPU.
    vae_device = _resolve_vae_device(args.vae_device, device)
    split_decode = vae_device != device
    if split_decode:
        print(f"[z-image] VAE decode on {vae_device} (transformer on {device})",
              file=sys.stderr)
        pipe.vae.to(vae_device)

    generator = None
    if args.seed is not None:
        # torch generators must live on the same device as the pipeline.
        try:
            generator = torch.Generator(device).manual_seed(int(args.seed))
        except (RuntimeError, ValueError):
            generator = torch.Generator("cpu").manual_seed(int(args.seed))

    print(
        f"[z-image] generating {args.width}x{args.height} "
        f"steps={args.steps} cfg={args.guidance} seed={args.seed}",
        file=sys.stderr,
    )

    if split_decode:
        # Run the transformer/scheduler loop on the GPU, get raw latents,
        # then decode them with the off-device VAE.
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            output_type="latent",
        )
        image = _decode_latents(pipe, result.images, vae_device)
    else:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
        image = result.images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(args.output))
    print(f"[z-image] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # Make sure the runner survives non-UTF8 default codepages on Windows.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
