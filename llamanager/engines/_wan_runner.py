"""Wan 2.2 video inference runner — invoked by ``wan.py`` as a subprocess.

This script ships with llamanager and runs inside the user-configured
Python environment (the one with ``diffusers`` / ``transformers`` /
``accelerate`` / ``torch`` + ``imageio-ffmpeg`` installed). It loads a
Wan pipeline and writes a single ``.mp4`` to disk.

Text-to-video uses ``WanPipeline``; when ``--init-image`` is supplied the
runner switches to ``WanImageToVideoPipeline`` (the image becomes the
first frame). The VAE is an ``AutoencoderKLWan`` loaded in fp32.

Progress: diffusers' samplers use tqdm, which writes "N/M [elapsed<eta]"
lines to stderr. The parent adapter's ``parse_progress`` keys off the
"N/M" portion, so we don't emit our own progress channel.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _select_device() -> tuple[str, str]:
    """Pick the best available device + dtype.

    Returns (device, dtype_name). Apple MPS doesn't support bfloat16, so we
    fall back to float16 there; CPU stays float32 to avoid bf16-emulation
    slowness (video on CPU is impractical, but we don't hard-block it).
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


def _resolve_vae_device(requested: str | None, main_device: str) -> str:
    """Pick the device for the VAE decode.

    'auto' keeps the VAE on the main device EXCEPT on AMD ROCm, where the
    conv decode hits MIOpen kernel paths that are unstable on newer archs
    (gfx1201/RDNA4) and can fault the GPU. There we default the VAE decode
    to CPU; the expensive transformer still runs on the GPU.
    """
    import torch
    req = (requested or "auto").lower()
    if req != "auto":
        return req
    if main_device == "cuda" and getattr(torch.version, "hip", None):
        return "cpu"
    return main_device


def _load_pipeline(model_path: Path, dtype_name: str, *, image_to_video: bool):
    """Load a Wan pipeline (text-to-video or image-to-video).

    The VAE is loaded separately in fp32 (Wan's VAE is numerically
    sensitive); the transformer/text-encoder load in the requested dtype.
    """
    from diffusers import AutoencoderKLWan
    torch_dtype = _torch_dtype(dtype_name)

    vae = AutoencoderKLWan.from_pretrained(
        str(model_path), subfolder="vae", torch_dtype=_torch_dtype("float32"))
    if image_to_video:
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(
            str(model_path), vae=vae, torch_dtype=torch_dtype)
    else:
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            str(model_path), vae=vae, torch_dtype=torch_dtype)
    return pipe


def main() -> int:
    p = argparse.ArgumentParser(description="Wan 2.2 video inference runner")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--num-frames", dest="num_frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="Override auto-detected device: cuda | mps | cpu.")
    p.add_argument("--dtype", default=None,
                   help="Override auto-detected dtype: bfloat16 | float16 | float32.")
    p.add_argument("--vae-device", default="auto",
                   help="Device for the VAE decode: auto | cpu | cuda | mps. "
                        "'auto' keeps it on the main device except on AMD ROCm, "
                        "where it defaults to cpu (unstable MIOpen conv kernels "
                        "on new archs like gfx1201).")
    p.add_argument("--init-image", default=None, type=Path,
                   help="Reference image: switches to image-to-video "
                        "(the image becomes the first frame).")
    p.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false",
                   help="Disable model CPU offload (keep the whole pipeline on "
                        "the GPU). Offload is on by default so the 5B fits.")
    p.set_defaults(cpu_offload=True)
    args = p.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"[wan] model path does not exist: {model_path}", file=sys.stderr)
        return 2

    device, default_dtype = _select_device()
    if args.device:
        device = args.device
    dtype_name = args.dtype or default_dtype
    print(f"[wan] device={device} dtype={dtype_name}", file=sys.stderr)

    init_image = None
    if args.init_image is not None:
        init_path = args.init_image.expanduser().resolve()
        if not init_path.is_file():
            print(f"[wan] init image not found: {init_path}", file=sys.stderr)
            return 2
        from diffusers.utils import load_image
        init_image = load_image(str(init_path))
        # Wan's image-to-video conditions on the supplied first frame at the
        # target resolution — resize so the aspect matches the output.
        if init_image.size != (args.width, args.height):
            init_image = init_image.resize((args.width, args.height))

    import torch
    from diffusers.utils import export_to_video

    pipe = _load_pipeline(
        model_path, dtype_name, image_to_video=init_image is not None)

    # VAE memory/stability hardening: tile + slice the decode so conv
    # workloads (and MIOpen kernels on AMD) stay small.
    for fn in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, fn)()
        except Exception:  # noqa: BLE001 — best-effort; diffusers versions vary
            pass

    # The 5B model is memory-heavy for a single consumer card. Prefer
    # model CPU offload (streams components onto the GPU as needed) unless
    # the operator disabled it. Falls back to a plain .to(device).
    vae_device = _resolve_vae_device(args.vae_device, device)
    if args.cpu_offload and device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("[wan] model CPU offload enabled", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[wan] cpu offload unavailable ({e}); moving to {device}",
                  file=sys.stderr)
            pipe.to(device)
    else:
        pipe.to(device)
        if vae_device != device:
            print(f"[wan] VAE decode on {vae_device} (transformer on {device})",
                  file=sys.stderr)
            pipe.vae.to(vae_device)

    generator = None
    if args.seed is not None:
        try:
            generator = torch.Generator(device).manual_seed(int(args.seed))
        except (RuntimeError, ValueError):
            generator = torch.Generator("cpu").manual_seed(int(args.seed))

    mode = "image-to-video" if init_image is not None else "text-to-video"
    print(
        f"[wan] generating {args.width}x{args.height} frames={args.num_frames} "
        f"steps={args.steps} cfg={args.guidance} seed={args.seed} ({mode})",
        file=sys.stderr,
    )

    call_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    if init_image is not None:
        call_kwargs["image"] = init_image

    result = pipe(**call_kwargs)
    frames = result.frames[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=int(args.fps))
    print(f"[wan] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
