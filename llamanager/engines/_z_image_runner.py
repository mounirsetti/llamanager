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
import os
import sys
from pathlib import Path


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


def _load_pipeline(model_path: Path, dtype_name: str):
    """Load the pipeline. Tries ``ZImagePipeline`` directly first; falls
    back to ``DiffusionPipeline.from_pretrained`` so models with a
    ``diffusers/`` subfolder layout (Z-Anime) also work via
    ``--subfolder diffusers`` on the command line."""
    import torch
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_name]

    try:
        from diffusers import ZImagePipeline
        return ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
        )
    except (ImportError, AttributeError):
        # Older / fork builds may not expose ZImagePipeline yet — try
        # the generic loader, which reads model_index.json and picks
        # the right pipeline class.
        from diffusers import DiffusionPipeline
        return DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
        )


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
    pipe = _load_pipeline(model_path, dtype_name)
    pipe.to(device)

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
