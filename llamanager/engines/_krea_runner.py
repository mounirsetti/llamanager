"""Krea 2 Turbo inference runner."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_REPO = "krea/Krea-2-Turbo"


def _select_device() -> tuple[str, str]:
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


def _load_pipeline(model_path: Path, gguf: Path | None, dtype_name: str,
                   base_repo: str):
    torch_dtype = _torch_dtype(dtype_name)
    if gguf is None:
        try:
            from diffusers import Krea2Pipeline
            return Krea2Pipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
            )
        except (ImportError, AttributeError):
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
            )

    try:
        from diffusers import (
            DiffusionPipeline,
            GGUFQuantizationConfig,
            QwenImageTransformer2DModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Krea GGUF needs a recent diffusers build with "
            "GGUFQuantizationConfig and QwenImageTransformer2DModel. "
            "Use the Diffusion engines page to install/update dependencies."
        ) from exc

    quant_config = GGUFQuantizationConfig(compute_dtype=torch_dtype)
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(gguf),
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
    )
    return DiffusionPipeline.from_pretrained(
        base_repo,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )


def _apply_lora(pipe, lora: str, scale: float | None) -> None:
    if not lora:
        return
    kwargs = {"adapter_name": "krea_lora"}
    try:
        pipe.load_lora_weights(lora, **kwargs)
    except TypeError:
        pipe.load_lora_weights(lora)
        kwargs["adapter_name"] = "default"
    if scale is None:
        return
    try:
        pipe.set_adapters([kwargs["adapter_name"]], adapter_weights=[float(scale)])
    except Exception as exc:  # noqa: BLE001 - older diffusers APIs vary here
        print(f"[krea] warning: could not set LoRA scale {scale}: {exc}",
              file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="Krea 2 Turbo runner")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--gguf", default=None, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--true-cfg", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--base-repo", default=BASE_REPO)
    p.add_argument("--lora", default="")
    p.add_argument("--lora-scale", type=float, default=None)
    args = p.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"[krea] model path does not exist: {model_path}", file=sys.stderr)
        return 2
    gguf = args.gguf.expanduser().resolve() if args.gguf else None
    if gguf is not None and not gguf.is_file():
        print(f"[krea] GGUF file does not exist: {gguf}", file=sys.stderr)
        return 2

    device, default_dtype = _select_device()
    if args.device:
        device = args.device
    dtype_name = args.dtype or default_dtype
    mode = f"gguf={gguf.name}" if gguf is not None else "original"
    print(f"[krea] device={device} dtype={dtype_name} {mode}", file=sys.stderr)

    import torch
    pipe = _load_pipeline(model_path, gguf, dtype_name, args.base_repo)
    if args.lora:
        print(f"[krea] loading LoRA {args.lora} scale={args.lora_scale}",
              file=sys.stderr)
        _apply_lora(pipe, args.lora, args.lora_scale)
    pipe.to(device)
    for fn in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, fn)()
        except Exception:
            pass

    generator = None
    if args.seed is not None:
        try:
            generator = torch.Generator(device).manual_seed(int(args.seed))
        except (RuntimeError, ValueError):
            generator = torch.Generator("cpu").manual_seed(int(args.seed))

    print(
        f"[krea] generating {args.width}x{args.height} "
        f"steps={args.steps} cfg={args.guidance} seed={args.seed}",
        file=sys.stderr,
    )
    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt or None,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": generator,
    }
    if args.true_cfg is not None:
        kwargs["true_cfg_scale"] = args.true_cfg
    result = pipe(**kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(str(args.output))
    print(f"[krea] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
