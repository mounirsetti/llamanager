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
    if gguf is not None:
        # Verified 2026-07-05: the community GGUF quants use Krea 2's own
        # tensor layout, which diffusers cannot load — Krea2Transformer2DModel
        # has no single-file/GGUF loader (upstream through 0.39 + main), and
        # QwenImageTransformer2DModel matches zero keys. The quants are
        # ComfyUI-only artifacts.
        raise RuntimeError(
            "Krea 2 Turbo GGUF quants are not loadable via diffusers — "
            "use the original krea/Krea-2-Turbo checkpoint instead "
            "(profile weights setting: original)."
        )
    try:
        from diffusers import Krea2Pipeline
        return Krea2Pipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Krea 2 needs diffusers >= 0.39 (Krea2Pipeline). Update the "
            "engine dependencies on the Diffusion engines page."
        ) from exc


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
    p.add_argument("--init-image", default=None, type=Path,
                   help="Reference image: switches to img2img "
                        "(QwenImageImg2ImgPipeline).")
    p.add_argument("--strength", type=float, default=0.6,
                   help="Img2img denoise strength: 0 keeps the init image, "
                        "1 ignores it.")
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

    if args.init_image is not None:
        # No Krea2 img2img pipeline exists upstream (checked diffusers 0.39
        # and main, 2026-07-05). Fail fast instead of silently ignoring the
        # reference image.
        print("[krea] reference images are not supported: diffusers has no "
              "Krea2 img2img pipeline yet. Use Z-Image for img2img.",
              file=sys.stderr)
        return 2

    import torch
    pipe = _load_pipeline(model_path, gguf, dtype_name, args.base_repo)
    if args.lora:
        print(f"[krea] loading LoRA {args.lora} scale={args.lora_scale}",
              file=sys.stderr)
        _apply_lora(pipe, args.lora, args.lora_scale)
    # bf16 transformer (26 GB) + text encoder (9 GB) exceed a 32 GB card
    # together; sequential offload keeps one component on-GPU at a time.
    if device == "cuda":
        # At >= 1024² the math-SDPA fallback needs ~4 GB for the attention
        # matrix, which doesn't fit next to the resident 26 GB transformer.
        # Stream the weights instead (slower, but it completes).
        big = (args.width * args.height) > 768 * 768
        try:
            if big:
                print("[krea] >768² on 32 GB VRAM: using sequential cpu "
                      "offload (slower, avoids attention OOM)", file=sys.stderr)
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        except Exception as exc:  # noqa: BLE001 — fall back to plain .to()
            print(f"[krea] cpu offload unavailable ({exc}); using .to(cuda)",
                  file=sys.stderr)
            pipe.to(device)
    else:
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
    # Prefer memory-efficient SDPA: the math fallback materializes the full
    # attention matrix (~4 GB at 1024²), which doesn't fit next to the
    # 26 GB bf16 transformer on a 32 GB card.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdp_ctx = sdpa_kernel(
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION,
             SDPBackend.MATH])
    except Exception:  # noqa: BLE001 — older torch: run unwrapped
        import contextlib
        sdp_ctx = contextlib.nullcontext()
    with sdp_ctx:
        result = pipe(**kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(str(args.output))
    print(f"[krea] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Headroom is razor-thin with the 26 GB bf16 transformer resident;
    # expandable segments avoids fragmentation-induced OOM.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
