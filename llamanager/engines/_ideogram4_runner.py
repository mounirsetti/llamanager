"""Ideogram 4 inference runner."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _select_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_dtype(dtype_name: str):
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> int:
    p = argparse.ArgumentParser(description="Ideogram 4 runner")
    p.add_argument("--weights-repo", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--sampler-preset", default="V4_QUALITY_48")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quantization", choices=["fp8", "nf4"], default="fp8")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--magic-prompt", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--magic-prompt-model", default=None)
    p.add_argument("--magic-prompt-key", default=None)
    p.add_argument("--hive-text-key", default=None)
    p.add_argument("--hive-visual-key", default=None)
    p.add_argument("--warn-on-caption-issues", action="store_true")
    args = p.parse_args()

    try:
        import torch
        from ideogram4 import (
            DEFAULT_MAGIC_PROMPT, MAGIC_PROMPTS, PRESETS,
            Ideogram4Pipeline, Ideogram4PipelineConfig,
            aspect_ratio_from_size, moderate_image, moderate_prompt,
        )
    except ImportError as exc:
        print(
            "[ideogram4] missing official ideogram4 package. Install "
            "dependencies from the Diffusion engines page.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2

    # The official package resolves every weight file through
    # hf_hub_download(repo_id=...), which rejects local paths. When the
    # operator points --weights-repo at an on-disk folder (a downloaded
    # snapshot of ideogram-ai/ideogram-4-fp8), reroute those lookups to
    # plain filesystem joins and leave real repo ids untouched.
    if Path(args.weights_repo).expanduser().is_dir():
        from ideogram4 import pipeline_ideogram4 as _pi
        _real_hf_hub_download = _pi.hf_hub_download

        def _local_or_hub(*a, repo_id: str = "", filename: str = "", **kw):
            if a:  # tolerate positional (repo_id, filename) call styles
                repo_id = a[0]
                if len(a) > 1:
                    filename = a[1]
            root = Path(repo_id).expanduser()
            if root.is_dir():
                p = root / filename
                if not p.is_file():
                    raise FileNotFoundError(
                        f"[ideogram4] {filename} not found in local "
                        f"weights dir {root}")
                return str(p)
            return _real_hf_hub_download(repo_id=repo_id, filename=filename,
                                         **kw)

        _pi.hf_hub_download = _local_or_hub

    device = args.device or _select_device()
    dtype = _torch_dtype(args.dtype)
    magic_model = args.magic_prompt_model or DEFAULT_MAGIC_PROMPT
    magic_key = (
        args.magic_prompt_key
        or os.environ.get("MAGIC_PROMPT_API_KEY")
        or os.environ.get("IDEOGRAM_API_KEY")
    )
    hive_text_key = args.hive_text_key or os.environ.get("HIVE_TEXT_MODERATION_KEY")
    hive_visual_key = args.hive_visual_key or os.environ.get("HIVE_VISUAL_MODERATION_KEY")

    if hive_text_key:
        flags = moderate_prompt(args.prompt, hive_text_key)
        if flags:
            print(f"[ideogram4] prompt rejected by Hive: {flags}", file=sys.stderr)
            return 2
    else:
        print("[ideogram4] warning: Hive text moderation disabled", file=sys.stderr)

    prompt = args.prompt
    if args.magic_prompt:
        if not magic_key:
            print(
                "[ideogram4] magic prompt requires IDEOGRAM_API_KEY or "
                "MAGIC_PROMPT_API_KEY. Disable with profile args "
                "{\"magic_prompt\": false} if passing structured JSON.",
                file=sys.stderr,
            )
            return 2
        aspect = aspect_ratio_from_size(args.width, args.height)
        print(f"[ideogram4] expanding prompt via {magic_model} ({aspect})",
              file=sys.stderr)
        magic = MAGIC_PROMPTS[magic_model](api_key=magic_key)  # type: ignore[call-arg]
        prompt = magic.expand(args.prompt, aspect_ratio=aspect)

    preset = PRESETS[args.sampler_preset]
    weights_repo = str(Path(args.weights_repo).expanduser())
    print(
        f"[ideogram4] loading {weights_repo} quant={args.quantization} "
        f"device={device} dtype={args.dtype}",
        file=sys.stderr,
    )
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=weights_repo),
        device=device,
        dtype=dtype,
    )
    images = pipe(
        prompt,
        height=args.height,
        width=args.width,
        num_steps=preset.num_steps,
        guidance_schedule=preset.guidance_schedule,
        mu=preset.mu,
        std=preset.std,
        seed=args.seed,
        raise_on_caption_issues=not args.warn_on_caption_issues,
    )
    if hive_visual_key:
        flags = moderate_image(images[0], hive_visual_key)
        if flags:
            print(f"[ideogram4] output rejected by Hive: {flags}", file=sys.stderr)
            return 2
    else:
        print("[ideogram4] warning: Hive visual moderation disabled", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(args.output))
    print(f"[ideogram4] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
