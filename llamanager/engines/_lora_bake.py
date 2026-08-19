"""Merge a LoRA into a diffusion transformer and write a quantised GGUF.

WHY THIS EXISTS. Applying a LoRA at request time costs far more than the
sampling it decorates. Measured on gfx1201 with Krea 2 Turbo at 1024x1024,
8 steps, and the 768-key Krea2-realism-V2 LoKr:

    Q6_K GGUF, no LoRA                29 s
    Q6_K GGUF + LoRA                 498 s   (448 s of it in step 1)
    fp8_scaled, no LoRA              365 s
    fp8_scaled + LoRA               1004 s

ComfyUI-GGUF dequantises every patched weight the first time it is used, and
the fp8 path is worse because a patch knocks the model off fp8 entirely and
expands it to bfloat16. No weight format avoids the work. Merging the LoRA
once, offline, does: the result is an ordinary model with nothing to patch,
so it generates at exactly the speed of the base it was merged into.

WHY IT GOES THROUGH COMFYUI. These weights are not always plain LoRA pairs —
this one is a LyCORIS LoKr, whose delta is a Kronecker product of two factors
with an alpha scale, and other packs ship LoHa, GLoRA or DoRA. Rather than
reimplement each decomposition, this applies patches with
``comfy.lora.calculate_weight``, the same code the live path uses, so a baked
model matches what the runtime patch would have produced.

QUANTISATION. The Python ``gguf`` package can write F32/F16/BF16/Q8_0 but no
K-quant; those need city96's patched llama.cpp. Q8_0 is therefore the default
target — slightly larger than a Q6_K and slightly better, not worse.

Run it with the ComfyUI venv's python, which is where torch and comfy live.
On AMD, the ROCm libraries must be on ``LD_LIBRARY_PATH`` before the process
starts — the dynamic linker reads it at exec, so the tool cannot set it for
itself, and ``import torch`` fails with ``libroctx64.so.4: cannot open
shared object file`` without it. ``gpu_detect.rocm_lib_dirs()`` returns the
directories to use (it returns [] off AMD, so the export is harmless)::

    LD_LIBRARY_PATH="$(python -c 'from llamanager.gpu_detect import \\
        rocm_lib_dirs; print(":".join(rocm_lib_dirs()))')" \\
    <comfy-venv>/bin/python -m llamanager.engines._lora_bake \\
        --comfy-repo /path/to/comfyui \\
        --base   .../diffusion_models/krea2_turbo_bf16.safetensors \\
        --lora   .../loras/Krea2-realism-V2.safetensors \\
        --out    .../diffusion_models/krea2_turbo-realism-v2-Q8_0.gguf \\
        --strength 1.0 --arch qwen_image
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Tensors below this many elements stay in full precision. Norms, biases and
# scales are tiny and quantising them costs accuracy for no meaningful space.
MIN_QUANTISED_ELEMENTS = 256 * 256


def should_quantise(shape: tuple[int, ...], block: int = 32) -> bool:
    """Is this tensor a candidate for Q8_0?

    Only 2-D tensors whose rows divide the block size, and only big ones.
    Everything else — 1-D norms, biases, odd-shaped embeddings — is written
    as F32, which is what the community quants of this model do too (their
    split is 167 F32 against 264 quantised).
    """
    if len(shape) != 2:
        return False
    if shape[-1] % block:
        return False
    return shape[0] * shape[1] >= MIN_QUANTISED_ELEMENTS


def lora_key_map(weight_keys: list[str]) -> dict[str, str]:
    """Map ComfyUI LoRA names to the weight keys they patch.

    ComfyUI names a diffusion-model patch ``diffusion_model.<key-without-
    .weight>``; ``comfy.lora.model_lora_keys_unet`` builds the same mapping
    from a live model object. Deriving it from the state dict keeps this a
    file-to-file tool: no model has to be constructed on a device first.
    """
    out: dict[str, str] = {}
    for key in weight_keys:
        if key.endswith(".weight"):
            out[f"diffusion_model.{key[: -len('.weight')]}"] = key
    return out


def _log(msg: str) -> None:
    print(f"[bake] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy-repo", required=True, type=Path,
                    help="ComfyUI checkout; its modules do the patch maths")
    ap.add_argument("--base", required=True, type=Path,
                    help="transformer to merge into (.safetensors)")
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--strength", type=float, required=True,
                    help="merge strength; the file is fixed at this value, "
                         "so state it rather than inheriting a default")
    ap.add_argument("--arch", required=True,
                    help="general.architecture for the GGUF. It is read only "
                         "by the loader's allowlist, but a wrong value there "
                         "means the file will not load at all "
                         "(qwen_image for Krea 2).")
    ap.add_argument("--qtype", default="Q8_0", choices=["Q8_0", "F16", "BF16"],
                    help="K-quants need city96's patched llama.cpp; these are "
                         "what the gguf writer can produce on its own.")
    args = ap.parse_args()

    sys.path.insert(0, str(args.comfy_repo))
    import numpy as np
    import torch
    import gguf
    import comfy.lora
    import comfy.utils

    if args.out.exists():
        _log(f"refusing to overwrite {args.out}")
        return 2

    t0 = time.time()
    _log(f"reading base {args.base.name}")
    # str(), not Path: comfy.utils.load_torch_file calls .lower() on it.
    base = comfy.utils.load_torch_file(str(args.base), safe_load=True)
    _log(f"{len(base)} tensors")

    lora_sd = comfy.utils.load_torch_file(str(args.lora), safe_load=True)
    key_map = lora_key_map(list(base.keys()))
    patches = comfy.lora.load_lora(lora_sd, key_map)
    if not patches:
        _log("no patch bound any weight — wrong LoRA for this model?")
        return 3
    _log(f"{len(patches)} of {len(base)} tensors are patched "
         f"(strength {args.strength})")

    writer = gguf.GGUFWriter(str(args.out), args.arch, use_temp_file=True)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    qtype = getattr(gguf.GGMLQuantizationType, args.qtype)
    patched = quantised = 0
    for i, (key, tensor) in enumerate(sorted(base.items())):
        weight = tensor.to(torch.float32)
        patch = patches.get(key)
        if patch is not None:
            # The live path's own maths: LoKr, LoHa, GLoRA and DoRA all land
            # here, so a baked weight equals the patched one it replaces.
            # (strength_patch, patch, strength_model, offset, function) —
            # strength_model is a multiplier on the base weight, so it is 1.0,
            # not None: calculate_weight multiplies by it unconditionally.
            weight = comfy.lora.calculate_weight(
                [(args.strength, patch, 1.0, None, None)], weight, key)
            patched += 1
        arr = weight.contiguous().numpy()
        if should_quantise(arr.shape) and args.qtype == "Q8_0":
            # No raw_shape: the writer derives the logical shape from the
            # quantised byte shape (6144 floats -> 6528 bytes at 34 per
            # 32-value block). Passing the logical shape makes it read 6144
            # as a byte count and reject the tensor.
            data = gguf.quants.quantize(arr, qtype)
            writer.add_tensor(key, data, raw_dtype=qtype)
            quantised += 1
        else:
            writer.add_tensor(key, arr.astype(np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)
        del weight, arr
        if (i + 1) % 100 == 0:
            _log(f"{i + 1}/{len(base)} tensors, {time.time() - t0:.0f}s")

    _log(f"writing {args.out.name}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = args.out.stat().st_size / 1e9
    _log(f"done in {time.time() - t0:.0f}s — {size_gb:.2f} GB, "
         f"{patched} patched, {quantised} quantised, "
         f"{len(base) - quantised} full precision")
    if patched != len(patches):
        _log(f"WARNING: {len(patches) - patched} patches never found their "
             "weight; the merge is incomplete")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
