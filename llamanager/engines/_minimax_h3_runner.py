"""MiniMax-H3 inference runner — invoked by ``minimax_h3.py`` as a subprocess.

MiniMax-H3 generates video **and its soundtrack** from one denoising loop:
a single transformer steps a packed sequence holding the text conditioning,
the keyframe/reference latents, the video latents and the audio latents at
once. There is no vocoder and no separate audio pass.

Three points shape this runner:

* **Modular Diffusers only.** Upstream ships MiniMax-H3 as ``ModularPipeline``
  blocks with no ``DiffusionPipeline`` half, so we build the pipeline with
  ``ModularPipeline.from_pretrained(..., workflow=...)`` and call
  ``load_components``. Selecting the workflow up front matters: it is what
  keeps a ``t2va``/``fl2va`` run from pulling the second 61.7 GB transformer
  partition it will never use.

* **Guidance is baked in.** Both transformer partitions are guidance-distilled
  — no guider, no ``negative_prompt``, no ``guidance_scale``, one forward pass
  per step. The adapter maps our generic guidance/negative-prompt profile
  fields to nothing here on purpose.

* **It is very large, and slow to load.** 61.7 GB of transformer plus
  62.1 GB of Qwen3-VL conditioner in bfloat16. Quantised to NF4 the weights
  fit a 32 GB card (see ``plan_memory``), but *getting them there* is the
  real cost: NF4 does not exist on disk, so every tensor is materialised in
  bf16 and quantised on the way in.

  Measured on gfx1201 / ROCm 7.2 / bitsandbytes 0.50.1, loading the
  conditioner alone with ``device_map={"": 0}``:

      284 of 1058 tensors in 5 min  =  5.8 s/tensor  ->  ~75 min projected
      host RSS climbed to 26.5 GB;  VRAM froze at 18.0 GB after 16 s

  So on this stack a single generation would spend hours in load before the
  first denoising step, and llamanager's one-shot runner pays that per
  request. Passing ``max_memory`` makes it strictly worse: accelerate sizes
  layers by their *unquantised* dtype, decides the model cannot fit the
  budget, and parks the remainder on CPU in bf16 — that path reached 50+ GB
  of host RAM and thrashed. Hence ``device_map={"": device}`` and no
  ``max_memory`` anywhere in this file.

  We size the request up front and refuse with the reason rather than
  thrashing — the same contract as the Krea and Wan runners.

Progress: diffusers' samplers use tqdm, which writes "N/M" lines to stderr;
the parent adapter's ``parse_progress`` keys off that.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# MiniMax-H3's own clock. Frames are snapped up to the next 17n+5 the video
# VAE can decode, and the resulting duration has to land in 5..15 seconds.
NATIVE_FPS = 24
_FRAME_BASE = 17
_FRAME_OFFSET = 5
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0

# bfloat16 component sizes from the upstream model card, used for sizing.
_TRANSFORMER_GIB = 61.7
_CONDITIONER_GIB = 62.1

# Weight-only quantisation options, with the size factor each actually
# achieved. Measured on gfx1201 / ROCm 7.2 / torch 2.10 against one 8192x8192
# bf16 linear (128.0 MiB resident), counting real
# ``torch.cuda.memory_allocated`` — the ratios are what carry to the model:
#
#   backend                     resident    ratio   forward
#   bf16 baseline               128.0 MiB    1.00x   0.35 ms
#   bitsandbytes NF4             33.0 MiB    3.88x   0.30 ms   <- best on ROCm
#   bitsandbytes int8            64.0 MiB    2.00x   0.20 ms
#   torchao int8 weight-only     64.0 MiB    2.00x   1.12 ms
#   torchao fp8 weight-only      64.0 MiB    2.00x   2.74 ms
#   torchao int8 dynamic-act     64.0 MiB    2.00x  31.42 ms   <- never offer
#   torchao int4                 unavailable ("Requires mslk >= 1.0.0")
#
# The headline: bitsandbytes has real ROCm kernels, so NF4 is both the
# smallest option AND faster than bf16 here, while torchao falls back to a
# slow dequant path (its cpp extensions need torch >= 2.11). torchao's int4
# has no ROCm kernels at all — which is why NF4, not torchao, is what makes
# a 4-bit run possible on AMD.
QUANT_FACTORS: dict[str, float] = {
    "none": 1.0,
    "int8": 0.5,        # torchao int8 weight-only
    "fp8": 0.5,         # torchao fp8 weight-only
    "int4": 0.25,       # torchao int4 (CUDA only)
    "nf4": 1.0 / 3.88,  # bitsandbytes 4-bit NF4, measured
    "bnb-int8": 0.5,    # bitsandbytes LLM.int8()
}

# Which library serves each option. bitsandbytes quantises at load time and
# pins the result to a device, so those two take a different code path.
_BNB_QUANTS = {"nf4", "bnb-int8"}
# Quantisation only shrinks the two large components; the VAEs stay bf16.
# Measured on the actual checkpoint: vae/ is 9.8 GB and audio_vae/ 0.58 GB.
# The earlier 5.5 estimate was low and made the fit look roomier than it is —
# NF4 split residency really peaks near 26.4 GB, not 21.5, on a 31.9 GB card.
_VAE_GIB = 10.4


def snap_num_frames(num_frames: int) -> int:
    """Snap ``num_frames`` up to the next ``17n + 5`` the video VAE decodes."""
    n = max(0, (int(num_frames) - _FRAME_OFFSET + _FRAME_BASE - 1) // _FRAME_BASE)
    return n * _FRAME_BASE + _FRAME_OFFSET


def _select_device() -> tuple[str, str]:
    """Pick the best available device + dtype.

    MiniMax-H3 is released in bfloat16. MPS has no bfloat16, so Apple falls
    back to float16; CPU stays float32 because bf16 there is emulated.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


def _accelerator_gib(device: str) -> float:
    """Usable accelerator memory in GiB, or 0.0 when it can't be determined."""
    import torch
    if device == "cuda":
        try:
            return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        except Exception:  # noqa: BLE001 — treated as "unknown"
            return 0.0
    if device == "mps":
        try:
            return torch.mps.recommended_max_memory() / 2 ** 30
        except Exception:  # noqa: BLE001
            return 0.0
    return 0.0


def _host_ram_gib() -> float:
    """Total host RAM in GiB, or 0.0 when it can't be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 0.0


def quant_config(quantize: str, *, for_transformers: bool = False):
    """Build the torchao config for ``quantize``, or None for no quantisation.

    ``for_transformers`` selects the ``transformers`` copy of TorchAoConfig,
    which the Qwen3-VL conditioner is loaded through; the diffusers copy is
    used for the transformer. Raises ImportError/AttributeError when the
    option is not available in the installed torchao — callers probe first.
    """
    if quantize == "none":
        return None
    if quantize in _BNB_QUANTS:
        import torch
        if for_transformers:
            from transformers import BitsAndBytesConfig as _Bnb
        else:
            from diffusers import BitsAndBytesConfig as _Bnb
        if quantize == "nf4":
            return _Bnb, dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
        return _Bnb, dict(load_in_8bit=True)
    if for_transformers:
        from transformers import TorchAoConfig as _Cfg
    else:
        from diffusers import TorchAoConfig as _Cfg
    import torchao.quantization as q
    if quantize == "int8":
        return _Cfg, q.Int8WeightOnlyConfig(version=2)
    if quantize == "fp8":
        return _Cfg, q.Float8WeightOnlyConfig()
    if quantize == "int4":
        return _Cfg, q.Int4WeightOnlyConfig()
    raise ValueError(f"unknown quantize option: {quantize!r}")


def probe_quant(quantize: str, device: str) -> tuple[bool, str]:
    """Can this backend actually run ``quantize``? Checked on a tiny layer.

    torchao advertises configs it has no kernels for on every backend — int4
    raises "Requires mslk >= 1.0.0" on ROCm, for instance. Finding that out
    after pulling 140 GB of weights is the expensive way to learn it, so we
    quantize one 256x256 linear first and actually run it.
    """
    if quantize == "none":
        return True, "no quantisation"
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"

    if quantize in _BNB_QUANTS:
        # bitsandbytes builds its own Linear subclasses; exercise one.
        try:
            import bitsandbytes as bnb
            if quantize == "nf4":
                lin = bnb.nn.Linear4bit(256, 256, bias=False,
                                        compute_dtype=torch.bfloat16,
                                        quant_type="nf4")
                lin.weight = bnb.nn.Params4bit(
                    torch.randn(256, 256, dtype=torch.bfloat16),
                    requires_grad=False, quant_type="nf4")
                dtype = torch.bfloat16
            else:
                lin = bnb.nn.Linear8bitLt(256, 256, bias=False,
                                          has_fp16_weights=False)
                lin.weight = bnb.nn.Int8Params(
                    torch.randn(256, 256, dtype=torch.float16),
                    requires_grad=False)
                dtype = torch.float16
            lin = lin.to(device).eval()
            with torch.no_grad():
                lin(torch.randn(2, 256, device=device, dtype=dtype))
            return True, f"{quantize} (bitsandbytes) verified on {device}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {str(exc)[:120]}"

    try:
        from torchao.quantization import quantize_
        _, cfg = quant_config(quantize)
    except Exception as exc:  # noqa: BLE001 — any import failure means "no"
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"
    try:
        lin = torch.nn.Linear(256, 256, bias=False).to(device, torch.bfloat16).eval()
        quantize_(lin, cfg)
        with torch.no_grad():
            lin(torch.randn(2, 256, device=device, dtype=torch.bfloat16))
        return True, f"{quantize} (torchao) verified on {device}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def weight_gib(quantize: str) -> tuple[float, float]:
    """Return (transformer_gib, conditioner_gib) at this precision."""
    factor = QUANT_FACTORS.get(quantize, 1.0)
    return _TRANSFORMER_GIB * factor, _CONDITIONER_GIB * factor


def plan_memory(device: str, quantize: str, offload: str,
                split: bool = True) -> tuple[str, str]:
    """Decide whether this machine can host the run.

    Returns ``(verdict, detail)``. Two dimensions matter:

    * **Quantisation** shrinks the two large components (see QUANT_FACTORS).
    * **split residency** (encode the prompt, free the conditioner, then load
      the transformer) makes the resident peak ``max(transformer,
      conditioner)`` instead of their sum — the same trick the Krea and Wan
      runners use. Upstream's own two-card example splits the pipeline the
      same way, so this is a supported shape rather than a hack.

    Together those decide whether the run can stay on the accelerator at all;
    only when it cannot do we consider parking weights in host RAM.
    """
    accel = _accelerator_gib(device)
    ram = _host_ram_gib()
    transformer, conditioner = weight_gib(quantize)
    peak = max(transformer, conditioner) if split else transformer + conditioner
    peak += _VAE_GIB          # the VAEs stay bf16 and stay resident
    shape = "split" if split else "joint"

    if offload == "none":
        # Everything on the accelerator — the only fully-GPU path.
        if accel and peak > accel * 0.92:
            hint = ""
            if quantize == "none":
                hint = " Try --quantize int8 (halves both components)."
            elif split is False:
                hint = " Try --residency split."
            return "refuse", (
                f"{shape} residency at {quantize} needs ~{peak:.0f} GiB on the "
                f"accelerator, which has {accel:.1f} GiB.{hint} Otherwise use "
                f"--offload block to stream from host RAM, or a larger card."
            )
        return "ok", (f"{peak:.0f} GiB resident on {device} "
                      f"({accel:.1f} GiB available, {shape} residency, {quantize})")

    # Offloaded: the weights live in host RAM and stream onto the device.
    hosted = transformer + conditioner
    if ram and hosted > ram * 0.92:
        return "refuse", (
            f"offload={offload} parks ~{hosted:.0f} GiB of {quantize} weights "
            f"in host RAM, but this machine has {ram:.1f} GiB. Either fit it "
            f"on the accelerator (needs ~{peak:.0f} GiB with {shape} "
            f"residency) or add host RAM."
        )
    return "ok", (f"~{hosted:.0f} GiB of {quantize} weights in host RAM "
                  f"({ram:.1f} GiB total), streamed onto {device} "
                  f"({accel:.1f} GiB)")


# Layers left in bf16, from upstream's own recipe: projections, embeddings
# and norms cost almost nothing to keep at full precision and quantising them
# measurably hurts quality.
_KEEP_BF16_TRANSFORMER = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
_KEEP_BF16_CONDITIONER = [
    "model.visual", "model.language_model.embed_tokens",
    "model.language_model.norm", "lm_head",
]


def load_quantized_transformer(repo: str, workflow: str, dtype, quantize: str,
                               device: str = ""):
    """Load just the denoising transformer at ``quantize`` precision.

    Quantising during ``from_pretrained`` matters: it never materialises the
    bf16 tensor on the accelerator, so peak load memory tracks the quantised
    size rather than spiking to 61.7 GB first.
    """
    from diffusers import MiniMaxH3Transformer3DModel
    subfolder = "transformer_ref" if workflow == "ref2va" else "transformer"
    kwargs = dict(dtype=dtype)
    if quantize in _BNB_QUANTS:
        cfg_cls, opts = quant_config(quantize)
        kwargs["quantization_config"] = cfg_cls(
            llm_int8_skip_modules=list(_KEEP_BF16_TRANSFORMER), **opts)
        # bitsandbytes quantises onto a device as it loads and the result
        # cannot be moved afterwards, so the placement has to be declared here
        # rather than with a later ``.to(device)``.
        kwargs["device_map"] = {"": device} if device else "auto"
    elif quantize != "none":
        cfg_cls, inner = quant_config(quantize)
        kwargs["quantization_config"] = cfg_cls(
            inner, modules_to_not_convert=list(_KEEP_BF16_TRANSFORMER))
        kwargs["low_cpu_mem_usage"] = False
    else:
        kwargs["low_cpu_mem_usage"] = False
    print(f"[minimax-h3] loading {subfolder} ({quantize})",
          file=sys.stderr, flush=True)
    return MiniMaxH3Transformer3DModel.from_pretrained(repo, subfolder=subfolder,
                                                       **kwargs)


def load_quantized_conditioner(repo: str, dtype, quantize: str,
                               device: str = ""):
    """Load just the Qwen3-VL conditioner at ``quantize`` precision."""
    from transformers import Qwen3VLForConditionalGeneration
    kwargs = dict(dtype=dtype)
    if quantize in _BNB_QUANTS:
        cfg_cls, opts = quant_config(quantize, for_transformers=True)
        kwargs["quantization_config"] = cfg_cls(
            llm_int8_skip_modules=list(_KEEP_BF16_CONDITIONER), **opts)
        kwargs["device_map"] = {"": device} if device else "auto"
    elif quantize != "none":
        cfg_cls, inner = quant_config(quantize, for_transformers=True)
        kwargs["quantization_config"] = cfg_cls(
            inner, modules_to_not_convert=list(_KEEP_BF16_CONDITIONER))
    print(f"[minimax-h3] loading text_encoder ({quantize})",
          file=sys.stderr, flush=True)
    return Qwen3VLForConditionalGeneration.from_pretrained(
        repo, subfolder="text_encoder", **kwargs)


def _apply_quant(pipe, repo: str, workflow: str, dtype, quantize: str,
                 device: str = ""):
    """Swap both large components for quantised copies before load_components."""
    if quantize == "none":
        return
    pipe.update_components(
        transformer=load_quantized_transformer(repo, workflow, dtype, quantize,
                                               device),
        text_encoder=load_quantized_conditioner(repo, dtype, quantize, device),
    )


def _apply_offload(pipe, device: str, offload: str) -> None:
    """Wire the chosen offload strategy."""
    import torch
    from diffusers.hooks import apply_group_offloading

    if offload == "none":
        pipe.transformer.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.audio_vae.to(device)
        return

    # Streamed offload needs frozen, pinnable weights.
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    common = dict(onload_device=torch.device(device),
                  offload_device=torch.device("cpu"), use_stream=(offload == "block"))
    if offload == "block":
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **common)
        apply_group_offloading(pipe.text_encoder.model,
                               offload_type="leaf_level", **common)
        # The VAEs are small enough to stay resident, which keeps decode fast.
        pipe.vae.to(device)
        pipe.audio_vae.to(device)
    else:  # "leaf" — the 12-16 GB recipe, VAE offloaded too, no stream
        pipe.transformer.enable_group_offload(
            offload_type="leaf_level", **common)
        apply_group_offloading(pipe.text_encoder.model,
                               offload_type="leaf_level", **common)
        apply_group_offloading(pipe.vae, offload_type="leaf_level", **common)
        pipe.audio_vae.to(device)


def _vram(device: str) -> str:
    import torch
    if device != "cuda":
        return ""
    free_b, total_b = torch.cuda.mem_get_info()
    return f"{(total_b - free_b) / 2 ** 30:.1f}/{total_b / 2 ** 30:.1f} GiB"


def run_split(repo: str, workflow: str, dtype, quantize: str, device: str,
              prompt: str, call_kwargs: dict):
    """Encode the prompt, free the conditioner, then load the transformer.

    The conditioner and the denoiser are never needed at the same moment, so
    running them in sequence makes the resident peak ``max(...)`` of the two
    rather than their sum. Upstream splits the pipeline exactly this way for
    its two-card recipe — popping the ``text_encoder`` sub-block into its own
    pipeline whose output ``state`` feeds the rest — and we reuse that shape
    on a single device, freeing between the halves.

    This is what makes a quantised MiniMax-H3 fit on one card without parking
    weights in host RAM.
    """
    import gc

    import torch
    from diffusers import ModularPipeline

    blocks = ModularPipeline.from_pretrained(repo).blocks.get_workflow(workflow)

    # --- half 1: the conditioner alone ---------------------------------
    conditioner = blocks.sub_blocks.pop("text_encoder").init_pipeline(repo)
    if quantize != "none":
        conditioner.update_components(
            text_encoder=load_quantized_conditioner(repo, dtype, quantize, device))
    conditioner.load_components(dtype=dtype)
    if quantize not in _BNB_QUANTS:
        # bitsandbytes already placed it via device_map and refuses .to().
        conditioner.text_encoder.to(device)
    print(f"[minimax-h3] conditioner resident; VRAM {_vram(device)}",
          file=sys.stderr, flush=True)
    with torch.no_grad():
        state = conditioner(prompt=prompt)

    # ``to("meta")`` drops the storage outright; a plain del would leave tens
    # of GB sitting in host RAM until the GC happened to run.
    try:
        # bitsandbytes params reject a device move; dropping the reference and
        # emptying the cache is what frees them.
        if quantize not in _BNB_QUANTS:
            conditioner.text_encoder.to("meta")
    except Exception:  # noqa: BLE001 — best effort; the del below still frees
        pass
    conditioner.text_encoder = None
    del conditioner
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[minimax-h3] conditioner freed; VRAM {_vram(device)}",
          file=sys.stderr, flush=True)

    # --- half 2: everything else ---------------------------------------
    rest = blocks.init_pipeline(repo)
    if quantize != "none":
        rest.update_components(
            transformer=load_quantized_transformer(repo, workflow, dtype,
                                                   quantize, device))
    rest.load_components(dtype=dtype)
    if quantize not in _BNB_QUANTS:
        rest.transformer.to(device)
    rest.vae.to(device)
    rest.audio_vae.to(device)
    print(f"[minimax-h3] transformer+vae resident; VRAM {_vram(device)}",
          file=sys.stderr, flush=True)

    # The prompt is already encoded into ``state``; passing it again would
    # ask a pipeline with no conditioner to encode it a second time.
    return rest(state=state, **call_kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="MiniMax-H3 video+audio runner")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--width", type=int, default=0,
                   help="Multiple of 32. 0 lets the model pick its own canvas.")
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--num-frames", dest="num_frames", type=int, default=124)
    p.add_argument("--fps", type=int, default=NATIVE_FPS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | mps | cpu")
    p.add_argument("--dtype", default=None,
                   help="bfloat16 | float16 | float32")
    p.add_argument("--init-image", default=None, type=Path,
                   help="First keyframe — selects the fl2va workflow.")
    p.add_argument("--last-image", default=None, type=Path,
                   help="Final keyframe. May be passed alone.")
    p.add_argument("--quantize", default="int8",
                   choices=sorted(QUANT_FACTORS),
                   help="Weight-only quantisation via torchao. int8/fp8 halve "
                        "the weights; int4 quarters them but needs kernels "
                        "that are not on every backend. Probed before load.")
    p.add_argument("--offload", default="none",
                   choices=["none", "block", "leaf"],
                   help="Where the weights live between steps. 'none' keeps "
                        "everything on the accelerator.")
    p.add_argument("--residency", default="split", choices=["split", "joint"],
                   help="'split' encodes the prompt, frees the conditioner, "
                        "then loads the transformer, so peak is max() of the "
                        "two rather than their sum.")
    p.add_argument("--repo", default="MiniMaxAI/MiniMax-H3",
                   help="Fallback repo id when --model_path isn't a local tree.")
    args = p.parse_args()

    model_path = args.model_path.expanduser()
    # Modular Diffusers resolves component subfolders from one root: a local
    # checkout when we have one, otherwise the hub id.
    source = str(model_path) if (model_path / "modular_model_index.json").is_file() \
        else args.repo
    if source == args.repo and not model_path.exists():
        print(f"[minimax-h3] no local checkout at {model_path}; "
              f"resolving {args.repo} from the hub", file=sys.stderr)

    device, default_dtype = _select_device()
    if args.device:
        device = args.device
    dtype_name = args.dtype or default_dtype

    # Ask the backend whether it can really run this quantisation before
    # committing to a multi-hundred-GB load. torchao advertises configs it has
    # no kernels for (int4 raises "Requires mslk >= 1.0.0" on ROCm), and
    # finding that out after the download is the expensive way to learn it.
    quantize = args.quantize
    if quantize != "none":
        ok, why = probe_quant(quantize, device)
        if not ok:
            print(f"[minimax-h3] {quantize} unavailable on {device} ({why})",
                  file=sys.stderr)
            # Fall back smallest-first: a 4-bit option is what decides whether
            # this model fits at all, so try NF4 (which has real ROCm kernels)
            # before dropping to a format that merely halves the weights.
            for candidate in ("nf4", "int8", "bnb-int8"):
                if candidate == quantize:
                    continue
                cand_ok, cand_why = probe_quant(candidate, device)
                if cand_ok:
                    print(f"[minimax-h3] falling back to {candidate}",
                          file=sys.stderr)
                    quantize = candidate
                    break
            else:
                print("[minimax-h3] no quantisation backend available; "
                      "using bf16 weights", file=sys.stderr)
                quantize = "none"
        else:
            print(f"[minimax-h3] {why}", file=sys.stderr)

    offload = args.offload
    if device == "cpu":
        offload = "none"
    split = args.residency == "split"

    verdict, detail = plan_memory(device, quantize, offload, split)
    print(f"[minimax-h3] device={device} dtype={dtype_name} "
          f"quantize={quantize} offload={offload} residency={args.residency}",
          file=sys.stderr)
    if verdict == "refuse":
        print(f"[minimax-h3] refusing: {detail}", file=sys.stderr)
        return 2
    print(f"[minimax-h3] memory plan: {detail}", file=sys.stderr)
    # The VRAM plan can pass while the *load* is still impractical: measured
    # 5.8 s/tensor for the conditioner on ROCm, i.e. over an hour before the
    # first step. Say so rather than letting it look like a hang.
    print("[minimax-h3] note: weights are quantised on the way in; on ROCm "
          "this measured ~5.8 s/tensor (~75 min for the conditioner alone). "
          "Expect a long load before the first denoising step.",
          file=sys.stderr)

    frames = snap_num_frames(args.num_frames)
    seconds = frames / NATIVE_FPS
    if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
        print(f"[minimax-h3] {args.num_frames} frames snaps to {frames} "
              f"({seconds:.2f}s at {NATIVE_FPS}fps); MiniMax-H3 generates "
              f"{MIN_SECONDS:.0f}-{MAX_SECONDS:.0f}s clips.", file=sys.stderr)
        return 2
    if frames != args.num_frames:
        print(f"[minimax-h3] frames {args.num_frames} -> {frames} (17n+5)",
              file=sys.stderr)
    for label, value in (("width", args.width), ("height", args.height)):
        if value and value % 32:
            print(f"[minimax-h3] {label}={value} must be a multiple of 32",
                  file=sys.stderr)
            return 2

    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    keyframe = load_image(str(args.init_image)) if args.init_image else None
    last_keyframe = load_image(str(args.last_image)) if args.last_image else None
    workflow = "fl2va" if (keyframe is not None or last_keyframe is not None) else "t2va"
    print(f"[minimax-h3] workflow={workflow} frames={frames} steps={args.steps}",
          file=sys.stderr)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype_name]

    generator = torch.Generator("cpu").manual_seed(int(args.seed)) \
        if args.seed is not None else None

    call_kwargs = {
        "num_frames": frames,
        "num_inference_steps": args.steps,
        "generator": generator,
        "output": ["videos", "audio", "sampling_rate"],
    }
    if args.width and args.height:
        call_kwargs["height"] = args.height
        call_kwargs["width"] = args.width
    if keyframe is not None:
        call_kwargs["image"] = keyframe
    if last_keyframe is not None:
        call_kwargs["last_image"] = last_keyframe

    if split and offload == "none":
        # Sequential residency: conditioner, free, then transformer. This is
        # the only path that keeps every byte on the accelerator.
        results = run_split(source, workflow, torch_dtype, quantize, device,
                            args.prompt, call_kwargs)
    else:
        # Joint pipeline. With offload set, the ComponentsManager moves whole
        # components on and off and group offload streams the transformer's
        # blocks — both of which mean weights live in host RAM.
        if offload == "none":
            pipe = ModularPipeline.from_pretrained(source, workflow=workflow)
        else:
            manager = ComponentsManager()
            pipe = ModularPipeline.from_pretrained(
                source, workflow=workflow, components_manager=manager)
        _apply_quant(pipe, source, workflow, torch_dtype, quantize, device)
        pipe.load_components(dtype=torch_dtype)
        _apply_offload(pipe, device, offload)
        results = pipe(prompt=args.prompt, **call_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Video and audio come out of the same loop but as separate outputs;
    # muxing them into one file is the caller's job.
    encode_video(
        results["videos"][0],
        fps=NATIVE_FPS,
        output_path=str(args.output),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    print(f"[minimax-h3] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
