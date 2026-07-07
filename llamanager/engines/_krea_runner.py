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


def _make_img2img_pipeline(pipe):
    """Build a Krea-2 img2img pipeline from a loaded ``Krea2Pipeline``.

    Upstream diffusers (through 0.39 + main) ships no ``Krea2Img2ImgPipeline``,
    but Krea 2 is a Qwen-Image-architecture model (same ``AutoencoderKLQwenImage``
    VAE, same packed-latent DiT), so img2img is the standard flow-matching
    recipe: VAE-encode the init image, add noise at a strength-derived starting
    sigma, then run the tail of the denoise loop. We subclass ``Krea2Pipeline``
    so prompt encoding, latent packing, position ids and the transformer call
    all stay byte-for-byte identical to text-to-image — only latent init and
    the timestep window change.
    """
    import numpy as np
    import torch
    from diffusers import Krea2Pipeline
    from diffusers.pipelines.krea2.pipeline_krea2 import (
        calculate_shift, retrieve_timesteps)
    from diffusers.pipelines.krea2.pipeline_output import Krea2PipelineOutput
    from diffusers.utils.torch_utils import randn_tensor

    def _retrieve_latents(encoder_output, generator):
        if hasattr(encoder_output, "latent_dist"):
            return encoder_output.latent_dist.sample(generator)
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        raise AttributeError("Could not access latents of VAE encoder output")

    class Krea2Img2ImgPipeline(Krea2Pipeline):
        def _encode_vae_image(self, image, generator):
            # image: (B, C, 1, H, W) pixel tensor in [-1, 1].
            image_latents = _retrieve_latents(self.vae.encode(image), generator)
            z = self.vae.config.z_dim
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, z, 1, 1, 1).to(image_latents.device, image_latents.dtype)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, z, 1, 1, 1).to(image_latents.device, image_latents.dtype)
            # Inverse of the decode-side ``latents / latents_std + latents_mean``.
            return (image_latents - latents_mean) * latents_std

        def get_timesteps(self, num_inference_steps, strength):
            init_timestep = min(num_inference_steps * strength, num_inference_steps)
            t_start = int(max(num_inference_steps - init_timestep, 0))
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
            return timesteps, max(num_inference_steps - t_start, 1)

        @torch.no_grad()
        def __call__(self, *, prompt, image, strength=0.6, negative_prompt=None,
                     height=1024, width=1024, num_inference_steps=8,
                     guidance_scale=1.0, generator=None, output_type="pil",
                     max_sequence_length=512):
            multiple = self.vae_scale_factor * self.patch_size
            if height % multiple or width % multiple:
                height = ((height + multiple - 1) // multiple) * multiple
                width = ((width + multiple - 1) // multiple) * multiple

            self._guidance_scale = guidance_scale
            self._attention_kwargs = None
            self._current_timestep = None
            self._interrupt = False
            device = self._execution_device
            batch_size = 1

            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt=prompt, device=device, num_images_per_prompt=1,
                max_sequence_length=max_sequence_length)
            negative_prompt_embeds = negative_prompt_embeds_mask = None
            if self.do_classifier_free_guidance:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                    prompt=[negative_prompt or ""], device=device,
                    num_images_per_prompt=1, max_sequence_length=max_sequence_length)

            # Timesteps first (img2img needs the starting sigma before latents).
            num_channels_latents = self.transformer.config.in_channels // (self.patch_size ** 2)
            latent_height = height // self.vae_scale_factor
            latent_width = width // self.vae_scale_factor
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = (latent_height // self.patch_size) * (latent_width // self.patch_size)
            if self.config.is_distilled:
                mu = 1.15
            else:
                mu = calculate_shift(
                    image_seq_len,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 6400),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15))
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
            timesteps, num_inference_steps = self.get_timesteps(
                len(timesteps), strength)
            latent_timestep = timesteps[:1].repeat(batch_size)

            # Init latents = VAE(image) noised to the starting sigma.
            init = self.image_processor.preprocess(
                image, height=height, width=width).to(device=device, dtype=torch.float32)
            init = init.unsqueeze(2)  # (B, C, 1, H, W) for the video-style VAE
            image_latents = self._encode_vae_image(
                init.to(self.vae.dtype), generator)[:, :, 0]  # (B, z, H', W')
            image_latents = image_latents.to(prompt_embeds.dtype)
            noise = randn_tensor(
                image_latents.shape, generator=generator, device=device,
                dtype=prompt_embeds.dtype)
            latents = self.scheduler.scale_noise(image_latents, latent_timestep, noise)
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, latent_height, latent_width)

            grid_height = height // (self.vae_scale_factor * self.patch_size)
            grid_width = width // (self.vae_scale_factor * self.patch_size)
            position_ids = self.prepare_position_ids(
                prompt_embeds.shape[1], grid_height, grid_width, device)

            with self.progress_bar(total=len(timesteps)) as progress_bar:
                for t in timesteps:
                    if self.interrupt:
                        continue
                    self._current_timestep = t
                    timestep = (t / self.scheduler.config.num_train_timesteps).expand(
                        latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latents, encoder_hidden_states=prompt_embeds,
                        timestep=timestep, position_ids=position_ids,
                        encoder_attention_mask=prompt_embeds_mask,
                        attention_kwargs=self.attention_kwargs, return_dict=False)[0]
                    if self.do_classifier_free_guidance:
                        neg = self.transformer(
                            hidden_states=latents,
                            encoder_hidden_states=negative_prompt_embeds,
                            timestep=timestep, position_ids=position_ids,
                            encoder_attention_mask=negative_prompt_embeds_mask,
                            attention_kwargs=self.attention_kwargs, return_dict=False)[0]
                        noise_pred = noise_pred + guidance_scale * (noise_pred - neg)
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False)[0]
                    progress_bar.update()

            self._current_timestep = None
            if output_type == "latent":
                return Krea2PipelineOutput(images=latents)
            latents = self._unpack_latents(latents, height, width).to(self.vae.dtype)
            z = self.vae.config.z_dim
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, z, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, z, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            decoded = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            images = self.image_processor.postprocess(decoded, output_type=output_type)
            self.maybe_free_model_hooks()
            return Krea2PipelineOutput(images=images)

    return Krea2Img2ImgPipeline(**pipe.components)


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

    init_image = None
    if args.init_image is not None:
        init_path = args.init_image.expanduser().resolve()
        if not init_path.is_file():
            print(f"[krea] init image not found: {init_path}", file=sys.stderr)
            return 2
        from PIL import Image
        init_image = Image.open(init_path).convert("RGB")

    import torch
    pipe = _load_pipeline(model_path, gguf, dtype_name, args.base_repo)
    if init_image is not None:
        # Upstream ships no Krea2Img2ImgPipeline; build one from the loaded
        # components (Krea 2 is Qwen-Image-architecture, so img2img is the
        # standard flow-matching recipe). See _make_img2img_pipeline.
        pipe = _make_img2img_pipeline(pipe)
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

    mode = f"img2img strength={args.strength}" if init_image is not None else "t2i"
    print(
        f"[krea] generating {args.width}x{args.height} "
        f"steps={args.steps} cfg={args.guidance} seed={args.seed} ({mode})",
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
    if init_image is not None:
        # The img2img subclass has a slimmer signature (no true_cfg passthrough)
        # and derives everything else from the init image.
        kwargs["image"] = init_image
        kwargs["strength"] = args.strength
    elif args.true_cfg is not None:
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
