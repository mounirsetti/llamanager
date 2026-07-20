"""llama-server subprocess lifecycle.

Spec §4.3 + §6.4. Owns one child process at a time. Start/stop/restart/
swap. Crash supervisor lives in supervisor.py and listens for unexpected
exits via the shared `wait_event`.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import runtime_state as rt
from . import exclusive as _exclusive
from .config import Config, Profile, detect_engine_for_path
from .db import DB
from .gguf_meta import GgufMeta, compute_n_gpu_layers, is_recurrent, read_gguf_meta
from . import mem_guard

log = logging.getLogger(__name__)

START_TIMEOUT_S = 60.0
STOP_GRACE_S = 10.0
HEALTH_POLL_INTERVAL_S = 0.5
# Extra grace for a process the KFD driver still holds a GPU context for.
# SIGKILL during an in-flight GPU operation can leave the driver unable to
# reclaim that context: on 2026-07-16 a killed llama-server leaked 26 GB
# (device VRAM, then host RAM via GTT) attributed to a dead PID, and every
# later launch silently fell back to CPU. Only a GPU reset or reboot frees it
# — both catastrophic on a remote box — so it is far cheaper to wait a long
# time for a clean exit than to kill early. A GPU-busy engine unloading a
# 20 GB model can legitimately need tens of seconds.
STOP_GRACE_GPU_S = 60.0
# How long to wait after exit for the driver to hand memory back before
# declaring a leak. Release is asynchronous but prompt when it works at all.
GPU_RELEASE_TIMEOUT_S = 20.0
GPU_RELEASE_POLL_S = 0.5
# Treat this much still-held memory after exit as a leak rather than noise.
GPU_LEAK_THRESHOLD_GB = 1.0


def _is_rocm_build(binary: str) -> bool:
    """Whether ``binary`` is a ROCm/HIP llama.cpp build, by its parent dir name
    (e.g. ``.../llama.cpp-hip/llama-server``). Vulkan/CPU/CUDA builds → False."""
    name = Path(binary).parent.name.lower()
    return "hip" in name or "rocm" in name


def _rocm_lib_env(binary: str) -> dict[str, str] | None:
    """LD_LIBRARY_PATH override for ROCm/HIP builds, or None to inherit.

    Only the ROCm/HIP build needs help: the llama.cpp HIP release does not
    bundle the ROCm runtime (no libamdhip64), and distros usually keep ROCm
    under /opt/rocm, which isn't on the default loader path. Without the
    system ROCm libs on LD_LIBRARY_PATH the HIP build enumerates ZERO GPUs
    and silently falls back to CPU (slow). We add the system ROCm lib dir
    only for hip/rocm variants — never for vulkan/cpu/cuda, and never the
    binary's own directory (that shadows system libs and crashed vulkan)."""
    if not _is_rocm_build(binary):
        return None
    rocm_lib = None
    candidates = [Path("/opt/rocm/lib")]
    try:
        candidates += sorted(Path("/opt").glob("rocm-*/lib"), reverse=True)
    except OSError:
        pass
    for d in candidates:
        if d.is_dir():
            rocm_lib = str(d)
            break
    if rocm_lib is None:
        return None
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = rocm_lib + (os.pathsep + existing if existing else "")
    return env


def model_needs_graph_disable(model_path: Path, binary: str) -> bool:
    """Whether CUDA graphs must be disabled for ``model_path`` on ``binary``.

    Recurrent / hybrid-SSM models (Mamba, gated-delta-net, Qwen3.6 "qwen35")
    run the ``gated_delta_net`` / SSM kernels. On ROCm/HIP builds, replaying a
    CUDA graph over recurrent state that was restored from a context checkpoint
    reads stale device memory and the queue aborts with
    HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION inside ``gated_delta_net_cuda``
    — reproducible within a few requests under agentic tool-call loads (Cline)
    on hybrid Qwen3.6 + MTP. We detect such models from their GGUF metadata so
    the workaround applies *only* where it's needed; plain transformers keep
    graphs (and their decode speedup). Scoped to ROCm builds, the only place
    this is verified; the env var also works for CUDA if that backend is ever
    found to need it. Best-effort: any read error → False (graphs left on)."""
    if not _is_rocm_build(binary):
        return False
    try:
        if not model_path.is_file():
            return False
        return is_recurrent(read_gguf_meta(model_path))
    except Exception as e:  # malformed header, IO error, etc.
        log.debug("graph-disable check failed for %s: %s", model_path, e)
        return False


def _engine_env(binary: str, *, gpu_device: str = "",
                disable_cuda_graphs: bool = False) -> dict[str, str] | None:
    """Environment override for the engine subprocess, or None to inherit.

    Layers three independent concerns onto the inherited environment:

    * ROCm/HIP runtime path (see ``_rocm_lib_env``).
    * GPU pin: when ``gpu_device`` names a card, set the backend's
      "visible devices" var so the engine runs on that GPU alone — this is
      what stops a model being silently split across a fast dGPU and a slow
      iGPU. The name is resolved against the *enumeration env* (so a ROCm
      build can actually see its GPUs), and a missing device is non-fatal.
    * ``GGML_CUDA_DISABLE_GRAPHS``: set when ``disable_cuda_graphs`` (the
      caller decides per-model via :func:`model_needs_graph_disable`) to avoid
      the recurrent-checkpoint graph-replay crash. ``setdefault`` so an
      operator's explicit value (e.g. forcing graphs back on) still wins.

    Returns None only when no concern applies (plain inherit).
    """
    env = _rocm_lib_env(binary)
    if gpu_device:
        # Enumerate with the same env the server will run with, so ROCm
        # builds get their lib path; falls back to the inherited env.
        from . import gpu_detect
        base = env if env is not None else dict(os.environ)
        pin = gpu_detect.visible_devices_env(binary, gpu_device, env=base)
        if pin:
            base.update(pin)
            env = base
    if disable_cuda_graphs:
        env = env if env is not None else dict(os.environ)
        env.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
    return env


# Args that don't translate cleanly to mlx_lm.server. mlx-lm has a much
# narrower flag surface than llama-server; we pass through what it accepts
# and silently drop the rest (they're llama.cpp-specific).
_MLX_SUPPORTED_ARGS = {
    "model", "host", "port", "trust-remote-code", "log-level",
    "chat-template", "use-default-chat-template",
    "temp", "top-p", "top-k", "min-p",
    "max-tokens",  # default; per-request override comes from the API
}


def _normalize_arg_key(k: str) -> str:
    """Engine flags are kebab-case on the command line. Accept snake_case
    keys (TOML-friendly) and emit kebab-case so config can stay readable
    without leaking quoting issues."""
    return k.replace("_", "-")


@dataclass
class StartSpec:
    """Resolved set of args to launch the inference engine with.

    ``cmdline`` adapts to either llama-server (the llama.cpp CLI) or
    ``mlx_lm.server`` based on the *engine* argument.
    """
    model_path: Path
    mmproj_path: Path | None
    extra_args: dict[str, Any]
    profile_name: str | None
    model_id: str  # the requested logical id (bare model path relative to models_dir)

    def cmdline(self, binary: str, port: int, engine: str = "llama") -> list[str]:
        if engine == "mlx":
            return self._mlx_cmdline(binary, port)
        return self._llama_cmdline(binary, port)

    def _llama_cmdline(self, binary: str, port: int) -> list[str]:
        cmd = [binary, "-m", str(self.model_path)]
        if self.mmproj_path:
            cmd += ["--mmproj", str(self.mmproj_path)]
        cmd += ["--host", "127.0.0.1", "--port", str(port)]
        for k, v in self.extra_args.items():
            k = _normalize_arg_key(k)
            if k in ("host", "port"):
                continue
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]
        return cmd

    def _mlx_cmdline(self, python: str, port: int) -> list[str]:
        # binary is the venv Python; mlx-lm is invoked as a module.
        cmd = [python, "-m", "mlx_lm", "server",
               "--model", str(self.model_path),
               "--host", "127.0.0.1", "--port", str(port)]
        for k, v in self.extra_args.items():
            k = _normalize_arg_key(k)
            if k in ("host", "port", "mmproj"):
                continue
            if k not in _MLX_SUPPORTED_ARGS:
                log.debug("dropping arg %r — not supported by mlx-lm", k)
                continue
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]
        return cmd


def _safe_under(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and require it to live under `base`.

    Rejects absolute escapes, .., symlinks pointing outside, and any other
    traversal trick. Returns the resolved (absolute) path on success.
    """
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"path escapes models_dir: {candidate} -> {resolved}"
        )
    return resolved


def _validate_model_id(model_id: str) -> None:
    """Reject obviously unsafe model identifiers before they hit the FS."""
    if not model_id or not isinstance(model_id, str):
        raise ValueError("model id is empty")
    if "\x00" in model_id:
        raise ValueError("model id contains NUL byte")
    # Backslashes are not legitimate in model ids (we always use forward
    # slashes); on POSIX they would survive Path() unsplit, so block here.
    if "\\" in model_id:
        raise ValueError(f"model id may not contain backslashes: {model_id!r}")
    p = Path(model_id)
    if p.is_absolute() or p.drive or p.root:
        raise ValueError(f"model id must be relative: {model_id!r}")
    parts = p.parts
    if any(seg in ("", "..", ".") for seg in parts):
        raise ValueError(f"model id contains forbidden segment: {model_id!r}")


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if the port can be bound. Uses SO_REUSEADDR so a socket lingering
    in TIME_WAIT (e.g. the old engine we just stopped during a swap) doesn't
    read as busy — that's exactly how llama-server binds, so it matches what
    the engine can actually do. Only a live LISTENing socket reports busy."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


async def _wait_port_free(port: int, host: str = "127.0.0.1",
                          timeout: float = 12.0) -> bool:
    """Poll until the port is bindable, up to ``timeout`` seconds. Lets a
    swap tolerate the brief window where a just-stopped engine is still
    releasing the port, instead of failing the bind check instantly."""
    deadline = time.monotonic() + timeout
    while True:
        if _port_free(port, host):
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.25)


def resolve_default(cfg: Config) -> tuple[str, str | None]:
    """Return the (model_id, profile_name) pair used when a request specifies
    neither. Profile is None when no per-model default exists.

    Returns ("", None) when no default_model is configured — callers should
    treat that as "nothing to start" rather than crashing.
    """
    model = cfg.default_model
    if not model:
        return "", None
    m = cfg.get_model(model)
    profile = (m.default_profile or None) if m else None
    return model, profile


def _basic_to_args(prof: Profile, engine: str, model_path: Path) -> dict[str, Any]:
    """Translate the structured 'basic' fields on a profile into engine
    flags. Returns a fresh dict — caller layers raw profile.args on top."""
    out: dict[str, Any] = {}
    if prof.ctx_size is not None and engine == "llama":
        out["ctx-size"] = int(prof.ctx_size)
    # Thinking-token cap → --reasoning-budget (llama only). Forces the model
    # to stop reasoning and answer once the budget is hit, so a runaway think
    # loop can't burn the whole output window with empty content.
    if getattr(prof, "reasoning_budget", None) is not None and engine == "llama":
        out["reasoning-budget"] = int(prof.reasoning_budget)
    # Concurrent request slots → --parallel (llama only). Each slot reserves
    # its own compute/KV headroom; lowering it (e.g. to 1) frees VRAM that can
    # otherwise push layers onto the CPU. Left unset, llama.cpp picks auto.
    if getattr(prof, "parallel", None) is not None and engine == "llama":
        out["parallel"] = int(prof.parallel)
    # Multi-Token Prediction (llama only). The MTP draft heads ship inside the
    # main model, so this is self-speculation — no separate draft model file.
    # llama.cpp can't run MTP with more than one slot, so we pin --parallel 1
    # here (overriding any slot count above); the profile validator already
    # blocks an explicit parallel>1 + mmproj alongside MTP.
    if getattr(prof, "mtp", False) and engine == "llama":
        out["spec-type"] = "draft-mtp"
        out["spec-draft-n-max"] = int(prof.mtp_n_max) if prof.mtp_n_max else 2
        out["parallel"] = 1
    # KV-cache quantization (llama only). Shrinks the per-token context memory
    # independently of the model's weight quant. Quantized KV needs flash
    # attention, so turn it on too. "" / "f16" leave the engine default.
    kv = (getattr(prof, "kv_cache_type", "") or "").strip().lower()
    fa = (getattr(prof, "flash_attn", "") or "").strip().lower()
    if engine == "llama" and kv and kv != "f16":
        out["cache-type-k"] = kv
        out["cache-type-v"] = kv
        # Quantized KV can't run without flash attention — force it on.
        out["flash-attn"] = "on"
    elif engine == "llama" and fa in ("on", "off", "auto"):
        # Independent flash-attn control for f16 KV. On backends where
        # quantized-KV FA is a slow fallback kernel (e.g. ROCm/HIP), this lets
        # f16 KV still get FA's large-context decode speedup.
        out["flash-attn"] = fa
    # VRAM/RAM caps → n-gpu-layers (llama only; mlx ignores it).
    if engine == "llama":
        if (prof.vram_limit_gb is not None
                or prof.ram_spill_policy != "default"):
            meta: GgufMeta | None = None
            try:
                if model_path.is_file():
                    meta = read_gguf_meta(model_path)
            except Exception as e:
                log.debug("gguf meta read failed for %s: %s", model_path, e)
            if meta is None:
                meta = GgufMeta(file_size=(model_path.stat().st_size
                                            if model_path.is_file() else 0))
            n = compute_n_gpu_layers(
                meta,
                vram_limit_gb=prof.vram_limit_gb,
                ram_spill_policy=prof.ram_spill_policy,
                ram_spill_limit_gb=prof.ram_spill_limit_gb,
                ctx_size=prof.ctx_size,
                kv_cache_type=kv,
            )
            if n is not None:
                out["n-gpu-layers"] = n
            if prof.ram_spill_policy == "none" and "fit" not in out:
                # "Every layer on the GPU" must mean exactly that. llama.cpp's
                # --fit (on by default) silently rewrites *unset* args — most
                # importantly ctx-size — to squeeze into whatever memory is
                # free, so a card that is unexpectedly occupied yields a
                # quietly downsized, partly-CPU run at ~1/100th the speed
                # instead of an error. That is precisely how a leaked 26 GB
                # GPU context went unnoticed for 12 hours. With --fit off the
                # engine refuses to start rather than degrading in silence.
                out["fit"] = "off"
    return out


def resolve_spec(cfg: Config, *, profile: str | None = None,
                 model: str | None = None,
                 mmproj: str | None = None,
                 args: dict[str, Any] | None = None) -> StartSpec:
    """Turn a request into a concrete StartSpec.

    Resolution rules:

    * Nothing specified           → ``cfg.default_model`` + its default profile.
    * Only model specified        → that model + its default profile (if any).
    * Only profile specified      → ``cfg.default_model`` + that profile.
    * Both specified              → look up ``cfg.models[model].profiles[profile]``.

    Merge order for ``extra_args``:
        engine-default args → profile basic-derived args → profile.args → request args.
    """
    args = dict(args or {})
    model = (model or "").strip() or None
    profile = (profile or "").strip() or None

    if not model:
        if cfg.default_model:
            model = cfg.default_model
        else:
            raise ValueError("no model specified and no default configured")

    if profile is None:
        m = cfg.get_model(model)
        profile = (m.default_profile or None) if m else None

    prof: Profile | None = None
    if profile:
        prof = cfg.get_profile(model, profile)
        if not prof:
            raise ValueError(
                f"unknown profile {profile!r} for model {model!r}"
            )

    _validate_model_id(model)
    model_path = _safe_under(cfg.models_dir, cfg.models_dir / model)

    engine = (detect_engine_for_path(model_path)
              if model_path.exists() else cfg.llama_server_engine)

    # Merge args: engine defaults → profile basic → profile.args → request args.
    # Normalize every key to kebab-case *before* merging. The cmdline builder
    # coerces keys with _normalize_arg_key at emit time, so an underscore key
    # from one layer (e.g. config's ``ctx_size``) and a dash key from another
    # (a profile's derived ``ctx-size``) would otherwise survive as two
    # distinct dict entries and both be emitted — a duplicated ``--ctx-size``.
    def _norm(d: dict[str, Any]) -> dict[str, Any]:
        return {_normalize_arg_key(k): v for k, v in d.items()}
    merged_args: dict[str, Any] = _norm(cfg.default_args.get(engine, {}))
    if prof:
        merged_args.update(_norm(_basic_to_args(prof, engine, model_path)))
        merged_args.update(_norm(prof.args))
    merged_args.update(_norm(args))

    chosen_mmproj = mmproj if mmproj is not None else (prof.mmproj if prof else "")
    mmproj_path: Path | None
    if chosen_mmproj:
        _validate_model_id(chosen_mmproj)
        mmproj_path = _safe_under(cfg.models_dir, cfg.models_dir / chosen_mmproj)
    else:
        mmproj_path = None

    return StartSpec(
        model_path=model_path,
        mmproj_path=mmproj_path,
        extra_args=merged_args,
        profile_name=profile,
        model_id=model,
    )


def _apply_launch_guardrails(spec: "StartSpec", cfg: Config, engine: str) -> None:
    """Inject memory-safety args and log a model-aware ctx warning in place.

    Two best-effort guardrails for the llama engine (mlx is left untouched):

    * **Bound the host prompt cache.** llama-server's ``--cache-ram`` defaults
      to 8192 MiB of host RAM for caching past prompts' KV. On a memory-tight
      box already holding a big model that is a lot; we lower it to a fraction
      of *currently-available* RAM unless the profile set it explicitly. This
      is what stops the cross-prompt cache accumulation seen in the incident.

    * **Warn on an oversized context.** Using the GGUF metadata we estimate the
      KV-cache footprint of the configured ``--ctx-size`` and compare it to
      VRAM (+RAM). A 'danger' verdict means the context cannot fit physical
      memory and will thrash swap — we log it loudly so it shows up in the
      activity log next to the launch. (We warn rather than silently clamp:
      clamping a value the operator set on purpose is its own surprise. Opt
      into clamping with ``cfg.mem_clamp_ctx = True``.)

    Never raises — a guardrail must not be able to block a launch.
    """
    if engine != "llama":
        return
    try:
        args = spec.extra_args
        # --- host prompt-cache cap ---
        if not any(k in args for k in ("cache-ram", "cram", "cache_ram")):
            avail = mem_guard.read_mem_state().ram_available_gb
            args["cache-ram"] = mem_guard.derive_cache_ram_mib(avail)

        # --- context/SWA checkpoint cap ---
        ckpt = int(getattr(cfg, "mem_ctx_checkpoints", 0) or 0)
        if ckpt > 0 and not any(
                k in args for k in ("ctx-checkpoints", "swa-checkpoints",
                                    "ctx_checkpoints")):
            args["ctx-checkpoints"] = ckpt

        # --- oversized-context warning / optional clamp ---
        ctx = args.get("ctx-size")
        try:
            ctx = int(ctx) if ctx is not None else None
        except (TypeError, ValueError):
            ctx = None
        if ctx:
            vram = getattr(cfg, "vram_total_gb", None)
            ram = mem_guard.read_mem_state().ram_total_gb
            # A vision projector also occupies VRAM — count it in the budget.
            mmproj_gb = (mem_guard.file_gb(spec.mmproj_path)
                         if spec.mmproj_path else 0.0)
            # Mirror the run's actual KV quant and slot count so the verdict
            # matches what will really be allocated. --parallel left unset means
            # llama-server auto-picks (assume DEFAULT_PARALLEL_SLOTS); KV type
            # comes from the --cache-type-k flag _basic_to_args emitted.
            kv_type = str(args.get("cache-type-k") or "")
            try:
                n_par = int(args.get("parallel")
                            or mem_guard.DEFAULT_PARALLEL_SLOTS)
            except (TypeError, ValueError):
                n_par = mem_guard.DEFAULT_PARALLEL_SLOTS
            verdict = mem_guard.ctx_safety(spec.model_path, ctx, vram, ram,
                                           extra_gb=mmproj_gb,
                                           kv_cache_type=kv_type,
                                           n_parallel=n_par)
            if verdict and verdict.is_unsafe:
                log.warning("ctx guardrail [%s]: %s — %s",
                            verdict.level, spec.model_id, verdict.message)
                if (getattr(cfg, "mem_clamp_ctx", False)
                        and verdict.level == "danger" and verdict.safe_ctx):
                    log.warning("ctx guardrail: clamping ctx-size %d -> %d for %s",
                                ctx, verdict.safe_ctx, spec.model_id)
                    args["ctx-size"] = verdict.safe_ctx
    except Exception as e:  # noqa: BLE001 — guardrails are best-effort
        log.debug("launch guardrails skipped: %s", e)


def _preflight_gpu_memory(spec: "StartSpec", cfg: Config, engine: str,
                          *, gpu_only: bool) -> None:
    """Compare *live* free VRAM against what this launch needs.

    The sizing guardrails above reason about the card's nominal capacity. That
    is not the same as what is actually free right now: another engine may
    still be resident, or the driver may be holding a leaked context from a
    killed process. On 2026-07-16 the latter left 7.5 GB free of 32 GB, and
    because nothing checked, every launch quietly ran on CPU.

    Warns always; raises ``ServerError`` only when the profile demands a
    GPU-only run (``ram_spill_policy = "none"``), where a spill is a
    correctness failure rather than a slow path.
    """
    if engine != "llama":
        return
    cards = mem_guard.read_gpu_mem()
    if not cards:
        return
    gpu = max(cards, key=lambda g: g.vram_total)
    need = mem_guard.file_gb(spec.model_path)
    if spec.mmproj_path:
        need += mem_guard.file_gb(spec.mmproj_path)
    ctx = spec.extra_args.get("ctx-size")
    try:
        ctx = int(ctx) if ctx is not None else None
    except (TypeError, ValueError):
        ctx = None
    if ctx:
        kv = mem_guard.kv_cache_gb(spec.model_path, ctx,
                                   kv_cache_type=str(
                                       spec.extra_args.get("cache-type-k") or ""))
        if kv:
            need += kv
    free = gpu.vram_free
    stale = mem_guard.stale_kfd_pids()
    if stale:
        log.error(
            "GPU has leaked contexts from dead PIDs %s — %.1f GB VRAM and "
            "%.1f GB host RAM (GTT) are pinned and unreclaimable. Clearing "
            "needs a GPU reset or reboot; not doing that automatically.",
            sorted(stale), gpu.vram_used, gpu.gtt_used)
    if need <= free:
        return
    detail = (f"{spec.model_id} needs ~{need:.1f} GB on the GPU "
              f"(weights + KV at ctx {ctx or '?'}) but only {free:.1f} GB of "
              f"{gpu.vram_total:.0f} GB is free "
              f"({gpu.vram_used:.1f} GB already in use"
              + (f", {gpu.gtt_used:.1f} GB pinned as host RAM/GTT"
                 if gpu.gtt_used > 1.0 else "") + ")")
    if stale:
        detail += (f"; leaked GPU contexts from dead PIDs {sorted(stale)} are "
                   "holding memory — a GPU reset or host reboot is required to "
                   "reclaim it")
    if gpu_only:
        raise ServerError(
            f"refusing to start: {detail}. This profile requires a full-GPU "
            f"run (ram_spill_policy='none'), and starting anyway would spill "
            f"to system RAM and run ~100x slower.")
    log.warning("preflight: %s — will spill to system RAM and run far slower",
                detail)


class ServerError(Exception):
    pass


class ServerManager:
    def __init__(self, cfg: Config, db: DB, *,
                 slot_id: int = 0,
                 port_override: int | None = None) -> None:
        """One llama-server child process.

        ``slot_id`` and ``port_override`` exist for the multi-slot beta
        (see ``server_pool.ServerPool``). With both at their defaults,
        behaviour is byte-identical to the legacy single-instance design:
        port from ``cfg.llama_server_port``, runtime persisted to
        ``cfg.runtime_path``, log at ``logs_dir/llama-server.log``.

        For non-default slots (``slot_id > 0``), the runtime state is
        kept in-memory — the slot manifest in ``slots.json`` is the
        persistence layer — and the log file is suffixed with the slot
        id so simultaneously-running children don't interleave logs.
        """
        self.cfg = cfg
        self.db = db
        self.slot_id = slot_id
        self._port = (port_override if port_override is not None
                      else cfg.llama_server_port)
        if slot_id == 0:
            self._log_name = "llama-server.log"
            self._runtime_path: Path | None = cfg.runtime_path
            self.runtime: rt.RuntimeState = rt.load(cfg.runtime_path)
        else:
            self._log_name = f"llama-server-slot{slot_id}.log"
            self._runtime_path = None  # in-memory only for non-default slots
            self.runtime = rt.RuntimeState()
        self.proc: asyncio.subprocess.Process | None = None
        self.spec: StartSpec | None = None
        self._wait_task: asyncio.Task[int] | None = None
        self._lock = asyncio.Lock()
        self._log_fp = None  # type: ignore[assignment]
        self._intentional_stop = False
        # Set once the driver has been seen failing to reclaim a dead engine's
        # GPU memory. Sticky: only a GPU reset or host reboot clears the
        # underlying condition, so it must not be reset by a later good stop.
        self.gpu_leak_detected = False
        # subscribers to unexpected-exit events
        # Each listener queue receives ``(slot_id, returncode)`` tuples so a
        # supervisor watching multiple slots can tell whose child died.
        self._exit_listeners: list[asyncio.Queue[tuple[int, int]]] = []

    # ---- internal: runtime persistence shim ----
    def _save_runtime(self) -> None:
        """Persist ``self.runtime`` to disk for slot 0; no-op for slot N>0.

        Multi-slot's slot 1..N runtime is purely in-memory — the
        persistent manifest in slots.json is the source of truth for
        what gets re-started on daemon boot. This avoids generating a
        runtime-slot1.json sibling for every slot and keeps the legacy
        runtime.json shape unchanged.
        """
        if self._runtime_path is not None:
            rt.save(self._runtime_path, self.runtime)

    # ---- lifecycle hooks ----
    def add_exit_listener(self, q: asyncio.Queue) -> None:
        """Subscribe to unexpected-exit events for this slot.

        The queue receives ``(slot_id, returncode)`` tuples. A single
        queue can be passed to multiple ServerManagers (see
        ``ServerPool.add_exit_listener``) so one supervisor consumer
        sees every slot's exits.
        """
        self._exit_listeners.append(q)

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def upstream_base(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    # ---- start ----
    async def start(self, spec: StartSpec) -> int:
        async with self._lock:
            if self.is_running:
                raise ServerError("inference engine already running")
            engine = (detect_engine_for_path(spec.model_path)
                      if spec.model_path.exists()
                      else (getattr(self.cfg, "llama_server_engine", "llama") or "llama"))
            if not spec.model_path.exists():
                raise ServerError(f"model not found: {spec.model_path}")
            if spec.mmproj_path and not spec.mmproj_path.exists():
                raise ServerError(f"mmproj file not found: {spec.mmproj_path}")
            # If exclusive mode is on, evict any foreign llama-server /
            # engine workers before we try to claim the port + VRAM.
            # Belt-and-suspenders: even if config drift somehow left the
            # value set, multi-slot is mutually exclusive — skip the sweep.
            mode = (getattr(self.cfg, "exclusive_mode", "off") or "off").lower()
            if (getattr(self.cfg, "multi_slot_enabled", False)
                    and mode not in ("off", "")):
                mode = "off"
            if mode not in ("off", ""):
                try:
                    await _exclusive.sweep_and_record(
                        mode,
                        grace_seconds=float(getattr(
                            self.cfg, "exclusive_grace_seconds", 5.0) or 5.0),
                    )
                except Exception:  # noqa: BLE001 — sweep must never block start
                    log.exception("exclusive: pre-start sweep failed")
            port = self._port
            # Wait briefly for the port to free up rather than failing
            # instantly. A swap stops the old llama-server then starts the
            # new one on the same port; the just-closed socket can linger in
            # TIME_WAIT for a moment, which would otherwise read as "busy"
            # mid-swap. _port_free uses SO_REUSEADDR (as llama-server does),
            # so it only reports busy when something is genuinely LISTENING.
            if not await _wait_port_free(port):
                raise ServerError(
                    f"port {port} is still in use after waiting — another "
                    "process may be holding it (see exclusive mode)."
                )

            _apply_launch_guardrails(spec, self.cfg, engine)
            # Live free-VRAM check. Runs after the guardrails so it sees the
            # final ctx-size, and before the process exists so a GPU-only
            # profile fails fast with a readable reason instead of starting a
            # doomed, silently-CPU-bound engine.
            _preflight_gpu_memory(
                spec, self.cfg, engine,
                gpu_only=(str(spec.extra_args.get("fit", "")).lower() == "off"
                          or spec.extra_args.get("n-gpu-layers") == -1))
            cmd = spec.cmdline(self.cfg.llama_server_binary, port, engine=engine)
            log_path = self.cfg.logs_dir / self._log_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = open(log_path, "ab", buffering=0)
            # Delimit each run so successive launches (different dates,
            # variants, models) don't blur together in the single shared log.
            banner = (
                f"\n{'=' * 78}\n"
                f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  START  engine={engine}\n"
                f"=== binary={self.cfg.llama_server_binary}\n"
                f"=== model={spec.model_id}\n"
                f"{'=' * 78}\n"
            )
            try:
                self._log_fp.write(banner.encode("utf-8", "replace"))
            except OSError:
                pass
            log.info("launching %s engine: %s", engine, shlex.join(cmd))
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=self._log_fp,
                    stderr=self._log_fp,
                    stdin=asyncio.subprocess.DEVNULL,
                    # None → inherit (vulkan/cpu/cuda); hip/rocm gets the
                    # system ROCm runtime added so it can see the GPU, and a
                    # configured GPU pin adds the backend's visible-devices
                    # var. The pin applies to the llama engine only — mlx
                    # (Apple) has no --list-devices to resolve against.
                    # Recurrent / hybrid-SSM models on ROCm additionally get
                    # CUDA graphs disabled to dodge the gated-delta-net
                    # checkpoint-restore crash (see model_needs_graph_disable).
                    env=_engine_env(
                        self.cfg.llama_server_binary,
                        gpu_device=(getattr(self.cfg, "llama_gpu_device", "") or "")
                        if engine == "llama" else "",
                        disable_cuda_graphs=(
                            engine == "llama"
                            and model_needs_graph_disable(
                                spec.model_path, self.cfg.llama_server_binary)
                        ),
                    ),
                )
            except FileNotFoundError as e:
                self._close_log()
                raise ServerError(
                    f"engine binary not found: {self.cfg.llama_server_binary}"
                ) from e

            self.spec = spec
            self._intentional_stop = False
            self.runtime = rt.RuntimeState(
                state="starting", pid=self.proc.pid,
                current_model=spec.model_id,
                current_profile=spec.profile_name,
                current_args=spec.extra_args,
                started_at=time.time(),
                last_event_at=time.time(),
            )
            self._save_runtime()
            self.db.log_event("server_starting", {"pid": self.proc.pid,
                                                  "cmd": cmd,
                                                  "model": spec.model_id,
                                                  "profile": spec.profile_name})
            self._wait_task = asyncio.create_task(self._watch_proc())

        # health-poll outside the lock so /admin/status can answer
        ok = await self._await_ready()
        if not ok:
            await self.stop()
            raise ServerError("llama-server failed to become healthy in time")

        self.runtime.state = "running"
        self.runtime.last_event_at = time.time()
        self._save_runtime()
        self.db.log_event("server_running", {"pid": self.proc.pid if self.proc else None,
                                             "model": spec.model_id})
        return self.proc.pid if self.proc else -1

    async def _await_ready(self) -> bool:
        url = f"{self.upstream_base}/health"
        deadline = time.time() + START_TIMEOUT_S
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if not self.is_running:
                    return False
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
        return False

    # ---- stop ----
    async def stop(self) -> None:
        async with self._lock:
            self._intentional_stop = True
            await self._terminate()
            self.runtime.state = "stopped"
            self.runtime.pid = None
            self.runtime.current_model = None
            self.runtime.current_profile = None
            self.runtime.last_event_at = time.time()
            self._save_runtime()
            self.db.log_event("server_stopped", {})

    async def _terminate(self) -> None:
        if not self.proc:
            return
        pid = self.proc.pid
        before = mem_guard.read_gpu_mem()
        killed = False
        if self.proc.returncode is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=STOP_GRACE_S)
            except asyncio.TimeoutError:
                # Before escalating, check whether the driver still holds a GPU
                # context for this PID. If it does, SIGKILL risks leaking that
                # context permanently (recoverable only by GPU reset/reboot),
                # so give it a much longer window to finish unwinding first.
                if pid in mem_guard.kfd_proc_pids():
                    log.warning(
                        "llama-server (pid %s) ignored SIGTERM and still holds a "
                        "GPU context — waiting up to %.0fs for a clean exit "
                        "rather than SIGKILLing into an in-flight GPU op "
                        "(that leaks VRAM/GTT until the next reboot)",
                        pid, STOP_GRACE_GPU_S)
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self.proc.wait(),
                                               timeout=STOP_GRACE_GPU_S)
            if self.proc.returncode is None:
                log.warning("llama-server did not exit on SIGTERM, escalating to SIGKILL")
                killed = True
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    log.error("llama-server still alive after SIGKILL — leaking")
        await self._verify_gpu_released(pid, before, killed=killed)
        self._close_log()
        self.proc = None
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
        self._wait_task = None

    async def _verify_gpu_released(self, pid: int,
                                   before: list[mem_guard.GpuMem],
                                   *, killed: bool) -> None:
        """Confirm the driver handed back the engine's GPU memory after exit.

        A leaked KFD context is silent: `rocm-smi` keeps reporting the card as
        mostly free once device VRAM drains into GTT, while the pinned host RAM
        never comes back and every subsequent launch quietly falls back to CPU.
        We detect it two ways — a KFD context still registered to a PID that no
        longer exists, and memory that never drops after exit.

        This only *records* the condition. Clearing it needs a GPU reset or a
        reboot, and on a remotely-managed box an unattended reboot can wedge
        the machine for days, so the decision belongs to the operator.
        """
        if not before:
            return
        deadline = time.time() + GPU_RELEASE_TIMEOUT_S
        stale: set[int] = set()
        after: list[mem_guard.GpuMem] = before
        while time.time() < deadline:
            await asyncio.sleep(GPU_RELEASE_POLL_S)
            stale = mem_guard.stale_kfd_pids()
            after = mem_guard.read_gpu_mem()
            if pid in stale:
                break          # definitive: dead PID still holds a context
            if not stale and self._gpu_freed(before, after):
                return         # memory came back; nothing to report
        by_card = {g.card: g for g in before}
        details = []
        for now in after:
            was = by_card.get(now.card)
            if not was:
                continue
            details.append(
                f"{now.card}: VRAM {was.vram_used:.1f}→{now.vram_used:.1f} GB, "
                f"GTT(host RAM) {was.gtt_used:.1f}→{now.gtt_used:.1f} GB")
        # A stale KFD context (dead PID, live context) is definitive. The
        # memory-delta check is only a heuristic, and it is wrong whenever
        # another engine legitimately holds the card — multi-slot keeps other
        # llama-servers resident — so only trust it when no live GPU process
        # remains to account for what's still held.
        leaked = bool(stale)
        if not leaked and not mem_guard.kfd_proc_pids():
            leaked = not self._gpu_freed(before, after)
        if not leaked:
            return
        self.gpu_leak_detected = True
        msg = ("GPU memory was NOT released after llama-server (pid %s) exited%s. "
               "%s%s This is a driver-level leak: the memory (including host RAM "
               "pinned as GTT) stays unusable until the GPU is reset or the host "
               "rebooted. Engines started from now on may silently fall back to "
               "CPU and run ~100x slower. NOT clearing it automatically — a GPU "
               "reset or reboot can wedge this machine.")
        log.error(msg, pid,
                  " (after SIGKILL)" if killed else "",
                  f"stale KFD contexts for dead PIDs: {sorted(stale)}. " if stale else "",
                  "; ".join(details))
        with contextlib.suppress(Exception):
            self.db.log_event("gpu_memory_leak", {
                "pid": pid, "killed": killed, "stale_kfd_pids": sorted(stale),
                "cards": [{"card": g.card, "vram_used_gb": round(g.vram_used, 2),
                           "gtt_used_gb": round(g.gtt_used, 2)} for g in after],
            })

    @staticmethod
    def _gpu_freed(before: list[mem_guard.GpuMem],
                   after: list[mem_guard.GpuMem]) -> bool:
        """True if every card gave back all but a trivial amount of memory.

        Compares VRAM + GTT together: a leak commonly migrates out of device
        VRAM into pinned host RAM, so watching VRAM alone reads as "released".
        """
        by_card = {g.card: g for g in before}
        for now in after:
            was = by_card.get(now.card)
            if not was:
                continue
            held_before = was.vram_used + was.gtt_used
            held_now = now.vram_used + now.gtt_used
            # Released means the memory came back in absolute terms, not merely
            # that the number moved. A leak migrating VRAM->GTT barely changes
            # the total (26.5 -> 25.0 GB), so a "dropped by more than a
            # threshold" test would wave it through.
            if held_now <= GPU_LEAK_THRESHOLD_GB:
                continue
            if held_now <= held_before * 0.25:
                continue
            return False
        return True

    def _close_log(self) -> None:
        try:
            if self._log_fp:
                self._log_fp.close()
        except Exception:
            pass
        self._log_fp = None

    # ---- restart / swap ----
    async def restart(self, spec: StartSpec | None = None) -> int:
        target = spec or self.spec
        if not target:
            raise ServerError("no previous spec to restart with")
        await self.stop()
        return await self.start(target)

    async def swap(self, new_spec: StartSpec) -> int:
        """Stop current, start new. On failure roll back to previous spec."""
        old_spec = self.spec
        self.runtime.state = "swapping"
        self.runtime.last_event_at = time.time()
        self._save_runtime()
        self.db.log_event("server_swap_begin",
                          {"from": old_spec.model_id if old_spec else None,
                           "to": new_spec.model_id})
        await self.stop()
        try:
            pid = await self.start(new_spec)
            self.db.log_event("server_swap_ok",
                              {"from": old_spec.model_id if old_spec else None,
                               "to": new_spec.model_id})
            return pid
        except Exception as e:
            self.db.log_event("server_swap_fail",
                              {"to": new_spec.model_id, "error": str(e)})
            if old_spec:
                try:
                    return await self.start(old_spec)
                except Exception as e2:
                    self.runtime.state = "degraded"
                    self.runtime.last_event_at = time.time()
                    self._save_runtime()
                    self.db.log_event("server_degraded",
                                      {"reason": "rollback_failed", "error": str(e2)})
                    raise ServerError(
                        f"swap failed and rollback also failed: {e2}"
                    ) from e2
            raise

    # ---- crash detection ----
    def _notify_exit_listeners(self, rc: int) -> None:
        """Push ``(slot_id, rc)`` to every registered listener queue.

        Extracted so paths that aren't a real ``proc.wait()`` — like a
        failed restore after an image-yield — can still trigger the
        supervisor's restart/backoff cycle. Without this, a
        yield-to-image whose restore raised would leave the daemon in
        a non-running state with no recovery plan.

        The slot id is included so a multi-slot supervisor can route
        the crash back to the right ServerManager via the pool.
        """
        for q in list(self._exit_listeners):
            try:
                q.put_nowait((self.slot_id, rc))
            except asyncio.QueueFull:
                pass

    async def _watch_proc(self) -> int:
        proc = self.proc
        assert proc
        rc = await proc.wait()
        # If we initiated the stop, this is benign.
        if self._intentional_stop:
            return rc
        log.warning("llama-server exited unexpectedly with code %s", rc)
        self.db.log_event("server_crashed", {"returncode": rc})
        self.runtime.state = "crashed"
        self.runtime.pid = None
        self.runtime.last_event_at = time.time()
        self._save_runtime()
        self._notify_exit_listeners(rc)
        return rc

    # ---- diagnostics ----
    async def health(self) -> bool:
        if not self.is_running:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.upstream_base}/health")
                return r.status_code == 200
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        try:
            import psutil
            mem_rss_mb: float | None = None
            if self.proc and self.proc.pid:
                try:
                    mem_rss_mb = psutil.Process(self.proc.pid).memory_info().rss / (1024 * 1024)
                except Exception:
                    mem_rss_mb = None
        except ImportError:
            mem_rss_mb = None
        uptime = (time.time() - self.runtime.started_at) if self.runtime.started_at else None
        return {
            "state": self.runtime.state,
            "current_model": self.runtime.current_model,
            "current_profile": self.runtime.current_profile,
            "pid": self.runtime.pid,
            "started_at": self.runtime.started_at,
            "uptime_s": uptime,
            "mem_rss_mb": mem_rss_mb,
        }

    # ---- supervisor coordination ----
    def mark_degraded(self, reason: str) -> None:
        self.runtime.state = "crashed"
        self.runtime.last_event_at = time.time()
        self._save_runtime()
        self.db.log_event("server_supervisor_giveup", {"reason": reason})

    # ---- image-family coordination ----
    @contextlib.asynccontextmanager
    async def yield_to_image(self, *, reason: str = "image"):
        """Temporarily release the GPU slot to an image task.

        Coexistence policy (see config.toml ``[coexistence]``):

        * ``allow_concurrent = true``  → no-op; text stays running.
        * ``unload_text_on_arrival = false`` → no-op (operator opted to
          keep text running; image task accepts the risk).
        * Otherwise: snapshot the active StartSpec, stop the server,
          yield, then optionally restart per ``restart_text_after_image``.
        """
        # Resolve the current policy each time — config hot-reloads.
        if self.cfg.allow_concurrent:
            yield
            return
        if not self.cfg.unload_text_on_arrival:
            yield
            return

        saved_spec: StartSpec | None = self.spec if self.is_running else None
        if saved_spec is None:
            # Nothing to unload, nothing to restore.
            yield
            return

        self.db.log_event("text_yield_to_image_begin", {
            "model": saved_spec.model_id, "reason": reason,
        })
        await self.stop()
        # Exclusive sweep before the image worker takes the slot — kills
        # any straggler llama-server / ComfyUI / etc. holding VRAM.
        # Belt-and-suspenders mutex with multi-slot (see start()).
        mode = (getattr(self.cfg, "exclusive_mode", "off") or "off").lower()
        if (getattr(self.cfg, "multi_slot_enabled", False)
                and mode not in ("off", "")):
            mode = "off"
        if mode not in ("off", ""):
            try:
                await _exclusive.sweep_and_record(
                    mode,
                    grace_seconds=float(getattr(
                        self.cfg, "exclusive_grace_seconds", 5.0) or 5.0),
                )
            except Exception:  # noqa: BLE001
                log.exception("exclusive: pre-image-yield sweep failed")
        try:
            yield
        finally:
            if self.cfg.restart_text_after_image:
                try:
                    await self.start(saved_spec)
                    self.db.log_event("text_yield_to_image_restored", {
                        "model": saved_spec.model_id,
                    })
                except Exception as e:
                    log.error("yield_to_image: restore failed: %s", e)
                    self.db.log_event("text_yield_to_image_restore_failed", {
                        "model": saved_spec.model_id, "error": str(e),
                    })
                    # Hand the failure to the supervisor so its
                    # backoff/cooldown logic kicks in. Without this,
                    # a single transient start() failure (port still
                    # held, VRAM not yet released, binary path moved,
                    # etc.) would leave the daemon parked with no
                    # llama-server running and no plan to restart.
                    # The spec we want is saved_spec, so re-set it
                    # before notifying — start() cleared it on failure.
                    if self.spec is None:
                        self.spec = saved_spec
                    self.runtime.state = "crashed"
                    self.runtime.pid = None
                    self.runtime.last_event_at = time.time()
                    self._save_runtime()
                    self._notify_exit_listeners(-1)
            else:
                self.db.log_event("text_yield_to_image_kept_unloaded", {
                    "prior_model": saved_spec.model_id,
                })
