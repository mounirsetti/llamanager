"""Tiny GGUF metadata reader + n-gpu-layers sizing heuristic.

We only need a handful of metadata fields — block_count, embedding_length,
head_count_kv, context_length — and the file size. A full GGUF parser
would be overkill, so this module pulls just enough of the header to
compute a sane n_gpu_layers value from a user-facing VRAM/RAM budget.

GGUF spec (v2/v3): magic "GGUF" + u32 version + u64 tensor_count + u64
kv_count, then kv_count entries of (key_string, value_type, value).
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"

# Value type ids per GGUF spec.
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12

_SCALAR_FMT: dict[int, tuple[str, int]] = {
    _T_UINT8: ("<B", 1),
    _T_INT8: ("<b", 1),
    _T_UINT16: ("<H", 2),
    _T_INT16: ("<h", 2),
    _T_UINT32: ("<I", 4),
    _T_INT32: ("<i", 4),
    _T_FLOAT32: ("<f", 4),
    _T_BOOL: ("<?", 1),
    _T_UINT64: ("<Q", 8),
    _T_INT64: ("<q", 8),
    _T_FLOAT64: ("<d", 8),
}


@dataclass
class GgufMeta:
    """Subset of GGUF metadata we care about for sizing."""
    architecture: str | None = None
    block_count: int | None = None
    embedding_length: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    # Explicit per-head K/V dimension. Some architectures decouple this from
    # embedding_length // head_count (e.g. a head_dim of 256 on a 5120-wide,
    # 24-head model), so we must read it rather than derive it for KV sizing.
    key_length: int | None = None
    value_length: int | None = None
    context_length: int | None = None
    # Hybrid SSM/attention models (e.g. Qwen3.6 "qwen35") interleave Mamba-style
    # SSM layers with full-attention layers, placing a full-attention layer only
    # once every ``full_attention_interval`` blocks. Only the attention layers
    # hold a KV cache that grows with context; the SSM layers keep a fixed-size
    # recurrent state. Without this, KV sizing over-counts by block_count /
    # n_attention_layers (≈4× for Qwen3.6-27B: 65 blocks, 16 attention layers).
    full_attention_interval: int | None = None
    # SSM recurrent-state geometry, used for the (constant, ctx-independent)
    # state footprint of the non-attention layers.
    ssm_inner_size: int | None = None
    ssm_state_size: int | None = None
    ssm_conv_kernel: int | None = None
    # Sliding-window attention (Gemma 3/4 and friends). These models interleave
    # *local* layers, whose KV cache is capped at ``sliding_window`` tokens and
    # therefore does NOT grow with context, with sparse *global* layers that do.
    # Gemma 4 additionally gives the two kinds different geometry: the local
    # layers use more KV heads but a smaller head_dim (16 × 256) than the global
    # ones (4 × 512). Treating every layer as full-context global attention
    # over-counts KV by ~13× on gemma-4-31B, which is what made the ctx
    # guardrail claim 15 GB at ctx 4096 for a model whose real KV is ~1.2 GB.
    sliding_window: int | None = None
    # Per-layer True=local(SWA) / False=global. Length == block_count.
    sliding_window_pattern: list[bool] | None = None
    # Per-layer KV head counts. Present when the arch varies KV heads by layer.
    head_count_kv_per_layer: list[int] | None = None
    # K/V head dimension for the sliding-window layers, when it differs.
    key_length_swa: int | None = None
    value_length_swa: int | None = None
    file_size: int = 0


def _read_string(buf: memoryview, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = bytes(buf[off:off + n]).decode("utf-8", errors="replace")
    return s, off + n


def _skip_value(buf: memoryview, off: int, vtype: int) -> int:
    if vtype in _SCALAR_FMT:
        _, size = _SCALAR_FMT[vtype]
        return off + size
    if vtype == _T_STRING:
        (n,) = struct.unpack_from("<Q", buf, off)
        return off + 8 + n
    if vtype == _T_ARRAY:
        (inner,) = struct.unpack_from("<I", buf, off)
        off += 4
        (count,) = struct.unpack_from("<Q", buf, off)
        off += 8
        for _ in range(count):
            off = _skip_value(buf, off, inner)
        return off
    raise ValueError(f"unsupported gguf value type: {vtype}")


def _read_scalar_array(buf: memoryview, off: int) -> tuple[list[Any] | None, int]:
    """Read an array of scalars, returning ``(values, new_off)``.

    ``off`` must point at the array payload (just past the value-type tag).
    Returns ``(None, new_off)`` for arrays of non-scalar element types, which
    we never need. Unlike ``_read_value`` — which deliberately skips arrays,
    since almost every array in a GGUF is the multi-megabyte tokenizer
    vocabulary — this materialises the values. Only call it for the handful of
    small per-layer geometry arrays we actually size from.
    """
    (inner,) = struct.unpack_from("<I", buf, off)
    (count,) = struct.unpack_from("<Q", buf, off + 4)
    end = _skip_value(buf, off, _T_ARRAY)
    if inner not in _SCALAR_FMT:
        return None, end
    o = off + 12
    vals: list[Any] = []
    for _ in range(count):
        v, o = _read_value(buf, o, inner)
        vals.append(v)
    return vals, end


def _read_value(buf: memoryview, off: int, vtype: int) -> tuple[Any, int]:
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        (v,) = struct.unpack_from(fmt, buf, off)
        return v, off + size
    if vtype == _T_STRING:
        return _read_string(buf, off)
    if vtype == _T_ARRAY:
        # Skipped — we never need an array value for sizing.
        return None, _skip_value(buf, off, vtype)
    raise ValueError(f"unsupported gguf value type: {vtype}")


def read_gguf_meta(path: Path, *, max_header_bytes: int = 4 * 1024 * 1024) -> GgufMeta:
    """Read the GGUF header and pull out the few fields we need for sizing.

    Reads up to ``max_header_bytes`` (4 MiB by default). That's enough for
    any normal model — the metadata section is tiny compared to the tensor
    payload. Raises ValueError on malformed headers.
    """
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(max_header_bytes)
    if len(header) < 16 or header[:4] != _GGUF_MAGIC:
        raise ValueError(f"not a GGUF file: {path}")
    buf = memoryview(header)
    (version,) = struct.unpack_from("<I", buf, 4)
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported gguf version: {version}")
    off = 8
    if version == 1:
        # v1 used 32-bit counts; we don't support pulling sizing from those.
        return GgufMeta(file_size=file_size)
    (tensor_count,) = struct.unpack_from("<Q", buf, off)
    off += 8
    (kv_count,) = struct.unpack_from("<Q", buf, off)
    off += 8
    _ = tensor_count  # unused

    meta = GgufMeta(file_size=file_size)
    arch: str | None = None
    pending: dict[str, Any] = {}
    try:
        for _i in range(kv_count):
            key, off = _read_string(buf, off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off += 4
            if key == "general.architecture":
                arch, off = _read_value(buf, off, vtype)
                meta.architecture = arch if isinstance(arch, str) else None
                continue
            # Hold sizing keys aside and resolve them once we know the arch.
            if key.endswith(".block_count") or key.endswith(".embedding_length") \
                    or key.endswith(".attention.head_count") \
                    or key.endswith(".attention.head_count_kv") \
                    or key.endswith(".attention.key_length") \
                    or key.endswith(".attention.value_length") \
                    or key.endswith(".attention.key_length_swa") \
                    or key.endswith(".attention.value_length_swa") \
                    or key.endswith(".attention.sliding_window") \
                    or key.endswith(".attention.sliding_window_pattern") \
                    or key.endswith(".context_length") \
                    or key.endswith(".full_attention_interval") \
                    or key.endswith(".ssm.inner_size") \
                    or key.endswith(".ssm.state_size") \
                    or key.endswith(".ssm.conv_kernel"):
                # Several of these are per-layer arrays on SWA architectures
                # (gemma4 ships head_count_kv and sliding_window_pattern as
                # 60-element arrays); read those rather than skipping them.
                if vtype == _T_ARRAY:
                    val, off = _read_scalar_array(buf, off)
                else:
                    val, off = _read_value(buf, off, vtype)
                pending[key] = val
            else:
                off = _skip_value(buf, off, vtype)
    except (struct.error, ValueError) as e:
        log.debug("gguf header parse stopped early for %s: %s", path, e)

    if arch:
        meta.block_count = pending.get(f"{arch}.block_count") or pending.get("block_count")
        meta.embedding_length = pending.get(f"{arch}.embedding_length")
        meta.head_count = pending.get(f"{arch}.attention.head_count")
        hckv = pending.get(f"{arch}.attention.head_count_kv")
        if isinstance(hckv, list):
            # Per-layer KV heads (gemma4: 16 on local layers, 4 on global).
            # Keep the list for exact sizing and use the max as the scalar
            # fallback so any legacy scalar path stays conservative.
            meta.head_count_kv_per_layer = [int(v) for v in hckv if v is not None]
            meta.head_count_kv = (max(meta.head_count_kv_per_layer)
                                  if meta.head_count_kv_per_layer else meta.head_count)
        else:
            meta.head_count_kv = hckv or meta.head_count
        meta.key_length = pending.get(f"{arch}.attention.key_length")
        meta.value_length = (pending.get(f"{arch}.attention.value_length")
                             or meta.key_length)
        meta.key_length_swa = pending.get(f"{arch}.attention.key_length_swa")
        meta.value_length_swa = (pending.get(f"{arch}.attention.value_length_swa")
                                 or meta.key_length_swa)
        sw = pending.get(f"{arch}.attention.sliding_window")
        meta.sliding_window = sw if isinstance(sw, int) else None
        swp = pending.get(f"{arch}.attention.sliding_window_pattern")
        if isinstance(swp, list):
            meta.sliding_window_pattern = [bool(v) for v in swp]
        meta.context_length = pending.get(f"{arch}.context_length")
        meta.full_attention_interval = pending.get(
            f"{arch}.full_attention_interval")
        meta.ssm_inner_size = pending.get(f"{arch}.ssm.inner_size")
        meta.ssm_state_size = pending.get(f"{arch}.ssm.state_size")
        meta.ssm_conv_kernel = pending.get(f"{arch}.ssm.conv_kernel")
    return meta


def is_recurrent(meta: GgufMeta) -> bool:
    """True if the model has SSM / recurrent layers (Mamba, gated-delta-net,
    and the hybrid SSM/attention archs like Qwen3.6 "qwen35").

    Detected from the SSM geometry keys in the GGUF header. These models run
    the ``gated_delta_net`` / SSM compute kernels, which on some ROCm/HIP (and
    CUDA) builds fault when a CUDA graph is replayed over recurrent state
    restored from a context checkpoint — see server_manager's graph-disable
    handling. Both pure-recurrent and hybrid models carry these keys."""
    return bool(meta.ssm_state_size or meta.ssm_inner_size
                or meta.ssm_conv_kernel or meta.full_attention_interval)


# ---------------------------------------------------------------------------
# Sizing heuristic
# ---------------------------------------------------------------------------

# Reserved VRAM for KV cache, overhead, and small allocations. Used when the
# caller doesn't supply a ctx_size that lets us compute KV exactly.
_DEFAULT_VRAM_OVERHEAD_FRACTION = 0.12
# Minimum reserve in GB regardless of model size.
_MIN_VRAM_OVERHEAD_GB = 0.5

# Bytes per KV-cache element for each cache dtype. f16 is the engine default;
# the quantized types shrink the per-token context memory independently of the
# model's weight quant. Values are the real llama.cpp block bit-widths / 8
# (q8_0 = 8.5 bpw, q5_1 = 6.0, q4_0 = 4.5).
_KV_CACHE_BYTES_PER_ELEM: dict[str, float] = {
    "": 2.0, "f16": 2.0, "f32": 4.0,
    "q8_0": 8.5 / 8, "q5_1": 6.0 / 8, "q4_0": 4.5 / 8,
}


def kv_bytes_per_elem(kv_cache_type: str | None) -> float:
    """Bytes per KV-cache element for a cache dtype. Unknown/blank → f16,
    the engine default and the conservative (largest) choice."""
    return _KV_CACHE_BYTES_PER_ELEM.get((kv_cache_type or "").strip().lower(), 2.0)


def _kv_cache_gb(meta: GgufMeta, ctx_size: int | None,
                 *, bytes_per_elem: float = 2.0,
                 include_recurrent: bool = True) -> float | None:
    """Estimate KV cache size in GB for ``ctx_size`` tokens.

    KV cache holds, per layer and token, an n_kv_heads × head_dim K vector and
    the same-shaped V vector. ``bytes_per_elem`` is the cache dtype's element
    size (2.0 = f16; see ``kv_bytes_per_elem`` for the quantized types).

    head_dim comes from the model's explicit ``attention.key_length`` /
    ``value_length`` when present (some archs decouple it from
    embedding_length // head_count); otherwise we fall back to that ratio.

    For hybrid SSM/attention models the result is ``linear_attention_KV(ctx) +
    constant_SSM_state``. Pass ``include_recurrent=False`` to get the
    ctx-proportional attention term ALONE — needed by callers that want a true
    per-token rate to extrapolate linearly (e.g. the slider's data-kv-per-1k);
    folding the constant into a "rate" would over-count it at higher ctx.

    Returns None if any required field is missing.
    """
    if not ctx_size or not meta.block_count:
        return None
    n_layers = meta.block_count
    embed = meta.embedding_length or 0
    head_count = meta.head_count or 0
    head_count_kv = meta.head_count_kv or head_count or 0
    fallback_dim = (embed // head_count) if (head_count and embed) else 0
    k_dim = meta.key_length or fallback_dim
    v_dim = meta.value_length or meta.key_length or fallback_dim
    if head_count_kv and (k_dim or v_dim):
        per_token_elems = head_count_kv * (k_dim + v_dim)
    elif embed:
        # Last resort: assume full attention over the embedding (K + V).
        per_token_elems = 2 * embed
    else:
        return None

    # Sliding-window attention: local layers hold at most ``sliding_window``
    # tokens of KV no matter how large ctx is, and may use different geometry
    # from the global layers. Size each layer kind separately — this is the
    # difference between a real 1.2 GB and a phantom 15 GB on gemma-4-31B.
    pattern = meta.sliding_window_pattern
    window = meta.sliding_window
    if pattern and window and len(pattern) == n_layers:
        per_layer_kv = meta.head_count_kv_per_layer
        swa_k = meta.key_length_swa or k_dim
        swa_v = meta.value_length_swa or meta.key_length_swa or v_dim
        bytes_total = 0.0
        for idx, is_local in enumerate(pattern):
            if per_layer_kv and idx < len(per_layer_kv):
                heads = per_layer_kv[idx]
            else:
                heads = head_count_kv
            if is_local:
                # Capped at the window; llama.cpp still rounds the local cache
                # up to at least the window size, so use min(ctx, window).
                tokens = min(ctx_size, window)
                elems = heads * (swa_k + swa_v)
            else:
                tokens = ctx_size
                elems = heads * (k_dim + v_dim)
            bytes_total += tokens * elems * bytes_per_elem
        return bytes_total / (1024 ** 3)

    # Hybrid SSM/attention models place a full-attention layer (the only kind
    # that grows a KV cache with context) once every full_attention_interval
    # blocks; the rest are SSM/Mamba layers with a fixed-size recurrent state.
    # Count only the attention layers for the ctx-scaling KV term.
    interval = meta.full_attention_interval or 0
    if interval > 1:
        n_kv_layers = max(1, n_layers // interval)
    else:
        n_kv_layers = n_layers
    bytes_total = n_kv_layers * ctx_size * per_token_elems * bytes_per_elem

    # Non-attention (SSM) layers carry a constant recurrent + conv state that
    # does NOT scale with ctx_size and is kept in f32 by llama.cpp. Small
    # (~0.15 GB for Qwen3.6-27B) but worth including so low-ctx estimates and
    # the absolute footprint stay accurate. Best-effort: only when we have the
    # SSM geometry.
    ssm_bytes = 0.0
    if (include_recurrent and interval > 1
            and meta.ssm_inner_size and meta.ssm_state_size):
        recurrent_layers = max(0, n_layers - n_kv_layers)
        state_elems = meta.ssm_inner_size * (
            meta.ssm_state_size + (meta.ssm_conv_kernel or 0))
        ssm_bytes = recurrent_layers * state_elems * 4.0  # f32 state

    return (bytes_total + ssm_bytes) / (1024 ** 3)


def compute_n_gpu_layers(
    meta: GgufMeta,
    *,
    vram_limit_gb: float | None,
    ram_spill_policy: str,
    ram_spill_limit_gb: float | None,
    ctx_size: int | None,
    kv_cache_type: str = "",
) -> int | None:
    """Translate the basic VRAM/RAM-spill budget into ``--n-gpu-layers``.

    Returns ``None`` when no constraint is set and the engine should pick
    its own default. ``-1`` (i.e. "offload everything") is represented as
    the explicit total layer count when we know it, or skipped when we don't.

    Policy semantics:
      * ``default``  — no caps from this profile; return None.
      * ``unlimited`` — RAM spill allowed without bound; only ``vram_limit_gb``
        constrains the layer split.
      * ``limited``   — RAM spill capped at ``ram_spill_limit_gb``.
      * ``none``      — every layer must fit on the GPU. Returns total layer
        count (so the launcher offloads everything); caller must size VRAM
        accordingly or the engine will OOM.
    """
    file_gb = meta.file_size / (1024 ** 3) if meta.file_size else None
    n_layers = meta.block_count

    if not file_gb or not n_layers:
        if ram_spill_policy == "none" and (vram_limit_gb is None):
            # No info but user wants all-on-GPU: pass -1 to mean "all".
            return -1
        if vram_limit_gb is None and ram_spill_policy in ("default", "unlimited"):
            return None
        # We can't compute a precise count without layer info. Bail out
        # rather than guess wildly; the engine's default will at least run.
        return None

    per_layer_gb = file_gb / n_layers

    # KV cache budget — subtract from VRAM before we count layers.
    kv_gb = _kv_cache_gb(
        meta, ctx_size, bytes_per_elem=kv_bytes_per_elem(kv_cache_type)) or 0.0
    overhead_gb = max(_MIN_VRAM_OVERHEAD_GB,
                      (vram_limit_gb or file_gb) * _DEFAULT_VRAM_OVERHEAD_FRACTION)

    if ram_spill_policy == "none":
        # All layers on GPU.
        return n_layers + 1  # +1 covers the output layer in llama.cpp's count

    if vram_limit_gb is None and ram_spill_policy == "default":
        return None

    layers_from_vram: int | None = None
    if vram_limit_gb is not None:
        usable = max(0.0, vram_limit_gb - kv_gb - overhead_gb)
        layers_from_vram = int(usable // per_layer_gb)
        layers_from_vram = max(0, min(n_layers + 1, layers_from_vram))

    layers_from_ram: int | None = None
    if ram_spill_policy == "limited" and ram_spill_limit_gb is not None:
        max_cpu_layers = int(ram_spill_limit_gb // per_layer_gb)
        # n_gpu_layers >= total - max_cpu_layers
        layers_from_ram = max(0, (n_layers + 1) - max_cpu_layers)

    # Take the more restrictive of (VRAM-derived ceiling) and (RAM-derived floor).
    # When both are set: choose the value that satisfies both.
    if layers_from_vram is not None and layers_from_ram is not None:
        if layers_from_ram > layers_from_vram:
            # Conflict: RAM constraint says we need more on GPU than VRAM allows.
            # Honor VRAM cap; user gets a smaller GPU footprint with more spill.
            return layers_from_vram
        return max(layers_from_ram, layers_from_vram)
    if layers_from_vram is not None:
        return layers_from_vram
    if layers_from_ram is not None:
        return layers_from_ram
    return None
