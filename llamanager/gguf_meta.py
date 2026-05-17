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
    context_length: int | None = None
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
                    or key.endswith(".context_length"):
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
        meta.head_count_kv = (pending.get(f"{arch}.attention.head_count_kv")
                              or meta.head_count)
        meta.context_length = pending.get(f"{arch}.context_length")
    return meta


# ---------------------------------------------------------------------------
# Sizing heuristic
# ---------------------------------------------------------------------------

# Reserved VRAM for KV cache, overhead, and small allocations. Used when the
# caller doesn't supply a ctx_size that lets us compute KV exactly.
_DEFAULT_VRAM_OVERHEAD_FRACTION = 0.12
# Minimum reserve in GB regardless of model size.
_MIN_VRAM_OVERHEAD_GB = 0.5


def _kv_cache_gb(meta: GgufMeta, ctx_size: int | None) -> float | None:
    """Estimate KV cache size in GB given a context size.

    KV cache (fp16): 2 * n_layers * ctx * n_kv_heads * head_dim * 2 bytes
                   = 4 * n_layers * ctx * embedding_per_head_kv (bytes)

    Returns None if any required field is missing.
    """
    if not ctx_size or not meta.block_count or not meta.embedding_length:
        return None
    n_layers = meta.block_count
    embed = meta.embedding_length
    # head_count gives us head_dim = embed / head_count.
    head_count = meta.head_count or 0
    head_count_kv = meta.head_count_kv or head_count or 0
    if not head_count or not head_count_kv:
        # Fallback: assume full attention (KV uses same embedding).
        kv_per_token = embed
    else:
        head_dim = embed // head_count
        kv_per_token = head_dim * head_count_kv
    bytes_total = 2 * n_layers * ctx_size * kv_per_token * 2  # K + V, fp16
    return bytes_total / (1024 ** 3)


def compute_n_gpu_layers(
    meta: GgufMeta,
    *,
    vram_limit_gb: float | None,
    ram_spill_policy: str,
    ram_spill_limit_gb: float | None,
    ctx_size: int | None,
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
    kv_gb = _kv_cache_gb(meta, ctx_size) or 0.0
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
