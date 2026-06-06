"""GPU family detection for the engine installer.

This module figures out what GPU stack is actually usable on the current
host so the installer can pick the right torch wheels and engine
configuration. It looks at what's installed (binaries, devices, groups),
not at the raw PCI bus, because the only thing that matters for picking
an install plan is what torch will see at runtime.

The detection is intentionally cheap (no torch import, no GPU init) so
it can be called from request handlers without latency cost.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

GpuKind = Literal["amd", "nvidia", "apple", "cpu"]


@dataclass(frozen=True)
class GpuProfile:
    """What kind of accelerator we expect torch to talk to.

    ``rocm_arch`` is set for AMD only — e.g. ``"gfx1201"`` for RDNA4
    (Radeon RX 9070 / R9700). ``needs_render_group`` is True iff
    ``/dev/kfd`` is gated by a POSIX group the daemon must belong to
    (i.e. the Linux AMD case). ``render_gid`` is the actual gid number
    so the installer can check ``os.getgroups()`` against it.
    """
    kind: GpuKind
    rocm_arch: str | None = None
    needs_render_group: bool = False
    render_gid: int | None = None

    @property
    def is_amd(self) -> bool:
        return self.kind == "amd"


def _detect_render_gid() -> int | None:
    """Look up the gid of the ``render`` group, if it exists."""
    try:
        import grp
        return grp.getgrnam("render").gr_gid
    except (KeyError, ImportError):
        return None


def _rocm_smi_arch() -> str | None:
    """Ask rocm-smi which arch the first GPU exposes (e.g. gfx1201).

    Returns ``None`` if rocm-smi isn't installed or the output doesn't
    parse — both are non-fatal; the installer still picks the AMD path
    when /dev/kfd is present.
    """
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"GFX\s*Version\s*:\s*(gfx\w+)", out.stdout, re.IGNORECASE)
    return m.group(1).lower() if m else None


def rocm_lib_dirs() -> list[str]:
    """Return existing ROCm shared-library directories, newest layout first.

    The lightweight AMD torch wheels (``+rocmX.Y.lw``) don't bundle the
    full ROCm runtime — they dlopen libs like ``libroctx64.so`` from the
    system install. Those live under ``/opt/rocm`` but, on recent ROCm,
    in versioned ``core-*/lib`` subdirs that aren't on the default linker
    path. We collect them so callers can prepend them to
    ``LD_LIBRARY_PATH`` when spawning a torch process. Returns [] when no
    ROCm install is found (non-AMD hosts), so callers can apply it
    unconditionally."""
    import glob
    roots: list[str] = []
    env_root = os.environ.get("ROCM_PATH")
    if env_root:
        roots.append(env_root)
    roots += sorted(glob.glob("/opt/rocm*"), reverse=True)

    dirs: list[str] = []
    seen: set[str] = set()
    for root in roots:
        # core-*/lib holds libroctx64/libroctracer on ROCm 7.x; plain
        # lib/lib64 hold the math libs; llvm/lib has the device libs.
        candidates = sorted(glob.glob(os.path.join(root, "core-*/lib")),
                            reverse=True)
        candidates += [
            os.path.join(root, "lib"),
            os.path.join(root, "lib64"),
            os.path.join(root, "lib", "llvm", "lib"),
        ]
        for d in candidates:
            if d not in seen and os.path.isdir(d):
                seen.add(d)
                dirs.append(d)
    return dirs


def detect_gpu() -> GpuProfile:
    """Return the GPU profile for this host.

    Order: AMD → NVIDIA → Apple → CPU. We pick the first one that
    plausibly works, not the "best" — if the host has both an AMD iGPU
    and an NVIDIA dGPU it's the operator's job to pick (we can't tell
    which one torch will end up using without importing torch).
    """
    sysname = platform.system()

    # AMD ROCm/HIP on Linux: /dev/kfd is the only reliable signal that
    # the kernel-side amdgpu+kfd stack is loaded and usable.
    if sysname == "Linux" and os.path.exists("/dev/kfd"):
        render_gid = _detect_render_gid()
        return GpuProfile(
            kind="amd",
            rocm_arch=_rocm_smi_arch(),
            needs_render_group=render_gid is not None,
            render_gid=render_gid,
        )

    # NVIDIA: presence of nvidia-smi is a good proxy. We don't run it
    # — that can be slow on driver-mismatch boxes.
    if shutil.which("nvidia-smi"):
        return GpuProfile(kind="nvidia")

    # Apple Silicon (MLX / MPS). The torch wheel will be the universal
    # one; no special install needed.
    if sysname == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        return GpuProfile(kind="apple")

    return GpuProfile(kind="cpu")


# ---------------------------------------------------------------------------
# Device enumeration for per-server GPU pinning
# ---------------------------------------------------------------------------
#
# llama-server can be told to use a single GPU by hiding the others via a
# backend-specific "visible devices" env var. The tricky part on multi-GPU
# boxes (e.g. an AMD dGPU next to an Intel iGPU) is that the raw device
# *order* is not stable across launches — so storing a bare index (the way
# the image engine does) can silently select the wrong card. We instead store
# the device *name* and resolve it to an index immediately before launch.


@dataclass(frozen=True)
class LlamaDevice:
    """One accelerator as reported by ``llama-server --list-devices``.

    ``backend`` is the prefix llama prints (``Vulkan`` / ``ROCm`` / ``CUDA`` /
    ``SYCL``); it determines which "visible devices" env var pins the choice.
    ``index`` is the number in ``Vulkan1:`` etc. — the value the env var takes.
    """
    backend: str
    index: int
    name: str
    mem_total_mib: int | None = None
    mem_free_mib: int | None = None


# Maps llama's device-list backend prefix → the env var that, set to a single
# index, hides every other device so the engine runs on that GPU alone.
_VISIBLE_DEVICES_ENV: dict[str, str] = {
    "Vulkan": "GGML_VK_VISIBLE_DEVICES",
    "ROCm": "ROCR_VISIBLE_DEVICES",
    "CUDA": "CUDA_VISIBLE_DEVICES",
    "SYCL": "GGML_SYCL_VISIBLE_DEVICES",
}

# Every var that could pre-filter enumeration. We strip these before running
# --list-devices so the dropdown always shows the *full* device set, even if
# the daemon was started with one already pinned in its environment.
_ALL_VISIBLE_ENV_VARS = tuple(_VISIBLE_DEVICES_ENV.values()) + (
    "HIP_VISIBLE_DEVICES",
)

# e.g. "  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 11445 MiB free)"
_DEVICE_LINE = re.compile(
    r"^\s*(Vulkan|ROCm|CUDA|SYCL)(\d+)\s*:\s*"
    r"(?P<name>.+?)"
    r"(?:\s*\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB\s*free\))?\s*$"
)


def parse_device_list(text: str) -> list[LlamaDevice]:
    """Parse the table printed by ``llama-server --list-devices``.

    Tolerant of the surrounding noise llama emits (the radv non-conformance
    warning, the ``Available devices:`` header) — anything that doesn't match
    a ``<Backend><N>: <name>`` line is ignored.
    """
    out: list[LlamaDevice] = []
    for line in text.splitlines():
        m = _DEVICE_LINE.match(line)
        if not m:
            continue
        total = m.group("total")
        free = m.group("free")
        out.append(LlamaDevice(
            backend=m.group(1),
            index=int(m.group(2)),
            name=m.group("name").strip(),
            mem_total_mib=int(total) if total else None,
            mem_free_mib=int(free) if free else None,
        ))
    return out


def list_llama_devices(
    binary: str,
    env: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> list[LlamaDevice]:
    """Enumerate accelerators visible to ``binary`` via ``--list-devices``.

    ``env`` is the base environment the *server* would run with (so a ROCm
    build gets its LD_LIBRARY_PATH and can actually see the GPU). Any
    pre-existing "visible devices" filter is stripped so the full set is
    reported. Returns ``[]`` on any failure — enumeration is best-effort and
    must never break the settings page or a launch.
    """
    base = dict(env if env is not None else os.environ)
    for var in _ALL_VISIBLE_ENV_VARS:
        base.pop(var, None)
    try:
        out = subprocess.run(
            [binary, "--list-devices"],
            capture_output=True, text=True, timeout=timeout, env=base,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("list-devices failed for %s: %s", binary, e)
        return []
    # llama prints the table to stdout; some builds use stderr — read both.
    return parse_device_list((out.stdout or "") + "\n" + (out.stderr or ""))


# Robust GPU identity matching
# -----------------------------
# We pin a GPU by its *model name* — the stable, human-meaningful identity a
# user recognises (what GPU-Z/lspci show) — never by a raw index, which
# reshuffles when devices/cables change. But the exact string llama prints
# varies by backend and driver: the ROCm build calls a card
# "AMD Radeon AI PRO R9700" while the Vulkan/RADV build appends the GPU's Mesa
# codename → "AMD Radeon AI PRO R9700 (RADV GFX1201)". Exact-string matching
# broke on that. So we compare a *normalised* form (driver/codename and memory
# annotations stripped) and fall back to substring / token overlap, using total
# VRAM as a tiebreaker — which matches the same physical card across backends.

def clean_gpu_name(name: str) -> str:
    """Display name with backend/driver and memory annotations removed.

    e.g. ``AMD Radeon AI PRO R9700 (RADV GFX1201)`` → ``AMD Radeon AI PRO
    R9700``; ``Intel(R) Graphics (ARL)`` → ``Intel Graphics``. Original casing
    is preserved (this is what the operator sees and what we store).
    """
    s = re.sub(r"\s*\([^()]*\)", "", name or "")   # drop "(...)" groups
    return re.sub(r"\s+", " ", s).strip()


def _normalize_gpu_name(name: str) -> str:
    """Lowercased, punctuation-free form used purely for matching."""
    s = clean_gpu_name(name)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _name_tokens(name: str) -> set[str]:
    return {t for t in _normalize_gpu_name(name).split() if len(t) > 1}


def match_device(devices: list[LlamaDevice], stored_name: str,
                 stored_total_mib: int | None = None) -> LlamaDevice | None:
    """Best device matching a stored model-name selection, or None.

    Tolerant of the naming differences above and of index reordering: scores
    each device by normalised-name equality → substring → shared-token ratio,
    with total VRAM as a tiebreaker, and returns the best provided it clears a
    minimum confidence. A unique exact normalised match always wins outright.
    """
    target = _normalize_gpu_name(stored_name)
    if not target or not devices:
        return None
    exact = [d for d in devices if _normalize_gpu_name(d.name) == target]
    if len(exact) == 1:
        return exact[0]
    cands = exact or devices            # if several exact, disambiguate by VRAM
    ttoks = _name_tokens(stored_name)
    best, best_score = None, 0.0
    for d in cands:
        dn = _normalize_gpu_name(d.name)
        score = 0.0
        if dn == target:
            score += 3.0
        elif target and (target in dn or dn in target):
            score += 2.0
        dtoks = _name_tokens(d.name)
        if ttoks and dtoks:
            score += len(ttoks & dtoks) / max(len(ttoks), len(dtoks))
        if stored_total_mib and d.mem_total_mib and \
                abs(d.mem_total_mib - stored_total_mib) <= 64:
            score += 1.0
        if score > best_score:
            best, best_score = d, score
    return best if best_score >= 0.5 else None


def visible_devices_env(
    binary: str,
    device_name: str,
    env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Resolve a stored device *name* to a ``{VAR: "<index>"}`` pin.

    Matching is robust (see ``match_device``), so a card pinned under one
    backend's name still resolves under another. Returns ``None`` when no pin
    is requested or the card genuinely isn't present (logged — we'd rather run
    on all GPUs than refuse to start because a card was unplugged).
    """
    name = (device_name or "").strip()
    if not name:
        return None
    devices = list_llama_devices(binary, env=env)
    d = match_device(devices, name)
    if d is None:
        log.warning(
            "configured GPU %r not matched among %s — running on all devices",
            name, [x.name for x in devices] or "no detected devices",
        )
        return None
    var = _VISIBLE_DEVICES_ENV.get(d.backend)
    if not var:
        log.warning("no visible-devices env for backend %r", d.backend)
        return None
    if _normalize_gpu_name(d.name) != _normalize_gpu_name(name):
        log.info("GPU pin %r resolved to %r [%s%d]",
                 name, d.name, d.backend, d.index)
    return {var: str(d.index)}


def render_group_ok(profile: GpuProfile) -> bool:
    """True iff the calling process can access /dev/kfd on AMD Linux.

    Returns True for non-AMD profiles (they don't need the group at all).
    On AMD without the group, the installer surfaces a warning but still
    proceeds — the venv will be valid once groups are fixed.
    """
    if not profile.is_amd or not profile.needs_render_group:
        return True
    if profile.render_gid is None:
        return True  # no render group on this distro; /dev/kfd is presumably world-readable
    return profile.render_gid in os.getgroups()
