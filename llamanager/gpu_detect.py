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
