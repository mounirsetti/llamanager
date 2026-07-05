"""Opinionated dependency installer for diffusion engines.

For each image engine, we know what Python packages or external binaries
are needed. This module creates a per-engine virtual environment under
``{data_dir}/venvs/{engine}/`` and pip-installs the right packages. Once
installed, the engine's "Python executable" field is auto-populated with
the venv's interpreter path.

GPU-aware resolution: ``resolve_plan(engine, gpu)`` returns a per-engine
plan tailored to the detected accelerator. For AMD/ROCm hosts the
hidream plan installs the official AMD wheels (torch / torchvision /
triton from ``repo.radeon.com``) and known-compatible pins for the
HuggingFace stack, then optionally applies the ``use_flash_attn=False``
patch to hidream-source/models/pipeline.py — because the upstream
pipeline hardcodes flash-attn which isn't available on the AMD wheel.
On CUDA/CPU/Apple it falls back to the previous generic plan.

The installer streams stdout/stderr lines from ``python -m venv`` and
``pip install`` into the ``engine_installs.log`` column so the UI can
show live output. Progress is reported as percentage milestones rather
than parsing pip's bar (which is unreliable across pip versions).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Config
from .db import DB
from .gpu_detect import GpuProfile, detect_gpu, render_group_ok, rocm_lib_dirs

log = logging.getLogger(__name__)

MAX_LOG_BYTES = 200_000  # cap stored log size so the DB doesn't bloat

# The diffusers release both diffusion engines are pinned to and tested
# against. This is the single source of truth for the version: it's what
# new installs get, and it's the *target* the auto-update-when-idle check
# compares the venv's installed diffusers against. Bump it (with testing)
# when moving the engines to a newer diffusers release. 0.38.0 is the first
# release line that exports ``ZImagePipeline`` (verified against the
# v0.38.0 tag), so Z-Image runs on the pinned release, not git main.
DIFFUSERS_PIN = "0.38.0"

# AMD ROCm wheel index. Single curated release we test against; the
# scraper picks the latest paired (torch, torchvision, triton) under
# this prefix matching the current Python ABI.
AMD_ROCM_REL = "rocm-rel-7.2.1"
AMD_ROCM_INDEX = f"https://repo.radeon.com/rocm/manylinux/{AMD_ROCM_REL}/"

# PyTorch's CPU-only wheel index. Used when the operator explicitly picks
# the "cpu" torch backend (or on a CPU-only machine) so we install the
# slim CPU build instead of pip's default CUDA wheel.
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Engines whose AMD path we know how to wire to repo.radeon.com wheels.
AMD_WHEEL_ENGINES = {"hidream", "z_image", "krea", "asr"}
# Valid values for the UI/CLI torch-backend selector.
TORCH_BACKENDS = ("auto", "rocm", "cuda", "cpu")

# Hard-pinned fallback wheels per Python ABI. Used when the index
# scrape fails (network down, AMD changed page layout, etc.). Bump
# these by hand when AMD ships a new ROCm release we want to track.
AMD_FALLBACK_WHEELS: dict[str, list[str]] = {
    "cp312": [
        AMD_ROCM_INDEX
        + "triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl",
        AMD_ROCM_INDEX
        + "torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl",
        AMD_ROCM_INDEX
        + "torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl",
    ],
}


@dataclass(frozen=True)
class EnginePackages:
    """Generic pip install plan (backwards-compatible shape).

    Used as the baseline for engines without a GPU-aware override and as
    the public surface for the setup page (``engine_plans`` context).
    """
    engine: str
    label: str
    packages: list[str]
    extra_index_url: str | None = None
    space_mb: int = 0
    notes: str = ""
    # Engines whose existing venv may *host* this one. When any candidate's
    # venv already satisfies the heavy deps (probed via ``reuse_probe``), the
    # installer skips building a new venv and layers only ``packages`` on top,
    # pointing this engine's python at the shared venv. Empty = always build a
    # dedicated venv. Used by ASR to reuse the diffusion torch+transformers
    # stack instead of re-downloading multi-GB torch wheels.
    reuse_from: tuple[str, ...] = ()
    # Import probe that decides whether a candidate venv is reusable. Run as
    # ``python -c "import <reuse_probe>"`` in the candidate interpreter.
    reuse_probe: str = ""


@dataclass(frozen=True)
class ResolvedPlan:
    """A per-install plan after GPU detection has run.

    ``wheel_urls`` are installed first with ``--no-deps`` because pip's
    resolver can't pin across local-version tags like ``+rocm7.2.1``
    (AMD's wheels declare deps via the same tag and pip refuses to
    bridge them). ``packages`` is the regular pypi step.
    """
    engine: str
    label: str
    notes: str
    space_mb: int
    target: str
    packages: list[str] = field(default_factory=list)
    wheel_urls: list[str] = field(default_factory=list)
    extra_index_url: str | None = None
    # When set, torch is installed in a dedicated step from this index
    # (``pip install torch --index-url ...``) before the regular packages,
    # and ``torch`` is dropped from ``packages``. Used for the CPU build,
    # which pip's default index won't select.
    torch_index_url: str = ""
    # The hidream-source pipeline.py defaults use_flash_attn=True. On
    # AMD (no flash_attn wheel) we need to flip it. The installer only
    # applies this when ``options['patch_flash_attn']`` is set AND
    # ``cfg.hidream_repo`` is configured.
    supports_flash_attn_patch: bool = False


# Generic baselines. The resolver consults ENGINE_PLANS only for engines
# it doesn't have a GPU-specific override for; the setup page also reads
# this dict to render the package list before any install runs.
ENGINE_PLANS: dict[str, EnginePackages] = {
    "z_image": EnginePackages(
        engine="z_image",
        label="Z-Image (Tongyi-MAI / Z-Anime)",
        # Pinned to the tested diffusers release (DIFFUSERS_PIN). 0.38.0 is
        # the first release that ships ``ZImagePipeline``, so we no longer
        # need git main — and the pin gives the auto-updater a real version
        # to compare the installed venv against.
        packages=[
            "torch", "transformers", "accelerate", "huggingface_hub",
            "safetensors", "Pillow", "sentencepiece",
            f"diffusers=={DIFFUSERS_PIN}",
        ],
        space_mb=8500,
        notes=(
            f"Installs diffusers {DIFFUSERS_PIN} (the tested release, which "
            "ships ZImagePipeline) plus torch and the Z-Image "
            "tokenizer/text-encoder deps. Auto-detects the GPU: AMD gets the "
            "official ROCm torch wheels from repo.radeon.com, NVIDIA the CUDA "
            "wheel, otherwise a CPU build. Override with the PyTorch-build "
            "selector."
        ),
    ),
    "krea": EnginePackages(
        engine="krea",
        label="Krea 2 Turbo GGUF",
        packages=[
            "torch", "transformers", "accelerate", "huggingface_hub",
            "safetensors", "Pillow", "sentencepiece", "gguf",
            f"diffusers=={DIFFUSERS_PIN}",
        ],
        space_mb=8500,
        notes=(
            "Installs the shared torch/diffusers stack used by Krea GGUF "
            "and Z-Image. Krea needs a recent diffusers build with Qwen-Image "
            "GGUF loader support; use the version picker here if the shipped "
            "pin is behind upstream support."
        ),
    ),
    "hidream": EnginePackages(
        engine="hidream",
        label="HiDream-O1-Image",
        packages=[
            "torch", "transformers==4.57.1", "accelerate==1.13.0",
            f"diffusers=={DIFFUSERS_PIN}", "huggingface_hub", "safetensors",
            "Pillow", "einops", "sentencepiece",
        ],
        space_mb=7500,
        notes=(
            "Auto-detects the GPU family. On AMD/ROCm 7.2+ this installs "
            "the official AMD wheels (torch + torchvision + triton from "
            "repo.radeon.com) and offers to patch hidream-source's "
            "pipeline.py to disable flash-attn. On NVIDIA, installs the "
            "generic CUDA torch wheel."
        ),
    ),
    "asr": EnginePackages(
        engine="asr",
        label="Whisper (transformers) — speech-to-text",
        # No torch here: ASR reuses the diffusion venv's torch+transformers
        # (reuse_from below). If no reusable venv exists, the GPU resolver's
        # dedicated-venv fallback supplies torch (AMD wheels / CUDA / CPU).
        packages=["transformers", "accelerate", "safetensors", "numpy"],
        reuse_from=("z_image", "hidream"),
        reuse_probe="torch, transformers",
        space_mb=300,
        notes=(
            "Runs Hugging Face Whisper checkpoints for speech-to-text. Reuses "
            "the Z-Image / HiDream diffusion venv's torch + transformers when "
            "present (no multi-GB re-download); otherwise builds a dedicated "
            "venv with the GPU-appropriate torch. Audio is decoded via the "
            "system ffmpeg."
        ),
    ),
}


# ---------- GPU-aware plan resolution ----------------------------------

def _python_abi() -> str:
    """Return the cpython ABI tag of the *llamanager* interpreter.

    The per-engine venv is created with ``[sys.executable, '-m', 'venv']``
    so it inherits the same Python version. Wheels must match this tag.
    """
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


@dataclass(frozen=True)
class _AmdWheel:
    url: str
    pkg: str          # 'torch' | 'torchvision' | 'triton'
    version: tuple[int, ...]
    date: str         # raw '28-Mar-2026 03:12' string from the directory listing


_AMD_LINE_RE = re.compile(
    r'<a\s+href="(?P<href>'
    r'(?P<pkg>torch|torchvision|triton)-'
    r'(?P<ver>\d+\.\d+\.\d+)'
    r'%2B[^"]+?'
    r'-(?P<abi>cp\d+)-(?P=abi)-linux_x86_64\.whl)">'
    r'[^<]*</a>\s+'
    r'(?P<date>\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2})'
)


def _parse_amd_index(html: str, abi: str,
                     base_url: str = AMD_ROCM_INDEX) -> list[_AmdWheel]:
    """Pull torch/torchvision/triton wheel rows for the given Python ABI."""
    out: list[_AmdWheel] = []
    for m in _AMD_LINE_RE.finditer(html):
        if m.group("abi") != abi:
            continue
        ver = tuple(int(p) for p in m.group("ver").split("."))
        out.append(_AmdWheel(
            url=base_url + m.group("href"),
            pkg=m.group("pkg"),
            version=ver,
            date=m.group("date"),
        ))
    return out


def _select_amd_wheels(wheels: list[_AmdWheel]) -> list[str]:
    """Pick the latest paired (torch, torchvision, triton) set.

    AMD ships paired builds — torch, torchvision and triton with the
    same build date are guaranteed-compatible. We pick the highest-semver
    torch, then find torchvision + triton entries from the same date.
    If a pair can't be assembled, return [] (caller falls back).
    """
    torch_rows = sorted(
        (w for w in wheels if w.pkg == "torch"),
        key=lambda w: w.version, reverse=True,
    )
    if not torch_rows:
        return []
    torch_pick = torch_rows[0]
    paired: dict[str, _AmdWheel] = {"torch": torch_pick}
    for needed in ("torchvision", "triton"):
        same_date = [
            w for w in wheels
            if w.pkg == needed and w.date == torch_pick.date
        ]
        if not same_date:
            return []
        # Multiple builds on one date can happen; take the highest version.
        paired[needed] = max(same_date, key=lambda w: w.version)
    # Order matters for pip: install triton first so torch picks it up
    # at install time. (Pip processes the list in order with --no-deps.)
    return [paired["triton"].url, paired["torch"].url,
            paired["torchvision"].url]


def _amd_rocm_index(target_release: str = "") -> str:
    """URL of the AMD wheel directory to scrape.

    Honours an explicit ``target_release`` (e.g. ``"rocm-rel-7.2.1"``)
    when set; otherwise uses the curated default ``AMD_ROCM_REL``. The
    fallback wheel list is only valid for ``AMD_ROCM_REL`` — a custom
    target_release falls through to "no wheels" on scrape failure
    rather than installing the wrong ROCm version.
    """
    rel = (target_release or AMD_ROCM_REL).strip()
    return f"https://repo.radeon.com/rocm/manylinux/{rel}/"


def _resolve_amd_wheel_set(abi: str, emit, target_release: str = "") -> list[str]:
    """Try scrape, fall back to hard-pinned curated set.

    ``emit`` is the installer's log emitter — we surface which path was
    taken so the operator can see why a specific wheel was chosen.
    ``target_release`` overrides the curated default; the curated
    fallback is only used when the override matches ``AMD_ROCM_REL``.
    """
    index_url = _amd_rocm_index(target_release)
    is_custom = bool(target_release and target_release != AMD_ROCM_REL)
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(index_url)
            r.raise_for_status()
        wheels = _parse_amd_index(r.text, abi, base_url=index_url)
        urls = _select_amd_wheels(wheels)
        if urls:
            emit(f"[amd] using scraped wheel set from {index_url}")
            for u in urls:
                emit(f"[amd]   {u.rsplit('/', 1)[-1]}")
            return urls
        emit(f"[amd] scrape returned no paired set for {abi}; "
             f"falling back to curated wheels")
    except Exception as e:
        emit(f"[amd] scrape of {index_url} failed ({e}); "
             f"falling back to curated wheels")
    if is_custom:
        emit(f"[amd] custom target_release {target_release!r} doesn't have "
             f"a curated fallback; pip will resolve generic torch")
        return []
    fallback = AMD_FALLBACK_WHEELS.get(abi)
    if not fallback:
        emit(f"[amd] no fallback wheels for ABI {abi}; "
             f"will let pip resolve generic torch")
        return []
    for u in fallback:
        emit(f"[amd]   {u.rsplit('/', 1)[-1]}")
    return fallback


def _effective_backend(backend: str | None, gpu: GpuProfile) -> str:
    """Resolve the requested torch backend against the detected GPU.

    ``backend`` is one of TORCH_BACKENDS. "auto" maps the detected GPU
    family to a concrete backend; anything else is honoured verbatim so
    the operator can force a build (e.g. CPU on a flaky GPU, or ROCm
    before the render-group fix lands)."""
    b = (backend or "auto").lower()
    if b not in TORCH_BACKENDS:
        b = "auto"
    if b != "auto":
        return b
    return {"amd": "rocm", "nvidia": "cuda",
            "apple": "cpu", "cpu": "cpu"}.get(gpu.kind, "cpu")


def _is_torch_pkg(spec: str) -> bool:
    """True if a pip requirement names torch/torchvision/torchaudio."""
    name = re.split(r"[=<>!~ \[]", spec, 1)[0].strip().lower()
    return name in {"torch", "torchvision", "torchaudio"}


def resolve_plan(engine: str, gpu: GpuProfile, emit=None,
                 cfg: Config | None = None,
                 backend: str = "auto") -> ResolvedPlan | None:
    """Tailor the install plan to the detected GPU and chosen torch backend.

    Returns None if the engine has no plan at all (callers treat that as
    "no auto-install path for this engine"). ``emit`` is the install
    log line emitter; when None, fall back to a no-op so resolve_plan
    can be called from the UI for pre-install previews too. ``cfg``
    enables operator overrides like ``hidream_target_rocm_release``;
    omit it for pure-detection previews. ``backend`` is the operator's
    torch-build choice from the UI (auto/rocm/cuda/cpu).
    """
    if emit is None:
        emit = lambda _line: None  # noqa: E731

    base = ENGINE_PLANS.get(engine)
    if base is None:
        return None

    abi = _python_abi()
    eff = _effective_backend(backend, gpu)
    chosen_caption = "" if (backend or "auto") == "auto" else " (operator choice)"

    # ---- AMD ROCm wheels (repo.radeon.com) -----------------------------
    if eff == "rocm" and engine in AMD_WHEEL_ENGINES:
        target_release = ""
        if cfg is not None:
            target_release = getattr(cfg, "hidream_target_rocm_release", "") or ""
        rel = (target_release or AMD_ROCM_REL).strip()
        wheels = _resolve_amd_wheel_set(abi, emit, target_release)
        # Keep the engine's own pinned HF stack, but let the ROCm wheels
        # provide torch/torchvision (installed first, with --no-deps).
        pkgs = [p for p in base.packages if not _is_torch_pkg(p)]
        arch_caption = f", arch {gpu.rocm_arch}" if gpu.rocm_arch else ""
        override_caption = (
            " (operator override)" if target_release
            and target_release != AMD_ROCM_REL else ""
        )
        notes = (
            f"AMD ROCm path (target {rel}{override_caption}{arch_caption})"
            f"{chosen_caption}. Installs official AMD torch/torchvision/triton "
            "wheels from repo.radeon.com, then the HuggingFace stack."
        )
        if engine == "hidream":
            notes += " Recommend enabling the pipeline.py flash-attn patch below."
        if not wheels:
            notes += (" NOTE: no ROCm wheels resolved for this Python ABI — "
                      "pip will fall back to a generic torch, which won't use "
                      "the GPU.")
        return ResolvedPlan(
            engine=engine,
            label=base.label,
            notes=notes,
            space_mb=9000,
            target=f"amd-{rel.replace('-', '')}",
            packages=pkgs,
            wheel_urls=wheels,
            supports_flash_attn_patch=(engine == "hidream"),
        )

    if eff == "rocm":
        emit(f"[plan] engine {engine!r} has no AMD wheel recipe; "
             f"falling back to generic torch")

    # ---- CPU-only build ------------------------------------------------
    if eff == "cpu":
        return ResolvedPlan(
            engine=engine,
            label=base.label,
            notes=(f"CPU-only torch build{chosen_caption} (from "
                   f"{CPU_TORCH_INDEX}). Image generation will be slow; "
                   "pick a GPU backend if you have one."),
            space_mb=base.space_mb,
            target="cpu",
            packages=[p for p in base.packages if not _is_torch_pkg(p)],
            torch_index_url=CPU_TORCH_INDEX,
            supports_flash_attn_patch=(engine == "hidream"),
        )

    # ---- NVIDIA CUDA / generic fallback --------------------------------
    return ResolvedPlan(
        engine=engine,
        label=base.label,
        notes=base.notes + (
            f" Torch build: CUDA{chosen_caption}." if eff == "cuda" else ""),
        space_mb=base.space_mb,
        target=eff,
        packages=list(base.packages),
        wheel_urls=[],
        extra_index_url=base.extra_index_url,
        supports_flash_attn_patch=(engine == "hidream"),
    )


# ---------- venv path helpers (unchanged) ------------------------------

def venv_root(cfg: Config) -> Path:
    """Where per-engine venvs live."""
    return cfg.data_dir / "venvs"


def venv_python(cfg: Config, engine: str) -> Path:
    """Predict the python interpreter path for a given engine's venv.
    Returned path may not exist until an install completes."""
    root = venv_root(cfg) / engine
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


# ---------- diffusion engine version check -----------------------------
#
# A diffusion engine's "version" is the version of its pinned ``diffusers``
# dependency. The *target* is the version this llamanager build ships
# (DIFFUSERS_PIN, parsed out of the engine's install plan); the *installed*
# version is read live from the engine's venv. The auto-update-when-idle
# loop uses this to fire only on a real, known-good version bump — i.e.
# when the operator updated llamanager to a build pinning a newer diffusers
# than what their venv currently has. It never chases a moving git branch
# or jumps ahead of the tested pin.

def _ver_tuple(v: str) -> tuple[int, ...]:
    """Numeric core of a version string ('0.39.0.dev0' -> (0, 39, 0))."""
    m = re.match(r"(\d+(?:\.\d+)*)", (v or "").strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else (0,)


def _plan_diffusers_pin(engine: str) -> str | None:
    """The diffusers version literally pinned in ``engine``'s install plan
    (``DIFFUSERS_PIN``), or None if the engine doesn't install diffusers."""
    plan = ENGINE_PLANS.get(engine)
    if plan is None:
        return None
    for pkg in plan.packages:
        m = re.match(r"diffusers==([0-9][^\s;]*)", pkg.strip())
        if m:
            return m.group(1)
    return None


def diffusion_version_override(cfg: Config, engine: str) -> str | None:
    """The operator's explicit diffusers pin for ``engine`` (set when they
    install a specific version from the UI/CLI), or None to use the shipped
    pin. Stored under ``[image].diffusers_version.<engine>``."""
    overrides = getattr(cfg, "image_diffusers_version", None) or {}
    v = overrides.get(engine)
    return v or None


def diffusion_target_version(engine: str, cfg: Config | None = None) -> str | None:
    """The diffusers version auto-update should converge ``engine`` to.

    Honors the operator's explicit override (a deliberate downgrade/pin) when
    one is set in ``cfg``; otherwise falls back to the tested ``DIFFUSERS_PIN``
    the install plan ships. None if the engine doesn't install diffusers."""
    if cfg is not None:
        override = diffusion_version_override(cfg, engine)
        if override:
            return override
    return _plan_diffusers_pin(engine)


def apply_diffusers_pin(packages: list[str], version: str) -> bool:
    """Rewrite the ``diffusers==`` entry in ``packages`` to ``version`` (or
    append one if absent). Mutates in place; returns True if an existing entry
    was replaced. Used by the install path when the operator picks a specific
    diffusers version."""
    replaced = False
    for i, pkg in enumerate(packages):
        if re.match(r"diffusers==", pkg.strip()):
            packages[i] = f"diffusers=={version}"
            replaced = True
    if not replaced:
        packages.append(f"diffusers=={version}")
    return replaced


def list_diffusers_versions(*, limit: int = 30) -> dict:
    """List released diffusers versions from PyPI, newest first. Never raises;
    failures come back as ``{"versions": [], "error": "..."}``."""
    url = "https://pypi.org/pypi/diffusers/json"
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI/CLI
        return {"versions": [], "error": str(exc)}
    releases: dict = data.get("releases") or {}
    rows: list[tuple[str, str]] = []
    for ver, files in releases.items():
        if not files:
            continue
        if any(t in ver for t in ("a", "b", "rc", "dev")) and not ver.replace(".", "").isdigit():
            continue
        try:
            upload = max((f.get("upload_time_iso_8601", "") or "") for f in files)
        except ValueError:
            upload = ""
        rows.append((ver, upload))
    rows.sort(key=lambda r: r[1], reverse=True)
    return {"versions": [v for v, _ in rows[:limit]], "error": None}


def installed_diffusers_version(cfg: Config, engine: str) -> str | None:
    """Read the diffusers version installed in ``engine``'s venv, or None if
    the venv doesn't exist / diffusers isn't importable there."""
    import subprocess
    py = venv_python(cfg, engine)
    if not py.exists():
        return None
    try:
        out = subprocess.run(
            [str(py), "-c",
             "import importlib.metadata as m; print(m.version('diffusers'))"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    ver = out.stdout.strip()
    return ver or None


def diffusion_update_info(cfg: Config, engine: str) -> dict[str, Any]:
    """Compare the installed diffusers against the shipped pin for ``engine``.

    Returns ``{installed, target, has_update}``. ``has_update`` is True only
    when the engine is installed AND its diffusers is strictly older than the
    target (the operator's override if set, else the shipped pin). Not-installed
    engines never report an update (auto-update updates; it doesn't first-install).
    """
    target = diffusion_target_version(engine, cfg)
    installed = installed_diffusers_version(cfg, engine)
    has_update = bool(
        target and installed and _ver_tuple(installed) < _ver_tuple(target)
    )
    return {"installed": installed, "target": target, "has_update": has_update}


# ---------- installer ---------------------------------------------------

class EngineInstaller:
    """Owns the background installer tasks for diffusion engines."""

    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel_flags: dict[str, asyncio.Event] = {}

    # ---- public surface ----
    def start(self, engine: str, *, options: dict[str, Any] | None = None) -> str:
        if engine not in ENGINE_PLANS:
            raise ValueError(f"no install plan for engine {engine!r}")
        active = self.active_for_engine(engine)
        if active:
            raise RuntimeError(
                f"an install for {engine} is already running "
                f"(id={active['id']})"
            )
        install_id = secrets.token_urlsafe(8)
        opts_json = json.dumps(options or {}, sort_keys=True)
        self.db.execute(
            "INSERT INTO engine_installs(id, engine, kind, status, "
            "message, started_at, options_json) "
            "VALUES (?, ?, 'pip', 'pending', ?, ?, ?)",
            (install_id, engine, "queued", time.time(), opts_json),
        )
        cancel = asyncio.Event()
        self._cancel_flags[install_id] = cancel
        task = asyncio.create_task(
            self._run(install_id, engine, options or {}, cancel),
        )
        self._tasks[install_id] = task
        self.db.log_event("install_started",
                          {"id": install_id, "engine": engine,
                           "options": options or {}})
        return install_id

    def cancel(self, install_id: str) -> bool:
        ev = self._cancel_flags.get(install_id)
        if not ev:
            return False
        ev.set()
        return True

    def get(self, install_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM engine_installs WHERE id=?", (install_id,),
        )
        if not row:
            return None
        return _row_to_dict(row)

    def list_for_engine(self, engine: str, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM engine_installs WHERE engine=? "
            "ORDER BY started_at DESC LIMIT ?", (engine, limit),
        )
        return [_row_to_dict(r) for r in rows]

    def active_for_engine(self, engine: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM engine_installs WHERE engine=? "
            "AND status IN ('pending', 'running') "
            "ORDER BY started_at DESC LIMIT 1",
            (engine,),
        )
        return _row_to_dict(row) if row else None

    # ---- internals ----
    async def _run(self, install_id: str, engine: str,
                   options: dict[str, Any],
                   cancel: asyncio.Event) -> None:
        log_buf: list[str] = []

        def emit(line: str) -> None:
            log_buf.append(line)
            joined = "\n".join(log_buf)
            if len(joined) > MAX_LOG_BYTES:
                joined = joined[-MAX_LOG_BYTES:]
                log_buf[:] = joined.splitlines()
            self._set(install_id, log=joined)

        def set_progress(pct: int, message: str) -> None:
            self._set(install_id, progress_pct=int(pct), message=message)
            emit(f"[{pct:3d}%] {message}")

        try:
            self._set(install_id, status="running",
                      message="Detecting GPU and resolving plan")
            set_progress(2, "Detecting GPU")
            gpu = detect_gpu()
            emit(f"[gpu] kind={gpu.kind}"
                 + (f" arch={gpu.rocm_arch}" if gpu.rocm_arch else "")
                 + (f" render_gid={gpu.render_gid}" if gpu.render_gid is not None else ""))

            # Reuse path: if this engine can host on another engine's venv
            # (e.g. ASR on the diffusion torch+transformers stack), probe the
            # candidates first and, on a hit, install only the lightweight
            # extras there instead of rebuilding a multi-GB torch venv.
            base = ENGINE_PLANS.get(engine)
            if base and base.reuse_from:
                reused = await self._try_reuse(
                    install_id, engine, base, cancel, emit, set_progress)
                if reused is not None:
                    return  # _try_reuse finished + persisted the python path

            backend = str(options.get("torch_backend") or "auto")
            plan = resolve_plan(engine, gpu, emit, cfg=self.cfg, backend=backend)
            if plan is None:
                raise RuntimeError(f"no resolvable plan for engine {engine!r}")
            emit(f"[plan] torch backend = {backend}")

            # Optional explicit diffusers version (UI/CLI version picker —
            # upgrade or downgrade). Rewrite the diffusers== entry in-place;
            # the persisted override is what makes auto-update respect this
            # choice (see config + diffusion_target_version).
            chosen = (options.get("diffusers_version") or "").strip()
            if chosen:
                apply_diffusers_pin(plan.packages, chosen)
                emit(f"[plan] diffusers pinned to {chosen} (operator override)")

            emit(f"[plan] target={plan.target}, "
                 f"wheel_urls={len(plan.wheel_urls)}, "
                 f"packages={len(plan.packages)}")

            # Render-group preflight (warn + continue).
            if gpu.is_amd and not render_group_ok(gpu):
                emit(
                    "[warn] this daemon is not a member of the 'render' "
                    "group, so HIP will report no devices at runtime "
                    "until that is fixed. Recipe: "
                    "`sudo usermod -aG render <user>` then log out and "
                    "back in, or `sudo systemctl restart "
                    "user@<uid>.service`. The install will continue."
                )

            set_progress(5, f"Creating venv at {venv_root(self.cfg) / engine}")
            venv_dir = venv_root(self.cfg) / engine
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
            python_path = venv_python(self.cfg, engine)

            if not python_path.exists():
                rc = await self._run_subprocess(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    cancel, emit,
                )
                if cancel.is_set():
                    raise asyncio.CancelledError()
                if rc != 0:
                    raise RuntimeError(f"venv create failed with exit {rc}")
            else:
                emit(f"[ok] venv already exists at {venv_dir}")

            set_progress(15, "Upgrading pip in the new environment")
            rc = await self._run_subprocess(
                [str(python_path), "-m", "pip", "install",
                 "--upgrade", "pip", "wheel", "setuptools"],
                cancel, emit,
            )
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc != 0:
                raise RuntimeError(f"pip self-upgrade failed (exit {rc})")

            # Step A: wheel URLs (AMD ROCm path).
            if plan.wheel_urls:
                set_progress(
                    25,
                    f"Installing {len(plan.wheel_urls)} GPU wheels ({plan.target})",
                )
                argv = [
                    str(python_path), "-m", "pip", "install", "--upgrade",
                    "--no-deps", "--progress-bar", "off",
                    *plan.wheel_urls,
                ]
                rc = await self._run_subprocess(argv, cancel, emit)
                if cancel.is_set():
                    raise asyncio.CancelledError()
                if rc != 0:
                    raise RuntimeError(f"GPU wheel install failed (exit {rc})")

            # Step A2: dedicated torch index (CPU build). pip's default
            # index serves the CUDA wheel under the same version, so the
            # CPU build can only be pinned via its own --index-url.
            if plan.torch_index_url:
                set_progress(
                    30, f"Installing torch from {plan.torch_index_url}")
                argv = [
                    str(python_path), "-m", "pip", "install", "--upgrade",
                    "--progress-bar", "off",
                    "--index-url", plan.torch_index_url, "torch",
                ]
                rc = await self._run_subprocess(argv, cancel, emit)
                if cancel.is_set():
                    raise asyncio.CancelledError()
                if rc != 0:
                    raise RuntimeError(f"CPU torch install failed (exit {rc})")

            # Step B: regular pypi packages.
            set_progress(
                60,
                f"Installing {len(plan.packages)} python packages",
            )
            argv = [
                str(python_path), "-m", "pip", "install", "--upgrade",
                "--progress-bar", "off",
            ]
            if plan.extra_index_url:
                argv += ["--extra-index-url", plan.extra_index_url]
            argv += plan.packages

            rc = await self._run_subprocess(argv, cancel, emit)
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc != 0:
                raise RuntimeError(f"pip install failed (exit {rc})")

            set_progress(90, "Verifying torch imports")
            # On AMD, the torch process needs the system ROCm libs on its
            # path (same as the runtime spawn env) or the import fails with
            # a misleading "not runnable" warning even though the GPU works.
            verify_env: dict[str, str] | None = None
            rocm_dirs = rocm_lib_dirs()
            if rocm_dirs:
                prior = os.environ.get("LD_LIBRARY_PATH", "")
                verify_env = {"LD_LIBRARY_PATH": os.pathsep.join(
                    rocm_dirs + ([prior] if prior else []))}
            rc = await self._run_subprocess(
                [str(python_path), "-c",
                 "import torch; print('torch', torch.__version__, "
                 "'hip', getattr(torch.version, 'hip', None), "
                 "'cuda_available', torch.cuda.is_available())"],
                cancel, emit, env=verify_env,
            )
            if rc != 0:
                emit("[warn] torch import check failed — installed packages "
                     "may not be runnable on this machine; pointing the "
                     "engine at the new venv anyway.")

            # Persist the new venv path into the engine's config field.
            self._persist_engine_python(engine, str(python_path))

            # Optional: patch hidream-source/models/pipeline.py to flip
            # use_flash_attn True → False. Only fires when the option
            # is set, the plan supports it, and the repo path is known.
            if plan.supports_flash_attn_patch and options.get("patch_flash_attn"):
                set_progress(95, "Patching pipeline.py (use_flash_attn=False)")
                self._apply_flash_attn_patch(engine, emit)

            # Always-on: add --num_inference_steps to hidream-source's
            # argparse and thread it into the pipeline call. Without this
            # patch, profile.image_steps / req.steps make the inference
            # script exit 2 ("unrecognized arguments"). Safe to run every
            # install: it's a no-op if the flag already exists.
            if engine == "hidream":
                set_progress(96, "Patching inference.py (--num_inference_steps)")
                self._apply_inference_steps_patch(engine, emit)

            set_progress(100, f"Installed at {python_path}")
            self._set(install_id, status="done", finished_at=time.time())
            self.db.log_event("install_done", {
                "id": install_id, "engine": engine,
                "python": str(python_path),
                "target": plan.target,
            })
        except asyncio.CancelledError:
            emit("[cancelled] install stopped")
            self._set(install_id, status="cancelled",
                      finished_at=time.time())
        except Exception as e:
            log.exception("install %s failed", install_id)
            emit(f"[error] {e}")
            self._set(install_id, status="failed",
                      finished_at=time.time(), error=str(e))
            self.db.log_event("install_failed",
                              {"id": install_id, "error": str(e)})
        finally:
            self._cancel_flags.pop(install_id, None)
            self._tasks.pop(install_id, None)

    async def _try_reuse(self, install_id: str, engine: str,
                         base: EnginePackages, cancel: asyncio.Event,
                         emit, set_progress) -> str | None:
        """Reuse a sibling engine's venv to host ``engine``.

        Probes each ``base.reuse_from`` candidate's interpreter for the heavy
        deps (``base.reuse_probe``). On the first hit, installs only the
        lightweight ``base.packages`` extras there — *without* ``--upgrade``,
        so the host venv's pinned versions are left untouched — points the
        engine at that interpreter, and marks the install done. Returns the
        python path on success, or None to fall through to a dedicated venv.
        """
        # AMD: torch needs the system ROCm libs on its path to import.
        verify_env: dict[str, str] | None = None
        rocm_dirs = rocm_lib_dirs()
        if rocm_dirs:
            prior = os.environ.get("LD_LIBRARY_PATH", "")
            verify_env = {"LD_LIBRARY_PATH": os.pathsep.join(
                rocm_dirs + ([prior] if prior else []))}

        set_progress(10, "Looking for a reusable engine venv")
        probe = base.reuse_probe or "torch"
        target_py: Path | None = None
        for cand in base.reuse_from:
            cand_py = venv_python(self.cfg, cand)
            if not cand_py.exists():
                continue
            emit(f"[reuse] probing {cand} venv: import {probe}")
            rc = await self._run_subprocess(
                [str(cand_py), "-c", f"import {probe}"],
                cancel, emit, env=verify_env,
            )
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc == 0:
                target_py = cand_py
                emit(f"[reuse] reusing {cand} venv at {cand_py}")
                break
            emit(f"[reuse] {cand} venv missing deps (exit {rc}); skipping")

        if target_py is None:
            emit("[reuse] no reusable venv found; building a dedicated venv")
            return None

        # Install only the extras, never --upgrade: leave the host venv's
        # pinned packages (e.g. z_image's diffusers/transformers) intact.
        if base.packages:
            set_progress(60, f"Adding {len(base.packages)} extra package(s) "
                             "to the shared venv")
            argv = [
                str(target_py), "-m", "pip", "install",
                "--upgrade-strategy", "only-if-needed",
                "--progress-bar", "off", *base.packages,
            ]
            rc = await self._run_subprocess(argv, cancel, emit, env=verify_env)
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc != 0:
                raise RuntimeError(f"extras install failed (exit {rc})")

        self._persist_engine_python(engine, str(target_py))
        set_progress(100, f"Reusing venv at {target_py}")
        self._set(install_id, status="done", finished_at=time.time())
        self.db.log_event("install_done", {
            "id": install_id, "engine": engine,
            "python": str(target_py), "target": "reuse",
        })
        return str(target_py)

    async def _run_subprocess(self, argv: list[str], cancel: asyncio.Event,
                              emit, env: dict[str, str] | None = None) -> int:
        """Run a subprocess, streaming each stdout/stderr line into the log.
        Honours ``cancel`` by terminating the child. Returns the exit code."""
        emit(f"$ {' '.join(argv)}")
        full_env = {**os.environ, **env} if env else None
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=full_env,
        )

        async def reader() -> None:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                emit(line.decode("utf-8", errors="replace").rstrip())

        reader_task = asyncio.create_task(reader())
        try:
            while True:
                if cancel.is_set():
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                    break
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            await reader_task
        return proc.returncode if proc.returncode is not None else -1

    def _apply_flash_attn_patch(self, engine: str, emit) -> None:
        """Flip ``use_flash_attn: True`` to ``False`` in pipeline.py.

        Only meaningful for hidream right now. The upstream pipeline
        hardcodes the flag at module level, and there's no flash_attn
        wheel for AMD ROCm. Skipped (with a note) when the repo path
        isn't configured yet — the operator can re-run after setting it.
        """
        if engine != "hidream":
            return
        repo = getattr(self.cfg, "hidream_repo", None)
        if not repo:
            emit("[warn] skipping flash-attn patch: hidream_repo not set "
                 "in config. After you point 'Source folder' at your "
                 "HiDream-O1-Image checkout, re-run install (or sed it "
                 "yourself: \"use_flash_attn\": True → False in "
                 "models/pipeline.py).")
            return
        pipeline = Path(repo).expanduser() / "models" / "pipeline.py"
        if not pipeline.is_file():
            emit(f"[warn] skipping flash-attn patch: {pipeline} not found")
            return
        try:
            text = pipeline.read_text(encoding="utf-8")
        except OSError as e:
            emit(f"[warn] skipping flash-attn patch: cannot read {pipeline}: {e}")
            return

        needle_true = '"use_flash_attn": True,'
        needle_false = '"use_flash_attn": False,'
        if needle_true not in text:
            if needle_false in text:
                emit(f"[ok] flash-attn patch already applied to {pipeline}")
            else:
                emit(f"[warn] flash-attn patch needle not found in {pipeline}; "
                     "upstream may have changed. Skipped.")
            return

        # Keep a one-shot backup the first time we touch this file.
        bak = pipeline.with_suffix(pipeline.suffix + ".bak")
        try:
            if not bak.exists():
                bak.write_text(text, encoding="utf-8")
            pipeline.write_text(
                text.replace(needle_true, needle_false), encoding="utf-8",
            )
            emit(f"[ok] patched {pipeline}: \"use_flash_attn\": True → False "
                 f"(backup at {bak.name})")
        except OSError as e:
            emit(f"[warn] flash-attn patch failed to write {pipeline}: {e}")

    def _apply_inference_steps_patch(self, engine: str, emit) -> None:
        """Add ``--num_inference_steps`` to hidream-source/inference.py.

        Vanilla hidream-source rejects the flag with ``error:
        unrecognized arguments`` and exits 2 before loading the model.
        We do two things:

        1. Add ``parser.add_argument("--num_inference_steps", type=int,
           default=None, ...)`` after the first existing add_argument.
        2. Find pipeline call sites of the shape ``num_inference_steps=
           <ident-or-int>`` and wrap them so the CLI value wins when set
           and the upstream default kicks in otherwise.

        Idempotent — re-running on an already-patched file is a no-op
        (step 1 detects the existing flag, step 2's regex won't match
        the wrapped form a second time).
        """
        if engine != "hidream":
            return
        repo = getattr(self.cfg, "hidream_repo", None)
        if not repo:
            emit("[warn] skipping --num_inference_steps patch: "
                 "hidream_repo not set in config.")
            return
        inference = Path(repo).expanduser() / "inference.py"
        if not inference.is_file():
            emit(f"[warn] skipping --num_inference_steps patch: "
                 f"{inference} not found")
            return
        try:
            text = inference.read_text(encoding="utf-8")
        except OSError as e:
            emit(f"[warn] cannot read {inference}: {e}")
            return

        original = text
        already_declared = ("--num_inference_steps" in text
                            or "--num-inference-steps" in text)

        # Step 1: inject argparse registration if missing. Anchor on the
        # first existing parser.add_argument(...) line so we land in the
        # right argparse block with the right indentation.
        if not already_declared:
            anchor = re.search(
                r'(^[ \t]*)parser\.add_argument\([^\n]*\n',
                text, flags=re.MULTILINE,
            )
            if not anchor:
                emit(f"[warn] --num_inference_steps patch: no "
                     f"parser.add_argument() found in {inference}; skipped.")
                return
            indent = anchor.group(1)
            injection = (
                f'{indent}parser.add_argument('
                '"--num_inference_steps", type=int, default=None, '
                'help="Override default step count (llamanager patch)")\n'
            )
            insert_at = anchor.end()
            text = text[:insert_at] + injection + text[insert_at:]

        # Step 2: route args.num_inference_steps into pipeline call sites.
        # Match `num_inference_steps=<ident-or-int>` but only when not
        # already wrapped (we look for the opening paren of our wrapper).
        call_re = re.compile(
            r'(num_inference_steps\s*=\s*)'
            r'(?!\(\s*args\.num_inference_steps\b)'  # not already wrapped
            r'([A-Za-z_][A-Za-z_0-9.]*|\d+)',
        )
        def _wrap(m: re.Match) -> str:
            return (f"{m.group(1)}(args.num_inference_steps "
                    f"if args.num_inference_steps is not None "
                    f"else {m.group(2)})")
        text, n_wraps = call_re.subn(_wrap, text)

        if text == original:
            emit(f"[ok] --num_inference_steps patch already applied to "
                 f"{inference}")
            return

        bak = inference.with_suffix(inference.suffix + ".bak")
        try:
            if not bak.exists():
                bak.write_text(original, encoding="utf-8")
            inference.write_text(text, encoding="utf-8")
            actions = []
            if not already_declared:
                actions.append("added argparse flag")
            if n_wraps:
                actions.append(f"wired {n_wraps} pipeline call(s)")
            emit(f"[ok] patched {inference}: "
                 f"{' + '.join(actions) or 'no-op'} "
                 f"(backup at {bak.name})")
        except OSError as e:
            emit(f"[warn] --num_inference_steps patch failed to write "
                 f"{inference}: {e}")

    def _set(self, install_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(
            f"UPDATE engine_installs SET {cols} WHERE id=?",
            (*fields.values(), install_id),
        )

    def _persist_engine_python(self, engine: str, python_path: str) -> None:
        """Save the venv's python path back to config.toml so the engine
        is ready to use without the operator having to copy/paste the path."""
        from .config import update_image_config
        kwargs: dict[str, Any] = {}
        if engine in ("z_image", "krea"):
            kwargs["z_image_python"] = python_path
            self.cfg.z_image_python = python_path
        elif engine == "hidream":
            kwargs["hidream_python"] = python_path
            self.cfg.hidream_python = python_path
        elif engine == "asr":
            kwargs["asr_python"] = python_path
            self.cfg.asr_python = python_path
        else:
            return
        try:
            update_image_config(self.cfg.config_path, **kwargs)
        except Exception:
            log.exception("failed to persist %s python path to config", engine)


def _row_to_dict(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "engine": row["engine"],
        "kind": row["kind"],
        "status": row["status"],
        "progress_pct": row["progress_pct"],
        "message": row["message"] or "",
        "log": row["log"] or "",
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
        "options_json": (row["options_json"] if "options_json" in row.keys()
                         else None),
    }
