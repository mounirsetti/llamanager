"""Detect and install the inference engine (llama.cpp builds, llama.cpp
forks, or MLX) that llamanager will manage.

A *variant* is the pair (source, backend). Each variant installs into its
own subdirectory under ~/.llamanager/bin/ so multiple builds coexist; the
active one is selected via the configured `llama_server_binary` path.

Two engine families are supported:

- ``"llama"`` — official llama.cpp and compatible forks. Installed by
  downloading a release archive from GitHub. The launched binary is
  ``llama-server``.
- ``"mlx"`` — Apple's MLX engine (``mlx-lm`` package). Installed via pip
  into a per-variant virtual environment. The launched binary is the
  venv Python; the server runs as ``python -m mlx_lm server``.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"

# ---------------------------------------------------------------------------
# Sources — upstream engines + repositories. Each declares an engine_type
# which determines the install mechanism and how server_manager spawns it.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "llama.cpp": {
        "label": "llama.cpp",
        "engine_type": "llama",
        "github_api": "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        "url": "https://github.com/ggml-org/llama.cpp",
        "description": "Official llama.cpp builds. Wide format support (GGUF), broad backend choice.",
    },
    "atomic": {
        "label": "Atomic TurboQuant",
        "engine_type": "llama",
        "github_api": "https://api.github.com/repos/AtomicBot-ai/atomic-llama-cpp-turboquant/releases/latest",
        "url": "https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant",
        "description": (
            "llama.cpp fork with TurboQuant compression and Gemma 4 MTP "
            "speculative decoding (~30–50 % throughput gains)."
        ),
    },
    "mlx": {
        "label": "MLX",
        "engine_type": "mlx",
        "pypi_url": "https://pypi.org/pypi/mlx-lm/json",
        "pip_package": "mlx-lm",
        "url": "https://github.com/ml-explore/mlx-lm",
        "description": (
            "Apple's native ML framework for Apple Silicon. Loads MLX-format "
            "checkpoints from Hugging Face. Best-in-class throughput on M-series chips."
        ),
    },
}


# ---------------------------------------------------------------------------
# Backends. ``platforms`` constrains where each is offered; ``sources``
# (optional) further constrains which sources offer the backend. The
# absence of a ``sources`` key means every source supports it.
# ---------------------------------------------------------------------------
BACKENDS: dict[str, dict] = {
    "cpu": {
        "label": "CPU",
        "description": "Portable CPU-only build. Works anywhere, no GPU required.",
        "platforms": ["win", "linux", "macos"],
        "sources": ["llama.cpp", "atomic"],
    },
    "cuda": {
        "label": "CUDA",
        "vendor": "NVIDIA",
        "description": "GPU acceleration for NVIDIA cards. Requires a recent driver.",
        "platforms": ["win", "linux"],
        "sources": ["llama.cpp", "atomic"],
    },
    "vulkan": {
        "label": "Vulkan",
        "vendor": "cross-vendor",
        "description": "GPU acceleration via Vulkan. Works on NVIDIA, AMD and Intel GPUs.",
        "platforms": ["win", "linux"],
        "sources": ["llama.cpp", "atomic"],
    },
    "hip": {
        "label": "HIP / ROCm",
        "vendor": "AMD",
        "description": "AMD GPU acceleration via HIP. Requires a supported Radeon GPU.",
        "platforms": ["win", "linux"],
        "sources": ["llama.cpp", "atomic"],
    },
    "sycl": {
        "label": "SYCL",
        "vendor": "Intel",
        "description": "Intel GPU acceleration via oneAPI / SYCL.",
        "platforms": ["win", "linux"],
        "sources": ["llama.cpp", "atomic"],
    },
    "metal": {
        "label": "Metal",
        "vendor": "Apple",
        "description": "Apple Silicon GPU. Built into the macOS arm64 build.",
        "platforms": ["macos"],
        "sources": ["llama.cpp", "atomic"],
    },
    "apple-silicon": {
        "label": "Apple Silicon",
        "vendor": "Apple",
        "description": "Native Apple Silicon (Metal + Neural Engine) via MLX.",
        "platforms": ["macos"],
        "sources": ["mlx"],
    },
}

GPU_TOKENS = ("cuda", "vulkan", "hip", "rocm", "sycl", "kompute",
              "openvino", "kleidiai", "musa")

_BACKEND_TOKENS: dict[str, tuple[str, ...]] = {
    "cuda": ("cuda",),
    "vulkan": ("vulkan",),
    "hip": ("hip",),
    "sycl": ("sycl",),
}


# ---------------------------------------------------------------------------
# Model → source hints. When a downloaded model matches one of these
# patterns, the UI shows a banner suggesting the user switch sources.
# ---------------------------------------------------------------------------
FORK_MODEL_HINTS: dict[str, dict[str, list[str] | str]] = {
    "atomic": {
        "repo_prefixes": ["AtomicBot-ai/"],
        "file_patterns": ["TQ1_", "TQ2_"],
        "message": (
            "This model uses TurboQuant quantization. For best results, "
            "install and switch to the Atomic TurboQuant engine."
        ),
    },
    "mlx": {
        "repo_prefixes": ["mlx-community/"],
        "file_patterns": [],
        "message": (
            "This is an MLX-format model. Install and switch to the MLX "
            "engine to serve it on Apple Silicon."
        ),
    },
}


def get_engine_hint(repo: str = "", filename: str = "") -> dict[str, str] | None:
    """Return a source hint if the repo or filename matches a known pattern."""
    for source_id, hint in FORK_MODEL_HINTS.items():
        for prefix in hint.get("repo_prefixes", []):
            if repo.startswith(prefix):
                return {
                    "source": source_id,
                    "fork": source_id,
                    "label": SOURCES[source_id]["label"],
                    "message": hint["message"],
                }
        for pattern in hint.get("file_patterns", []):
            if pattern in filename:
                return {
                    "source": source_id,
                    "fork": source_id,
                    "label": SOURCES[source_id]["label"],
                    "message": hint["message"],
                }
    return None


# ---------------------------------------------------------------------------
# Variant identity
# ---------------------------------------------------------------------------
def variant_id(source: str, backend: str) -> str:
    return f"{source}-{backend}"


def parse_variant_id(vid: str) -> tuple[str, str] | None:
    for src in SOURCES:
        prefix = f"{src}-"
        if vid.startswith(prefix):
            backend = vid[len(prefix):]
            if backend in BACKENDS:
                return src, backend
    return None


def engine_type_for(source: str) -> str:
    """Return the engine_type for a given source, defaulting to 'llama'."""
    return SOURCES.get(source, {}).get("engine_type", "llama")


def current_platform() -> str:
    s = platform.system()
    if s == "Darwin":
        return "macos"
    if s == "Linux":
        return "linux"
    return "win"


def _backend_offered_for(source: str, backend: str) -> bool:
    be = BACKENDS.get(backend)
    if not be:
        return False
    if current_platform() not in be["platforms"]:
        return False
    sources = be.get("sources")
    if sources is not None and source not in sources:
        return False
    return True


def list_variants() -> list[dict]:
    """Return every valid (source, backend) combination for this platform."""
    plat = current_platform()
    machine = platform.machine().lower()
    out: list[dict] = []
    for src_id, src_meta in SOURCES.items():
        for be_id, be_meta in BACKENDS.items():
            if plat not in be_meta["platforms"]:
                continue
            allowed = be_meta.get("sources")
            if allowed is not None and src_id not in allowed:
                continue
            # MLX only runs on Apple Silicon
            if src_meta["engine_type"] == "mlx" and machine not in ("arm64", "aarch64"):
                continue
            out.append({
                "id": variant_id(src_id, be_id),
                "source": src_id,
                "backend": be_id,
                "engine_type": src_meta["engine_type"],
                "source_label": src_meta["label"],
                "source_url": src_meta["url"],
                "source_description": src_meta["description"],
                "backend_label": be_meta["label"],
                "backend_vendor": be_meta.get("vendor", ""),
                "backend_description": be_meta["description"],
            })
    return out


# ---------------------------------------------------------------------------
# Hardware detection — pick a sensible default backend.
# ---------------------------------------------------------------------------
def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _command_succeeds(args: list[str], timeout: float = 3.0) -> bool:
    try:
        subprocess.run(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, check=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def detect_default_backend() -> str:
    """Best-effort suggestion. Returns 'cpu' if nothing better is detected."""
    plat = current_platform()
    if plat == "macos":
        if platform.machine().lower() in ("arm64", "aarch64"):
            return "metal"
        return "cpu"

    if _has_command("nvidia-smi") and _command_succeeds(["nvidia-smi", "-L"]):
        return "cuda"
    if plat == "linux" and _has_command("rocm-smi"):
        return "hip"
    if _has_command("vulkaninfo"):
        return "vulkan"
    return "cpu"


def detect_default_source() -> str:
    """Suggested source for this hardware. MLX on Apple Silicon, else llama.cpp."""
    if (current_platform() == "macos"
            and platform.machine().lower() in ("arm64", "aarch64")):
        return "mlx"
    return "llama.cpp"


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
def _bin_root() -> Path:
    return Path.home() / ".llamanager" / "bin"


def variant_dir(source: str, backend: str) -> Path:
    return _bin_root() / variant_id(source, backend)


def variant_install_path(source: str, backend: str) -> Path:
    """The path llamanager will spawn for a variant.

    For llama.cpp engines, this is the ``llama-server`` executable. For
    MLX, it's the venv Python interpreter (server_manager runs it as
    ``python -m mlx_lm server …``).
    """
    if engine_type_for(source) == "mlx":
        if sys.platform == "win32":
            return variant_dir(source, backend) / "venv" / "Scripts" / "python.exe"
        return variant_dir(source, backend) / "venv" / "bin" / "python"
    suffix = ".exe" if sys.platform == "win32" else ""
    return variant_dir(source, backend) / f"llama-server{suffix}"


def detect_variant_binary(source: str, backend: str) -> str | None:
    path = variant_install_path(source, backend)
    return str(path) if path.exists() else None


def detect_binary(configured: str) -> str | None:
    """Resolve the configured binary path through PATH, installed variants,
    and the legacy install location. Returns ``None`` if nothing exists."""
    p = Path(configured)
    if p.is_absolute() and p.exists():
        return str(p)

    found = shutil.which(configured)
    if found:
        return found

    for variant in list_variants():
        path = variant_install_path(variant["source"], variant["backend"])
        if path.exists():
            return str(path)

    suffix = ".exe" if sys.platform == "win32" else ""
    legacy = _bin_root() / f"llama-server{suffix}"
    if legacy.exists():
        return str(legacy)
    return None


def detect_variant_for_binary(binary_path: str) -> tuple[str, str] | None:
    """If ``binary_path`` lives inside a known variant directory, return
    (source, backend). Otherwise None — the binary is external."""
    try:
        p = Path(binary_path).resolve()
    except OSError:
        return None
    bin_root = _bin_root().resolve()
    try:
        rel = p.relative_to(bin_root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    vid = rel.parts[0]
    return parse_variant_id(vid)


# ---------------------------------------------------------------------------
# Install metadata — per-variant marker capturing installed version so the
# "check for updates" button has something to compare against.
# ---------------------------------------------------------------------------
def _marker_path(source: str, backend: str) -> Path:
    return variant_dir(source, backend) / ".installed.json"


def read_install_meta(source: str, backend: str) -> dict | None:
    p = _marker_path(source, backend)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_install_meta(source: str, backend: str, **fields) -> None:
    p = _marker_path(source, backend)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": source, "backend": backend, **fields}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# GitHub release fetching + asset matching
# ---------------------------------------------------------------------------
def _platform_tags() -> list[str]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return ["macos-arm64"]
        return ["macos-x64"]
    if system == "Linux":
        if machine in ("aarch64", "arm64"):
            return ["ubuntu-arm64", "linux-arm64"]
        return ["ubuntu-x64", "linux-x64"]
    if machine in ("arm64", "aarch64"):
        return ["win-arm64"]
    return ["win-x64", "win-amd64"]


def _asset_extensions() -> tuple[str, ...]:
    if sys.platform == "win32":
        return (".zip",)
    return (".tar.gz", ".zip")


def _select_asset(assets: list[dict], backend: str) -> dict | None:
    tags = _platform_tags()
    exts = _asset_extensions()
    backend_tokens = _BACKEND_TOKENS.get(backend, ())

    def matches(name: str) -> bool:
        low = name.lower()
        if not any(low.endswith(ext) for ext in exts):
            return False
        if not any(tag in low for tag in tags):
            return False
        if backend_tokens:
            if not all(tok in low for tok in backend_tokens):
                return False
        else:
            if any(tok in low for tok in GPU_TOKENS):
                return False
        return True

    candidates = [a for a in assets if matches(a.get("name", ""))]
    if not candidates:
        return None
    candidates.sort(key=lambda a: (len(a["name"]), a["name"]))
    return candidates[0]


def _fetch_latest_release(github_api_url: str) -> dict:
    req = urllib.request.Request(
        github_api_url,
        headers={"User-Agent": "llamanager/0.1"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_pypi_latest(pypi_url: str) -> str:
    req = urllib.request.Request(
        pypi_url, headers={"User-Agent": "llamanager/0.1"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["info"]["version"]


# ---------------------------------------------------------------------------
# Update check
# ---------------------------------------------------------------------------
@dataclass
class UpdateInfo:
    installed: str | None
    latest: str | None
    has_update: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "latest": self.latest,
            "has_update": self.has_update,
            "error": self.error,
        }


def check_for_update(source: str, backend: str) -> UpdateInfo:
    """Compare the installed version of a variant against upstream's latest.

    Returns an :class:`UpdateInfo`. Never raises — failures are reported
    via ``UpdateInfo.error``.
    """
    src_meta = SOURCES.get(source)
    if not src_meta:
        return UpdateInfo(None, None, False, f"unknown source: {source}")
    if detect_variant_binary(source, backend) is None:
        return UpdateInfo(None, None, False, "not installed")

    meta = read_install_meta(source, backend) or {}
    installed = meta.get("version")
    try:
        if src_meta["engine_type"] == "mlx":
            latest = _fetch_pypi_latest(src_meta["pypi_url"])
        else:
            release = _fetch_latest_release(src_meta["github_api"])
            latest = release.get("tag_name") or release.get("name")
    except Exception as exc:
        return UpdateInfo(installed, None, False, str(exc))

    has_update = bool(latest and installed and latest != installed)
    # If we have a latest but no installed marker, treat as updatable so the
    # user can record the current version by reinstalling.
    if latest and not installed:
        has_update = True
    return UpdateInfo(installed, latest, has_update)


# ---------------------------------------------------------------------------
# Install state + entrypoint
# ---------------------------------------------------------------------------
@dataclass
class InstallState:
    status: str = "idle"  # idle | running | done | error
    lines: list[str] = field(default_factory=list)
    installed_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "lines": list(self.lines),
            "installed_path": self.installed_path,
            "error": self.error,
        }


async def install_variant(state: InstallState, source: str, backend: str) -> None:
    """Install (source, backend) and update *state* throughout."""
    src_meta = SOURCES.get(source)
    be_meta = BACKENDS.get(backend)
    if src_meta is None:
        state.error = f"Unknown source: {source}"
        state.status = "error"
        return
    if be_meta is None:
        state.error = f"Unknown backend: {backend}"
        state.status = "error"
        return
    if not _backend_offered_for(source, backend):
        state.error = (
            f"{src_meta['label']} does not ship a {be_meta['label']} "
            f"build for this platform."
        )
        state.status = "error"
        return

    if src_meta["engine_type"] == "mlx":
        await _install_mlx(state, source, backend, src_meta)
    else:
        await _install_llama(state, source, backend, src_meta, be_meta)


# ---- llama.cpp install (GitHub release archive) ---------------------------
async def _install_llama(state: InstallState, source: str, backend: str,
                         src_meta: dict, be_meta: dict) -> None:
    loop = asyncio.get_running_loop()

    def _emit(line: str) -> None:
        state.lines.append(line)

    try:
        _emit(f"Fetching latest release for {src_meta['label']} ({be_meta['label']})…")
        release = await loop.run_in_executor(
            None, lambda: _fetch_latest_release(src_meta["github_api"])
        )
        tag = release.get("tag_name", "unknown")
        _emit(f"Latest release: {tag}")

        asset = _select_asset(release.get("assets", []), backend)
        if asset is None:
            names = [a.get("name", "") for a in release.get("assets", [])]
            raise RuntimeError(
                f"No release asset found for {be_meta['label']} on this "
                f"platform in release {tag}. Available: {names}"
            )

        size_mb = (asset.get("size", 0) or 0) // 1024 // 1024
        _emit(f"Downloading {asset['name']} ({size_mb} MB)…")

        reported: set[int] = set()

        def _progress(pct: int) -> None:
            bucket = (pct // 10) * 10
            if bucket not in reported:
                reported.add(bucket)
                _emit(f"  {bucket}% downloaded")

        archive_path = await loop.run_in_executor(
            None, lambda: _download_asset(asset["browser_download_url"], _progress)
        )
        _emit("Download complete. Extracting…")

        install_path = await loop.run_in_executor(
            None, lambda: _extract_binary(archive_path, source, backend)
        )

        write_install_meta(source, backend, version=tag, engine_type="llama",
                           asset=asset["name"])
        _emit(f"Installed to {install_path}")

        state.installed_path = str(install_path)
        state.status = "done"

    except Exception as exc:
        state.error = str(exc)
        state.status = "error"


# ---- MLX install (pip into per-variant venv) ------------------------------
async def _install_mlx(state: InstallState, source: str, backend: str,
                       src_meta: dict) -> None:
    loop = asyncio.get_running_loop()

    def _emit(line: str) -> None:
        state.lines.append(line)

    try:
        if current_platform() != "macos":
            raise RuntimeError("MLX only runs on macOS Apple Silicon.")
        if platform.machine().lower() not in ("arm64", "aarch64"):
            raise RuntimeError("MLX requires Apple Silicon (arm64) hardware.")

        venv_dir = variant_dir(source, backend) / "venv"
        _emit(f"Creating virtual environment at {venv_dir}…")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        # Use the running interpreter to bootstrap the venv.
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True, capture_output=True, text=True,
            ),
        )

        python = variant_install_path(source, backend)
        _emit(f"Upgrading pip…")
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [str(python), "-m", "pip", "install", "--upgrade", "pip"],
                check=True, capture_output=True, text=True,
            ),
        )

        _emit(f"Installing {src_meta['pip_package']} from PyPI (this may take a few minutes)…")
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [str(python), "-m", "pip", "install", "--upgrade",
                 src_meta["pip_package"]],
                check=True, capture_output=True, text=True,
            ),
        )
        # Surface the tail of pip's output for the operator.
        for tail_line in result.stdout.splitlines()[-3:]:
            _emit(f"  {tail_line}")

        # Resolve the actually-installed version for the metadata marker.
        version_proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [str(python), "-c",
                 "import importlib.metadata as m; print(m.version('mlx-lm'))"],
                capture_output=True, text=True, check=True,
            ),
        )
        version = version_proc.stdout.strip() or "unknown"
        write_install_meta(source, backend, version=version, engine_type="mlx",
                           pip_package=src_meta["pip_package"])

        _emit(f"Installed mlx-lm {version} to {python}")
        state.installed_path = str(python)
        state.status = "done"

    except subprocess.CalledProcessError as exc:
        # pip / venv emits useful errors on stderr.
        stderr_tail = (exc.stderr or "").splitlines()[-5:]
        state.error = f"command failed: {' '.join(exc.cmd)}"
        for line in stderr_tail:
            state.lines.append(f"  {line}")
        state.status = "error"
    except Exception as exc:
        state.error = str(exc)
        state.status = "error"


def _download_asset(url: str, emit) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "llamanager/0.1"})
    suffix = ".zip" if url.endswith(".zip") else ".tar.gz"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_pct = -1
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct:
                        emit(pct)
                        last_pct = pct
        tmp.flush()
        return Path(tmp.name)
    except Exception:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    finally:
        tmp.close()


def _extract_binary(archive_path: Path, source: str, backend: str) -> Path:
    target_path = variant_install_path(source, backend)
    dest_dir = target_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if str(archive_path).lower().endswith(".tar.gz") or archive_path.suffix == ".gz":
        with tarfile.open(archive_path, "r:gz") as tf:
            candidates = [m for m in tf.getmembers()
                          if Path(m.name).name == BINARY_NAME and m.isfile()]
            if not candidates:
                raise RuntimeError(
                    f"'{BINARY_NAME}' not found inside the downloaded archive. "
                    f"Contents: {[m.name for m in tf.getmembers()[:20]]}"
                )
            entry = min(candidates, key=lambda m: len(m.name))
            prefix = str(Path(entry.name).parent)
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if str(Path(m.name).parent) == prefix:
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    out_path = dest_dir / Path(m.name).name
                    with open(out_path, "wb") as out:
                        out.write(src.read())
                    if sys.platform != "win32":
                        os.chmod(out_path, 0o755)
    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            candidates = [n for n in zf.namelist() if Path(n).name == BINARY_NAME]
            if not candidates:
                raise RuntimeError(
                    f"'{BINARY_NAME}' not found inside the downloaded zip. "
                    f"Contents: {zf.namelist()[:20]}"
                )
            entry = min(candidates, key=lambda n: len(n))
            prefix = str(Path(entry).parent)
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                if str(Path(name).parent) == prefix:
                    out_path = dest_dir / Path(name).name
                    with zf.open(name) as src, open(out_path, "wb") as out:
                        out.write(src.read())
                    if sys.platform != "win32":
                        os.chmod(out_path, 0o755)

    _create_lib_symlinks(dest_dir)

    try:
        os.unlink(archive_path)
    except OSError:
        pass

    return target_path


def _create_lib_symlinks(dest_dir: Path) -> None:
    for p in dest_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if ".dylib" in name:
            m = re.match(r'^(lib[^.]+\.\d+)\.\d+(?:\.\d+)?\.dylib$', name)
            if m:
                short = m.group(1) + ".dylib"
                link = dest_dir / short
                if not link.exists():
                    link.symlink_to(p.name)
        elif ".so." in name:
            m = re.match(r'^(lib[^.]+\.so\.\d+)\.\d+(?:\.\d+)?$', name)
            if m:
                short = m.group(1)
                link = dest_dir / short
                if not link.exists():
                    link.symlink_to(p.name)


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------
def patch_config_binary(config_path: Path, binary: str, engine: str | None = None) -> None:
    """Update ``llama_server_binary`` (and optionally ``llama_server_engine``)
    in the TOML config file."""
    binary_line = f'llama_server_binary = {json.dumps(binary)}'
    engine_line = f'llama_server_engine = {json.dumps(engine)}' if engine else None

    def _upsert(text: str, key: str, line: str) -> str:
        pattern = rf'^{re.escape(key)}\s*=\s*.*$'
        new_text, count = re.subn(pattern, line, text, flags=re.MULTILINE)
        if count:
            return new_text
        # Not present — append under [server] section.
        server_match = re.search(r'^\[server\]', text, flags=re.MULTILINE)
        if server_match:
            insert_at = server_match.end()
            return text[:insert_at] + f"\n{line}" + text[insert_at:]
        return text.rstrip("\n") + f"\n\n[server]\n{line}\n"

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        body = f"[server]\n{binary_line}\n"
        if engine_line:
            body += f"{engine_line}\n"
        config_path.write_text(body, encoding="utf-8")
        return

    text = config_path.read_text(encoding="utf-8")
    text = _upsert(text, "llama_server_binary", binary_line)
    if engine_line:
        text = _upsert(text, "llama_server_engine", engine_line)
    config_path.write_text(text, encoding="utf-8")
