"""Detect and install llama-server from official llama.cpp GitHub releases
(and compatible forks)."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"
GITHUB_API_LATEST = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

# ---------------------------------------------------------------------------
# Fork registry — each entry describes a llama-server source that can be
# installed side-by-side.  The default fork ("llama.cpp") installs into
# ~/.llamanager/bin/; alternate forks install into a subdirectory so they
# never overwrite the original.
# ---------------------------------------------------------------------------
FORKS: dict[str, dict[str, str]] = {
    "llama.cpp": {
        "label": "llama.cpp (official)",
        "github_api": GITHUB_API_LATEST,
        "subdir": "",
        "description": "Official llama.cpp CPU build.",
        "url": "https://github.com/ggerganov/llama.cpp",
    },
    "atomic": {
        "label": "Atomic TurboQuant",
        "github_api": "https://api.github.com/repos/AtomicBot-ai/atomic-llama-cpp-turboquant/releases/latest",
        "subdir": "atomic",
        "description": (
            "llama.cpp fork with TurboQuant compression and Gemma\u00a04 MTP "
            "speculative decoding (\u223c30\u201350\u2009% throughput gains)."
        ),
        "url": "https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant",
    },
}


# ---------------------------------------------------------------------------
# Model → fork hints.  When a downloaded model matches one of these
# patterns, the UI shows a banner suggesting the user switch engines.
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
}


def get_engine_hint(repo: str = "", filename: str = "") -> dict[str, str] | None:
    """Return a fork hint if the repo or filename matches a known pattern."""
    for fork_id, hint in FORK_MODEL_HINTS.items():
        for prefix in hint.get("repo_prefixes", []):
            if repo.startswith(prefix):
                return {
                    "fork": fork_id,
                    "label": FORKS[fork_id]["label"],
                    "message": hint["message"],
                }
        for pattern in hint.get("file_patterns", []):
            if pattern in filename:
                return {
                    "fork": fork_id,
                    "label": FORKS[fork_id]["label"],
                    "message": hint["message"],
                }
    return None


def detect_binary(configured: str) -> str | None:
    """Return the resolved path to llama-server, or None if not found.

    Checks in order:
      1. If configured is an absolute path and the file exists.
      2. shutil.which(configured) — i.e. on PATH.
      3. ~/.llamanager/bin/llama-server[.exe].
    """
    p = Path(configured)
    if p.is_absolute() and p.exists():
        return str(p)

    found = shutil.which(configured)
    if found:
        return found

    default = _default_install_path()
    if default.exists():
        return str(default)

    return None


def detect_fork_binary(fork: str) -> str | None:
    """Return the path to a fork's llama-server if it is installed at its
    default location, or *None* otherwise."""
    path = _default_install_path(fork)
    if path.exists():
        return str(path)
    return None


def _default_install_path(fork: str = "llama.cpp") -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    base = Path.home() / ".llamanager" / "bin"
    subdir = FORKS.get(fork, {}).get("subdir", "")
    if subdir:
        base = base / subdir
    return base / f"llama-server{suffix}"


def _asset_suffix() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        return "macos-x64"
    if system == "Linux":
        if machine in ("aarch64", "arm64"):
            return "ubuntu-arm64"
        return "ubuntu-x64"
    # Windows — CPU build, no CUDA dependency
    return "win-cpu-x64"


def _asset_extension() -> str:
    """Release assets use .tar.gz on macOS/Linux and .zip on Windows."""
    return ".zip" if sys.platform == "win32" else ".tar.gz"


def install_instructions(fork: str = "llama.cpp") -> dict:
    if fork != "llama.cpp":
        fork_info = FORKS.get(fork, {})
        url = fork_info.get("url", "")
        return {
            "platform": platform.system() or "Unknown",
            "steps": [
                "# Auto-install via the button below, or:",
                f"# Build from source: {url}",
            ],
            "note": (
                "The auto-installer downloads the latest release build. "
                "For custom builds with specific GPU backends, clone the "
                "repo and build from source."
            ),
        }

    system = platform.system()
    if system == "Darwin":
        return {
            "platform": "macOS",
            "steps": [
                "brew install llama.cpp",
            ],
            "note": "This installs a Metal-accelerated build. After installing, run `llamanager serve`. Detection is automatic.",
        }
    if system == "Linux":
        return {
            "platform": "Linux",
            "steps": [
                "# Auto-install (CPU build) via the button below, or:",
                "# Download a CUDA / ROCm / Vulkan build from:",
                "# https://github.com/ggerganov/llama.cpp/releases",
            ],
            "note": "The auto-installer downloads a CPU-only build. For GPU acceleration, download a CUDA or ROCm binary from the releases page and set the path manually.",
        }
    return {
        "platform": "Windows",
        "steps": [
            "# Auto-install (AVX2 CPU build) via the button below, or:",
            "# Download a CUDA build from:",
            "# https://github.com/ggerganov/llama.cpp/releases",
        ],
        "note": "The auto-installer downloads a CPU build (no GPU). For CUDA acceleration, download a CUDA binary from the releases page and set the path manually.",
    }


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


async def install_llama_server(state: InstallState,
                               fork: str = "llama.cpp") -> None:
    """Fetch the latest release for *fork* and install llama-server.

    Updates *state* throughout so the UI can poll for progress.
    """
    fork_info = FORKS.get(fork)
    if not fork_info:
        state.error = f"Unknown fork: {fork}"
        state.status = "error"
        return

    loop = asyncio.get_running_loop()

    def _emit(line: str) -> None:
        state.lines.append(line)

    try:
        _emit(f"Fetching latest release info for {fork_info['label']}…")
        github_api = fork_info["github_api"]
        release = await loop.run_in_executor(
            None, lambda: _fetch_latest_release(github_api)
        )
        tag = release.get("tag_name", "unknown")
        _emit(f"Latest release: {tag}")

        suffix = _asset_suffix()
        ext = _asset_extension()
        assets = release.get("assets", [])
        asset = next(
            (a for a in assets
             if suffix in a["name"] and a["name"].endswith(ext)
             # Prefer the plain build (no kleidiai/vulkan/rocm/sycl qualifiers)
             and not any(q in a["name"] for q in ("kleidiai", "vulkan", "rocm", "sycl", "openvino", "cuda"))),
            None,
        )
        if asset is None:
            raise RuntimeError(
                f"No {ext} asset matching '{suffix}' found in release {tag}. "
                f"Available: {[a['name'] for a in assets]}"
            )

        _emit(f"Downloading {asset['name']} ({asset.get('size', 0) // 1024 // 1024} MB)…")

        reported: set[int] = set()

        def _progress(pct: int) -> None:
            bucket = (pct // 10) * 10
            if bucket not in reported:
                reported.add(bucket)
                _emit(f"  {bucket}% downloaded")

        zip_path = await loop.run_in_executor(
            None, lambda: _download_asset(asset["browser_download_url"], _progress)
        )
        _emit("Download complete. Extracting…")

        install_path = await loop.run_in_executor(
            None, lambda: _extract_binary(zip_path, fork)
        )
        _emit(f"Installed to {install_path}")

        state.installed_path = str(install_path)
        state.status = "done"

    except Exception as exc:
        state.error = str(exc)
        state.status = "error"


def _fetch_latest_release(github_api_url: str = GITHUB_API_LATEST) -> dict:
    req = urllib.request.Request(
        github_api_url,
        headers={"User-Agent": "llamanager/0.1 (https://github.com/llamanager/llamanager)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_asset(url: str, emit) -> Path:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "llamanager/0.1"},
    )
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


def _extract_binary(archive_path: Path, fork: str = "llama.cpp") -> Path:
    """Extract llama-server and all sibling files (shared libraries, etc.)
    from the archive into ~/.llamanager/bin/ (or a fork-specific subdirectory)."""
    dest_dir = _default_install_path(fork).parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            # Find the binary to determine its directory prefix
            candidates = [m for m in tf.getmembers()
                          if Path(m.name).name == BINARY_NAME and m.isfile()]
            if not candidates:
                raise RuntimeError(
                    f"'{BINARY_NAME}' not found inside the downloaded archive. "
                    f"Contents: {[m.name for m in tf.getmembers()[:20]]}"
                )
            entry = min(candidates, key=lambda m: len(m.name))
            prefix = str(Path(entry.name).parent)
            # Extract all files from the same directory (libs, etc.)
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                m_parent = str(Path(m.name).parent)
                if m_parent == prefix:
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

    # Create versioned-to-short symlinks for shared libraries.
    # e.g. libllama-common.0.0.9097.dylib -> libllama-common.0.dylib
    # The binary links against the short names via @rpath.
    _create_lib_symlinks(dest_dir)

    try:
        os.unlink(archive_path)
    except OSError:
        pass

    return _default_install_path(fork)


def _create_lib_symlinks(dest_dir: Path) -> None:
    """Create short-name symlinks for versioned shared libraries.

    Archives ship files like libfoo.0.11.1.dylib but the binary
    references @rpath/libfoo.0.dylib. Create the missing symlinks.
    Same pattern applies to .so files on Linux.
    """
    import re as _re
    for p in dest_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        # Match: libfoo.0.11.1.dylib or libfoo.0.0.9097.dylib or libfoo.so.0.11.1
        # Create: libfoo.0.dylib or libfoo.so.0
        if ".dylib" in name:
            # libggml-base.0.11.1.dylib -> libggml-base.0.dylib
            m = _re.match(r'^(lib[^.]+\.\d+)\.\d+(?:\.\d+)?\.dylib$', name)
            if m:
                short = m.group(1) + ".dylib"
                link = dest_dir / short
                if not link.exists():
                    link.symlink_to(p.name)
        elif ".so." in name:
            # libfoo.so.0.11.1 -> libfoo.so.0
            m = _re.match(r'^(lib[^.]+\.so\.\d+)\.\d+(?:\.\d+)?$', name)
            if m:
                short = m.group(1)
                link = dest_dir / short
                if not link.exists():
                    link.symlink_to(p.name)


def patch_config_binary(config_path: Path, binary: str) -> None:
    """Update or add `llama_server_binary = ...` in the TOML config file."""
    quoted = json.dumps(binary)  # produces a valid TOML double-quoted string
    new_line = f'llama_server_binary = {quoted}'

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(f"[server]\n{new_line}\n", encoding="utf-8")
        return

    text = config_path.read_text(encoding="utf-8")
    pattern = r'^llama_server_binary\s*=\s*.*$'
    new_text, count = re.subn(pattern, new_line, text, flags=re.MULTILINE)

    if count == 0:
        # Key not present — append under [server] section if it exists,
        # otherwise append a new [server] section.
        server_match = re.search(r'^\[server\]', new_text, flags=re.MULTILINE)
        if server_match:
            # Insert after [server] line.
            insert_at = server_match.end()
            new_text = new_text[:insert_at] + f"\n{new_line}" + new_text[insert_at:]
        else:
            new_text = new_text.rstrip("\n") + f"\n\n[server]\n{new_line}\n"

    config_path.write_text(new_text, encoding="utf-8")
