"""Detect and install llama-server from official llama.cpp GitHub releases."""
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


def _default_install_path() -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path.home() / ".llamanager" / "bin" / f"llama-server{suffix}"


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


def install_instructions() -> dict:
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


async def install_llama_server(state: InstallState) -> None:
    """Fetch the latest llama.cpp release and install llama-server.

    Updates `state` throughout so the UI can poll for progress.
    """
    loop = asyncio.get_running_loop()

    def _emit(line: str) -> None:
        state.lines.append(line)

    try:
        _emit("Fetching latest release info from GitHub…")
        release = await loop.run_in_executor(None, _fetch_latest_release)
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

        install_path = await loop.run_in_executor(None, lambda: _extract_binary(zip_path))
        _emit(f"Installed to {install_path}")

        state.installed_path = str(install_path)
        state.status = "done"

    except Exception as exc:
        state.error = str(exc)
        state.status = "error"


def _fetch_latest_release() -> dict:
    req = urllib.request.Request(
        GITHUB_API_LATEST,
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


def _extract_binary(archive_path: Path) -> Path:
    dest = _default_install_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            candidates = [m for m in tf.getmembers()
                          if Path(m.name).name == BINARY_NAME and m.isfile()]
            if not candidates:
                raise RuntimeError(
                    f"'{BINARY_NAME}' not found inside the downloaded archive. "
                    f"Contents: {[m.name for m in tf.getmembers()[:20]]}"
                )
            entry = min(candidates, key=lambda m: len(m.name))
            src = tf.extractfile(entry)
            if src is None:
                raise RuntimeError(f"Could not read {entry.name} from archive")
            with open(dest, "wb") as out:
                out.write(src.read())
    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            candidates = [n for n in zf.namelist() if Path(n).name == BINARY_NAME]
            if not candidates:
                raise RuntimeError(
                    f"'{BINARY_NAME}' not found inside the downloaded zip. "
                    f"Contents: {zf.namelist()[:20]}"
                )
            entry = min(candidates, key=lambda n: len(n))
            with zf.open(entry) as src, open(dest, "wb") as out:
                out.write(src.read())

    if sys.platform != "win32":
        os.chmod(dest, 0o755)

    try:
        os.unlink(archive_path)
    except OSError:
        pass

    return dest


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
