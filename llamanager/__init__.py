"""llamanager — single-host manager and proxy for llama-server."""

from pathlib import Path as _Path

_version_file = _Path(__file__).resolve().parent.parent / "VERSION"
if _version_file.exists():
    __version__ = _version_file.read_text(encoding="utf-8").strip()
else:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("llamanager")
