#!/usr/bin/env python3
"""Build ``dist/llamanager.mcpb`` — the one-click Claude Desktop bundle.

An ``.mcpb`` is a zip holding a ``manifest.json`` (and, here, an icon).
The user double-clicks it and Claude Desktop installs an MCP server with
no config file editing.

This bundle deliberately ships **no code**. It points at the
``llamanager`` binary the user already has and runs its ``mcp-stdio``
verb, which proxies the running daemon. Vendoring a copy of llamanager
here would give the host a second process fighting the first one for the
GPU, and it would go stale the moment llamanager is updated.

Usage:
    python tools/build_mcpb.py [--out dist]
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_VERSION = "0.3"


def build_manifest(version: str) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "name": "llamanager",
        "display_name": "llamanager",
        "version": version,
        "description": "Drive your local llamanager daemon: models, GPU, "
                       "generation, transcription.",
        "long_description": (
            "Connects Claude to llamanager running on this machine. List and "
            "load models, watch VRAM and memory pressure, pull weights from "
            "Hugging Face, generate images and video, run local text "
            "inference, and transcribe audio — all on your own hardware, "
            "with nothing leaving the box.\n\n"
            "Requires llamanager to be installed and running (`llamanager "
            "serve`, or the llamanager user service). This bundle contains no "
            "code of its own: it runs `llamanager mcp-stdio`, which proxies "
            "the daemon that already owns the GPU."
        ),
        "author": {"name": "llamanager contributors"},
        "license": "MIT",
        "icon": "icon.png",
        "keywords": ["llm", "local", "gpu", "llama.cpp", "diffusion", "asr"],
        "server": {
            "type": "binary",
            "entry_point": "llamanager",
            "mcp_config": {
                "command": "llamanager",
                "args": ["mcp-stdio"],
                "env": {
                    # Both are optional on the machine running the daemon:
                    # llamanager falls back to its loopback-only control key
                    # and its configured bind address. Left blank, the
                    # placeholders resolve to empty strings, and the CLI's
                    # own resolution order takes over.
                    "LLAMANAGER_ADMIN_KEY": "${user_config.admin_key}",
                    "LLAMANAGER_URL": "${user_config.url}",
                },
            },
        },
        "user_config": {
            "admin_key": {
                "type": "string",
                "title": "API key",
                "description": (
                    "An origin key from llamanager's Connect page. Leave "
                    "blank when Claude Desktop runs on the same machine as "
                    "the daemon — llamanager then uses its local control key."
                ),
                "sensitive": True,
                "required": False,
            },
            "url": {
                "type": "string",
                "title": "Daemon URL",
                "description": "Where llamanager is listening.",
                "default": "http://127.0.0.1:7200",
                "required": False,
            },
        },
        "compatibility": {"platforms": ["darwin", "linux", "win32"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "dist"),
                    help="output directory (default: dist/)")
    args = ap.parse_args()

    version = (ROOT / "VERSION").read_text().strip()
    manifest = build_manifest(version)

    icon = ROOT / "assets" / "icon-light-512.png"
    if not icon.is_file():
        # Fail loudly: a bundle whose declared icon is missing installs with
        # a broken tile, and the manifest would be lying about its contents.
        raise SystemExit(f"icon not found: {icon}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "llamanager.mcpb"

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        z.write(icon, "icon.png")

    size_kb = bundle.stat().st_size / 1024
    print(f"built {bundle} ({size_kb:.1f} KB) for llamanager {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
