"""The Claude Desktop bundle: a manifest that installs, and no vendored code.

The bundle is the one artifact a user installs without reading, so the
things worth pinning are that it declares a schema Claude Desktop
understands, that its version tracks the project, and that it stays a
pointer at the installed binary rather than a second copy of llamanager.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def bundle(tmp_path):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_mcpb.py"),
         "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    path = tmp_path / "llamanager.mcpb"
    assert path.is_file()
    return path


def test_bundle_holds_a_manifest_and_an_icon(bundle):
    with zipfile.ZipFile(bundle) as z:
        assert set(z.namelist()) == {"manifest.json", "icon.png"}


def test_manifest_has_the_fields_the_installer_requires(bundle):
    with zipfile.ZipFile(bundle) as z:
        m = json.loads(z.read("manifest.json"))
    for field in ("manifest_version", "name", "version", "description",
                  "author", "server"):
        assert field in m, field
    assert m["manifest_version"] == "0.3"
    assert m["author"]["name"]
    assert m["icon"] == "icon.png"


def test_version_tracks_the_project(bundle):
    with zipfile.ZipFile(bundle) as z:
        m = json.loads(z.read("manifest.json"))
    assert m["version"] == (ROOT / "VERSION").read_text().strip()


def test_it_runs_the_installed_binary_rather_than_shipping_one(bundle):
    """A vendored copy would fight the real daemon for the GPU."""
    with zipfile.ZipFile(bundle) as z:
        m = json.loads(z.read("manifest.json"))
        # No Python, no server directory — the bundle carries no code.
        assert not [n for n in z.namelist() if n.endswith(".py")]
    cfg = m["server"]["mcp_config"]
    assert cfg["command"] == "llamanager"
    assert cfg["args"] == ["mcp-stdio"]


def test_credentials_are_declared_sensitive_and_optional(bundle):
    """Optional because same-box installs use the local control key."""
    with zipfile.ZipFile(bundle) as z:
        m = json.loads(z.read("manifest.json"))
    key = m["user_config"]["admin_key"]
    assert key["sensitive"] is True
    assert key["required"] is False
    assert m["user_config"]["url"]["default"] == "http://127.0.0.1:7200"
    env = m["server"]["mcp_config"]["env"]
    assert env["LLAMANAGER_ADMIN_KEY"] == "${user_config.admin_key}"
    assert env["LLAMANAGER_URL"] == "${user_config.url}"
