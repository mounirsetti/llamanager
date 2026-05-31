"""Tests for the engine version listing + downgrade (install a chosen version).

Covers llama_installer release/version listing and tag installs, the diffusers
version listing + pin override, and the admin endpoints — all with the network
and subprocess boundaries stubbed.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# llama_installer: list_versions
# ---------------------------------------------------------------------------

def _release(tag, asset_name):
    return {"tag_name": tag, "prerelease": False, "published_at": "2026-01-01T00:00:00Z",
            "assets": [{"name": asset_name, "size": 1, "browser_download_url": "http://x/" + asset_name}]}


def test_list_versions_github_filters_by_asset(monkeypatch):
    from llamanager import llama_installer as L
    # Two releases; stub asset selection so only the "good" one is installable
    # here (platform-independent — the real _select_asset is exercised elsewhere).
    good = _release("b6000", "llama-good.tar.gz")
    bad = _release("b5999", "llama-nomatch.tar.gz")
    monkeypatch.setattr(L, "_fetch_release_list", lambda url, per_page=30: [good, bad])
    monkeypatch.setattr(
        L, "_select_asset",
        lambda assets, backend: assets[0] if "good" in assets[0]["name"] else None)

    out = L.list_versions("llama.cpp", "cpu")
    assert out["error"] is None
    tags = [r["version"] for r in out["versions"]]
    assert "b6000" in tags          # has a matching asset
    assert "b5999" not in tags      # filtered out (no installable asset)


def test_list_versions_mlx_uses_pypi(monkeypatch):
    from llamanager import llama_installer as L
    monkeypatch.setattr(L, "_fetch_pypi_versions",
                        lambda url, limit=30: ["0.20.0", "0.19.1", "0.19.0"])
    out = L.list_versions("mlx", "apple-silicon")
    assert out["error"] is None
    assert [r["version"] for r in out["versions"]] == ["0.20.0", "0.19.1", "0.19.0"]


def test_list_versions_surfaces_errors(monkeypatch):
    from llamanager import llama_installer as L
    def boom(*a, **k):
        raise RuntimeError("github down")
    monkeypatch.setattr(L, "_fetch_release_list", boom)
    out = L.list_versions("llama.cpp", "cpu")
    assert out["versions"] == []
    assert "github down" in out["error"]


# ---------------------------------------------------------------------------
# llama_installer: installing a specific tag (downgrade path)
# ---------------------------------------------------------------------------

def test_install_llama_fetches_chosen_tag(monkeypatch, tmp_path):
    from llamanager import llama_installer as L

    calls = {}

    def fake_by_tag(api, tag):
        calls["tag"] = tag
        return _release(tag, "llama-x.tar.gz")

    def fake_latest(api):
        calls["latest"] = True
        return _release("b9999", "llama-x.tar.gz")

    monkeypatch.setattr(L, "_fetch_release_by_tag", fake_by_tag)
    monkeypatch.setattr(L, "_fetch_latest_release", fake_latest)
    monkeypatch.setattr(L, "_select_asset", lambda assets, backend: assets[0])
    monkeypatch.setattr(L, "_download_asset", lambda url, emit: Path("/tmp/x.tar.gz"))
    monkeypatch.setattr(L, "_extract_binary", lambda p, s, b: Path("/tmp/bin/llama-server"))
    written = {}
    monkeypatch.setattr(L, "write_install_meta",
                        lambda s, b, **kw: written.update(kw))

    state = L.InstallState()
    asyncio.run(L._install_llama(state, "llama.cpp", "cpu",
                                 L.SOURCES["llama.cpp"], L.BACKENDS["cpu"],
                                 version="b6000"))
    assert state.status == "done"
    assert calls.get("tag") == "b6000"      # fetched the chosen tag
    assert "latest" not in calls            # not the latest path
    assert written.get("version") == "b6000"  # marker records the chosen tag


# ---------------------------------------------------------------------------
# engine_installer: diffusers pin override + version listing
# ---------------------------------------------------------------------------

def test_apply_diffusers_pin_replaces_in_place():
    from llamanager.engine_installer import apply_diffusers_pin
    pkgs = ["torch", "diffusers==0.38.0", "safetensors"]
    assert apply_diffusers_pin(pkgs, "0.36.0") is True
    assert "diffusers==0.36.0" in pkgs and "diffusers==0.38.0" not in pkgs
    # appends when absent
    pkgs2 = ["torch"]
    assert apply_diffusers_pin(pkgs2, "0.36.0") is False
    assert pkgs2[-1] == "diffusers==0.36.0"


def test_list_diffusers_versions(monkeypatch):
    from llamanager import engine_installer as ei

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"releases": {
                "0.38.0": [{"upload_time_iso_8601": "2026-03-01T00:00:00Z"}],
                "0.37.1": [{"upload_time_iso_8601": "2026-02-01T00:00:00Z"}],
                "0.39.0.dev0": [{"upload_time_iso_8601": "2026-04-01T00:00:00Z"}],
                "0.0.0": [],   # no files -> skipped
            }}

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): return _Resp()

    monkeypatch.setattr(ei.httpx, "Client", _Client)
    out = ei.list_diffusers_versions()
    assert out["error"] is None
    assert out["versions"] == ["0.38.0", "0.37.1"]   # newest-first, dev + empty dropped


# ---------------------------------------------------------------------------
# admin endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_key(app) -> str:
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    return am.rotate_key(boot.id)


@pytest.fixture
def admin_client(app, admin_key):
    from llamanager.admin_client import AdminClient
    from fastapi.testclient import TestClient
    http = TestClient(app, base_url="http://test")
    yield AdminClient(base_url="http://test", admin_key=admin_key, client=http)
    http.close()


def test_admin_engine_versions(admin_client, monkeypatch):
    from llamanager import llama_installer as L
    monkeypatch.setattr(L, "list_versions", lambda s, b: {
        "versions": [{"version": "b6000", "prerelease": False, "published_at": ""}],
        "installed": "b5000", "error": None})
    res = admin_client.setup_engine_versions("llama.cpp-cpu")
    assert res["variant"] == "llama.cpp-cpu"
    assert res["installed"] == "b5000"
    assert res["versions"][0]["version"] == "b6000"


def test_admin_install_llama_passes_version(admin_client, monkeypatch):
    from llamanager import llama_installer as L

    async def fake_install(state, source, backend, *, version=None):
        state.status = "done"

    monkeypatch.setattr(L, "install_variant", fake_install)
    res = admin_client.setup_install_llama_server(
        source="llama.cpp", backend="cpu", version="b6000")
    assert res["version"] == "b6000"


def test_admin_diffusion_install_persists_override(admin_client, app, monkeypatch):
    # Don't actually spawn a venv install — stub the installer.
    app.state.engine_installer.start = lambda engine, options=None: "fake-id"
    res = admin_client.diffusion_install("z_image", diffusers_version="0.36.0")
    assert res["id"] == "fake-id"
    # Override persisted to live config + on disk.
    assert app.state.cfg.image_diffusers_version.get("z_image") == "0.36.0"
    from llamanager.config import load_config
    assert load_config(app.state.cfg.config_path).image_diffusers_version.get("z_image") == "0.36.0"

    # Reset clears it.
    res = admin_client.diffusion_install("z_image", reset_diffusers=True)
    assert "z_image" not in (app.state.cfg.image_diffusers_version or {})


def test_admin_diffusion_versions(admin_client, monkeypatch):
    from llamanager import engine_installer as ei
    monkeypatch.setattr(ei, "list_diffusers_versions",
                        lambda: {"versions": ["0.38.0", "0.37.1"], "error": None})
    # installed older than the pin -> has_update True
    monkeypatch.setattr(ei, "installed_diffusers_version", lambda c, e: "0.36.0")
    res = admin_client.diffusion_versions("z_image")
    assert res["engine"] == "z_image"
    assert res["installed"] == "0.36.0"
    assert res["pin"] == ei.DIFFUSERS_PIN
    assert res["target"] == ei.DIFFUSERS_PIN
    assert res["has_update"] is True
    assert res["versions"] == ["0.38.0", "0.37.1"]


def test_admin_setup_check_updates(admin_client, monkeypatch):
    from llamanager import llama_installer as L

    def fake_check(source, backend):
        return L.UpdateInfo(installed="b5000", latest="b6000", has_update=True)

    monkeypatch.setattr(L, "check_for_update", fake_check)
    res = admin_client.setup_check_updates("llama.cpp-cpu")
    upd = res["updates"]["llama.cpp-cpu"]
    assert upd["installed"] == "b5000"
    assert upd["latest"] == "b6000"
    assert upd["has_update"] is True


def test_admin_setup_check_updates_bad_variant(admin_client):
    from llamanager.admin_client import AdminClientError
    with pytest.raises(AdminClientError):
        admin_client.setup_check_updates("not-a-real-variant")
