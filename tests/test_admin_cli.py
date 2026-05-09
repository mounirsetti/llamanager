"""Tests for the admin CLI verbs and the underlying AdminClient.

The CLI verbs talk to a running daemon over /admin/*. We use
httpx.ASGITransport to drive the FastAPI app directly (no socket), then
inject the resulting httpx.Client into AdminClient.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_key(app) -> str:
    """Mint a fresh admin key for the bootstrap origin (the original is
    suppressed by the `app` fixture). This stands in for whatever the user
    would put in $LLAMANAGER_ADMIN_KEY in production."""
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    assert boot is not None
    return am.rotate_key(boot.id)


@pytest.fixture
def admin_client(app, admin_key):
    """An AdminClient whose HTTP traffic is routed in-process to the
    FastAPI app via ASGITransport — no ports, no bootstrap race."""
    from llamanager.admin_client import AdminClient
    from fastapi.testclient import TestClient
    http = TestClient(app, base_url="http://test")
    yield AdminClient(base_url="http://test", admin_key=admin_key, client=http)
    http.close()


# ---------------------------------------------------------------------------
# Auth + URL resolution
# ---------------------------------------------------------------------------


def test_resolve_admin_key_priority(monkeypatch, tmp_path):
    """Explicit > env > config."""
    from llamanager.admin_client import resolve_admin_key
    from llamanager.config import Config

    cfg = Config(data_dir=tmp_path)
    cfg.raw = {"cli": {"admin_key": "from-config"}}

    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    assert resolve_admin_key(cfg) == "from-config"

    monkeypatch.setenv("LLAMANAGER_ADMIN_KEY", "from-env")
    assert resolve_admin_key(cfg) == "from-env"

    assert resolve_admin_key(cfg, "from-flag") == "from-flag"


def test_resolve_admin_key_missing_raises(monkeypatch, tmp_path):
    from llamanager.admin_client import AdminClientError, resolve_admin_key
    from llamanager.config import Config

    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    cfg = Config(data_dir=tmp_path)  # no [cli] section
    with pytest.raises(AdminClientError):
        resolve_admin_key(cfg)


def test_resolve_base_url_rewrites_wildcard(monkeypatch, tmp_path):
    from llamanager.admin_client import resolve_base_url
    from llamanager.config import Config

    monkeypatch.delenv("LLAMANAGER_URL", raising=False)
    cfg = Config(data_dir=tmp_path, bind="0.0.0.0", port=7200)
    assert resolve_base_url(cfg) == "http://127.0.0.1:7200"

    monkeypatch.setenv("LLAMANAGER_URL", "http://elsewhere:9000")
    assert resolve_base_url(cfg) == "http://elsewhere:9000"

    assert resolve_base_url(cfg, "http://flag:1234") == "http://flag:1234"


# ---------------------------------------------------------------------------
# AdminClient end-to-end against the real FastAPI app
# ---------------------------------------------------------------------------


def test_status_through_client(admin_client):
    body = admin_client.status()
    assert body["state"] == "stopped"
    assert body["queue_depth"] == 0


def test_disk_through_client(admin_client):
    body = admin_client.disk()
    assert "free_gb" in body
    assert "models_dir" in body


def test_origins_create_list_delete(admin_client):
    created = admin_client.origin_create("alice", priority=42,
                                          allowed_models=["*"])
    assert created["api_key"].startswith("lm_")
    oid = created["origin"]["id"]

    listed = admin_client.origins_list()
    assert any(o["name"] == "alice" for o in listed)

    rotated = admin_client.origin_rotate_key(oid)
    assert rotated["api_key"] != created["api_key"]

    admin_client.origin_delete(oid)
    listed = admin_client.origins_list()
    assert not any(o["name"] == "alice" for o in listed)


def test_queue_pause_resume_through_client(admin_client):
    admin_client.queue_pause()
    snap = admin_client.queue_list()
    assert snap["paused"] is True
    admin_client.queue_resume()
    snap = admin_client.queue_list()
    assert snap["paused"] is False


def test_models_list_through_client(admin_client):
    # Empty registry on a fresh data dir.
    assert admin_client.models_list() == []


def test_unauthorized_raises(app):
    """A bad token surfaces as AdminClientError (not a traceback)."""
    from llamanager.admin_client import AdminClient, AdminClientError
    from fastapi.testclient import TestClient
    http = TestClient(app, base_url="http://test")
    try:
        c = AdminClient(base_url="http://test", admin_key="lm_bogus", client=http)
        with pytest.raises(AdminClientError) as ei:
            c.status()
        assert "401" in str(ei.value) or "403" in str(ei.value)
    finally:
        http.close()


# ---------------------------------------------------------------------------
# CLI surface — argparse + dispatch
# ---------------------------------------------------------------------------


def test_cli_parser_builds():
    """If any subparser was wired wrong this will raise."""
    from llamanager.cli import main
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0


def test_cli_models_help():
    from llamanager.cli import main
    with pytest.raises(SystemExit) as ei:
        main(["models", "--help"])
    assert ei.value.code == 0


def test_cli_admin_verb_uses_injected_client(monkeypatch, capsys, app, admin_key):
    """Drive `llamanager origins list` end-to-end. We monkeypatch the
    client factory so the CLI talks to the FastAPI app in-process."""
    from llamanager import cli as cli_mod
    from llamanager.admin_client import AdminClient

    from fastapi.testclient import TestClient
    http = TestClient(app, base_url="http://test")

    def fake_factory(args):
        return AdminClient(base_url="http://test", admin_key=admin_key,
                           client=http)

    monkeypatch.setattr(cli_mod, "_make_admin_client", fake_factory)
    try:
        rc = cli_mod.main(["origins", "list"])
    finally:
        http.close()
    assert rc == 0
    out = capsys.readouterr().out
    parsed: list[dict[str, Any]] = json.loads(out)
    assert any(o["name"] == "bootstrap" for o in parsed)


def test_cli_kv_arg_parsing():
    """`--arg key=value` coerces ints/floats/bools."""
    from llamanager.cli import _parse_kv_args
    out = _parse_kv_args(["ctx-size=8192", "temp=0.7", "verbose=true",
                          "alias=qwen"])
    assert out == {"ctx-size": 8192, "temp": 0.7,
                   "verbose": True, "alias": "qwen"}


def test_cli_kv_arg_rejects_bad_input():
    from llamanager.cli import _parse_kv_args
    with pytest.raises(SystemExit):
        _parse_kv_args(["nokey"])
