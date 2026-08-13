"""Zero-config admin for same-machine tools.

The daemon drops a 0600 control key in its data dir; the tray and CLI read it
with no configuration. Being able to read the file is the authorization, so
the tests care about: it exists, only the owner can read it, it actually
works, it is stable across restarts (a long-lived tray caches its client),
explicit credentials still win, and it is refused off loopback.
"""
from __future__ import annotations

import stat

import pytest
from fastapi.testclient import TestClient

from llamanager.auth import LOCAL_KEY_FILENAME, LOCAL_ORIGIN_NAME


def _key_path(app):
    return app.state.cfg.data_dir / LOCAL_KEY_FILENAME


# --------------------------------------------------------------------------
# provisioning
# --------------------------------------------------------------------------

def test_key_file_written_0600_and_origin_is_admin(app):
    p = _key_path(app)
    assert p.exists(), "daemon did not write a local control key"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    key = p.read_text().strip()
    assert key.startswith("lm_")

    origin = app.state.auth.get_origin_by_name(LOCAL_ORIGIN_NAME)
    assert origin is not None and origin.is_admin and origin.enabled


def test_key_works_for_admin_calls_with_no_configuration(app):
    key = _key_path(app).read_text().strip()
    with TestClient(app) as client:
        h = {"Authorization": f"Bearer {key}"}
        assert client.get("/admin/status", headers=h).status_code == 200
        # And it can drive the intake switch — the thing the tray needs.
        assert client.post("/admin/intake/pause", headers=h).status_code == 200
        assert client.get("/admin/intake", headers=h).json()["accepting"] is False
        assert client.post("/admin/intake/resume", headers=h).status_code == 200


def test_key_is_stable_across_restarts(data_dir, app):
    """A running tray caches its client, so restarting the daemon must not
    invalidate the key underneath it."""
    from llamanager.app import create_app
    first = _key_path(app).read_text().strip()
    again = create_app(app.state.cfg.config_path, print_bootstrap=False)
    assert (again.state.cfg.data_dir / LOCAL_KEY_FILENAME).read_text().strip() == first


def test_deleting_the_file_mints_a_fresh_key(app):
    """The old key is unreadable by anyone once the file is gone, so the next
    start replaces it rather than leaving a key nobody can use."""
    from llamanager.app import create_app
    old = _key_path(app).read_text().strip()
    _key_path(app).unlink()

    again = create_app(app.state.cfg.config_path, print_bootstrap=False)
    new = (again.state.cfg.data_dir / LOCAL_KEY_FILENAME).read_text().strip()
    assert new and new != old
    with TestClient(again) as client:
        assert client.get("/admin/status",
                          headers={"Authorization": f"Bearer {new}"}).status_code == 200
        assert client.get("/admin/status",
                          headers={"Authorization": f"Bearer {old}"}).status_code == 403


# --------------------------------------------------------------------------
# resolution precedence
# --------------------------------------------------------------------------

def test_resolve_falls_back_to_the_local_key(app, monkeypatch):
    from llamanager.admin_client import resolve_admin_key
    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    cfg = app.state.cfg
    assert resolve_admin_key(cfg) == _key_path(app).read_text().strip()


def test_explicit_and_env_still_win(app, monkeypatch):
    from llamanager.admin_client import resolve_admin_key
    cfg = app.state.cfg
    monkeypatch.setenv("LLAMANAGER_ADMIN_KEY", "from-env")
    assert resolve_admin_key(cfg) == "from-env"
    assert resolve_admin_key(cfg, "explicit") == "explicit"


def test_missing_file_still_raises_the_helpful_error(app, monkeypatch):
    from llamanager.admin_client import AdminClientError, resolve_admin_key
    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    _key_path(app).unlink()
    with pytest.raises(AdminClientError, match="no admin key found"):
        resolve_admin_key(app.state.cfg)


# --------------------------------------------------------------------------
# it must not become remote admin
# --------------------------------------------------------------------------

def test_local_key_refused_from_a_non_loopback_peer(app):
    """Binding to 0.0.0.0 must not turn a readable file into remote admin."""
    key = _key_path(app).read_text().strip()
    with TestClient(app, client=("203.0.113.9", 51234)) as client:
        r = client.get("/admin/status",
                       headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 403
        assert "only accepted from this machine" in r.json()["detail"]


def test_ordinary_admin_origins_are_unaffected_by_the_peer_check(app):
    """The loopback rule is scoped to the local key, not to admin in general."""
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    key = am.rotate_key(boot.id)
    with TestClient(app, client=("203.0.113.9", 51234)) as client:
        assert client.get("/admin/status",
                          headers={"Authorization": f"Bearer {key}"}).status_code == 200


def test_local_key_still_works_from_loopback(app):
    key = _key_path(app).read_text().strip()
    with TestClient(app, client=("127.0.0.1", 4242)) as client:
        assert client.get("/admin/status",
                          headers={"Authorization": f"Bearer {key}"}).status_code == 200


# --------------------------------------------------------------------------
# a caller that is NOT on this machine gets nothing for free
# --------------------------------------------------------------------------

def test_a_remote_cli_finds_no_key_and_must_be_given_one(app, tmp_path,
                                                         monkeypatch):
    """The fallback is client-side: it reads a file in *this* machine's data
    dir. A CLI run anywhere else has no such file and fails closed."""
    from llamanager.admin_client import AdminClientError, resolve_admin_key
    from llamanager.config import Config

    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    elsewhere = Config(data_dir=tmp_path / "some-other-box")
    with pytest.raises(AdminClientError, match="no admin key found"):
        resolve_admin_key(elsewhere)
    # Given a real key, that same remote caller works — the key is the gate,
    # not the location.
    assert resolve_admin_key(elsewhere, "lm_explicit") == "lm_explicit"


def test_another_machines_local_key_is_not_accepted(app, data_dir, tmp_path):
    """Two boxes each have a `local` origin; theirs must not open ours."""
    from llamanager.app import create_app

    other_dir = tmp_path / "other-box"
    other_dir.mkdir()
    other_cfg = other_dir / "config.toml"
    other_cfg.write_text(
        f'[server]\ndata_dir = "{other_dir.as_posix()}"\n', encoding="utf-8")
    other = create_app(other_cfg, print_bootstrap=False)
    other_key = (other.state.cfg.data_dir / LOCAL_KEY_FILENAME).read_text().strip()

    assert other_key != _key_path(app).read_text().strip()
    with TestClient(app) as client:
        r = client.get("/admin/status",
                       headers={"Authorization": f"Bearer {other_key}"})
        assert r.status_code == 403


@pytest.mark.parametrize("path,method,body", [
    ("/admin/status", "get", None),
    ("/admin/intake/pause", "post", None),
    ("/v1/models", "get", None),
    ("/v1/chat/completions", "post",
     {"messages": [{"role": "user", "content": "hi"}]}),
    ("/anthropic/v1/messages", "post",
     {"model": "test/model.gguf", "max_tokens": 8,
      "messages": [{"role": "user", "content": "hi"}]}),
])
def test_local_key_is_refused_off_box_on_every_surface(app, path, method, body):
    """A copy of the file that escapes the machine (synced home, backup) must
    be inert everywhere — inference included, not just /admin."""
    key = _key_path(app).read_text().strip()
    with TestClient(app, client=("198.51.100.7", 4444)) as client:
        r = getattr(client, method)(
            path, headers={"Authorization": f"Bearer {key}"},
            **({"json": body} if body else {}))
        assert r.status_code == 403, f"{path} accepted the local key off-box"
        assert "only accepted from this machine" in r.text


def test_a_real_remote_origin_still_works_off_box(app):
    """The rule is scoped to the auto-minted local credential — ordinary keys
    are exactly as usable from elsewhere as before."""
    am = app.state.auth
    _, key = am.create_origin(name="remote-agent", allowed_models=["*"])
    with TestClient(app, client=("198.51.100.7", 4444)) as client:
        assert client.get("/v1/models",
                          headers={"Authorization": f"Bearer {key}"}).status_code == 200


# --------------------------------------------------------------------------
# the tray, end to end
# --------------------------------------------------------------------------

def test_tray_builds_a_working_client_with_no_configuration(app, monkeypatch):
    """The whole point: a tray started from a desktop session with no env and
    no [cli] section can still drive the daemon."""
    pytest.importorskip("pystray")
    import threading

    from llamanager import service_ctl, tray as tray_mod
    from llamanager.admin_client import AdminClient

    monkeypatch.delenv("LLAMANAGER_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        tray_mod.service_ctl, "state",
        lambda c: service_ctl.DaemonState(reachable=True, installed=True,
                                          autostart=True, detail="up"))

    http = TestClient(app, base_url="http://test")
    real_from_config = AdminClient.from_config

    def in_process(cfg, **kw):
        return real_from_config(cfg, base_url="http://test", client=http, **kw)

    monkeypatch.setattr(AdminClient, "from_config", staticmethod(in_process))
    try:
        t = tray_mod.TrayApp(app.state.cfg)
        assert t._client is not None, "tray found no credential"
        t._icon = None
        t._notify = lambda m: None
        t._poll_once()
        menu = t._build_menu()
        item = next(i for i in menu if "Accepting requests" in str(i.text))
        assert item.enabled is True and item.checked is True
        item(None)                                   # click → pause
        assert t._client.intake_status()["accepting"] is False
        item2 = next(i for i in t._build_menu() if "Accepting requests" in str(i.text))
        item2(None)                                  # click → resume
        assert t._client.intake_status()["accepting"] is True
    finally:
        http.close()
