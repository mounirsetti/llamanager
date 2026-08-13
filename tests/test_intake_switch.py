"""The master intake switch: top-bar toggle, admin API, CLI verb.

Covers the contract from llamanager/intake.py — closed intake refuses
inference with 503 (everyone, including admin keys), leaves the admin API,
the UI and the model listings reachable, drops the queued backlog, and
survives a config reload.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from llamanager.api_ui import COOKIE_NAME


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _admin_key(app) -> str:
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    assert boot is not None
    return am.rotate_key(boot.id)


def _ui_client(app, key: str) -> TestClient:
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303 and COOKIE_NAME in r.headers.get("set-cookie", "")
    return client


def _csrf(html: str) -> str:
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert m, "no csrf token in page"
    return m.group(1)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --------------------------------------------------------------------------
# config persistence
# --------------------------------------------------------------------------

def test_accepting_requests_defaults_true_and_round_trips(tmp_path):
    from llamanager.config import (load_config, update_queue_settings,
                                   write_default_config)
    cfg_path = tmp_path / "config.toml"
    write_default_config(cfg_path)
    assert load_config(cfg_path).accepting_requests is True

    update_queue_settings(cfg_path, accepting_requests=False)
    assert load_config(cfg_path).accepting_requests is False
    assert "accepting_requests" in cfg_path.read_text()

    update_queue_settings(cfg_path, accepting_requests=True)
    assert load_config(cfg_path).accepting_requests is True


def test_missing_key_reads_as_accepting(tmp_path):
    """An older config.toml without the key must not read as 'closed'."""
    from llamanager.config import load_config
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[queue]\nmax_concurrent = 1\n", encoding="utf-8")
    assert load_config(cfg_path).accepting_requests is True


# --------------------------------------------------------------------------
# admin API
# --------------------------------------------------------------------------

def test_admin_pause_resume_round_trip(app):
    key = _admin_key(app)
    with TestClient(app) as client:
        r = client.get("/admin/intake", headers=_auth(key))
        assert r.status_code == 200 and r.json()["accepting"] is True

        r = client.post("/admin/intake/pause", headers=_auth(key))
        assert r.status_code == 200
        assert r.json() == {"ok": True, "accepting": False, "dropped_queued": 0}
        assert app.state.cfg.accepting_requests is False
        # Persisted, not just in memory — a restart comes back up closed.
        assert "accepting_requests = false" in app.state.cfg.config_path.read_text()

        assert client.get("/admin/status", headers=_auth(key)
                          ).json()["accepting_requests"] is False

        r = client.post("/admin/intake/resume", headers=_auth(key))
        assert r.status_code == 200 and r.json()["accepting"] is True
        assert app.state.cfg.accepting_requests is True


def test_admin_intake_requires_admin_scope(app):
    am = app.state.auth
    _, plain_key = am.create_origin(name="plain", allowed_models=["*"])
    with TestClient(app) as client:
        r = client.post("/admin/intake/pause", headers=_auth(plain_key))
        assert r.status_code == 403


# --------------------------------------------------------------------------
# the gate itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi"}),
    ("/v1/images/generations", {"prompt": "a cat", "model": "test/model.gguf"}),
    ("/v1/videos/generations", {"prompt": "a cat", "model": "test/model.gguf"}),
    ("/anthropic/v1/messages",
     {"model": "test/model.gguf", "max_tokens": 8,
      "messages": [{"role": "user", "content": "hi"}]}),
    ("/anthropic/v1/messages/count_tokens",
     {"model": "test/model.gguf",
      "messages": [{"role": "user", "content": "hi"}]}),
])
def test_inference_refused_while_paused(app, path, body):
    key = _admin_key(app)
    with TestClient(app) as client:
        assert client.post("/admin/intake/pause",
                           headers=_auth(key)).status_code == 200
        r = client.post(path, json=body, headers=_auth(key))
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "60"
        assert "not accepting requests" in r.json()["detail"]


def test_audio_transcription_refused_while_paused(app):
    """The multipart endpoint is gated before the form is even parsed — and
    it really is the gate talking: the same request answers 400 when open."""
    key = _admin_key(app)
    with TestClient(app) as client:
        r = client.post("/v1/audio/transcriptions", headers=_auth(key),
                        files={"nothing": ("x.txt", b"x")})
        assert r.status_code == 400          # open: complains about 'file'

        client.post("/admin/intake/pause", headers=_auth(key))
        r = client.post("/v1/audio/transcriptions", headers=_auth(key),
                        files={"nothing": ("x.txt", b"x")})
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "60"


def test_audio_stream_websocket_refused_while_paused(app):
    """A websocket can't carry a 503, so it says so in the error frame and
    closes with 1013 ("try again later")."""
    from starlette.websockets import WebSocketDisconnect

    key = _admin_key(app)
    with TestClient(app) as client:
        client.post("/admin/intake/pause", headers=_auth(key))
        with client.websocket_connect(f"/v1/audio/stream?key={key}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not accepting requests" in msg["error"]
            with pytest.raises(WebSocketDisconnect) as ei:
                ws.receive_text()
            assert ei.value.code == 1013


def test_audio_stream_websocket_still_rejects_bad_keys_first(app):
    """Auth precedes the gate here too — an unauthorized socket gets 1008."""
    from starlette.websockets import WebSocketDisconnect

    with TestClient(app) as client:
        client.post("/admin/intake/pause", headers=_auth(_admin_key(app)))
        with client.websocket_connect("/v1/audio/stream?key=lm_garbage") as ws:
            assert ws.receive_json()["error"] == "unauthorized"
            with pytest.raises(WebSocketDisconnect) as ei:
                ws.receive_text()
            assert ei.value.code == 1008


def test_admin_keys_are_not_exempt(app):
    """The operator's own UI chat runs on an admin key — it is gated too."""
    key = _admin_key(app)
    with TestClient(app) as client:
        client.post("/admin/intake/pause", headers=_auth(key))
        boot = app.state.auth.get_origin_by_name("bootstrap")
        assert boot.is_admin
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]},
                        headers=_auth(key))
        assert r.status_code == 503


def test_gate_runs_after_auth(app):
    """A bad key still gets 401, not a 503 that leaks the daemon's state."""
    with TestClient(app) as client:
        client.post("/admin/intake/pause", headers=_auth(_admin_key(app)))
        r = client.post("/v1/chat/completions",
                        json={"messages": []},
                        headers=_auth("lm_garbage"))
        assert r.status_code == 401


def test_listings_and_admin_stay_reachable_while_paused(app):
    key = _admin_key(app)
    with TestClient(app) as client:
        client.post("/admin/intake/pause", headers=_auth(key))
        # Discovery costs the machine nothing and keeps clients sane.
        assert client.get("/v1/models", headers=_auth(key)).status_code == 200
        assert client.get("/anthropic/v1/models",
                          headers=_auth(key)).status_code == 200
        # And the door back out is never locked.
        assert client.get("/admin/status", headers=_auth(key)).status_code == 200
        assert client.post("/admin/intake/resume",
                           headers=_auth(key)).status_code == 200


def test_cancel_pending_drops_queued_but_not_in_flight(cfg, tmp_path):
    """The backlog is cancelled; work already running is left alone."""
    import asyncio

    from llamanager.auth import AuthManager, load_or_create_lookup_secret
    from llamanager.db import DB
    from llamanager.queue_mgr import QueueManager

    db = DB(tmp_path / "state.db")
    am = AuthManager(db, lookup_secret=load_or_create_lookup_secret(tmp_path))
    _, key = am.create_origin(name="client", allowed_models=["*"])

    async def _run() -> tuple[int, int]:
        origin = await am.verify(key)
        qm = QueueManager(cfg, db, sm=None)
        reqs = [await qm.enqueue(origin=origin, model_required=None)
                for _ in range(3)]
        # Pretend the dispatcher picked one up.
        qm._heap = [e for e in qm._heap if e[1] is not reqs[0]]
        reqs[0].status = "running"
        reqs[0].dispatched = True
        qm._in_flight[reqs[0].request_id] = reqs[0]
        return qm.cancel_pending(), qm.snapshot()["depth"]

    try:
        dropped, depth = asyncio.run(_run())
    finally:
        db.close()
    assert dropped == 2
    assert depth == 0


def test_pause_drops_the_queued_backlog(app, monkeypatch):
    """Closing intake asks the queue to shed its backlog, and says how much."""
    key = _admin_key(app)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.queue, "cancel_pending", lambda: 3)
        r = client.post("/admin/intake/pause", headers=_auth(key))
        assert r.json()["dropped_queued"] == 3

        # Idempotent: a second pause has no fresh backlog to blame.
        monkeypatch.setattr(app.state.queue, "cancel_pending", lambda: 9)
        r = client.post("/admin/intake/pause", headers=_auth(key))
        assert r.json()["dropped_queued"] == 0


# --------------------------------------------------------------------------
# top bar
# --------------------------------------------------------------------------

def test_topbar_renders_switch_and_toggles(app):
    key = _admin_key(app)
    with _ui_client(app, key) as client:
        body = client.get("/ui/").text
        assert 'action="/ui/topbar/intake"' in body
        assert 'aria-checked="false"' in body       # accepting
        assert ">accepting" in body.replace("\n", "").replace("  ", "")

        r = client.post("/ui/topbar/intake",
                        data={"csrf_token": _csrf(body), "accepting": "off"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert app.state.cfg.accepting_requests is False

        body = client.get("/ui/").text
        assert "is-intake-paused" in body
        assert 'aria-checked="true"' in body
        assert "not accepting" in body

        # The switch posts the opposite state, so clicking it again reopens.
        r = client.post("/ui/topbar/intake",
                        data={"csrf_token": _csrf(body), "accepting": "on"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert app.state.cfg.accepting_requests is True


def test_topbar_intake_requires_csrf(app):
    """A forged post can't close the door. (require_csrf answers a stale/bad
    token with a redirect back to the page, not a 403 — what matters here is
    that the switch did not move.)"""
    key = _admin_key(app)
    with _ui_client(app, key) as client:
        client.post("/ui/topbar/intake",
                    data={"csrf_token": "wrong", "accepting": "off"},
                    follow_redirects=False)
        assert app.state.cfg.accepting_requests is True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("verb", ["status", "pause", "resume"])
def test_cli_intake_help(verb):
    """The subparser is wired (a mis-wired one raises instead of exiting 0)."""
    from llamanager.cli import main
    with pytest.raises(SystemExit) as ei:
        main(["intake", verb, "--help"])
    assert ei.value.code == 0


def test_cli_intake_pause_end_to_end(monkeypatch, capsys, app):
    """`llamanager intake pause` closes the door on the running daemon."""
    from llamanager import cli as cli_mod
    from llamanager.admin_client import AdminClient

    key = _admin_key(app)
    http = TestClient(app, base_url="http://test")
    monkeypatch.setattr(
        cli_mod, "_make_admin_client",
        lambda args: AdminClient(base_url="http://test", admin_key=key,
                                 client=http))
    try:
        assert cli_mod.main(["intake", "pause"]) == 0
        assert app.state.cfg.accepting_requests is False
        assert '"accepting": false' in capsys.readouterr().out

        assert cli_mod.main(["intake", "resume"]) == 0
        assert app.state.cfg.accepting_requests is True
    finally:
        http.close()


def test_admin_client_intake_calls(app):
    from llamanager.admin_client import AdminClient
    key = _admin_key(app)
    http = TestClient(app, base_url="http://test")
    c = AdminClient(base_url="http://test", admin_key=key, client=http)
    try:
        assert c.intake_status()["accepting"] is True
        assert c.intake_pause()["accepting"] is False
        assert c.intake_status()["accepting"] is False
        assert c.intake_resume()["accepting"] is True
    finally:
        http.close()
