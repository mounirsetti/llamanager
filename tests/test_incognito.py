"""Incognito requests: an admin's chat / image / video leaves nothing behind.

Covers the request-field contract (admin-only, boolean), the storage gate
in the queue manager (no text even when retention is on, row flagged), the
image runner's ephemeral output (outside the gallery, no sidecar/thumbnail,
gone after ``discard_ephemeral``), and the request-detail view.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _mk(am, name, **kw):
    return am.create_origin(name=name, allowed_models=["*"], **kw)


# --------------------------------------------------------------------------
# Request-field contract
# --------------------------------------------------------------------------

def test_non_admin_cannot_request_incognito(app):
    _, key = _mk(app.state.auth, "plain-user", is_admin=False)
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {key}"}
    for path in ("/v1/chat/completions", "/v1/images/generations",
                 "/v1/videos/generations"):
        r = client.post(path, headers=hdr,
                        json={"model": "test/model.gguf", "prompt": "x",
                              "messages": [{"role": "user", "content": "x"}],
                              "incognito": True})
        assert r.status_code == 403, (path, r.status_code, r.text)
        assert "admin" in r.json()["detail"]


def test_incognito_must_be_boolean(app):
    _, key = _mk(app.state.auth, "adm-bool", is_admin=True)
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"messages": [{"role": "user", "content": "x"}],
                          "incognito": "yes"})
    assert r.status_code == 400
    assert "boolean" in r.json()["detail"]


def test_incognito_image_requires_b64_json(app):
    _, key = _mk(app.state.auth, "adm-b64", is_admin=True)
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {key}"}
    for path in ("/v1/images/generations", "/v1/videos/generations"):
        r = client.post(path, headers=hdr,
                        json={"model": "test/model.gguf", "prompt": "x",
                              "response_format": "url", "incognito": True})
        # The b64 check must fire before any model resolution: a bad
        # combination is a bad request regardless of what's installed.
        assert r.status_code == 400, (path, r.status_code, r.text)
        assert "b64_json" in r.json()["detail"]


def test_pop_incognito_strips_the_field_from_the_forwarded_body():
    from llamanager.api_v1 import _pop_incognito
    from llamanager.auth import Origin
    admin = Origin(id=1, name="a", priority=50, allowed_models=["*"],
                   is_admin=True, created_at=0.0)
    body = {"messages": [], "incognito": True}
    assert _pop_incognito(body, admin) is True
    assert "incognito" not in body
    assert _pop_incognito({"messages": []}, admin) is False


# --------------------------------------------------------------------------
# Storage gate
# --------------------------------------------------------------------------

def _qm_and_db(cfg):
    from llamanager.db import DB
    from llamanager.queue_mgr import QueueManager
    from llamanager.server_manager import ServerManager
    db = DB(cfg.db_path)
    sm = ServerManager(cfg, db)
    return QueueManager(cfg, db, sm), db


def test_incognito_row_is_flagged_and_textless_even_with_retention_on(cfg):
    from llamanager.auth import Origin
    from llamanager.queue_mgr import QueuedRequest
    cfg.conversation_retention_days = 30
    qm, db = _qm_and_db(cfg)
    try:
        origin = Origin(id=1, name="t", priority=50, allowed_models=["*"],
                        is_admin=True, created_at=0.0)
        qr = QueuedRequest(request_id="inc1", origin=origin, priority=50,
                           model_required="m", enqueued_at=time.time(),
                           seq=0, incognito=True)
        db.insert_request(request_id="inc1", origin_id=None, model="m",
                          priority=50, incognito=True)
        qm._in_flight["inc1"] = qr
        qm.mark_in_flight_done(qr, error=None, cancelled=False,
                               prompt_tokens=3, completion_tokens=4,
                               prompt_text="secret prompt",
                               response_text="secret answer")
        row = db.query_one(
            "SELECT prompt_text, response_text, incognito, prompt_tokens "
            "FROM requests WHERE id=?", ("inc1",))
        assert row["prompt_text"] is None and row["response_text"] is None
        assert row["incognito"] == 1
        assert row["prompt_tokens"] == 3      # accounting is kept
        ev = db.query_one(
            "SELECT payload_json AS payload FROM events WHERE kind='request_done' "
            "ORDER BY rowid DESC LIMIT 1")
        assert '"incognito": true' in ev["payload"]
        assert "secret" not in ev["payload"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_enqueue_records_the_flag(cfg):
    from llamanager.auth import Origin
    qm, db = _qm_and_db(cfg)
    try:
        origin = Origin(id=1, name="t", priority=50, allowed_models=["*"],
                        is_admin=True, created_at=0.0)
        qr = await qm.enqueue(origin=origin, model_required="m",
                              incognito=True)
        assert qr.incognito is True
        row = db.query_one("SELECT incognito FROM requests WHERE id=?",
                           (qr.request_id,))
        assert row["incognito"] == 1
        snap = qm.snapshot()
        assert any(r["id"] == qr.request_id and r["incognito"]
                   for r in snap["pending"])
        qm.cancel(qr.request_id)
    finally:
        db.close()


# --------------------------------------------------------------------------
# Ephemeral output
# --------------------------------------------------------------------------

def test_incognito_dir_is_outside_the_gallery_and_discardable(cfg):
    from llamanager.image_runner import (ImageResult, _incognito_dir,
                                         discard_ephemeral,
                                         sweep_incognito_dir)
    d = _incognito_dir(cfg, "req-1")
    assert d.is_dir()
    with pytest.raises(ValueError):
        d.resolve().relative_to(cfg.images_dir.resolve())
    (d / "img0001.png").write_bytes(b"\x89PNG")
    res = ImageResult(request_id="req-1", engine="e", model_id="m",
                      profile_name=None, output_path=d / "img0001.png",
                      seed=1, duration_s=0.1, ephemeral_dir=d)
    discard_ephemeral(res)
    assert not d.exists()
    assert res.ephemeral_dir is None
    discard_ephemeral(res)   # idempotent
    # Startup sweep clears whatever a crash left behind.
    _incognito_dir(cfg, "req-2")
    _incognito_dir(cfg, "req-3")
    assert sweep_incognito_dir(cfg) == 2
    assert not (cfg.data_dir / "incognito" / "req-2").exists()


@pytest.mark.asyncio
async def test_run_one_incognito_writes_no_sidecar_or_thumbnail(cfg, tmp_path,
                                                                monkeypatch):
    """Drive ``_run_one`` with a stand-in 'engine' (a shell one-liner that
    writes the output file) and check what lands on disk."""
    import asyncio
    from llamanager.db import DB
    from llamanager.image_runner import ImageTaskRunner
    from llamanager import thumbs

    db = DB(cfg.db_path)
    runner = ImageTaskRunner(cfg, db, sm=None)
    out = tmp_path / "scratch" / "img0001.png"
    out.parent.mkdir(parents=True)
    argv = ["sh", "-c", f"printf '\\211PNG' > '{out}'"]

    class _Adapter:
        def parse_progress(self, line):
            return None

    warmed = []
    monkeypatch.setattr(thumbs, "warm_thumbnail",
                        lambda *a, **k: warmed.append(a))
    try:
        await runner._run_one(engine="fake", model_id="m", profile_name="p",
                              request_id="r", argv=argv, env={},
                              out_path=out, adapter=_Adapter(),
                              progress_cb=None,
                              sidecar={"prompt": "SECRET"}, incognito=True)
        assert out.exists()
        assert not out.with_suffix(".png.json").exists()
        assert warmed == []
        ev = db.query_one(
            "SELECT payload_json AS payload FROM events WHERE kind='image_generate_done' "
            "ORDER BY rowid DESC LIMIT 1")
        assert '"incognito": true' in ev["payload"]
        assert "img0001" not in ev["payload"]     # no output path recorded
        assert "SECRET" not in ev["payload"]
    finally:
        db.close()


# --------------------------------------------------------------------------
# Request detail view
# --------------------------------------------------------------------------

def test_request_detail_explains_incognito(app):
    from llamanager.api_ui import COOKIE_NAME
    am = app.state.auth
    _, key = _mk(am, "adm-view", is_admin=True)
    db = app.state.db
    db.insert_request(request_id="rd-inc", origin_id=None, model="m",
                      priority=50, incognito=True)
    db.update_request_status("rd-inc", "done", finished_at=time.time())
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert COOKIE_NAME in r.cookies
    html = client.get("/ui/requests/rd-inc").text
    assert "incognito" in html.lower()
    assert "Not recorded" not in html
