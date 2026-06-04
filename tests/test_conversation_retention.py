"""Conversation-text retention setting: 0 disables capture, N keeps text
N days. Covers config round-trip, the storage gate, the prune, and the
settings route + page."""
from __future__ import annotations

import re
import time

from fastapi.testclient import TestClient

from llamanager.api_ui import COOKIE_NAME


# --------------------------------------------------------------------------
# Config load/save round-trip
# --------------------------------------------------------------------------

def test_config_retention_roundtrip(tmp_path):
    from llamanager.config import (load_config, update_conversation_retention)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[server]\ndata_dir = "%s"\n' % (tmp_path / "d").as_posix(),
        encoding="utf-8",
    )
    # Default when the section is absent.
    assert load_config(cfg_path).conversation_retention_days == 30
    # Persist and reload — including the meaningful 0.
    update_conversation_retention(cfg_path, retention_days=7)
    assert load_config(cfg_path).conversation_retention_days == 7
    assert "[conversation]" in cfg_path.read_text()
    update_conversation_retention(cfg_path, retention_days=0)
    assert load_config(cfg_path).conversation_retention_days == 0


# --------------------------------------------------------------------------
# Storage gate: retention 0 -> nothing written
# --------------------------------------------------------------------------

def _qm_and_db(cfg):
    from llamanager.db import DB
    from llamanager.queue_mgr import QueueManager
    from llamanager.server_manager import ServerManager
    db = DB(cfg.db_path)
    sm = ServerManager(cfg, db)
    return QueueManager(cfg, db, sm), db


def _complete(qm, db, rid, *, prompt="p", resp="r"):
    from llamanager.auth import Origin
    from llamanager.queue_mgr import QueuedRequest
    origin = Origin(id=1, name="t", priority=50, allowed_models=["*"],
                    is_admin=False, created_at=0.0)
    qr = QueuedRequest(request_id=rid, origin=origin, priority=50,
                       model_required="m", enqueued_at=time.time(), seq=0)
    db.insert_request(request_id=rid, origin_id=None, model="m", priority=50)
    qm._in_flight[rid] = qr
    qm.mark_in_flight_done(qr, error=None, cancelled=False,
                           prompt_tokens=1, completion_tokens=1,
                           prompt_text=prompt, response_text=resp)


def test_retention_zero_skips_storage(cfg):
    cfg.conversation_retention_days = 0
    qm, db = _qm_and_db(cfg)
    try:
        _complete(qm, db, "r0")
        row = db.query_one(
            "SELECT prompt_text, response_text FROM requests WHERE id=?", ("r0",))
        assert row["prompt_text"] is None and row["response_text"] is None
    finally:
        db.close()


def test_retention_positive_stores(cfg):
    cfg.conversation_retention_days = 30
    qm, db = _qm_and_db(cfg)
    try:
        _complete(qm, db, "r1", prompt="hi", resp="yo")
        row = db.query_one(
            "SELECT prompt_text, response_text FROM requests WHERE id=?", ("r1",))
        assert row["prompt_text"] == "hi" and row["response_text"] == "yo"
    finally:
        db.close()


# --------------------------------------------------------------------------
# prune_conversations
# --------------------------------------------------------------------------

def test_prune_conversations_clears_old_keeps_recent(cfg):
    from llamanager.db import DB
    db = DB(cfg.db_path)
    try:
        now = time.time()
        for rid, age_days in [("old", 40), ("recent", 1)]:
            db.insert_request(request_id=rid, origin_id=None, model="m", priority=50)
            db.update_request_status(
                rid, "done",
                enqueued_at=now - age_days * 86400,
                prompt_text="p", response_text="r")
        cleared = db.prune_conversations(30)
        assert cleared == 1
        assert db.query_one("SELECT prompt_text FROM requests WHERE id='old'")["prompt_text"] is None
        assert db.query_one("SELECT prompt_text FROM requests WHERE id='recent'")["prompt_text"] == "p"
        # retention 0 wipes everything remaining.
        assert db.prune_conversations(0) == 1
        assert db.query_one("SELECT prompt_text FROM requests WHERE id='recent'")["prompt_text"] is None
    finally:
        db.close()


# --------------------------------------------------------------------------
# Settings route + page
# --------------------------------------------------------------------------

def _admin_client(app) -> TestClient:
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    key = am.rotate_key(boot.id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303 and COOKIE_NAME in r.headers.get("set-cookie", "")
    return client


def _csrf(html: str) -> str:
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert m, "no csrf token in page"
    return m.group(1)


def test_settings_page_has_retention_field(app):
    with _admin_client(app) as client:
        body = client.get("/ui/settings").text
        assert "Conversation retention" in body
        assert 'action="/ui/settings/conversation-retention"' in body
        assert 'name="retention_days"' in body


def test_settings_save_persists_applies_and_clears(app):
    with _admin_client(app) as client:
        # Seed a stored conversation, then set retention to 0 — it should be
        # persisted, applied to the live cfg, and wipe the stored text now.
        db = app.state.db
        db.insert_request(request_id="seed", origin_id=None, model="m", priority=50)
        db.update_request_status("seed", "done", prompt_text="secret", response_text="reply")

        body = client.get("/ui/settings").text
        r = client.post("/ui/settings/conversation-retention",
                        data={"csrf_token": _csrf(body), "retention_days": "0"},
                        follow_redirects=False)
        assert r.status_code == 303
        cfg = app.state.cfg
        assert cfg.conversation_retention_days == 0
        assert "[conversation]" in cfg.config_path.read_text()
        row = db.query_one(
            "SELECT prompt_text, response_text FROM requests WHERE id='seed'")
        assert row["prompt_text"] is None and row["response_text"] is None

        # And a positive value round-trips back through the live cfg.
        r = client.post("/ui/settings/conversation-retention",
                        data={"csrf_token": _csrf(body), "retention_days": "14"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert app.state.cfg.conversation_retention_days == 14
