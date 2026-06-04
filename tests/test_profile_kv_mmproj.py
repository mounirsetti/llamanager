"""Profile editor: KV-cache type control + mmproj dropdown.

Verifies the new first-class controls render, the KV type round-trips through
save → reload → engine flags (with flash-attn auto-enabled), and the mmproj
field is now a real <select> rather than a free-text input. Login mirrors
test_smoke (rotate the bootstrap key).
"""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from llamanager.api_ui import COOKIE_NAME


def _admin_client(app) -> TestClient:
    am = app.state.auth
    key = am.rotate_key(am.get_origin_by_name("bootstrap").id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303
    return client


def _csrf(html: str) -> str:
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert m
    return m.group(1)


def _seed_model(app, *, mmproj=False) -> None:
    """A zero-byte test/model.gguf so the registry lists it and renders a
    card (the editor controls live on the card). Optionally seed projectors
    in the model's folder and elsewhere."""
    md = app.state.cfg.models_dir
    (md / "test").mkdir(parents=True, exist_ok=True)
    (md / "test" / "model.gguf").write_bytes(b"")
    if mmproj:
        (md / "other").mkdir(parents=True, exist_ok=True)
        (md / "test" / "mmproj-beside.gguf").write_bytes(b"GGUF\x00")
        (md / "other" / "mmproj-far.gguf").write_bytes(b"GGUF\x00")


def test_models_page_has_kv_and_mmproj_controls(app):
    _seed_model(app)
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        # KV cache type is a first-class select now.
        assert 'name="kv_cache_type"' in body
        assert "q8_0" in body
        # mmproj is a <select>, not a free-text input.
        assert re.search(r'<select[^>]*name="mmproj"', body)
        assert 'list="mmproj-options"' not in body  # old datalist input gone


def test_kv_cache_type_saves_and_translates(app):
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        r = client.post("/ui/models/profiles/test/update", data={
            "csrf_token": _csrf(body),
            "model_id": "test/model.gguf",
            "new_name": "test",
            "ctx_size": "65536",
            "kv_cache_type": "q8_0",
            "ram_spill_policy": "default",
            "args_json": "{}",
        }, follow_redirects=False)
        assert r.status_code in (303, 200)
        prof = app.state.cfg.models["test/model.gguf"].profiles["test"]
        assert prof.kv_cache_type == "q8_0"
        # translates to the engine flags + flash attention
        from llamanager.server_manager import _basic_to_args
        from pathlib import Path
        args = _basic_to_args(prof, "llama", Path("/none.gguf"))
        assert args["cache-type-k"] == "q8_0"
        assert args["cache-type-v"] == "q8_0"
        assert args["flash-attn"] == "on"


def test_invalid_kv_type_rejected(app):
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        r = client.post("/ui/models/profiles/test/update", data={
            "csrf_token": _csrf(body),
            "model_id": "test/model.gguf",
            "new_name": "test",
            "ctx_size": "4096",
            "kv_cache_type": "bogus",
            "ram_spill_policy": "default",
            "args_json": "{}",
        }, follow_redirects=False)
        assert r.status_code == 400
        # unchanged
        assert app.state.cfg.models["test/model.gguf"].profiles["test"].kv_cache_type == ""


def test_mmproj_prioritizes_same_folder(app):
    """The per-model split lists same-folder projectors before others."""
    _seed_model(app, mmproj=True)
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        assert "mmproj-beside.gguf" in body          # same-folder, by basename
        assert "other/mmproj-far.gguf" in body       # other folder, full path
        i_same = body.find("In this model's folder")
        i_other = body.find("Other folders")
        assert i_same != -1 and i_other != -1 and i_same < i_other
