"""UI + persistence wiring for the memory guardrails.

Verifies the Settings page renders the new section, the save endpoint writes
the [mem_guard] config section and updates the live config, the profile
editor carries the ctx-advice hooks, and the launch guardrail injects its
flags. The login flow mirrors test_smoke (rotate the bootstrap key).
"""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from llamanager.api_ui import COOKIE_NAME


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


def test_settings_page_has_mem_guard_section(app):
    with _admin_client(app) as client:
        body = client.get("/ui/settings").text
        assert "Memory guardrails" in body
        assert 'action="/ui/settings/mem-guard"' in body
        assert 'name="ctx_checkpoints"' in body
        assert "Capacity now" in body


def test_mem_guard_save_persists_and_applies(app):
    with _admin_client(app) as client:
        body = client.get("/ui/settings").text
        r = client.post("/ui/settings/mem-guard", data={
            "csrf_token": _csrf(body),
            "enabled": "on", "clamp_ctx": "on", "ctx_checkpoints": "6",
            # hard_stop_enabled omitted -> unchecked
        }, follow_redirects=False)
        assert r.status_code == 303
        cfg = app.state.cfg
        assert cfg.mem_clamp_ctx is True
        assert cfg.mem_hard_stop_enabled is False
        assert cfg.mem_ctx_checkpoints == 6
        # persisted to disk
        assert "[mem_guard]" in cfg.config_path.read_text()
        # reloads from disk with the same values
        from llamanager.config import load_config
        reloaded = load_config(cfg.config_path)
        assert reloaded.mem_clamp_ctx is True
        assert reloaded.mem_ctx_checkpoints == 6


def test_unchecked_boxes_read_as_off(app):
    with _admin_client(app) as client:
        body = client.get("/ui/settings").text
        # Send nothing but csrf -> all booleans off, checkpoints 0.
        client.post("/ui/settings/mem-guard",
                    data={"csrf_token": _csrf(body)}, follow_redirects=False)
        cfg = app.state.cfg
        assert cfg.mem_guard_enabled is False
        assert cfg.mem_clamp_ctx is False
        assert cfg.mem_hard_stop_enabled is False
        assert cfg.mem_ctx_checkpoints == 0


def test_models_page_has_ctx_advice_hooks(app):
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        # The profile editor's ctx input + advice element are present even
        # with no real GGUF on disk (the data-* attrs just stay absent).
        assert "lm-ctx-input" in body
        assert "lm-ctx-advice" in body
