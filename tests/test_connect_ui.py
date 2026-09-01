"""The Connect page: minting an MCP key and handing back a usable config.

The snippets are the whole point of the page, so these tests assert that a
freshly minted key actually reaches them — a page that renders but still
says YOUR_KEY would send people to a client that cannot authenticate.
The login flow mirrors test_mem_guard_ui (rotate the bootstrap key).
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


def test_connect_page_renders_with_nav_entry(app):
    with _admin_client(app) as client:
        body = client.get("/ui/connect").text
    assert 'action="/ui/connect/create-key"' in body
    # The rail's refresh button derives its URL from `active`, so the nav
    # key and the path segment must agree.
    assert 'href="/ui/connect"' in body
    assert "claude mcp add --transport http" in body
    assert "mcp-stdio" in body


def test_page_shows_placeholder_until_a_key_exists(app):
    with _admin_client(app) as client:
        body = client.get("/ui/connect").text
    assert "YOUR_KEY" in body
    assert "Shown once" not in body


def test_minting_a_key_fills_every_snippet(app):
    with _admin_client(app) as client:
        page = client.get("/ui/connect").text
        r = client.post("/ui/connect/create-key",
                        data={"csrf_token": _csrf(page), "name": "mcp",
                              "is_admin": "true"})
    assert r.status_code == 200
    body = r.text
    key = re.search(r'<div class="lm-key">(lm_[A-Za-z0-9_\-]+)</div>', body)
    assert key, "minted key not revealed"
    secret = key.group(1)

    assert "Shown once" in body
    assert "YOUR_KEY" not in body
    # Every client snippet must carry the real credential.
    assert f'Authorization: Bearer {secret}' in body
    assert f'"LLAMANAGER_ADMIN_KEY": "{secret}"' in body
    assert f'"Authorization": "Bearer {secret}"' in body


def test_snippets_use_this_daemons_real_address(app):
    """A non-default port must not be silently rendered as 7200."""
    app.state.cfg.port = 7311
    with _admin_client(app) as client:
        body = client.get("/ui/connect").text
    assert "http://127.0.0.1:7311/mcp" in body
    assert "http://127.0.0.1:7200/mcp" not in body


def test_minted_origin_is_a_real_manageable_origin(app):
    # Assertions stay inside the client block: leaving it runs the app's
    # lifespan shutdown, which closes the database.
    with _admin_client(app) as client:
        page = client.get("/ui/connect").text
        client.post("/ui/connect/create-key",
                    data={"csrf_token": _csrf(page), "name": "claude-desktop",
                          "is_admin": "true"})
        origin = app.state.auth.get_origin_by_name("claude-desktop")
        assert origin is not None and origin.is_admin
        # It appears on the Origins page, so it can be rotated or revoked
        # there like any other credential.
        assert "claude-desktop" in client.get("/ui/origins").text


def test_non_admin_key_can_be_minted(app):
    """Opting out of admin must actually produce a non-admin origin."""
    with _admin_client(app) as client:
        page = client.get("/ui/connect").text
        client.post("/ui/connect/create-key",
                    data={"csrf_token": _csrf(page), "name": "gen-only"})
        origin = app.state.auth.get_origin_by_name("gen-only")
        assert origin is not None and not origin.is_admin


def test_duplicate_name_is_reported_inline(app):
    with _admin_client(app) as client:
        page = client.get("/ui/connect").text
        token = _csrf(page)
        client.post("/ui/connect/create-key",
                    data={"csrf_token": token, "name": "mcp"})
        page2 = client.get("/ui/connect").text
        r = client.post("/ui/connect/create-key",
                        data={"csrf_token": _csrf(page2), "name": "mcp"})
    assert r.status_code == 409
    assert "already exists" in r.text


def test_create_key_requires_csrf(app):
    """No token, no origin.

    require_csrf answers a missing token with a redirect rather than a 403
    (a stale token must not dead-end the user), so the property worth
    asserting is that nothing was minted.
    """
    with _admin_client(app) as client:
        r = client.post("/ui/connect/create-key", data={"name": "nope"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert app.state.auth.get_origin_by_name("nope") is None
