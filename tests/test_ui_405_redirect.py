"""A GET landing on a POST-only /ui action redirects to its section page.

hx-boost can leave a form's POST URL (e.g. /ui/models/load, the "Reload"
button) in the address bar/history; a later browser refresh or back/forward
(htmx history restore uses GET) would otherwise hit the POST-only route and
return a raw 405 "Method Not Allowed" in the UI. The app bounces those to the
section root instead. The API keeps the standard 405.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("path,section", [
    ("/ui/models/load", "/ui/models"),
    ("/ui/models/profiles/p/update", "/ui/models"),
    ("/ui/models/set-default", "/ui/models"),
    ("/ui/launch/server/start", "/ui/launch"),
    ("/ui/settings/mem-guard", "/ui/settings"),
    # These POST paths sit under a section that no longer owns them: image and
    # audio engines are configured from their own pages, not /ui/setup.
    ("/ui/setup/image/z-image", "/ui/diffusion"),
    ("/ui/setup/audio/asr", "/ui/asr"),
    ("/ui/diffusion-models/activate", "/ui/diffusion"),
    ("/ui/asr-models/pull", "/ui/asr"),
])
def test_get_on_post_only_ui_action_redirects(app, path, section):
    client = TestClient(app)
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == section


def test_api_405_is_not_redirected(app):
    # A POST to a GET-only API route keeps the real 405 (not a UI redirect).
    client = TestClient(app)
    r = client.post("/health", follow_redirects=False)
    assert r.status_code == 405
    assert "/ui/" not in (r.headers.get("location") or "")
