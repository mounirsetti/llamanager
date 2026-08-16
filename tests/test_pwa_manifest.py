"""The PWA manifest is per-surface.

A home-screen icon opens the manifest's ``start_url``, so an icon installed
from /images must reopen /images. The app-wide manifest that preceded this
sent every install to /ui/, dropping non-admin key holders on the operator
login with no way out (standalone display has no address bar).
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEMPLATES = Path(__file__).resolve().parents[1] / "llamanager" / "templates"

SURFACES = {
    "admin": "/ui/",
    "chat": "/chat",
    "images": "/images",
    "videos": "/videos",
}

# Which surface each installable page must declare.
PAGE_SURFACE = {
    "base.html": "admin",
    "login.html": "admin",
    "chat_public.html": "chat",
    "images_public.html": "images",
    "videos_public.html": "videos",
}


@pytest.mark.parametrize("surface,start_url", sorted(SURFACES.items()))
def test_manifest_start_url_is_per_surface(app, surface, start_url):
    client = TestClient(app)
    r = client.get(f"/manifest.json?app={surface}")
    assert r.status_code == 200
    body = r.json()
    assert body["start_url"] == start_url
    # Scope stays app-wide so in-app links don't bounce out to a browser tab.
    assert body["scope"] == "/"
    assert body["icons"]


def test_manifest_requires_a_surface(app):
    """No default: a page that forgets ?app= fails loudly instead of
    silently inheriting somebody else's start_url."""
    client = TestClient(app)
    assert client.get("/manifest.json").status_code == 400
    assert client.get("/manifest.json?app=nope").status_code == 400


@pytest.mark.parametrize("page,surface", sorted(PAGE_SURFACE.items()))
def test_each_page_links_its_own_manifest(page, surface):
    html = (TEMPLATES / page).read_text()
    assert f'href="/manifest.json?app={surface}"' in html
    assert 'href="/manifest.json"' not in html
