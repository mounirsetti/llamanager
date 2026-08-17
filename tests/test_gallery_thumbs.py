"""Gallery thumbnails and the URLs the listing hands out.

Two regressions guarded here:

* The public (bearer) listing used to emit ``/ui/images/file/...`` URLs —
  the cookie-authenticated admin route — so a bearer-only client's tile
  fetches were redirected to the login page and the grid stayed blank.
* Tiles used to download the full original. Now each file has a JPEG
  thumbnail built on demand under ``images_dir/.thumbs`` and served by
  ``/images/thumb`` (public) and ``/ui/images/thumb`` (admin), and the
  ``.thumbs`` cache never shows up in the listing.
"""
import io

import pytest
from fastapi.testclient import TestClient

from llamanager import thumbs


def _png_bytes(w=64, h=32, color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _mk(am, name, **kw):
    return am.create_origin(name=name, allowed_models=["*"], **kw)


@pytest.fixture
def seeded(app):
    am = app.state.auth
    cfg = app.state.cfg
    _, key = _mk(am, "thumb-owner")
    day = "2026-03-04"
    d = cfg.images_dir / day / "thumb-owner"
    d.mkdir(parents=True)
    (d / "pic.png").write_bytes(_png_bytes(640, 320))
    (d / "pic.png.json").write_text('{"width": 640, "height": 320, "prompt": "p"}')
    return {"key": key, "day": day, "cfg": cfg}


def test_public_listing_emits_public_urls_and_thumb_url(app, seeded):
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {seeded['key']}"}
    items = client.get("/images/gallery", headers=hdr).json()["items"]
    assert len(items) == 1
    it = items[0]
    assert it["url"] == f"/images/file/{seeded['day']}/thumb-owner/pic.png"
    assert it["thumb_url"] == f"/images/thumb/{seeded['day']}/thumb-owner/pic.png"
    assert (it["width"], it["height"]) == (640, 320)
    # And the URL it hands out actually works with the same bearer (no
    # redirect to /ui/login).
    r = client.get(it["url"], headers=hdr, follow_redirects=False)
    assert r.status_code == 200
    assert r.headers["cache-control"].startswith("private, max-age=")


def test_thumb_route_builds_and_caches_a_small_jpeg(app, seeded):
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {seeded['key']}"}
    url = f"/images/thumb/{seeded['day']}/thumb-owner/pic.png"
    r = client.get(url, headers=hdr)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"].startswith("private, max-age=")
    from PIL import Image
    im = Image.open(io.BytesIO(r.content))
    assert im.format == "JPEG"
    assert max(im.size) <= thumbs.THUMB_PX
    assert im.size[0] / im.size[1] == pytest.approx(2.0, abs=0.02)
    cached = thumbs.thumb_path(seeded["cfg"].images_dir, seeded["day"],
                               "thumb-owner", "pic.png")
    assert cached.is_file()
    mtime = cached.stat().st_mtime
    assert client.get(url, headers=hdr).status_code == 200
    assert cached.stat().st_mtime == mtime  # served from cache, not rebuilt


def test_thumb_cache_dir_is_invisible_to_the_listing(app, seeded):
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {seeded['key']}"}
    assert client.get(f"/images/thumb/{seeded['day']}/thumb-owner/pic.png",
                      headers=hdr).status_code == 200
    # Admin listing walks every day dir; ``.thumbs`` must not be one.
    from llamanager.api_ui import _list_gallery
    listing = _list_gallery(seeded["cfg"].images_dir)
    assert [i["name"] for i in listing["items"]] == ["pic.png"]
    assert all(i["url"].startswith("/ui/images/file/") for i in listing["items"])
    assert all(i["thumb_url"].startswith("/ui/images/thumb/") for i in listing["items"])


def test_thumb_route_is_origin_confined(app, seeded):
    am = app.state.auth
    _, other = _mk(am, "thumb-other")
    client = TestClient(app)
    r = client.get(f"/images/thumb/{seeded['day']}/thumb-owner/pic.png",
                   headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 403


def test_thumb_of_undecodable_png_is_a_loud_503(app, seeded):
    cfg = seeded["cfg"]
    d = cfg.images_dir / seeded["day"] / "thumb-owner"
    (d / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\nnot really")
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {seeded['key']}"}
    r = client.get(f"/images/thumb/{seeded['day']}/thumb-owner/broken.png",
                   headers=hdr)
    assert r.status_code == 503
    assert "thumbnail" in r.json()["detail"]
    # Nothing half-written was left behind to be served next time.
    assert not thumbs.thumb_path(cfg.images_dir, seeded["day"],
                                 "thumb-owner", "broken.png").exists()


def test_thumb_regenerates_when_source_changes(tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(_png_bytes(64, 64, (0, 0, 255)))
    dst = tmp_path / ".thumbs" / "a.png.jpg"
    thumbs.ensure_thumbnail(src, dst)
    first = dst.read_bytes()
    import os, time
    src.write_bytes(_png_bytes(64, 64, (0, 255, 0)))
    os.utime(src, (time.time() + 10, time.time() + 10))
    thumbs.ensure_thumbnail(src, dst)
    assert dst.read_bytes() != first


def test_disk_cap_drops_thumbnail_with_original(tmp_path):
    src = tmp_path / "2026-01-01" / "o" / "x.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(_png_bytes())
    dst = thumbs.thumb_path(tmp_path, "2026-01-01", "o", "x.png")
    thumbs.ensure_thumbnail(src, dst)
    assert dst.exists()
    thumbs.drop_thumbnail(tmp_path, src)
    assert not dst.exists()
