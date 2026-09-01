"""The admin gallery browses by origin folder, not as one flat stream.

Disk layout is ``<images_dir>/<day>/<origin>/<file>``, so an origin already
IS a folder. The admin image/video pages open inside the session origin's own
folder and can step up to a parent listing of every origin's folder. These
cover the two endpoints that make that possible: the per-origin filter on the
listing, and the folder listing itself.
"""
import io

from fastapi.testclient import TestClient


def _png_bytes(color=(10, 200, 10)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), color).save(buf, format="PNG")
    return buf.getvalue()


def _mk(am, name, **kw):
    return am.create_origin(name=name, allowed_models=["*"], **kw)


def _seed(cfg, day, origin, name, sidecar=None):
    d = cfg.images_dir / day / origin
    d.mkdir(parents=True, exist_ok=True)
    if name.endswith(".png"):
        (d / name).write_bytes(_png_bytes())
    else:
        (d / name).write_bytes(b"\x00\x00\x00\x18ftypmp42")
    if sidecar is not None:
        (d / (name + ".json")).write_text(sidecar, encoding="utf-8")
    return d / name


def _admin(app, name="folder-adm"):
    """Log a *named* admin origin into the cookie UI and return its client."""
    am = app.state.auth
    _, key = _mk(am, name, is_admin=True)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return client


# --------------------------------------------------------------- the filter

def test_listing_scopes_to_one_origin_folder(app):
    cfg = app.state.cfg
    _seed(cfg, "2026-05-06", "alice", "a1.png")
    _seed(cfg, "2026-05-07", "alice", "a2.png")
    _seed(cfg, "2026-05-07", "bob", "b1.png")
    client = _admin(app)

    everything = client.get("/ui/images/gallery").json()
    assert sorted(i["name"] for i in everything["items"]) == ["a1.png", "a2.png", "b1.png"]

    scoped = client.get("/ui/images/gallery", params={"origin": "alice"}).json()
    assert sorted(i["name"] for i in scoped["items"]) == ["a1.png", "a2.png"]
    assert {i["origin"] for i in scoped["items"]} == {"alice"}


def test_an_illegal_origin_name_is_a_400_not_a_walk(app):
    """The name is the directory; anything else must be refused by name."""
    client = _admin(app)
    for bad in ("../..", "al ice", "a/b", ""):
        r = client.get("/ui/images/gallery", params={"origin": bad})
        assert r.status_code == 400, (bad, r.status_code)
    # A well-formed name with nothing on disk is an empty folder, not an error.
    r = client.get("/ui/images/gallery", params={"origin": "nobody-here"})
    assert r.status_code == 200 and r.json()["items"] == []


# ---------------------------------------------------------- the parent level

def test_origin_folders_carry_counts_and_a_cover(app):
    cfg = app.state.cfg
    _seed(cfg, "2026-05-06", "alice", "a1.png", sidecar='{"width": 64, "height": 32}')
    _seed(cfg, "2026-05-07", "alice", "a2.png")
    _seed(cfg, "2026-05-07", "bob", "b1.png")
    _seed(cfg, "2026-05-07", "bob", "clip.mp4")
    client = _admin(app)

    payload = client.get("/ui/images/gallery/origins").json()
    by_name = {e["origin"]: e for e in payload["origins"]}
    assert by_name["alice"]["count"] == 2
    assert by_name["bob"]["count"] == 2
    assert by_name["alice"]["bytes"] > 0
    cover = by_name["alice"]["cover"]
    assert cover["origin"] == "alice"
    assert cover["thumb_url"].startswith("/ui/images/thumb/")
    assert cover["url"].startswith("/ui/images/file/")


def test_folders_are_scoped_by_kind(app):
    """The video page's folder view must not count PNGs, and vice versa."""
    cfg = app.state.cfg
    _seed(cfg, "2026-05-07", "bob", "b1.png")
    _seed(cfg, "2026-05-07", "bob", "clip.mp4")
    _seed(cfg, "2026-05-07", "carol", "c1.png")
    client = _admin(app)

    vids = {e["origin"]: e for e in
            client.get("/ui/images/gallery/origins", params={"kind": "video"}).json()["origins"]}
    assert vids["bob"]["count"] == 1
    assert vids["bob"]["cover"]["name"] == "clip.mp4"
    # carol has no clips, so her folder is absent from the video view.
    assert "carol" not in vids

    imgs = {e["origin"]: e for e in
            client.get("/ui/images/gallery/origins", params={"kind": "image"}).json()["origins"]}
    assert imgs["bob"]["count"] == 1 and imgs["carol"]["count"] == 1


def test_the_callers_own_folder_is_always_listed(app):
    """The admin steps up from their own folder; it must still be there,
    at an honest zero, even before they have generated anything."""
    cfg = app.state.cfg
    _seed(cfg, "2026-05-07", "bob", "b1.png")
    client = _admin(app, name="lonely-admin")
    payload = client.get("/ui/images/gallery/origins").json()
    assert payload["self"] == "lonely-admin"
    mine = [e for e in payload["origins"] if e["origin"] == "lonely-admin"]
    assert len(mine) == 1
    assert mine[0]["count"] == 0
    assert mine[0]["cover"] is None
    assert mine[0]["latest"] is None
    # And an origin with media sorts ahead of the empty one.
    assert payload["origins"][0]["origin"] == "bob"


def test_folder_view_needs_an_admin_session(app):
    """Same gate as the rest of /ui: no cookie, no folder listing."""
    r = TestClient(app).get("/ui/images/gallery/origins", follow_redirects=False)
    assert r.status_code in (302, 303, 307, 401, 403)


# ------------------------------------------------------------- the page wiring

def test_pages_open_in_the_session_origins_folder(app):
    """Both admin pages must hand the client its own folder name and the
    folder-listing URL, or the feed cannot open scoped."""
    client = _admin(app, name="page-adm")
    for path in ("/ui/images", "/ui/videos"):
        html = client.get(path).text
        assert 'selfOrigin: "page-adm"' in html, path
        assert 'originsUrl: "/ui/images/gallery/origins"' in html, path
        assert 'id="gen-scopebar"' in html, path
        assert "LMGalleryScope" in html, path


def test_pagination_stays_inside_the_folder(app):
    """"Load older" must not walk out of the folder being browsed."""
    cfg = app.state.cfg
    for i in range(4):
        _seed(cfg, "2026-05-07", "alice", f"a{i}.png")
    for i in range(4):
        _seed(cfg, "2026-05-07", "bob", f"b{i}.png")
    client = _admin(app)

    seen, before, pages = [], None, 0
    while pages < 10:
        params = {"origin": "alice", "limit": 2}
        if before is not None:
            params["before"] = before
        payload = client.get("/ui/images/gallery", params=params).json()
        assert all(i["origin"] == "alice" for i in payload["items"]), payload
        seen += [i["name"] for i in payload["items"]]
        before = payload["next_before"]
        pages += 1
        if before is None:
            break
    assert sorted(seen) == ["a0.png", "a1.png", "a2.png", "a3.png"]
