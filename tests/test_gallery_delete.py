"""Admin-only permanent removal of a gallery item (original + sidecar +
cached thumbnail), via DELETE /admin/gallery/{day}/{origin}/{name}."""
import io

from fastapi.testclient import TestClient

from llamanager import thumbs


def _png_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


def _mk(am, name, **kw):
    return am.create_origin(name=name, allowed_models=["*"], **kw)


def _seed(cfg, day="2026-05-06", origin="someone", name="pic.png"):
    d = cfg.images_dir / day / origin
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(_png_bytes())
    (d / (name + ".json")).write_text('{"prompt": "p"}')
    thumbs.ensure_thumbnail(d / name, thumbs.thumb_path(cfg.images_dir, day, origin, name))
    return d / name


def test_admin_delete_removes_file_sidecar_thumbnail_and_empty_dirs(app):
    cfg = app.state.cfg
    p = _seed(cfg)
    thumb = thumbs.thumb_path(cfg.images_dir, "2026-05-06", "someone", "pic.png")
    assert p.exists() and thumb.exists()
    _, key = _mk(app.state.auth, "adm-del", is_admin=True)
    client = TestClient(app)
    r = client.delete("/admin/gallery/2026-05-06/someone/pic.png",
                      headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["removed"]["sidecar"] and body["removed"]["thumbnail"]
    assert not p.exists()
    assert not p.with_name("pic.png.json").exists()
    assert not thumb.exists()
    # Nothing else was in that origin/day → folders pruned.
    assert not (cfg.images_dir / "2026-05-06").exists()
    assert not (cfg.images_dir / thumbs.THUMBS_DIRNAME / "2026-05-06").exists()
    # Audited.
    ev = app.state.db.query_one(
        "SELECT payload_json FROM events WHERE kind='gallery_delete' "
        "ORDER BY rowid DESC LIMIT 1")
    assert '"by": "adm-del"' in ev["payload_json"]
    # Gone from the listing; second delete is a 404, not a 500.
    assert client.get("/admin/gallery/x", headers={"Authorization": f"Bearer {key}"}).status_code in (404, 405)
    r2 = client.delete("/admin/gallery/2026-05-06/someone/pic.png",
                       headers={"Authorization": f"Bearer {key}"})
    assert r2.status_code == 404


def test_delete_keeps_siblings_and_their_folders(app):
    cfg = app.state.cfg
    _seed(cfg, name="a.png")
    b = _seed(cfg, name="b.png")
    _, key = _mk(app.state.auth, "adm-del2", is_admin=True)
    client = TestClient(app)
    r = client.delete("/admin/gallery/2026-05-06/someone/a.png",
                      headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert b.exists()
    assert thumbs.thumb_path(cfg.images_dir, "2026-05-06", "someone", "b.png").exists()


def test_non_admin_cannot_delete(app):
    cfg = app.state.cfg
    p = _seed(cfg, origin="victim")
    _, key = _mk(app.state.auth, "victim", is_admin=False)
    client = TestClient(app)
    r = client.delete("/admin/gallery/2026-05-06/victim/pic.png",
                      headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 403
    assert p.exists()
    assert client.delete("/admin/gallery/2026-05-06/victim/pic.png").status_code == 401
    assert p.exists()


def test_delete_rejects_traversal_and_foreign_suffixes(app):
    cfg = app.state.cfg
    _seed(cfg)
    _, key = _mk(app.state.auth, "adm-del3", is_admin=True)
    client = TestClient(app)
    hdr = {"Authorization": f"Bearer {key}"}
    assert client.delete("/admin/gallery/2026-05-06/someone/pic.png.json",
                         headers=hdr).status_code == 400
    assert client.delete("/admin/gallery/2026-05-06/..%2F..%2Fx/pic.png",
                         headers=hdr).status_code in (400, 404)
    assert (cfg.images_dir / "2026-05-06" / "someone" / "pic.png").exists()
