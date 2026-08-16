"""One origin must never see another origin's generated history.

The gallery boundary is the per-origin directory under ``images_dir``, so it
only holds while name -> directory is injective. It once was not: names were
sanitised at write time ("".join(c for c in name if c.isalnum() or c in "-_")),
so the distinct origins "gal probe" and "gal.probe" both wrote to "galprobe"
and each could list and download the other's images.
"""
import pytest
from fastapi.testclient import TestClient

from llamanager.auth import validate_origin_name


def _mk(am, name, **kw):
    return am.create_origin(name=name, allowed_models=["*"], **kw)


def test_colliding_names_are_rejected_at_creation(app):
    """The two names from the original leak can no longer both exist."""
    am = app.state.auth
    _mk(am, "galprobe")
    for bad in ("gal probe", "gal.probe", "gal/probe", "../escape", "",
                "-leading", "a" * 65):
        with pytest.raises(ValueError):
            _mk(am, bad)


def test_gallery_dir_is_the_origin_name_verbatim(app, tmp_path):
    from llamanager.image_runner import _gallery_dir
    cfg = app.state.cfg
    d = _gallery_dir(cfg, "design-to-html")
    assert d.name == "design-to-html"
    assert d.parent.parent == cfg.images_dir
    # An origin predating validation must fail loudly rather than write into
    # a folder that is not its own.
    with pytest.raises(ValueError):
        _gallery_dir(cfg, "design to html")


def test_one_origin_cannot_read_anothers_gallery(app):
    """End to end: B may not list or download A's file."""
    am = app.state.auth
    cfg = app.state.cfg
    _, key_a = _mk(am, "origin-a")
    _, key_b = _mk(am, "origin-b")

    day = "2026-01-02"
    d = cfg.images_dir / day / "origin-a"
    d.mkdir(parents=True)
    (d / "private.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    client = TestClient(app)
    a_hdr = {"Authorization": f"Bearer {key_a}"}
    b_hdr = {"Authorization": f"Bearer {key_b}"}

    owner = client.get("/images/gallery", headers=a_hdr).json()
    assert [i["name"] for i in owner["items"]] == ["private.png"]

    other = client.get("/images/gallery", headers=b_hdr).json()
    assert other["items"] == []

    assert client.get(f"/images/file/{day}/origin-a/private.png",
                      headers=a_hdr).status_code == 200
    r = client.get(f"/images/file/{day}/origin-a/private.png", headers=b_hdr)
    assert r.status_code == 403
    assert "another origin" in r.json()["detail"]


def test_missing_file_is_404_not_500(app):
    """The error paths raised NameError before HTTPException was imported."""
    am = app.state.auth
    _, key = _mk(am, "origin-c")
    client = TestClient(app)
    r = client.get("/images/file/2026-01-02/origin-c/nope.png",
                   headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 404


@pytest.mark.parametrize("name", ["bootstrap", "local", "Aya", "read95",
                                  "continue-agent", "design-to-html"])
def test_names_in_use_stay_valid(name):
    """Names this deployment already relies on must keep working."""
    assert validate_origin_name(name) == name
