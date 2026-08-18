"""The merged Diffusion page: tiles, detail fragments, and the reopen hint.

/ui/setup-diffusion and /ui/diffusion-models described the same seven engines
and ten models twice — the first as seven fully-expanded setup cards ~10,000px
tall, the second as a catalog whose only advice for an uninstalled model was to
go back to the first page. These cover the one page that replaced them.
"""
import re

import pytest
from fastapi.testclient import TestClient

from llamanager import diffusion_catalog
from llamanager.api_ui import _DIFFUSION_ENGINE_TILES


def _admin(app):
    am = app.state.auth
    key = am.rotate_key(am.get_origin_by_name("bootstrap").id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303
    return client


@pytest.mark.parametrize("old", ["/ui/setup-diffusion", "/ui/diffusion-models"])
def test_the_old_pages_redirect_permanently(app, old):
    r = _admin(app).get(old, follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"] == "/ui/diffusion"


def test_page_shows_one_tile_per_engine_and_per_model(app):
    html = _admin(app).get("/ui/diffusion").text
    for engine in _DIFFUSION_ENGINE_TILES:
        assert f'data-engine="{engine}"' in html, engine
    for entry in diffusion_catalog.CATALOG:
        assert f'data-model="{entry.canonical_id}"' in html, entry.canonical_id


def test_page_does_not_render_the_audio_engines(app):
    """The old models page looped every adapter, so asr / whispercpp / sherpa
    each got a section announcing it had nothing — on the diffusion page."""
    html = _admin(app).get("/ui/diffusion").text
    for audio in ("whispercpp", "sherpa"):
        assert f'data-engine="{audio}"' not in html


@pytest.mark.parametrize("engine", _DIFFUSION_ENGINE_TILES)
def test_every_engine_fragment_renders(app, engine):
    r = _admin(app).get(f"/ui/diffusion/engine/{engine}")
    assert r.status_code == 200, r.text[:2000]
    # A fragment, not a page: it must not carry the whole chrome into the modal.
    assert "<html" not in r.text
    assert 'id="lm-modal-content"' in r.text


@pytest.mark.parametrize("model_id",
                         [e.canonical_id for e in diffusion_catalog.CATALOG])
def test_every_model_fragment_renders(app, model_id):
    """Four catalog ids contain a slash, which is why the id travels as a
    query parameter rather than a path segment."""
    r = _admin(app).get("/ui/diffusion/model", params={"id": model_id})
    assert r.status_code == 200, r.text[:2000]
    assert "<html" not in r.text


def test_unknown_ids_are_not_found(app):
    client = _admin(app)
    assert client.get("/ui/diffusion/engine/nope").status_code == 404
    assert client.get("/ui/diffusion/model", params={"id": "nope"}).status_code == 404


def test_open_hint_reopens_a_known_modal(app):
    html = _admin(app).get("/ui/diffusion?open=engine:z_image").text
    assert 'hx-get="/ui/diffusion/engine/z_image"' in html
    assert 'hx-trigger="load"' in html


def test_open_hint_survives_a_slash_in_the_model_id(app):
    html = _admin(app).get(
        "/ui/diffusion", params={"open": "model:MiniMaxAI/MiniMax-H3"}).text
    assert "/ui/diffusion/model?id=MiniMaxAI%2FMiniMax-H3" in html


@pytest.mark.parametrize("bad", [
    "engine:nope", "model:nope", "garbage", "engine:<script>alert(1)</script>",
    "model:../../etc/passwd",
])
def test_a_bad_open_hint_opens_nothing_and_is_not_reflected(app, bad):
    """The hint is validated against the ids this install actually has and
    rebuilt from them, so a crafted value never reaches the markup."""
    html = _admin(app).get("/ui/diffusion", params={"open": bad}).text
    assert 'hx-trigger="load"' not in html
    assert "alert(1)" not in html
    assert "etc/passwd" not in html


def test_the_page_only_polls_while_something_is_running(app):
    client = _admin(app)
    idle = client.get("/ui/diffusion").text
    assert 'hx-get="/ui/diffusion/_body"' not in idle

    # A component pull: it names the directory it lands in rather than a
    # repo the catalog knows, which is how a tile learns whose download it is.
    import json
    import time
    app.state.db.execute(
        "INSERT INTO downloads (id, source, files_json, status, bytes_done, "
        "bytes_total, started_at, family) VALUES (?,?,?,?,?,?,?,?)",
        ("dl1", "hf://Comfy-Org/Krea-2",
         json.dumps({"files": ["vae/qwen_image_vae.safetensors"],
                     "target_dir": "Krea-2-Turbo-Comfy/vae"}),
         "running", 5, 10, time.time(), "image"))

    busy = client.get("/ui/diffusion").text
    assert 'hx-get="/ui/diffusion/_body"' in busy
    # A poll that let its forms inherit the morph swap would morph <body>.
    assert 'hx-disinherit="hx-swap"' in busy
    assert "Downloading" in busy


def test_the_body_fragment_renders_on_its_own(app):
    r = _admin(app).get("/ui/diffusion/_body")
    assert r.status_code == 200
    assert "<html" not in r.text
    assert 'id="diffusion-body"' in r.text


def test_saving_an_engine_path_reopens_that_engine(app):
    client = _admin(app)
    m = re.search(r'name="csrf_token" value="([^"]+)"',
                  client.get("/ui/diffusion").text)
    r = client.post("/ui/setup/image/z-image",
                    data={"csrf_token": m.group(1), "z_image_python": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/diffusion?open=engine:z_image"


def _install_h3(app):
    """Enough of a MiniMax-H3 tree for detect() to claim it, so the modal
    renders its installed body (the profiles live there)."""
    from llamanager.engines import minimax_h3_comfy as m
    root = app.state.cfg.models_dir / "MiniMax-H3-Comfy"
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "diffusion_models" / m.UNET_FILE).write_bytes(b"x")
    (root / "vae" / m.AUDIO_VAE_FILE).write_bytes(b"x")
    return root


def _save_profile(app, model_id, name, fields):
    """Save one profile and reload it into the app, as the UI routes do."""
    from llamanager.config import Profile, load_config, save_profile
    path = app.state.cfg.config_path
    save_profile(path, model_id, name, Profile(name=name, **fields))
    app.state.cfg = load_config(path)


def test_a_model_with_profiles_is_still_offered_the_missing_builtins(app):
    """Built-ins are seeded at registration, so a profile shipped by a later
    update never reaches a model registered before it. Offering them only on
    the empty state is how two engine updates went unnoticed for a week."""
    _install_h3(app)
    _save_profile(app, "MiniMax-H3-Comfy",
                  "mine", {"image_steps": 4})
    html = _admin(app).get("/ui/diffusion/model",
                           params={"id": "MiniMax-H3-Comfy"}).text

    assert "Add the missing built-in" in html
    assert "built-ins not added" in html
    # The shipped names it does not have, and not the one it does.
    assert "h3-turbo-baked-8step" in html and "h3-ref2va-4step" in html
    assert "materialize-defaults" in html


def test_a_model_holding_every_builtin_is_offered_nothing(app):
    """The block must disappear once there is nothing left to add, or it
    reads as an action that never completes."""
    _install_h3(app)
    from llamanager.engines import minimax_h3_comfy as m
    for name, fields in m.default_profiles().items():
        _save_profile(app, "MiniMax-H3-Comfy",
                      name, fields)
    html = _admin(app).get("/ui/diffusion/model",
                           params={"id": "MiniMax-H3-Comfy"}).text

    assert "Add the missing built-in" not in html
    assert "not added" not in html


def test_materialize_adds_only_what_is_missing(app):
    """The button is additive: it must never overwrite a profile the
    operator has edited."""
    _install_h3(app)
    from llamanager.config import load_config
    _save_profile(app, "MiniMax-H3-Comfy",
                  "h3-turbo-4step", {"image_steps": 99})
    client = _admin(app)
    m = re.search(r'name="csrf_token" value="([^"]+)"',
                  client.get("/ui/diffusion").text)
    r = client.post("/ui/diffusion-models/profiles/materialize-defaults",
                    data={"csrf_token": m.group(1),
                          "model_id": "MiniMax-H3-Comfy",
                          "engine": "minimax_h3_comfy"},
                    follow_redirects=False)
    assert r.status_code == 303

    saved = load_config(app.state.cfg.config_path).get_model(
        "MiniMax-H3-Comfy").profiles
    assert "h3-ref2va-4step" in saved
    assert saved["h3-turbo-4step"].image_steps == 99, "edited profile clobbered"
