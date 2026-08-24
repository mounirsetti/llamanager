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


def test_keep_warm_is_settable_from_the_coexistence_form(app):
    """The warm-server window belongs to the coexistence form: it is the same
    trade, since a warm server holds its VRAM for the whole window."""
    from llamanager.config import load_config
    client = _admin(app)
    page = client.get("/ui/diffusion").text
    assert 'name="comfy_keep_warm_s"' in page

    m = re.search(r'name="csrf_token" value="([^"]+)"', page)
    r = client.post("/ui/setup/coexistence",
                    data={"csrf_token": m.group(1),
                          "unload_text_on_arrival": "on",
                          "comfy_keep_warm_s": "300"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert load_config(app.state.cfg.config_path).comfy_keep_warm_s == 300
    assert 'value="300"' in client.get("/ui/diffusion").text


@pytest.mark.parametrize("bad", ["-1", "5000", "soon"])
def test_a_bad_keep_warm_is_refused_not_coerced(app, bad):
    """Silently clamping would leave a server holding the card for a window
    nobody asked for."""
    from llamanager.config import load_config
    client = _admin(app)
    m = re.search(r'name="csrf_token" value="([^"]+)"',
                  client.get("/ui/diffusion").text)
    r = client.post("/ui/setup/coexistence",
                    data={"csrf_token": m.group(1), "comfy_keep_warm_s": bad},
                    follow_redirects=False)
    assert r.status_code == 400
    assert load_config(app.state.cfg.config_path).comfy_keep_warm_s == 0


# ------------------------------------------------------- warm / prewarm


def test_warm_status_reports_cold_for_a_model_with_no_server(app):
    _install_h3(app)
    r = _admin(app).get("/ui/comfy/warm", params={"model": "MiniMax-H3-Comfy"})
    assert r.status_code == 200
    body = r.json()
    assert body["warm"] is False
    assert body["model"] == "MiniMax-H3-Comfy"


def test_warm_status_needs_a_model(app):
    assert _admin(app).get("/ui/comfy/warm").status_code == 400


def test_prewarm_refuses_without_a_keep_warm_window(app):
    """With a window of 0 the prewarmed server is reaped immediately, so the
    button would report success and change nothing."""
    _install_h3(app)
    client = _admin(app)
    m = re.search(r'name="csrf_token" value="([^"]+)"',
                  client.get("/ui/diffusion").text)
    r = client.post("/ui/comfy/prewarm",
                    data={"csrf_token": m.group(1),
                          "model": "MiniMax-H3-Comfy"})
    assert r.status_code == 400
    assert "keep-warm" in r.json()["detail"]


def test_unwarm_gives_the_card_back(app):
    client = _admin(app)
    m = re.search(r'name="csrf_token" value="([^"]+)"',
                  client.get("/ui/diffusion").text)
    r = client.post("/ui/comfy/unwarm", data={"csrf_token": m.group(1)})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_warm_status_says_whether_prewarming_can_work(app):
    """can_prewarm is the window, not the server: the page uses it to
    explain why the button is disabled instead of just disabling it."""
    from llamanager.config import load_config, update_image_config
    _install_h3(app)
    client = _admin(app)
    assert client.get("/ui/comfy/warm",
                      params={"model": "MiniMax-H3-Comfy"}
                      ).json()["can_prewarm"] is False

    update_image_config(app.state.cfg.config_path, comfy_keep_warm_s=300)
    app.state.cfg = load_config(app.state.cfg.config_path)
    body = client.get("/ui/comfy/warm",
                      params={"model": "MiniMax-H3-Comfy"}).json()
    assert body["can_prewarm"] is True and body["keep_warm_s"] == 300


# ------------------------------------------------- reattaching to a job


def test_every_generation_page_can_recover_a_running_job(app):
    """A generation belongs to the queue, not to the browser connection that
    started it: closing the tab during a four-minute clip used to leave the
    page looking idle while the GPU worked."""
    client = _admin(app)
    for url in ("/ui/images", "/ui/videos"):
        html = client.get(url).text
        assert "statusUrl" in html, url
        # Either the page's own reattach, or the shared banner.
        assert ("LM_HAS_REATTACH" in html) or ("lm-reattach" in html), url


def test_the_shared_banner_stands_down_where_a_page_has_its_own(app):
    html = _admin(app).get("/ui/images").text
    assert "window.LM_HAS_REATTACH = true" in html
    assert "lm-reattach" in html, "banner still included, just inert"


def test_public_status_needs_a_bearer(app):
    from fastapi.testclient import TestClient
    assert TestClient(app).get("/v1/images/status").status_code == 401


def test_public_status_reports_idle_for_an_authorised_caller(app):
    from fastapi.testclient import TestClient
    am = app.state.auth
    am.ensure_bootstrap()
    _, key = am.create_origin(name="public-page")
    r = TestClient(app).get("/v1/images/status",
                            headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["busy"] is False and "queued" in body


def test_prewarm_loads_the_profile_the_page_has_selected(app):
    """Measured: prewarming the engine's first default while the request used
    a different profile gave 35.8 s, where prewarming the selected one gave
    15.5 s — the profile picks the transformer, so warming the wrong one
    reports warm and still pays the load."""
    import inspect
    from llamanager import api_ui
    from pathlib import Path

    src = inspect.getsource(api_ui._spawn_prewarm)
    assert "cfg.get_profile(model_id, profile_name)" in src
    # And the button has to send it, or the parameter is decorative.
    partial = (Path(api_ui.__file__).parent / "templates"
               / "_composer_warm.html").read_text()
    assert 'getElementById("lm-img-profile")' in partial
    assert 'fd.append("profile"' in partial


def test_prewarm_prefers_the_models_own_default_over_the_engines(app):
    """With no profile named, the model's configured default is closer to
    what the next request will use than the engine's first built-in."""
    import inspect
    from llamanager import api_ui

    src = inspect.getsource(api_ui._spawn_prewarm)
    own = src.find("default_profile")
    engine_default = src.find("adapter.default_profiles()")
    assert own != -1 and own < engine_default


# ------------------------------------------------- composer partials


def _partial(name):
    from pathlib import Path
    from llamanager import api_ui
    return (Path(api_ui.__file__).parent / "templates" / name).read_text()


def test_the_progress_banner_is_hidden_until_there_is_a_job():
    """`display: flex` outranks the hidden attribute, so without an explicit
    rule the banner sat on every page announcing a job nobody started."""
    css = _partial("_composer_reattach.html")
    assert ".lm-reattach[hidden] { display: none; }" in css


@pytest.mark.parametrize("name", ["_composer_advanced.html",
                                  "_composer_warm.html",
                                  "_composer_reattach.html",
                                  "_composer_page.html"])
def test_partials_wait_for_the_page_config(name):
    """They are included above the page's inline LM_IMAGES_CFG assignment, so
    reading it at parse time found nothing and every one of them returned
    early — the advanced fold never ran on any page."""
    js = _partial(name)
    assert "DOMContentLoaded" in js, name
    assert "function boot()" in js, name


@pytest.mark.parametrize("page", ["images.html", "videos.html",
                                  "images_public.html", "videos_public.html"])
def test_the_banner_is_not_buried_in_the_settings_popover(page):
    """It first shipped inside #gen-settings, which is hidden: a progress
    banner nobody could see unless they opened the settings drawer."""
    html = _partial(page)
    banner = html.find('{% include "_composer_reattach.html" %}')
    popover = html.find('id="gen-settings"')
    assert banner != -1, page
    assert banner < popover, f"{page}: banner is inside the settings popover"


def test_the_reload_button_is_for_pages_with_no_menu():
    """The admin pages reach a refresh from the rail footer in base.html; the
    public pages have no menu at all, so the button is theirs. It navigates
    to the canonical GET URL rather than reloading, for the same reason the
    rail's does: reload repeats however the page was reached."""
    js = _partial("_composer_page.html")
    assert 'querySelector(".gen-topbar__actions")' in js
    assert ".gen-feedhead" not in js, "would put a second button on admin pages"
    assert "window.location.assign(window.location.pathname)" in js
    # Comments may discuss reload(); no line may call it.
    code = "\n".join(l for l in js.splitlines()
                     if not l.lstrip().startswith("//"))
    assert "location.reload()" not in code


def test_the_admin_pages_still_have_their_menu_refresh():
    """Removing the injected button must not leave admin with no way back."""
    from pathlib import Path
    from llamanager import api_ui
    base = (Path(api_ui.__file__).parent / "templates" / "base.html").read_text()
    assert "Reload page and refresh status" in base


def test_nothing_out_votes_the_hidden_attribute():
    """Component display rules were quietly beating it: `Load older` painted
    on an empty feed, the attach button showed for engines that take no
    reference image, and the progress banner announced a job nobody started.
    One rule covers the class rather than each instance."""
    css = _partial("_composer_page.html")
    assert "[hidden] { display: none !important; }" in css


def test_an_empty_feed_leaves_masonry_so_its_message_has_room():
    """The empty message is one 4px masonry row, so the grid collapsed to
    4px, its text overflowed, and whatever followed the grid painted on top
    of it."""
    css = _partial("_composer_page.html")
    assert ".gen-grid:has(> .gen-empty:not([hidden]))" in css
    assert "display: block" in css.split(
        ".gen-grid:has(> .gen-empty:not([hidden]))")[1][:80]


def test_a_long_error_cannot_swallow_the_settings_pane():
    """The popovers open upwards from the dock, so a tall error pushes their
    top off-screen — measured at -100px on a 390x844 phone, which put the
    Model and Profile controls out of reach because the popover scrolls
    internally. Two guards: cap the error, and size the popover to the room
    that is actually left."""
    css = _partial("_composer_page.html")
    assert ".gen-error {" in css and "max-height: 25dvh" in css
    assert "overflow-y: auto" in css.split(".gen-error {")[1][:120]

    js = css  # same partial
    assert "dock.getBoundingClientRect().top" in js
    assert 'attributeFilter: ["hidden"]' in js, "must resize when opened"
    assert "Math.max(180" in js, "never shrink below a usable height"


def test_the_feed_holds_still_while_a_popover_is_open():
    """A scroll gesture over the open sheet — or a flick that began on it —
    otherwise ran the feed underneath and moved what was being read. The
    scroller is .gen-feed, not the body: the shell is a fixed 100dvh column,
    so locking the body alone would do nothing."""
    css = _partial("_composer_page.html")
    assert "body.lm-pop-open .gen-feed { overflow: hidden; }" in css
    assert "body.lm-pop-open { overflow: hidden;" in css
    # And the class has to follow both popovers, not just the settings one.
    assert 'getElementById("gen-settings")' in css
    assert 'getElementById("gen-history")' in css
    assert 'classList.toggle("lm-pop-open"' in css


# ------------------------------------------------------- the guided flow


@pytest.mark.parametrize("page", ["images.html", "videos.html",
                                  "images_public.html", "videos_public.html"])
def test_every_generation_page_carries_the_flow_partial_once(page):
    html = _partial(page)
    assert html.count('{% include "_composer_flow.html" %}') == 1, page


def test_the_flow_blocks_generation_before_the_server_would():
    """H3 with no opening frame used to travel to the server and fail there;
    the Ctrl/Cmd+Enter path bypasses submit entirely, so both are guarded."""
    js = _partial("_composer_flow.html")
    assert '"submit"' in js and "blockIfUnmet" in js
    assert "metaKey || e.ctrlKey" in js
    assert "stopImmediatePropagation" in js


def test_the_flow_derives_modes_from_caps_not_heuristics():
    js = _partial("_composer_flow.html")
    assert "caps.mode" in js and "mode_label" in js
    # And options are filtered, never rebuilt (Safari ignores hidden alone).
    assert "opt.hidden = outside" in js
    assert "opt.disabled = outside" in js


@pytest.mark.parametrize("page", ["videos.html", "videos_public.html"])
def test_the_video_pages_send_every_reference(page):
    """REF2VA advertises nine slots; sending only state.refs[0] made every
    slot past the first a silent no-op."""
    html = _partial(page)
    assert "body.images = state.refs.map(r => r.dataUrl)" in html, page


def test_engines_caps_is_the_empty_profile_answer(app):
    """The blank "(use engine defaults)" option falls back to this map, and
    the engine-wide answer offered Krea reference slots that do nothing."""
    import json
    import re
    _install_h3(app)
    html = _admin(app).get("/ui/videos").text
    m = re.search(r"enginesCaps:\s*(\{.*?\}),\n", html, re.S)
    assert m, "enginesCaps missing from the page config"
    caps = json.loads(m.group(1))
    h3 = caps.get("minimax_h3_comfy")
    assert h3 and h3["ref_images_max"] == 1, h3
