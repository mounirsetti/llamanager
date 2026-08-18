"""The Diffusion engines page as a LoRA manager for the Comfy Krea model.

Downloading a LoRA was always one click; everything else about that folder
was invisible. These cover the three things the page now has to get right:
list what is actually installed, say which files change the *graph* (and so
need reference images), and remove one without a shell.
"""
import re

from fastapi.testclient import TestClient

from llamanager.api_ui import KREA_COMFY_MODEL_ID


def _admin_client(app):
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    key = am.rotate_key(boot.id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303
    return client


KREA_MODAL = ("/ui/diffusion/model", {"id": KREA_COMFY_MODEL_ID})


def _csrf(client) -> str:
    m = re.search(r'name="csrf_token" value="([^"]+)"',
                  client.get("/ui/diffusion").text)
    assert m, "no csrf token on the diffusion page"
    return m.group(1)


def _modal(client) -> str:
    """The Krea model's detail modal — where its LoRA folder is managed."""
    r = client.get(KREA_MODAL[0], params=KREA_MODAL[1])
    assert r.status_code == 200, r.text[:2000]
    return r.text


def _delete(client, filename):
    return client.post("/ui/setup-diffusion/krea-lora/delete",
                       data={"filename": filename, "csrf_token": _csrf(client)},
                       follow_redirects=False)


def _loras_dir(app):
    d = app.state.cfg.models_dir / KREA_COMFY_MODEL_ID / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_page_lists_installed_loras_and_flags_the_editors(app):
    d = _loras_dir(app)
    (d / "krea2_darkbrush.safetensors").write_bytes(b"x" * 4096)
    (d / "krea2_identity_edit_v1_2.safetensors").write_bytes(b"x" * 8192)
    (d / "krea2_style_reference.safetensors").write_bytes(b"x" * 2048)
    html = _modal(_admin_client(app))

    assert "krea2_darkbrush.safetensors" in html
    assert "krea2_identity_edit_v1_2.safetensors" in html
    # The distinction that matters: which files demand a reference image.
    assert "needs 1-2 reference images" in html      # identity edit, pack A
    assert "needs 1-3 reference images" in html      # style reference, pack B
    assert "no reference image" in html              # the plain style LoRA


def test_page_offers_a_form_for_any_other_hub_lora(app):
    """The curated rows and Krea's own collection do not cover the Hub."""
    _loras_dir(app)
    html = _modal(_admin_client(app))
    assert 'name="target_dir" value="Krea-2-Turbo-Comfy/loras"' in html
    assert "Add a LoRA from Hugging Face" in html


def test_deleting_a_lora_removes_the_file(app):
    d = _loras_dir(app)
    doomed = d / "krea2_darkbrush.safetensors"
    doomed.write_bytes(b"x")
    keep = d / "krea2_style_reference.safetensors"
    keep.write_bytes(b"x")
    r = _delete(_admin_client(app), "krea2_darkbrush.safetensors")
    assert r.status_code == 303, r.text
    # Back to the model it was deleted from, not the top of the page.
    assert r.headers["location"].startswith("/ui/diffusion?open=model")
    assert not doomed.exists()
    assert keep.exists()


def test_delete_cannot_escape_the_loras_folder(app):
    """The filename is operator input; it must stay one path component."""
    d = _loras_dir(app)
    outside = d.parent.parent / "victim.safetensors"
    outside.write_bytes(b"x")
    r = _delete(_admin_client(app), "../../victim.safetensors")
    assert r.status_code == 400
    assert outside.exists()


def test_delete_refuses_a_non_lora_file(app):
    """Same folder, but this endpoint is not a general file remover."""
    d = _loras_dir(app)
    other = d / "notes.txt"
    other.write_bytes(b"x")
    r = _delete(_admin_client(app), "notes.txt")
    assert r.status_code == 400
    assert other.exists()
