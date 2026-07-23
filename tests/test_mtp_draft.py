"""External MTP drafter (mtp-*.gguf) handling.

Some repos (e.g. unsloth's gemma-4-26B-A4B MTP pack) ship the speculative
draft heads as a separate ``mtp-*.gguf`` beside the main model (or in an
``MTP/`` subdir) instead of embedding them. Verifies the drafter is treated
as an attachment — hidden from the models page / pickers, wired in via
``--model-draft`` when a profile enables MTP — and never seeded as a main
model. Login mirrors test_smoke (rotate the bootstrap key).
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from llamanager.api_ui import _is_mtp_draft
from llamanager.config import Profile
from llamanager.server_manager import _basic_to_args, _find_mtp_draft


def _admin_client(app) -> TestClient:
    am = app.state.auth
    key = am.rotate_key(am.get_origin_by_name("bootstrap").id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303
    return client


def _seed_mtp_repo(md: Path, *, subdir: bool = False) -> Path:
    """A repo dir with a main GGUF and an mtp drafter, mirroring unsloth's
    two layouts (drafter beside the model, or in an MTP/ subfolder)."""
    repo = md / "unsloth" / "gemma-mtp-GGUF"
    repo.mkdir(parents=True, exist_ok=True)
    main = repo / "gemma-it-UD-Q4_K_XL.gguf"
    main.write_bytes(b"GGUF\x00")
    draft_dir = (repo / "MTP") if subdir else repo
    draft_dir.mkdir(exist_ok=True)
    (draft_dir / "mtp-gemma-it-BF16.gguf").write_bytes(b"GGUF\x00")
    return main


def test_is_mtp_draft():
    assert _is_mtp_draft("repo/mtp-gemma-it-BF16.gguf")
    assert _is_mtp_draft("repo/MTP/mtp-gemma-it-BF16.gguf")
    assert not _is_mtp_draft("repo/gemma-it-UD-Q4_K_XL.gguf")
    assert not _is_mtp_draft("repo/mmproj-BF16.gguf")
    # "mtp" must be a filename prefix, not just a substring of the path.
    assert not _is_mtp_draft("mtp-repo/gemma.gguf")


def test_find_mtp_draft_beside_model(app):
    main = _seed_mtp_repo(app.state.cfg.models_dir)
    found = _find_mtp_draft(main)
    assert found is not None and found.name == "mtp-gemma-it-BF16.gguf"


def test_find_mtp_draft_in_subdir(app):
    main = _seed_mtp_repo(app.state.cfg.models_dir, subdir=True)
    found = _find_mtp_draft(main)
    assert found is not None
    assert found.parent.name == "MTP"


def test_find_mtp_draft_absent(app):
    md = app.state.cfg.models_dir
    (md / "plain").mkdir(parents=True, exist_ok=True)
    main = md / "plain" / "model.gguf"
    main.write_bytes(b"GGUF\x00")
    assert _find_mtp_draft(main) is None


def test_mtp_profile_passes_model_draft(app):
    main = _seed_mtp_repo(app.state.cfg.models_dir)
    prof = Profile(name="mtp", mtp=True, mtp_n_max=4)
    args = _basic_to_args(prof, "llama", main)
    assert args["spec-type"] == "draft-mtp"
    assert args["spec-draft-n-max"] == 4
    assert args["parallel"] == 1
    assert args["model-draft"].endswith("mtp-gemma-it-BF16.gguf")


def test_mtp_profile_without_drafter_omits_model_draft(app):
    """Embedded-heads models (no mtp-*.gguf on disk) keep self-speculating
    with no --model-draft."""
    md = app.state.cfg.models_dir
    (md / "embedded").mkdir(parents=True, exist_ok=True)
    main = md / "embedded" / "model.gguf"
    main.write_bytes(b"GGUF\x00")
    prof = Profile(name="mtp", mtp=True)
    args = _basic_to_args(prof, "llama", main)
    assert args["spec-type"] == "draft-mtp"
    assert "model-draft" not in args


def test_models_page_hides_drafter_and_hints(app):
    _seed_mtp_repo(app.state.cfg.models_dir)
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        # The drafter is not its own card…
        assert 'data-model-id="unsloth/gemma-mtp-GGUF/mtp-gemma-it-BF16.gguf"' \
            not in body
        # …the main model is, with an MTP hint pill.
        assert 'data-model-id="unsloth/gemma-mtp-GGUF/gemma-it-UD-Q4_K_XL.gguf"' \
            in body
        assert re.search(r'>\s*MTP\s*</span>', body)


def test_profile_editor_mtp_controls(app):
    """Drafter-shipping models get the positive framing (drafter name +
    "drafter available" pill); the drafted-tokens input starts hidden and
    is revealed by the checkbox (hidden attr on the extra block)."""
    _seed_mtp_repo(app.state.cfg.models_dir)
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        assert "drafter available" in body
        assert "mtp-gemma-it-BF16.gguf" in body        # named in the hint
        assert "Faster generation (MTP)" in body
        # No profile has MTP on yet → the extra block renders hidden.
        assert re.search(r'class="lm-mtp-extra"\s+hidden', body)
        # Checkbox and mmproj select clear each other client-side.
        assert "select[name=mmproj]" in body
        assert "input[name=mtp]" in body


def test_plain_model_keeps_mtp_warning(app):
    """Models without a shipped drafter keep the cautionary hint — enabling
    MTP on a non-MTP model fails to launch."""
    md = app.state.cfg.models_dir
    (md / "plain").mkdir(parents=True, exist_ok=True)
    (md / "plain" / "model.gguf").write_bytes(b"GGUF\x00")
    with _admin_client(app) as client:
        body = client.get("/ui/models").text
        assert "Only works on MTP-trained models" in body
        assert "drafter available" not in body


def test_pull_seeds_mtp_profile(app):
    """A fresh pull of a repo shipping an MTP drafter (and no mmproj) seeds
    its default profile with MTP enabled."""
    _seed_mtp_repo(app.state.cfg.models_dir)
    reg = app.state.registry
    reg._maybe_create_profile("hf://unsloth/gemma-mtp-GGUF")
    m = app.state.cfg.get_model("unsloth/gemma-mtp-GGUF/gemma-it-UD-Q4_K_XL.gguf")
    assert m and m.profiles
    prof = m.profiles[m.default_profile]
    assert prof.mtp is True
    assert not prof.mmproj


def _seed_zoo(app) -> None:
    """A registry zoo: main model + attachments + shards + a diffusion dir."""
    md = app.state.cfg.models_dir
    repo = md / "unsloth" / "zoo-GGUF"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "main-Q4_K_M.gguf").write_bytes(b"GGUF\x00")
    (repo / "mmproj-BF16.gguf").write_bytes(b"GGUF\x00")
    (repo / "mtp-main-BF16.gguf").write_bytes(b"GGUF\x00")
    (repo / "big-00001-of-00002.gguf").write_bytes(b"GGUF\x00")
    (repo / "big-00002-of-00002.gguf").write_bytes(b"GGUF\x00")
    zdir = md / "Z-Fake"
    zdir.mkdir(parents=True, exist_ok=True)
    (zdir / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline"}', encoding="utf-8")
    (zdir / "x.safetensors").write_bytes(b"")


def test_admin_models_annotated(app):
    """/admin/models carries engine/family/role for every entry."""
    _seed_zoo(app)
    am = app.state.auth
    key = am.rotate_key(am.get_origin_by_name("bootstrap").id)
    with TestClient(app) as client:
        rows = client.get("/admin/models",
                          headers={"Authorization": f"Bearer {key}"}).json()
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["unsloth/zoo-GGUF/main-Q4_K_M.gguf"]["role"] == "model"
    assert by_id["unsloth/zoo-GGUF/main-Q4_K_M.gguf"]["family"] == "text"
    assert by_id["unsloth/zoo-GGUF/mmproj-BF16.gguf"]["role"] == "mmproj"
    assert by_id["unsloth/zoo-GGUF/mtp-main-BF16.gguf"]["role"] == "mtp_draft"
    assert by_id["unsloth/zoo-GGUF/big-00001-of-00002.gguf"]["role"] == "model"
    assert by_id["unsloth/zoo-GGUF/big-00002-of-00002.gguf"]["role"] == "shard"
    assert by_id["Z-Fake"]["family"] == "image"


def test_v1_models_wildcard_hides_non_llms(app):
    """A '*' origin's /v1/models only lists launchable LLMs — no diffusion
    dirs, attachments, or non-first shards."""
    _seed_zoo(app)
    am = app.state.auth
    _o, key = am.create_origin(name="wild", allowed_models=["*"])
    with TestClient(app) as client:
        ids = {d["id"] for d in client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {key}"}).json()["data"]}
    assert "unsloth/zoo-GGUF/main-Q4_K_M.gguf" in ids
    assert "unsloth/zoo-GGUF/big-00001-of-00002.gguf" in ids
    assert not any("mmproj" in i for i in ids)
    assert not any(i.rsplit("/", 1)[-1].startswith("mtp-") for i in ids)
    assert "unsloth/zoo-GGUF/big-00002-of-00002.gguf" not in ids
    assert "Z-Fake" not in ids


def test_cli_models_list_filters(app, monkeypatch, capsys):
    """CLI ``models list`` shows launchable LLMs only; ``--all`` shows the
    whole registry."""
    import json
    from types import SimpleNamespace
    from llamanager import cli as cli_mod

    rows = [
        {"model_id": "a/main.gguf", "role": "model", "family": "text"},
        {"model_id": "a/mmproj.gguf", "role": "mmproj", "family": "text"},
        {"model_id": "a/mtp-x.gguf", "role": "mtp_draft", "family": "text"},
        {"model_id": "a/b-00002-of-00002.gguf", "role": "shard",
         "family": "text"},
        {"model_id": "Z-Fake", "role": "model", "family": "image"},
        {"model_id": "legacy.gguf"},   # old daemon: no annotations → shown
    ]

    class FakeClient:
        def models_list(self):
            return rows

    monkeypatch.setattr(cli_mod, "_make_admin_client", lambda a: FakeClient())

    args = SimpleNamespace(all=False)
    assert cli_mod.cmd_models_list(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert {m["model_id"] for m in shown} == {"a/main.gguf", "legacy.gguf"}

    args = SimpleNamespace(all=True)
    assert cli_mod.cmd_models_list(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert len(shown) == len(rows)


def test_pull_prefers_vision_when_both_ship(app):
    """Repos shipping both an mmproj and an MTP drafter keep the existing
    vision-first seeding (MTP+mmproj can't run together); MTP stays a
    one-click flip in the editor."""
    md = app.state.cfg.models_dir
    main = _seed_mtp_repo(md)
    (main.parent / "mmproj-BF16.gguf").write_bytes(b"GGUF\x00")
    reg = app.state.registry
    reg._maybe_create_profile("hf://unsloth/gemma-mtp-GGUF")
    m = app.state.cfg.get_model("unsloth/gemma-mtp-GGUF/gemma-it-UD-Q4_K_XL.gguf")
    assert m and m.profiles
    prof = m.profiles[m.default_profile]
    assert prof.mmproj.endswith("mmproj-BF16.gguf")
    assert prof.mtp is False
