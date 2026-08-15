"""Downloads carry the family they belong to, and each page shows its own.

Before this, a download row was just "some HF pull". The models page listed
every failed/cancelled row under "Language models", so a failed MiniMax-H3
component pull (a video model, started from the diffusion setup page) read as
a broken LLM. The ``family`` column is what lets each page filter to its own
rows — and the v8 migration backfills legacy rows from the diffusion catalog
so the existing clutter moves too.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from llamanager.db import DB
from llamanager.registry import Registry


def _db(tmp_path: Path) -> DB:
    return DB(tmp_path / "state.db")


def _registry(tmp_path: Path, db: DB) -> Registry:
    from llamanager.config import Config
    cfg = Config()
    cfg.models_dir_override = tmp_path / "models"
    return Registry(cfg, db)


def _pull(reg: Registry, **kw) -> str:
    """start_pull inside a loop, with the actual fetch stubbed out.

    We only care about the row it writes; the download task itself would hit
    the network.
    """
    async def _noop(*a, **k):
        return None

    reg._run_pull = _noop  # type: ignore[method-assign]

    async def _go() -> str:
        return reg.start_pull(**kw)

    return asyncio.run(_go())


# ------------------------------------------------------------ enqueue tagging


def test_pull_defaults_to_the_text_family(tmp_path):
    db = _db(tmp_path)
    reg = _registry(tmp_path, db)
    did = _pull(reg, source="hf://org/some-llm-gguf", files=["m.gguf"])
    assert reg.get_download(did)["family"] == "text"


def test_pull_records_the_requested_family(tmp_path):
    db = _db(tmp_path)
    reg = _registry(tmp_path, db)
    did = _pull(reg, source="hf://MiniMaxAI/MiniMax-H3", files=None,
                whole_repo=True, family="video")
    assert reg.get_download(did)["family"] == "video"


def test_pull_coerces_an_unknown_family_to_text(tmp_path):
    """A bad value must not create a row that no page ever lists."""
    db = _db(tmp_path)
    reg = _registry(tmp_path, db)
    did = _pull(reg, source="hf://org/model", files=None, family="banana")
    assert reg.get_download(did)["family"] == "text"


# ------------------------------------------------------- the v8 backfill


def _insert_legacy_row(conn, ident: str, source: str) -> None:
    """A downloads row as v7 wrote them — no family column involved."""
    conn.execute(
        "INSERT INTO downloads(id, source, files_json, status, started_at)"
        " VALUES (?, ?, '{}', 'failed', ?)",
        (ident, source, time.time()),
    )


def test_migration_backfills_legacy_rows_from_the_catalog(tmp_path):
    import sqlite3

    from llamanager.db import SCHEMA_VERSIONS, migrate

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Migrate to v7 only, then write rows the way the old code did.
    for script in SCHEMA_VERSIONS[:7]:
        conn.executescript(script)
    conn.execute("PRAGMA user_version=7")
    _insert_legacy_row(conn, "h3", "hf://MiniMaxAI/MiniMax-H3")
    _insert_legacy_row(conn, "h3parts", "hf://Comfy-Org/MiniMax-H3")
    _insert_legacy_row(conn, "krea", "hf://vantagewithai/Krea-2-Turbo-GGUF")
    _insert_legacy_row(conn, "wan", "hf://Wan-AI/Wan2.2-TI2V-5B")
    _insert_legacy_row(conn, "llm", "hf://unsloth/Qwen3.6-27B-GGUF")

    migrate(conn)

    fam = {r["id"]: r["family"]
           for r in conn.execute("SELECT id, family FROM downloads")}
    assert fam["h3"] == "video"
    assert fam["h3parts"] == "video"
    assert fam["krea"] == "image"
    # The catalog names the -Diffusers re-host; a pull of the base repo is
    # still the same video model.
    assert fam["wan"] == "video"
    # Anything the catalog doesn't know was an LLM pull — the common case.
    assert fam["llm"] == "text"


# ------------------------------------------------------------ page filtering


def test_each_page_lists_only_its_own_downloads(tmp_path, monkeypatch):
    """The LLM page must not show diffusion pulls, and vice versa."""
    from types import SimpleNamespace

    from llamanager import api_ui

    db = _db(tmp_path)
    reg = _registry(tmp_path, db)
    _pull(reg, source="hf://org/llm-gguf", files=["m.gguf"])
    _pull(reg, source="hf://MiniMaxAI/MiniMax-H3", files=None, family="video")
    _pull(reg, source="hf://Comfy-Org/Krea-2", files=None, family="image")

    def _families(scope):
        return sorted(d["family"] for d in reg.list_downloads()
                      if scope is None or d["family"] in scope)

    # The filter _models_ctx applies, exercised on the same data it would see.
    assert _families(("text",)) == ["text"]
    assert _families(("image", "video")) == ["image", "video"]
    assert _families(None) == ["image", "text", "video"]

    # And the context builder really passes those scopes through.
    captured: list = []
    real_ctx = api_ui._models_ctx

    def _spy(request, download_families=None):
        captured.append(download_families)
        return {"downloads": []}

    monkeypatch.setattr(api_ui, "_models_ctx", _spy)
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(api_ui.templates, "TemplateResponse",
                        lambda *a, **k: SimpleNamespace(headers={}))
    asyncio.run(api_ui.models_view(req, _=None))
    asyncio.run(api_ui.models_list_partial(req, _=None))
    assert captured == [("text",), ("text",)]
    assert real_ctx is not _spy


# ------------------------------------------------------------ the rendered UI


def _admin_client(app):
    from fastapi.testclient import TestClient
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    key = am.rotate_key(boot.id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303
    return client


def _insert_failed(app, ident: str, source: str, family: str) -> None:
    app.state.db.execute(
        "INSERT INTO downloads(id, source, files_json, status, started_at,"
        " finished_at, error, family) VALUES (?, ?, '{}', 'failed', ?, ?,"
        " 'interrupted by daemon restart', ?)",
        (ident, source, time.time(), time.time(), family),
    )


def test_failed_pulls_render_on_their_own_page(app):
    """The reported symptom: a failed video pull listed under Language models."""
    _insert_failed(app, "llm", "hf://org/llm-gguf", "text")
    _insert_failed(app, "h3", "hf://MiniMaxAI/MiniMax-H3", "video")
    client = _admin_client(app)

    models = client.get("/ui/models").text
    assert "hf://org/llm-gguf" in models
    assert "MiniMax-H3" not in models

    diffusion = client.get("/ui/setup-diffusion").text
    assert "hf://MiniMaxAI/MiniMax-H3" in diffusion
    assert "hf://org/llm-gguf" not in diffusion


def test_the_post_download_refresh_keeps_diffusion_cards_off_the_llm_page(app):
    """/ui/models/_list is morphed into the page when a download finishes.

    It must render the same groups models.html asks for — otherwise the swap
    injects the diffusion model cards into the LLM page.
    """
    client = _admin_client(app)
    body = client.get("/ui/models/_list").text
    assert "Language models" in body
    assert "Diffusion models" not in body
