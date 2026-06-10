"""Smoke tests: import surface, DB migrations, config defaults, auth."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_imports():
    import llamanager
    import llamanager.app  # noqa: F401
    import llamanager.api_v1  # noqa: F401
    import llamanager.api_admin  # noqa: F401
    import llamanager.api_ui  # noqa: F401
    import llamanager.queue_mgr  # noqa: F401
    import llamanager.server_manager  # noqa: F401
    import llamanager.supervisor  # noqa: F401
    import llamanager.registry  # noqa: F401
    import llamanager.installer  # noqa: F401
    assert llamanager.__version__


def test_config_defaults_and_paths(tmp_path: Path):
    from llamanager.config import load_config
    c = load_config(tmp_path / "missing.toml")
    assert c.port == 7200
    assert c.llama_server_port == 7201
    assert c.models == {}  # no bundled models
    # Bundled defaults seed per-engine default_args.
    assert "llama" in c.default_args
    assert c.models_dir.is_dir()


def test_db_migrations(tmp_path: Path):
    from llamanager.db import DB
    db = DB(tmp_path / "state.db")
    # Re-open to ensure migrate is idempotent.
    db2 = DB(tmp_path / "state.db")
    db.close(); db2.close()


def test_auth_bootstrap_and_verify(tmp_path: Path):
    import asyncio
    from llamanager.auth import AuthManager, load_or_create_lookup_secret
    from llamanager.db import DB

    db = DB(tmp_path / "state.db")
    am = AuthManager(db, lookup_secret=load_or_create_lookup_secret(tmp_path))
    key = am.ensure_bootstrap()
    assert key and key.startswith("lm_")
    # Second call should return None (origin already exists).
    assert am.ensure_bootstrap() is None
    origin = asyncio.run(am.verify(key))
    assert origin and origin.is_admin and origin.name == "bootstrap"
    # Wrong key fails.
    assert asyncio.run(am.verify("lm_garbage")) is None


def test_create_origin_and_priorities(tmp_path: Path):
    import asyncio
    from llamanager.auth import AuthManager, load_or_create_lookup_secret
    from llamanager.db import DB
    db = DB(tmp_path / "state.db")
    am = AuthManager(db,
                     lookup_secret=load_or_create_lookup_secret(tmp_path),
                     default_priority=42)
    am.ensure_bootstrap()
    o, key = am.create_origin(name="alice", allowed_models=["*"])
    assert o.priority == 42
    verified = asyncio.run(am.verify(key))
    assert verified is not None and verified.id == o.id
    am.update_origin(o.id, priority=99)
    assert am.get_origin(o.id).priority == 99  # type: ignore[union-attr]


def test_registry_disk_safety(tmp_path: Path):
    from llamanager.config import Config
    from llamanager.db import DB
    from llamanager.registry import Registry

    cfg = Config(data_dir=tmp_path)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.max_disk_gb = 0  # disable cap
    db = DB(cfg.db_path)
    reg = Registry(cfg, db)
    assert reg.list() == []
    # Pretend we have a model on disk.
    p = cfg.models_dir / "fake/repo/model.gguf"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x" * 1024)
    found = reg.list()
    assert len(found) == 1
    assert found[0].model_id == "fake/repo/model.gguf"


def test_health_endpoint(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["daemon_ok"] is True
    assert body["llama_server_ok"] is False
    assert body["state"] == "stopped"


def test_v1_requires_auth(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 401


def test_admin_requires_admin(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/admin/status", headers={"Authorization": "Bearer lm_x"})
    assert r.status_code in (401, 403)


def test_admin_status_with_bootstrap(app):
    """Pull the bootstrap origin's hash, mint a fresh key for it via
    rotate-key (we can't recover the original because we suppressed the
    print), and verify /admin/status responds."""
    from fastapi.testclient import TestClient
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    assert boot
    new_key = am.rotate_key(boot.id)
    client = TestClient(app)
    r = client.get("/admin/status", headers={"Authorization": f"Bearer {new_key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "stopped"
    assert body["queue_depth"] == 0


def _seed_fake_model(cfg) -> None:
    """Create a zero-byte test/model.gguf under cfg.models_dir so resolve_spec
    doesn't reject the path."""
    target = cfg.models_dir / "test" / "model.gguf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")


def test_resolve_spec_scenario_1_nothing_specified(app):
    """No model, no profile -> default_model + its default profile."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    spec = resolve_spec(cfg)
    assert spec.model_id == "test/model.gguf"
    assert spec.profile_name == "test"  # from [model_defaults]
    assert spec.extra_args.get("ctx-size") == 1024


def test_resolve_spec_scenario_2_model_only(app):
    """Model only -> that model + its default profile (if any)."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    spec = resolve_spec(cfg, model="test/model.gguf")
    assert spec.model_id == "test/model.gguf"
    assert spec.profile_name == "test"
    assert spec.extra_args.get("ctx-size") == 1024


def test_resolve_spec_scenario_2_model_only_no_default(app):
    """Model only, with no per-model default profile -> the model loads
    with engine defaults + the request's args. Per-engine defaults seeded
    by the bundled config provide the baseline."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    m = cfg.get_model("test/model.gguf")
    if m:
        m.default_profile = ""
    cfg.default_args.pop("llama", None)
    spec = resolve_spec(cfg, model="test/model.gguf", args={"temp": 0.1})
    assert spec.profile_name is None
    assert spec.extra_args == {"temp": 0.1}


def test_resolve_spec_scenario_3_model_and_matching_profile(app):
    """Model + bound profile -> use the pair."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    spec = resolve_spec(cfg, model="test/model.gguf", profile="test",
                        args={"temp": 0.4})
    assert spec.profile_name == "test"
    assert spec.extra_args.get("temp") == 0.4
    assert spec.extra_args.get("ctx-size") == 1024  # from profile


def test_resolve_spec_profile_alone_uses_default_model(app):
    """Profile alone falls back to cfg.default_model + that profile."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    spec = resolve_spec(cfg, profile="test")
    assert spec.model_id == "test/model.gguf"
    assert spec.profile_name == "test"


def test_resolve_spec_unknown_profile_for_model(app):
    """A profile name that doesn't exist under the chosen model is rejected."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_spec(cfg, model="test/model.gguf", profile="nonexistent")


def test_resolve_spec_default_args_applied(app):
    """cfg.default_args[engine] provides a minimum baseline that profile
    and request args layer on top of."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    cfg.default_args["llama"] = {"top-p": 0.92, "temp": 0.5}
    spec = resolve_spec(cfg, model="test/model.gguf", args={"temp": 0.1})
    # default_args supplies top-p; request overrides temp.
    assert spec.extra_args.get("top-p") == 0.92
    assert spec.extra_args.get("temp") == 0.1
    # ctx-size still comes from the profile (translated from ctx_size).
    assert spec.extra_args.get("ctx-size") == 1024


def test_load_config_migrates_legacy_layout(tmp_path: Path):
    """Legacy [profiles.X] with model= flat layout migrates into nested
    [models."x/m.gguf".profiles.fast] plus a per-model default_profile."""
    from llamanager.config import load_config
    p = tmp_path / "config.toml"
    p.write_text(f'''
[server]
data_dir = "{tmp_path.as_posix()}"

[defaults]
model = "x/m.gguf"
profile = "fast"

[model_defaults]
"x/m.gguf" = "fast"

[profiles.fast]
model = "x/m.gguf"
[profiles.fast.args]
temp = 0.3
ctx-size = 2048

[profiles.house]
model = ""
[profiles.house.args]
top-p = 0.9
''', encoding="utf-8")
    cfg = load_config(p)
    # Model-bound profile migrated into nested layout.
    m = cfg.get_model("x/m.gguf")
    assert m is not None
    assert "fast" in m.profiles
    fast = m.profiles["fast"]
    assert fast.args.get("temp") == 0.3
    assert fast.ctx_size == 2048
    assert m.default_profile == "fast"
    # Empty-model "house" profile merged into engine defaults.
    assert cfg.default_args.get("llama", {}).get("top-p") == 0.9
    # Reloading is a no-op (migration idempotent).
    cfg2 = load_config(p)
    m2 = cfg2.get_model("x/m.gguf")
    assert m2 is not None and m2.default_profile == "fast"


def test_v1_rejects_legacy_profile_shorthand(app):
    """X-Llamanager-Model: profile:foo returns 400 with a migration hint."""
    from fastapi.testclient import TestClient
    am = app.state.auth
    new_key = am.rotate_key(am.get_origin_by_name("bootstrap").id)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={
            "Authorization": f"Bearer {new_key}",
            "X-Llamanager-Model": "profile:fast",
        },
    )
    assert r.status_code == 400
    assert "no longer supported" in r.json()["detail"]


def test_v1_accepts_profile_header_without_model(app):
    """X-Llamanager-Profile without X-Llamanager-Model is allowed; the
    dispatcher will resolve it against the default model. Tested at the
    request-parse layer so we don't have to spin up a real upstream."""
    from llamanager.api_v1 import _model_required

    class _StubRequest:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    class _StubOrigin:
        name = "test"
        allowed_models = ["*"]

    req = _StubRequest({"x-llamanager-profile": "test"})
    model, profile = _model_required(req, _StubOrigin())  # type: ignore[arg-type]
    assert model is None
    assert profile == "test"


class _StubRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _origin(allowed):
    from llamanager.auth import Origin
    return Origin(id=1, name="test", priority=50, allowed_models=allowed,
                  is_admin=False, created_at=0.0)


def test_resolve_request_model_body_routing(app):
    """Stock OpenAI clients (Continue, openai SDK) put the model in the body.
    A known + allowed body model routes; the header still wins over the body."""
    from llamanager.api_v1 import _resolve_request_model
    cfg, sm = app.state.cfg, app.state.sm
    o = _origin(["*"])

    # Known + allowed body model → routed, no fallback.
    m, p, fb = _resolve_request_model(
        _StubRequest({}), o, {"model": "test/model.gguf"}, cfg, sm)
    assert (m, p, fb) == ("test/model.gguf", None, None)

    # Header wins over body.
    m, _, _ = _resolve_request_model(
        _StubRequest({"x-llamanager-model": "test/model.gguf"}),
        o, {"model": "ignored"}, cfg, sm)
    assert m == "test/model.gguf"

    # Whitespace tolerated.
    m, _, _ = _resolve_request_model(
        _StubRequest({}), o, {"model": "  test/model.gguf  "}, cfg, sm)
    assert m == "test/model.gguf"


def test_resolve_request_model_falls_back_to_default(app):
    """Unknown, placeholder, or not-permitted models degrade to the default
    (model_id None) with a reason — never a 403."""
    from llamanager.api_v1 import _resolve_request_model
    cfg, sm = app.state.cfg, app.state.sm

    # No model named → default, no fallback reason.
    m, p, fb = _resolve_request_model(_StubRequest({}), _origin(["*"]), {}, cfg, sm)
    assert (m, p, fb) == (None, None, None)

    # Unknown / placeholder ids → default + "not installed" reason.
    for bad in ("gpt-4", "any"):
        m, _, fb = _resolve_request_model(
            _StubRequest({}), _origin(["*"]), {"model": bad}, cfg, sm)
        assert m is None and fb and "not installed" in fb

    # "default" / empty / non-string → default, silently.
    for val in ("default", "", 123):
        m, _, fb = _resolve_request_model(
            _StubRequest({}), _origin(["*"]), {"model": val}, cfg, sm)
        assert (m, fb) == (None, None)

    # Known but NOT permitted for this origin → default + "not permitted".
    m, _, fb = _resolve_request_model(
        _StubRequest({}), _origin(["other/model.gguf"]),
        {"model": "test/model.gguf"}, cfg, sm)
    assert m is None and fb and "not permitted" in fb


def test_model_allowed_predicate():
    from llamanager.api_v1 import _model_allowed
    assert _model_allowed(_origin(["*"]), "anything") is True
    assert _model_allowed(_origin(["default"]), "default") is True
    assert _model_allowed(_origin(["default"]), "a/b.gguf") is False
    assert _model_allowed(_origin(["a/b.gguf"]), "a/b.gguf") is True
    assert _model_allowed(_origin(["a/b.gguf"]), "c/d.gguf") is False


def test_parse_allowed_models():
    from llamanager.api_ui import _parse_allowed_models
    assert _parse_allowed_models(True, []) == ["*"]
    assert _parse_allowed_models(True, ["a/b.gguf"]) == ["*"]  # checkbox wins
    assert _parse_allowed_models(False, []) == ["*"]           # empty → allow all
    assert _parse_allowed_models(False, ["a/b.gguf", " c/d.gguf "]) == \
        ["a/b.gguf", "c/d.gguf"]


def test_reasoning_budget_form_to_launch_arg():
    """Reasoning budget flows form → Profile → --reasoning-budget, survives a
    TOML round-trip, and the blank/0 edges behave."""
    from pathlib import Path
    from llamanager.api_ui import _build_profile_from_form
    from llamanager.server_manager import _basic_to_args
    from llamanager.config import _profile_to_tomlkit, _parse_profile
    import tomlkit

    def build(rb):
        return _build_profile_from_form(
            "p", mmproj="", ctx_size="", vram_limit_gb="", vram_unlimited="on",
            ram_spill_policy="default", ram_spill_limit_gb="", thinking="",
            args_json="{}", kv_cache_type="", reasoning_budget=rb)

    prof = build("2000")
    assert prof.reasoning_budget == 2000
    assert _basic_to_args(prof, "llama", Path("/x.gguf"))["reasoning-budget"] == 2000
    # blank → unbounded (no arg emitted)
    assert build("").reasoning_budget is None
    assert "reasoning-budget" not in _basic_to_args(build(""), "llama", Path("/x.gguf"))
    # 0 → no thinking (arg emitted as 0)
    assert build("0").reasoning_budget == 0
    assert _basic_to_args(build("0"), "llama", Path("/x.gguf"))["reasoning-budget"] == 0
    # negative rejected
    import pytest
    with pytest.raises(ValueError):
        build("-5")
    # mlx ignores it (llama-only basic arg)
    assert "reasoning-budget" not in _basic_to_args(prof, "mlx", Path("/x.gguf"))
    # TOML round-trip
    tbl = _profile_to_tomlkit(prof)
    reparsed = _parse_profile("p", tomlkit.loads(tomlkit.dumps({"x": tbl}))["x"])
    assert reparsed.reasoning_budget == 2000


def test_parallel_form_to_launch_arg():
    """Parallel slots flow form → Profile → --parallel, survive a TOML
    round-trip, and the blank/invalid edges behave."""
    from pathlib import Path
    from llamanager.api_ui import _build_profile_from_form
    from llamanager.server_manager import _basic_to_args
    from llamanager.config import _profile_to_tomlkit, _parse_profile
    import tomlkit

    def build(par):
        return _build_profile_from_form(
            "p", mmproj="", ctx_size="", vram_limit_gb="", vram_unlimited="on",
            ram_spill_policy="default", ram_spill_limit_gb="", thinking="",
            args_json="{}", kv_cache_type="", reasoning_budget="", parallel=par)

    prof = build("1")
    assert prof.parallel == 1
    assert _basic_to_args(prof, "llama", Path("/x.gguf"))["parallel"] == 1
    # blank → auto (no arg emitted)
    assert build("").parallel is None
    assert "parallel" not in _basic_to_args(build(""), "llama", Path("/x.gguf"))
    # < 1 rejected
    import pytest
    with pytest.raises(ValueError):
        build("0")
    # mlx ignores it (llama-only basic arg)
    assert "parallel" not in _basic_to_args(prof, "mlx", Path("/x.gguf"))
    # TOML round-trip
    tbl = _profile_to_tomlkit(prof)
    reparsed = _parse_profile("p", tomlkit.loads(tomlkit.dumps({"x": tbl}))["x"])
    assert reparsed.parallel == 1


def test_recommended_reasoning_budget():
    from llamanager.api_ui import _recommended_reasoning_budget
    assert _recommended_reasoning_budget(None) is None
    assert _recommended_reasoning_budget(0) is None
    # ~20s of thinking at the measured rate, rounded to 250, floor 500
    assert _recommended_reasoning_budget(60.0) == 1250   # 1200 → 1250
    assert _recommended_reasoning_budget(150.0) == 3000
    assert _recommended_reasoning_budget(5.0) == 500     # floor


def test_priority_queue_ordering(app):
    """Two queued requests at different priorities: the higher one comes out first."""
    import asyncio
    from llamanager.auth import AuthManager
    from llamanager.queue_mgr import QueueManager

    am: AuthManager = app.state.auth
    qm: QueueManager = app.state.queue
    low, _ = am.create_origin(name="low", priority=10)
    high, _ = am.create_origin(name="high", priority=90)

    async def run() -> tuple[str, str]:
        a = await qm.enqueue(origin=low, model_required=None)
        b = await qm.enqueue(origin=high, model_required=None)
        # Inspect heap order without starting the dispatcher.
        ordered = sorted(qm._heap)
        first = ordered[0][1]
        second = ordered[1][1]
        return first.origin.name, second.origin.name

    first, second = asyncio.run(run())
    assert first == "high"
    assert second == "low"


def test_installer_writes_files(tmp_path: Path, monkeypatch):
    """Just check the templating; we redirect HOME to tmp so we don't
    pollute the real user dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from llamanager.installer import (
        install_launchd, install_systemd, install_windows_task,
    )
    plist = install_launchd(label="com.llamanager.test", port=7200,
                            binary="/bin/echo")
    assert plist.exists()
    assert "com.llamanager.test" in plist.read_text(encoding="utf-8")
    unit = install_systemd(unit_name="llamanager-test.service", port=7200,
                           binary="/bin/echo")
    assert unit.exists()
    assert "ExecStart=/bin/echo" in unit.read_text(encoding="utf-8")
    xml = install_windows_task(task_name="llamanager-test", port=7200,
                               binary="C:\\bin\\llamanager.exe")
    assert xml.exists()
    raw = xml.read_bytes()
    assert raw.startswith(b"\xff\xfe")  # UTF-16 LE BOM
    text = raw[2:].decode("utf-16-le")
    assert "llamanager.exe" in text
    assert "<LogonTrigger>" in text
    assert "serve --port 7200" in text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
def test_windows_service_module_importable():
    """If pywin32 is available on this Windows host, the service
    wrapper should import cleanly and expose its expected attrs."""
    pytest.importorskip("win32serviceutil")
    from llamanager import win_service
    cls = win_service.LlamanagerService
    assert cls._svc_name_ == "llamanager"
    assert "llama-server" in cls._svc_display_name_
    # main() with no args would try to dispatch to the SCM, which would
    # fail outside the service host. We don't exercise that path here;
    # we just check the symbol exists.
    assert callable(win_service.main)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
def test_cli_remove_windows_service_without_pywin32(monkeypatch, capsys):
    """If pywin32 is missing, the CLI should print a helpful error
    and return a non-zero exit code rather than crashing."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "win32serviceutil":
            raise ImportError("simulated missing pywin32")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from llamanager.cli import _run_win_service_module
    rc = _run_win_service_module(["stop"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "pywin32 is not installed" in err
