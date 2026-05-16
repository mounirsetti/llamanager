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
    assert c.default_profile == ""
    assert c.profiles == {}  # no bundled profiles
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
    """Model only, with no per-model default and no global default
    -> the model loads with the request's args only."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    cfg.model_default_profiles.pop("test/model.gguf", None)
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


def test_resolve_spec_rejects_profile_without_model(app):
    """Profile alone is no longer accepted."""
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    with pytest.raises(ValueError, match="profile requires a model"):
        resolve_spec(cfg, profile="test")


def test_resolve_spec_rejects_mismatched_profile(app):
    """Profile bound to model A cannot be used with model B."""
    from llamanager.config import Profile
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    # Add a second fake model + a profile bound to it.
    other = cfg.models_dir / "other" / "model.gguf"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"")
    cfg.profiles["other-only"] = Profile(name="other-only",
                                          model="other/model.gguf",
                                          mmproj="", args={"temp": 0.2})
    with pytest.raises(ValueError, match="bound to"):
        resolve_spec(cfg, model="test/model.gguf", profile="other-only")


def test_resolve_spec_global_profile_applies_to_any_model(app):
    """A profile with model='' is a valid args-only fallback for any model."""
    from llamanager.config import Profile
    from llamanager.server_manager import resolve_spec
    cfg = app.state.cfg
    _seed_fake_model(cfg)
    cfg.profiles["balanced"] = Profile(name="balanced", model="",
                                        mmproj="", args={"top-p": 0.95})
    spec = resolve_spec(cfg, model="test/model.gguf", profile="balanced")
    assert spec.profile_name == "balanced"
    assert spec.extra_args.get("top-p") == 0.95


def test_load_config_migrates_legacy_default_profile(tmp_path: Path):
    """A legacy [defaults] profile that names a model-bound profile becomes
    a [model_defaults] entry. default_profile is cleared in memory."""
    from llamanager.config import load_config
    p = tmp_path / "config.toml"
    p.write_text(f'''
[server]
data_dir = "{tmp_path.as_posix()}"

[defaults]
model = "x/m.gguf"
profile = "fast"

[profiles.fast]
model = "x/m.gguf"
[profiles.fast.args]
temp = 0.3
''', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.default_profile == ""
    assert cfg.model_default_profiles == {"x/m.gguf": "fast"}
    # Re-loading should be a no-op (migration is idempotent).
    cfg2 = load_config(p)
    assert cfg2.default_profile == ""
    assert cfg2.model_default_profiles == {"x/m.gguf": "fast"}


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


def test_v1_rejects_profile_header_without_model(app):
    """X-Llamanager-Profile without X-Llamanager-Model returns 400."""
    from fastapi.testclient import TestClient
    am = app.state.auth
    new_key = am.rotate_key(am.get_origin_by_name("bootstrap").id)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={
            "Authorization": f"Bearer {new_key}",
            "X-Llamanager-Profile": "fast",
        },
    )
    assert r.status_code == 400
    assert "requires X-Llamanager-Model" in r.json()["detail"]


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
