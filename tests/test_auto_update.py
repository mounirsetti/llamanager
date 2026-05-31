"""Tests for auto-update-when-idle: queue idle detection, the config
writer, the admin toggle endpoints, and the AutoUpdater decision path.

No real network or installs happen — the upstream check is pre-seeded and
the installer entry points are stubbed, so these assert the *decision*
logic (when does an update fire, when is it deferred) rather than a live
download.
"""
from __future__ import annotations

import asyncio
import time
import types
from pathlib import Path

import pytest

from llamanager.auth import Origin
from llamanager.config import Config, load_config, update_auto_update, write_default_config
from llamanager.db import DB
from llamanager.queue_mgr import QueueManager
from llamanager.server_manager import ServerManager
from llamanager import auto_update as au_mod
from llamanager.auto_update import AutoUpdater


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, **kw) -> Config:
    data = tmp_path / "lm"
    data.mkdir(exist_ok=True)
    (data / "logs").mkdir(exist_ok=True)
    (data / "models").mkdir(exist_ok=True)
    return Config(data_dir=data, **kw)


def _origin() -> Origin:
    return Origin(id=1, name="t", priority=50, allowed_models=["*"],
                  is_admin=False, created_at=0.0)


class _RecordingDB:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]


class _FakeInstaller:
    def __init__(self) -> None:
        self.started: list[str] = []

    def active_for_engine(self, engine: str):
        return None

    def start(self, engine: str, *, options=None) -> str:
        self.started.append(engine)
        return "fake-install-id"


def _fake_app(cfg: Config, *, idle: float, db=None, installer=None) -> types.SimpleNamespace:
    queue = types.SimpleNamespace(idle_seconds=lambda: idle)
    state = types.SimpleNamespace(
        cfg=cfg,
        queue=queue,
        db=db or _RecordingDB(),
        engine_installer=installer or _FakeInstaller(),
        install_states={},
        sm=types.SimpleNamespace(is_running=False),
    )
    return types.SimpleNamespace(state=state)


# ---------------------------------------------------------------------------
# idle detection
# ---------------------------------------------------------------------------

def test_idle_seconds_zero_when_pending(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        qm = QueueManager(cfg, db, ServerManager(cfg, db))

        async def go():
            # Fresh queue with nothing in it is idle.
            assert qm.idle_seconds() >= 0.0
            await qm.enqueue(origin=_origin(), model_required=None)
            # A pending request means "not idle".
            assert qm.idle_seconds() == 0.0
        asyncio.run(go())
    finally:
        db.close()


def test_idle_seconds_grows_when_drained(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        qm = QueueManager(cfg, db, ServerManager(cfg, db))
        # Simulate a queue that drained 10s ago.
        qm._last_busy_monotonic = time.monotonic() - 10.0
        assert qm.idle_seconds() >= 10.0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# config round-trip
# ---------------------------------------------------------------------------

def test_update_auto_update_roundtrip(tmp_path):
    p = tmp_path / "config.toml"
    write_default_config(p)
    update_auto_update(p, engine="llama.cpp-cuda", enabled=True)
    update_auto_update(p, engine="z_image", enabled=False)
    update_auto_update(p, idle_seconds=120, check_interval_seconds=3600)

    cfg = load_config(p)
    assert cfg.auto_update_engines == {"llama.cpp-cuda": True, "z_image": False}
    assert cfg.auto_update_idle_seconds == 120
    assert cfg.auto_update_check_interval_seconds == 3600


# ---------------------------------------------------------------------------
# admin endpoints (in-process, via the app fixture from conftest)
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_key(app) -> str:
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    assert boot is not None
    return am.rotate_key(boot.id)


@pytest.fixture
def admin_client(app, admin_key):
    from llamanager.admin_client import AdminClient
    from fastapi.testclient import TestClient
    http = TestClient(app, base_url="http://test")
    yield AdminClient(base_url="http://test", admin_key=admin_key, client=http)
    http.close()


def test_admin_auto_update_toggle_and_read(admin_client, app):
    # Default: nothing enabled.
    body = admin_client.setup_auto_update_list()
    assert body["engines"] == {}
    assert body["idle_seconds"] == app.state.cfg.auto_update_idle_seconds

    # Toggle the self-update key (valid on every platform).
    res = admin_client.setup_auto_update("llamanager", True)
    assert res["engines"]["llamanager"] is True
    # Mirrored into live config + persisted.
    assert app.state.cfg.auto_update_engines.get("llamanager") is True
    assert admin_client.setup_auto_update_list()["engines"]["llamanager"] is True

    # Settings round-trip through the endpoint.
    res = admin_client.setup_auto_update_settings(idle_seconds=60,
                                                  check_interval_seconds=120)
    assert res["idle_seconds"] == 60
    assert res["check_interval_seconds"] == 120


def test_admin_auto_update_rejects_unknown_engine(admin_client):
    from llamanager.admin_client import AdminClientError
    with pytest.raises(AdminClientError):
        admin_client.setup_auto_update("definitely-not-an-engine", True)


# ---------------------------------------------------------------------------
# AutoUpdater decision path
# ---------------------------------------------------------------------------

def test_autoupdater_fires_llama_when_idle(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path,
        auto_update_engines={"llama.cpp-cpu": True},
        auto_update_idle_seconds=300,
        auto_update_check_interval_seconds=21600,
        llama_server_binary="definitely-no-such-binary",
    )
    db = _RecordingDB()
    app = _fake_app(cfg, idle=9999.0, db=db)
    au = AutoUpdater(app)

    calls: list[tuple[str, str]] = []

    async def fake_install(state, source, backend):
        calls.append((source, backend))
        state.status = "done"
        state.installed_path = "/fake/llama-server"

    monkeypatch.setattr(au_mod, "install_variant", fake_install)

    # Pre-seed the upstream check as "update available" so _tick doesn't
    # hit the network.
    au._update_available["llama.cpp-cpu"] = True
    au._last_check["llama.cpp-cpu"] = time.monotonic()

    asyncio.run(au._tick())

    assert calls == [("llama.cpp", "cpu")]
    assert "auto_update_started" in db.kinds()
    assert "auto_update_done" in db.kinds()


def test_autoupdater_defers_when_busy(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path,
        auto_update_engines={"llama.cpp-cpu": True},
        auto_update_idle_seconds=300,
        auto_update_check_interval_seconds=21600,
        llama_server_binary="definitely-no-such-binary",
    )
    db = _RecordingDB()
    app = _fake_app(cfg, idle=5.0, db=db)   # below the 300s threshold
    au = AutoUpdater(app)

    calls: list[tuple[str, str]] = []

    async def fake_install(state, source, backend):
        calls.append((source, backend))

    monkeypatch.setattr(au_mod, "install_variant", fake_install)

    au._update_available["llama.cpp-cpu"] = True
    au._last_check["llama.cpp-cpu"] = time.monotonic()

    asyncio.run(au._tick())

    assert calls == []                              # nothing installed
    assert "auto_update_skipped_busy" in db.kinds()
    assert "auto_update_started" not in db.kinds()


def test_autoupdater_diffusion_starts_reinstall(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path,
        auto_update_engines={"z_image": True},
        auto_update_idle_seconds=300,
        auto_update_check_interval_seconds=21600,
    )
    db = _RecordingDB()
    installer = _FakeInstaller()
    app = _fake_app(cfg, idle=9999.0, db=db, installer=installer)
    au = AutoUpdater(app)

    # Pre-mark as due (the version check itself is covered separately below).
    au._update_available["z_image"] = True
    au._last_check["z_image"] = time.monotonic()

    asyncio.run(au._tick())

    assert installer.started == ["z_image"]
    assert "auto_update_started" in db.kinds()


# ---------------------------------------------------------------------------
# diffusion version check (the real signal, not a timer)
# ---------------------------------------------------------------------------

def test_diffusion_target_versions():
    from llamanager.engine_installer import diffusion_target_version, DIFFUSERS_PIN
    assert diffusion_target_version("z_image") == DIFFUSERS_PIN
    assert diffusion_target_version("hidream") == DIFFUSERS_PIN
    assert diffusion_target_version("flux2") is None


@pytest.mark.parametrize("installed,expected", [
    ("0.36.0", True),       # older than the pin -> update
    ("0.38.0", False),      # exactly the pin -> no update
    ("0.39.0.dev0", False), # newer (e.g. someone on git main) -> never downgrade
    (None, False),          # not installed -> never first-install
])
def test_diffusion_update_info(tmp_path, monkeypatch, installed, expected):
    from llamanager import engine_installer as ei
    cfg = _make_cfg(tmp_path)
    monkeypatch.setattr(ei, "installed_diffusers_version",
                        lambda _cfg, _engine: installed)
    info = ei.diffusion_update_info(cfg, "z_image")
    assert info["target"] == ei.DIFFUSERS_PIN
    assert info["has_update"] is expected


def test_autoupdater_diffusion_check_gates_on_version(tmp_path, monkeypatch):
    from llamanager import engine_installer as ei
    cfg = _make_cfg(tmp_path, auto_update_engines={"z_image": True})
    db = _RecordingDB()
    au = AutoUpdater(_fake_app(cfg, idle=9999.0, db=db))

    # Installed diffusers newer than / equal to the pin -> no update due.
    monkeypatch.setattr(ei, "installed_diffusers_version",
                        lambda _c, _e: ei.DIFFUSERS_PIN)
    assert asyncio.run(au._check_engine("z_image")) is False

    # Installed diffusers older than the pin -> update due.
    monkeypatch.setattr(ei, "installed_diffusers_version",
                        lambda _c, _e: "0.36.0")
    assert asyncio.run(au._check_engine("z_image")) is True
