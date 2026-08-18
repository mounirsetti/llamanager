"""Graceful stop / restart from the tray, and the drain switch behind them.

"Graceful" means: stop taking NEW work, let what was already accepted finish,
then bounce the service. It is deliberately not a suspend-and-resume — the
queue is an in-memory heap of live HTTP requests, so a queued request cannot
outlive the process holding its connection open. Finishing the backlog before
the restart is the only way not to lose it.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from llamanager.tray import TrayApp


# ----------------------------------------------------------- the drain switch

def _admin_key(app):
    am = app.state.auth
    return am.rotate_key(am.get_origin_by_name("bootstrap").id)


def test_drain_closes_the_door_but_keeps_the_backlog(app):
    """Pausing cancels the queue; draining must not. A graceful stop waits
    for exactly that backlog, so cancelling it would destroy the work the
    wait exists to protect."""
    key = _admin_key(app)
    client = TestClient(app)
    h = {"Authorization": f"Bearer {key}"}
    cancelled: list[str] = []
    app.state.queue.cancel_pending = lambda: cancelled.append("dropped") or 0

    r = client.post("/admin/intake/drain", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["accepting"] is False
    assert cancelled == [], "drain must not cancel the queued backlog"

    # …whereas pause still does, which is the distinction being preserved.
    client.post("/admin/intake/resume", headers=h)
    client.post("/admin/intake/pause", headers=h)
    assert cancelled == ["dropped"]


def test_status_reports_work_a_restart_would_destroy(app):
    """Downloads and installs don't pause across a restart — a pull resumes
    from byte 0 — so a graceful stop has to be able to see them."""
    key = _admin_key(app)
    r = TestClient(app).get("/admin/status",
                            headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["active_downloads"] == 0
    assert body["active_installs"] == 0


# ------------------------------------------------------- the tray's own logic

def _tray(cfg):
    t = TrayApp.__new__(TrayApp)          # no pystray, no icon, no threads
    t.cfg = cfg
    t._graceful = None
    t._graceful_since = 0.0
    t._icon = None
    t._client = object()
    t._notices: list[str] = []
    t._notify = t._notices.append
    return t


@pytest.mark.parametrize("status,expected", [
    ({}, []),
    ({"in_flight_count": 1}, ["1 running"]),
    ({"queue_depth": 3}, ["3 queued"]),
    ({"in_flight_count": 1, "queue_depth": 2}, ["1 running", "2 queued"]),
    ({"active_downloads": 1}, ["1 download"]),
    ({"active_downloads": 2, "active_installs": 1},
     ["2 downloads", "1 install"]),
])
def test_what_a_graceful_action_waits_on(status, expected):
    assert TrayApp._waiting_on(status) == expected


def test_graceful_fires_only_once_the_queue_is_empty(cfg, monkeypatch):
    """The whole point: a busy tick must not bounce the service."""
    from llamanager import service_ctl
    t = _tray(cfg)
    calls: list[str] = []
    monkeypatch.setattr(service_ctl, "restart_daemon",
                        lambda c: calls.append("restart") or (True, "ok"))
    monkeypatch.setattr(service_ctl, "stop_daemon",
                        lambda c: calls.append("stop") or (True, "ok"))
    t._safe_admin = lambda fn, label: calls.append(label) or True
    t._poll_once = lambda: None
    up = SimpleNamespace(reachable=True)

    t._graceful = "restart"
    t._graceful_tick(up, {"in_flight_count": 1, "queue_depth": 4})
    assert calls == [] and t._graceful == "restart", "fired while still busy"

    t._graceful_tick(up, {"in_flight_count": 0, "queue_depth": 2})
    assert calls == [] and t._graceful == "restart", "fired with a backlog left"

    # A download alone is enough to hold it: a restart would send it back to
    # byte 0 rather than pausing it.
    t._graceful_tick(up, {"active_downloads": 1})
    assert calls == [] and t._graceful == "restart"

    t._graceful_tick(up, {"in_flight_count": 0, "queue_depth": 0})
    assert "restart" in calls
    assert t._graceful is None


def test_graceful_reopens_intake_before_bouncing(cfg, monkeypatch):
    """The switch is persisted. Leaving it closed would bring the service
    back up silently refusing every request."""
    from llamanager import service_ctl
    t = _tray(cfg)
    order: list[str] = []
    monkeypatch.setattr(service_ctl, "restart_daemon",
                        lambda c: order.append("restart") or (True, "ok"))
    t._safe_admin = lambda fn, label: order.append(label) or True
    t._poll_once = lambda: None

    t._graceful = "restart"
    t._graceful_tick(SimpleNamespace(reachable=True), {})
    assert order.index("Resume intake") < order.index("restart")


def test_graceful_stop_on_an_already_stopped_service_does_not_start_it(cfg, monkeypatch):
    from llamanager import service_ctl
    t = _tray(cfg)
    calls: list[str] = []
    monkeypatch.setattr(service_ctl, "start_daemon",
                        lambda c: calls.append("start") or (True, "ok"))
    t._safe_admin = lambda fn, label: True
    t._poll_once = lambda: None

    t._graceful = "stop"
    t._graceful_tick(SimpleNamespace(reachable=False), {})
    assert calls == []
    assert t._graceful is None

    # A restart, though, still owes you a running service.
    t._graceful = "restart"
    t._graceful_tick(SimpleNamespace(reachable=False), {})
    assert calls == ["start"]


def test_cancelling_a_graceful_action_reopens_the_door(cfg):
    t = _tray(cfg)
    resumed: list[str] = []
    t._safe_admin = lambda fn, label: resumed.append(label) or True
    t._graceful = "stop"
    t._cancel_graceful()
    assert t._graceful is None
    assert resumed == ["Resume intake"]


def test_the_menu_actually_carries_the_new_items(cfg, monkeypatch):
    """_build_menu is where a label typo or a bad lambda would surface, and
    nothing else in this file exercises it."""
    pystray = pytest.importorskip("pystray")
    from llamanager import service_ctl, tray as tray_mod

    t = _tray(cfg)
    t.state = tray_mod.TrayState()
    t.state.update(
        daemon=service_ctl.DaemonState(reachable=True, installed=True,
                                      autostart=True, detail="up"),
        status={"state": "running", "queue_depth": 2, "in_flight_count": 1,
                "accepting_requests": True},
        models=[], last_error="")
    labels = [i.text for i in t._build_menu().items]

    def _sub(name):
        item = next(i for i in t._build_menu().items if i.text == name)
        return [s.text for s in item.submenu.items]

    service = _sub(next(l for l in labels if "Service" in str(l)))
    assert "Graceful stop (waiting: 1 running, 2 queued)" in service
    assert "Graceful restart (waiting: 1 running, 2 queued)" in service

    # Armed, the pair collapses into one cancel item naming what it waits on.
    t._graceful = "restart"
    service = _sub(next(l for l in labels if "Service" in str(l)))
    assert any(s.startswith("Cancel graceful restart") for s in service)
    assert not any(s.startswith("Graceful stop") for s in service)
