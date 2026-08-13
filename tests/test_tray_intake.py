"""The tray's intake toggle.

The menu is built headless (pystray imports fine without a display; only
running an Icon needs one), so these drive the real _build_menu output and
the real action callbacks against a stub AdminClient.
"""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pystray")
pytest.importorskip("PIL")


class FakeClient:
    """Stands in for AdminClient — records the admin calls the tray makes."""

    def __init__(self, accepting: bool = True) -> None:
        self.accepting = accepting
        self.calls: list[str] = []

    def status(self) -> dict[str, Any]:
        return {"state": "running", "queue_depth": 0, "in_flight_count": 0,
                "accepting_requests": self.accepting}

    def models_list(self) -> list[dict[str, Any]]:
        return [{"id": "org/model.gguf"}]

    def intake_pause(self) -> dict[str, Any]:
        self.calls.append("pause")
        self.accepting = False
        return {"ok": True, "accepting": False}

    def intake_resume(self) -> dict[str, Any]:
        self.calls.append("resume")
        self.accepting = True
        return {"ok": True, "accepting": True}


@pytest.fixture
def tray(cfg, monkeypatch):
    """A TrayApp wired to a FakeClient and a reachable fake daemon."""
    from llamanager import service_ctl, tray as tray_mod

    monkeypatch.setattr(
        service_ctl, "state",
        lambda c: service_ctl.DaemonState(reachable=True, installed=True,
                                          autostart=True, detail="up"))
    monkeypatch.setattr(tray_mod.service_ctl, "state", service_ctl.state)
    app = tray_mod.TrayApp.__new__(tray_mod.TrayApp)   # skip AdminClient build
    app.cfg = cfg
    app.state = tray_mod.TrayState()
    app._stop = __import__("threading").Event()
    app._icon = None
    app._web_url = "http://127.0.0.1:7200/ui/"
    app._last_sig = None
    app._last_ok = None
    app._client = FakeClient()
    app.notes: list[str] = []
    monkeypatch.setattr(app, "_notify", lambda m: app.notes.append(m))
    app._poll_once()
    return app


def _items(menu) -> list:
    return list(menu)


def _find(menu, text: str):
    for it in _items(menu):
        if text.lower() in str(it.text).lower():
            return it
    raise AssertionError(f"no menu item matching {text!r} in "
                         f"{[str(i.text) for i in _items(menu)]}")


# --------------------------------------------------------------------------

def test_menu_has_intake_toggle_checked_when_accepting(tray):
    item = _find(tray._build_menu(), "Accepting requests")
    assert item.checked is True
    assert item.enabled is True


def test_toggle_pauses_then_resumes(tray):
    # Uncheck → pause.
    _find(tray._build_menu(), "Accepting requests")(tray._icon)
    assert tray._client.calls == ["pause"]
    assert any("not taking requests" in n.lower() for n in tray.notes)

    # The menu now shows it unchecked, and the header says so.
    menu = tray._build_menu()
    assert _find(menu, "Accepting requests").checked is False
    assert "NOT taking requests" in str(_items(menu)[0].text)

    # Check again → resume.
    _find(menu, "Accepting requests")(tray._icon)
    assert tray._client.calls == ["pause", "resume"]
    assert _find(tray._build_menu(), "Accepting requests").checked is True
    assert "NOT taking requests" not in str(_items(tray._build_menu())[0].text)


def test_toggle_reflects_a_flip_made_elsewhere(tray):
    """Flipped from the top bar or CLI, the tray's next poll picks it up."""
    tray._client.accepting = False
    tray._poll_once()
    assert _find(tray._build_menu(), "Accepting requests").checked is False
    # ...and the poller notices, so the menu actually gets rebuilt.
    sig_closed = tray._display_signature(tray.state.snapshot())
    tray._client.accepting = True
    tray._poll_once()
    assert tray._display_signature(tray.state.snapshot()) != sig_closed


def test_older_daemon_without_the_key_reads_as_accepting(tray):
    """A status payload predating the switch must not read as 'closed'."""
    tray._client.status = lambda: {"state": "running", "queue_depth": 0}
    tray._poll_once()
    assert _find(tray._build_menu(), "Accepting requests").checked is True


def test_failed_call_does_not_claim_success(tray, monkeypatch):
    from llamanager.admin_client import AdminClientError

    def boom():
        raise AdminClientError("daemon went away")

    tray._client.intake_pause = boom
    _find(tray._build_menu(), "Accepting requests")(tray._icon)
    assert any("failed" in n.lower() for n in tray.notes)
    assert not any("not taking requests" in n.lower() for n in tray.notes)


def test_toggle_disabled_when_daemon_unreachable(tray, monkeypatch):
    from llamanager import service_ctl, tray as tray_mod
    monkeypatch.setattr(
        tray_mod.service_ctl, "state",
        lambda c: service_ctl.DaemonState(reachable=False, installed=True,
                                          autostart=True, detail="down"))
    tray._poll_once()
    assert _find(tray._build_menu(), "Accepting requests").enabled is False


def test_no_admin_key_reports_instead_of_crashing(tray):
    tray._client = None
    _find(tray._build_menu(), "Accepting requests")(tray._icon)
    assert any("admin key" in n.lower() for n in tray.notes)


def test_no_admin_key_shows_why_instead_of_a_dead_switch(tray):
    """A keyless tray can't drive the daemon at all — the item says so and is
    greyed out rather than looking clickable and doing nothing."""
    tray._client = None
    item = _find(tray._build_menu(), "Accepting requests")
    assert item.enabled is False
    assert "needs an admin key" in str(item.text)
