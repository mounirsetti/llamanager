"""The tray's MCP submenu.

The tray is where someone is standing when they want to point an agent at
this machine, so these drive the real menu and the real copy actions. The
key property under test: the tray never claims to have copied something
it did not, because a silent no-op is indistinguishable from success and
the operator would paste stale content into their MCP client.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("pystray")
pytest.importorskip("PIL")


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def status(self) -> dict[str, Any]:
        return {"state": "running", "queue_depth": 0, "in_flight_count": 0,
                "accepting_requests": True}

    def models_list(self) -> list[dict[str, Any]]:
        return [{"id": "org/model.gguf"}]


@pytest.fixture
def tray(cfg, monkeypatch):
    import threading

    from llamanager import service_ctl, tray as tray_mod

    monkeypatch.setattr(
        service_ctl, "state",
        lambda c: service_ctl.DaemonState(reachable=True, installed=True,
                                          autostart=True, detail="up"))
    monkeypatch.setattr(tray_mod.service_ctl, "state", service_ctl.state)
    app = tray_mod.TrayApp.__new__(tray_mod.TrayApp)
    app.cfg = cfg
    app.state = tray_mod.TrayState()
    app._stop = threading.Event()
    app._icon = None
    app._web_url = "http://127.0.0.1:7200/ui/"
    app._last_sig = None
    app._last_ok = None
    app._graceful = None
    app._graceful_since = 0.0
    app._client = FakeClient()
    app.notes = []
    monkeypatch.setattr(app, "_notify", lambda m: app.notes.append(m))
    app._poll_once()
    return app


def _find(menu, text: str):
    for it in menu:
        if text.lower() in str(it.text).lower():
            return it
    raise AssertionError(f"no item matching {text!r} in "
                         f"{[str(i.text) for i in menu]}")


@pytest.fixture
def clip(monkeypatch):
    """Capture what the tray puts on the clipboard."""
    from llamanager import tray as tray_mod

    box: dict[str, str] = {}
    monkeypatch.setattr(tray_mod, "_copy_to_clipboard",
                        lambda t: box.update(text=t) or None)
    return box


# --------------------------------------------------------------- menu ----

def test_menu_has_an_mcp_submenu(tray):
    item = _find(tray._build_menu(), "MCP")
    labels = [str(i.text) for i in item.submenu]
    assert any("Endpoint:" in x for x in labels)
    assert any("Connect page" in x for x in labels)
    assert any("Claude Code" in x for x in labels)
    assert any("Claude Desktop" in x for x in labels)
    assert any("Cursor" in x for x in labels)


def test_endpoint_line_shows_this_daemons_address(tray):
    tray.cfg.port = 7355
    item = _find(tray._build_menu(), "MCP")
    line = _find(item.submenu, "Endpoint:")
    assert "http://127.0.0.1:7355/mcp" in str(line.text)


def test_connect_page_opens_the_ui(tray, monkeypatch):
    from llamanager import tray as tray_mod

    opened: list[str] = []
    monkeypatch.setattr(tray_mod.webbrowser, "open", opened.append)
    tray._act_open_connect()
    assert opened == ["http://127.0.0.1:7200/ui/connect"]


# ---------------------------------------------------------- clipboard ----

def test_copy_endpoint_url(tray, clip):
    tray._act_copy_mcp_url()
    assert clip["text"] == "http://127.0.0.1:7200/mcp"
    assert "Copied" in tray.notes[-1]


def test_copy_claude_code_command_is_runnable(tray, clip):
    tray._act_copy_mcp_claude_code()
    text = clip["text"]
    assert text.startswith("claude mcp add --transport http llamanager ")
    assert "http://127.0.0.1:7200/mcp" in text
    assert 'Authorization: Bearer YOUR_KEY' in text


def test_copied_configs_are_valid_json(tray, clip):
    tray._act_copy_mcp_stdio()
    stdio = json.loads(clip["text"])
    entry = stdio["mcpServers"]["llamanager"]
    assert entry["command"] == "llamanager"
    assert entry["args"] == ["mcp-stdio"]
    # stdio resolves its own credential on this machine, so no key is
    # embedded — the tray has none to embed.
    assert "env" not in entry or not entry.get("env")

    tray._act_copy_mcp_http()
    http = json.loads(clip["text"])
    entry = http["mcpServers"]["llamanager"]
    assert entry["url"] == "http://127.0.0.1:7200/mcp"
    assert entry["headers"]["Authorization"] == "Bearer YOUR_KEY"


def test_a_failed_copy_says_so_rather_than_claiming_success(tray, monkeypatch):
    """A desktop with no clipboard tool must not look like a successful copy."""
    from llamanager import tray as tray_mod

    monkeypatch.setattr(tray_mod, "_copy_to_clipboard",
                        lambda t: "no clipboard tool found (tried wl-copy)")
    tray._act_copy_mcp_url()
    note = tray.notes[-1]
    assert "Could not copy" in note
    assert "no clipboard tool" in note


def test_clipboard_helper_reports_when_no_tool_exists(monkeypatch):
    from llamanager import tray as tray_mod

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(tray_mod.shutil, "which", lambda name: None)
    err = tray_mod._copy_to_clipboard("hello")
    assert err and "no clipboard tool" in err


def test_clipboard_helper_says_so_with_no_graphical_session(monkeypatch):
    """Tools can be installed on a box with no display — SSH, headless."""
    from llamanager import tray as tray_mod

    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(tray_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    err = tray_mod._copy_to_clipboard("hello")
    assert err and "no graphical session" in err


def test_clipboard_helper_prefers_the_tool_for_the_session(monkeypatch):
    from llamanager import tray as tray_mod

    seen: dict[str, Any] = {}

    class _Proc:
        returncode = 0

    monkeypatch.setattr(tray_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(tray_mod.subprocess, "run",
                        lambda cmd, **kw: seen.update(cmd=cmd, kw=kw) or _Proc())

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert tray_mod._copy_to_clipboard("hello") is None
    assert seen["cmd"][0] == "wl-copy"

    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert tray_mod._copy_to_clipboard("hello") is None
    assert seen["cmd"][0] == "xclip"
    assert seen["kw"]["input"] == b"hello"


def test_clipboard_helper_does_not_wait_on_the_selection_owner(monkeypatch):
    """wl-copy/xclip stay alive holding the clipboard.

    Capturing their output would read pipes that never close, hanging the
    tray's UI thread on every copy. Assert we redirect instead.
    """
    from llamanager import tray as tray_mod

    seen: dict[str, Any] = {}

    class _Proc:
        returncode = 0

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(tray_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(tray_mod.subprocess, "run",
                        lambda cmd, **kw: seen.update(kw) or _Proc())
    tray_mod._copy_to_clipboard("hello")

    assert not seen.get("capture_output"), "would block on a daemonised tool"
    assert seen["stdout"] is tray_mod.subprocess.DEVNULL
    assert seen["stderr"] is tray_mod.subprocess.DEVNULL
    assert seen["timeout"] == 5


def test_a_hanging_clipboard_tool_is_reported(monkeypatch):
    from llamanager import tray as tray_mod

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(tray_mod.shutil, "which", lambda name: "/usr/bin/" + name)

    def _hang(cmd, **kw):
        raise tray_mod.subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(tray_mod.subprocess, "run", _hang)
    err = tray_mod._copy_to_clipboard("hello")
    assert err and "did not return within 5s" in err
