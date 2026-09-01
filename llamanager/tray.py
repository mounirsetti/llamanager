"""System-tray / menu-bar app for llamanager (cross-platform).

This is a **thin client**: it does not run the daemon, it talks to an
already-running one over the ``/admin/*`` control plane
(:class:`~llamanager.admin_client.AdminClient`) and reflects/controls it
from the desktop session. The daemon itself is an OS-managed background
service (see :mod:`llamanager.service_ctl`) so it can be always-on, even
before login; the tray just shows up when you sign in and disappears
when you sign out — the daemon keeps serving either way.

Run it directly with ``llamanager tray``. On macOS the tray event loop
must own the main thread, so the status poller runs on a background
thread and the icon loop blocks in :func:`main`.

Optional dependency: install with ``pip install 'llamanager[tray]'``
(pulls in ``pystray`` + ``pillow``).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import service_ctl
from .admin_client import AdminClient, AdminClientError, resolve_base_url
from .config import Config, ConfigError, expand, load_config

log = logging.getLogger("llamanager.tray")

# How often the background poller refreshes daemon/LLM state.
_POLL_SECONDS = 4.0


def _require_pystray():
    """Import pystray + Pillow lazily with an actionable error."""
    try:
        import pystray  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on optional extra
        msg = ("error: the tray needs pystray + Pillow.\n"
               "  Install with: pip install 'llamanager[tray]'\n"
               f"  (import failed: {e})")
        if sys.platform.startswith("linux"):
            msg += ("\n  On Linux the icon also needs PyGObject + an "
                    "AppIndicator typelib:\n"
                    "    sudo apt install python3-gi "
                    "gir1.2-ayatanaappindicator3-0.1\n"
                    "  and a venv built with --system-site-packages (or "
                    "include-system-site-packages = true in pyvenv.cfg).")
        log.error("%s", msg)
        raise SystemExit(msg)
    return pystray, Image


def _assets_dir() -> Path:
    # app.py serves assets from the wheel root next to the package.
    return Path(__file__).resolve().parent.parent / "assets"


def _load_icon_image(Image, *, ok: bool):
    """Pick a tray image. Uses the shipped PNG; falls back to a plain
    generated dot tinted by reachability so the icon still renders if the
    asset is missing."""
    name = "icon-light-512.png" if ok else "icon-dark-512.png"
    candidate = _assets_dir() / name
    try:
        return Image.open(candidate)
    except Exception:
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        try:
            from PIL import ImageDraw  # type: ignore[import-not-found]
            color = (46, 160, 67, 255) if ok else (128, 128, 128, 255)
            ImageDraw.Draw(img).ellipse((8, 8, 56, 56), fill=color)
        except Exception:
            pass
        return img


class TrayState:
    """Shared, lock-guarded snapshot the poller writes and the menu reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.daemon = service_ctl.DaemonState(reachable=False, installed=None,
                                              autostart=None)
        self.status: dict[str, Any] = {}
        self.models: list[dict[str, Any]] = []
        self.last_error: str = ""

    def update(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "daemon": self.daemon,
                "status": dict(self.status),
                "models": list(self.models),
                "last_error": self.last_error,
            }


def _model_id(m: dict[str, Any]) -> str:
    """Models endpoint dicts have varied id keys across versions; accept
    whichever is present."""
    return str(m.get("id") or m.get("model_id") or m.get("name") or "")


def _open_path(path: Path) -> None:
    """Open a folder in the OS file manager."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        log.warning("could not open %s: %s", path, e)


def _copy_to_clipboard(text: str) -> str | None:
    """Put ``text`` on the system clipboard. Returns an error message or None.

    The tray's whole value for MCP is handing over a config line without
    making the operator go find the web UI, and that needs a clipboard.
    There is no Python clipboard in llamanager's dependencies, so this
    shells out to whatever the desktop provides. If nothing does, say so —
    a silent no-op looks exactly like a successful copy, and the operator
    would paste stale content into their MCP client.

    Two traps, both learned the hard way:

    * ``wl-copy`` (and ``xclip``) **stay alive** holding the selection —
      that is how X11/Wayland clipboards work, the owner serves the data.
      Capturing their output therefore waits on pipes that never close, so
      output goes to /dev/null and we only wait for the fork to return.
    * Tools can be installed on a machine with no display at all (an SSH
      session, a headless service). Pick by the session actually present
      rather than by which binary exists.
    """
    if sys.platform == "win32":
        candidates = [["clip"]]
    elif sys.platform == "darwin":
        candidates = [["pbcopy"]]
    elif os.environ.get("WAYLAND_DISPLAY"):
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"],
                      ["xsel", "--clipboard", "--input"]]
    elif os.environ.get("DISPLAY"):
        candidates = [["xclip", "-selection", "clipboard"],
                      ["xsel", "--clipboard", "--input"], ["wl-copy"]]
    else:
        return ("no graphical session (no DISPLAY or WAYLAND_DISPLAY), so "
                "there is no clipboard to copy to")

    tried = []
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            tried.append(cmd[0])
            continue
        try:
            proc = subprocess.run(
                cmd, input=text.encode("utf-8"),
                # Not capture_output: these tools keep running to serve the
                # selection, and reading their pipes to EOF would block.
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return f"{cmd[0]} did not return within 5s"
        except Exception as e:  # noqa: BLE001 — reported, not raised
            return f"{cmd[0]} failed: {e}"
        if proc.returncode == 0:
            return None
        return f"{cmd[0]} exited {proc.returncode}"
    return ("no clipboard tool found (tried " + ", ".join(tried) +
            ") — install one, or copy from the Connect page in the web UI")


def _autorun_label() -> str:
    """Cheap, file-based detection of the current autorun-at-startup state —
    'off' / 'at login' / 'before login' / 'unknown'. No subprocess (the menu
    rebuilds on every poll). 'before login' means linger (Linux) / a system
    LaunchDaemon (macOS)."""
    home = Path.home()
    if sys.platform.startswith("linux"):
        unit = ((home / ".config/systemd/user/llamanager.service").exists() or
                (home / ".config/systemd/user/default.target.wants/"
                        "llamanager.service").exists())
        tray = (home / ".config/autostart/llamanager-tray.desktop").exists()
        if not (unit or tray):
            return "off"
        user = os.environ.get("USER", "")
        linger = bool(user) and Path(f"/var/lib/systemd/linger/{user}").exists()
        return "before login" if linger else "at login"
    if sys.platform == "darwin":
        sysd = Path("/Library/LaunchDaemons/com.llamanager.plist").exists()
        agent = (home / "Library/LaunchAgents/com.llamanager.plist").exists()
        tray = (home / "Library/LaunchAgents/com.llamanager.tray.plist").exists()
        if not (sysd or agent or tray):
            return "off"
        return "before login" if sysd else "at login"
    return "unknown"  # Windows needs sc/schtasks to tell


class TrayApp:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = TrayState()
        self._stop = threading.Event()
        self._icon = None  # set in run()
        # The dashboard lives at /ui/, not the bare root.
        self._web_url = resolve_base_url(cfg).rstrip("/") + "/ui/"
        # Flicker guard: only touch the icon/menu when something visible
        # actually changed (see _poll_loop). Reassigning icon.icon every poll
        # makes the AppIndicator redraw, which looks like flickering.
        self._last_sig: tuple | None = None
        self._last_ok: bool | None = None
        # Armed graceful action: "stop" | "restart" | None. The poller checks
        # the drain condition on every tick and fires when it is met, so the
        # wait survives the menu being closed and reopened.
        self._graceful: str | None = None
        self._graceful_since: float = 0.0
        # Build a client once; admin key may be missing — degrade to a
        # reachability-only tray in that case rather than refusing to start.
        try:
            self._client: AdminClient | None = AdminClient.from_config(cfg)
        except AdminClientError as e:
            self._client = None
            self.state.update(last_error=f"no admin key: {e}")

    # ---- background poller ----

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            if self._icon is not None:
                try:
                    snap = self.state.snapshot()
                    sig = self._display_signature(snap)
                    # Only rebuild the menu when a displayed value changed.
                    if sig != self._last_sig:
                        self._last_sig = sig
                        self._icon.menu = self._build_menu()
                        self._icon.update_menu()
                    # Only swap the icon image when up/down actually flips —
                    # reassigning it every tick is what caused the flicker.
                    ok = snap["daemon"].reachable
                    if ok != self._last_ok:
                        self._last_ok = ok
                        self._refresh_icon_image()
                except Exception as e:  # never let a UI hiccup kill the poller
                    log.debug("menu refresh failed: %s", e)
            self._stop.wait(_POLL_SECONDS)

    def _display_signature(self, snap: dict[str, Any]) -> tuple:
        """A tuple of everything the menu/icon shows. When it's unchanged
        between polls we leave the tray untouched (no flicker)."""
        d = snap["daemon"]
        st = snap["status"]
        models = tuple(_model_id(m) for m in snap["models"])
        return (
            d.reachable, d.detail,
            st.get("state"), st.get("current_model"), st.get("current_profile"),
            st.get("queue_depth"), st.get("in_flight_count"),
            # Included so the menu rebuilds when the switch is flipped
            # elsewhere (top bar, CLI) — not only when the tray flips it.
            st.get("accepting_requests"),
            st.get("active_downloads"), st.get("active_installs"),
            self._graceful,
            models, _autorun_label(), snap["last_error"][:60],
        )

    def _poll_once(self) -> None:
        dstate = service_ctl.state(self.cfg)
        status: dict[str, Any] = {}
        models: list[dict[str, Any]] = []
        err = ""
        if dstate.reachable and self._client is not None:
            try:
                status = self._client.status()
            except AdminClientError as e:
                err = str(e)
            try:
                models = self._client.models_list()
            except AdminClientError as e:
                err = err or str(e)
        self.state.update(daemon=dstate, status=status, models=models,
                          last_error=err)
        if self._graceful:
            self._graceful_tick(dstate, status)

    def _refresh_icon_image(self) -> None:
        snap = self.state.snapshot()
        _, Image = _require_pystray()
        self._icon.icon = _load_icon_image(Image, ok=snap["daemon"].reachable)

    # ---- menu actions ----

    def _safe_admin(self, fn, label: str) -> bool:
        """Run an admin call, reporting failures. Returns whether it worked so
        callers can keep their own success notification honest."""
        if self._client is None:
            self._notify("No admin key configured")
            return False
        ok = True
        try:
            fn(self._client)
        except AdminClientError as e:
            ok = False
            self._notify(f"{label} failed: {e}")
        # Refresh promptly so the menu reflects the new state.
        self._poll_once()
        if self._icon is not None:
            self._icon.menu = self._build_menu()
            self._icon.update_menu()
        return ok

    def _notify(self, msg: str) -> None:
        log.info(msg)
        try:
            if self._icon is not None and self._icon.HAS_NOTIFICATION:
                self._icon.notify(msg, "llamanager")
        except Exception:
            pass

    # ---- graceful stop / restart --------------------------------------
    #
    # "Graceful" here means: stop taking NEW work, let everything already
    # accepted finish, and only then bounce the service. It is not a
    # suspend-and-resume — the queue is an in-memory heap of live HTTP
    # requests, so a queued request cannot outlive the process that is
    # holding its connection open. Finishing the backlog BEFORE the restart
    # is therefore the only way not to lose it, which is what this does.
    #
    # Draining uses /admin/intake/drain rather than the pause switch: pausing
    # cancels the backlog, and cancelling the queue we are waiting for would
    # defeat the entire point.

    @staticmethod
    def _waiting_on(status: dict[str, Any]) -> list[str]:
        """What still has to finish before a graceful action can fire.

        Downloads and engine installs are counted alongside requests because
        a restart does not pause them: an HF pull resumes from byte 0 and an
        install leaves a half-built venv behind.
        """
        bits: list[str] = []
        running = int(status.get("in_flight_count") or 0)
        queued = int(status.get("queue_depth") or 0)
        dls = int(status.get("active_downloads") or 0)
        ins = int(status.get("active_installs") or 0)
        if running:
            bits.append(f"{running} running")
        if queued:
            bits.append(f"{queued} queued")
        if dls:
            bits.append(f"{dls} download{'s' if dls != 1 else ''}")
        if ins:
            bits.append(f"{ins} install{'s' if ins != 1 else ''}")
        return bits

    def _act_graceful(self, action: str):
        """Arm (or cancel) a graceful stop/restart."""
        def run(*_: Any) -> None:
            if self._graceful:
                self._cancel_graceful()
                return
            if self._client is None:
                self._notify("No admin key configured")
                return
            if not self._safe_admin(lambda c: c.intake_drain(), "Drain"):
                return
            self._graceful = action
            self._graceful_since = time.time()
            status = self.state.snapshot()["status"]
            waiting = self._waiting_on(status)
            self._notify(
                f"Graceful {action}: not taking new requests; waiting for "
                + (", ".join(waiting) if waiting else "nothing")
                + ". Queued work still runs.")
            # Nothing outstanding? Then this tick is the one that fires.
            self._graceful_tick(self.state.snapshot()["daemon"], status)
        return run

    def _cancel_graceful(self) -> None:
        action, self._graceful = self._graceful, None
        self._safe_admin(lambda c: c.intake_resume(), "Resume intake")
        self._notify(f"Graceful {action} cancelled — taking requests again")

    def _graceful_tick(self, dstate, status: dict[str, Any]) -> None:
        """One check of the drain condition; fires the action when it is met."""
        if not dstate.reachable:
            # It went away on its own (crash, or someone else stopped it).
            # A restart still has a job to do; a stop is already done.
            action, self._graceful = self._graceful, None
            if action == "restart":
                ok, msg = service_ctl.start_daemon(self.cfg)
                self._notify(f"Graceful restart: service was down, started it — {msg}")
            else:
                self._notify("Graceful stop: service already stopped")
            return
        if self._waiting_on(status):
            return                      # still busy; try again next tick

        action, self._graceful = self._graceful, None
        # Re-open intake BEFORE bouncing, so the daemon that comes back is
        # taking requests. The switch is persisted, so leaving it closed here
        # would silently start a shut daemon next time.
        self._safe_admin(lambda c: c.intake_resume(), "Resume intake")
        if action == "restart":
            ok, msg = service_ctl.restart_daemon(self.cfg)
            self._notify(f"Graceful restart: {msg}")
        else:
            ok, msg = service_ctl.stop_daemon(self.cfg)
            self._notify(f"Graceful stop: {msg}")
        self._poll_once()

    # daemon (OS service) controls
    def _act_daemon_start(self, *_: Any) -> None:
        ok, msg = service_ctl.start_daemon(self.cfg)
        self._notify(f"Service start: {msg}")
        self._poll_once()

    def _act_daemon_stop(self, *_: Any) -> None:
        ok, msg = service_ctl.stop_daemon(self.cfg)
        self._notify(f"Service stop: {msg}")
        self._poll_once()

    def _act_daemon_restart(self, *_: Any) -> None:
        ok, msg = service_ctl.restart_daemon(self.cfg)
        self._notify(f"Service restart: {msg}")
        self._poll_once()

    # autorun at startup — shell out to `llamanager autostart --mode ...`,
    # which runs in this desktop session (no extra privilege on Linux/macOS).
    def _act_set_autorun(self, mode: str):
        def run(*_: Any) -> None:
            cmd = [sys.executable, "-m", "llamanager", "autostart",
                   "--mode", mode]
            threading.Thread(
                target=self._run_autorun, args=(cmd, mode), daemon=True).start()
        return run

    def _run_autorun(self, cmd: list[str], mode: str) -> None:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=120, check=False)
            ok = r.returncode == 0
            self._notify(f"Autorun → {mode}: {'done' if ok else 'failed'}")
            if not ok:
                log.error("autostart --mode %s failed: %s", mode,
                          (r.stderr or r.stdout).strip())
        except (OSError, subprocess.SubprocessError) as e:
            self._notify(f"Autorun change failed: {e}")
        # Rebuild the menu so the radio reflects the new state.
        if self._icon is not None:
            try:
                self._icon.menu = self._build_menu()
                self._icon.update_menu()
            except Exception:
                pass

    # LLM controls
    def _act_launch_model(self, model_id: str):
        # No profile arg → daemon uses this model's default profile.
        return lambda *_: self._safe_admin(
            lambda c: c.server_start(model=model_id), f"Launch {model_id}")

    def _act_llm_stop(self, *_: Any) -> None:
        self._safe_admin(lambda c: c.server_stop(), "Stop LLM")

    def _act_llm_restart(self, *_: Any) -> None:
        self._safe_admin(lambda c: c.server_restart(), "Restart LLM")

    # intake switch — the daemon-wide "stop taking requests" toggle. Same
    # state the top bar and `llamanager intake` drive; the tray reads it from
    # the status poll it already makes.
    def _act_intake_toggle(self, *_: Any) -> None:
        if self._accepting(self.state.snapshot()["status"]):
            if self._safe_admin(lambda c: c.intake_pause(), "Pause intake"):
                self._notify("Not taking requests — new ones get 503 until "
                             "you switch this back on")
        else:
            if self._safe_admin(lambda c: c.intake_resume(), "Resume intake"):
                self._notify("Taking requests again")

    @staticmethod
    def _accepting(status: dict[str, Any]) -> bool:
        """Whether the daemon is taking requests. Absent on a daemon older
        than the intake switch — treat that as 'yes', which is what such a
        daemon does."""
        return bool(status.get("accepting_requests", True))

    def _act_open_ui(self, *_: Any) -> None:
        webbrowser.open(self._web_url)

    def _act_open_logs(self, *_: Any) -> None:
        _open_path(self.cfg.logs_dir)

    def _act_open_models(self, *_: Any) -> None:
        _open_path(self.cfg.models_dir)

    # ---- MCP ----------------------------------------------------------
    #
    # The tray is where someone is standing when they want to point an
    # agent at this machine, so the snippets live one click away here as
    # well as on the Connect page. The key is deliberately NOT included:
    # origin keys are shown once at mint time and only hashed after that,
    # so the tray cannot know one. The snippets carry a placeholder and
    # the Connect page is one item below.

    @property
    def _mcp_url(self) -> str:
        return resolve_base_url(self.cfg).rstrip("/") + "/mcp"

    @property
    def _mcp_connect_url(self) -> str:
        return resolve_base_url(self.cfg).rstrip("/") + "/ui/connect"

    def _act_open_connect(self, *_: Any) -> None:
        webbrowser.open(self._mcp_connect_url)

    def _copy(self, what: str, text: str) -> None:
        err = _copy_to_clipboard(text)
        self._notify(f"Could not copy {what}: {err}" if err
                     else f"Copied {what} to the clipboard")

    def _act_copy_mcp_url(self, *_: Any) -> None:
        self._copy("the MCP endpoint", self._mcp_url)

    def _act_copy_mcp_claude_code(self, *_: Any) -> None:
        self._copy("the Claude Code command", (
            f'claude mcp add --transport http llamanager {self._mcp_url} '
            f'--header "Authorization: Bearer YOUR_KEY"'))

    def _act_copy_mcp_stdio(self, *_: Any) -> None:
        self._copy("the Claude Desktop config", json.dumps({
            "mcpServers": {
                "llamanager": {
                    "command": "llamanager",
                    "args": ["mcp-stdio"],
                }
            }
        }, indent=2))

    def _act_copy_mcp_http(self, *_: Any) -> None:
        self._copy("the Cursor / VS Code config", json.dumps({
            "mcpServers": {
                "llamanager": {
                    "url": self._mcp_url,
                    "headers": {"Authorization": "Bearer YOUR_KEY"},
                }
            }
        }, indent=2))

    def _act_quit(self, *_: Any) -> None:
        # Quitting the tray tears the whole stack down: stop the LLM, then the
        # daemon service, then exit the icon loop. We notify first because the
        # shutdown below can take a few seconds, and the icon (and its ability
        # to post notifications) goes away once _icon.stop() runs.
        self._notify("Shutting down llamanager…")
        self._stop.set()  # halt the poller so it doesn't fight the teardown
        # Stop the LLM server it proxies (best-effort; needs an admin key and a
        # reachable daemon). Stopping the service below would take it down too,
        # but doing it explicitly is cleaner and lets it drain in-flight work.
        if self._client is not None and self.state.snapshot()["daemon"].reachable:
            try:
                self._client.server_stop()
            except AdminClientError as e:
                log.warning("LLM stop on quit failed: %s", e)
        # Stop the OS-managed daemon service itself.
        try:
            ok, msg = service_ctl.stop_daemon(self.cfg)
            log.info("service stop on quit: %s", msg)
        except Exception as e:  # never let teardown block the exit
            log.warning("service stop on quit failed: %s", e)
        if self._icon is not None:
            self._icon.stop()

    # ---- menu construction (rebuilt each poll so labels/state are live) ----

    def _build_menu(self):
        pystray, _ = _require_pystray()
        Item = pystray.MenuItem
        Menu = pystray.Menu
        snap = self.state.snapshot()
        d: service_ctl.DaemonState = snap["daemon"]
        st = snap["status"]
        up = d.reachable
        llm_state = st.get("state") or "stopped"
        cur_model = st.get("current_model")
        cur_profile = st.get("current_profile")

        accepting = self._accepting(st)

        # Header line (disabled). A closed door is called out here as well as
        # on the toggle below — from the client side a forgotten pause looks
        # exactly like an outage, so it should be visible without opening a
        # submenu.
        if up:
            header = "● Running" if accepting else "● Running — NOT taking requests"
        else:
            header = "○ Stopped"
        qd = st.get("queue_depth")
        inflight = st.get("in_flight_count")
        queue_line = (f"Queue: {inflight or 0} running / {qd or 0} queued"
                      if up else (snap["last_error"][:60] or "not reachable"))

        # Service submenu ("service" reads friendlier than "daemon").
        #
        # Plain Stop/Restart bounce the service now, killing whatever is
        # mid-generation. The graceful pair close the door, let the backlog
        # finish, and only then bounce — so the label has to say what is
        # still holding them up, or an operator watching a long queue has no
        # way to tell waiting from hung.
        waiting = self._waiting_on(st) if up else []
        if self._graceful:
            held = ", ".join(waiting) if waiting else "finishing up"
            graceful_items = [
                Item(f"Cancel graceful {self._graceful} (waiting: {held})",
                     self._act_graceful(self._graceful)),
            ]
        else:
            suffix = f" (waiting: {', '.join(waiting)})" if waiting else ""
            graceful_items = [
                Item(f"Graceful stop{suffix}", self._act_graceful("stop"),
                     enabled=up),
                Item(f"Graceful restart{suffix}", self._act_graceful("restart"),
                     enabled=up),
            ]
        service_menu = Menu(
            Item(lambda i: f"Status: {'running' if up else 'stopped'}",
                 None, enabled=False),
            Item("Start", self._act_daemon_start, enabled=not up),
            Item("Stop", self._act_daemon_stop, enabled=up),
            Item("Restart", self._act_daemon_restart, enabled=up),
            Menu.SEPARATOR,
            *graceful_items,
        )

        # Autorun-at-startup submenu. Radio over the three meaningful states;
        # each shells out to `llamanager autostart --mode ...` (runs in this
        # desktop session, so no extra privilege on Linux/macOS). The current
        # state is detected cheaply from files (see _autorun_label).
        cur_autorun = _autorun_label()
        def _ar(item, want):  # checked-predicate for a radio item
            return cur_autorun == want
        autorun_menu = Menu(
            Item("Off (don't start automatically)",
                 self._act_set_autorun("off"),
                 checked=lambda i: _ar(i, "off"), radio=True),
            Item("At login",
                 self._act_set_autorun("login-tray"),
                 checked=lambda i: _ar(i, "at login"), radio=True),
            Item("Before login (always on)",
                 self._act_set_autorun("tray+service"),
                 checked=lambda i: _ar(i, "before login"), radio=True),
        )

        # LLM "Launch ▸" submenu — one item per installed model.
        launch_items = []
        for m in snap["models"]:
            mid = _model_id(m)
            if not mid:
                continue
            launch_items.append(
                Item(mid, self._act_launch_model(mid),
                     checked=lambda item, mid_=mid: mid_ == cur_model,
                     radio=True)
            )
        if not launch_items:
            launch_items = [Item("(no models found)", None, enabled=False)]

        running_label = (f"Running: {cur_model} "
                         f"({cur_profile})" if cur_model else "Running: none")
        llm_menu = Menu(
            Item(running_label, None, enabled=False),
            Menu.SEPARATOR,
            Item("Launch", Menu(*launch_items), enabled=up),
            Item("Stop LLM", self._act_llm_stop,
                 enabled=up and llm_state == "running"),
            Item("Restart LLM", self._act_llm_restart,
                 enabled=up and llm_state == "running"),
        )

        # MCP submenu. The endpoint is up exactly when the daemon is, so
        # the items follow `up` rather than probing separately — a tray
        # poll should not cost an MCP handshake every few seconds.
        mcp_menu = Menu(
            Item(lambda i: f"Endpoint: {self._mcp_url}", None, enabled=False),
            Item("Connect page (mint a key)…", self._act_open_connect,
                 enabled=up),
            Menu.SEPARATOR,
            Item("Copy endpoint URL", self._act_copy_mcp_url),
            Item("Copy Claude Code command", self._act_copy_mcp_claude_code),
            Item("Copy Claude Desktop config (stdio)", self._act_copy_mcp_stdio),
            Item("Copy Cursor / VS Code config", self._act_copy_mcp_http),
        )

        return Menu(
            Item(header, None, enabled=False),
            Item(queue_line, None, enabled=False),
            Menu.SEPARATOR,
            Item("Open Web UI", self._act_open_ui, default=True),
            Menu.SEPARATOR,
            # Checked = taking requests. Unchecking closes the door: new
            # requests get 503 and the queued backlog is dropped, while
            # in-flight work finishes. Persists until switched back on.
            #
            # Without an admin key every control here is inert (the tray is
            # then reachability-only), so say so on the item instead of
            # offering a switch that silently does nothing when clicked.
            Item("Accepting requests" if self._client is not None
                 else "Accepting requests  (needs an admin key)",
                 self._act_intake_toggle,
                 checked=lambda i: accepting,
                 enabled=up and self._client is not None),
            Menu.SEPARATOR,
            Item("Service", service_menu),
            Item("LLM", llm_menu),
            Item("MCP", mcp_menu),
            Item(f"Autorun at startup  ({cur_autorun})", autorun_menu),
            Menu.SEPARATOR,
            Item("Open logs folder", self._act_open_logs),
            Item("Open models folder", self._act_open_models),
            Menu.SEPARATOR,
            Item("Quit tray", self._act_quit),
        )

    # ---- run ----

    def run(self) -> int:
        pystray, Image = _require_pystray()
        # Prime the snapshot before the first paint so the menu isn't empty.
        self._poll_once()
        self._icon = pystray.Icon(
            "llamanager",
            icon=_load_icon_image(Image, ok=self.state.snapshot()["daemon"].reachable),
            title="llamanager",
            menu=self._build_menu(),
        )
        poller = threading.Thread(target=self._poll_loop, daemon=True,
                                  name="tray-poller")
        poller.start()
        # Blocks on the main thread (required on macOS). Returns when
        # _act_quit calls icon.stop().
        self._icon.run()
        self._stop.set()
        return 0


def _setup_tray_logging(cfg: Config | None = None) -> Path | None:
    """Log to logs_dir/tray.log so a failed *autostart* launch (no console)
    leaves a trace instead of vanishing. Also keeps the console handler.

    ``cfg`` is optional so this can run *before* the config is loaded — a
    config failure is exactly the case that used to vanish silently. Without
    a config we use the default data dir, which lives in $HOME and so is
    available even when a configured models volume is not."""
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logs_dir = cfg.logs_dir if cfg is not None else expand("~/.llamanager") / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "tray.log"
        fh = RotatingFileHandler(log_path, maxBytes=1 * 1024 * 1024,
                                 backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        return log_path
    except OSError:
        return None


def main(cfg: Config | None = None,
         config_path: Path | None = None) -> int:
    # Logging first, config second. An autostart launch has no console and no
    # systemd unit of its own, so anything that fails before the log file is
    # open leaves no trace at all — which is precisely what a bad models_dir
    # used to do.
    log_path = _setup_tray_logging(cfg)
    log.info("tray starting (pid=%s, log=%s)", os.getpid(), log_path)
    if cfg is None:
        try:
            cfg = load_config(config_path)
        except ConfigError as e:
            log.error("tray aborting — %s", e)
            return 1
        except Exception:
            log.exception("tray aborting: configuration could not be loaded")
            return 1
        # Now that the real config is known, attach its log file too if it
        # differs from the default we guessed above.
        if cfg.logs_dir / "tray.log" != log_path:
            log_path = _setup_tray_logging(cfg)
    try:
        return TrayApp(cfg).run()
    except SystemExit:
        raise  # already logged (e.g. missing pystray)
    except Exception:
        log.exception("tray exited with an unexpected error")
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
