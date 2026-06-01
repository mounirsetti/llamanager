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

import logging
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from . import service_ctl
from .admin_client import AdminClient, AdminClientError, resolve_base_url
from .config import Config, load_config

log = logging.getLogger("llamanager.tray")

# How often the background poller refreshes daemon/LLM state.
_POLL_SECONDS = 4.0


def _require_pystray():
    """Import pystray + Pillow lazily with an actionable error."""
    try:
        import pystray  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "error: the tray needs pystray + Pillow.\n"
            "  Install with: pip install 'llamanager[tray]'\n"
            f"  (import failed: {e})"
        )
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


class TrayApp:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = TrayState()
        self._stop = threading.Event()
        self._icon = None  # set in run()
        self._web_url = resolve_base_url(cfg)
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
                    self._icon.menu = self._build_menu()
                    self._icon.update_menu()
                    self._refresh_icon_image()
                except Exception as e:  # never let a UI hiccup kill the poller
                    log.debug("menu refresh failed: %s", e)
            self._stop.wait(_POLL_SECONDS)

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

    def _refresh_icon_image(self) -> None:
        snap = self.state.snapshot()
        _, Image = _require_pystray()
        self._icon.icon = _load_icon_image(Image, ok=snap["daemon"].reachable)

    # ---- menu actions ----

    def _safe_admin(self, fn, label: str) -> None:
        if self._client is None:
            self._notify("No admin key configured")
            return
        try:
            fn(self._client)
        except AdminClientError as e:
            self._notify(f"{label} failed: {e}")
        # Refresh promptly so the menu reflects the new state.
        self._poll_once()
        if self._icon is not None:
            self._icon.menu = self._build_menu()
            self._icon.update_menu()

    def _notify(self, msg: str) -> None:
        log.info(msg)
        try:
            if self._icon is not None and self._icon.HAS_NOTIFICATION:
                self._icon.notify(msg, "llamanager")
        except Exception:
            pass

    # daemon (OS service) controls
    def _act_daemon_start(self, *_: Any) -> None:
        ok, msg = service_ctl.start_daemon(self.cfg)
        self._notify(f"Daemon start: {msg}")
        self._poll_once()

    def _act_daemon_stop(self, *_: Any) -> None:
        ok, msg = service_ctl.stop_daemon(self.cfg)
        self._notify(f"Daemon stop: {msg}")
        self._poll_once()

    def _act_daemon_restart(self, *_: Any) -> None:
        ok, msg = service_ctl.restart_daemon(self.cfg)
        self._notify(f"Daemon restart: {msg}")
        self._poll_once()

    def _act_toggle_boot(self, icon, item) -> None:
        # `item.checked` reflects the state *before* the click.
        ok, msg = service_ctl.set_autostart(self.cfg, not item.checked)
        self._notify(f"Start at boot: {msg}")
        self._poll_once()

    # app-level autolaunch (daemon auto-loads default LLM)
    def _act_toggle_autolaunch(self, icon, item) -> None:
        self._safe_admin(lambda c: c.setup_autolaunch(not item.checked),
                         "Auto-load default LLM")

    # LLM controls
    def _act_launch_model(self, model_id: str):
        # No profile arg → daemon uses this model's default profile.
        return lambda *_: self._safe_admin(
            lambda c: c.server_start(model=model_id), f"Launch {model_id}")

    def _act_llm_stop(self, *_: Any) -> None:
        self._safe_admin(lambda c: c.server_stop(), "Stop LLM")

    def _act_llm_restart(self, *_: Any) -> None:
        self._safe_admin(lambda c: c.server_restart(), "Restart LLM")

    def _act_open_ui(self, *_: Any) -> None:
        webbrowser.open(self._web_url)

    def _act_open_logs(self, *_: Any) -> None:
        _open_path(self.cfg.logs_dir)

    def _act_open_models(self, *_: Any) -> None:
        _open_path(self.cfg.models_dir)

    def _act_quit(self, *_: Any) -> None:
        self._stop.set()
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

        # Header line (disabled).
        if up:
            header = f"● daemon UP · {d.detail or 'serving'}"
        else:
            header = "○ daemon DOWN"
        qd = st.get("queue_depth")
        inflight = st.get("in_flight_count")
        queue_line = (f"Queue: {inflight or 0} running / {qd or 0} queued"
                      if up else (snap["last_error"][:60] or "not reachable"))

        # Daemon submenu.
        daemon_menu = Menu(
            Item(lambda i: f"Status: {d.detail or ('up' if up else 'down')}",
                 None, enabled=False),
            Item("Start daemon", self._act_daemon_start, enabled=not up),
            Item("Stop daemon", self._act_daemon_stop, enabled=up),
            Item("Restart daemon", self._act_daemon_restart, enabled=up),
            Menu.SEPARATOR,
            Item("Start daemon at boot", self._act_toggle_boot,
                 checked=lambda i: bool(d.autostart),
                 enabled=d.installed is not False),
            Item("Auto-load default LLM", self._act_toggle_autolaunch,
                 checked=lambda i: bool(st.get("autolaunch")),
                 enabled=up),
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

        return Menu(
            Item(header, None, enabled=False),
            Item(queue_line, None, enabled=False),
            Menu.SEPARATOR,
            Item("Open Web UI", self._act_open_ui, default=True),
            Menu.SEPARATOR,
            Item("Daemon", daemon_menu),
            Item("LLM", llm_menu),
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


def main(cfg: Config | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return TrayApp(cfg or load_config()).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
