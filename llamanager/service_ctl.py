"""Cross-platform control of the llamanager *daemon* as an OS-managed
background service, plus autostart (start-at-boot/login) management.

This is what the tray's **Daemon** submenu drives. It is deliberately
separate from :class:`~llamanager.admin_client.AdminClient`, which
controls the *llama-server* the daemon proxies — not the daemon process
itself.

Three platforms, three mechanisms:

* **Linux**   — a ``systemd --user`` unit ``llamanager.service``.
  ``loginctl enable-linger`` lets it run at boot before login. Control
  (start/stop/enable/disable) needs no root.
* **macOS**   — a per-user **LaunchAgent** ``com.llamanager``. Driven with
  ``launchctl``; needs no root. Runs at login (not before).
* **Windows** — a Windows **service** ``llamanager`` (see
  :mod:`llamanager.win_service`). The installer grants the current
  account START/STOP/QUERY via the service security descriptor, so
  control needs no UAC prompt afterward.

Every function degrades gracefully: detection never raises (returns
``None``/``False`` on any error), and control functions return a
``(ok, message)`` tuple rather than throwing, so the tray can surface a
short status line instead of crashing its event loop.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import dataclass

from .config import Config

# Identifiers must match the installers (installer.py) and win_service.py.
SYSTEMD_UNIT = "llamanager.service"
LAUNCHD_LABEL = "com.llamanager"
WINDOWS_SERVICE = "llamanager"

# How long to wait on a native control tool before giving up.
_CTL_TIMEOUT = 20.0


@dataclass
class DaemonState:
    """Snapshot of the daemon's OS-service status for the tray header."""

    reachable: bool  # HTTP port answers — the authoritative "is it up?"
    installed: bool | None  # service/unit/agent registered with the OS
    autostart: bool | None  # starts automatically at boot/login
    detail: str = ""  # short human note (e.g. raw state, or an error)


# ---------------------------------------------------------------------------
# Reachability — platform-agnostic and privilege-free. A TCP connect to the
# bound port is the most reliable "is the daemon actually serving?" signal,
# independent of how (or whether) it was registered as a service.
# ---------------------------------------------------------------------------


def daemon_reachable(cfg: Config, *, timeout: float = 1.0) -> bool:
    host = cfg.bind if cfg.bind not in ("0.0.0.0", "::") else "127.0.0.1"
    try:
        with socket.create_connection((host, cfg.port), timeout=timeout):
            return True
    except OSError:
        return False


def _run(cmd: list[str], *, timeout: float = _CTL_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a native control command, capturing output. Never raises for a
    non-zero exit — the caller inspects ``returncode``/``stdout``."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )


# ---------------------------------------------------------------------------
# Platform dispatch
# ---------------------------------------------------------------------------


def _platform() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def state(cfg: Config) -> DaemonState:
    """Best-effort snapshot. Reachability is always trustworthy; the
    installed/autostart flags are None when we can't determine them."""
    reachable = daemon_reachable(cfg)
    try:
        plat = _platform()
        if plat == "linux":
            installed, autostart, detail = _linux_state()
        elif plat == "macos":
            installed, autostart, detail = _macos_state()
        else:
            installed, autostart, detail = _windows_state()
    except Exception as e:  # detection must never crash the tray
        installed, autostart, detail = None, None, f"state probe failed: {e!r}"
    return DaemonState(reachable=reachable, installed=installed,
                       autostart=autostart, detail=detail)


def start_daemon(cfg: Config) -> tuple[bool, str]:
    return _dispatch_start()


def stop_daemon(cfg: Config) -> tuple[bool, str]:
    return _dispatch_stop()


def restart_daemon(cfg: Config) -> tuple[bool, str]:
    plat = _platform()
    if plat == "linux":
        r = _run(["systemctl", "--user", "restart", SYSTEMD_UNIT])
        return (r.returncode == 0, r.stderr.strip() or "restarted")
    # macOS/Windows: stop then start (no atomic restart verb we rely on).
    ok, msg = _dispatch_stop()
    if not ok:
        return ok, msg
    return _dispatch_start()


def set_autostart(cfg: Config, enabled: bool) -> tuple[bool, str]:
    plat = _platform()
    if plat == "linux":
        return _linux_set_autostart(enabled)
    if plat == "macos":
        return _macos_set_autostart(enabled)
    return _windows_set_autostart(enabled)


def _dispatch_start() -> tuple[bool, str]:
    plat = _platform()
    if plat == "linux":
        r = _run(["systemctl", "--user", "start", SYSTEMD_UNIT])
        return (r.returncode == 0, r.stderr.strip() or "started")
    if plat == "macos":
        target, is_system = _macos_target()
        r = _run(["launchctl", "kickstart", "-k", target])
        if r.returncode != 0 and is_system:
            return (False, "system daemon — run: "
                           f"sudo launchctl kickstart -k {target}")
        return (r.returncode == 0, r.stderr.strip() or "started")
    r = _run(["sc", "start", WINDOWS_SERVICE])
    return (r.returncode == 0, (r.stdout or r.stderr).strip() or "started")


def _dispatch_stop() -> tuple[bool, str]:
    plat = _platform()
    if plat == "linux":
        r = _run(["systemctl", "--user", "stop", SYSTEMD_UNIT])
        return (r.returncode == 0, r.stderr.strip() or "stopped")
    if plat == "macos":
        target, is_system = _macos_target()
        r = _run(["launchctl", "bootout", target])
        # bootout returns non-zero if already stopped — treat "no such
        # process" as success so the tray toggle is idempotent.
        ok = r.returncode == 0 or "No such process" in (r.stderr or "")
        if not ok and is_system:
            return (False, f"system daemon — run: sudo launchctl bootout {target}")
        return (ok, r.stderr.strip() or "stopped")
    r = _run(["sc", "stop", WINDOWS_SERVICE])
    return (r.returncode == 0, (r.stdout or r.stderr).strip() or "stopped")


# ---------------------------------------------------------------------------
# Linux — systemd --user
# ---------------------------------------------------------------------------


def _linux_state() -> tuple[bool | None, bool | None, str]:
    active = _run(["systemctl", "--user", "is-active", SYSTEMD_UNIT])
    enabled = _run(["systemctl", "--user", "is-enabled", SYSTEMD_UNIT])
    # is-active prints "active"/"inactive"/"failed"; is-enabled prints
    # "enabled"/"disabled"/"static". A unit that doesn't exist yields a
    # non-zero exit and "Failed to get..."/"not-found" on stderr.
    out_enabled = (enabled.stdout or "").strip()
    installed = "not-found" not in (enabled.stderr or "") and out_enabled != ""
    autostart = out_enabled == "enabled"
    detail = (active.stdout or "").strip() or (active.stderr or "").strip()
    return (installed or None, autostart, detail)


def _linux_set_autostart(enabled: bool) -> tuple[bool, str]:
    verb = "enable" if enabled else "disable"
    r = _run(["systemctl", "--user", verb, SYSTEMD_UNIT])
    if r.returncode != 0:
        return (False, r.stderr.strip() or f"{verb} failed")
    if enabled:
        # Linger lets the --user manager (and thus the unit) run at boot
        # before anyone logs in. May require polkit auth; report but don't
        # treat its failure as fatal — the unit is still enabled-at-login.
        linger = _run(["loginctl", "enable-linger", os.environ.get("USER", "")])
        if linger.returncode != 0:
            return (True, "enabled (login only — enable-linger needs auth "
                          "for boot-before-login)")
    return (True, f"autostart {verb}d")


# ---------------------------------------------------------------------------
# macOS — per-user LaunchAgent
# ---------------------------------------------------------------------------


def _macos_system_daemon_installed() -> bool:
    """A pre-login install drops a plist in /Library/LaunchDaemons."""
    from pathlib import Path
    return Path(f"/Library/LaunchDaemons/{LAUNCHD_LABEL}.plist").exists()


def _macos_target() -> tuple[str, bool]:
    """Return (launchctl-target, is_system). Prefers the system daemon
    (pre-login install) when present, else the per-user GUI agent."""
    if _macos_system_daemon_installed():
        return f"system/{LAUNCHD_LABEL}", True
    uid = os.getuid()  # type: ignore[attr-defined]
    return f"gui/{uid}/{LAUNCHD_LABEL}", False


def _macos_state() -> tuple[bool | None, bool | None, str]:
    target, is_system = _macos_target()
    printed = _run(["launchctl", "print", target])
    installed = printed.returncode == 0
    if is_system:
        # System daemon: RunAtLoad means it autostarts at boot. We can't
        # easily read its enabled bit without root, so report installed.
        detail = "system daemon (pre-login)" if installed else "not loaded"
        return (installed or _macos_system_daemon_installed() or None,
                installed or None, detail)
    uid = os.getuid()  # type: ignore[attr-defined]
    # "disabled" services list under `launchctl print-disabled gui/$UID`.
    disabled = _run(["launchctl", "print-disabled", f"gui/{uid}"])
    autostart = installed and f'"{LAUNCHD_LABEL}" => disabled' not in (
        disabled.stdout or "")
    detail = "loaded" if installed else "not loaded"
    return (installed or None, autostart if installed else None, detail)


def _macos_set_autostart(enabled: bool) -> tuple[bool, str]:
    target, is_system = _macos_target()
    if is_system:
        # A system LaunchDaemon autostarts via RunAtLoad; toggling it means
        # boot-out/in under root. Not something the unprivileged tray can do.
        return (False, "system daemon autostarts at boot; change it with "
                       f"sudo launchctl (bootout/bootstrap system) — {target}")
    verb = "enable" if enabled else "disable"
    r = _run(["launchctl", verb, target])
    return (r.returncode == 0, r.stderr.strip() or f"autostart {verb}d")


# ---------------------------------------------------------------------------
# Windows — Service Control Manager (control rights granted at install time)
# ---------------------------------------------------------------------------


def _windows_state() -> tuple[bool | None, bool | None, str]:
    q = _run(["sc", "query", WINDOWS_SERVICE])
    installed = "FAILED 1060" not in (q.stdout + q.stderr)  # 1060 = no such service
    running = "RUNNING" in (q.stdout or "")
    qc = _run(["sc", "qc", WINDOWS_SERVICE])
    # qc prints "START_TYPE : 2  AUTO_START" / "3  DEMAND_START" / "4 DISABLED".
    autostart = "AUTO_START" in (qc.stdout or "")
    detail = "running" if running else ("stopped" if installed else "not installed")
    return (installed or None, autostart if installed else None, detail)


def _windows_set_autostart(enabled: bool) -> tuple[bool, str]:
    start_type = "auto" if enabled else "demand"
    # `sc config` needs SERVICE_CHANGE_CONFIG, granted to the user by the
    # installer's SD grant. Note the required space after "start=".
    r = _run(["sc", "config", WINDOWS_SERVICE, f"start={start_type}"])
    return (r.returncode == 0, (r.stdout or r.stderr).strip()
            or f"start type set to {start_type}")


__all__ = [
    "DaemonState", "daemon_reachable", "state",
    "start_daemon", "stop_daemon", "restart_daemon", "set_autostart",
    "SYSTEMD_UNIT", "LAUNCHD_LABEL", "WINDOWS_SERVICE",
]
