"""CLI entrypoint."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .config import AUDIO_ENGINES, expand, load_config, write_default_config

# Selectable ASR engines for `llamanager asr {install,setup,cancel-install}`.
# Derived from the engine registry so new audio engines appear automatically;
# ``asr`` (the default) is listed first.
AUDIO_ENGINE_CHOICES = ["asr"] + sorted(AUDIO_ENGINES - {"asr"})


def cmd_serve(args: argparse.Namespace) -> int:
    import os
    import uvicorn
    from .app import create_app

    cfg_path = Path(args.config) if args.config else None
    app = create_app(cfg_path)
    cfg = app.state.cfg
    # uvicorn's default LOGGING_CONFIG re-enables an INFO-level access
    # log handler on stdout that drowns llamanager.log with per-poll
    # "GET /ui/_partials/dashboard" lines. Skip it unless the operator
    # explicitly asks for verbose logs. ``log_config=None`` keeps the
    # uvicorn.* loggers attached only to whatever the root logger has
    # (the rotating file handler set up in ``_setup_logging``).
    verbose = bool(os.environ.get("LLAMANAGER_VERBOSE_LOGS"))
    uvicorn.run(
        app,
        host=args.host or cfg.bind,
        port=args.port or cfg.port,
        log_level=args.log_level,
        access_log=verbose,
        log_config=None,
    )
    return 0


def cmd_tray(args: argparse.Namespace) -> int:
    """Run the system-tray / menu-bar app (thin client over /admin/*).

    Talks to an already-running daemon; does not start one. The daemon
    is expected to run as an OS-managed background service so it can be
    always-on independent of the desktop session.

    Foreground by default — pystray's event loop blocks the terminal.
    Pass --background to detach (and silence GUI-toolkit warnings)."""
    if getattr(args, "background", False):
        if _spawn_detached(args.binary, "tray"):
            print("tray launched in the background (logs: see tray.log)")
            return 0
        return 1
    from .tray import main as tray_main
    cfg = load_config(Path(args.config) if args.config else None)
    return tray_main(cfg)


def cmd_init_config(args: argparse.Namespace) -> int:
    target = expand(args.path) if args.path else expand("~/.llamanager/config.toml")
    out = write_default_config(target)
    print(f"wrote default config to {out}")
    return 0


# ---------------------------------------------------------------------------
# `llamanager init` — the single guided front door. Stitches together the
# steps that used to be scattered: config, the bootstrap-key dance, and
# the autostart (background service + tray) setup. Engine/binary install is
# intentionally NOT here — that lives in the web UI (/ui/setup), which
# detects the GPU and installs the right build.
# ---------------------------------------------------------------------------


def _show_bootstrap_key(cfg, key: str) -> None:
    """Persist the bootstrap key 0600 and print it. Mirrors
    app._emit_bootstrap_key but without importing the FastAPI stack —
    `init` should stay lightweight."""
    import os
    p = cfg.data_dir / "bootstrap-key.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    line = "=" * 78
    print(f"\n{line}")
    print("  BOOTSTRAP ADMIN KEY (shown once — also saved to:")
    print(f"  {p})")
    print(f"  {key}")
    print("  Paste it at /ui/login, create a real origin, then revoke "
          "'bootstrap' and delete that file.")
    print(f"{line}")


def cmd_init(args: argparse.Namespace) -> int:
    """Guided first-run setup: config → admin key → start the daemon + tray now.

    `init` deliberately does NOT configure run-at-startup. It just gets you
    running immediately (the tray icon appears without a restart); whether
    llamanager comes back at boot/login is chosen later from the tray icon's
    right-click menu or the web UI. Engine install also lives in the web UI."""
    from . import service_ctl
    from .auth import AuthManager, load_or_create_lookup_secret
    from .db import DB

    print("llamanager — first-run setup\n" + "=" * 40)

    # 1. Config -------------------------------------------------------------
    cfg_path = (Path(args.config) if args.config
                else expand("~/.llamanager/config.toml"))
    if cfg_path.exists():
        print(f"[1/2] config: found {cfg_path}")
    else:
        out = write_default_config(cfg_path)
        print(f"[1/2] config: wrote default to {out}")
    cfg = load_config(cfg_path)

    # 2. Bootstrap admin key (reliable — no running daemon needed) ----------
    # We create the bootstrap origin directly so the key is captured even
    # when the user goes on to install a headless service (the old failure
    # mode where the service swallowed the only printout).
    db = DB(cfg.db_path)
    try:
        am = AuthManager(
            db, lookup_secret=load_or_create_lookup_secret(cfg.data_dir),
            default_priority=cfg.default_origin_priority)
        key = am.ensure_bootstrap()
    finally:
        db.close()
    if key:
        print("[2/2] admin key: created bootstrap key")
        _show_bootstrap_key(cfg, key)
    else:
        keyfile = cfg.data_dir / "bootstrap-key.txt"
        print("[2/2] admin key: already initialized (origins exist).")
        if keyfile.exists():
            print(f"       bootstrap key still on disk: {keyfile}")
        else:
            print("       if you've lost all keys, create one via an existing "
                  "admin origin, or reset state.db.")

    # Start now — daemon (if not already up) + tray, so the icon appears
    # immediately, no restart. Persistence across reboots is opt-in later.
    if not args.no_launch:
        print("\nStarting now (no restart needed):")
        _ensure_tray_deps()
        if service_ctl.daemon_reachable(cfg):
            print("  daemon: already running")
        else:
            # If a service/autostart is already installed, start it through
            # the service manager so systemd/launchd owns the process. A bare
            # detached `serve` here would grab the port and leave the service
            # stuck in a bind-failure restart loop (an orphan competing with
            # the managed daemon). Only spawn an unmanaged serve when there's
            # no service to manage it.
            if service_ctl.state(cfg).installed:
                ok, msg = service_ctl.start_daemon(cfg)
                print(f"  daemon (service): {msg}")
            else:
                _spawn_serve(args.binary)
        _spawn_tray(args.binary)

    # Next steps ------------------------------------------------------------
    host = cfg.bind if cfg.bind not in ("0.0.0.0", "::") else "127.0.0.1"
    print("\n" + "=" * 40 + "\nDone.")
    if args.no_launch:
        print("  • start it:          llamanager serve   (and `llamanager tray`)")
    else:
        print("  • the tray icon should now be in your menu bar / notification area")
    print(f"  • open the UI:       http://{host}:{cfg.port}/ui/login  "
          "(paste the bootstrap key)")
    print("  • run at startup:    choose it from the tray icon "
          "(right-click → Autorun at startup) or in the web UI")
    print("  • install an engine: /ui/setup   •   download a model: /ui/models")
    return 0


def cmd_install_launchd(args: argparse.Namespace) -> int:
    from .installer import install_launchd
    cfg = load_config(Path(args.config) if args.config else None)
    plist_path = install_launchd(label=args.label, port=args.port,
                                 binary=args.binary, cfg=cfg)
    print(f"installed LaunchAgent at {plist_path}")
    print("To load now: `launchctl load -w` followed by the plist path.")
    return 0


def cmd_install_systemd(args: argparse.Namespace) -> int:
    from .installer import install_systemd
    cfg = load_config(Path(args.config) if args.config else None)
    unit_path = install_systemd(unit_name=args.unit, port=args.port,
                                binary=args.binary, cfg=cfg)
    print(f"installed user systemd unit at {unit_path}")
    print("To enable: `systemctl --user daemon-reload && "
          "systemctl --user enable --now llamanager.service`.")
    return 0


def cmd_install_windows(args: argparse.Namespace) -> int:
    from .installer import install_windows_task
    cfg = load_config(Path(args.config) if args.config else None)
    xml_path = install_windows_task(task_name=args.task, port=args.port,
                                    binary=args.binary, cfg=cfg)
    print(f"wrote Task Scheduler XML to {xml_path}")
    print("To register (runs at user logon):")
    print(f'  schtasks /Create /XML "{xml_path}" /TN {args.task}')
    print("To start it now:")
    print(f"  schtasks /Run /TN {args.task}")
    print("To remove:")
    print(f"  schtasks /Delete /TN {args.task} /F")
    return 0


def _run_win_service_module(extra_args: list[str]) -> int:
    """Subprocess into `python -m llamanager.win_service <args>`.

    We use a subprocess (rather than calling `win_service.main()` in
    this process) so that the pywin32 imports happen in a fresh
    interpreter — this keeps `import llamanager` clean on hosts that
    don't have pywin32 and avoids polluting our argparse namespace
    with pywin32's CLI conventions."""
    import subprocess
    if sys.platform != "win32":
        print("error: Windows service commands are only available on Windows",
              file=sys.stderr)
        return 2
    try:
        import win32serviceutil  # noqa: F401  (probe only)
    except ImportError:
        print("error: pywin32 is not installed.\n"
              "  Install with: pip install 'llamanager[windows-service]'\n"
              "  Or directly:  pip install pywin32",
              file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "llamanager.win_service", *extra_args]
    return subprocess.call(cmd)


def cmd_install_windows_service(args: argparse.Namespace) -> int:
    """Register llamanager as a real Windows service via pywin32.

    Requires admin rights — run from an elevated shell. After install,
    the service is set to start automatically at boot."""
    extra: list[str] = []
    if args.username:
        extra += ["--username", args.username]
    if args.password:
        extra += ["--password", args.password]
    extra += ["--startup", args.startup, "install"]
    rc = _run_win_service_module(extra)
    if rc != 0:
        return rc
    if args.start:
        print("starting service...")
        rc = _run_win_service_module(["start"])
    print("\nManagement commands:")
    print("  sc query llamanager                       (status)")
    print("  python -m llamanager.win_service start    (start)")
    print("  python -m llamanager.win_service stop     (stop)")
    print("  python -m llamanager.win_service restart  (restart)")
    print("  llamanager remove-windows-service         (remove)")
    return rc


def cmd_remove_windows_service(args: argparse.Namespace) -> int:
    # stop first if running; ignore failure (service might not be running)
    _run_win_service_module(["stop"])
    return _run_win_service_module(["remove"])


# ---------------------------------------------------------------------------
# Autostart: how llamanager runs at boot/login. One command, four modes,
# three platforms. The tray icon is produced here (modes that include it),
# so `init` just asks which mode and delegates to `cmd_autostart`.
#
#   off            tear down all autostart
#   boot-service   always-on headless daemon, no tray (before login where
#                  the OS allows: Linux linger / Windows service / macOS
#                  needs --pre-login, else falls back to login)
#   login-tray     daemon + tray, both at login (no boot service)
#   tray+service   always-on daemon + login tray (recommended; "mode #3")
# ---------------------------------------------------------------------------

# mode -> (daemon_kind, want_tray); daemon_kind in {"none","login","boot"}.
_AUTOSTART_MODES = {
    "off": ("none", False),
    "boot-service": ("boot", False),
    "login-tray": ("login", True),
    "tray+service": ("boot", True),
}


def _sh(cmd: list[str], *, label: str | None = None) -> int:
    """Run a setup step, echoing the command and its outcome. Returns the
    exit code; never raises (so one optional step can't abort the rest)."""
    import subprocess
    print(f"  $ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           check=False, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"    ! skipped ({e})")
        return 1
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if out:
        print("    " + out.replace("\n", "\n    "))
    if r.returncode != 0 and err:
        print(f"    ! {err.splitlines()[0]}")
    return r.returncode


def _ensure_tray_deps() -> None:
    """The tray needs pystray + Pillow (pip), and on Linux also PyGObject
    (`gi`) + an AppIndicator typelib (system packages). Auto-install the pip
    side into this interpreter; for the Linux system side we can only guide,
    since it needs apt/sudo. Best-effort — never aborts the autostart setup.
    The daemon runs fine regardless; only the icon depends on these."""
    import importlib
    import subprocess

    print("  tray dependencies:")
    try:
        importlib.import_module("pystray")
        importlib.import_module("PIL")
        print("    pystray + Pillow: present")
    except ImportError:
        print("    installing pystray + Pillow into this environment...")
        rc = subprocess.call([sys.executable, "-m", "pip", "install",
                              "pystray", "pillow"])
        if rc != 0:
            print("    ! install failed — run manually: "
                  "pip install 'llamanager[tray]'")
            return

    if not sys.platform.startswith("linux"):
        return
    # Linux GUI backend: pystray draws the icon via PyGObject + AppIndicator.
    # On Wayland/GNOME the X11 fallback won't show, so this matters.
    try:
        import gi
        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import AyatanaAppIndicator3  # noqa: F401
        print("    AppIndicator backend: ok")
        return
    except (ImportError, ValueError):
        pass
    print("    ! the icon needs PyGObject + an AppIndicator typelib. Install "
          "the system packages:")
    print("        sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1")
    # If this is a venv that can't see the system gi, point at the fix.
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print("      then let this venv use them (either):")
        print("        • recreate it:  python3 -m venv --system-site-packages "
              f"{sys.prefix}")
        print("        • or set  include-system-site-packages = true  in "
              f"{Path(sys.prefix) / 'pyvenv.cfg'}")
    print("      (the daemon still runs without this — only the tray icon is "
          "affected.)")


def cmd_autostart(args: argparse.Namespace) -> int:
    """Configure how llamanager starts at boot/login (see --mode)."""
    from . import installer, service_ctl
    cfg = load_config(Path(args.config) if args.config else None)
    daemon_kind, want_tray = _AUTOSTART_MODES[args.mode]
    binary = args.binary
    write_only = bool(args.write_only)
    start_now = not args.no_start

    if args.mode == "off":
        return _autostart_off(cfg)

    if want_tray and not write_only:
        _ensure_tray_deps()

    if sys.platform.startswith("linux"):
        return _autostart_linux(cfg, installer, service_ctl,
                                daemon_kind=daemon_kind, want_tray=want_tray,
                                binary=binary, write_only=write_only,
                                start_now=start_now)
    if sys.platform == "darwin":
        return _autostart_macos(cfg, installer,
                                daemon_kind=daemon_kind, want_tray=want_tray,
                                binary=binary, write_only=write_only,
                                start_now=start_now, pre_login=bool(args.pre_login))
    if sys.platform == "win32":
        return _autostart_windows(cfg, installer,
                                  daemon_kind=daemon_kind, want_tray=want_tray,
                                  binary=binary, write_only=write_only,
                                  start_now=start_now,
                                  username=args.username, password=args.password)
    print(f"error: unsupported platform {sys.platform!r}", file=sys.stderr)
    return 2


def _autostart_linux(cfg, installer, service_ctl, *, daemon_kind, want_tray,
                     binary, write_only, start_now) -> int:
    if daemon_kind != "none":
        print("\n[daemon] systemd --user unit")
        unit = installer.install_systemd(command="serve", binary=binary, cfg=cfg)
        print(f"  wrote {unit}")
        if not write_only:
            _sh(["systemctl", "--user", "daemon-reload"])
            _sh(["systemctl", "--user", "enable", "--now", "llamanager.service"])
            if daemon_kind == "boot":
                # Linger lets the --user manager run at boot before login.
                _sh(["loginctl", "enable-linger", os.environ.get("USER", "")])
        else:
            print("  --write-only: systemctl --user daemon-reload && "
                  "systemctl --user enable --now llamanager.service"
                  + ("  (+ loginctl enable-linger \"$USER\")" if daemon_kind == "boot" else ""))
    if want_tray:
        print("\n[tray] XDG autostart entry")
        desktop = installer.install_xdg_autostart(binary=binary)
        print(f"  wrote {desktop}")
        if start_now and not write_only:
            _spawn_tray(binary)
    print("\nDone." + (" The tray launches at your next login."
                       if want_tray else ""))
    return 0


def _autostart_macos(cfg, installer, *, daemon_kind, want_tray, binary,
                     write_only, start_now, pre_login) -> int:
    import os
    uid = os.getuid()
    if want_tray:
        print("\n[tray] LaunchAgent com.llamanager.tray")
        tray_plist = installer.install_launchd(label="com.llamanager.tray",
                                              command="tray", binary=binary, cfg=cfg)
        print(f"  wrote {tray_plist}")

    if daemon_kind == "boot" and pre_login:
        staging, dest = installer.install_launchdaemon(label="com.llamanager",
                                                       binary=binary, cfg=cfg)
        print(f"\n[daemon] system LaunchDaemon (pre-login) staged at {staging}")
        print(f"  → installs to {dest}; this needs root:")
        print(f"  sudo cp {staging} {dest}")
        print(f"  sudo launchctl bootstrap system {dest}")
        if not write_only and start_now:
            _sh(["sudo", "cp", str(staging), str(dest)])
            _sh(["sudo", "launchctl", "bootstrap", "system", str(dest)])
        print("  NOTE: stopping/starting a system daemon needs sudo, so the "
              "tray's daemon controls will require it.")
    elif daemon_kind != "none":
        if daemon_kind == "boot":
            print("\n  note: macOS can't start a per-user daemon before login; "
                  "installing as a login agent. Re-run with --pre-login for "
                  "true before-login start.")
        print("\n[daemon] LaunchAgent com.llamanager (login-triggered)")
        daemon_plist = installer.install_launchd(label="com.llamanager",
                                                 command="serve", binary=binary, cfg=cfg)
        print(f"  wrote {daemon_plist}")
        if not write_only:
            _sh(["launchctl", "bootout", f"gui/{uid}/{daemon_plist.stem}"])
            _sh(["launchctl", "bootstrap", f"gui/{uid}", str(daemon_plist)])

    if want_tray and not write_only and start_now:
        _sh(["launchctl", "bootout", f"gui/{uid}/com.llamanager.tray"])
        _sh(["launchctl", "bootstrap", f"gui/{uid}",
             str(installer.expand("~/Library/LaunchAgents/com.llamanager.tray.plist"))])
    if write_only:
        print("\n--write-only: load each plist with "
              f"`launchctl bootstrap gui/{uid} <plist>`.")
    print("\nDone.")
    return 0


def _autostart_windows(cfg, installer, *, daemon_kind, want_tray, binary,
                       write_only, start_now, username, password) -> int:
    if daemon_kind == "boot":
        print("\n[daemon] Windows service (needs an elevated shell)")
        extra: list[str] = []
        if username:
            extra += ["--username", username]
        if password:
            extra += ["--password", password]
        extra += ["--startup", "auto", "install"]
        if not write_only:
            rc = _run_win_service_module(extra)
            if rc != 0:
                print("  ! service install failed — re-run from an "
                      "Administrator shell (the service step needs elevation).")
            elif start_now:
                _run_win_service_module(["start"])
            ok, msg = installer.grant_windows_service_control("llamanager", username)
            print(f"  control grant: {'ok' if ok else '!'} {msg}")
            if not ok:
                print("    (run elevated — editing the service SD needs admin)")
        else:
            print("  --write-only: install with: llamanager install-windows-service")
    elif daemon_kind == "login":
        print("\n[daemon] logon Task Scheduler entry")
        xml = installer.install_windows_task(task_name="llamanager",
                                             command="serve", binary=binary, cfg=cfg)
        print(f"  wrote {xml}")
        if not write_only and start_now:
            _sh(["schtasks", "/Create", "/XML", str(xml), "/TN", "llamanager", "/F"])
            _sh(["schtasks", "/Run", "/TN", "llamanager"])

    if want_tray:
        print("\n[tray] logon Task Scheduler entry")
        xml = installer.install_windows_task(task_name="llamanager-tray",
                                             command="tray", binary=binary, cfg=cfg)
        print(f"  wrote {xml}")
        print("  register it (no elevation needed):")
        print(f'    schtasks /Create /XML "{xml}" /TN llamanager-tray /F')
        if not write_only and start_now:
            _sh(["schtasks", "/Create", "/XML", str(xml), "/TN",
                 "llamanager-tray", "/F"])
            _sh(["schtasks", "/Run", "/TN", "llamanager-tray"])
    print("\nDone.")
    return 0


def _autostart_off(cfg) -> int:
    """Tear down every autostart artifact on this platform."""
    from .config import expand
    print("Removing llamanager autostart:")
    if sys.platform.startswith("linux"):
        _sh(["systemctl", "--user", "disable", "--now", "llamanager.service"])
        desktop = expand("~/.config/autostart/llamanager-tray.desktop")
        if desktop.exists():
            desktop.unlink(); print(f"  removed {desktop}")
        return 0
    if sys.platform == "darwin":
        from .installer import LAUNCHDAEMON_DIR
        uid = os.getuid()
        for label in ("com.llamanager.tray", "com.llamanager"):
            _sh(["launchctl", "bootout", f"gui/{uid}/{label}"])
            plist = expand(f"~/Library/LaunchAgents/{label}.plist")
            if plist.exists():
                plist.unlink(); print(f"  removed {plist}")
        sysd = LAUNCHDAEMON_DIR / "com.llamanager.plist"
        if sysd.exists():
            print(f"  found system daemon {sysd} — removing (needs sudo):")
            _sh(["sudo", "launchctl", "bootout", "system", str(sysd)])
            _sh(["sudo", "rm", "-f", str(sysd)])
        return 0
    if sys.platform == "win32":
        _sh(["schtasks", "/Delete", "/TN", "llamanager-tray", "/F"])
        _sh(["schtasks", "/Delete", "/TN", "llamanager", "/F"])
        print("  daemon service left intact — remove with: "
              "llamanager remove-windows-service")
        return 0
    print(f"error: unsupported platform {sys.platform!r}", file=sys.stderr)
    return 2


def _spawn_detached(binary: str | None, verb: str) -> bool:
    """Spawn `llamanager <verb>` as a detached background process. Returns
    True on launch. Used by `init` to bring the daemon/tray up immediately
    without installing any persistent autostart entry."""
    import shutil
    import subprocess
    exe = binary or shutil.which("llamanager") or sys.executable
    cmd = ([exe, verb] if exe.endswith(("llamanager", "llamanager.exe"))
           else [exe, "-m", "llamanager", verb])
    try:
        subprocess.Popen(cmd, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as e:
        print(f"  ! could not launch `{verb}` ({e})")
        return False


def _spawn_serve(binary: str | None) -> None:
    if _spawn_detached(binary, "serve"):
        print("  daemon: started (detached)")
    else:
        print("    run it manually: llamanager serve")


def _spawn_tray(binary: str | None) -> None:
    """Launch the tray now, detached, so the user sees it without a re-login."""
    if _spawn_detached(binary, "tray"):
        print("  tray: launched (detached)")
    else:
        print("    it will start when autorun is configured, or run: llamanager tray")


def cmd_install_tray(args: argparse.Namespace) -> int:
    """Deprecated alias for `autostart --mode tray+service`."""
    print("note: `install-tray` is now `autostart --mode tray+service`.\n")
    args.mode = "tray+service"
    return cmd_autostart(args)


def cmd_remove_tray(args: argparse.Namespace) -> int:
    """Deprecated alias for `autostart --mode off`."""
    cfg = load_config(Path(args.config) if args.config else None)
    return _autostart_off(cfg)


# ---------------------------------------------------------------------------
# `llamanager uninstall` — stop + remove all OS integration, optionally purge
# data/models. The package itself is removed with pip (we can't pip-uninstall
# the interpreter we're running in).
# ---------------------------------------------------------------------------


def _purge_data(cfg, *, include_models: bool) -> None:
    """Delete app data under data_dir (config, state.db, logs, secrets,
    staged plists, images). Keeps the models dir unless include_models."""
    import shutil
    data = cfg.data_dir
    models = cfg.models_dir

    def _models_under(p: Path) -> bool:
        try:
            return models == p or models.is_relative_to(p)
        except ValueError:
            return False

    if data.exists():
        for child in sorted(data.iterdir()):
            if not include_models and _models_under(child):
                print(f"  keeping models: {child}")
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            print(f"  removed {child}")
        # Remove the now-(near-)empty data dir only if nothing is left.
        try:
            data.rmdir()
            print(f"  removed {data}")
        except OSError:
            pass
    # A custom --config path may live outside data_dir; remove it too.
    cfg_path = cfg.config_path
    if cfg_path and cfg_path.exists() and not cfg_path.is_relative_to(data):
        cfg_path.unlink(missing_ok=True)
        print(f"  removed {cfg_path}")
    # Models dir outside data_dir.
    if include_models and models.exists() and not models.is_relative_to(data):
        shutil.rmtree(models, ignore_errors=True)
        print(f"  removed {models}")


def cmd_uninstall(args: argparse.Namespace) -> int:
    """Stop the daemon, remove all autostart/service integration, and
    (with --purge / --purge-models) delete app data and models."""
    from . import service_ctl
    cfg = load_config(Path(args.config) if args.config else None)

    print("llamanager uninstall — this will:")
    print("  • stop the daemon and remove ALL autostart entries "
          "(service, agents, tasks, tray)")
    if args.purge or args.purge_models:
        print(f"  • delete app data in {cfg.data_dir} "
              "(config, state.db, logs, keys)")
    if args.purge_models:
        print(f"  • delete ALL models in {cfg.models_dir}")
    else:
        print(f"  • KEEP models in {cfg.models_dir}")
    print("  (the Python package itself is removed separately with pip)")

    if not args.yes:
        try:
            reply = input("\nProceed? [y/N]: ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("aborted — nothing changed")
            return 1

    # 1. Stop the daemon (best-effort; service-managed daemons stop here,
    #    a bare `serve` is stopped by the autostart teardown below).
    print("\n[1/4] stopping daemon")
    ok, msg = service_ctl.stop_daemon(cfg)
    print(f"  {msg}")

    # 2. Remove all autostart artifacts (service unit / agents / tasks / tray).
    print("\n[2/4] removing autostart")
    _autostart_off(cfg)

    # 3. Windows service: _autostart_off leaves it intact on purpose, so a
    #    full uninstall removes it here (needs an elevated shell).
    if sys.platform == "win32":
        print("\n[2b] removing Windows service (needs elevation)")
        _run_win_service_module(["stop"])
        _run_win_service_module(["remove"])

    # 4. Purge data / models.
    if args.purge or args.purge_models:
        print("\n[3/4] purging data")
        _purge_data(cfg, include_models=bool(args.purge_models))
    else:
        print("\n[3/4] data kept (pass --purge to delete config/db/logs/keys)")

    print("\n[4/4] remove the package:")
    print("  pip uninstall llamanager")
    print("\nDone.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print runtime.json contents — no need to spin up the server."""
    from . import runtime_state as rt
    cfg = load_config(Path(args.config) if args.config else None)
    state = rt.load(cfg.runtime_path)
    print(json.dumps(state.to_dict(), indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Admin verbs — drive a running daemon over /admin/*. Useful for agents and
# shell scripts that want lifecycle / queue / model control without speaking
# the full HTTP API by hand.
# ---------------------------------------------------------------------------


def _make_admin_client(args: argparse.Namespace):
    """Build an AdminClient from CLI flags + config + env, or print a
    human-friendly error and raise SystemExit on failure."""
    from .admin_client import AdminClient, AdminClientError
    cfg = None
    try:
        cfg = load_config(Path(args.config) if args.config else None)
    except Exception:
        # Config is optional for admin verbs — env vars and flags can carry
        # everything we need. We still try to load it for default URL/key.
        cfg = None
    try:
        return AdminClient.from_config(
            cfg,
            admin_key=getattr(args, "admin_key", None),
            base_url=getattr(args, "url", None),
        )
    except AdminClientError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)


def _emit(payload: Any) -> None:
    """Pretty-print JSON to stdout. Strings pass through unchanged so
    `llamanager logs` etc. can be piped to a pager."""
    if isinstance(payload, str):
        print(payload, end="" if payload.endswith("\n") else "\n")
        return
    print(json.dumps(payload, indent=2, default=str, sort_keys=False))


def _run_admin(fn) -> int:
    """Wrap an admin-verb handler so AdminClientError becomes a clean
    exit code instead of a traceback."""
    from .admin_client import AdminClientError
    try:
        result = fn()
    except AdminClientError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if result is not None:
        _emit(result)
    return 0


def _parse_kv_args(items: list[str] | None) -> dict[str, Any]:
    """Parse `key=value` pairs into a dict. Values are coerced to int/float/
    bool when the literal form matches; everything else stays a string."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"error: --arg expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = _coerce(v)
    return out


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# ---- models ----

def cmd_models_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.models_list())


def cmd_models_pull(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.models_pull(args.source, args.file))


def cmd_models_delete(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.model_delete(args.model_id, force=args.force))


# ---- downloads ----

def cmd_downloads_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.downloads_list())


def cmd_downloads_get(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.download_get(args.download_id))


def cmd_downloads_cancel(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.download_cancel(args.download_id))


# ---- server ----

def cmd_server_status(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.status())


def cmd_server_start(args):
    c = _make_admin_client(args)
    extra = _parse_kv_args(args.arg)
    return _run_admin(lambda: c.server_start(
        profile=args.profile, model=args.model,
        mmproj=args.mmproj, args=extra,
    ))


def cmd_server_stop(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.server_stop())


def cmd_server_restart(args):
    c = _make_admin_client(args)
    extra = _parse_kv_args(args.arg)
    return _run_admin(lambda: c.server_restart(
        profile=args.profile, model=args.model,
        mmproj=args.mmproj, args=extra,
    ))


def cmd_server_swap(args):
    c = _make_admin_client(args)
    extra = _parse_kv_args(args.arg)
    return _run_admin(lambda: c.server_swap(
        profile=args.profile, model=args.model,
        mmproj=args.mmproj, args=extra,
    ))


# ---- queue ----

def cmd_queue_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.queue_list())


def cmd_queue_cancel(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.queue_cancel(args.request_id))


def cmd_queue_pause(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.queue_pause())


def cmd_queue_resume(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.queue_resume())


# ---- origins ----

def cmd_origins_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.origins_list())


def cmd_origins_create(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.origin_create(
        args.name, priority=args.priority,
        allowed_models=args.allowed, is_admin=args.admin,
    ))


def cmd_origins_delete(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.origin_delete(args.origin_id))


def cmd_origins_rotate_key(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.origin_rotate_key(args.origin_id))


# ---- misc ----

def cmd_events(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.events(limit=args.limit))


def cmd_disk(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.disk())


def cmd_reload(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.reload())


def cmd_logs(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.logs(source=args.source, tail=args.tail))


# ---- exclusive mode ----

def cmd_exclusive_status(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.exclusive_status())


def cmd_exclusive_set(args):
    c = _make_admin_client(args)
    kwargs: dict[str, Any] = {}
    if args.mode is not None:
        kwargs["mode"] = args.mode
    if args.grace is not None:
        kwargs["grace_seconds"] = args.grace
    if args.heartbeat is not None:
        kwargs["heartbeat_seconds"] = args.heartbeat
    if not kwargs:
        print("error: nothing to update — pass --mode, --grace, or --heartbeat",
              file=sys.stderr)
        return 2
    return _run_admin(lambda: c.exclusive_set(**kwargs))


def cmd_exclusive_sweep(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.exclusive_sweep())


# ---- slots (multi-slot LLM beta) ----

def cmd_slots_status(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_status())


def cmd_slots_enable(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_set_enabled(True))


def cmd_slots_disable(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_set_enabled(False))


def cmd_slots_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_status())


def cmd_slots_add(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_add())


def cmd_slots_remove(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_remove(args.slot_id))


def cmd_slots_load(args):
    c = _make_admin_client(args)
    extra = _parse_kv_args(args.arg) if args.arg else {}
    return _run_admin(lambda: c.slots_load(
        args.slot_id,
        model=args.model,
        profile=args.profile,
        args=extra or None,
        force=bool(args.force),
    ))


def cmd_slots_unload(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.slots_unload(args.slot_id))


def cmd_slots_coex(args):
    c = _make_admin_client(args)
    allow = args.state == "on"
    return _run_admin(lambda: c.slots_diffusion_coex(allow))


# ---- diffusion ----
#
# Talk to /admin/diffusion/* so the CLI can do what the
# /ui/diffusion-models page does without a browser.

def cmd_diffusion_engines(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_engines())


def cmd_diffusion_install(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_install(
        args.engine, patch_flash_attn=args.patch_flash_attn,
        diffusers_version=(args.diffusers_version or ""),
        reset_diffusers=args.reset_diffusers))


def cmd_diffusion_versions(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_versions(args.engine))


def cmd_diffusion_cancel_install(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_cancel_install(args.engine))


def cmd_diffusion_models(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_models())


def cmd_diffusion_activate(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_activate(args.model_id))


def _parse_field_kv(arg_list: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--field key=value`` flags into a dict.

    Values are passed through verbatim; the server validates against the
    engine's profile_schema and rejects unknown / mistyped fields.
    """
    out: dict[str, Any] = {}
    for entry in (arg_list or []):
        if "=" not in entry:
            raise SystemExit(f"--field expects key=value, got {entry!r}")
        k, _, v = entry.partition("=")
        out[k.strip()] = _coerce_field_value(v.strip())
    return out


def _coerce_field_value(v: str) -> Any:
    """Best-effort coercion: int → float → keep as string. The server's
    schema-driven validation will reject anything the chosen kind
    doesn't accept."""
    if v == "":
        return ""
    try:
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def cmd_diffusion_profiles_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_profiles(args.model_id))


def cmd_diffusion_profile_create(args):
    c = _make_admin_client(args)
    fields = _parse_field_kv(args.field)
    return _run_admin(lambda: c.diffusion_profile_create(
        args.model_id, args.name, fields=fields,
        make_default=args.make_default))


def cmd_diffusion_profile_update(args):
    c = _make_admin_client(args)
    fields = _parse_field_kv(args.field)
    return _run_admin(lambda: c.diffusion_profile_update(
        args.name, args.model_id, fields=fields, new_name=args.rename))


def cmd_diffusion_profile_delete(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_profile_delete(args.name, args.model_id))


def cmd_diffusion_profile_clone(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_profile_clone(
        args.name, args.model_id, args.new_name))


def cmd_diffusion_set_model_default(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_set_model_default_profile(
        args.model_id, profile_name=(args.profile or "")))


def cmd_diffusion_materialize_defaults(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.diffusion_materialize_defaults(
        args.model_id, args.engine))


# ---- ASR (speech-to-text) ----

def cmd_asr_engines(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_engines())


def cmd_asr_install(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_install(engine=args.engine,
                                            torch_backend=args.backend))


def cmd_asr_cancel_install(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_cancel_install(engine=args.engine))


def cmd_asr_setup(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_setup(args.python, engine=args.engine))


def cmd_asr_models(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_models())


def cmd_asr_models_dir(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_models_dir(args.path or ""))


def cmd_asr_defaults(args):
    c = _make_admin_client(args)
    coexist = None if args.coexist is None else (args.coexist == "on")
    return _run_admin(lambda: c.asr_defaults(
        vram_budget_gb=args.vram_budget_gb, coexist=coexist,
        idle_timeout_s=args.idle_timeout_s,
        decode_interval_s=args.decode_interval_s))


def _asr_stream(args, base: str, key: str) -> int:
    """Live streaming client: pipe a local file (paced real-time with ffmpeg
    -re) or stdin (``-``) to the ``/v1/audio/stream`` WebSocket and print the
    revised transcripts as they arrive."""
    import asyncio
    import os
    from urllib.parse import urlencode

    ws_base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    params = {"key": key, "model": args.model, "task": args.task}
    if args.language:
        params["language"] = args.language
    if args.profile:
        params["profile"] = args.profile
    uri = f"{ws_base}/v1/audio/stream?{urlencode(params)}"

    is_stdin = args.file == "-"
    if not is_stdin and not os.path.isfile(os.path.expanduser(args.file)):
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 2

    async def go() -> int:
        import websockets
        if is_stdin:
            ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                  "-f", "wav", "pipe:1"]
            ff_stdin = sys.stdin.buffer
        else:
            ff = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                  "-re", "-i", os.path.expanduser(args.file), "-f", "wav", "pipe:1"]
            ff_stdin = asyncio.subprocess.DEVNULL
        proc = await asyncio.create_subprocess_exec(
            *ff, stdin=ff_stdin, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        try:
            async with websockets.connect(uri, max_size=None) as ws:
                async def send() -> None:
                    assert proc.stdout is not None
                    while True:
                        chunk = await proc.stdout.read(8192)
                        if not chunk:
                            break
                        await ws.send(chunk)
                    await ws.send(json.dumps({"type": "stop"}))

                sender = asyncio.create_task(send())
                final_text = ""
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "transcript":
                        tag = "FINAL" if msg.get("final") else f"rev {msg.get('rev')}"
                        txt = msg.get("text") or " ".join(
                            w["w"] for w in msg.get("words", []))
                        print(f"[{tag} · {msg.get('audio_ms', 0)}ms] {txt}")
                        if msg.get("final"):
                            final_text = txt
                    elif t == "error":
                        print(f"error: {msg.get('error')}", file=sys.stderr)
                        return 1
                    elif t == "done":
                        break
                sender.cancel()
                with contextlib_suppress():
                    await sender
            return 0
        finally:
            with contextlib_suppress(ProcessLookupError):
                if proc.returncode is None:
                    proc.kill()

    try:
        return asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        print(f"error: streaming failed: {e}", file=sys.stderr)
        return 1


def contextlib_suppress(*exc):
    import contextlib
    return contextlib.suppress(*(exc or (Exception,)))


def cmd_asr_transcribe(args):
    """Transcribe a *local* audio file by uploading it to the daemon's
    OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.

    Unlike the other ``asr`` verbs (which drive the admin control plane), this
    is a client operation: it authenticates with a plain **origin** bearer key
    — not an admin key — and uploads the file, so any user on the tailnet with
    a valid key can transcribe against a remote daemon, choosing the model,
    profile, language, and task per request.
    """
    import os
    import httpx
    from .admin_client import resolve_base_url

    cfg = None
    try:
        cfg = load_config(Path(args.config) if args.config else None)
    except Exception:
        cfg = None
    base = resolve_base_url(cfg, getattr(args, "url", None))
    # Bearer key: an ORIGIN key (any enabled origin). Falls back to the admin
    # key resolution because an admin key is also a valid origin.
    key = (getattr(args, "key", None)
           or os.environ.get("LLAMANAGER_API_KEY")
           or getattr(args, "admin_key", None)
           or os.environ.get("LLAMANAGER_ADMIN_KEY"))
    if not key and cfg is not None:
        cli_section = (cfg.raw or {}).get("cli") or {}
        key = cli_section.get("api_key") or cli_section.get("admin_key")
    if not key:
        print("error: no API key. Pass --key (an origin key), set "
              "$LLAMANAGER_API_KEY, or add `api_key`/`admin_key` under [cli] "
              "in config.toml.", file=sys.stderr)
        return 2

    if args.stream:
        return _asr_stream(args, base, key)

    path = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.isfile(path):
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    rf = {"text": "text", "json": "verbose_json", "words": "words"}[args.format]
    data: dict[str, str] = {
        "model": args.model, "response_format": rf, "task": args.task,
    }
    if args.word_timestamps or args.format == "words":
        data["word_timestamps"] = "true"
    if args.language:
        data["language"] = args.language
    if args.profile:
        data["profile"] = args.profile
    try:
        with open(path, "rb") as f:
            r = httpx.post(
                f"{base}/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data=data, files={"file": (os.path.basename(path), f)},
                timeout=1800.0,
            )
    except httpx.HTTPError as e:
        print(f"error: could not reach {base}: {e}", file=sys.stderr)
        return 1
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail") or r.text
        except Exception:
            detail = r.text
        print(f"error: {r.status_code}: {detail}", file=sys.stderr)
        return 1
    if args.format == "text":
        print(r.text)
    else:  # json / words → full envelope
        _emit(r.json())
    return 0


def cmd_asr_profiles_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_profiles(args.model_id))


def cmd_asr_profile_create(args):
    c = _make_admin_client(args)
    fields = _parse_field_kv(args.field)
    return _run_admin(lambda: c.asr_profile_create(
        args.model_id, args.name, fields=fields, make_default=args.make_default))


def cmd_asr_profile_delete(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_profile_delete(args.name, args.model_id))


def cmd_asr_profile_set_default(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.asr_profile_set_default(
        args.model_id, profile_name=(args.profile or "")))


# ---- self-update ----

def cmd_update_check(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.check_update())


def cmd_update(args):
    if args.check:
        return cmd_update_check(args)
    c = _make_admin_client(args)
    from .admin_client import AdminClientError
    try:
        res = c.self_update()
    except AdminClientError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # Print whichever the daemon returned: pip's tail output on
    # success, or the error log if the update failed.
    log = res.get("log") or json.dumps(res, indent=2, default=str)
    if res.get("ok"):
        print(log)
        if res.get("restarting"):
            print("\nDaemon SIGTERM scheduled. Your supervisor will restart it.")
        return 0
    print(log, file=sys.stderr)
    return 1


# ---- LLM profiles ----

def _on_off_or_none(v: str | None) -> bool | None:
    """Parse an ``on|off`` string flag into bool, or None when absent."""
    if v is None:
        return None
    v = v.strip().lower()
    if v in ("on", "true", "yes", "1"):
        return True
    if v in ("off", "false", "no", "0"):
        return False
    raise SystemExit(f"error: expected 'on' or 'off', got {v!r}")


def cmd_profiles_list(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.profiles_list(args.model_id))


def cmd_profile_create(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.profile_create(
        args.model_id, args.name,
        mmproj=args.mmproj or "",
        ctx_size=args.ctx_size,
        vram_limit_gb=(None if args.vram_unlimited
                       else args.vram_limit_gb),
        ram_spill_policy=args.ram_spill_policy or "default",
        ram_spill_limit_gb=args.ram_spill_limit_gb,
        thinking=args.thinking or "",
        args=_parse_kv_args(args.arg),
        make_default=args.make_default,
    ))


def cmd_profile_update(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.profile_update(
        args.name, args.model_id,
        mmproj=args.mmproj,
        ctx_size=args.ctx_size,
        vram_limit_gb=(None if args.vram_unlimited
                       else args.vram_limit_gb),
        ram_spill_policy=args.ram_spill_policy,
        ram_spill_limit_gb=args.ram_spill_limit_gb,
        thinking=args.thinking,
        args=(_parse_kv_args(args.arg) if args.arg is not None else None),
        new_name=args.rename,
    ))


def cmd_profile_delete(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.profile_delete(args.name, args.model_id))


def cmd_profile_clone(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.profile_clone(
        args.name, args.model_id, args.new_name))


def cmd_profile_set_model_default(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.profile_set_model_default(
        args.model_id, profile_name=(args.profile or "")))


# ---- models housekeeping ----

def cmd_models_set_default(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.model_set_default(args.model_id))


def cmd_models_add_existing(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.model_add_existing(args.file_path))


def cmd_models_set_dir(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.models_set_dir(args.models_dir))


# ---- queue extras ----

def cmd_queue_cancel_all(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c._post("/admin/queue/cancel-all"))


# ---- origins extras ----

def cmd_origins_update(args):
    c = _make_admin_client(args)
    is_admin_flag = _on_off_or_none(args.admin)
    return _run_admin(lambda: c.origin_update(
        args.origin_id,
        priority=args.priority,
        allowed_models=args.allowed,
        is_admin=is_admin_flag,
    ))


# ---- setup / config ----

def cmd_setup_llama_binary(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_llama_binary(args.path))


def cmd_setup_hidream(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_hidream(
        python=args.python, repo=args.repo))


def cmd_setup_z_image(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_z_image(args.python))


def cmd_setup_flux2(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_flux2(
        sd_cli=args.sd_cli,
        device_index=args.device_index,
        clear_device_index=args.clear_device_index,
    ))


def cmd_setup_coexistence(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_coexistence(
        unload_text_on_arrival=_on_off_or_none(args.unload_text_on_arrival),
        restart_text_after_image=_on_off_or_none(args.restart_text_after_image),
        allow_concurrent=_on_off_or_none(args.allow_concurrent),
    ))


def cmd_setup_default_args(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_default_args(
        args.engine, _parse_kv_args(args.arg)))


def cmd_setup_autolaunch(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_autolaunch(args.enabled == "on"))


def cmd_setup_autorestart(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_autorestart(args.enabled == "on"))


def cmd_setup_install_llama_server(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_install_llama_server(
        source=args.source, backend=args.backend,
        version=(args.version or "")))


def cmd_setup_engine_versions(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_engine_versions(args.variant))


def cmd_setup_check_updates(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_check_updates(args.variant or ""))


def cmd_setup_install_llama_server_status(args):
    c = _make_admin_client(args)
    return _run_admin(lambda:
                      c.setup_install_llama_server_status(args.variant))


def cmd_setup_switch_variant(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_switch_variant(args.variant))


def cmd_setup_auto_update(args):
    c = _make_admin_client(args)
    action = args.action
    if action == "list":
        return _run_admin(lambda: c.setup_auto_update_list())
    if action == "settings":
        if args.idle_seconds is None and args.check_interval_seconds is None:
            print("error: settings needs --idle-seconds and/or "
                  "--check-interval-seconds", file=sys.stderr)
            return 2
        return _run_admin(lambda: c.setup_auto_update_settings(
            idle_seconds=args.idle_seconds,
            check_interval_seconds=args.check_interval_seconds))
    # Otherwise ``action`` is an engine key and ``state`` is on|off.
    if not args.state:
        print("error: usage: setup auto-update <engine-key> on|off",
              file=sys.stderr)
        return 2
    enabled = _on_off_or_none(args.state)
    return _run_admin(lambda: c.setup_auto_update(action, bool(enabled)))


def _add_admin_flags(p: argparse.ArgumentParser) -> None:
    """Flags shared by every admin verb."""
    p.add_argument("--url", default=None,
                   help="llamanager base URL (default: $LLAMANAGER_URL or "
                        "http://<bind>:<port> from config)")
    p.add_argument("--admin-key", default=None,
                   help="admin bearer token (default: $LLAMANAGER_ADMIN_KEY "
                        "or `[cli].admin_key` in config)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llamanager",
                                description="single-host llama-server manager")
    p.add_argument("--config", help="path to config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="run the daemon in the foreground")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--log-level", default="info")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("tray",
                        help="run the system-tray / menu-bar app "
                             "(thin client; needs the [tray] extra)")
    sp.add_argument("-b", "--background", action="store_true",
                    help="detach and return immediately, instead of blocking "
                         "the terminal with the tray event loop")
    sp.add_argument("--binary", default=None,
                    help="path to the llamanager entrypoint (for --background)")
    sp.set_defaults(func=cmd_tray)

    sp = sub.add_parser("init-config", help="write a default config.toml")
    sp.add_argument("--path", default=None)
    sp.set_defaults(func=cmd_init_config)

    sp = sub.add_parser("status", help="print last persisted runtime state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("install-launchd", help="install macOS LaunchAgent")
    sp.add_argument("--label", default="com.llamanager")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--binary", default=None)
    sp.set_defaults(func=cmd_install_launchd)

    sp = sub.add_parser("install-systemd", help="install Linux user systemd unit")
    sp.add_argument("--unit", default="llamanager.service")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--binary", default=None)
    sp.set_defaults(func=cmd_install_systemd)

    sp = sub.add_parser("install-windows",
                        help="write a Windows Task Scheduler XML "
                             "(no extra deps; user-logon trigger)")
    sp.add_argument("--task", default="llamanager")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--binary", default=None)
    sp.set_defaults(func=cmd_install_windows)

    sp = sub.add_parser("install-windows-service",
                        help="register llamanager as a real Windows "
                             "service (requires pywin32 + admin)")
    sp.add_argument("--startup", default="auto",
                    choices=["auto", "manual", "delayed", "disabled"],
                    help="service start type (default: auto)")
    sp.add_argument("--username", default=None,
                    help="run service as this account "
                         "(default: LocalSystem)")
    sp.add_argument("--password", default=None,
                    help="password for --username")
    sp.add_argument("--no-start", dest="start", action="store_false",
                    default=True,
                    help="don't start the service after installing")
    sp.set_defaults(func=cmd_install_windows_service)

    sp = sub.add_parser("remove-windows-service",
                        help="stop and unregister the Windows service")
    sp.set_defaults(func=cmd_remove_windows_service)

    def _add_autostart_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--binary", default=None,
                       help="path to the llamanager entrypoint (default: "
                            "autodetect)")
        p.add_argument("--no-start", action="store_true",
                       help="install but don't start the daemon/tray now")
        p.add_argument("--write-only", action="store_true",
                       help="only write units/plists/XML; skip enable/start "
                            "(print the manual commands instead)")
        p.add_argument("--pre-login", action="store_true",
                       help="macOS only: install the daemon as a system "
                            "LaunchDaemon (runs before login; needs sudo, and "
                            "tray daemon-controls then require sudo too)")
        p.add_argument("--username", default=None,
                       help="Windows only: run the service as this account "
                            "and grant it control")
        p.add_argument("--password", default=None,
                       help="Windows only: password for --username")

    sp = sub.add_parser("autostart",
                        help="configure how llamanager runs at boot/login "
                             "(the single front door for OS integration)")
    sp.add_argument("--mode", required=True,
                    choices=["off", "boot-service", "login-tray", "tray+service"],
                    help="off=tear down; boot-service=always-on headless; "
                         "login-tray=daemon+tray at login; "
                         "tray+service=always-on daemon + login tray "
                         "(recommended)")
    _add_autostart_flags(sp)
    sp.set_defaults(func=cmd_autostart)

    # Back-compat aliases for the original commands.
    sp = sub.add_parser("install-tray",
                        help="alias for `autostart --mode tray+service`")
    _add_autostart_flags(sp)
    sp.set_defaults(func=cmd_install_tray)

    sp = sub.add_parser("remove-tray",
                        help="alias for `autostart --mode off`")
    sp.set_defaults(func=cmd_remove_tray)

    sp = sub.add_parser("uninstall",
                        help="stop the daemon + remove all autostart/service "
                             "integration; --purge also deletes data")
    sp.add_argument("--purge", action="store_true",
                    help="also delete app data (config, state.db, logs, keys) "
                         "in data_dir — keeps models")
    sp.add_argument("--purge-models", action="store_true",
                    help="also delete the models dir (implies data purge)")
    sp.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    sp.set_defaults(func=cmd_uninstall)

    sp = sub.add_parser("init",
                        help="guided first-run setup: config + admin key, then "
                             "starts the daemon + tray now (no restart). "
                             "Run-at-startup and engine install live in the UI.")
    sp.add_argument("--binary", default=None,
                    help="path to the llamanager entrypoint (default: autodetect)")
    sp.add_argument("--no-launch", action="store_true",
                    help="don't start the daemon/tray now — just write config "
                         "and the admin key")
    sp.set_defaults(func=cmd_init)

    # ---- admin verbs (drive a running daemon) ----

    # models
    msp = sub.add_parser("models", help="manage downloaded models").add_subparsers(
        dest="models_cmd", required=True)
    sp = msp.add_parser("list", help="list models on disk")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_models_list)
    sp = msp.add_parser("pull", help="start a HuggingFace GGUF download")
    sp.add_argument("source",
                    help="hf://<user>/<repo> or <user>/<repo>")
    sp.add_argument("--file", action="append", default=None,
                    help="specific file inside the repo (may be repeated)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_models_pull)
    sp = msp.add_parser("delete", help="delete a model from disk")
    sp.add_argument("model_id", help="<repo>/<file>.gguf")
    sp.add_argument("--force", action="store_true",
                    help="stop llama-server first if this model is loaded")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_models_delete)

    # downloads
    dsp = sub.add_parser("downloads", help="track in-progress model pulls").add_subparsers(
        dest="downloads_cmd", required=True)
    sp = dsp.add_parser("list", help="list downloads (running and historical)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_downloads_list)
    sp = dsp.add_parser("get", help="show one download")
    sp.add_argument("download_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_downloads_get)
    sp = dsp.add_parser("cancel", help="cancel a running download")
    sp.add_argument("download_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_downloads_cancel)

    # server
    ssp = sub.add_parser("server", help="control the upstream llama-server").add_subparsers(
        dest="server_cmd", required=True)
    sp = ssp.add_parser("status", help="full daemon status (queue, runtime, model)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_server_status)

    def _server_lifecycle(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--profile", default=None)
        parser.add_argument("--model", default=None)
        parser.add_argument("--mmproj", default=None)
        parser.add_argument("--arg", action="append", default=None,
                            metavar="KEY=VALUE",
                            help="extra llama-server flag override (repeatable)")
        _add_admin_flags(parser)

    sp = ssp.add_parser("start", help="start llama-server")
    _server_lifecycle(sp); sp.set_defaults(func=cmd_server_start)
    sp = ssp.add_parser("stop", help="stop llama-server (drains queue)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_server_stop)
    sp = ssp.add_parser("restart", help="restart llama-server")
    _server_lifecycle(sp); sp.set_defaults(func=cmd_server_restart)
    sp = ssp.add_parser("swap", help="hot-swap to a different model/profile")
    _server_lifecycle(sp); sp.set_defaults(func=cmd_server_swap)

    # queue
    qsp = sub.add_parser("queue", help="manage the request queue").add_subparsers(
        dest="queue_cmd", required=True)
    sp = qsp.add_parser("list", help="snapshot of pending and in-flight requests")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_queue_list)
    sp = qsp.add_parser("cancel", help="cancel a queued or in-flight request")
    sp.add_argument("request_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_queue_cancel)
    sp = qsp.add_parser("pause", help="stop dispatching new requests")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_queue_pause)
    sp = qsp.add_parser("resume", help="resume after pause")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_queue_resume)

    # origins
    osp = sub.add_parser("origins", help="manage API-key origins").add_subparsers(
        dest="origins_cmd", required=True)
    sp = osp.add_parser("list", help="list origins (no key material)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_origins_list)
    sp = osp.add_parser("create", help="create an origin and print its api key")
    sp.add_argument("name")
    sp.add_argument("--priority", type=int, default=None)
    sp.add_argument("--allowed", action="append", default=None,
                    help="allowed model id (repeatable; '*' for any)")
    sp.add_argument("--admin", action="store_true",
                    help="grant admin scope")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_origins_create)
    sp = osp.add_parser("delete", help="delete an origin by id")
    sp.add_argument("origin_id", type=int)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_origins_delete)
    sp = osp.add_parser("rotate-key", help="mint a new api key for an origin")
    sp.add_argument("origin_id", type=int)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_origins_rotate_key)

    # misc admin verbs
    sp = sub.add_parser("events", help="recent server events")
    sp.add_argument("--limit", type=int, default=200)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_events)

    sp = sub.add_parser("disk", help="models-dir disk usage")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_disk)

    sp = sub.add_parser("reload", help="reload config.toml in-place")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_reload)

    sp = sub.add_parser("logs", help="tail the activity stream (default) or raw logs")
    sp.add_argument("--source", default="activity",
                    help="activity | llama-server | llamanager | <engine name> "
                         "(any *.log file in logs_dir is acceptable)")
    sp.add_argument("--tail", type=int, default=200)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_logs)

    # diffusion
    dfp = sub.add_parser("diffusion",
                         help="manage diffusion engines + image models + profiles"
                         ).add_subparsers(dest="diffusion_cmd", required=True)

    sp = dfp.add_parser("engines", help="per-engine install state + GPU detection")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_engines)

    sp = dfp.add_parser("install", help="kick the auto-installer for an engine")
    sp.add_argument("engine", help="hidream | z_image | flux2")
    sp.add_argument("--patch-flash-attn", action="store_true",
                    help="apply the use_flash_attn=False patch to "
                         "hidream-source/pipeline.py (recommended on AMD)")
    sp.add_argument("--diffusers-version", default="", dest="diffusers_version",
                    help="install a specific diffusers version (upgrade or "
                         "downgrade). Overrides the tested pin and persists as "
                         "the auto-update target. See `diffusion versions`.")
    sp.add_argument("--reset-diffusers", action="store_true", dest="reset_diffusers",
                    help="clear any diffusers override and reinstall the "
                         "tested pin")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_install)

    sp = dfp.add_parser("versions",
                        help="list installable diffusers versions + the "
                             "engine's installed/target version")
    sp.add_argument("engine", help="hidream | z_image")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_versions)

    sp = dfp.add_parser("cancel-install", help="cancel an in-progress install")
    sp.add_argument("engine")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_cancel_install)

    sp = dfp.add_parser("models",
                        help="list installed image models + catalog of installable ones")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_models)

    sp = dfp.add_parser("activate",
                        help="set this image model as the dashboard/API default")
    sp.add_argument("model_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_activate)

    # diffusion profiles (nested under diffusion to keep verbs short)
    pfp = dfp.add_parser("profiles", help="manage per-model diffusion profiles"
                         ).add_subparsers(dest="profiles_cmd", required=True)

    sp = pfp.add_parser("list", help="show profiles + built-in defaults for a model")
    sp.add_argument("model_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_profiles_list)

    sp = pfp.add_parser("create", help="create a new profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    sp.add_argument("--field", action="append", default=None,
                    metavar="KEY=VALUE",
                    help="schema field value (repeatable, e.g. --field image_steps=50)")
    sp.add_argument("--make-default", action="store_true",
                    help="also set the new profile as this model's default")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_profile_create)

    sp = pfp.add_parser("update", help="patch fields on an existing profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    sp.add_argument("--field", action="append", default=None,
                    metavar="KEY=VALUE")
    sp.add_argument("--rename", default=None,
                    help="optionally rename the profile in the same call")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_profile_update)

    sp = pfp.add_parser("delete", help="delete a profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_profile_delete)

    sp = pfp.add_parser("clone", help="duplicate a profile under a new name")
    sp.add_argument("model_id"); sp.add_argument("name"); sp.add_argument("new_name")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_profile_clone)

    sp = pfp.add_parser("set-default",
                        help="set the per-model default profile (blank to clear)")
    sp.add_argument("model_id")
    sp.add_argument("--profile", default="")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_set_model_default)

    sp = pfp.add_parser("materialize-defaults",
                        help="copy the engine's built-in defaults into config.toml")
    sp.add_argument("model_id"); sp.add_argument("engine")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_materialize_defaults)

    # ---- asr (speech-to-text) ----
    afp = sub.add_parser(
        "asr",
        help="speech-to-text: transcribe audio, manage the ASR engine + models"
    ).add_subparsers(dest="asr_cmd", required=True)

    sp = afp.add_parser("transcribe",
                        help="transcribe a local audio file (uploads to the daemon "
                             "with an origin key; works over the tailnet)")
    sp.add_argument("file", help="path to a LOCAL audio file (wav/mp3/m4a/ogg/flac/…)")
    sp.add_argument("--model", required=True, help="audio-family model id")
    sp.add_argument("--language", default="", help="ISO hint (e.g. ar); blank = auto-detect")
    sp.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    sp.add_argument("--profile", default="", help="audio profile name")
    sp.add_argument("--format", default="text", choices=["text", "json", "words"],
                    help="text = transcription only (default); json = full result; "
                         "words = word-level {w,t0,t1,p} envelope")
    sp.add_argument("--word-timestamps", action="store_true",
                    help="include per-word timing + confidence")
    sp.add_argument("--stream", action="store_true",
                    help="live streaming over WebSocket: pace the file real-time "
                         "(or read '-' from stdin) and print revised partials")
    sp.add_argument("--key", default=None,
                    help="origin bearer token (default: $LLAMANAGER_API_KEY, "
                         "[cli].api_key, or the admin key). Any enabled origin works.")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_transcribe)

    sp = afp.add_parser("engines",
                        help="list ASR engines and their status "
                             "(available? configured? installed?)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_engines)

    sp = afp.add_parser("install",
                        help="install/build a chosen ASR engine's dependencies")
    sp.add_argument("--engine", default="asr", choices=AUDIO_ENGINE_CHOICES,
                    help="which engine to install: asr (Whisper/transformers), "
                         "whispercpp (whisper.cpp, built with Vulkan), or "
                         "sherpa (sherpa-onnx streaming). Default: asr")
    sp.add_argument("--backend", default="auto",
                    choices=["auto", "rocm", "cuda", "cpu"],
                    help="torch build for the dedicated-venv fallback "
                         "(asr only; ignored by whispercpp/sherpa)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_install)

    sp = afp.add_parser("cancel-install",
                        help="cancel an in-progress ASR install/build")
    sp.add_argument("--engine", default="asr", choices=AUDIO_ENGINE_CHOICES,
                    help="which engine's install to cancel (default: asr)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_cancel_install)

    sp = afp.add_parser("setup",
                        help="point a chosen ASR engine at its interpreter "
                             "(asr/sherpa) or built binary (whispercpp)")
    sp.add_argument("python",
                    help="path: a python with the engine's deps, or the "
                         "whisper-cli binary for whispercpp")
    sp.add_argument("--engine", default="asr", choices=AUDIO_ENGINE_CHOICES,
                    help="which engine to point (default: asr)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_setup)

    sp = afp.add_parser("models", help="list installed speech-to-text models")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_models)

    sp = afp.add_parser("models-dir",
                        help="set the dedicated ASR models folder (blank = shared LLM models dir)")
    sp.add_argument("path", nargs="?", default="",
                    help="folder to scan for Whisper models; omit to revert to the shared dir")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_models_dir)

    sp = afp.add_parser("defaults",
                        help="set ASR service defaults: VRAM budget, coexistence, "
                             "idle timeout, streaming decode cadence")
    sp.add_argument("--vram-budget-gb", type=float, default=None,
                    help="cap ASR VRAM (GB); admits concurrent tasks under it (0 = uncapped)")
    sp.add_argument("--coexist", choices=["on", "off"], default=None,
                    help="on = run ASR alongside the LLM; off = unload the LLM per task")
    sp.add_argument("--idle-timeout-s", type=int, default=None,
                    help="stop the warm worker after N idle seconds (0 = never)")
    sp.add_argument("--decode-interval-s", type=float, default=None,
                    help="streaming decode cadence (live WebSocket mode)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_defaults)

    pfp = afp.add_parser("profiles", help="manage per-model ASR profiles"
                         ).add_subparsers(dest="asr_profiles_cmd", required=True)

    sp = pfp.add_parser("list", help="show profiles + built-in defaults for a model")
    sp.add_argument("model_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_profiles_list)

    sp = pfp.add_parser("create", help="create a new audio profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    sp.add_argument("--field", action="append", default=None, metavar="KEY=VALUE",
                    help="schema field (repeatable, e.g. --field audio_language=ar)")
    sp.add_argument("--make-default", action="store_true",
                    help="also set the new profile as this model's default")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_profile_create)

    sp = pfp.add_parser("delete", help="delete an audio profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_profile_delete)

    sp = pfp.add_parser("set-default",
                        help="set the per-model default audio profile (blank to clear)")
    sp.add_argument("model_id")
    sp.add_argument("--profile", default="")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_asr_profile_set_default)

    # profiles (LLM-side, mirror of `diffusion profiles`)
    pfp = sub.add_parser("profiles",
                         help="manage per-LLM-model profiles "
                              "(mmproj, ctx_size, vram, ram-spill, thinking, args)"
                         ).add_subparsers(dest="profiles_cmd", required=True)

    def _add_llm_profile_fields(parser: argparse.ArgumentParser, *,
                                require_make_default: bool) -> None:
        parser.add_argument("--mmproj", default=None,
                            help="path to mmproj .gguf for vision models")
        parser.add_argument("--ctx-size", type=int, default=None,
                            dest="ctx_size",
                            help="context length the engine launches with")
        parser.add_argument("--vram-limit-gb", type=float, default=None,
                            dest="vram_limit_gb",
                            help="cap on VRAM usage (GB); omit for unlimited")
        parser.add_argument("--vram-unlimited", action="store_true",
                            dest="vram_unlimited",
                            help="explicitly mark VRAM unlimited (overrides --vram-limit-gb)")
        parser.add_argument("--ram-spill-policy", default=None,
                            dest="ram_spill_policy",
                            choices=["default", "ram_only", "limited"],
                            help="RAM-spill behaviour when VRAM is full")
        parser.add_argument("--ram-spill-limit-gb", type=float, default=None,
                            dest="ram_spill_limit_gb",
                            help="cap on RAM-spill GB (only with policy=limited)")
        parser.add_argument("--thinking", default=None,
                            choices=["on", "off", ""],
                            help="force thinking on/off (overrides template)")
        parser.add_argument("--arg", action="append", default=None,
                            metavar="KEY=VALUE",
                            help="extra llama-server arg (repeatable). "
                                 "Replaces the whole args bucket on update.")
        if require_make_default:
            parser.add_argument("--make-default", action="store_true",
                                help="also set the new profile as this model's default")

    sp = pfp.add_parser("list", help="show LLM profiles for a model")
    sp.add_argument("model_id")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_profiles_list)

    sp = pfp.add_parser("create", help="create a new LLM profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    _add_llm_profile_fields(sp, require_make_default=True)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_profile_create)

    sp = pfp.add_parser("update", help="patch fields on an existing LLM profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    _add_llm_profile_fields(sp, require_make_default=False)
    sp.add_argument("--rename", default=None,
                    help="optionally rename the profile in the same call")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_profile_update)

    sp = pfp.add_parser("delete", help="delete an LLM profile")
    sp.add_argument("model_id"); sp.add_argument("name")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_profile_delete)

    sp = pfp.add_parser("clone", help="duplicate an LLM profile under a new name")
    sp.add_argument("model_id"); sp.add_argument("name"); sp.add_argument("new_name")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_profile_clone)

    sp = pfp.add_parser("set-default",
                        help="set the per-model default LLM profile (empty to clear)")
    sp.add_argument("model_id"); sp.add_argument("--profile", default="")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_profile_set_model_default)

    # New verbs are added to the existing `models` / `queue` / `origins`
    # subparser groups by reaching into their already-built parser via
    # ``sub.choices[name]``. argparse doesn't expose the inner
    # _SubParsersAction except via _actions, so the helper below walks
    # it. ``sub`` itself is a _SubParsersAction whose .choices maps
    # top-level subcommand names to their ArgumentParser instances.
    def _inner_subparsers(parser_name: str) -> argparse._SubParsersAction | None:
        parser = sub.choices.get(parser_name)
        if parser is None:
            return None
        return next(
            (a for a in parser._actions
             if isinstance(a, argparse._SubParsersAction)),
            None,
        )

    models_sub = _inner_subparsers("models")
    if models_sub is not None:
        sp = models_sub.add_parser("set-default",
                                   help="set the configured default LLM model")
        sp.add_argument("model_id")
        _add_admin_flags(sp); sp.set_defaults(func=cmd_models_set_default)

        sp = models_sub.add_parser("add-existing",
                                   help="register an existing .gguf file "
                                        "(symlinks/copies it into models_dir)")
        sp.add_argument("file_path")
        _add_admin_flags(sp); sp.set_defaults(func=cmd_models_add_existing)

        sp = models_sub.add_parser("set-dir",
                                   help="change models_dir at runtime (and persist)")
        sp.add_argument("models_dir")
        _add_admin_flags(sp); sp.set_defaults(func=cmd_models_set_dir)

    queue_sub = _inner_subparsers("queue")
    if queue_sub is not None:
        sp = queue_sub.add_parser("cancel-all",
                                  help="cancel every queued + in-flight request")
        _add_admin_flags(sp); sp.set_defaults(func=cmd_queue_cancel_all)

    origins_sub = _inner_subparsers("origins")
    if origins_sub is not None:
        sp = origins_sub.add_parser("update",
                                    help="patch priority / allowed_models / "
                                         "admin scope on an existing origin")
        sp.add_argument("origin_id", type=int)
        sp.add_argument("--priority", type=int, default=None)
        sp.add_argument("--allowed", action="append", default=None,
                        help="allowed model id (repeatable; pass once "
                             "with '*' for all)")
        sp.add_argument("--admin", default=None, choices=["on", "off"],
                        help="grant or revoke admin scope")
        _add_admin_flags(sp); sp.set_defaults(func=cmd_origins_update)

    # setup / config (paths, coexistence, default-args, autolaunch,
    # autorestart, llama-server installer + variant switch).
    setup = sub.add_parser("setup",
                           help="manage daemon paths, coexistence, autolaunch, "
                                "and the llama-server installer"
                           ).add_subparsers(dest="setup_cmd", required=True)

    sp = setup.add_parser("llama-binary",
                          help="set the llama-server binary path")
    sp.add_argument("path")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_llama_binary)

    sp = setup.add_parser("hidream",
                          help="set hidream Python and/or source folder")
    sp.add_argument("--python", default=None,
                    help="path to the hidream venv's python interpreter")
    sp.add_argument("--repo", default=None,
                    help="path to the HiDream-O1-Image source folder")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_hidream)

    sp = setup.add_parser("z-image", help="set z_image Python interpreter")
    sp.add_argument("python")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_z_image)

    sp = setup.add_parser("flux2",
                          help="set flux2 sd-cli path and/or device index")
    sp.add_argument("--sd-cli", default=None, dest="sd_cli")
    sp.add_argument("--device-index", type=int, default=None,
                    dest="device_index")
    sp.add_argument("--clear-device-index", action="store_true",
                    dest="clear_device_index",
                    help="remove the device_index setting (let sd-cli auto-pick)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_flux2)

    sp = setup.add_parser("coexistence",
                          help="toggle text<->image coexistence policy flags")
    sp.add_argument("--unload-text-on-arrival", choices=["on", "off"],
                    default=None, dest="unload_text_on_arrival")
    sp.add_argument("--restart-text-after-image", choices=["on", "off"],
                    default=None, dest="restart_text_after_image")
    sp.add_argument("--allow-concurrent", choices=["on", "off"],
                    default=None, dest="allow_concurrent")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_coexistence)

    sp = setup.add_parser("default-args",
                          help="replace [default_args.<engine>] in config.toml")
    sp.add_argument("engine", choices=["llama", "mlx"])
    sp.add_argument("--arg", action="append", default=None,
                    metavar="KEY=VALUE", required=True,
                    help="default arg (repeatable). Pass with no --arg "
                         "to clear via an empty list isn't supported here "
                         "— edit config.toml directly to wipe the bucket.")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_default_args)

    sp = setup.add_parser("autolaunch",
                          help="enable/disable autolaunch of the default LLM "
                               "on daemon startup")
    sp.add_argument("enabled", choices=["on", "off"])
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_autolaunch)

    sp = setup.add_parser("autorestart",
                          help="enable/disable the crash supervisor's "
                               "auto-restart (in-memory only)")
    sp.add_argument("enabled", choices=["on", "off"])
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_autorestart)

    sp = setup.add_parser("install-llama-server",
                          help="kick the llama-server variant installer "
                               "(asynchronous; poll status separately)")
    sp.add_argument("--source", default="llama.cpp")
    sp.add_argument("--backend", default="",
                    help="backend variant; leave empty to auto-pick "
                         "(e.g. cuda, vulkan, metal, cpu)")
    sp.add_argument("--version", default="",
                    help="install a specific upstream version (GitHub release "
                         "tag, or mlx-lm PyPI version) — upgrade or downgrade. "
                         "Omit for latest. See `setup engine-versions`.")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_install_llama_server)

    sp = setup.add_parser("engine-versions",
                          help="list installable versions for an LLM variant "
                               "(newest first)")
    sp.add_argument("variant", help="variant id, e.g. llama.cpp-cuda")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_engine_versions)

    sp = setup.add_parser("check-updates",
                          help="check upstream for a newer build of one variant "
                               "(--variant) or every installed variant")
    sp.add_argument("--variant", default="",
                    help="variant id, e.g. llama.cpp-cuda; omit to check all "
                         "installed variants")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_check_updates)

    sp = setup.add_parser("install-llama-server-status",
                          help="poll an in-flight or completed llama-server install")
    sp.add_argument("variant", help="variant id from install-llama-server")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_install_llama_server_status)

    sp = setup.add_parser("switch-variant",
                          help="switch the active llama-server to a previously-installed variant")
    sp.add_argument("variant")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_switch_variant)

    sp = setup.add_parser(
        "auto-update",
        help="manage auto-update-when-idle per engine "
             "(llama variant id / diffusion engine / 'llamanager')")
    sp.add_argument(
        "action",
        help="'list', 'settings', or an engine key to toggle "
             "(e.g. llama.cpp-cuda, z_image, llamanager)")
    sp.add_argument(
        "state", nargs="?",
        help="on|off when toggling an engine key (omit for list/settings)")
    sp.add_argument("--idle-seconds", type=int, default=None, dest="idle_seconds",
                    help="settings: quiet window before an update fires")
    sp.add_argument("--check-interval-seconds", type=int, default=None,
                    dest="check_interval_seconds",
                    help="settings: how often to check upstream (seconds)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_auto_update)

    # self-update — same code path as the /ui/about Update button.
    sp = sub.add_parser("update",
                        help="pull + reinstall llamanager + restart "
                             "(same as the /ui/about Update button)")
    sp.add_argument("--check", action="store_true",
                    help="only check for a newer release, don't update")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_update)

    # exclusive — host-wide process isolation toggle.
    esp = sub.add_parser(
        "exclusive",
        help=(
            "manage exclusive mode (kill foreign llama-server / "
            "image-engine workers so llamanager owns the GPU)"
        ),
    ).add_subparsers(dest="exclusive_cmd", required=True)

    sp = esp.add_parser("status",
                        help="current mode + last sweep result")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_exclusive_status)

    sp = esp.add_parser("set",
                        help="set the mode and/or grace/heartbeat seconds")
    sp.add_argument("--mode", choices=["off", "warn", "exclusive", "aggressive"],
                    default=None,
                    help="off=disabled, warn=scan-only, "
                         "exclusive=kill llama-server + workers, "
                         "aggressive=also kill ComfyUI/vLLM/Ollama/etc.")
    sp.add_argument("--grace", type=float, default=None,
                    help="SIGTERM → SIGKILL grace window in seconds (default 5)")
    sp.add_argument("--heartbeat", type=int, default=None,
                    help="background re-sweep interval in seconds (default 120)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_exclusive_set)

    sp = esp.add_parser("sweep",
                        help="run one sweep now (warn-scan if mode is off)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_exclusive_sweep)

    # slots — multi-slot LLM (beta). When the master switch is on,
    # multiple llama-servers run in parallel, each holding a different
    # model. Routing is by model id; cold-cache requests are rejected.
    ssp = sub.add_parser(
        "slots",
        help="manage the multi-slot LLM beta (parallel models on their "
             "own ports; routing by model id)",
    ).add_subparsers(dest="slots_cmd", required=True)

    sp = ssp.add_parser("status",
                        help="enabled flag + per-slot dashboard")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_status)

    sp = ssp.add_parser("list",
                        help="alias for `slots status`")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_list)

    sp = ssp.add_parser("enable",
                        help="turn on the multi-slot beta (force-disables "
                             "exclusive mode)")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_enable)

    sp = ssp.add_parser("disable",
                        help="turn off — drains and stops slots 1..N")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_disable)

    sp = ssp.add_parser("add",
                        help="allocate the next free slot id + port")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_add)

    sp = ssp.add_parser("remove",
                        help="stop and remove a slot (slot 0 cannot be removed)")
    sp.add_argument("slot_id", type=int)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_remove)

    sp = ssp.add_parser("load",
                        help="load (or swap) a model into a specific slot")
    sp.add_argument("slot_id", type=int)
    sp.add_argument("--model", required=True,
                    help="model id (e.g. org/repo-GGUF/Q4_K_M.gguf)")
    sp.add_argument("--profile", default=None,
                    help="profile bound to that model (optional)")
    sp.add_argument("--arg", action="append", default=None,
                    metavar="KEY=VALUE",
                    help="extra llama-server arg override (repeatable)")
    sp.add_argument("--force", action="store_true",
                    help="bypass the VRAM admission warning")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_load)

    sp = ssp.add_parser("unload",
                        help="stop the model running in a specific slot")
    sp.add_argument("slot_id", type=int)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_unload)

    sp = ssp.add_parser("coex",
                        help="diffusion-coexistence policy "
                             "(on = image task keeps LLM slots loaded)")
    sp.add_argument("state", choices=["on", "off"])
    _add_admin_flags(sp); sp.set_defaults(func=cmd_slots_coex)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
