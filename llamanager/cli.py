"""CLI entrypoint."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .config import expand, load_config, write_default_config


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .app import create_app
    from .config import expand

    cfg_path = Path(args.config) if args.config else None
    app = create_app(cfg_path)
    cfg = app.state.cfg

    ssl_certfile = args.ssl_certfile or cfg.ssl_certfile or None
    ssl_keyfile = args.ssl_keyfile or cfg.ssl_keyfile or None

    # Auto-generate a locally-trusted cert via mkcert if the configured
    # cert files don't exist yet (first run).
    if ssl_certfile and ssl_keyfile:
        cert_path = expand(ssl_certfile)
        key_path = expand(ssl_keyfile)
        if not cert_path.exists() or not key_path.exists():
            if _auto_generate_cert(cert_path, key_path):
                print(f"  certificate: {cert_path}")
                print(f"  private key: {key_path}")
                print()
            else:
                ssl_certfile = None
                ssl_keyfile = None
        ssl_certfile = str(cert_path) if ssl_certfile else None
        ssl_keyfile = str(key_path) if ssl_keyfile else None

    uvicorn.run(app, host=args.host or cfg.bind, port=args.port or cfg.port,
                log_level=args.log_level,
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
    return 0


def _auto_generate_cert(cert_path: Path, key_path: Path,
                        extra_hosts: list[str] | None = None) -> bool:
    """Generate a locally-trusted cert using mkcert. Returns True on success."""
    import subprocess
    import shutil
    if not shutil.which("mkcert"):
        print("  mkcert is not installed. Install it for automatic HTTPS:",
              file=sys.stderr)
        print("    brew install mkcert    (macOS)", file=sys.stderr)
        print("    choco install mkcert   (Windows)", file=sys.stderr)
        print("    apt install mkcert     (Debian/Ubuntu)", file=sys.stderr)
        print("  Then run: mkcert -install", file=sys.stderr)
        print("  Falling back to HTTP.", file=sys.stderr)
        print()
        return False
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    hosts = ["localhost", "127.0.0.1", "::1"] + (extra_hosts or [])
    try:
        # Try to install the local CA (needs sudo the first time).
        # If it fails (no tty for sudo), the cert will still be created
        # but won't be trusted until the user runs `mkcert -install`.
        ca_result = subprocess.run(["mkcert", "-install"], capture_output=True)
        subprocess.run([
            "mkcert",
            "-cert-file", str(cert_path),
            "-key-file", str(key_path),
            *hosts,
        ], check=True, capture_output=True)
        import os
        os.chmod(key_path, 0o600)
        if ca_result.returncode != 0:
            print("  created TLS certificate via mkcert")
            print("  note: run `mkcert -install` in a terminal to trust it "
                  "(needs your password once)")
        else:
            print("  created locally-trusted TLS certificate via mkcert")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  mkcert failed: {e}", file=sys.stderr)
        print("  Falling back to HTTP.", file=sys.stderr)
        return False


def cmd_init_config(args: argparse.Namespace) -> int:
    target = expand(args.path) if args.path else expand("~/.llamanager/config.toml")
    out = write_default_config(target)
    print(f"wrote default config to {out}")
    return 0


def cmd_install_launchd(args: argparse.Namespace) -> int:
    from .installer import install_launchd
    plist_path = install_launchd(label=args.label, port=args.port,
                                 binary=args.binary)
    print(f"installed LaunchAgent at {plist_path}")
    print("To load now: `launchctl load -w` followed by the plist path.")
    return 0


def cmd_install_systemd(args: argparse.Namespace) -> int:
    from .installer import install_systemd
    unit_path = install_systemd(unit_name=args.unit, port=args.port,
                                binary=args.binary)
    print(f"installed user systemd unit at {unit_path}")
    print("To enable: `systemctl --user daemon-reload && "
          "systemctl --user enable --now llamanager.service`.")
    return 0


def cmd_install_windows(args: argparse.Namespace) -> int:
    from .installer import install_windows_task
    xml_path = install_windows_task(task_name=args.task, port=args.port,
                                    binary=args.binary)
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


def cmd_generate_cert(args: argparse.Namespace) -> int:
    """Generate or regenerate the locally-trusted TLS certificate via mkcert."""
    cfg = load_config(Path(args.config) if args.config else None)
    certfile = expand(cfg.ssl_certfile) if cfg.ssl_certfile else cfg.data_dir / "tls" / "cert.pem"
    keyfile = expand(cfg.ssl_keyfile) if cfg.ssl_keyfile else cfg.data_dir / "tls" / "key.pem"
    if certfile.exists() and keyfile.exists() and not args.force:
        print(f"certificate already exists at {certfile}")
        print("use --force to overwrite")
        return 1
    if _auto_generate_cert(certfile, keyfile, extra_hosts=args.host):
        return 0
    return 2


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
    sp.add_argument("--ssl-certfile", default=None,
                    help="path to TLS certificate (PEM)")
    sp.add_argument("--ssl-keyfile", default=None,
                    help="path to TLS private key (PEM)")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("init-config", help="write a default config.toml")
    sp.add_argument("--path", default=None)
    sp.set_defaults(func=cmd_init_config)

    sp = sub.add_parser("generate-cert",
                        help="generate a locally-trusted TLS certificate via mkcert")
    sp.add_argument("--force", action="store_true",
                    help="overwrite existing certificate")
    sp.add_argument("--host", action="append", default=None,
                    help="extra hostname or IP to include (repeatable)")
    sp.set_defaults(func=cmd_generate_cert)

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

    sp = sub.add_parser("logs", help="tail llama-server or llamanager logs")
    sp.add_argument("--source", default="llama-server",
                    choices=["llama-server", "llamanager"])
    sp.add_argument("--tail", type=int, default=200)
    _add_admin_flags(sp); sp.set_defaults(func=cmd_logs)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
