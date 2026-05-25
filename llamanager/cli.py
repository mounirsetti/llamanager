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
        args.engine, patch_flash_attn=args.patch_flash_attn))


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
    # Print whichever the daemon returned. For success: pip's tail
    # output. For editable refusal: the multi-line manual-update
    # instructions. Either way the operator gets actionable text.
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
        source=args.source, backend=args.backend))


def cmd_setup_install_llama_server_status(args):
    c = _make_admin_client(args)
    return _run_admin(lambda:
                      c.setup_install_llama_server_status(args.variant))


def cmd_setup_switch_variant(args):
    c = _make_admin_client(args)
    return _run_admin(lambda: c.setup_switch_variant(args.variant))


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
    _add_admin_flags(sp); sp.set_defaults(func=cmd_diffusion_install)

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
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_install_llama_server)

    sp = setup.add_parser("install-llama-server-status",
                          help="poll an in-flight or completed llama-server install")
    sp.add_argument("variant", help="variant id from install-llama-server")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_install_llama_server_status)

    sp = setup.add_parser("switch-variant",
                          help="switch the active llama-server to a previously-installed variant")
    sp.add_argument("variant")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_setup_switch_variant)

    # self-update — same code path as the /ui/about Update button.
    sp = sub.add_parser("update",
                        help="pull + reinstall llamanager + restart "
                             "(same as the /ui/about Update button)")
    sp.add_argument("--check", action="store_true",
                    help="only check for a newer release, don't update")
    _add_admin_flags(sp); sp.set_defaults(func=cmd_update)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
