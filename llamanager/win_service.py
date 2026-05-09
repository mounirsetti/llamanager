"""Proper Windows service wrapper using pywin32.

This module is imported only when running as a Windows service (or when
the user explicitly invokes `python -m llamanager.win_service ...`).
Other parts of the codebase do NOT import it — that way `pywin32` stays
optional and platform-specific.

Install / manage:

    python -m llamanager.win_service install     [--startup auto]
    python -m llamanager.win_service start
    python -m llamanager.win_service stop
    python -m llamanager.win_service restart
    python -m llamanager.win_service remove

Or use the CLI shortcut:

    llamanager install-windows-service           (registers + starts)
    llamanager remove-windows-service

Service identity:
    name:         llamanager
    display name: llamanager — manager and proxy for llama-server
    startup:      auto (configurable via --startup)
    account:      LocalSystem by default; pass --username / --password
                  on the install command to run as a different account.
"""
from __future__ import annotations

import logging
import socket
import sys
import threading
from logging.handlers import RotatingFileHandler

# pywin32 imports — only valid on Windows with pywin32 installed.
import servicemanager  # type: ignore[import-not-found]
import win32event  # type: ignore[import-not-found]
import win32service  # type: ignore[import-not-found]
import win32serviceutil  # type: ignore[import-not-found]


SERVICE_NAME = "llamanager"
SERVICE_DISPLAY_NAME = "llamanager — manager and proxy for llama-server"
SERVICE_DESCRIPTION = (
    "Single-host manager and queueing OpenAI-compatible proxy in front "
    "of llama-server."
)


def _setup_service_logging() -> Path:  # type: ignore[name-defined]
    """Wire root logger to the llamanager log file.

    The service runs detached from any console, so without a file
    handler we'd lose every log line. We intentionally do NOT touch
    stdout/stderr — pywin32 redirects those to the service event log
    via `servicemanager.LogMsg`, which is what we use for lifecycle
    notifications below."""
    from pathlib import Path

    from .config import load_config
    cfg = load_config()
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        cfg.logs_dir / "llamanager.log",
        maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(h, "baseFilename", None) == handler.baseFilename
               for h in root.handlers):
        root.addHandler(handler)
    return cfg.logs_dir / "llamanager.log"


class LlamanagerService(win32serviceutil.ServiceFramework):
    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = SERVICE_DISPLAY_NAME
    _svc_description_ = SERVICE_DESCRIPTION

    def __init__(self, args: list[str]):
        super().__init__(args)
        self._stop_event = win32event.CreateEvent(None, 0, 0, None)
        # Sockets default to no timeout; clamp so a stuck connection
        # can't block service stop indefinitely.
        socket.setdefaulttimeout(60)
        self._server = None  # type: ignore[assignment]
        self._uvicorn_thread: threading.Thread | None = None

    # ---- service lifecycle callbacks (called by SCM) ----

    def SvcStop(self) -> None:
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        servicemanager.LogInfoMsg(f"{SERVICE_NAME}: stop requested")
        # Tell uvicorn to shut down. uvicorn's lifespan will call our
        # FastAPI lifespan exit handlers, which in turn stop the queue
        # dispatcher, supervisor, and llama-server child.
        if self._server is not None:
            self._server.should_exit = True
        # If uvicorn is mid-startup, signal the thread to bail out.
        win32event.SetEvent(self._stop_event)
        # Wait up to ~30s for the uvicorn thread to wind down.
        if self._uvicorn_thread is not None:
            self._uvicorn_thread.join(timeout=30)

    def SvcDoRun(self) -> None:
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        try:
            log_path = _setup_service_logging()
            servicemanager.LogInfoMsg(
                f"{SERVICE_NAME}: logging to {log_path}"
            )
            self._run_uvicorn()
        except Exception as e:
            servicemanager.LogErrorMsg(
                f"{SERVICE_NAME}: fatal error: {e!r}"
            )
            raise

    # ---- uvicorn driver ----

    def _run_uvicorn(self) -> None:
        # Imported lazily so that an `import llamanager.win_service` on
        # a misconfigured host can't take the whole package down at
        # import time.
        import uvicorn

        from .app import create_app
        from .config import load_config

        cfg = load_config()
        # Service has no console — never print the bootstrap key from
        # in here. The operator should run `llamanager serve` once
        # interactively before installing the service so the bootstrap
        # banner is visible. If the DB is empty when the service
        # starts, the key is still generated; it's just lost. Log a
        # very loud event-log line so the operator sees something is
        # off.
        from .auth import AuthManager, load_or_create_lookup_secret
        from .db import DB
        db = DB(cfg.db_path)
        am = AuthManager(db,
                         lookup_secret=load_or_create_lookup_secret(cfg.data_dir),
                         default_priority=cfg.default_origin_priority)
        if am.list_origins() == []:
            servicemanager.LogWarningMsg(
                f"{SERVICE_NAME}: no origins exist in state.db — the "
                "bootstrap key will be generated but discarded because "
                "the service has no stdout. Stop the service, run "
                "`llamanager serve` once to capture the bootstrap key, "
                "then start the service again."
            )
        db.close()

        app = create_app(print_bootstrap=False)
        config = uvicorn.Config(
            app,
            host=cfg.bind,
            port=cfg.port,
            log_level="info",
            # uvicorn auto-installs signal handlers on POSIX; on Windows
            # it falls back to setting `should_exit` from KeyboardInterrupt.
            # We trigger should_exit ourselves from SvcStop above.
            log_config=None,
        )
        self._server = uvicorn.Server(config)

        # uvicorn's `Server.run()` is blocking and creates its own
        # event loop. Run it in a worker thread so SvcStop can
        # interrupt SCM's wait without us blocking the main service
        # thread. Actually, simpler: run it in this thread and let
        # SvcStop set should_exit + wait via the event loop's natural
        # shutdown path.
        self._server.run()


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m llamanager.win_service ...`.

    With no args, runs as a service (this is what the SCM invokes).
    With args, dispatches to pywin32's CLI handler (install/start/stop/
    remove/etc.).
    """
    args = list(sys.argv if argv is None else [sys.argv[0], *argv])
    if len(args) == 1:
        # Started by Service Control Manager.
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(LlamanagerService)
        servicemanager.StartServiceCtrlDispatcher()
        return 0
    sys.argv = args
    win32serviceutil.HandleCommandLine(LlamanagerService)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
