"""FastAPI app factory."""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .api_admin import router as admin_router
from .api_ui import SessionStore, router as ui_router
from .api_v1 import router as v1_router
from .auth import AuthManager, load_or_create_lookup_secret
from .config import Config, load_config
from .db import DB
from .queue_mgr import QueueManager
from .registry import Registry
from .server_manager import ServerManager
from .supervisor import Supervisor
from .llama_installer import detect_binary, InstallState, list_variants
from . import __version__

log = logging.getLogger(__name__)


def _setup_logging(cfg: Config) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.logs_dir / "llamanager.log"
    handler = RotatingFileHandler(
        logfile, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    # don't double-add handlers if reloaded
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == str(logfile):
            return
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _load_or_create_session_secret(cfg: Config) -> str:
    p = cfg.session_secret_path
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    secret = secrets.token_urlsafe(48)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(secret, encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return secret


def _emit_bootstrap_key(cfg: Config, boot_key: str) -> None:
    """Persist the bootstrap admin key to a 0600 file and only echo it
    to stdout when stdout is an interactive terminal.

    Service managers (launchd, systemd, Task Scheduler) redirect stdout
    to a logfile; printing the key there would leave a cleartext copy
    in the journal indefinitely, so we never write it to a non-TTY.
    The operator is told where the file lives so they can read it once
    and delete it.
    """
    p = cfg.data_dir / "bootstrap-key.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically-ish; create-then-chmod has a tiny window where
    # the file exists with default perms, but we're under the user's
    # data_dir which is itself owner-only on a sane install.
    p.write_text(boot_key + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass

    line = "=" * 78
    if sys.stdout.isatty():
        print(f"\n{line}", flush=True)
        print("  llamanager BOOTSTRAP ADMIN KEY (shown ONCE — also written to:",
              flush=True)
        print(f"  {p})", flush=True)
        print(f"  {boot_key}", flush=True)
        print("  Use it to create real origins via /admin/origins, then "
              "revoke 'bootstrap'", flush=True)
        print("  AND delete the file above.", flush=True)
        print(f"{line}\n", flush=True)
    else:
        log.warning(
            "bootstrap admin key written to %s (mode 0600). Read it once, "
            "create real origins via /admin/origins, then revoke 'bootstrap' "
            "and delete the file.", p,
        )


def create_app(config_path: Path | None = None,
               *, print_bootstrap: bool = True) -> FastAPI:
    cfg = load_config(config_path)
    _setup_logging(cfg)

    db = DB(cfg.db_path)
    lookup_secret = load_or_create_lookup_secret(cfg.data_dir)
    auth = AuthManager(db, lookup_secret=lookup_secret,
                       default_priority=cfg.default_origin_priority)
    sm = ServerManager(cfg, db)
    queue = QueueManager(cfg, db, sm)
    supervisor = Supervisor(cfg, sm)
    registry = Registry(cfg, db)

    # Warn if llama-server binary is not found
    if not detect_binary(cfg.llama_server_binary):
        log.warning(
            "llama-server binary not found (configured: %r). "
            "Open the UI > Setup to install it or set the path.",
            cfg.llama_server_binary,
        )

    boot_key = auth.ensure_bootstrap()
    if boot_key and print_bootstrap:
        _emit_bootstrap_key(cfg, boot_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[override]
        queue.start()
        supervisor.start()

        # Autolaunch the configured default model (+ its default profile if
        # any) when autolaunch is on and a binary is available.
        if cfg.autolaunch and detect_binary(cfg.llama_server_binary):
            try:
                from .server_manager import resolve_spec, resolve_default
                model, profile = resolve_default(cfg)
                if not model:
                    log.info("autolaunch: no default_model configured, skipping")
                else:
                    spec = resolve_spec(cfg, model=model, profile=profile)
                    asyncio.create_task(sm.start(spec))
                    log.info("autolaunch: starting %s (profile=%s)", model, profile)
            except Exception as e:
                log.warning("autolaunch failed: %s", e)

        # Periodic database maintenance (daily prune of old records)
        async def _periodic_prune() -> None:
            while True:
                await asyncio.sleep(86400)  # 24 hours
                try:
                    counts = db.prune(max_age_days=90)
                    if any(v > 0 for v in counts.values()):
                        log.info("db prune: deleted %s", counts)
                except Exception as e:
                    log.warning("db prune failed: %s", e)

        prune_task = asyncio.create_task(_periodic_prune())

        # SIGHUP -> reload config (POSIX only)
        loop = None
        try:
            loop = asyncio.get_running_loop()
            if hasattr(signal, "SIGHUP"):
                def _on_sighup() -> None:
                    new = load_config(cfg.path)
                    new.bind = cfg.bind
                    new.port = cfg.port
                    app.state.cfg = new
                    db.log_event("config_reloaded", {"via": "SIGHUP"})
                    log.info("config reloaded via SIGHUP")
                loop.add_signal_handler(signal.SIGHUP, _on_sighup)
        except (NotImplementedError, RuntimeError):
            pass

        try:
            yield
        finally:
            prune_task.cancel()
            await queue.stop()
            await supervisor.stop()
            await sm.stop()
            db.close()

    app = FastAPI(title="llamanager", version=__version__, lifespan=lifespan)
    app.state.cfg = cfg
    app.state.db = db
    app.state.auth = auth
    app.state.sm = sm
    app.state.queue = queue
    app.state.supervisor = supervisor
    app.state.registry = registry
    # One InstallState per installable variant (source + backend), so the UI
    # can poll progress per variant independently.
    app.state.install_states = {v["id"]: InstallState() for v in list_variants()}
    app.state.session_secret = _load_or_create_session_secret(cfg)
    # Server-side admin UI session store. Lives in-process; restart logs
    # everyone out, which is acceptable for a single-host operator UI.
    app.state.sessions = SessionStore()

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        # If a remember-me auto-login happened, set the session cookie
        new_sid = getattr(request.state, "_new_session_sid", None)
        if new_sid:
            from .api_ui import COOKIE_NAME, SESSION_TTL_S
            response.set_cookie(
                COOKIE_NAME,
                new_sid,
                httponly=True,
                samesite="lax",
                secure=(request.url.scheme == "https"),
                max_age=SESSION_TTL_S,
            )
        # Restrictive CSP scoped to the admin UI: third-party origins are
        # explicitly listed (HTMX on cdnjs, Google Fonts CSS + woff2). The
        # `'unsafe-inline'` allowance covers the inline <style>/<script>
        # shipped with the templates. A compromised origin is still bad, but
        # it is constrained to the listed hosts and cannot exfiltrate
        # cross-origin or run plugins.
        if request.url.path.startswith("/ui"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' "
                "https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' "
                "https://fonts.googleapis.com; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "font-src 'self' data: https://fonts.gstatic.com; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    app.include_router(v1_router)
    app.include_router(admin_router)
    app.include_router(ui_router)

    _assets_dir = Path(__file__).resolve().parent.parent / "assets"

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon_svg():
        return FileResponse(_assets_dir / "favicon.svg", media_type="image/svg+xml")

    @app.get("/favicon-dark.svg", include_in_schema=False)
    async def favicon_dark_svg():
        return FileResponse(_assets_dir / "favicon-dark.svg", media_type="image/svg+xml")

    @app.get("/logo.svg", include_in_schema=False)
    async def logo_svg():
        return FileResponse(_assets_dir / "logo.svg", media_type="image/svg+xml")

    @app.get("/logo-dark.svg", include_in_schema=False)
    async def logo_dark_svg():
        return FileResponse(_assets_dir / "logo-dark.svg", media_type="image/svg+xml")

    @app.get("/logo.png", include_in_schema=False)
    async def logo_png():
        return FileResponse(_assets_dir / "logo.png", media_type="image/png")

    @app.get("/logo-dark.png", include_in_schema=False)
    async def logo_dark_png():
        return FileResponse(_assets_dir / "logo-dark.png", media_type="image/png")

    # ---- PWA support ----

    @app.get("/manifest.json", include_in_schema=False)
    async def pwa_manifest():
        import json as _json
        manifest = {
            "name": "llamanager",
            "short_name": "llamanager",
            "description": "Local LLM inference manager",
            "start_url": "/ui/",
            "scope": "/",
            "display": "standalone",
            "background_color": "#201e17",
            "theme_color": "#201e17",
            "icons": [
                {"src": "/icon-light-192.png", "type": "image/png", "sizes": "192x192"},
                {"src": "/icon-light-512.png", "type": "image/png", "sizes": "512x512"},
                {"src": "/icon-dark-192.png", "type": "image/png", "sizes": "192x192"},
                {"src": "/icon-dark-512.png", "type": "image/png", "sizes": "512x512"},
            ],
        }
        return Response(
            content=_json.dumps(manifest),
            media_type="application/manifest+json",
        )

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker():
        sw_js = (
            "// llamanager service worker — enables PWA install\n"
            "self.addEventListener('install', e => self.skipWaiting());\n"
            "self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));\n"
            "self.addEventListener('fetch', e => e.respondWith(fetch(e.request)));\n"
        )
        return Response(content=sw_js, media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/"})

    for _icon_name in ("icon-light-192.png", "icon-light-512.png",
                       "icon-dark-192.png", "icon-dark-512.png"):
        def _make_icon_route(name: str):
            @app.get(f"/{name}", include_in_schema=False)
            async def _serve_icon(n=name):
                return FileResponse(_assets_dir / n, media_type="image/png")
        _make_icon_route(_icon_name)

    @app.get("/health")
    async def health() -> JSONResponse:
        ok = await sm.health() if sm.is_running else False
        import shutil
        free_gb = round(shutil.disk_usage(cfg.models_dir).free / (1024 ** 3), 2)
        return JSONResponse({
            "daemon_ok": True,
            "llama_server_ok": ok,
            "state": sm.runtime.state,
            "disk_free_gb": free_gb,
        })

    @app.get("/chat")
    async def chat_public(request: Request):
        from fastapi.templating import Jinja2Templates
        from pathlib import Path as _Path
        import json as _json
        _templates = Jinja2Templates(directory=str(_Path(__file__).parent / "templates"))
        # Profiles are emitted as [name, bound_model] pairs so the page can
        # filter the dropdown to the user's chosen model.
        profile_pairs = [(p.name, p.model) for p in cfg.profiles.values()]
        model_ids = [m.model_id for m in registry.list()]
        return _templates.TemplateResponse(request, "chat_public.html", {
            "request": request,
            "profiles_json": _json.dumps(profile_pairs),
            "models_json": _json.dumps(model_ids),
        })

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"name": "llamanager", "version": __version__,
                             "ui": "/ui/", "chat": "/chat"})

    return app
