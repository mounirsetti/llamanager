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
from .api_anthropic import router as anthropic_router
from .api_ui import SessionStore, router as ui_router
from .api_v1 import router as v1_router
from .auth import AuthManager, load_or_create_lookup_secret
from .config import Config, load_config
from .db import DB
from . import exclusive as _exclusive
from .image_runner import ImageTaskRunner
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

    # Demote chatty loggers that otherwise drown the file in per-poll noise:
    #   - uvicorn.access: every HTMX partial refresh prints a line
    #   - httpx: every proxied request to llama-server prints "HTTP/1.1 200 OK"
    # Set LLAMANAGER_VERBOSE_LOGS=1 to keep them at INFO for debugging.
    if not os.environ.get("LLAMANAGER_VERBOSE_LOGS"):
        for noisy in ("uvicorn.access", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


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
    image_runner = ImageTaskRunner(cfg, db, sm=sm)

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

        # Exclusive-mode heartbeat: re-sweep every N seconds (config knob).
        # Always running; it's the read of cfg.exclusive_mode on each tick
        # that decides whether to actually do anything, so toggling the
        # mode via UI/CLI takes effect on the next interval without a
        # daemon restart.
        async def _exclusive_boot_sweep() -> None:
            try:
                cur_cfg = app.state.cfg
                if (getattr(cur_cfg, "exclusive_mode", "off") or "off") != "off":
                    await _exclusive.sweep_and_record(
                        cur_cfg.exclusive_mode,
                        grace_seconds=float(getattr(cur_cfg, "exclusive_grace_seconds", 5.0)),
                    )
            except Exception:  # noqa: BLE001 — boot sweep must never crash startup
                log.exception("exclusive: boot sweep failed")

        boot_sweep_task = asyncio.create_task(_exclusive_boot_sweep())

        excl_stop = asyncio.Event()
        excl_task = asyncio.create_task(
            _exclusive.heartbeat(
                get_mode=lambda: getattr(app.state.cfg, "exclusive_mode", "off"),
                get_grace=lambda: getattr(app.state.cfg, "exclusive_grace_seconds", 5.0),
                get_interval=lambda: getattr(app.state.cfg, "exclusive_heartbeat_seconds", 120),
                stop_event=excl_stop,
            )
        )

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
            boot_sweep_task.cancel()
            excl_stop.set()
            excl_task.cancel()
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
    app.state.image_runner = image_runner
    from .engine_installer import EngineInstaller
    app.state.engine_installer = EngineInstaller(cfg, db)
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
    app.include_router(anthropic_router)
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
        from .config import ENGINE_FAMILY, detect_engine_for_id

        _templates = Jinja2Templates(directory=str(_Path(__file__).parent / "templates"))
        # LLM-family only; image-family models live on /images.
        llm_models: list[str] = []
        for m in registry.list():
            engine = detect_engine_for_id(m.model_id, cfg.models_dir)
            if ENGINE_FAMILY.get(engine, "text") == "text":
                llm_models.append(m.model_id)
        llm_set = set(llm_models)
        # Profile triples: (name, bound_model, vision_capable). Only
        # profiles bound to LLM models are surfaced; ``vision`` is True
        # iff the profile has an mmproj set, which the chat UI uses to
        # enable the image-attach button. Per-bearer gating
        # (allowed_models) is applied client-side after login.
        profile_triples = [
            [p.name, mid, bool(p.mmproj)]
            for mid, p in cfg.iter_profiles()
            if mid in llm_set
        ]
        return _templates.TemplateResponse(request, "chat_public.html", {
            "request": request,
            "profiles_json": _json.dumps(profile_triples),
            "models_json": _json.dumps(llm_models),
        })

    @app.get("/images")
    async def images_public(request: Request):
        """Public images page — any valid origin key, no admin session.

        Sibling to /chat. Re-uses the same schema-driven context the admin
        page builds so the two surfaces stay structurally identical; the
        only difference is which gallery endpoint and which bearer the
        page calls.
        """
        from fastapi.templating import Jinja2Templates
        from pathlib import Path as _Path
        from .api_ui import _build_image_page_context

        _templates = Jinja2Templates(directory=str(_Path(__file__).parent / "templates"))
        ctx = _build_image_page_context(cfg, registry)
        return _templates.TemplateResponse(request, "images_public.html", {
            "request": request,
            **ctx,
        })

    @app.get("/images/gallery")
    async def images_gallery_public(request: Request,
                                    limit: int = 60,
                                    before: float | None = None) -> JSONResponse:
        """Public counterpart of /ui/images/gallery, scoped to the bearer's
        origin. Same payload shape; only this origin's subdirectory under
        ``images_dir`` is enumerated."""
        from .api_v1 import _origin_from_request
        from .api_ui import _list_gallery
        if limit < 1 or limit > 500:
            return JSONResponse({"detail": "limit must be 1..500"},
                                status_code=400)
        origin = await _origin_from_request(request)
        # Mirror the on-disk name produced by image_runner._gallery_dir
        safe_origin = "".join(c for c in origin.name
                              if c.isalnum() or c in "-_") or "anon"
        payload = _list_gallery(cfg.images_dir,
                                origin_filter=safe_origin,
                                limit=limit, before=before)
        return JSONResponse(payload)

    @app.get("/images/file/{day}/{origin}/{name}")
    async def images_file_serve_public(request: Request,
                                       day: str, origin: str,
                                       name: str) -> Response:
        """Bearer-authenticated PNG serving for the public gallery. The path
        is constrained to this origin's own directory to avoid leaking
        other users' galleries."""
        from .api_v1 import _origin_from_request
        from .api_ui import _safe_path_components
        from fastapi.responses import FileResponse as _FileResponse
        bearer_origin = await _origin_from_request(request)
        safe_origin = "".join(c for c in bearer_origin.name
                              if c.isalnum() or c in "-_") or "anon"
        if origin != safe_origin:
            raise HTTPException(status_code=403,
                                detail="cannot access another origin's gallery")
        _safe_path_components(day, origin, name)
        if not name.lower().endswith(".png"):
            raise HTTPException(status_code=400,
                                detail="only .png files are served here")
        p = (cfg.images_dir / day / origin / name).resolve()
        try:
            p.relative_to(cfg.images_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="path escapes images_dir")
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return _FileResponse(p, media_type="image/png")

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"name": "llamanager", "version": __version__,
                             "ui": "/ui/", "chat": "/chat"})

    return app
