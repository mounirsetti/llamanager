"""HTMX-driven web UI.

Auth model:
  * Operator pastes their admin key on /ui/login.
  * The server creates a server-side session keyed by an opaque random
    session id; the cookie carries only that id, never the cleartext key.
  * State-changing routes verify a per-session CSRF token (form field
    `csrf_token` or `X-CSRF-Token` header) plus a same-origin Origin/
    Referer check.

No build step — Pico CSS + HTMX, Jinja2 templates, escape-by-default.
"""
from __future__ import annotations

import json
import secrets
import time
from html import escape as html_escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .auth import AuthManager, Origin
from .queue_mgr import QueueManager
from .registry import Registry
from .server_manager import ServerManager, resolve_spec, ServerError

router = APIRouter(prefix="/ui", tags=["ui"])

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

COOKIE_NAME = "llamanager_session"
SESSION_TTL_S = 60 * 60 * 24 * 7  # 7 days


# ---------- session store (server-side) ----------

class SessionStore:
    """In-process session map. Single-process daemon, so an in-memory dict
    is sufficient; restart logs everyone out, which is acceptable for an
    operator UI. Each session holds the admin key (so we can re-verify
    against the live origin table, since the key may have been rotated)
    and a CSRF token bound to the session.
    """

    def __init__(self, ttl: float = SESSION_TTL_S):
        self.ttl = ttl
        self._data: dict[str, dict[str, Any]] = {}

    def _gc(self, now: float) -> None:
        for sid in [s for s, v in self._data.items() if v["expires"] <= now]:
            self._data.pop(sid, None)

    def create(self, key: str) -> tuple[str, str]:
        sid = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        now = time.time()
        self._gc(now)
        self._data[sid] = {"key": key, "csrf": csrf, "expires": now + self.ttl}
        return sid, csrf

    def get(self, sid: str | None) -> dict[str, Any] | None:
        if not sid:
            return None
        v = self._data.get(sid)
        if not v:
            return None
        if v["expires"] <= time.time():
            self._data.pop(sid, None)
            return None
        return v

    def delete(self, sid: str | None) -> None:
        if sid:
            self._data.pop(sid, None)


def _session_store(request: Request) -> SessionStore:
    return request.app.state.sessions


def _set_session(request: Request, response: Response, key: str) -> str:
    sid, csrf = _session_store(request).create(key)
    response.set_cookie(
        COOKIE_NAME,
        sid,
        httponly=True,
        samesite="lax",
        # Mark Secure when the request looks HTTPS-y. Behind a TLS proxy
        # the X-Forwarded-Proto header tells us; uvicorn's `proxy_headers`
        # surfaces that as request.url.scheme.
        secure=(request.url.scheme == "https"),
        max_age=SESSION_TTL_S,
    )
    return csrf


def _read_session(request: Request) -> dict[str, Any] | None:
    sid = request.cookies.get(COOKIE_NAME)
    return _session_store(request).get(sid)


async def require_admin_ui(request: Request) -> Origin:
    sess = _read_session(request)
    if not sess:
        raise HTTPException(status_code=302, headers={"Location": "/ui/login"})
    am: AuthManager = request.app.state.auth
    origin = await am.verify(sess["key"])
    if not origin or not origin.is_admin:
        # Key rotated, origin removed, or admin scope dropped.
        _session_store(request).delete(request.cookies.get(COOKIE_NAME))
        raise HTTPException(status_code=302, headers={"Location": "/ui/login"})
    # Surface the CSRF token to handlers that render templates.
    request.state.csrf_token = sess["csrf"]
    return origin


# ---------- CSRF helpers ----------

def _same_origin(request: Request) -> bool:
    """Best-effort same-origin check on the Origin/Referer header."""
    ref = request.headers.get("origin") or request.headers.get("referer")
    if not ref:
        # No Origin/Referer at all -> conservatively reject for browser POSTs.
        return False
    try:
        netloc = urlparse(ref).netloc
    except Exception:
        return False
    host = request.headers.get("host", "")
    return bool(netloc) and bool(host) and netloc.lower() == host.lower()


async def _extract_csrf_token(request: Request) -> str | None:
    """Pull the CSRF token from header (HTMX) or form (plain POST)."""
    hdr = request.headers.get("x-csrf-token")
    if hdr:
        return hdr
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/x-www-form-urlencoded") or \
            ctype.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception:
            return None
        val = form.get("csrf_token")
        return str(val) if val is not None else None
    return None


async def require_csrf(request: Request,
                       _: Origin = Depends(require_admin_ui)) -> None:
    """Dependency: validate per-session CSRF token + same-origin Origin/Referer.

    Must be added to every state-changing UI route.
    """
    sess = _read_session(request)
    if not sess:
        raise HTTPException(status_code=403, detail="missing session")
    if not _same_origin(request):
        raise HTTPException(status_code=403, detail="bad origin")
    token = await _extract_csrf_token(request)
    if not token or not secrets.compare_digest(str(token), sess["csrf"]):
        raise HTTPException(status_code=403, detail="invalid csrf token")


def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
    """Build a base template context that always includes the CSRF token
    when an authenticated session is in flight."""
    base: dict[str, Any] = {"request": request,
                            "csrf_token": getattr(request.state, "csrf_token", "")}
    base.update(extra)
    return base


def _error_html(message: str, status_code: int = 400) -> HTMLResponse:
    """Escape arbitrary text into an error fragment. Never use f-strings
    with untrusted data here."""
    safe = html_escape(message, quote=True)
    return HTMLResponse(
        f"<div class='error'>{safe}</div>",
        status_code=status_code,
    )


# ---------- login / logout ----------

@router.get("/login", response_class=HTMLResponse)
async def login_get(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("login.html", {"request": request,
                                                     "error": None})


@router.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, api_key: str = Form(...)) -> Response:
    am: AuthManager = request.app.state.auth
    origin = await am.verify(api_key.strip())
    if not origin or not origin.is_admin:
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "invalid admin key"},
            status_code=401,
        )
    resp = RedirectResponse(url="/ui/", status_code=303)
    _set_session(request, resp, api_key.strip())
    return resp


@router.post("/logout")
async def logout(request: Request,
                 _: None = Depends(require_csrf)) -> Response:
    _session_store(request).delete(request.cookies.get(COOKIE_NAME))
    resp = RedirectResponse(url="/ui/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


# ---------- dashboard ----------

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    db = request.app.state.db
    recent = db.query(
        "SELECT id, origin_id, model, status, enqueued_at, started_at,"
        " finished_at, prompt_tokens, completion_tokens"
        " FROM requests ORDER BY enqueued_at DESC LIMIT 10"
    )
    return templates.TemplateResponse("dashboard.html", _ctx(
        request,
        status=sm.status(),
        queue=qm.snapshot(),
        recent=[dict(r) for r in recent],
    ))


@router.get("/_partials/dashboard", response_class=HTMLResponse)
async def dashboard_partial(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    db = request.app.state.db
    recent = db.query(
        "SELECT id, origin_id, model, status, enqueued_at, finished_at,"
        " prompt_tokens, completion_tokens FROM requests"
        " ORDER BY enqueued_at DESC LIMIT 10"
    )
    return templates.TemplateResponse("_dashboard_partial.html", _ctx(
        request,
        status=sm.status(),
        queue=qm.snapshot(),
        recent=[dict(r) for r in recent],
    ))


# ---------- queue ----------

@router.get("/queue", response_class=HTMLResponse)
async def queue_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    return templates.TemplateResponse("queue.html", _ctx(
        request, queue=qm.snapshot(),
    ))


@router.get("/_partials/queue", response_class=HTMLResponse)
async def queue_partial(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    return templates.TemplateResponse("_queue_partial.html", _ctx(
        request, queue=qm.snapshot(),
    ))


@router.post("/queue/{request_id}/cancel", response_class=HTMLResponse)
async def queue_cancel_ui(request: Request, request_id: str,
                          _: None = Depends(require_csrf)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    qm.cancel(request_id)
    return templates.TemplateResponse("_queue_partial.html", _ctx(
        request, queue=qm.snapshot(),
    ))


# ---------- models ----------

@router.get("/models", response_class=HTMLResponse)
async def models_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    reg: Registry = request.app.state.registry
    return templates.TemplateResponse("models.html", _ctx(
        request,
        models=[m.to_dict() for m in reg.list()],
        downloads=reg.list_downloads(),
        current_model=request.app.state.sm.runtime.current_model,
    ))


@router.post("/models/load", response_class=HTMLResponse)
async def models_load_ui(request: Request, model_id: str = Form(...),
                         _: None = Depends(require_csrf)) -> Response:
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    try:
        spec = resolve_spec(cfg, model=model_id)
        if sm.is_running:
            await sm.swap(spec)
        else:
            await sm.start(spec)
    except (ServerError, ValueError) as e:
        return _error_html(f"load failed: {e}", status_code=400)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/delete", response_class=HTMLResponse)
async def models_delete_ui(request: Request, model_id: str = Form(...),
                           force: bool = Form(False),
                           _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    sm: ServerManager = request.app.state.sm
    loaded = sm.runtime.current_model
    if loaded == model_id and force:
        await sm.stop()
        loaded = None
    ok, err = reg.delete(model_id, currently_loaded=loaded, force=force)
    if not ok:
        return _error_html(f"delete failed: {err}", status_code=400)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/pull", response_class=HTMLResponse)
async def models_pull_ui(request: Request, source: str = Form(...),
                         files: str = Form(""),
                         _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    file_list = [f.strip() for f in files.split(",") if f.strip()] or None
    try:
        reg.start_pull(source=source.strip(), files=file_list)
    except Exception as e:
        return _error_html(f"pull failed: {e}", status_code=400)
    return RedirectResponse("/ui/models", status_code=303)


# ---------- origins ----------

@router.get("/origins", response_class=HTMLResponse)
async def origins_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    return templates.TemplateResponse("origins.html", _ctx(
        request,
        origins=[o.to_public() for o in am.list_origins()],
        new_key=None,
    ))


@router.post("/origins/create", response_class=HTMLResponse)
async def origins_create_ui(request: Request, name: str = Form(...),
                            priority: int = Form(50),
                            allowed_models: str = Form("default"),
                            is_admin: bool = Form(False),
                            _: None = Depends(require_csrf)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    if am.get_origin_by_name(name):
        # Render a normal page with an inline notice; no f-string HTML.
        return templates.TemplateResponse("origins.html", _ctx(
            request,
            origins=[o.to_public() for o in am.list_origins()],
            new_key=None,
            error=f"origin '{name}' already exists",
        ), status_code=409)
    al = [a.strip() for a in allowed_models.split(",") if a.strip()] or ["default"]
    origin, key = am.create_origin(name=name, priority=priority,
                                   allowed_models=al, is_admin=is_admin)
    return templates.TemplateResponse("origins.html", _ctx(
        request,
        origins=[o.to_public() for o in am.list_origins()],
        new_key={"name": origin.name, "key": key},
    ))


@router.post("/origins/{origin_id}/priority", response_class=HTMLResponse)
async def origins_priority_ui(request: Request, origin_id: int,
                              priority: int = Form(...),
                              _: None = Depends(require_csrf)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    am.update_origin(origin_id, priority=priority)
    # Integer is safe to interpolate (FastAPI already coerced to int).
    return HTMLResponse(f"<span>priority: {int(priority)}</span>")


@router.post("/origins/{origin_id}/delete", response_class=HTMLResponse)
async def origins_delete_ui(request: Request, origin_id: int,
                            _: None = Depends(require_csrf)) -> Response:
    am: AuthManager = request.app.state.auth
    am.delete_origin(origin_id)
    return RedirectResponse("/ui/origins", status_code=303)


@router.post("/origins/{origin_id}/rotate", response_class=HTMLResponse)
async def origins_rotate_ui(request: Request, origin_id: int,
                            _: None = Depends(require_csrf)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    key = am.rotate_key(origin_id)
    origin = am.get_origin(origin_id)
    return templates.TemplateResponse("origins.html", _ctx(
        request,
        origins=[o.to_public() for o in am.list_origins()],
        new_key={"name": origin.name if origin else f"#{origin_id}", "key": key},
    ))


# ---------- profiles ----------

@router.get("/profiles", response_class=HTMLResponse)
async def profiles_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    cfg = request.app.state.cfg
    rows: list[dict[str, Any]] = []
    for name, p in cfg.profiles.items():
        rows.append({"name": name, "model": p.model, "mmproj": p.mmproj,
                     "args": json.dumps(p.args, indent=2, sort_keys=True)})
    return templates.TemplateResponse("profiles.html", _ctx(
        request, profiles=rows, config_path=str(cfg.config_path),
    ))


# ---------- logs ----------

@router.get("/logs", response_class=HTMLResponse)
async def logs_view(request: Request, source: str = "llama-server",
                    tail: int = 200,
                    _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    cfg = request.app.state.cfg
    p = cfg.logs_dir / ("llama-server.log" if source == "llama-server"
                        else "llamanager.log")
    text = ""
    if p.exists():
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = 8192
                data = b""
                while size > 0 and data.count(b"\n") <= tail:
                    step = min(block, size)
                    size -= step
                    f.seek(size)
                    data = f.read(step) + data
                text = b"\n".join(data.splitlines()[-tail:]).decode("utf-8", errors="replace")
        except OSError as e:
            text = f"(error reading log: {e})"
    return templates.TemplateResponse("logs.html", _ctx(
        request, source=source, tail=tail, text=text,
    ))


@router.get("/_partials/logs", response_class=HTMLResponse)
async def logs_partial(request: Request, source: str = "llama-server",
                       tail: int = 200,
                       _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return await logs_view(request, source=source, tail=tail, _=_)
