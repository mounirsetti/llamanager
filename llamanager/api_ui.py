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

import asyncio
import json
import logging
import platform
import secrets
import shutil
import subprocess
import time

log = logging.getLogger(__name__)
from html import escape as html_escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .auth import AuthManager, Origin
from .config import (
    load_config, save_profile, delete_profile, update_defaults,
)
from .llama_installer import (
    InstallState,
    detect_binary,
    install_instructions,
    install_llama_server,
    patch_config_binary,
)
from .queue_mgr import QueueManager
from .registry import Registry
from .server_manager import ServerManager, resolve_spec, ServerError
from .supervisor import Supervisor

router = APIRouter(prefix="/ui", tags=["ui"])

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

COOKIE_NAME = "llamanager_session"
REMEMBER_COOKIE = "llamanager_remember"
SESSION_TTL_S = 60 * 60 * 24 * 7  # 7 days
REMEMBER_TTL_S = 60 * 60 * 24 * 30  # 30 days


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


def _set_session(request: Request, response: Response, key: str,
                  remember: bool = False) -> str:
    sid, csrf = _session_store(request).create(key)
    is_secure = (request.url.scheme == "https")
    response.set_cookie(
        COOKIE_NAME,
        sid,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        max_age=SESSION_TTL_S,
    )
    if remember:
        # Store a signed token so the session survives server restarts.
        # The token is the key signed with itsdangerous using the
        # session secret that persists on disk.
        from itsdangerous import URLSafeTimedSerializer
        secret = request.app.state.session_secret
        s = URLSafeTimedSerializer(secret)
        token = s.dumps(key, salt="remember")
        response.set_cookie(
            REMEMBER_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            secure=is_secure,
            max_age=REMEMBER_TTL_S,
        )
    return csrf


def _try_remember(request: Request) -> str | None:
    """Try to recover a session from the remember-me cookie.

    Returns the API key if valid, None otherwise.
    """
    token = request.cookies.get(REMEMBER_COOKIE)
    if not token:
        return None
    try:
        from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
        secret = request.app.state.session_secret
        s = URLSafeTimedSerializer(secret)
        key = s.loads(token, salt="remember", max_age=REMEMBER_TTL_S)
        return key
    except Exception:
        return None


def _read_session(request: Request) -> dict[str, Any] | None:
    sid = request.cookies.get(COOKIE_NAME)
    return _session_store(request).get(sid)


async def require_admin_ui(request: Request) -> Origin:
    sess = _read_session(request)

    # If no active session, try to restore from remember-me cookie
    if not sess:
        remembered_key = _try_remember(request)
        if remembered_key:
            am: AuthManager = request.app.state.auth
            origin = await am.verify(remembered_key)
            if origin and origin.is_admin:
                # Recreate session in-memory (cookie already exists from last login)
                sid, csrf = _session_store(request).create(remembered_key)
                # Stash the new session id so the response middleware can set the cookie
                request.state._new_session_sid = sid
                request.state.csrf_token = csrf
                return origin

    if not sess:
        raise HTTPException(status_code=302, headers={"Location": "/ui/login"})
    am_: AuthManager = request.app.state.auth
    origin = await am_.verify(sess["key"])
    if not origin or not origin.is_admin:
        _session_store(request).delete(request.cookies.get(COOKIE_NAME))
        raise HTTPException(status_code=302, headers={"Location": "/ui/login"})
    request.state.csrf_token = sess["csrf"]
    return origin


# ---------- CSRF helpers ----------

def _normalise_host(netloc: str) -> str:
    """Treat localhost and 127.0.0.1 as the same host."""
    return netloc.lower().replace("localhost", "127.0.0.1")


def _same_origin(request: Request) -> bool:
    """Best-effort same-origin check on the Origin/Referer header.

    When Referrer-Policy: no-referrer is active (which we set), Chrome sends
    Origin: null on form POSTs. In that case we skip the origin check and rely
    solely on the CSRF token, which is the real protection.
    """
    origin = request.headers.get("origin")
    # "null" origin is sent by browsers when referrer policy suppresses it.
    # Treat it as same-origin; CSRF token validation is the actual guard.
    if origin == "null":
        return True
    ref = origin or request.headers.get("referer")
    if not ref:
        return False
    try:
        netloc = urlparse(ref).netloc
    except Exception:
        return False
    host = request.headers.get("host", "")
    return bool(netloc) and bool(host) and _normalise_host(netloc) == _normalise_host(host)


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


def _get_system_info(models_dir: str | Path) -> dict[str, Any]:
    """Gather hardware info relevant to llama-server inference."""
    import psutil

    info: dict[str, Any] = {}

    # CPU
    info["cpu"] = platform.processor() or platform.machine()
    info["cpu_cores"] = psutil.cpu_count(logical=False) or 0
    info["cpu_threads"] = psutil.cpu_count(logical=True) or 0
    info["arch"] = platform.machine()

    # On macOS, get the chip name from sysctl
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=2,
            ).strip()
            if chip:
                info["cpu"] = chip
        except Exception:
            pass

    # RAM
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024**3), 1)
    info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    info["ram_used_pct"] = mem.percent

    # Top 5 memory-consuming processes
    try:
        procs = []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                mi = p.info["memory_info"]
                if mi is None:
                    continue
                mem_mb = round(mi.rss / (1024**2), 1)
                procs.append({"name": p.info["name"], "pid": p.info["pid"], "mem_mb": mem_mb})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
                continue
        procs.sort(key=lambda x: x["mem_mb"], reverse=True)
        info["top_mem_procs"] = procs[:5]
    except Exception:
        info["top_mem_procs"] = []

    # Swap
    swap = psutil.swap_memory()
    info["swap_total_gb"] = round(swap.total / (1024**3), 1)
    info["swap_used_gb"] = round(swap.used / (1024**3), 1)

    # GPU info
    system = platform.system()
    if system == "Darwin":
        # Apple Silicon uses Metal with unified memory
        info["gpu_type"] = "Metal (unified memory)"
        info["gpu_name"] = ""
        try:
            sp = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, timeout=5,
            )
            displays = json.loads(sp).get("SPDisplaysDataType", [])
            if displays:
                gpu = displays[0]
                info["gpu_name"] = gpu.get("sppci_model", "")
                cores = gpu.get("sppci_cores", "")
                if cores:
                    info["gpu_cores"] = cores
        except Exception:
            pass
    else:
        # Try nvidia-smi for CUDA GPUs
        info["gpu_type"] = "unknown"
        info["gpu_name"] = ""
        try:
            nv = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip()
            if nv:
                parts = nv.split(",")
                info["gpu_type"] = "CUDA"
                info["gpu_name"] = parts[0].strip()
                info["gpu_vram_total_gb"] = round(int(parts[1].strip()) / 1024, 1)
                info["gpu_vram_free_gb"] = round(int(parts[2].strip()) / 1024, 1)
        except Exception:
            info["gpu_type"] = "CPU-only (no CUDA detected)"

    # Disk free in models dir
    try:
        disk = shutil.disk_usage(str(models_dir))
        info["disk_free_gb"] = round(disk.free / (1024**3), 1)
        info["disk_total_gb"] = round(disk.total / (1024**3), 1)
    except Exception:
        info["disk_free_gb"] = 0
        info["disk_total_gb"] = 0

    # OS
    info["os"] = f"{platform.system()} {platform.release()}"

    return info


def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
    """Build a base template context that always includes the CSRF token
    when an authenticated session is in flight."""
    cfg = request.app.state.cfg
    binary_path = detect_binary(cfg.llama_server_binary)
    sm: ServerManager = request.app.state.sm
    server_state = sm.runtime.state  # stopped | starting | running | swapping | crashed | degraded
    has_models = bool(request.app.state.registry.list())
    # Dot color: red = no binary, orange = binary but not running, green = running
    if not binary_path:
        dot_status = "missing"
    elif server_state == "running":
        dot_status = "running"
    else:
        dot_status = "idle"
    base: dict[str, Any] = {
        "request": request,
        "csrf_token": getattr(request.state, "csrf_token", ""),
        "binary_missing": not binary_path,
        "server_state": server_state,
        "dot_status": dot_status,
        "has_models": has_models,
    }
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
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, api_key: str = Form(...),
                     remember: str = Form("off")) -> Response:
    am: AuthManager = request.app.state.auth
    origin = await am.verify(api_key.strip())
    if not origin or not origin.is_admin:
        return templates.TemplateResponse(
            request, "login.html", {"error": "invalid admin key"},
            status_code=401,
        )
    resp = RedirectResponse(url="/ui/", status_code=303)
    _set_session(request, resp, api_key.strip(), remember=(remember == "on"))
    return resp


@router.post("/logout")
async def logout(request: Request,
                 _: None = Depends(require_csrf)) -> Response:
    _session_store(request).delete(request.cookies.get(COOKIE_NAME))
    resp = RedirectResponse(url="/ui/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    resp.delete_cookie(REMEMBER_COOKIE)
    return resp


# ---------- dashboard ----------

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    cfg = request.app.state.cfg
    db = request.app.state.db
    recent = db.query(
        "SELECT id, origin_id, model, status, enqueued_at, started_at,"
        " finished_at, prompt_tokens, completion_tokens"
        " FROM requests ORDER BY enqueued_at DESC LIMIT 10"
    )
    # Build the base URL for cheat sheet examples
    host = request.headers.get("host", f"{cfg.bind}:{cfg.port}")
    base_url = f"http://{host}"
    sysinfo = _get_system_info(cfg.models_dir)
    return templates.TemplateResponse(request, "dashboard.html", _ctx(
        request,
        status=sm.status(),
        queue=qm.snapshot(),
        recent=[dict(r) for r in recent],
        base_url=base_url,
        sysinfo=sysinfo,
    ))


@router.get("/_partials/dashboard", response_class=HTMLResponse)
async def dashboard_partial(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    cfg = request.app.state.cfg
    db = request.app.state.db
    recent = db.query(
        "SELECT id, origin_id, model, status, enqueued_at, started_at,"
        " finished_at, prompt_tokens, completion_tokens FROM requests"
        " ORDER BY enqueued_at DESC LIMIT 10"
    )
    sysinfo = _get_system_info(cfg.models_dir)
    return templates.TemplateResponse(request, "_dashboard_partial.html", _ctx(
        request,
        status=sm.status(),
        queue=qm.snapshot(),
        recent=[dict(r) for r in recent],
        sysinfo=sysinfo,
    ))


# ---------- queue ----------

@router.get("/queue", response_class=HTMLResponse)
async def queue_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    return templates.TemplateResponse(request, "queue.html", _ctx(
        request, queue=qm.snapshot(),
    ))


@router.get("/_partials/queue", response_class=HTMLResponse)
async def queue_partial(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    return templates.TemplateResponse(request, "_queue_partial.html", _ctx(
        request, queue=qm.snapshot(),
    ))


@router.post("/queue/{request_id}/cancel", response_class=HTMLResponse)
async def queue_cancel_ui(request: Request, request_id: str,
                          _: None = Depends(require_csrf)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    qm.cancel(request_id)
    return templates.TemplateResponse(request, "_queue_partial.html", _ctx(
        request, queue=qm.snapshot(),
    ))


# ---------- models ----------

def _models_ctx(request: Request) -> dict:
    import sys as _sys
    reg: Registry = request.app.state.registry
    cfg = request.app.state.cfg
    if _sys.platform == "darwin":
        open_label = "Open in Finder"
    elif _sys.platform == "win32":
        open_label = "Open in Explorer"
    else:
        open_label = "Open folder"
    return _ctx(
        request,
        models=[m.to_dict() for m in reg.list()],
        downloads=reg.list_downloads(),
        current_model=request.app.state.sm.runtime.current_model,
        models_dir=str(cfg.models_dir),
        open_label=open_label,
    )


@router.get("/models", response_class=HTMLResponse)
async def models_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "models.html", _models_ctx(request))


@router.get("/models/_list", response_class=HTMLResponse)
async def models_list_partial(request: Request,
                              _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    reg: Registry = request.app.state.registry
    return templates.TemplateResponse(request, "_models_list_partial.html", _ctx(
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


@router.post("/models/add-existing", response_class=HTMLResponse)
async def models_add_existing(request: Request,
                              file_path: str = Form(...),
                              _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    src = Path(file_path.strip()).expanduser().resolve()
    if not src.exists():
        return _error_html(f"file not found: {src}", status_code=400)
    if not src.is_file():
        return _error_html(f"not a file: {src}", status_code=400)
    if not src.name.lower().endswith(".gguf"):
        return _error_html("only .gguf files are supported", status_code=400)
    dest = cfg.models_dir / src.name
    if dest.exists():
        return _error_html(f"a model named {src.name} already exists", status_code=409)
    try:
        dest.symlink_to(src)
    except OSError:
        # Symlinks may fail on some Windows setups; fall back to copy
        import shutil
        shutil.copy2(src, dest)
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
    # Render immediately so the HTMX polling div is present from the start
    return templates.TemplateResponse(request, "models.html", _models_ctx(request))


@router.post("/models/set-dir", response_class=HTMLResponse)
async def models_set_dir(request: Request,
                         models_dir: str = Form(...),
                         _: None = Depends(require_csrf)) -> Response:
    import re as _re
    cfg = request.app.state.cfg
    new_dir = Path(models_dir.strip()).expanduser().resolve()
    new_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir_override = new_dir
    cfg.registry = request.app.state.registry
    request.app.state.registry.models_dir = new_dir
    # Persist to config.toml
    config_path = cfg.config_path
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    new_line = f'models_dir = {json.dumps(str(new_dir))}'
    text, n = _re.subn(r'^models_dir\s*=\s*.*$', new_line, text, flags=_re.MULTILINE)
    if n == 0:
        server_match = _re.search(r'^\[server\]', text, flags=_re.MULTILINE)
        if server_match:
            text = text[:server_match.end()] + f"\n{new_line}" + text[server_match.end():]
        else:
            text = text.rstrip("\n") + f"\n\n[server]\n{new_line}\n"
    config_path.write_text(text, encoding="utf-8")
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/downloads/{download_id}/cancel", response_class=HTMLResponse)
async def download_cancel_ui(request: Request, download_id: str,
                              _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    reg.cancel_pull(download_id)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/downloads/{download_id}/delete", response_class=HTMLResponse)
async def download_delete_ui(request: Request, download_id: str,
                              _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    reg.delete_download(download_id)
    return RedirectResponse("/ui/models", status_code=303)


def _open_path(path: str) -> None:
    import subprocess, sys as _sys
    if _sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    elif _sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", path])
    else:
        # xdg-open opens the parent directory
        subprocess.Popen(["xdg-open", str(Path(path).parent)])


@router.post("/models/open-dir", response_class=HTMLResponse)
async def models_open_dir(request: Request,
                          _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    _open_path(str(cfg.models_dir))
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/locate", response_class=HTMLResponse)
async def models_locate(request: Request,
                        model_id: str = Form(...),
                        _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    m = reg.get(model_id)
    if m:
        _open_path(str(m.path))
    return RedirectResponse("/ui/models", status_code=303)


# ---------- origins ----------

@router.get("/origins", response_class=HTMLResponse)
async def origins_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    return templates.TemplateResponse(request, "origins.html", _ctx(
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
        return templates.TemplateResponse(request, "origins.html", _ctx(
            request,
            origins=[o.to_public() for o in am.list_origins()],
            new_key=None,
            error=f"origin '{name}' already exists",
        ), status_code=409)
    al = [a.strip() for a in allowed_models.split(",") if a.strip()] or ["default"]
    origin, key = am.create_origin(name=name, priority=priority,
                                   allowed_models=al, is_admin=is_admin)
    return templates.TemplateResponse(request, "origins.html", _ctx(
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
    return templates.TemplateResponse(request, "origins.html", _ctx(
        request,
        origins=[o.to_public() for o in am.list_origins()],
        new_key={"name": origin.name if origin else f"#{origin_id}", "key": key},
    ))


# ---------- profiles (redirect to launch) ----------

@router.get("/profiles")
async def profiles_redirect(request: Request,
                            _: Origin = Depends(require_admin_ui)) -> Response:
    return RedirectResponse("/ui/launch", status_code=301)


# ---------- about ----------

LLAMANAGER_VERSION = "0.1.0"
GITHUB_REPO = "mounirsetti/llamanager"


def _about_ctx(request: Request, **extra: Any) -> dict:
    import datetime
    ctx = _ctx(
        request,
        version=LLAMANAGER_VERSION,
        year=datetime.date.today().year,
        update_available=None,
        latest_version=None,
        update_check_error=None,
        update_log=None,
    )
    ctx.update(extra)
    return ctx


@router.get("/about", response_class=HTMLResponse)
async def about_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "about.html", _about_ctx(request))


@router.post("/about/check-update", response_class=HTMLResponse)
async def about_check_update(request: Request,
                             _: None = Depends(require_csrf)) -> HTMLResponse:
    import urllib.request
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            headers={"User-Agent": "llamanager", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            import json as _json
            data = _json.loads(r.read())
        latest = data.get("tag_name", "").lstrip("v")
        if not latest:
            raise ValueError("no tag_name in release")
        is_newer = latest != LLAMANAGER_VERSION
        return templates.TemplateResponse(request, "about.html", _about_ctx(
            request,
            update_available=is_newer,
            latest_version=latest,
        ))
    except Exception as e:
        return templates.TemplateResponse(request, "about.html", _about_ctx(
            request,
            update_check_error=f"Could not check for updates: {e}",
        ))


@router.post("/about/update", response_class=HTMLResponse)
async def about_update(request: Request,
                       _: None = Depends(require_csrf)) -> Response:
    import subprocess, sys, os, signal

    project_dir = Path(__file__).parent.parent
    log_lines: list[str] = []

    try:
        # Step 1: git pull
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(project_dir),
            capture_output=True, text=True, timeout=60,
        )
        log_lines.append("$ git pull --ff-only")
        log_lines.append(result.stdout.strip())
        if result.stderr.strip():
            log_lines.append(result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(f"git pull failed (exit {result.returncode})")

        # Step 2: pip install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=str(project_dir),
            capture_output=True, text=True, timeout=120,
        )
        log_lines.append("\n$ pip install -e .")
        # Only show last few lines of pip output
        pip_lines = result.stdout.strip().splitlines()
        log_lines.extend(pip_lines[-5:])
        if result.stderr.strip():
            err_lines = result.stderr.strip().splitlines()
            log_lines.extend(l for l in err_lines if "WARNING" not in l)
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed (exit {result.returncode})")

        log_lines.append("\nUpdate complete. Restarting...")

        # Step 3: stop llama-server and restart
        sm: ServerManager = request.app.state.sm
        if sm.is_running:
            await sm.stop()

        # Send SIGTERM after a brief delay to let the response reach the browser
        async def _delayed_restart():
            await asyncio.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)
        asyncio.create_task(_delayed_restart())

        return templates.TemplateResponse(request, "about.html", _about_ctx(
            request,
            update_log="\n".join(log_lines),
            update_available=False,
            latest_version="restarting",
        ))

    except Exception as e:
        log_lines.append(f"\nError: {e}")
        return templates.TemplateResponse(request, "about.html", _about_ctx(
            request,
            update_check_error=str(e),
            update_log="\n".join(log_lines),
        ))


# ---------- chat ----------

@router.get("/chat", response_class=HTMLResponse)
async def chat_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    cfg = request.app.state.cfg
    sm: ServerManager = request.app.state.sm
    reg: Registry = request.app.state.registry
    sess = _read_session(request)
    profiles = [{"name": name} for name in cfg.profiles]
    model_ids = [m.model_id for m in reg.list()]
    return templates.TemplateResponse(request, "chat.html", _ctx(
        request,
        status=sm.status(),
        profiles=profiles,
        model_ids=model_ids,
        api_key=sess["key"] if sess else "",
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
    return templates.TemplateResponse(request, "logs.html", _ctx(
        request, source=source, tail=tail, text=text,
    ))


@router.get("/_partials/logs", response_class=HTMLResponse)
async def logs_partial(request: Request, source: str = "llama-server",
                       tail: int = 200,
                       _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return await logs_view(request, source=source, tail=tail, _=_)


# ---------- setup ----------

@router.get("/setup", response_class=HTMLResponse)
async def setup_get(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "setup.html", _setup_ctx(request))


@router.post("/setup/binary-path", response_class=HTMLResponse)
async def setup_set_binary(request: Request,
                           binary_path: str = Form(...),
                           _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    binary_path = binary_path.strip()
    cfg.llama_server_binary = binary_path
    patch_config_binary(cfg.config_path, binary_path)
    return RedirectResponse("/ui/setup", status_code=303)


def _setup_ctx(request: Request) -> dict:
    import sys as _sys
    cfg = request.app.state.cfg
    if _sys.platform == "darwin":
        open_config_label = "Open in Finder"
    elif _sys.platform == "win32":
        open_config_label = "Open in Explorer"
    else:
        open_config_label = "Open folder"
    return _ctx(
        request,
        binary_path=detect_binary(cfg.llama_server_binary),
        configured_binary=cfg.llama_server_binary,
        instructions=install_instructions(),
        install=request.app.state.install_state.to_dict(),
        config_dir=str(cfg.config_path.parent),
        config_file=str(cfg.config_path),
        open_config_label=open_config_label,
    )


@router.post("/setup/install", response_class=HTMLResponse)
async def setup_install(request: Request,
                        _: None = Depends(require_csrf)) -> Response:
    state: InstallState = request.app.state.install_state
    if state.status != "running":
        state.status = "running"
        state.lines = []
        state.error = None
        state.installed_path = None
        asyncio.create_task(install_llama_server(state))
    # Render immediately so the HTMX polling div is present from the start
    return templates.TemplateResponse(request, "setup.html", _setup_ctx(request))


@router.get("/setup/_install_progress", response_class=HTMLResponse)
async def setup_install_progress(request: Request,
                                 _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    state: InstallState = request.app.state.install_state
    cfg = request.app.state.cfg
    if state.status == "done" and state.installed_path:
        cfg.llama_server_binary = state.installed_path
        patch_config_binary(cfg.config_path, state.installed_path)
    binary_path = detect_binary(cfg.llama_server_binary)
    return templates.TemplateResponse(request, "_install_progress.html", _ctx(
        request,
        install=state.to_dict(),
        binary_path=binary_path,
    ))


@router.post("/setup/open-config", response_class=HTMLResponse)
async def setup_open_config(request: Request,
                            _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    _open_path(str(cfg.config_path))
    return RedirectResponse("/ui/setup", status_code=303)


@router.post("/setup/restart", response_class=HTMLResponse)
async def setup_restart(request: Request,
                        _: None = Depends(require_csrf)) -> Response:
    """Restart the llamanager process.

    Stops llama-server cleanly, then exits with code 0. When running under
    a service manager (launchd, systemd, Task Scheduler), the manager
    restarts the process automatically. When running interactively, the
    user will need to run `llamanager serve` again.
    """
    import os, signal
    sm: ServerManager = request.app.state.sm
    # Stop llama-server first so it exits cleanly
    if sm.is_running:
        await sm.stop()
    log.info("restart requested via UI, shutting down")
    # Send SIGTERM to ourselves — uvicorn handles this gracefully
    os.kill(os.getpid(), signal.SIGTERM)
    return HTMLResponse(
        "<p style='padding:40px;font-family:sans-serif;color:#999;text-align:center;'>"
        "Restarting llamanager. This page will reload automatically.</p>"
        "<script>setTimeout(function(){location.href='/ui/';},5000);</script>",
    )


# ---------- launch ----------

def _reload_config(request: Request) -> None:
    """Reload config.toml into app.state.cfg, preserving runtime-only fields."""
    cfg = request.app.state.cfg
    new_cfg = load_config(cfg.config_path)
    new_cfg.bind = cfg.bind
    new_cfg.port = cfg.port
    # Preserve models_dir if it was overridden at runtime
    if hasattr(cfg, 'models_dir_override'):
        new_cfg.models_dir_override = cfg.models_dir_override
    request.app.state.cfg = new_cfg
    # Update registry to use potentially new models_dir
    request.app.state.registry.models_dir = new_cfg.models_dir


def _read_log_tail(cfg, lines: int = 15) -> str:
    log_path = cfg.logs_dir / "llama-server.log"
    if not log_path.exists():
        return ""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(4096, size)
            if block <= 0:
                return ""
            f.seek(size - block)
            data = f.read(block)
            return b"\n".join(data.splitlines()[-lines:]).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _launch_ctx(request: Request) -> dict:
    cfg = request.app.state.cfg
    sm: ServerManager = request.app.state.sm
    supervisor: Supervisor = request.app.state.supervisor
    reg: Registry = request.app.state.registry
    model_ids = [m.model_id for m in reg.list()]
    profiles = []
    for name, p in cfg.profiles.items():
        available = p.model in model_ids
        profiles.append({
            "name": name,
            "model": p.model,
            "mmproj": p.mmproj,
            "args": p.args,
            "args_json": json.dumps(p.args, indent=2, sort_keys=True),
            "available": available,
        })
    return _ctx(
        request,
        status=sm.status(),
        profiles=profiles,
        profile_names=list(cfg.profiles.keys()),
        model_ids=model_ids,
        default_model=cfg.default_model,
        default_profile=cfg.default_profile,
        autorestart=supervisor.enabled,
        autolaunch=cfg.autolaunch,
        max_restarts=cfg.max_restarts_in_window,
        log_tail=_read_log_tail(cfg),
    )


@router.get("/launch", response_class=HTMLResponse)
async def launch_view(request: Request,
                      _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "launch.html", _launch_ctx(request))


@router.post("/launch/server/start", response_class=HTMLResponse)
async def launch_server_start(request: Request,
                              profile: str = Form(""),
                              _: None = Depends(require_csrf)) -> Response:
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    try:
        spec = resolve_spec(cfg, profile=profile or None)
    except (ServerError, ValueError) as e:
        return _error_html(str(e), status_code=400)
    # Fire and forget — the page will poll for status
    asyncio.create_task(_start_server_bg(sm, spec))
    # Wait briefly so the state transitions from stopped to starting
    await asyncio.sleep(0.3)
    ctx = _launch_ctx(request)
    # Force-show the status partial even if state hasn't transitioned yet
    ctx["launching"] = True
    return templates.TemplateResponse(request, "launch.html", ctx)


async def _start_server_bg(sm: ServerManager, spec) -> None:
    try:
        await sm.start(spec)
    except Exception:
        pass  # state is already set to crashed/stopped by ServerManager


@router.get("/launch/_server_status", response_class=HTMLResponse)
async def launch_server_status(request: Request,
                               _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    cfg = request.app.state.cfg
    sm: ServerManager = request.app.state.sm
    return templates.TemplateResponse(request, "_server_status.html", _ctx(
        request,
        status=sm.status(),
        log_tail=_read_log_tail(cfg),
    ))


@router.post("/launch/server/stop", response_class=HTMLResponse)
async def launch_server_stop(request: Request,
                             _: None = Depends(require_csrf)) -> Response:
    sm: ServerManager = request.app.state.sm
    await sm.stop()
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/autorestart", response_class=HTMLResponse)
async def launch_autorestart(request: Request,
                             enabled: str = Form("off"),
                             _: None = Depends(require_csrf)) -> Response:
    request.app.state.supervisor.enabled = (enabled == "on")
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/autolaunch", response_class=HTMLResponse)
async def launch_autolaunch(request: Request,
                            enabled: str = Form("off"),
                            _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    on = (enabled == "on")
    cfg.autolaunch = on
    update_defaults(cfg.config_path, autolaunch=on)
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/profiles/create", response_class=HTMLResponse)
async def launch_profile_create(request: Request,
                                name: str = Form(...),
                                model: str = Form(...),
                                mmproj: str = Form(""),
                                args_json: str = Form("{}"),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    try:
        args = json.loads(args_json.strip() or "{}")
    except json.JSONDecodeError as e:
        return _error_html(f"invalid JSON in args: {e}", status_code=400)
    if name.strip().lower() in cfg.profiles:
        return _error_html(f"profile '{name}' already exists", status_code=409)
    try:
        save_profile(cfg.config_path, name, model.strip(), mmproj.strip(), args)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    _reload_config(request)
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/profiles/{profile_name}/update", response_class=HTMLResponse)
async def launch_profile_update(request: Request, profile_name: str,
                                new_name: str = Form(""),
                                model: str = Form(...),
                                mmproj: str = Form(""),
                                args_json: str = Form("{}"),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    try:
        args = json.loads(args_json.strip() or "{}")
    except json.JSONDecodeError as e:
        return _error_html(f"invalid JSON in args: {e}", status_code=400)
    target_name = new_name.strip().lower() or profile_name
    # If renamed, check for conflicts and delete the old profile
    if target_name != profile_name:
        if target_name in cfg.profiles:
            return _error_html(f"profile '{target_name}' already exists", status_code=409)
        delete_profile(cfg.config_path, profile_name)
    try:
        save_profile(cfg.config_path, target_name, model.strip(), mmproj.strip(), args)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    _reload_config(request)
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/profiles/{profile_name}/delete", response_class=HTMLResponse)
async def launch_profile_delete(request: Request, profile_name: str,
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    delete_profile(cfg.config_path, profile_name)
    _reload_config(request)
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/defaults", response_class=HTMLResponse)
async def launch_defaults(request: Request,
                          default_model: str = Form(""),
                          default_profile: str = Form(""),
                          _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    update_defaults(
        cfg.config_path,
        default_model=default_model.strip() or None,
        default_profile=default_profile.strip() or None,
    )
    _reload_config(request)
    return RedirectResponse("/ui/launch", status_code=303)

