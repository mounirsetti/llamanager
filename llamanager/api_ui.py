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
    BACKENDS,
    SOURCES,
    InstallState,
    check_for_update,
    current_platform,
    detect_binary,
    detect_default_backend,
    detect_default_source,
    detect_variant_binary,
    detect_variant_for_binary,
    engine_type_for,
    get_engine_hint,
    install_variant,
    list_variants,
    parse_variant_id,
    patch_config_binary,
    read_install_meta,
    variant_id,
    variant_install_path,
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

    # Collect per-process memory; separate llamanager processes from the rest
    LLAMA_NAMES = {"llama-server", "llamanager", "llama_server"}
    try:
        procs = []
        llama_mem_bytes = 0
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                mi = p.info["memory_info"]
                if mi is None:
                    continue
                name = p.info["name"] or ""
                rss = mi.rss
                if name in LLAMA_NAMES:
                    llama_mem_bytes += rss
                else:
                    procs.append({"name": name, "pid": p.info["pid"], "mem_mb": round(rss / (1024**2), 1)})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
                continue
        procs.sort(key=lambda x: x["mem_mb"], reverse=True)
        info["top_mem_procs"] = procs[:5]
    except Exception:
        llama_mem_bytes = 0
        info["top_mem_procs"] = []

    info["llama_mem_gb"] = round(llama_mem_bytes / (1024**3), 1)
    # Effective available = OS available + what llamanager/llama-server is using
    info["ram_effective_avail_gb"] = round((mem.available + llama_mem_bytes) / (1024**3), 1)

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
        # Try nvidia-smi for CUDA GPUs, then rocm-smi for AMD GPUs
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
            # Try AMD ROCm
            try:
                rocm = subprocess.check_output(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    text=True, timeout=5,
                ).strip()
                # CSV format: GPU, VRAM Total Used (B), VRAM Total (B)
                for line in rocm.splitlines()[1:]:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        total_b = int(parts[2])
                        used_b = int(parts[1])
                        info["gpu_type"] = "ROCm"
                        info["gpu_vram_total_gb"] = round(total_b / (1024**3), 1)
                        info["gpu_vram_free_gb"] = round((total_b - used_b) / (1024**3), 1)
                        break
                # Get GPU name
                try:
                    name_out = subprocess.check_output(
                        ["rocm-smi", "--showproductname", "--csv"],
                        text=True, timeout=5,
                    ).strip()
                    for line in name_out.splitlines()[1:]:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            info["gpu_name"] = parts[1]
                            break
                except Exception:
                    pass
            except Exception:
                # Try Intel Arc / Data Center GPU via xpu-smi
                try:
                    # Get device name and total VRAM
                    disc = subprocess.check_output(
                        ["xpu-smi", "discovery", "--dump", "1,2,16"],
                        text=True, timeout=5,
                    ).strip()
                    for line in disc.splitlines()[1:]:
                        parts = [p.strip().strip('"') for p in line.split(",")]
                        if len(parts) >= 3:
                            info["gpu_type"] = "Intel Arc (SYCL)"
                            info["gpu_name"] = parts[1]
                            # Total VRAM, e.g. "16384.00 MiB"
                            total_mib = float(parts[2].split()[0])
                            info["gpu_vram_total_gb"] = round(total_mib / 1024, 1)
                            break
                    # Get used VRAM (metric 18 = GPU Memory Used MiB)
                    used_out = subprocess.check_output(
                        ["xpu-smi", "dump", "-d", "0", "-m", "18", "-i", "1", "-n", "1"],
                        text=True, timeout=5,
                    ).strip()
                    for line in used_out.splitlines()[1:]:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            used_mib = float(parts[2])
                            info["gpu_vram_free_gb"] = round(
                                (info.get("gpu_vram_total_gb", 0) * 1024 - used_mib) / 1024, 1
                            )
                            break
                except Exception:
                    info["gpu_type"] = "No compatible GPU detected"

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

    # Group profiles by parent model so the template can render them inline
    # under each model row. The "global" bucket holds profiles whose `model`
    # field is empty — these are the args-only fallbacks.
    profiles_by_model: dict[str, list[dict]] = {}
    global_profiles: list[dict] = []
    for p in cfg.profiles.values():
        entry = {
            "name": p.name,
            "model": p.model,
            "mmproj": p.mmproj,
            "args": p.args,
            "args_json": json.dumps(p.args) if p.args else "",
        }
        if p.model:
            profiles_by_model.setdefault(p.model, []).append(entry)
        else:
            global_profiles.append(entry)
    for v in profiles_by_model.values():
        v.sort(key=lambda e: e["name"])
    global_profiles.sort(key=lambda e: e["name"])

    models = []
    for m in reg.list():
        d = m.to_dict()
        d["profiles"] = profiles_by_model.get(m.model_id, [])
        d["default_profile"] = cfg.model_default_profiles.get(m.model_id, "")
        models.append(d)

    return _ctx(
        request,
        models=models,
        downloads=reg.list_downloads(),
        current_model=request.app.state.sm.runtime.current_model,
        current_profile=request.app.state.sm.runtime.current_profile,
        models_dir=str(cfg.models_dir),
        open_label=open_label,
        global_profiles=global_profiles,
        global_default_profile=cfg.default_profile,
        default_model=cfg.default_model,
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


# ---------- model search / browse (Hugging Face) ----------

def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


@router.get("/models/search", response_class=HTMLResponse)
async def models_search(request: Request, q: str = "",
                        _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Search Hugging Face for GGUF models.  Returns an HTML partial."""
    import httpx as _httpx

    async def _do_search(query: str) -> list[dict]:
        params: dict[str, str] = {
            "filter": "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": "20",
        }
        if query.strip():
            params["search"] = query.strip()
        async with _httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://huggingface.co/api/models", params=params,
            )
            r.raise_for_status()
            return r.json()

    try:
        raw = await _do_search(q)
        results = [
            {
                "id": m.get("id", ""),
                "author": m.get("author", m.get("id", "").split("/")[0]),
                "downloads": m.get("downloads", 0),
                "downloads_fmt": _fmt_count(m.get("downloads", 0)),
                "likes": m.get("likes", 0),
            }
            for m in raw
        ]
    except Exception as e:
        log.warning("HF search failed: %s", e)
        return templates.TemplateResponse(
            request, "_model_search_results.html",
            _ctx(request, search_results=[], search_error=str(e), search_query=q),
        )

    return templates.TemplateResponse(
        request, "_model_search_results.html",
        _ctx(request, search_results=results, search_error=None, search_query=q),
    )


@router.get("/models/browse", response_class=HTMLResponse)
async def models_browse(request: Request, repo: str = "",
                        _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """List GGUF files in a Hugging Face repo.  Returns an HTML partial."""
    import httpx as _httpx

    if not repo.strip():
        return _error_html("repo is required", status_code=400)

    async def _do_browse(repo_id: str) -> list[dict]:
        # The /tree/main endpoint includes file sizes; /api/models/{id} does not.
        async with _httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://huggingface.co/api/models/{repo_id}/tree/main",
            )
            r.raise_for_status()
            return r.json()

    try:
        tree = await _do_browse(repo.strip())
        gguf_files = []
        for entry in tree:
            path = entry.get("path", "")
            if not path.lower().endswith(".gguf"):
                continue
            size = entry.get("size") or 0
            gguf_files.append({
                "name": path,
                "size": size,
                "size_fmt": _fmt_size(size) if size else "unknown",
                "engine_hint": get_engine_hint(repo=repo, filename=path),
            })
        gguf_files.sort(key=lambda f: f["size"])
    except Exception as e:
        log.warning("HF browse failed for %s: %s", repo, e)
        return templates.TemplateResponse(
            request, "_model_files.html",
            _ctx(request, repo=repo, gguf_files=[], browse_error=str(e)),
        )

    engine_hint = get_engine_hint(repo=repo)
    return templates.TemplateResponse(
        request, "_model_files.html",
        _ctx(request, repo=repo, gguf_files=gguf_files,
             browse_error=None, engine_hint=engine_hint),
    )


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


# ---------- profiles (redirect to models — profile CRUD lives there now) ----------

@router.get("/profiles")
async def profiles_redirect(request: Request,
                            _: Origin = Depends(require_admin_ui)) -> Response:
    return RedirectResponse("/ui/models", status_code=301)


# ---------- about ----------


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string like '0.2.1' into (major, minor, patch).

    Pre-release suffixes (e.g. '0.2.1-beta', '0.2.1-rc1') are ignored
    so comparison is purely on the numeric core.
    """
    import re as _re
    # Take only the leading dotted-number portion (e.g. "0.2.1" from "0.2.1-rc1")
    m = _re.match(r"(\d+(?:\.\d+)*)", v.strip())
    if not m:
        return (0,)
    return tuple(int(p) for p in m.group(1).split("."))


def _version_newer(remote: str, local: str) -> bool:
    """Return True only if *remote* is strictly greater than *local*."""
    return _parse_version(remote) > _parse_version(local)


from importlib.metadata import version as _pkg_version
LLAMANAGER_VERSION = _pkg_version("llamanager")
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
        latest: str | None = None

        # Try releases first, then fall back to tags
        for api_path in ("releases/latest", "tags"):
            req = urllib.request.Request(
                f"https://api.github.com/repos/{GITHUB_REPO}/{api_path}",
                headers={"User-Agent": "llamanager",
                         "Accept": "application/vnd.github+json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    import json as _json
                    data = _json.loads(r.read())
            except urllib.error.HTTPError as he:
                if he.code == 404 and api_path == "releases/latest":
                    continue  # no releases yet — try tags
                raise

            if api_path == "releases/latest":
                latest = data.get("tag_name", "").lstrip("v") or None
            else:
                # tags endpoint returns a list
                if isinstance(data, list) and data:
                    latest = data[0].get("name", "").lstrip("v") or None
            if latest:
                break

        if not latest:
            raise ValueError("no releases or tags found on GitHub")
        is_newer = _version_newer(latest, LLAMANAGER_VERSION)
        return templates.TemplateResponse(request, "_update_area.html", _about_ctx(
            request,
            update_available=is_newer,
            latest_version=latest,
        ))
    except Exception as e:
        return templates.TemplateResponse(request, "_update_area.html", _about_ctx(
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
    # Each profile carries its bound model id so the template can show
    # `(global)` and the JS can route the right model+profile header pair.
    profiles = [{"name": p.name, "model": p.model} for p in cfg.profiles.values()]
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


def _resolve_variant(source: str, backend: str) -> tuple[str, str]:
    """Validate (source, backend) — fall back to sensible defaults."""
    if source not in SOURCES:
        source = detect_default_source()
    plat = current_platform()
    allowed = BACKENDS.get(backend, {}).get("sources")
    if (backend not in BACKENDS
            or plat not in BACKENDS[backend]["platforms"]
            or (allowed is not None and source not in allowed)):
        # Pick the first backend valid for this (source, platform).
        for be_id, be_meta in BACKENDS.items():
            if plat not in be_meta["platforms"]:
                continue
            be_allowed = be_meta.get("sources")
            if be_allowed is not None and source not in be_allowed:
                continue
            backend = be_id
            break
        else:
            backend = "cpu"
    return source, backend


def _setup_ctx(request: Request,
               selected_source: str | None = None,
               selected_backend: str | None = None) -> dict:
    import sys as _sys
    cfg = request.app.state.cfg
    states: dict[str, InstallState] = request.app.state.install_states
    if _sys.platform == "darwin":
        open_config_label = "Open in Finder"
    elif _sys.platform == "win32":
        open_config_label = "Open in Explorer"
    else:
        open_config_label = "Open folder"

    active_binary = detect_binary(cfg.llama_server_binary)
    active_variant = detect_variant_for_binary(active_binary) if active_binary else None

    # Build the variant catalogue for the dropdowns + installed list. The
    # per-variant view is purely presentational; the source of truth for
    # install state lives in app.state.install_states.
    variants = list_variants()
    variants_by_id: dict[str, dict] = {}
    for v in variants:
        bin_path = detect_variant_binary(v["source"], v["backend"])
        installed_meta = read_install_meta(v["source"], v["backend"]) or {}
        variants_by_id[v["id"]] = {
            **v,
            "installed": bin_path is not None,
            "binary_path": bin_path,
            "installed_version": installed_meta.get("version"),
            "install": states[v["id"]].to_dict() if v["id"] in states else InstallState().to_dict(),
            "active": (active_variant is not None
                       and active_variant == (v["source"], v["backend"])),
        }

    # Group by source for the UI: each source row lists its backends.
    sources_view: list[dict] = []
    for src_id, src_meta in SOURCES.items():
        backends_for_src = [variants_by_id[v["id"]] for v in variants
                            if v["source"] == src_id]
        if not backends_for_src:
            continue
        sources_view.append({
            "id": src_id,
            **src_meta,
            "backends": backends_for_src,
        })

    suggested_source = detect_default_source()
    suggested_backend = detect_default_backend()
    sel_source, sel_backend = _resolve_variant(
        selected_source or suggested_source,
        selected_backend or suggested_backend,
    )

    # Available backends for the currently selected source.
    plat = current_platform()
    backends_for_selected: list[dict] = []
    for be_id, be_meta in BACKENDS.items():
        if plat not in be_meta["platforms"]:
            continue
        allowed = be_meta.get("sources")
        if allowed is not None and sel_source not in allowed:
            continue
        backends_for_selected.append({"id": be_id, **be_meta})

    # The currently-running progress block (if any) — pick the most recently
    # touched state so the UI shows the install the user just started.
    progress_state = None
    progress_variant = None
    for vid, st in states.items():
        if st.status == "running":
            progress_state = st
            progress_variant = vid
            break
    if progress_state is None:
        vid = variant_id(sel_source, sel_backend)
        if vid in states:
            progress_state = states[vid]
            progress_variant = vid

    updates = getattr(request.app.state, "install_updates", {}) or {}

    return _ctx(
        request,
        binary_path=active_binary,
        active_variant_id=variant_id(*active_variant) if active_variant else None,
        configured_binary=cfg.llama_server_binary,
        engine=getattr(cfg, "llama_server_engine", "llama"),
        sources=sources_view,
        backends=BACKENDS,
        backends_for_selected=backends_for_selected,
        selected_source=sel_source,
        selected_backend=sel_backend,
        suggested_source=suggested_source,
        suggested_backend=suggested_backend,
        platform_tag=plat,
        machine=platform.machine(),
        install=progress_state.to_dict() if progress_state else InstallState().to_dict(),
        install_variant=progress_variant,
        updates=updates,
        config_dir=str(cfg.config_path.parent),
        config_file=str(cfg.config_path),
        open_config_label=open_config_label,
    )


@router.post("/setup/install", response_class=HTMLResponse)
async def setup_install(request: Request,
                        source: str = Form("llama.cpp"),
                        backend: str = Form(""),
                        _: None = Depends(require_csrf)) -> Response:
    source, backend = _resolve_variant(source, backend)
    vid = variant_id(source, backend)
    states: dict[str, InstallState] = request.app.state.install_states
    state = states.setdefault(vid, InstallState())
    if state.status != "running":
        state.status = "running"
        state.lines = []
        state.error = None
        state.installed_path = None
        asyncio.create_task(install_variant(state, source, backend))
    return templates.TemplateResponse(
        request, "setup.html",
        _setup_ctx(request, selected_source=source, selected_backend=backend),
    )


@router.get("/setup/_install_progress", response_class=HTMLResponse)
async def setup_install_progress(request: Request,
                                 variant: str = "",
                                 _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    states: dict[str, InstallState] = request.app.state.install_states
    state = states.get(variant)
    if state is None:
        return templates.TemplateResponse(request, "_install_progress.html", _ctx(
            request,
            install=InstallState().to_dict(),
            binary_path=detect_binary(request.app.state.cfg.llama_server_binary),
            install_variant=variant,
        ))

    cfg = request.app.state.cfg
    # When an install finishes, point the configured binary at it so the new
    # variant becomes active immediately. Engine type is set in lockstep so
    # server_manager builds the right cmdline.
    if state.status == "done" and state.installed_path:
        if cfg.llama_server_binary != state.installed_path:
            cfg.llama_server_binary = state.installed_path
            parsed = parse_variant_id(variant)
            engine = engine_type_for(parsed[0]) if parsed else "llama"
            cfg.llama_server_engine = engine
            patch_config_binary(cfg.config_path, state.installed_path, engine=engine)
    binary_path = detect_binary(cfg.llama_server_binary)
    return templates.TemplateResponse(request, "_install_progress.html", _ctx(
        request,
        install=state.to_dict(),
        binary_path=binary_path,
        install_variant=variant,
    ))


@router.post("/setup/switch-variant", response_class=HTMLResponse)
async def setup_switch_variant(request: Request,
                               variant: str = Form(...),
                               _: None = Depends(require_csrf)) -> Response:
    """Switch the active engine to a previously-installed variant."""
    cfg = request.app.state.cfg
    parsed = parse_variant_id(variant)
    if parsed is not None:
        source, backend = parsed
        path = variant_install_path(source, backend)
        if path.exists():
            engine = engine_type_for(source)
            cfg.llama_server_binary = str(path)
            cfg.llama_server_engine = engine
            patch_config_binary(cfg.config_path, str(path), engine=engine)
    return RedirectResponse("/ui/setup", status_code=303)


@router.get("/setup/check-updates", response_class=HTMLResponse)
async def setup_check_updates(request: Request,
                              variant: str = "",
                              _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Check upstream for a newer release of a specific variant (or all
    installed variants if ``variant`` is empty). Renders the updated
    variants list partial.
    """
    parsed = parse_variant_id(variant) if variant else None
    updates: dict[str, dict] = {}
    loop = asyncio.get_running_loop()
    if parsed is not None:
        source, backend = parsed
        info = await loop.run_in_executor(None, check_for_update, source, backend)
        updates[variant] = info.to_dict()
    else:
        for v in list_variants():
            if detect_variant_binary(v["source"], v["backend"]) is None:
                continue
            info = await loop.run_in_executor(
                None, check_for_update, v["source"], v["backend"]
            )
            updates[v["id"]] = info.to_dict()
    request.app.state.install_updates = updates
    return templates.TemplateResponse(
        request, "_installed_variants.html",
        _setup_ctx(request) | {"updates": updates},
    )


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
    """Context for the launcher page. Profile management lives on the Models
    page now, so this context exposes only what the launcher needs: the
    inventory of models + their bound profiles for the start-form dropdowns.
    """
    cfg = request.app.state.cfg
    sm: ServerManager = request.app.state.sm
    supervisor: Supervisor = request.app.state.supervisor
    reg: Registry = request.app.state.registry
    model_ids = [m.model_id for m in reg.list()]

    # Group profiles by their bound model for the JS-side filtering of the
    # profile dropdown.
    profiles_by_model: dict[str, list[str]] = {}
    global_profile_names: list[str] = []
    for name, p in cfg.profiles.items():
        if p.model:
            profiles_by_model.setdefault(p.model, []).append(name)
        else:
            global_profile_names.append(name)
    for v in profiles_by_model.values():
        v.sort()
    global_profile_names.sort()

    return _ctx(
        request,
        status=sm.status(),
        model_ids=model_ids,
        profiles_by_model=profiles_by_model,
        global_profile_names=global_profile_names,
        model_default_profiles=cfg.model_default_profiles,
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
                              model: str = Form(""),
                              profile: str = Form(""),
                              _: None = Depends(require_csrf)) -> Response:
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    try:
        spec = resolve_spec(
            cfg,
            model=model or None,
            profile=profile or None,
        )
    except (ServerError, ValueError) as e:
        return _error_html(str(e), status_code=400)
    asyncio.create_task(_start_server_bg(sm, spec))
    await asyncio.sleep(0.3)
    ctx = _launch_ctx(request)
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


# ---------- profile CRUD (model-scoped) ----------
#
# Profiles always live as a child of a model (or as a global "args-only"
# fallback when model_id is empty). All profile-mutating endpoints redirect
# back to /ui/models — the launcher no longer owns profile management.

from .config import (
    clear_model_default_profile as _clear_mdp,
    set_model_default_profile as _set_mdp,
)


@router.post("/models/profiles/create", response_class=HTMLResponse)
async def models_profile_create(request: Request,
                                name: str = Form(...),
                                model_id: str = Form(""),
                                mmproj: str = Form(""),
                                args_json: str = Form("{}"),
                                make_default: str = Form(""),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    try:
        args = json.loads(args_json.strip() or "{}")
    except json.JSONDecodeError as e:
        return _error_html(f"invalid JSON in args: {e}", status_code=400)
    profile_name = name.strip().lower()
    if profile_name in cfg.profiles:
        return _error_html(f"profile '{profile_name}' already exists", status_code=409)
    try:
        save_profile(cfg.config_path, profile_name,
                     model_id.strip(), mmproj.strip(), args)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    if make_default == "on":
        if model_id.strip():
            _set_mdp(cfg.config_path, model_id.strip(), profile_name)
        else:
            update_defaults(cfg.config_path, default_profile=profile_name)
    _reload_config(request)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/profiles/{profile_name}/update", response_class=HTMLResponse)
async def models_profile_update(request: Request, profile_name: str,
                                new_name: str = Form(""),
                                model_id: str = Form(""),
                                mmproj: str = Form(""),
                                args_json: str = Form("{}"),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    if profile_name not in cfg.profiles:
        return _error_html(f"profile '{profile_name}' not found", status_code=404)
    try:
        args = json.loads(args_json.strip() or "{}")
    except json.JSONDecodeError as e:
        return _error_html(f"invalid JSON in args: {e}", status_code=400)
    target_name = new_name.strip().lower() or profile_name
    if target_name != profile_name:
        if target_name in cfg.profiles:
            return _error_html(f"profile '{target_name}' already exists",
                               status_code=409)
        delete_profile(cfg.config_path, profile_name)
    try:
        save_profile(cfg.config_path, target_name,
                     model_id.strip(), mmproj.strip(), args)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    _reload_config(request)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/profiles/{profile_name}/delete", response_class=HTMLResponse)
async def models_profile_delete(request: Request, profile_name: str,
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    delete_profile(cfg.config_path, profile_name)
    _reload_config(request)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/profiles/set-model-default", response_class=HTMLResponse)
async def models_set_model_default(request: Request,
                                   model_id: str = Form(...),
                                   profile_name: str = Form(""),
                                   _: None = Depends(require_csrf)) -> Response:
    """Set the default profile for a specific model. Empty profile_name
    clears the per-model default (falling back to the global default)."""
    cfg = request.app.state.cfg
    mid = model_id.strip()
    pname = profile_name.strip()
    if not mid:
        return _error_html("model_id required", status_code=400)
    if pname:
        prof = cfg.profiles.get(pname)
        if not prof:
            return _error_html(f"unknown profile: {pname}", status_code=400)
        if prof.model and prof.model != mid:
            return _error_html(
                f"profile '{pname}' is bound to '{prof.model}', not '{mid}'",
                status_code=400,
            )
        _set_mdp(cfg.config_path, mid, pname)
    else:
        _clear_mdp(cfg.config_path, mid)
    _reload_config(request)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/profiles/set-global-default", response_class=HTMLResponse)
async def models_set_global_default(request: Request,
                                    profile_name: str = Form(""),
                                    _: None = Depends(require_csrf)) -> Response:
    """Set or clear the global default profile (the args-only fallback
    applied when neither the request nor the model has a profile)."""
    cfg = request.app.state.cfg
    pname = profile_name.strip()
    if pname:
        prof = cfg.profiles.get(pname)
        if not prof:
            return _error_html(f"unknown profile: {pname}", status_code=400)
        if prof.model:
            return _error_html(
                f"profile '{pname}' is bound to '{prof.model}' and cannot be "
                "used as the global default. Create a model-less profile to "
                "use as the global default.",
                status_code=400,
            )
    update_defaults(cfg.config_path, default_profile=pname)
    _reload_config(request)
    return RedirectResponse("/ui/models", status_code=303)


@router.post("/models/set-default", response_class=HTMLResponse)
async def models_set_default(request: Request,
                             model_id: str = Form(""),
                             _: None = Depends(require_csrf)) -> Response:
    """Set the configured default model. Empty model_id clears it."""
    cfg = request.app.state.cfg
    update_defaults(cfg.config_path, default_model=model_id.strip())
    _reload_config(request)
    return RedirectResponse("/ui/models", status_code=303)

