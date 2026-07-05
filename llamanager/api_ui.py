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
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
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
    ENGINE_FAMILY, Profile, VALID_FLASH_ATTN, VALID_KV_CACHE_TYPES,
    VALID_RAM_SPILL_POLICIES,
    VALID_THINKING,
    delete_profile, detect_engine_for_id, load_config, rename_profile,
    save_profile, set_default_args, set_model_default_profile,
    update_coexistence_policy, update_defaults, update_image_config,
)
from . import engines as image_engines
from . import diffusion_catalog
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
from .server_manager import ServerManager, resolve_spec, ServerError, _engine_env
from .supervisor import Supervisor

router = APIRouter(prefix="/ui", tags=["ui"])

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

KREA_LORA_COLLECTION_API = (
    "https://huggingface.co/api/collections/krea/krea-2-loras"
)
HF_MODEL_TREE_API = "https://huggingface.co/api/models/{repo}/tree/main"


def _localdt(ts: Any, fmt: str = "%b %d %H:%M") -> str:
    """Format a unix timestamp in the server's local time. Empty for
    falsy/invalid input so templates can use ``{{ ts | localdt }}`` freely."""
    try:
        if not ts:
            return ""
        return time.strftime(fmt, time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return ""


templates.env.filters["localdt"] = _localdt

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


def _current_session(request: Request) -> dict[str, Any] | None:
    """Return the admin session resolved by ``require_admin_ui``.

    Always prefer this over ``_read_session`` from inside a handler that
    depends on ``require_admin_ui`` — on the remember-me restore path the
    new session cookie hasn't been delivered to the client yet, so
    ``_read_session`` would return ``None`` and the template's
    ``api_key`` would render empty. ``require_admin_ui`` stashes the
    resolved session on ``request.state.session`` for us.
    """
    cached = getattr(request.state, "session", None)
    if cached is not None:
        return cached
    return _read_session(request)


def _wants_htmx(request: Request) -> bool:
    """True when the request was issued by htmx (vs a full-page navigation)."""
    return request.headers.get("hx-request") == "true"


def _auth_redirect(request: Request, location: str) -> HTTPException:
    """Build a redirect-to-login that works for both htmx and plain requests.

    A plain 3xx is transparently followed by htmx's fetch and the login HTML
    gets swapped into the current page, which is broken. For htmx we instead
    return HX-Redirect so the browser does a real top-level navigation.
    """
    if _wants_htmx(request):
        return HTTPException(status_code=200, headers={"HX-Redirect": location})
    return HTTPException(status_code=302, headers={"Location": location})


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
                # Stash the freshly-built session so handlers calling
                # ``_current_session`` get the right api_key on this same
                # request (before the new cookie has been delivered).
                request.state.session = _session_store(request).get(sid)
                return origin

    if not sess:
        raise _auth_redirect(request, "/ui/login")
    am_: AuthManager = request.app.state.auth
    origin = await am_.verify(sess["key"])
    if not origin or not origin.is_admin:
        _session_store(request).delete(request.cookies.get(COOKIE_NAME))
        raise _auth_redirect(request, "/ui/login")
    request.state.csrf_token = sess["csrf"]
    request.state.session = sess
    return origin


# ---------- CSRF helpers ----------

def _normalise_host(netloc: str) -> str:
    """Treat localhost and 127.0.0.1 as the same host."""
    return netloc.lower().replace("localhost", "127.0.0.1")


def _same_origin(request: Request) -> bool:
    """Best-effort same-origin check on the Origin/Referer header.

    We set Referrer-Policy: no-referrer, which strips the Referer header and
    changes how browsers report the origin on same-origin form POSTs: Chrome
    sends Origin: null, while Firefox omits the Origin header entirely. In
    either case there is no usable origin to compare against, so we skip the
    origin check and rely solely on the per-session CSRF token, which is the
    real protection against forged cross-origin requests.
    """
    origin = request.headers.get("origin")
    # "null" origin is sent by browsers when referrer policy suppresses it.
    # Treat it as same-origin; CSRF token validation is the actual guard.
    if origin == "null":
        return True
    ref = origin or request.headers.get("referer")
    if not ref:
        # Firefox omits Origin (rather than sending "null") on same-origin
        # POSTs under our no-referrer policy, and Referer is stripped too. With
        # no header to compare, defer to the CSRF token downstream — the same
        # treatment as the origin == "null" case above.
        return True
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

    Must be added to every state-changing UI route. On a recoverable failure
    (the session was rotated out from under an already-rendered page, leaving a
    stale CSRF token) we tell the browser to reload/redirect instead of
    returning a dead 403 the user can only escape by clearing cookies.
    """
    sess = _current_session(request)
    if not sess:
        # Session expired or rotated away entirely: bounce to login.
        raise _auth_redirect(request, "/ui/login")
    if not _same_origin(request):
        # Genuine cross-origin POST — a real security signal, not a stale
        # cookie. Keep it a hard failure.
        raise HTTPException(status_code=403, detail="bad origin")
    token = await _extract_csrf_token(request)
    if not token or not secrets.compare_digest(str(token), sess["csrf"]):
        # The session is valid but the page was rendered before a session
        # rotation, so its embedded token is stale. Reload the page to pick up
        # a fresh token (rendered from sess["csrf"]) so the user can retry,
        # rather than dead-ending on a 403.
        if _wants_htmx(request):
            raise HTTPException(status_code=200, headers={"HX-Refresh": "true"})
        raise HTTPException(
            status_code=303,
            headers={"Location": request.headers.get("referer") or "/ui/setup"},
        )


_VRAM_USAGE_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_PER_PROCESS_VRAM_CACHE: dict[str, Any] = {"at": 0.0, "value": {}}


def _pdh_query(counter_path: str) -> list[tuple[str, int]] | None:
    """Run a single PDH wildcard query and return (instance_name, value)
    pairs. Uses ``PdhAddEnglishCounterW`` so the path is locale-independent.
    Returns ``None`` on any failure."""
    try:
        import ctypes
        from ctypes import POINTER, byref, c_int64, c_uint32, c_void_p, c_wchar_p
        from ctypes.wintypes import DWORD, LPCWSTR

        pdh = ctypes.WinDLL("pdh")
        PDH_FMT_LARGE = 0x00000400
        PDH_MORE_DATA = 0x800007D2

        class PDH_FMT_COUNTERVALUE(ctypes.Structure):
            _fields_ = [("CStatus", DWORD), ("largeValue", c_int64)]

        class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
            _fields_ = [("szName", c_wchar_p),
                        ("FmtValue", PDH_FMT_COUNTERVALUE)]

        PdhOpenQueryW = pdh.PdhOpenQueryW
        PdhOpenQueryW.argtypes = [LPCWSTR, c_void_p, POINTER(c_void_p)]
        PdhOpenQueryW.restype = c_uint32
        PdhAddEnglishCounterW = pdh.PdhAddEnglishCounterW
        PdhAddEnglishCounterW.argtypes = [c_void_p, LPCWSTR, c_void_p,
                                          POINTER(c_void_p)]
        PdhAddEnglishCounterW.restype = c_uint32
        PdhCollectQueryData = pdh.PdhCollectQueryData
        PdhCollectQueryData.argtypes = [c_void_p]
        PdhCollectQueryData.restype = c_uint32
        PdhGetFormattedCounterArrayW = pdh.PdhGetFormattedCounterArrayW
        PdhGetFormattedCounterArrayW.argtypes = [c_void_p, DWORD,
                                                 POINTER(DWORD),
                                                 POINTER(DWORD), c_void_p]
        PdhGetFormattedCounterArrayW.restype = c_uint32
        PdhCloseQuery = pdh.PdhCloseQuery
        PdhCloseQuery.argtypes = [c_void_p]
        PdhCloseQuery.restype = c_uint32

        query = c_void_p()
        if PdhOpenQueryW(None, 0, byref(query)) != 0:
            return None
        try:
            counter = c_void_p()
            if PdhAddEnglishCounterW(query, counter_path, 0,
                                     byref(counter)) != 0:
                return None
            if PdhCollectQueryData(query) != 0:
                return None
            buf_size = DWORD(0)
            item_count = DWORD(0)
            rc = PdhGetFormattedCounterArrayW(
                counter, PDH_FMT_LARGE,
                byref(buf_size), byref(item_count), None,
            )
            if rc != PDH_MORE_DATA or buf_size.value == 0:
                return None
            buf = (ctypes.c_ubyte * buf_size.value)()
            rc = PdhGetFormattedCounterArrayW(
                counter, PDH_FMT_LARGE,
                byref(buf_size), byref(item_count),
                ctypes.cast(buf, c_void_p),
            )
            if rc != 0:
                return None
            items = ctypes.cast(buf, POINTER(PDH_FMT_COUNTERVALUE_ITEM_W))
            out: list[tuple[str, int]] = []
            for i in range(item_count.value):
                it = items[i]
                if it.FmtValue.CStatus != 0:
                    continue
                out.append((it.szName or "", int(it.FmtValue.largeValue)))
            return out
        finally:
            PdhCloseQuery(query)
    except Exception:
        return None


def _windows_per_process_vram() -> dict[int, int]:
    """Map of PID → dedicated VRAM bytes, summed across GPU engines/adapters
    on Windows. Cached for 2 s. Empty dict if the counter isn't available."""
    now = time.monotonic()
    if now - _PER_PROCESS_VRAM_CACHE["at"] < 2.0:
        return _PER_PROCESS_VRAM_CACHE["value"]

    by_pid: dict[int, int] = {}
    rows = _pdh_query(r"\GPU Process Memory(*)\Dedicated Usage")
    if rows:
        import re
        pid_re = re.compile(r"pid_(\d+)")
        for name, value in rows:
            m = pid_re.search(name)
            if not m or value <= 0:
                continue
            pid = int(m.group(1))
            by_pid[pid] = by_pid.get(pid, 0) + value
    _PER_PROCESS_VRAM_CACHE.update({"at": now, "value": by_pid})
    return by_pid


def _windows_vram_usage_bytes() -> int | None:
    """System-wide dedicated-VRAM usage in bytes, summed across all
    GPU adapters. Uses Windows Performance Data Helper (PDH) with the
    locale-independent English counter API — works on Windows 10+ for
    any GPU vendor (NVIDIA, AMD, Intel) without vendor SDKs.

    Returns ``None`` if the counter is unavailable (older Windows,
    counter not registered, etc.). Result cached for 2 s.

    Note: DXGI's ``QueryVideoMemoryInfo`` is per-process, not system-wide
    — Task Manager itself uses the same perfcounter source we query here.
    """
    now = time.monotonic()
    if now - _VRAM_USAGE_CACHE["at"] < 2.0:
        return _VRAM_USAGE_CACHE["value"]
    rows = _pdh_query(r"\GPU Adapter Memory(*)\Dedicated Usage")
    used: int | None = None
    if rows is not None:
        used = sum(v for _, v in rows if v > 0)
    _VRAM_USAGE_CACHE.update({"at": now, "value": used})
    return used


def _linux_per_process_vram() -> dict[int, int]:
    """Return ``{pid: vram_bytes}`` for processes using the GPU on Linux.

    Tries ``nvidia-smi --query-compute-apps`` first, then ``rocm-smi
    --showpids --json``, then a CSV/text fallback. Empty dict if no
    vendor CLI is available or no compute processes are running.
    """
    out: dict[int, int] = {}

    # --- NVIDIA path ------------------------------------------------------
    try:
        nv = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for line in nv.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                mib = int(parts[1])  # used_memory is reported in MiB
            except ValueError:
                continue
            if pid > 0 and mib > 0:
                out[pid] = mib * 1024 * 1024
        if out:
            return out
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    # --- AMD ROCm path (JSON) --------------------------------------------
    # `rocm-smi --showpids --json` prints either a single dict keyed by
    # PID strings, or {"WARNING": "No JSON data to report"} when idle.
    try:
        rj = subprocess.check_output(
            ["rocm-smi", "--showpids", "--json"],
            text=True, timeout=5,
        ).strip()
        if rj and not rj.startswith("WARNING"):
            try:
                data = json.loads(rj)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        pid = int(k)
                    except (TypeError, ValueError):
                        continue
                    # Header names differ across versions: try a few.
                    raw = None
                    for key in ("VRAM USED",
                                "VRAM used",
                                "VRAM Used (B)",
                                "VRAM USED (B)",
                                "VRAM used (B)"):
                        if key in v:
                            raw = v[key]
                            break
                    if raw is None:
                        continue
                    try:
                        used = int(str(raw).strip())
                    except (TypeError, ValueError):
                        continue
                    if used > 0:
                        out[pid] = used
                if out:
                    return out
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    # --- AMD ROCm path (CSV fallback) ------------------------------------
    try:
        rc = subprocess.check_output(
            ["rocm-smi", "--showpids", "--csv"],
            text=True, timeout=5,
        ).strip()
        if rc and not rc.startswith("WARNING"):
            lines = rc.splitlines()
            if len(lines) >= 2:
                header = [h.strip().lower() for h in lines[0].split(",")]
                pid_idx = next(
                    (i for i, h in enumerate(header) if h == "pid"),
                    None,
                )
                vram_idx = next(
                    (i for i, h in enumerate(header)
                     if "vram" in h and "used" in h),
                    None,
                )
                for line in lines[1:]:
                    parts = [p.strip() for p in line.split(",")]
                    if pid_idx is None or vram_idx is None:
                        break
                    if max(pid_idx, vram_idx) >= len(parts):
                        continue
                    try:
                        pid = int(parts[pid_idx])
                        used = int(parts[vram_idx])
                    except ValueError:
                        continue
                    if pid > 0 and used > 0:
                        out[pid] = used
                if out:
                    return out
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    # Older rocm-smi versions (pre-5.x) only emit the human-readable
    # table without --json/--csv. We don't try to parse that — the
    # multi-word column headers ("VRAM USED") don't align with
    # whitespace-split data tokens, which makes the fallback fragile
    # and gives wrong numbers. If both JSON and CSV come back empty
    # on those rare hosts, the dashboard simply omits the panel.
    return out


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

    # On Linux, platform.processor() is usually empty or just "x86_64".
    # The marketing name lives in /proc/cpuinfo under "model name".
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        _, _, val = line.partition(":")
                        val = val.strip()
                        if val:
                            info["cpu"] = val
                        break
        except OSError:
            pass

    # On Windows, platform.processor() returns the bare family/model
    # identifier ("Intel64 Family 6 Model 198 Stepping 2, GenuineIntel").
    # The marketing name lives in the registry.
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                name = (name or "").strip()
                if name:
                    info["cpu"] = name
        except OSError:
            pass

    # RAM
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024**3), 1)
    info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    info["ram_used_pct"] = mem.percent

    # Collect per-process RAM; separate llamanager processes from the rest.
    # Also build a (pid → name) lookup we can reuse for the per-process VRAM
    # table below.
    LLAMA_NAMES = {"llama-server", "llamanager", "llama_server"}
    pid_names: dict[int, str] = {}
    try:
        procs = []
        llama_mem_bytes = 0
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                mi = p.info["memory_info"]
                if mi is None:
                    continue
                name = p.info["name"] or ""
                pid = p.info["pid"]
                pid_names[pid] = name
                rss = mi.rss
                if name in LLAMA_NAMES:
                    llama_mem_bytes += rss
                else:
                    procs.append({"name": name, "pid": pid,
                                  "mem_mb": round(rss / (1024**2), 1)})
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess, AttributeError):
                continue
        procs.sort(key=lambda x: x["mem_mb"], reverse=True)
        info["top_ram_procs"] = procs[:5]
    except Exception:
        llama_mem_bytes = 0
        info["top_ram_procs"] = []
    # Back-compat alias for any external template that still reads this.
    info["top_mem_procs"] = info["top_ram_procs"]

    # Per-process VRAM. Windows uses PDH GPU perfcounters; Linux tries
    # nvidia-smi compute-apps then rocm-smi --showpids. macOS doesn't
    # expose per-process VRAM through any first-party tool we trust.
    vram_procs: list[dict[str, Any]] = []
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            by_pid = _windows_per_process_vram()
        elif sys_name == "Linux":
            by_pid = _linux_per_process_vram()
        else:
            by_pid = {}
        rows = []
        for pid, used_bytes in by_pid.items():
            if used_bytes <= 0:
                continue
            name = pid_names.get(pid)
            if not name:
                # Process may have started after our psutil sweep, or
                # we don't have permission to read its name.
                try:
                    name = psutil.Process(pid).name()
                except (psutil.NoSuchProcess, psutil.AccessDenied,
                        psutil.ZombieProcess):
                    name = f"pid {pid}"
            rows.append({
                "name": name,
                "pid": pid,
                "mem_mb": round(used_bytes / (1024**2), 1),
            })
        rows.sort(key=lambda x: x["mem_mb"], reverse=True)
        vram_procs = rows[:5]
    except Exception:
        vram_procs = []
    info["top_vram_procs"] = vram_procs

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
                # rocm-smi CSV column order has flipped across versions
                # (older: "VRAM Total Used, VRAM Total"; newer 6.x/7.x:
                # "VRAM Total Memory, VRAM Total Used Memory"). Match
                # by header substring so we don't silently mis-label.
                lines = rocm.splitlines()
                if len(lines) >= 2:
                    header = [h.strip().lower() for h in lines[0].split(",")]
                    total_idx = next(
                        (i for i, h in enumerate(header)
                         if "total" in h and "used" not in h),
                        None,
                    )
                    used_idx = next(
                        (i for i, h in enumerate(header) if "used" in h),
                        None,
                    )
                    for line in lines[1:]:
                        parts = [p.strip() for p in line.split(",")]
                        if total_idx is None or used_idx is None:
                            break
                        if max(total_idx, used_idx) >= len(parts):
                            continue
                        try:
                            total_b = int(parts[total_idx])
                            used_b = int(parts[used_idx])
                        except ValueError:
                            continue
                        if total_b <= 0:
                            continue
                        info["gpu_type"] = "ROCm"
                        info["gpu_vram_total_gb"] = round(total_b / (1024**3), 1)
                        # Clamp free ≥ 0 as a backstop against future
                        # header drift or transient used > total reads.
                        info["gpu_vram_free_gb"] = round(
                            max(0, total_b - used_b) / (1024**3), 1)
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
                    pass

        # Windows fallback: ROCm/xpu-smi aren't shipped on Windows for
        # most users (AMD cards run llama.cpp via Vulkan/HIP-SDK), and
        # nvidia-smi only covers NVIDIA. Read the display-adapter info
        # from the Windows registry so we still surface the card.
        if system == "Windows" and not info.get("gpu_name"):
            try:
                import winreg
                base = (r"SYSTEM\CurrentControlSet\Control\Class"
                        r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
                best: tuple[int, str, int] | None = None  # (priority, name, mem_bytes)
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
                    i = 0
                    while True:
                        try:
                            sub = winreg.EnumKey(root, i)
                        except OSError:
                            break
                        i += 1
                        if not sub.isdigit():
                            continue
                        try:
                            with winreg.OpenKey(root, sub) as k:
                                name = winreg.QueryValueEx(k, "DriverDesc")[0]
                                mem_bytes = 0
                                # qwMemorySize is 64-bit (reliable for >4 GB)
                                for key_name in ("HardwareInformation.qwMemorySize",
                                                 "HardwareInformation.MemorySize"):
                                    try:
                                        raw = winreg.QueryValueEx(k, key_name)[0]
                                        if isinstance(raw, bytes):
                                            mem_bytes = int.from_bytes(raw, "little")
                                        else:
                                            mem_bytes = int(raw)
                                            if mem_bytes < 0:  # 32-bit signed → wrap
                                                mem_bytes += 2**32
                                        if mem_bytes > 0:
                                            break
                                    except (FileNotFoundError, OSError, ValueError):
                                        continue
                        except OSError:
                            continue
                        low = name.lower()
                        # Prefer discrete GPUs over integrated/basic display
                        if "nvidia" in low or "geforce" in low or "quadro" in low:
                            pri = 0
                        elif ("amd" in low or "radeon" in low
                              or low.startswith("ati ") or "rx " in low):
                            pri = 0
                        elif "arc" in low:
                            pri = 1
                        elif "intel" in low:
                            pri = 2
                        elif "microsoft basic" in low or "remote" in low:
                            continue
                        else:
                            pri = 3
                        if best is None or pri < best[0]:
                            best = (pri, name, mem_bytes)
                if best is not None:
                    _, name, mem_bytes = best
                    low = name.lower()
                    if "nvidia" in low or "geforce" in low or "quadro" in low:
                        info["gpu_type"] = "NVIDIA (driver)"
                    elif ("amd" in low or "radeon" in low
                          or low.startswith("ati ") or "rx " in low):
                        info["gpu_type"] = "AMD (Vulkan/HIP)"
                    elif "arc" in low:
                        info["gpu_type"] = "Intel Arc"
                    elif "intel" in low:
                        info["gpu_type"] = "Intel iGPU"
                    else:
                        info["gpu_type"] = "GPU"
                    info["gpu_name"] = name
                    if mem_bytes > 0:
                        info["gpu_vram_total_gb"] = round(mem_bytes / (1024**3), 1)
            except Exception:
                pass

        # Live VRAM usage on Windows — DXGI works for any vendor when we
        # don't have a vendor-specific CLI (rocm-smi/nvidia-smi are rare
        # on Windows for AMD users). Only fill ``gpu_vram_free_gb`` if it
        # isn't already set by an earlier vendor-specific path.
        if (system == "Windows" and "gpu_vram_total_gb" in info
                and "gpu_vram_free_gb" not in info):
            used_bytes = _windows_vram_usage_bytes()
            if used_bytes is not None:
                used_gb = used_bytes / (1024**3)
                free_gb = max(0.0, info["gpu_vram_total_gb"] - used_gb)
                info["gpu_vram_free_gb"] = round(free_gb, 1)

        if not info.get("gpu_name") and info.get("gpu_type") in ("unknown", ""):
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


def _topbar_models(request: Request) -> dict[str, Any]:
    """Build the dataset the sticky top-bar model selector needs:
    text/image model lists, each model's profiles, and the currently
    loaded LLM + saved default-image model."""
    cfg = request.app.state.cfg
    reg: Registry = request.app.state.registry
    sm: ServerManager = request.app.state.sm

    text: list[dict[str, Any]] = []
    image: list[dict[str, Any]] = []
    for entry in reg.list():
        # Same de-clutter as the models list: projectors aren't launchable
        # and only a split GGUF's first shard is the load target.
        if _is_mmproj(entry.model_id):
            continue
        part = _shard_index(entry.model_id)
        if part is not None and part > 1:
            continue
        engine = detect_engine_for_id(entry.model_id, cfg.models_dir)
        m = cfg.get_model(entry.model_id)
        profile_names = sorted(m.profiles.keys()) if m else []
        default_profile = m.default_profile if m else ""
        row = {
            "model_id": entry.model_id,
            "engine": engine,
            "profiles": profile_names,
            "default_profile": default_profile,
        }
        fam = ENGINE_FAMILY.get(engine, "text")
        if fam == "image":
            image.append(row)
        elif fam == "audio":
            # ASR models aren't launchable LLMs — keep them out of the LLM
            # model picker (they have their own page + transcription API).
            continue
        else:
            text.append(row)

    text.sort(key=lambda r: r["model_id"])
    image.sort(key=lambda r: r["model_id"])
    # Multi-slot indicator strip: when the beta is on, the topbar
    # renders one read-only row per slot below the main LLM lane. The
    # views carry just enough to render the strip — id, port, model,
    # and engine state. Empty list when the feature is off.
    topbar_slots: list[dict[str, Any]] = []
    if getattr(cfg, "multi_slot_enabled", False) and hasattr(sm, "slots"):
        for sv in sm.slots():
            topbar_slots.append({
                "id": sv.id,
                "port": sv.port,
                "model": sv.model,
                "profile": sv.profile,
                "state": sv.state,
            })
    return {
        "topbar_text_models": text,
        "topbar_image_models": image,
        "topbar_current_llm": sm.runtime.current_model or "",
        "topbar_current_llm_profile": sm.runtime.current_profile or "",
        "topbar_default_llm": cfg.default_model or "",
        "topbar_default_image": cfg.default_image_model or "",
        "topbar_default_image_profile": cfg.default_image_profile or "",
        "topbar_slots_enabled": bool(getattr(cfg, "multi_slot_enabled", False)),
        "topbar_slots": topbar_slots,
    }


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
    base.update(_topbar_models(request))
    base.update(extra)
    return base


def _error_html(message: str, status_code: int = 400) -> HTMLResponse:
    """Escape arbitrary text into an error fragment. Never use f-strings
    with untrusted data here."""
    safe = html_escape(message, quote=True)
    return HTMLResponse(
        f"<div class='lm-error'>{safe}</div>",
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

def _autostart_installed() -> bool | None:
    """Cheap, file-based check for whether an autostart entry exists. No
    subprocess (the dashboard renders often). Returns None when we can't
    tell cheaply (Windows, where it needs `sc`/`schtasks`)."""
    home = Path.home()
    if sys.platform.startswith("linux"):
        return ((home / ".config/systemd/user/llamanager.service").exists()
                or (home / ".config/autostart/llamanager-tray.desktop").exists())
    if sys.platform == "darwin":
        return (
            (home / "Library/LaunchAgents/com.llamanager.plist").exists()
            or (home / "Library/LaunchAgents/com.llamanager.tray.plist").exists()
            or Path("/Library/LaunchDaemons/com.llamanager.plist").exists()
        )
    return None  # Windows: unknown without sc/schtasks


def _autorun_label() -> str:
    """Cheap, file-based current autorun-at-startup state for the Setup page:
    'off' / 'at login' / 'before login' / 'unknown'. Mirrors the tray's
    detection. 'before login' = linger (Linux) / system LaunchDaemon (macOS)."""
    home = Path.home()
    if sys.platform.startswith("linux"):
        unit = ((home / ".config/systemd/user/llamanager.service").exists() or
                (home / ".config/systemd/user/default.target.wants/"
                        "llamanager.service").exists())
        tray = (home / ".config/autostart/llamanager-tray.desktop").exists()
        if not (unit or tray):
            return "off"
        user = os.environ.get("USER", "")
        linger = bool(user) and Path(f"/var/lib/systemd/linger/{user}").exists()
        return "before login" if linger else "at login"
    if sys.platform == "darwin":
        sysd = Path("/Library/LaunchDaemons/com.llamanager.plist").exists()
        agent = (home / "Library/LaunchAgents/com.llamanager.plist").exists()
        tray = (home / "Library/LaunchAgents/com.llamanager.tray.plist").exists()
        if not (sysd or agent or tray):
            return "off"
        return "before login" if sysd else "at login"
    return "unknown"


def _onboarding_status(request: Request) -> dict[str, Any]:
    """First-run checklist for the dashboard banner. Each step is cheap to
    compute (binary detect, registry list, one COUNT, file existence)."""
    cfg = request.app.state.cfg
    db = request.app.state.db
    binary_ok = bool(detect_binary(cfg.llama_server_binary))
    has_models = bool(request.app.state.registry.list())
    row = db.query_one(
        "SELECT COUNT(*) AS c FROM origins WHERE name != 'bootstrap'")
    has_origin = bool(row and row["c"] > 0)
    autostart = _autostart_installed()

    steps = [
        {"key": "binary", "label": "Install the llama-server binary",
         "done": binary_ok, "href": "/ui/setup",
         "hint": "Open Setup to install a build for your GPU."},
        {"key": "model", "label": "Download a model",
         "done": has_models, "href": "/ui/models",
         "hint": "Pull a GGUF from Hugging Face."},
        {"key": "origin", "label": "Create an API key (origin)",
         "done": has_origin, "href": "/ui/origins",
         "hint": "Replace the bootstrap key with a real origin."},
    ]
    # Autostart is optional/nudge-only — include it only when we can tell,
    # and never let it alone keep the banner open.
    if autostart is not None:
        steps.append({
            "key": "autostart", "label": "Run at startup (+ tray icon)",
            "done": bool(autostart), "href": "/ui/setup",
            "hint": "Run `llamanager autostart --mode tray+service`.",
            "optional": True})

    required_done = all(s["done"] for s in steps if not s.get("optional"))
    return {"steps": steps, "show": not required_done,
            "done_count": sum(1 for s in steps if s["done"]),
            "total": len(steps)}


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    cfg = request.app.state.cfg
    db = request.app.state.db
    # JOIN origins so the Recent table can show the readable origin
    # name instead of a bare numeric id. LEFT JOIN keeps requests
    # whose origin was deleted (origin_id no longer resolvable).
    recent = db.query(
        "SELECT r.id, r.origin_id, o.name AS origin_name, r.model,"
        " r.status, r.enqueued_at, r.started_at, r.finished_at,"
        " r.prompt_tokens, r.completion_tokens"
        " FROM requests r LEFT JOIN origins o ON r.origin_id = o.id"
        " ORDER BY r.enqueued_at DESC LIMIT 10"
    )
    # Build the base URL for cheat sheet examples
    host = request.headers.get("host", f"{cfg.bind}:{cfg.port}")
    base_url = f"http://{host}"
    sysinfo = _get_system_info(cfg.models_dir)
    image_runner = getattr(request.app.state, "image_runner", None)
    image_status = image_runner.status() if image_runner else None
    return templates.TemplateResponse(request, "dashboard.html", _ctx(
        request,
        status=sm.status(),
        queue=qm.snapshot(),
        recent=[dict(r) for r in recent],
        base_url=base_url,
        sysinfo=sysinfo,
        image_status=image_status,
        onboarding=_onboarding_status(request),
    ))


@router.get("/_partials/dashboard", response_class=HTMLResponse)
async def dashboard_partial(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    cfg = request.app.state.cfg
    db = request.app.state.db
    # JOIN origins so the polled partial keeps showing the readable origin
    # name. Without this the 3s refresh would fall back to the numeric id
    # (the full-page render JOINs but this one used to not).
    recent = db.query(
        "SELECT r.id, r.origin_id, o.name AS origin_name, r.model,"
        " r.status, r.enqueued_at, r.started_at, r.finished_at,"
        " r.prompt_tokens, r.completion_tokens"
        " FROM requests r LEFT JOIN origins o ON r.origin_id = o.id"
        " ORDER BY r.enqueued_at DESC LIMIT 10"
    )
    sysinfo = _get_system_info(cfg.models_dir)
    image_runner = getattr(request.app.state, "image_runner", None)
    image_status = image_runner.status() if image_runner else None
    return templates.TemplateResponse(request, "_dashboard_partial.html", _ctx(
        request,
        status=sm.status(),
        queue=qm.snapshot(),
        recent=[dict(r) for r in recent],
        sysinfo=sysinfo,
        image_status=image_status,
    ))


# ---------- queue ----------

VALID_QUEUE_STATUSES = (
    "queued", "swapping_model", "running", "done", "failed", "cancelled",
)


def _queue_ctx(request: Request) -> dict[str, Any]:
    """Shared context for the Queue page + its polled partial.

    Layers the live snapshot (in-flight + pending from QueueManager)
    with a DB-backed history view that supports filtering by model,
    origin name, and status. Filters arrive as query params so the
    polled partial sees them via ``hx-include``.
    """
    qm: QueueManager = request.app.state.queue
    db = request.app.state.db
    qp = request.query_params

    f_model = (qp.get("f_model") or "").strip()
    f_origin = (qp.get("f_origin") or "").strip()
    f_status = (qp.get("f_status") or "").strip()
    try:
        f_limit = max(10, min(500, int(qp.get("f_limit") or 50)))
    except ValueError:
        f_limit = 50

    where: list[str] = []
    args: list[Any] = []
    if f_model:
        where.append("r.model LIKE ?")
        args.append(f"%{f_model}%")
    if f_origin:
        where.append("o.name LIKE ?")
        args.append(f"%{f_origin}%")
    if f_status and f_status in VALID_QUEUE_STATUSES:
        where.append("r.status = ?")
        args.append(f_status)
    sql = (
        "SELECT r.id, r.origin_id, o.name AS origin_name, r.model,"
        " r.status, r.enqueued_at, r.started_at, r.finished_at,"
        " r.prompt_tokens, r.completion_tokens, r.error"
        " FROM requests r LEFT JOIN origins o ON r.origin_id = o.id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.enqueued_at DESC LIMIT ?"
    args.append(f_limit)
    history = [dict(row) for row in db.query(sql, tuple(args))]

    # Distinct values for the filter dropdowns. Cheap LIMIT 50 covers
    # the realistic operator scenarios (a few origins, a few models).
    origins = [dict(row) for row in db.query(
        "SELECT id, name FROM origins ORDER BY name LIMIT 100"
    )]
    models_in_history = sorted({
        h["model"] for h in history if h.get("model")
    })

    return _ctx(
        request,
        queue=qm.snapshot(),
        history=history,
        f_model=f_model,
        f_origin=f_origin,
        f_status=f_status,
        f_limit=f_limit,
        origins=origins,
        valid_statuses=VALID_QUEUE_STATUSES,
        history_models=models_in_history,
    )


@router.get("/queue", response_class=HTMLResponse)
async def queue_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "queue.html",
                                      _queue_ctx(request))


@router.get("/_partials/queue", response_class=HTMLResponse)
async def queue_partial(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "_queue_partial.html",
                                      _queue_ctx(request))


@router.post("/queue/{request_id}/cancel", response_class=HTMLResponse)
async def queue_cancel_ui(request: Request, request_id: str,
                          _: None = Depends(require_csrf)) -> HTMLResponse:
    qm: QueueManager = request.app.state.queue
    qm.cancel(request_id)
    return templates.TemplateResponse(request, "_queue_partial.html",
                                      _queue_ctx(request))


@router.get("/requests/{request_id}", response_class=HTMLResponse)
async def request_detail(request: Request, request_id: str,
                         _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Detail fragment for one request — the prompt sent and the response
    generated, plus timing/token stats. Loaded into the shared modal when
    an operator clicks a row on the dashboard or queue.

    For a request that's still queued or running we read the live in-memory
    buffer (prompt + partial response) off the QueueManager so the operator
    can watch it stream; the fragment then polls itself until the request
    reaches a terminal state, at which point we serve the persisted DB row.
    """
    qm: QueueManager = request.app.state.queue
    db = request.app.state.db
    retain = getattr(request.app.state.cfg, "conversation_retention_days", 0) > 0

    live = qm.get(request_id)
    if live is not None and live.status in ("queued", "swapping_model", "running"):
        partial = "".join(live.response_parts)
        req = {
            "id": live.request_id,
            "origin_id": live.origin.id,
            "origin_name": live.origin.name,
            "model": live.model_required,
            "status": live.status,
            "enqueued_at": live.enqueued_at,
            "started_at": live.started_at,
            "finished_at": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": live.error,
            "prompt_text": live.prompt_text if retain else None,
            "response_text": (partial if retain else None) or None,
        }
        return templates.TemplateResponse(
            request, "_request_detail.html",
            # Only self-refresh when there's live content to stream.
            _ctx(request, req=req, active=retain),
        )

    row = db.query_one(
        "SELECT r.id, r.origin_id, o.name AS origin_name, r.model, r.priority,"
        " r.status, r.enqueued_at, r.started_at, r.finished_at,"
        " r.prompt_tokens, r.completion_tokens, r.error,"
        " r.prompt_text, r.response_text"
        " FROM requests r LEFT JOIN origins o ON r.origin_id = o.id"
        " WHERE r.id = ?",
        (request_id,),
    )
    return templates.TemplateResponse(
        request, "_request_detail.html",
        _ctx(request, req=dict(row) if row else None, active=False),
    )


# ---------- models ----------

_CTX_MAX_CACHE: dict[tuple[str, float], int] = {}


def _ctx_max_for(path: Path, engine: str) -> int:
    """Best-effort trained context length for a model. Cached by (path,
    mtime). Returns a generous fallback when unknown."""
    if engine != "llama" or not path.is_file():
        return 131072
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return 131072
    cached = _CTX_MAX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from .gguf_meta import read_gguf_meta
        meta = read_gguf_meta(path)
        value = int(meta.context_length) if meta.context_length else 131072
    except Exception:
        value = 131072
    _CTX_MAX_CACHE[key] = value
    return value


# A split/sharded GGUF is named ``<name>-00001-of-00005.gguf``. llama.cpp
# loads the whole set from the first shard, so only that one is a launch
# target; the rest are pieces of the same model.
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _is_mmproj(model_id: str) -> bool:
    """True for vision-projector files. These aren't launchable models —
    they're attachments referenced by a profile's ``mmproj`` field — so the
    models list shouldn't show them as their own cards."""
    return "mmproj" in model_id.rsplit("/", 1)[-1].lower()


def _shard_index(model_id: str) -> int | None:
    """1-based index of a split-GGUF shard, or None if not sharded."""
    m = _SHARD_RE.search(model_id)
    return int(m.group(1)) if m else None


def _model_dir(model_id: str) -> str:
    """Parent directory (repo) of a model id, or '' for a root-level file."""
    return model_id.rsplit("/", 1)[0] if "/" in model_id else ""


def _decode_rates_by_model(db) -> dict[str, float]:
    """Estimate decode throughput (tokens/sec) per model from recent finished
    requests, for advising a reasoning budget. A thinking-token budget is
    spent at decode speed, so we approximate that with a high percentile of
    completion_tokens/duration over generation-substantial requests — the
    fastest samples are the most generation-dominant (least prefill overhead),
    so they best reflect pure decode. Best-effort; returns {} on any trouble.
    """
    try:
        rows = db.query(
            "SELECT model, completion_tokens AS ct, started_at AS s, "
            "finished_at AS f FROM requests WHERE status='done' "
            "AND completion_tokens IS NOT NULL AND completion_tokens >= 80 "
            "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "ORDER BY enqueued_at DESC LIMIT 1000"
        )
    except Exception as e:  # noqa: BLE001 — advice is best-effort
        log.debug("decode-rate query failed: %s", e)
        return {}
    buckets: dict[str, list[float]] = {}
    for r in rows:
        dur = (r["f"] or 0) - (r["s"] or 0)
        ct = r["ct"] or 0
        if dur <= 0 or ct <= 0 or not r["model"]:
            continue
        buckets.setdefault(r["model"], []).append(ct / dur)
    out: dict[str, float] = {}
    for mid, rates in buckets.items():
        rates.sort()
        out[mid] = rates[min(int(len(rates) * 0.9), len(rates) - 1)]
    return out


def _recommended_reasoning_budget(tok_s: float | None) -> int | None:
    """A reasoning budget that keeps thinking to ~20s of wall-clock at the
    model's measured decode rate — a sane interactive default. Rounded to 250.
    """
    if not tok_s or tok_s <= 0:
        return None
    return max(500, int(round((tok_s * 20) / 250.0) * 250))


def _models_ctx(request: Request) -> dict:
    import sys as _sys
    import psutil as _psutil
    reg: Registry = request.app.state.registry
    cfg = request.app.state.cfg
    if _sys.platform == "darwin":
        open_label = "Open in Finder"
    elif _sys.platform == "win32":
        open_label = "Open in Explorer"
    else:
        open_label = "Open folder"

    sysinfo = _get_system_info(cfg.models_dir)
    ram_total_gb = float(sysinfo.get("ram_total_gb") or 0) or round(
        _psutil.virtual_memory().total / (1024**3), 1
    )
    vram_total_gb = sysinfo.get("gpu_vram_total_gb")
    # On Apple Silicon (unified memory) and CPU-only systems we have no
    # distinct VRAM total; fall back to system RAM as the cap for the
    # vram_limit slider so the control is still useful.
    if not vram_total_gb:
        vram_total_gb = ram_total_gb

    # Profiles live nested under their parent model in cfg.models. We hand
    # the template a flattened per-model list of dicts so Jinja can render
    # without poking dataclasses.
    text_models: list[dict] = []
    image_models: list[dict] = []
    audio_models: list[dict] = []

    # Measured decode throughput per model, for the reasoning-budget advice.
    decode_rates = _decode_rates_by_model(request.app.state.db)

    entries = reg.list()
    # Pre-pass: which repo dirs ship a vision projector (so the model in that
    # repo can show a "vision" hint), and the combined size of each split-GGUF
    # series (so the first shard's card shows the whole model's size, not just
    # part 1). Projector + non-first-shard files are then skipped as cards.
    mmproj_dirs: set[str] = set()
    shard_series_bytes: dict[str, int] = {}
    for entry in entries:
        if _is_mmproj(entry.model_id):
            mmproj_dirs.add(_model_dir(entry.model_id))
        if _shard_index(entry.model_id) is not None:
            key = _SHARD_RE.sub("", entry.model_id)
            shard_series_bytes[key] = shard_series_bytes.get(key, 0) + entry.size

    for entry in entries:
        mid = entry.model_id
        # Vision projectors are attachments, not models — never list them.
        if _is_mmproj(mid):
            continue
        # For a split GGUF, only the first shard represents the model.
        part = _shard_index(mid)
        if part is not None and part > 1:
            continue
        d = entry.to_dict()
        # Clean display name: drop the "-00001-of-00003" suffix so a folded
        # split model reads as one model (model_id keeps the real first-shard
        # path, which is what llama.cpp loads from).
        base = mid.rsplit("/", 1)[-1]
        d["display_name"] = (_SHARD_RE.sub("", base) + ".gguf") if part else base
        if part == 1:
            # Show the whole split model's size, not just shard 1.
            d["size"] = shard_series_bytes.get(_SHARD_RE.sub("", mid), entry.size)
        engine = detect_engine_for_id(entry.model_id, cfg.models_dir)
        d["engine"] = engine
        d["engine_family"] = ENGINE_FAMILY.get(engine, "text")
        d["ctx_max"] = _ctx_max_for(entry.path, engine)
        # Capacity advice for the profile editor's ctx control: per-model KV
        # rate (GB per 1k tokens), weight footprint, and the largest ctx that
        # still fits VRAM. The slider reads these to show a live estimate and
        # flag a context that would spill to RAM. llama text models only —
        # KV sizing doesn't apply to mlx/diffusion.
        d["kv_per_1k_gb"] = None
        d["safe_ctx"] = None
        d["weights_gb"] = None
        if engine == "llama" and d["engine_family"] != "image":
            try:
                from . import mem_guard
                from .gguf_meta import read_gguf_meta
                _meta = read_gguf_meta(entry.path) if entry.path.is_file() else None
                if _meta is not None:
                    # Per-1k rate must be the ctx-proportional (attention-only)
                    # term — the JS extrapolates it linearly, so folding in the
                    # constant SSM state of hybrid models would over-count it at
                    # high ctx. The constant is added to the fixed base below.
                    kv1k = mem_guard.kv_cache_gb(entry.path, 1024, meta=_meta,
                                                 include_recurrent=False)
                    d["kv_per_1k_gb"] = round(kv1k, 4) if kv1k else None
                    weights = mem_guard.weights_gb(entry.path, meta=_meta)
                    # Constant SSM recurrent state (0 for non-hybrid models):
                    # full footprint at any ctx minus the attention-only part.
                    full1k = mem_guard.kv_cache_gb(entry.path, 1024, meta=_meta)
                    ssm_const = (full1k - kv1k) if (full1k and kv1k) else 0.0
                    d["weights_gb"] = round(weights + max(0.0, ssm_const), 1)
                    _vram = getattr(cfg, "vram_total_gb", None)
                    if _vram:
                        # Conservative ceiling for the slider hint: f16 KV and
                        # the engine's auto slot count (the live JS estimate
                        # refines this per-profile as the operator edits).
                        d["safe_ctx"] = mem_guard.safe_max_ctx(
                            entry.path, _vram, meta=_meta,
                            n_parallel=mem_guard.DEFAULT_PARALLEL_SLOTS)
            except Exception as e:  # noqa: BLE001 — advice is best-effort
                log.debug("ctx advice sizing failed for %s: %s", mid, e)
        # Flag models whose repo ships a projector so the card can hint that
        # it's vision-capable (the projector is wired up via a profile).
        d["has_mmproj"] = bool(_model_dir(mid)) and _model_dir(mid) in mmproj_dirs
        # Reasoning-budget advice: measured decode rate + a suggested budget
        # (~20s of thinking at that rate). llama text models only.
        rate = decode_rates.get(mid)
        if engine == "llama" and d["engine_family"] != "image":
            d["decode_tok_s"] = round(rate, 1) if rate else None
            d["rec_reasoning_budget"] = _recommended_reasoning_budget(rate)
        else:
            d["decode_tok_s"] = None
            d["rec_reasoning_budget"] = None
        m = cfg.get_model(entry.model_id)
        prof_entries: list[dict] = []
        default_profile = ""
        if m:
            default_profile = m.default_profile
            for p in m.profiles.values():
                prof_entries.append({
                    "name": p.name,
                    "mmproj": p.mmproj,
                    "ctx_size": p.ctx_size,
                    "vram_limit_gb": p.vram_limit_gb,
                    "ram_spill_policy": p.ram_spill_policy or "default",
                    "ram_spill_limit_gb": p.ram_spill_limit_gb,
                    "kv_cache_type": getattr(p, "kv_cache_type", "") or "",
                    "flash_attn": getattr(p, "flash_attn", "") or "",
                    "thinking": getattr(p, "thinking", "") or "",
                    "reasoning_budget": getattr(p, "reasoning_budget", None),
                    "parallel": getattr(p, "parallel", None),
                    "mtp": getattr(p, "mtp", False),
                    "mtp_n_max": getattr(p, "mtp_n_max", None),
                    "args": p.args,
                    "args_json": json.dumps(p.args, indent=2) if p.args else "{}",
                })
            prof_entries.sort(key=lambda e: e["name"])
        d["profiles"] = prof_entries
        d["default_profile"] = default_profile
        if d["engine_family"] == "image":
            image_models.append(d)
        elif d["engine_family"] == "audio":
            # ASR models have their own page (/ui/asr-models) — keep them out
            # of the LLM model list, the same way diffusion models are.
            audio_models.append(d)
        else:
            text_models.append(d)

    # Combined list kept for any callers that still expect ``models``.
    all_models = text_models + image_models

    # Detect mmproj-style GGUFs on disk so the profile editor can offer a
    # pickable dropdown instead of a free-text path. Same heuristic the
    # registry uses internally when auto-seeding profiles (filename contains
    # "mmproj").
    mmproj_options: list[str] = []
    mmproj_sizes: dict[str, float] = {}   # rel path → GB, for the ctx estimate
    if cfg.models_dir.exists():
        for p in sorted(cfg.models_dir.rglob("*.gguf")):
            if not p.is_file():
                continue
            if "mmproj" not in p.name.lower():
                continue
            rel = p.relative_to(cfg.models_dir).as_posix()
            mmproj_options.append(rel)
            try:
                mmproj_sizes[rel] = round(p.stat().st_size / (1024 ** 3), 2)
            except OSError:
                pass

    # Per-model split so each editor's dropdown lists projectors in the
    # model's own folder first (the overwhelmingly common case) and the rest
    # under "other folders". Only meaningful for llama text models.
    import posixpath as _pp
    for d in all_models:
        if d.get("engine") != "llama":
            continue
        mfolder = _pp.dirname(d.get("model_id") or "")
        d["mmproj_same"] = [o for o in mmproj_options if _pp.dirname(o) == mfolder]
        d["mmproj_other"] = [o for o in mmproj_options if _pp.dirname(o) != mfolder]
        d["mmproj_sizes"] = mmproj_sizes

    return _ctx(
        request,
        models=all_models,
        text_models=text_models,
        image_models=image_models,
        downloads=reg.list_downloads(),
        current_model=request.app.state.sm.runtime.current_model,
        current_profile=request.app.state.sm.runtime.current_profile,
        models_dir=str(cfg.models_dir),
        open_label=open_label,
        default_args=cfg.default_args,
        default_model=cfg.default_model,
        ram_spill_policies=VALID_RAM_SPILL_POLICIES,
        ram_total_gb=round(ram_total_gb, 1),
        vram_total_gb=round(float(vram_total_gb), 1),
        # Actual GPU VRAM (None on CPU-only / not-yet-detected), distinct from
        # vram_total_gb above which falls back to RAM for the slider max. The
        # profile editor's ctx advice compares KV against real GPU memory.
        gpu_vram_gb=getattr(cfg, "vram_total_gb", None),
        mmproj_options=mmproj_options,
    )


@router.get("/models", response_class=HTMLResponse)
async def models_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "models.html", _models_ctx(request))


@router.get("/models/_list", response_class=HTMLResponse)
async def models_list_partial(request: Request,
                              _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_models_list_partial.html", _models_ctx(request),
    )


@router.get("/models/_downloads_strip", response_class=HTMLResponse)
async def models_downloads_strip(request: Request,
                                 _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Tiny partial that only contains the active-downloads table.

    Polled every 2 s while a download is running. Splitting this out
    from ``_models_list_partial`` is what lets open profile editors
    survive the poll — the model cards (and their inline forms) live
    in a sibling div that this endpoint never touches."""
    ctx = _models_ctx(request)
    resp = templates.TemplateResponse(
        request, "_models_downloads_strip.html", ctx,
    )
    # The strip only polls while a download is active. When a poll finds
    # nothing active, a download just finished — tell the page (once) to
    # refresh the model list so the new model appears in its card group.
    active = [d for d in ctx["downloads"]
              if d.get("status") in ("running", "pending")]
    if not active:
        resp.headers["HX-Trigger"] = "lmDownloadsIdle"
    return resp


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
    return _models_redirect(request)


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
    return _models_redirect(request)


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
    return _models_redirect(request)


@router.post("/models/pull", response_class=HTMLResponse)
async def models_pull_ui(request: Request, source: str = Form(...),
                         files: str = Form(""),
                         _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    src = source.strip()
    file_list = [f.strip() for f in files.split(",") if f.strip()] or None
    # Seed bytes_total up-front for HF pulls so the progress bar shows a
    # real percentage instead of only bytes-so-far. Best-effort: a failed
    # estimate (no token, network blip) just leaves the total at 0 and the
    # strip falls back to showing bytes downloaded.
    bytes_total = 0
    if src.startswith("hf://") or src.startswith("hf:"):
        repo = src.removeprefix("hf://").removeprefix("hf:")
        try:
            bytes_total = await reg.estimate_repo_size(repo, files=file_list)
        except Exception:
            bytes_total = 0
    try:
        reg.start_pull(source=src, files=file_list, bytes_total=bytes_total)
    except Exception as e:
        return _error_html(f"pull failed: {e}", status_code=400)
    # 303 back to the models page (standard post-redirect-get). The download
    # progress strip is rendered there, so the pull is visible immediately.
    # We deliberately do NOT render the full page here with an HX-Trigger
    # toast: returning a whole document to a boosted POST makes htmx swap the
    # <body> element and crash mid-swap ("document.body is null") on this page,
    # because its content wires up `from:body` triggers during the swap.
    return _models_redirect(request)


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
    # Use a function replacement: a plain-string replacement makes re treat
    # backslash sequences specially (\U, \1, …), which mangles Windows paths
    # like C:\Users\… into invalid TOML and corrupts the config. A function
    # returns the text verbatim.
    text, n = _re.subn(r'^models_dir\s*=\s*.*$', lambda _m: new_line, text,
                       flags=_re.MULTILINE)
    if n == 0:
        server_match = _re.search(r'^\[server\]', text, flags=_re.MULTILINE)
        if server_match:
            text = text[:server_match.end()] + f"\n{new_line}" + text[server_match.end():]
        else:
            text = text.rstrip("\n") + f"\n\n[server]\n{new_line}\n"
    config_path.write_text(text, encoding="utf-8")
    return _models_redirect(request)


@router.post("/downloads/{download_id}/cancel", response_class=HTMLResponse)
async def download_cancel_ui(request: Request, download_id: str,
                              _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    reg.cancel_pull(download_id)
    return _models_redirect(request)


@router.post("/downloads/{download_id}/delete", response_class=HTMLResponse)
async def download_delete_ui(request: Request, download_id: str,
                              _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    reg.delete_download(download_id)
    return _models_redirect(request)


def _open_path(path: str) -> None:
    """Reveal ``path`` in the OS file manager (selecting the file when possible).

    On Linux we deliberately avoid bare ``xdg-open`` as the primary strategy:
    it chooses a handler from desktop-environment env vars, and when the server
    process doesn't inherit those (launched from a stripped shell, a launcher,
    or a service) it falls back to opening the path in the default *web
    browser* — which renders the directory as a broken ``file://`` page. We talk
    to the running file manager directly over the session bus instead, and only
    fall back to ``xdg-open`` as a last resort.
    """
    import subprocess, sys as _sys
    if _sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
        return
    if _sys.platform == "win32":
        # explorer selects a file; for a directory it opens it directly.
        if Path(path).is_dir():
            subprocess.Popen(["explorer", path])
        else:
            subprocess.Popen(["explorer", "/select,", path])
        return

    p = Path(path)
    is_dir = p.is_dir()
    # For a directory, open it so its contents show; for a file, open the
    # containing directory and highlight the file.
    if is_dir:
        method, uri, target_dir = "ShowFolders", p.as_uri(), str(p)
    else:
        method, uri, target_dir = "ShowItems", p.as_uri(), str(p.parent)
    # 1) freedesktop FileManager1 — talks to the running file manager directly,
    #    no XDG desktop detection (which can otherwise leak to a web browser).
    try:
        r = subprocess.run(
            ["gdbus", "call", "--session",
             "--dest", "org.freedesktop.FileManager1",
             "--object-path", "/org/freedesktop/FileManager1",
             "--method", f"org.freedesktop.FileManager1.{method}",
             f"['{uri}']", ""],
            timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        pass
    # 2) gio open — uses the configured handler for inode/directory (file mgr).
    try:
        r = subprocess.run(["gio", "open", target_dir], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        pass
    # 3) Last resort: xdg-open the directory.
    try:
        subprocess.Popen(["xdg-open", target_dir])
    except (FileNotFoundError, OSError):
        pass


@router.post("/models/open-dir", response_class=HTMLResponse)
async def models_open_dir(request: Request,
                          _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    _open_path(str(cfg.models_dir))
    return _models_redirect(request)


@router.post("/models/locate", response_class=HTMLResponse)
async def models_locate(request: Request,
                        model_id: str = Form(...),
                        _: None = Depends(require_csrf)) -> Response:
    reg: Registry = request.app.state.registry
    m = reg.get(model_id)
    if m:
        _open_path(str(m.path))
    return _models_redirect(request)


# ---------- local filesystem browser (for path inputs) ----------

def _fs_default_start(kind: str) -> Path:
    """Where to start browsing if the caller didn't specify a path."""
    return Path.home()


def _fs_shortcuts() -> list[dict[str, str]]:
    """Quick-jump locations shown above the directory listing. Lets users
    cross drive/root boundaries that ``..`` cannot reach (Windows drive
    letters, mounted volumes on macOS/Linux). Only entries that exist on
    disk are returned."""
    import sys as _sys
    out: list[dict[str, str]] = []
    home = Path.home()
    if home.exists():
        out.append({"label": "Home", "path": str(home)})

    if _sys.platform == "win32":
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            try:
                if drive.exists():
                    out.append({"label": f"{letter}:\\", "path": str(drive)})
            except OSError:
                continue
        return out

    out.append({"label": "/", "path": "/"})
    # Surface mounted volumes (one level under common mount roots).
    mount_roots = ["/Volumes"] if _sys.platform == "darwin" else ["/mnt", "/media"]
    for root in mount_roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        try:
            for entry in sorted(root_path.iterdir(), key=lambda p: p.name.lower()):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    out.append({"label": entry.name, "path": str(entry)})
        except (PermissionError, OSError):
            continue
    return out


def _fs_safe_listdir(path: Path) -> tuple[list[Path], list[Path], str | None]:
    """Return (dirs, files, error_message). Hidden entries are skipped.
    Errors (permission, missing) are returned, not raised."""
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        return [], [], str(exc)
    dirs, files = [], []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append(entry)
            elif entry.is_file():
                files.append(entry)
        except OSError:
            # broken symlink, race — skip silently
            continue
    return dirs, files, None


@router.get("/fs/browse", response_class=HTMLResponse)
async def fs_browse(request: Request,
                    path: str = "",
                    kind: str = "file",
                    ext: str = "",
                    target: str = "",
                    _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Render an inline directory browser for one of the path inputs.

    Query params:
      path:   directory to list (defaults to $HOME)
      kind:   "file" or "dir" — controls which selection action is shown
      ext:    optional extension filter for files (e.g. ".gguf"); empty = all
      target: id of the form input whose value should be set on selection
    """
    if kind not in ("file", "dir"):
        kind = "file"
    raw = (path or "").strip()
    current = Path(raw).expanduser() if raw else _fs_default_start(kind)
    try:
        current = current.resolve()
    except OSError:
        current = _fs_default_start(kind)

    if not current.is_dir():
        # Fall back to the parent if a file path was passed in.
        current = current.parent if current.parent.exists() else _fs_default_start(kind)

    dirs, files, err = _fs_safe_listdir(current)
    ext_norm = ext.strip().lower()
    if ext_norm and not ext_norm.startswith("."):
        ext_norm = "." + ext_norm
    if ext_norm:
        files = [f for f in files if f.suffix.lower() == ext_norm]

    parent = current.parent if current.parent != current else None
    return templates.TemplateResponse(
        request, "_fs_browser.html",
        _ctx(
            request,
            fs_current=str(current),
            fs_parent=str(parent) if parent else "",
            fs_dirs=[{"name": d.name, "path": str(d)} for d in dirs],
            fs_files=[{"name": f.name, "path": str(f)} for f in files],
            fs_error=err,
            fs_kind=kind,
            fs_ext=ext_norm,
            fs_target=target,
            fs_shortcuts=_fs_shortcuts(),
        ),
    )


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


_HF_VALID_SORTS = {"downloads", "likes", "trending", "lastModified", "createdAt"}
_HF_VALID_LIBRARIES = {"gguf", "mlx"}


@router.get("/models/search", response_class=HTMLResponse)
async def models_search(request: Request, q: str = "",
                        sort: str = "downloads",
                        library: str = "gguf",
                        author: str = "",
                        _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Search Hugging Face for inference-ready model repos.

    Filters (all optional, all surface in the UI):

    * ``q``       — free-text search across repo id and tags.
    * ``sort``    — ``downloads`` (default), ``likes``, ``trending``,
                    ``lastModified``, ``createdAt``.
    * ``library`` — ``gguf`` (llama.cpp, default) or ``mlx`` (Apple silicon).
                    The HF ``filter`` tag is set accordingly so only repos
                    that ship that format are returned.
    * ``author``  — restrict to a single org / user. Accepts the bare
                    handle (``mlx-community``) or a ``user/repo`` prefix
                    (we strip after the slash).
    """
    import httpx as _httpx

    if sort not in _HF_VALID_SORTS:
        sort = "downloads"
    if library not in _HF_VALID_LIBRARIES:
        library = "gguf"
    author_clean = (author.split("/", 1)[0]).strip()

    async def _do_search() -> list[dict]:
        params: dict[str, str] = {
            "filter": library,
            "sort": sort,
            "direction": "-1",
            "limit": "20",
        }
        if q.strip():
            params["search"] = q.strip()
        if author_clean:
            params["author"] = author_clean
        async with _httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://huggingface.co/api/models", params=params,
            )
            r.raise_for_status()
            return r.json()

    try:
        raw = await _do_search()
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
            _ctx(request, search_results=[], search_error=str(e),
                 search_query=q, search_library=library),
        )

    return templates.TemplateResponse(
        request, "_model_search_results.html",
        _ctx(request, search_results=results, search_error=None,
             search_query=q, search_library=library),
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

def _all_model_ids(request: Request) -> list[str]:
    """Installed model ids, for the origin allow-list picker."""
    reg: Registry = request.app.state.registry
    return sorted(m.model_id for m in reg.list())


def _parse_allowed_models(allow_all: bool, allowed_models: list[str]) -> list[str]:
    """Turn the create/edit form's picker into an allow-list.

    ``allow_all`` (or an empty selection) means every model (``["*"]``);
    otherwise only the explicitly selected ids are permitted.
    """
    if allow_all:
        return ["*"]
    picked = [a.strip() for a in allowed_models if a.strip()]
    return picked or ["*"]


@router.get("/origins", response_class=HTMLResponse)
async def origins_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    return templates.TemplateResponse(request, "origins.html", _ctx(
        request,
        origins=[o.to_public() for o in am.list_origins()],
        all_model_ids=_all_model_ids(request),
        new_key=None,
    ))


@router.post("/origins/create", response_class=HTMLResponse)
async def origins_create_ui(request: Request, name: str = Form(...),
                            priority: int = Form(50),
                            allow_all: bool = Form(False),
                            allowed_models: list[str] = Form([]),
                            is_admin: bool = Form(False),
                            _: None = Depends(require_csrf)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    if am.get_origin_by_name(name):
        # Render a normal page with an inline notice; no f-string HTML.
        return templates.TemplateResponse(request, "origins.html", _ctx(
            request,
            origins=[o.to_public() for o in am.list_origins()],
            all_model_ids=_all_model_ids(request),
            new_key=None,
            error=f"origin '{name}' already exists",
        ), status_code=409)
    al = _parse_allowed_models(allow_all, allowed_models)
    origin, key = am.create_origin(name=name, priority=priority,
                                   allowed_models=al, is_admin=is_admin)
    return templates.TemplateResponse(request, "origins.html", _ctx(
        request,
        origins=[o.to_public() for o in am.list_origins()],
        all_model_ids=_all_model_ids(request),
        new_key={"name": origin.name, "key": key},
    ))


@router.post("/origins/{origin_id}/allowed-models", response_class=HTMLResponse)
async def origins_allowed_models_ui(request: Request, origin_id: int,
                                    allow_all: bool = Form(False),
                                    allowed_models: list[str] = Form([]),
                                    _: None = Depends(require_csrf)) -> Response:
    """Edit an existing origin's allow-list (the only field create couldn't
    change afterwards). Mirrors the create-form picker."""
    am: AuthManager = request.app.state.auth
    am.update_origin(origin_id,
                     allowed_models=_parse_allowed_models(allow_all, allowed_models))
    return RedirectResponse("/ui/origins", status_code=303)


@router.post("/origins/{origin_id}/priority", response_class=HTMLResponse)
async def origins_priority_ui(request: Request, origin_id: int,
                              priority: int = Form(...),
                              _: None = Depends(require_csrf)) -> HTMLResponse:
    am: AuthManager = request.app.state.auth
    am.update_origin(origin_id, priority=priority)
    # Integer is safe to interpolate (FastAPI already coerced to int).
    return HTMLResponse(f"<span>priority: {int(priority)}</span>")


@router.post("/origins/{origin_id}/enabled", response_class=HTMLResponse)
async def origins_enabled_ui(request: Request, origin_id: int,
                             enabled: str = Form("off"),
                             _: None = Depends(require_csrf)) -> Response:
    """Enable/disable an origin. A disabled origin authenticates but can't
    submit inference/image requests (rejected with 403). Takes effect on the
    next request (the auth cache is cleared)."""
    am: AuthManager = request.app.state.auth
    am.set_enabled(origin_id, enabled == "on")
    return RedirectResponse("/ui/origins", status_code=303)


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
        all_model_ids=_all_model_ids(request),
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


from . import __version__ as LLAMANAGER_VERSION
GITHUB_REPO = "mounirsetti/llamanager"


def _about_ctx(request: Request, **extra: Any) -> dict:
    import datetime
    cfg = request.app.state.cfg
    from .auto_update import SELF_KEY
    ctx = _ctx(
        request,
        version=LLAMANAGER_VERSION,
        year=datetime.date.today().year,
        update_available=None,
        latest_version=None,
        update_check_error=None,
        update_log=None,
        # Always surface the install mode so the auto-update toggle can be
        # disabled for editable installs even before a "Check for updates".
        install_mode=_detect_install_mode()["mode"],
        # Auto-update-when-idle switch for the daemon's own self-update.
        auto_update_self=bool((cfg.auto_update_engines or {}).get(SELF_KEY)),
    )
    ctx.update(extra)
    return ctx


@router.get("/about", response_class=HTMLResponse)
async def about_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "about.html", _about_ctx(request))


def _detect_install_mode() -> dict[str, Any]:
    """Figure out how llamanager was installed.

    Reads ``direct_url.json`` (PEP 610) which pip writes alongside every
    package install. The file's ``dir_info.editable: true`` flag is the
    canonical marker for editable installs (``pip install -e .``); when
    it's absent we assume a regular wheel install which can be upgraded
    in place with ``pip install --upgrade git+https://github.com/<repo>``.

    Returns ``{mode, location, source}``:
      - ``mode`` is ``"editable"`` | ``"pypi"`` | ``"unknown"``.
      - ``location`` is the local path the package resolves to (the
        checkout dir for editable, the site-packages dir otherwise).
      - ``source`` is the direct_url ``url`` if present (useful for
        editable installs to point the operator at the right checkout).
    """
    import importlib.metadata as _im
    import json as _json
    try:
        dist = _im.distribution("llamanager")
    except _im.PackageNotFoundError:
        return {"mode": "unknown", "location": "", "source": ""}

    # ``locate_file`` resolves the canonical path of an installed file.
    # We use it to get the package location without dragging in
    # pip-internal APIs.
    try:
        location = str(dist.locate_file("llamanager")).rsplit("/llamanager", 1)[0]
    except Exception:
        location = ""

    raw = None
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None

    mode = "pypi"
    source = ""
    if raw:
        try:
            info = _json.loads(raw)
        except _json.JSONDecodeError:
            info = {}
        if info.get("dir_info", {}).get("editable"):
            mode = "editable"
        source = info.get("url", "") or ""
    return {"mode": mode, "location": location, "source": source}


def _check_latest_release() -> dict[str, Any]:
    """Resolve the newest tag on the GitHub repo, releases first then tags.

    Pure-stdlib (urllib) so this function works from the CLI/admin
    paths that don't carry httpx. Returns ``{latest, current, is_newer,
    install_mode}`` or raises if neither releases nor tags are
    available. ``install_mode`` is carried in the result so the UI /
    CLI can render the right call-to-action: PyPI installs get an
    Update button; editable installs get an instructions block.
    """
    import urllib.request as _ur
    import urllib.error as _ue
    import json as _json
    latest: str | None = None
    for api_path in ("releases/latest", "tags"):
        req = _ur.Request(
            f"https://api.github.com/repos/{GITHUB_REPO}/{api_path}",
            headers={"User-Agent": "llamanager",
                     "Accept": "application/vnd.github+json"},
        )
        try:
            with _ur.urlopen(req, timeout=10) as r:
                data = _json.loads(r.read())
        except _ue.HTTPError as he:
            if he.code == 404 and api_path == "releases/latest":
                continue
            raise
        if api_path == "releases/latest":
            latest = data.get("tag_name", "").lstrip("v") or None
        else:
            if isinstance(data, list) and data:
                latest = data[0].get("name", "").lstrip("v") or None
        if latest:
            break
    if not latest:
        raise ValueError("no releases or tags found on GitHub")
    install = _detect_install_mode()
    return {
        "latest": latest,
        "current": LLAMANAGER_VERSION,
        "is_newer": _version_newer(latest, LLAMANAGER_VERSION),
        "install_mode": install["mode"],
        "install_location": install["location"],
    }


@router.post("/about/check-update", response_class=HTMLResponse)
async def about_check_update(request: Request,
                             _: None = Depends(require_csrf)) -> HTMLResponse:
    try:
        info = _check_latest_release()
        return templates.TemplateResponse(request, "_update_area.html", _about_ctx(
            request,
            update_available=info["is_newer"],
            latest_version=info["latest"],
            install_mode=info["install_mode"],
            install_location=info["install_location"],
        ))
    except Exception as e:
        return templates.TemplateResponse(request, "_update_area.html", _about_ctx(
            request,
            update_check_error=f"Could not check for updates: {e}",
            install_mode=_detect_install_mode()["mode"],
        ))


def _run_self_update(project_dir: Path | None = None) -> dict[str, Any]:
    """Upgrade llamanager in-place via ``pip install --upgrade`` from GitHub.

    llamanager isn't published to PyPI — the canonical distribution is
    the GitHub repo (see ``GITHUB_REPO``). Pip installs git URLs
    natively (needs ``git`` on PATH), so the same command works for
    every install pattern:

    - Regular install: pip replaces the snapshot with the latest tag.
    - Editable install: pip replaces the editable link with a regular
      install. The daemon picks up the new code on supervisor restart.
      If the developer wants editable back afterwards, ``pip install -e
      <their checkout>`` restores it. We do not block the update on
      install mode; the operator owns that choice.

    Pins to the latest GitHub release tag when reachable (so the action
    matches the "Update to vX.Y.Z" the UI advertises); falls back to
    the default branch if the tag lookup fails.

    The ``project_dir`` arg is kept for back-compat and ignored; pip
    uses ``sys.executable``'s venv automatically. Returns ``{ok, log,
    mode}``.
    """
    import subprocess as _sp
    import sys as _sys
    log_lines: list[str] = []
    install = _detect_install_mode()
    # Pin to the latest tag when reachable; fall back to the default
    # branch if GitHub is unreachable. Either way the request goes to
    # the canonical source, not PyPI.
    ref = "main"
    try:
        info = _check_latest_release()
        if info.get("latest"):
            ref = f"v{info['latest']}"
    except Exception:
        pass
    url = f"git+https://github.com/{GITHUB_REPO}.git@{ref}"
    try:
        argv = [_sys.executable, "-m", "pip", "install", "--upgrade",
                "--no-input", url]
        log_lines.append("$ " + " ".join(argv))
        result = _sp.run(
            argv, capture_output=True, text=True, timeout=600,
        )
        if result.stdout.strip():
            # Keep the tail — pip's output can be very long with deps.
            tail = result.stdout.strip().splitlines()[-30:]
            log_lines.extend(tail)
        if result.stderr.strip():
            err_lines = result.stderr.strip().splitlines()
            log_lines.extend(l for l in err_lines if "WARNING" not in l)
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install --upgrade {url} failed "
                f"(exit {result.returncode}). Common causes: ``git`` is "
                f"missing on PATH, the venv is read-only, the host has "
                f"no network access to GitHub, or pip itself needs "
                f"upgrading."
            )
        return {"ok": True, "log": "\n".join(log_lines), "mode": install["mode"]}
    except Exception as e:
        log_lines.append(f"\nError: {e}")
        return {"ok": False, "log": "\n".join(log_lines),
                "error": str(e), "mode": install["mode"]}


async def _schedule_self_restart() -> None:
    """SIGTERM ourselves after a brief delay so the HTTP response reaches
    the caller before the daemon dies. Whatever process supervisor we
    run under (systemd / launchd / pm2) restarts us."""
    import os as _os
    import signal as _signal
    async def _delayed():
        await asyncio.sleep(1)
        _os.kill(_os.getpid(), _signal.SIGTERM)
    asyncio.create_task(_delayed())


@router.post("/about/update", response_class=HTMLResponse)
async def about_update(request: Request,
                       _: None = Depends(require_csrf)) -> Response:
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _run_self_update)
    install_mode = res.get("mode", "unknown")
    if not res["ok"]:
        return templates.TemplateResponse(request, "about.html", _about_ctx(
            request,
            update_check_error=res.get("error", "update failed"),
            update_log=res["log"],
            install_mode=install_mode,
        ))
    # Stop llama-server and schedule a self-restart so the new code loads.
    sm: ServerManager = request.app.state.sm
    if sm.is_running:
        await sm.stop()
    await _schedule_self_restart()
    return templates.TemplateResponse(request, "about.html", _about_ctx(
        request,
        update_log=res["log"] + "\n\nUpdate complete. Restarting…",
        update_available=False,
        latest_version="restarting",
    ))


@router.post("/about/auto-update", response_class=HTMLResponse)
async def about_auto_update_toggle(request: Request,
                                   enabled: str = Form(""),
                                   _: None = Depends(require_csrf)) -> Response:
    """Toggle auto-update-when-idle for llamanager itself."""
    from .auto_update import SELF_KEY
    _set_auto_update_flag(request, SELF_KEY, enabled.strip().lower() == "on")
    return RedirectResponse("/ui/about", status_code=303)


# ---------- settings ----------
#
# One page for daemon-level preferences that used to be scattered across
# About (software update), Setup (run-at-startup), and Launch (auto-launch /
# auto-restart). The autostart action shells out to `llamanager autostart`,
# the same per-platform logic the CLI and tray use.

def _gpu_device_ctx(cfg) -> dict:
    """Enumerate selectable GPUs for the text engine + the current pin.

    Best-effort: any failure (no engine installed, enumeration error) yields
    an empty list, and the template degrades to just the stored name. Only
    meaningful for the llama engine — mlx has no --list-devices.
    """
    current = getattr(cfg, "llama_gpu_device", "") or ""
    devices: list[dict] = []
    matched_value = ""           # the option that corresponds to the stored pin
    if (getattr(cfg, "llama_server_engine", "llama") or "llama") == "llama":
        binary = detect_binary(cfg.llama_server_binary)
        if binary:
            try:
                from .gpu_detect import (
                    list_llama_devices, match_device, clean_gpu_name)
                raw = list_llama_devices(binary, _engine_env(binary))
                for d in raw:
                    devices.append({
                        # The clean model name is both shown and stored, so the
                        # pin is backend-agnostic (see gpu_detect.match_device).
                        "value": clean_gpu_name(d.name),
                        "name": clean_gpu_name(d.name),
                        "raw_name": d.name,
                        "backend": d.backend,
                        "index": d.index,
                        "mem_free_mib": d.mem_free_mib,
                        "mem_total_mib": d.mem_total_mib,
                    })
                # Resolve the stored pin to a live device robustly, so the
                # right option shows selected even if its name changed form.
                if current:
                    m = match_device(raw, current)
                    if m is not None:
                        matched_value = clean_gpu_name(m.name)
            except Exception as e:  # noqa: BLE001 — never break settings
                log.debug("gpu enumeration failed: %s", e)
    # Pinned but not matched to any live device → still offer it so the
    # operator sees their selection rather than a silent reset to "all".
    pinned_missing = bool(current) and not matched_value
    return {
        "gpu_devices": devices,
        "gpu_device_current": current,
        "gpu_device_matched": matched_value or current,
        "gpu_pinned_missing": pinned_missing,
    }


def _mem_guard_ctx(cfg) -> dict:
    """Current memory-guardrail settings + a live capacity snapshot for the
    Settings page."""
    from . import mem_guard
    out = {
        "mem_guard_enabled": bool(getattr(cfg, "mem_guard_enabled", True)),
        "mem_clamp_ctx": bool(getattr(cfg, "mem_clamp_ctx", False)),
        "mem_hard_stop_enabled": bool(getattr(cfg, "mem_hard_stop_enabled", False)),
        "mem_guard_interval_s": float(getattr(cfg, "mem_guard_interval_s", 5.0)),
        "mem_ctx_checkpoints": int(getattr(cfg, "mem_ctx_checkpoints", 0) or 0),
        "mem_vram_total_gb": getattr(cfg, "vram_total_gb", None),
    }
    try:
        st = mem_guard.read_mem_state()
        out["mem_ram_total_gb"] = round(st.ram_total_gb, 1)
        out["mem_ram_available_gb"] = round(st.ram_available_gb, 1)
        out["mem_swap_total_gb"] = round(st.swap_total_gb, 1)
        out["mem_swap_used_gb"] = round(st.swap_used_gb, 1)
        out["mem_pressure"] = mem_guard.classify_pressure(
            st, mem_guard.MemThresholds.from_cfg(cfg)).name.lower()
    except Exception as e:  # noqa: BLE001 — display-only
        log.debug("mem state read failed: %s", e)
    return out


def _settings_ctx(request: Request, **extra: Any) -> dict:
    cfg = request.app.state.cfg
    supervisor = request.app.state.supervisor
    ctx = _about_ctx(
        request,
        autorun_mode=_autorun_label(),
        autolaunch=bool(cfg.autolaunch),
        autorestart=bool(supervisor.enabled),
        update_action_base="/ui/settings",
    )
    ctx.update(_gpu_device_ctx(cfg))
    ctx.update(_mem_guard_ctx(cfg))
    ctx["conversation_retention_days"] = int(
        getattr(cfg, "conversation_retention_days", 0) or 0)
    ctx["lock_model_loading"] = bool(getattr(cfg, "lock_model_loading", False))
    ctx.update(extra)
    return ctx


@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request,
                        _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "settings.html",
                                      _settings_ctx(request))


@router.post("/settings/check-update", response_class=HTMLResponse)
async def settings_check_update(request: Request,
                                _: None = Depends(require_csrf)) -> HTMLResponse:
    try:
        info = _check_latest_release()
        return templates.TemplateResponse(request, "_update_area.html", _settings_ctx(
            request,
            update_available=info["is_newer"],
            latest_version=info["latest"],
            install_mode=info["install_mode"],
            install_location=info["install_location"],
        ))
    except Exception as e:
        return templates.TemplateResponse(request, "_update_area.html", _settings_ctx(
            request,
            update_check_error=f"Could not check for updates: {e}",
            install_mode=_detect_install_mode()["mode"],
        ))


@router.post("/settings/update", response_class=HTMLResponse)
async def settings_update(request: Request,
                          _: None = Depends(require_csrf)) -> Response:
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _run_self_update)
    install_mode = res.get("mode", "unknown")
    if not res["ok"]:
        return templates.TemplateResponse(request, "settings.html", _settings_ctx(
            request,
            update_check_error=res.get("error", "update failed"),
            update_log=res["log"],
            install_mode=install_mode,
        ))
    sm: ServerManager = request.app.state.sm
    if sm.is_running:
        await sm.stop()
    await _schedule_self_restart()
    return templates.TemplateResponse(request, "settings.html", _settings_ctx(
        request,
        update_log=res["log"] + "\n\nUpdate complete. Restarting…",
        update_available=False,
        latest_version="restarting",
    ))


@router.post("/settings/autostart", response_class=HTMLResponse)
async def settings_autostart(request: Request,
                             mode: str = Form(...),
                             _: None = Depends(require_csrf)) -> Response:
    """Configure run-at-startup. Maps the three user-facing choices onto
    `llamanager autostart --mode`, run as a subprocess so it reuses the exact
    per-platform logic the CLI/tray use. Runs in the daemon's session, so for
    a user-session daemon it needs no prompt; a system/headless daemon may
    lack the rights for 'before login' — the result is logged either way."""
    allowed = {"off": "off", "login": "login-tray", "boot": "tray+service"}
    cli_mode = allowed.get(mode)
    if cli_mode is None:
        raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}")
    cmd = [sys.executable, "-m", "llamanager", "autostart", "--mode", cli_mode]
    try:
        r = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            log.info("autostart set to %s via UI", cli_mode)
        else:
            log.error("autostart %s via UI failed: %s", cli_mode,
                      (r.stderr or r.stdout).strip())
    except (OSError, subprocess.SubprocessError) as e:
        log.error("autostart %s via UI errored: %s", cli_mode, e)
    return RedirectResponse("/ui/settings", status_code=303)


@router.post("/settings/autolaunch", response_class=HTMLResponse)
async def settings_autolaunch(request: Request,
                              enabled: str = Form("off"),
                              _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    on = (enabled == "on")
    cfg.autolaunch = on
    update_defaults(cfg.config_path, autolaunch=on)
    return RedirectResponse("/ui/settings", status_code=303)


@router.post("/settings/autorestart", response_class=HTMLResponse)
async def settings_autorestart(request: Request,
                               enabled: str = Form("off"),
                               _: None = Depends(require_csrf)) -> Response:
    request.app.state.supervisor.enabled = (enabled == "on")
    return RedirectResponse("/ui/settings", status_code=303)


@router.post("/settings/lock-model-loading", response_class=HTMLResponse)
async def settings_lock_model_loading(request: Request,
                                      enabled: str = Form("off"),
                                      _: None = Depends(require_csrf)) -> Response:
    """Toggle the operator lock that forbids request-triggered model loads.

    Takes effect immediately (the dispatcher reads ``cfg.lock_model_loading``
    on each dispatch) and is persisted to the [queue] section so it survives
    a restart."""
    from .config import update_queue_settings
    cfg = request.app.state.cfg
    on = (enabled == "on")
    cfg.lock_model_loading = on
    update_queue_settings(cfg.config_path, lock_model_loading=on)
    log.info("lock_model_loading set to %s via UI", on)
    return RedirectResponse("/ui/settings", status_code=303)


@router.post("/settings/gpu", response_class=HTMLResponse)
async def settings_gpu(request: Request,
                       llama_gpu_device: str = Form(""),
                       _: None = Depends(require_csrf)) -> Response:
    """Pin the text engine to one GPU (or clear the pin with empty value).

    Stored by device name; takes effect on the next engine start/restart.
    """
    from .config import update_server_gpu
    cfg = request.app.state.cfg
    name = (llama_gpu_device or "").strip()
    cfg.llama_gpu_device = name
    update_server_gpu(cfg.config_path, device_name=name)
    return RedirectResponse("/ui/settings", status_code=303)


@router.post("/settings/mem-guard", response_class=HTMLResponse)
async def settings_mem_guard(request: Request,
                             enabled: str = Form("off"),
                             clamp_ctx: str = Form("off"),
                             hard_stop_enabled: str = Form("off"),
                             ctx_checkpoints: str = Form(""),
                             _: None = Depends(require_csrf)) -> Response:
    """Persist the memory-guardrail knobs (the [mem_guard] config section).

    Booleans arrive only when their checkbox is ticked; an unchecked box sends
    nothing, so the Form default "off" correctly reads as disabled. The whole
    form posts on any change so each toggle saves immediately.
    """
    from .config import update_mem_guard
    cfg = request.app.state.cfg
    on_enabled = (enabled == "on")
    on_clamp = (clamp_ctx == "on")
    on_hard = (hard_stop_enabled == "on")
    try:
        ckpt = max(0, int(ctx_checkpoints)) if str(ctx_checkpoints).strip() else 0
    except (TypeError, ValueError):
        ckpt = 0
    cfg.mem_guard_enabled = on_enabled
    cfg.mem_clamp_ctx = on_clamp
    cfg.mem_hard_stop_enabled = on_hard
    cfg.mem_ctx_checkpoints = ckpt
    update_mem_guard(cfg.config_path, enabled=on_enabled, clamp_ctx=on_clamp,
                     hard_stop_enabled=on_hard, ctx_checkpoints=ckpt)
    # Apply the enable/disable to the live watchdog without a restart.
    wd = getattr(request.app.state, "mem_watchdog", None)
    if wd is not None:
        try:
            wd.enabled = on_enabled
            if on_enabled:
                wd.start()
            else:
                await wd.stop()
        except Exception as e:  # noqa: BLE001
            log.debug("watchdog toggle failed: %s", e)
    return RedirectResponse("/ui/settings", status_code=303)


@router.post("/settings/conversation-retention", response_class=HTMLResponse)
async def settings_conversation_retention(
        request: Request,
        retention_days: str = Form(""),
        _: None = Depends(require_csrf)) -> Response:
    """Persist conversation-text retention (the [conversation] section).

    0 disables prompt/response capture; any positive value keeps text that
    many days. The change is applied to existing rows immediately so lowering
    the window (or setting 0) clears stored text now, not just at the next
    daily prune.
    """
    from .config import update_conversation_retention
    cfg = request.app.state.cfg
    db = request.app.state.db
    try:
        days = max(0, int(retention_days)) if str(retention_days).strip() else 0
    except (TypeError, ValueError):
        days = int(getattr(cfg, "conversation_retention_days", 0) or 0)
    cfg.conversation_retention_days = days
    update_conversation_retention(cfg.config_path, retention_days=days)
    try:
        db.prune_conversations(days)
    except Exception as e:  # noqa: BLE001
        log.debug("conversation prune failed: %s", e)
    return RedirectResponse("/ui/settings", status_code=303)


# ---------- chat ----------

@router.get("/chat", response_class=HTMLResponse)
async def chat_view(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    cfg = request.app.state.cfg
    sm: ServerManager = request.app.state.sm
    reg: Registry = request.app.state.registry
    sess = _current_session(request)
    # Filter to LLM-family (text engines only). Image-family models
    # don't belong in the chat picker even though they show up in the
    # registry — confirms via ENGINE_FAMILY.
    llm_models: list[str] = []
    for m in reg.list():
        engine = detect_engine_for_id(m.model_id, cfg.models_dir)
        if ENGINE_FAMILY.get(engine, "text") == "text":
            llm_models.append(m.model_id)
    llm_model_set = set(llm_models)
    # Only profiles bound to LLM models are surfaced. ``vision`` is
    # True iff the profile has an ``mmproj`` set, which the chat UI
    # uses to decide whether to surface the image-upload button.
    profiles = [
        {"name": p.name, "model": mid, "vision": bool(p.mmproj)}
        for mid, p in cfg.iter_profiles()
        if mid in llm_model_set
    ]
    return templates.TemplateResponse(request, "chat.html", _ctx(
        request,
        status=sm.status(),
        profiles=profiles,
        profiles_json=json.dumps(profiles),
        model_ids=llm_models,
        api_key=sess["key"] if sess else "",
    ))


# ---------- images ----------

def _build_image_page_context(cfg, reg) -> dict[str, Any]:
    """Shared context for the admin + public image-gen pages.

    Walks each image engine's ``profile_schema()`` so the composer can
    auto-render the right inputs per kind. Also flattens profile field
    values so the page can pre-fill the override inputs from whatever
    the currently-selected profile already sets.
    """
    image_models: list[dict[str, str]] = []
    engines_in_use: set[str] = set()
    for m in reg.list():
        engine = detect_engine_for_id(m.model_id, cfg.models_dir)
        if ENGINE_FAMILY.get(engine, "text") == "image":
            image_models.append({"model_id": m.model_id, "engine": engine})
            engines_in_use.add(engine)

    # Per-engine schema, shape matches what diffusion_models.html consumes.
    engines_schema: dict[str, list[dict[str, Any]]] = {}
    for eng in engines_in_use:
        try:
            mod = image_engines.get(eng)
        except KeyError:
            continue
        engines_schema[eng] = [
            _serialize_profile_field(f) for f in mod.profile_schema()
        ]

    image_model_ids = {m["model_id"] for m in image_models}
    profiles_with_fields: list[dict[str, Any]] = []
    for mid, p in cfg.iter_profiles():
        if mid not in image_model_ids:
            continue
        fields: dict[str, Any] = {}
        eng = next((m["engine"] for m in image_models if m["model_id"] == mid),
                   None)
        if eng and eng in engines_schema:
            for s in engines_schema[eng]:
                v = getattr(p, s["key"], None)
                if v is None or v == "":
                    continue
                fields[s["key"]] = str(v)
        profiles_with_fields.append({
            "name": p.name, "model": mid, "fields": fields,
        })

    # Preselect the operator's configured default image model/profile so the
    # composer opens on it instead of whatever happens to be first in the
    # registry (which is why the page used to default to HiDream).
    default_model = cfg.default_image_model or ""
    if default_model not in image_model_ids:
        default_model = image_models[0]["model_id"] if image_models else ""
    default_profile = cfg.default_image_profile or ""

    # Per-engine reference-image capabilities, so the composer can show the
    # right attach / strength / keep-aspect controls for the chosen model.
    engines_caps = {eng: image_engines.capabilities(eng)
                    for eng in engines_in_use}

    return {
        "image_models": image_models,
        "image_models_json": json.dumps(image_models),
        "profiles_json": json.dumps(profiles_with_fields),
        "engines_schema_json": json.dumps(engines_schema),
        "engines_caps_json": json.dumps(engines_caps),
        "default_image_model": default_model,
        "default_image_profile": default_profile,
    }


@router.get("/images", response_class=HTMLResponse)
async def images_view(request: Request,
                      _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    cfg = request.app.state.cfg
    reg: Registry = request.app.state.registry
    sess = _current_session(request)
    ctx = _build_image_page_context(cfg, reg)
    return templates.TemplateResponse(request, "images.html", _ctx(
        request,
        api_key=sess["key"] if sess else "",
        **ctx,
    ))


def _safe_path_components(*parts: str) -> None:
    """Reject path components that contain separators / traversal / nulls."""
    for seg in parts:
        if "/" in seg or "\\" in seg or ".." in seg or "\x00" in seg:
            raise HTTPException(status_code=400, detail="invalid path component")


def _list_gallery(images_dir: Path, *, origin_filter: str | None = None,
                  limit: int = 200,
                  before: float | None = None) -> dict[str, Any]:
    """Walk ``images_dir`` newest-first and return a JSON-ready gallery page.

    Layout on disk is ``<images_dir>/<day>/<origin>/<name>.png`` (written by
    ``image_runner._gallery_dir``). For each PNG we attempt to read its
    ``*.png.json`` sidecar (written at ``image_runner.py:461-465``) — when
    the sidecar is absent or unreadable we still surface the file with
    minimal metadata so the gallery remains usable for legacy items.

    ``origin_filter`` scopes the listing to one origin name (already
    sanitised by the caller); ``before`` paginates via mtime cursor.
    """
    if not images_dir.exists():
        return {"items": [], "next_before": None}
    items: list[dict[str, Any]] = []
    # Walk the dated subdirs newest-first. We rely on the YYYY-MM-DD
    # naming used by _gallery_dir, but fall back to mtime sort for any
    # directory that doesn't follow the convention.
    try:
        day_dirs = sorted(
            (p for p in images_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError:
        return {"items": [], "next_found": None}
    for day_dir in day_dirs:
        day = day_dir.name
        try:
            origin_dirs = [p for p in day_dir.iterdir() if p.is_dir()]
        except OSError:
            continue
        if origin_filter:
            origin_dirs = [p for p in origin_dirs if p.name == origin_filter]
        for origin_dir in origin_dirs:
            try:
                files = list(origin_dir.iterdir())
            except OSError:
                continue
            for f in files:
                if not f.is_file() or f.suffix.lower() != ".png":
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                items.append({
                    "_mtime": st.st_mtime,
                    "_path": f,
                    "day": day,
                    "origin": origin_dir.name,
                    "name": f.name,
                    "size": st.st_size,
                })
    items.sort(key=lambda x: x["_mtime"], reverse=True)
    if before is not None:
        items = [it for it in items if it["_mtime"] < before]
    page = items[:limit]
    next_before = page[-1]["_mtime"] if len(items) > limit else None
    out: list[dict[str, Any]] = []
    for it in page:
        sidecar: dict[str, Any] = {}
        sc = it["_path"].with_suffix(it["_path"].suffix + ".json")
        if sc.is_file():
            try:
                sidecar = json.loads(sc.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                sidecar = {}
        out.append({
            "day": it["day"],
            "origin": it["origin"],
            "name": it["name"],
            "mtime": it["_mtime"],
            "size": it["size"],
            "url": f"/ui/images/file/{it['day']}/{it['origin']}/{it['name']}",
            "sidecar": sidecar,
        })
    return {"items": out, "next_before": next_before}


@router.get("/images/gallery")
async def images_gallery(request: Request,
                         limit: int = 60,
                         before: float | None = None,
                         _: Origin = Depends(require_admin_ui)) -> Response:
    """JSON listing of every image on disk, newest first.

    Pagination uses an mtime cursor (``before=<float>``) so the client can
    request the next page once the previous page's last image has been
    rendered. The admin endpoint sees every origin's gallery; the public
    variant in ``app.py`` scopes by the bearer's origin name.
    """
    from fastapi.responses import JSONResponse as _JSONResponse
    cfg = request.app.state.cfg
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1..500")
    payload = _list_gallery(cfg.images_dir, limit=limit, before=before)
    return _JSONResponse(payload)


@router.get("/images/file/{day}/{origin}/{name}")
async def images_file_serve(request: Request, day: str, origin: str, name: str,
                            _: Origin = Depends(require_admin_ui)) -> Response:
    """Serve a previously generated PNG. Authenticated by the same admin
    session that owns the rest of /ui. Path components are sanitised
    against traversal."""
    from fastapi.responses import FileResponse as _FileResponse
    cfg = request.app.state.cfg
    for seg in (day, origin, name):
        if "/" in seg or "\\" in seg or ".." in seg or "\x00" in seg:
            raise HTTPException(status_code=400, detail="invalid path component")
    if not name.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="only .png files are served here")
    p = (cfg.images_dir / day / origin / name).resolve()
    try:
        p.relative_to(cfg.images_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes images_dir")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return _FileResponse(p, media_type="image/png")


# ---------- logs ----------

@router.get("/logs", response_class=HTMLResponse)
async def logs_view(request: Request, source: str = "activity",
                    tail: int = 200,
                    _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    from .activity import discover_engine_logs
    cfg = request.app.state.cfg
    engine_logs = discover_engine_logs(cfg.logs_dir)
    text = ""
    if source == "activity":
        from .activity import build_activity, render_activity
        entries = build_activity(
            request.app.state.db, cfg.logs_dir, tail=tail,
        )
        text = render_activity(entries)
    else:
        if source == "llama-server":
            p = cfg.logs_dir / "llama-server.log"
        elif source == "llamanager":
            p = cfg.logs_dir / "llamanager.log"
        else:
            engines = {n: pth for n, pth in engine_logs}
            if source not in engines:
                text = f"(unknown log source: {source})"
                p = None
            else:
                p = engines[source]
        if p is not None and p.exists():
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
        engine_logs=[name for name, _p in engine_logs],
    ))


@router.get("/_partials/logs", response_class=HTMLResponse)
async def logs_partial(request: Request, source: str = "activity",
                       tail: int = 200,
                       _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return await logs_view(request, source=source, tail=tail, _=_)


# ---------- setup ----------

@router.get("/setup", response_class=HTMLResponse)
async def setup_get(request: Request, _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "setup.html", _setup_ctx(request))


def _krea_lora_cache(request: Request) -> dict[str, Any]:
    return getattr(request.app.state, "krea_lora_collection", None) or {
        "items": [],
        "error": "",
        "last_checked": 0.0,
        "title": "Krea 2 LoRAs",
        "description": "",
    }


async def _fetch_krea_lora_collection() -> dict[str, Any]:
    import httpx as _httpx

    async with _httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(KREA_LORA_COLLECTION_API)
        resp.raise_for_status()
        payload = resp.json()
        raw_items = [
            item for item in payload.get("items", [])
            if item.get("repoType") in (None, "model") and item.get("id")
        ]

        async def enrich(item: dict[str, Any]) -> dict[str, Any]:
            repo_id = str(item.get("id") or "")
            filename = ""
            for provider in item.get("availableInferenceProviders") or []:
                candidate = provider.get("adapterWeightsPath")
                if candidate:
                    filename = str(candidate)
                    break
            size = 0
            if not filename:
                try:
                    tree_url = HF_MODEL_TREE_API.format(repo=repo_id)
                    tree_resp = await client.get(tree_url)
                    tree_resp.raise_for_status()
                    for f in tree_resp.json():
                        path = str(f.get("path") or "")
                        if path.lower().endswith(".safetensors"):
                            filename = path
                            size = int(f.get("size") or (f.get("lfs") or {}).get("size") or 0)
                            break
                except Exception:
                    pass
            elif repo_id:
                try:
                    tree_url = HF_MODEL_TREE_API.format(repo=repo_id)
                    tree_resp = await client.get(tree_url)
                    tree_resp.raise_for_status()
                    for f in tree_resp.json():
                        if str(f.get("path") or "") == filename:
                            size = int(f.get("size") or (f.get("lfs") or {}).get("size") or 0)
                            break
                except Exception:
                    pass
            label = repo_id.rsplit("/", 1)[-1].removeprefix("Krea-2-LoRA-")
            previews = item.get("widgetOutputUrls") or []
            return {
                "repo": repo_id,
                "filename": filename,
                "label": label,
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "updated": item.get("lastModified") or "",
                "size": size,
                "size_gb": (size / (1024 ** 3)) if size else 0,
                "preview": previews[0] if previews else "",
            }

        items = await asyncio.gather(*(enrich(item) for item in raw_items))
    return {
        "items": sorted(items, key=lambda x: x.get("label", "")),
        "error": "",
        "last_checked": time.time(),
        "title": payload.get("title") or "Krea 2 LoRAs",
        "description": payload.get("description") or "",
    }


def _setup_diffusion_ctx(request: Request) -> dict[str, Any]:
    """Context for the Diffusion engines page (full or partial reload).

    Combines:
      * the standard model-list context (text_models / image_models,
        current_model, default_model, csrf, etc.)
      * image-engine paths (HiDream / Flux2 / Z-Image)
      * coexistence policy
      * per-engine install plans + the most recent install job per engine
      * a tiny ``downloads_by_repo`` index keyed by HF repo id so each
        engine card can show "downloading model X — 42%" inline.
    """
    cfg = request.app.state.cfg
    installer = request.app.state.engine_installer
    from .engine_installer import ENGINE_PLANS, resolve_plan, venv_python
    from .gpu_detect import detect_gpu, render_group_ok

    ctx = _models_ctx(request)
    ctx["image_cfg"] = {
        "hidream_python": cfg.hidream_python,
        "hidream_repo": cfg.hidream_repo,
        "flux2_sd_cli": cfg.flux2_sd_cli,
        "flux2_device_index": cfg.flux2_device_index,
        "z_image_python": cfg.z_image_python,
        "ideogram4_python": cfg.ideogram4_python,
    }
    ctx["coex"] = {
        "unload_text_on_arrival": cfg.unload_text_on_arrival,
        "restart_text_after_image": cfg.restart_text_after_image,
        "allow_concurrent": cfg.allow_concurrent,
    }

    # Surface detected GPU so the engine cards can render the right
    # pre-install options (e.g. pre-check the flash-attn patch on AMD).
    gpu = detect_gpu()
    ctx["gpu"] = {
        "kind": gpu.kind,
        "rocm_arch": gpu.rocm_arch,
        "needs_render_group": gpu.needs_render_group,
        "render_group_ok": render_group_ok(gpu),
    }

    # Per-engine install state — surface the most-relevant row so the
    # card can show "running 42%", "done", or "click to install".
    engines = ("z_image", "krea", "ideogram4", "hidream", "flux2")
    install_state: dict[str, Any] = {}
    for eng in engines:
        active = installer.active_for_engine(eng)
        if active:
            install_state[eng] = active
        else:
            recent = installer.list_for_engine(eng, limit=1)
            install_state[eng] = recent[0] if recent else None
    ctx["engine_installs"] = install_state
    # Render the GPU-resolved plan (which on AMD swaps in the ROCm wheels
    # and notes) when one exists; otherwise fall back to the generic
    # baseline so the card still has packages/notes to show.
    plans: dict[str, dict[str, Any]] = {}
    for e, base in ENGINE_PLANS.items():
        resolved = resolve_plan(e, gpu, emit=None, cfg=cfg)
        if resolved is not None:
            wheel_basenames = [u.rsplit("/", 1)[-1] for u in resolved.wheel_urls]
            plans[e] = {
                "engine": e,
                "label": resolved.label,
                "packages": resolved.packages,
                "wheel_urls": wheel_basenames,
                "space_mb": resolved.space_mb,
                "notes": resolved.notes,
                "target": resolved.target,
                "supports_flash_attn_patch": resolved.supports_flash_attn_patch,
                "has_diffusers": any(
                    str(p).startswith("diffusers==")
                    for p in resolved.packages
                ),
                "has_plan": True,
            }
        else:
            plans[e] = {
                "engine": e,
                "label": base.label,
                "packages": base.packages,
                "wheel_urls": [],
                "space_mb": base.space_mb,
                "notes": base.notes,
                "target": gpu.kind,
                "supports_flash_attn_patch": False,
                "has_diffusers": any(
                    str(p).startswith("diffusers==")
                    for p in base.packages
                ),
                "has_plan": True,
            }
    ctx["engine_plans"] = plans
    # Predicted venv interpreter path so the UI can show where pip will
    # plant things (or where it already lives).
    ctx["engine_venv_python"] = {
        e: str(venv_python(cfg, e)) for e in engines
    }
    try:
        from .engines import krea as _krea
        ctx["krea_quant_files"] = [
            {
                "filename": fn,
                "label": fn.removeprefix("krea2_turbo-").removesuffix(".gguf"),
                "size_gb": _krea.QUANT_SIZE_GB.get(fn),
            }
            for fn in _krea.QUANT_FILES
        ]
    except Exception:
        ctx["krea_quant_files"] = []
    ctx["krea_lora_collection"] = _krea_lora_cache(request)

    # Index downloads by repo so each engine card can show in-flight
    # pulls for its own models.
    downloads_by_repo: dict[str, dict[str, Any]] = {}
    for d in ctx.get("downloads", []) or []:
        src = d.get("source") or ""
        if src.startswith("hf://") or src.startswith("hf:"):
            repo = src.removeprefix("hf://").removeprefix("hf:")
            downloads_by_repo[repo] = d
    ctx["downloads_by_repo"] = downloads_by_repo

    # Per-engine "auto-update when idle" switch state.
    ctx["auto_update_engines"] = dict(cfg.auto_update_engines or {})
    # Diffusers version-picker cache (populated on demand by the Versions
    # button), keyed by engine → {installed, target, pin, versions, error}.
    ctx["diffusers_versions"] = (
        getattr(request.app.state, "diffusers_versions", {}) or {})

    return ctx


@router.get("/setup-diffusion", response_class=HTMLResponse)
async def setup_diffusion_get(request: Request,
                              _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Diffusion-engines page: image engine paths, coexistence policy,
    and the installed diffusion model list."""
    return templates.TemplateResponse(
        request, "setup_diffusion.html", _setup_diffusion_ctx(request),
    )


@router.post("/setup/binary-path", response_class=HTMLResponse)
async def setup_set_binary(request: Request,
                           binary_path: str = Form(...),
                           _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    binary_path = binary_path.strip()
    cfg.llama_server_binary = binary_path
    patch_config_binary(cfg.config_path, binary_path)
    return RedirectResponse("/ui/setup", status_code=303)


@router.post("/setup/image/hidream", response_class=HTMLResponse)
async def setup_hidream(request: Request,
                        hidream_python: str = Form(""),
                        hidream_repo: str = Form(""),
                        _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    cfg.hidream_python = hidream_python.strip()
    cfg.hidream_repo = hidream_repo.strip()
    update_image_config(
        cfg.config_path,
        hidream_python=cfg.hidream_python,
        hidream_repo=cfg.hidream_repo,
    )
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.post("/setup/image/flux2", response_class=HTMLResponse)
async def setup_flux2(request: Request,
                      flux2_sd_cli: str = Form(""),
                      flux2_device_index: str = Form(""),
                      _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    cfg.flux2_sd_cli = flux2_sd_cli.strip()
    idx_raw = flux2_device_index.strip()
    idx = None
    if idx_raw:
        try:
            idx = int(idx_raw)
        except ValueError:
            idx = None
    cfg.flux2_device_index = idx
    update_image_config(
        cfg.config_path,
        flux2_sd_cli=cfg.flux2_sd_cli,
        flux2_device_index=idx,
        clear_flux2_device_index=(idx is None),
    )
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.post("/setup/image/z-image", response_class=HTMLResponse)
async def setup_z_image(request: Request,
                        z_image_python: str = Form(""),
                        _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    cfg.z_image_python = z_image_python.strip()
    update_image_config(cfg.config_path, z_image_python=cfg.z_image_python)
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.post("/setup/image/ideogram4", response_class=HTMLResponse)
async def setup_ideogram4(request: Request,
                          ideogram4_python: str = Form(""),
                          _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    cfg.ideogram4_python = ideogram4_python.strip()
    update_image_config(cfg.config_path, ideogram4_python=cfg.ideogram4_python)
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.post("/setup/coexistence", response_class=HTMLResponse)
async def setup_coexistence(request: Request,
                            unload_text_on_arrival: str = Form(""),
                            restart_text_after_image: str = Form(""),
                            allow_concurrent: str = Form(""),
                            _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    cfg.unload_text_on_arrival = bool(unload_text_on_arrival)
    cfg.restart_text_after_image = bool(restart_text_after_image)
    cfg.allow_concurrent = bool(allow_concurrent)
    update_coexistence_policy(
        cfg.config_path,
        unload_text_on_arrival=cfg.unload_text_on_arrival,
        restart_text_after_image=cfg.restart_text_after_image,
        allow_concurrent=cfg.allow_concurrent,
    )
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


# ---- per-engine install + download ----

@router.post("/setup-diffusion/engine/{engine}/install-deps",
             response_class=HTMLResponse)
async def install_engine_deps(request: Request, engine: str,
                              patch_flash_attn: str = Form(""),
                              diffusers_version: str = Form(""),
                              reset_diffusers: str = Form(""),
                              torch_backend: str = Form("auto"),
                              _: None = Depends(require_csrf)) -> Response:
    """Kick off an opinionated venv + pip install for one engine.

    ``diffusers_version`` pins a specific diffusers (upgrade/downgrade) and
    persists as an override; ``reset_diffusers`` clears the override and
    reinstalls the shipped pin. ``torch_backend`` (auto/rocm/cuda/cpu)
    chooses which torch build to install."""
    from .config import set_diffusers_override
    from .engine_installer import TORCH_BACKENDS
    cfg = request.app.state.cfg
    installer = request.app.state.engine_installer
    options: dict[str, Any] = {}
    # Checkbox values arrive as "on" when checked, "" when not.
    if patch_flash_attn:
        options["patch_flash_attn"] = True
    tb = (torch_backend or "auto").strip().lower()
    if tb in TORCH_BACKENDS and tb != "auto":
        options["torch_backend"] = tb
    chosen = (diffusers_version or "").strip()
    if reset_diffusers:
        set_diffusers_override(cfg.config_path, engine, None)
        cfg.image_diffusers_version = {
            k: v for k, v in (cfg.image_diffusers_version or {}).items()
            if k != engine
        }
    elif chosen:
        options["diffusers_version"] = chosen
        set_diffusers_override(cfg.config_path, engine, chosen)
        cfg.image_diffusers_version = dict(cfg.image_diffusers_version or {})
        cfg.image_diffusers_version[engine] = chosen
    try:
        installer.start(engine, options=options)
    except (ValueError, RuntimeError) as e:
        return _error_html(f"install failed: {e}", status_code=400)
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.post("/setup-diffusion/engine/{engine}/install-deps/{install_id}/cancel",
             response_class=HTMLResponse)
async def cancel_engine_install(request: Request, engine: str,
                                install_id: str,
                                _: None = Depends(require_csrf)) -> Response:
    installer = request.app.state.engine_installer
    installer.cancel(install_id)
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.post("/setup-diffusion/engine/{engine}/download-model",
             response_class=HTMLResponse)
async def download_engine_model(request: Request, engine: str,
                                repo: str = Form(...),
                                subfolder: str = Form(""),
                                filename: str = Form(""),
                                _: None = Depends(require_csrf)) -> Response:
    """Start an HF whole-repo, subfolder, or single-file diffusion download."""
    reg: Registry = request.app.state.registry
    repo_clean = repo.strip()
    if not repo_clean:
        return _error_html("repo is required", status_code=400)
    # Normalise: accept a full HF URL or a plain org/name.
    if repo_clean.startswith("http"):
        # _normalise_hf_url will rewrite when start_pull runs; pass as-is.
        source = repo_clean
    else:
        # Strip an accidental hf:// prefix; we'll re-add canonically.
        source = "hf://" + repo_clean.removeprefix("hf://").removeprefix("hf:")
    sub = subfolder.strip().strip("/") or None
    fn = filename.strip()
    try:
        # Estimate size up-front so the progress bar can show a total.
        bare_repo = source.removeprefix("hf://").removeprefix("hf:")
        if fn:
            size = await reg.estimate_repo_size(bare_repo, files=[fn])
            reg.start_pull(source=source, files=[fn],
                           subfolder=None, whole_repo=False,
                           bytes_total=size)
        else:
            size = await reg.estimate_repo_size(bare_repo, sub)
            reg.start_pull(source=source, files=None,
                           subfolder=sub, whole_repo=True,
                           bytes_total=size)
    except Exception as e:
        return _error_html(f"pull failed: {e}", status_code=400)
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


@router.get("/setup-diffusion/_partial", response_class=HTMLResponse)
async def setup_diffusion_partial(request: Request,
                                  _: Origin = Depends(require_admin_ui)
                                  ) -> HTMLResponse:
    """HTMX-polled fragment so the install/download status updates live
    on the Diffusion engines page without a full reload."""
    return templates.TemplateResponse(
        request, "_setup_diffusion_partial.html",
        _setup_diffusion_ctx(request),
    )


@router.get("/setup-diffusion/krea-loras", response_class=HTMLResponse)
async def setup_diffusion_krea_loras(request: Request,
                                     _: Origin = Depends(require_admin_ui)
                                     ) -> HTMLResponse:
    try:
        request.app.state.krea_lora_collection = await _fetch_krea_lora_collection()
    except Exception as e:
        previous = _krea_lora_cache(request)
        request.app.state.krea_lora_collection = {
            **previous,
            "error": str(e),
            "last_checked": time.time(),
        }
    return templates.TemplateResponse(
        request, "_setup_diffusion_partial.html",
        _setup_diffusion_ctx(request),
    )


@router.get("/setup-diffusion/versions", response_class=HTMLResponse)
async def setup_diffusion_versions(request: Request,
                                   engine: str = "",
                                   _: Origin = Depends(require_admin_ui)
                                   ) -> HTMLResponse:
    """Fetch the installable diffusers versions for one engine (+ its
    installed/target) and re-render the diffusion partial with the picker
    populated. Manual (button-triggered) to avoid per-load network/subprocess
    cost."""
    from .engine_installer import (
        list_diffusers_versions, installed_diffusers_version,
        diffusion_target_version, _plan_diffusers_pin,
    )
    cfg = request.app.state.cfg
    cache: dict[str, dict] = getattr(request.app.state, "diffusers_versions", None) or {}
    if engine:
        loop = asyncio.get_running_loop()
        listing = await loop.run_in_executor(None, list_diffusers_versions)
        installed = await loop.run_in_executor(
            None, installed_diffusers_version, cfg, engine)
        cache[engine] = {
            "installed": installed,
            "target": diffusion_target_version(engine, cfg),
            "pin": _plan_diffusers_pin(engine),
            **listing,
        }
        request.app.state.diffusers_versions = cache
    return templates.TemplateResponse(
        request, "_setup_diffusion_partial.html",
        _setup_diffusion_ctx(request),
    )


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
        version_lists=getattr(request.app.state, "install_versions", {}) or {},
        auto_update_engines=dict(cfg.auto_update_engines or {}),
        auto_update_idle_seconds=cfg.auto_update_idle_seconds,
        auto_update_check_interval_seconds=cfg.auto_update_check_interval_seconds,
        config_dir=str(cfg.config_path.parent),
        config_file=str(cfg.config_path),
        open_config_label=open_config_label,
        image_cfg={
            "hidream_python": cfg.hidream_python,
            "hidream_repo": cfg.hidream_repo,
            "flux2_sd_cli": cfg.flux2_sd_cli,
            "flux2_device_index": cfg.flux2_device_index,
        },
        coex={
            "unload_text_on_arrival": cfg.unload_text_on_arrival,
            "restart_text_after_image": cfg.restart_text_after_image,
            "allow_concurrent": cfg.allow_concurrent,
        },
    )


@router.post("/setup/install", response_class=HTMLResponse)
async def setup_install(request: Request,
                        source: str = Form("llama.cpp"),
                        backend: str = Form(""),
                        version: str = Form(""),
                        _: None = Depends(require_csrf)) -> Response:
    source, backend = _resolve_variant(source, backend)
    vid = variant_id(source, backend)
    pinned = (version or "").strip() or None
    states: dict[str, InstallState] = request.app.state.install_states
    state = states.setdefault(vid, InstallState())
    if state.status != "running":
        state.status = "running"
        state.lines = []
        state.error = None
        state.installed_path = None
        asyncio.create_task(install_variant(state, source, backend, version=pinned))
    return templates.TemplateResponse(
        request, "setup.html",
        _setup_ctx(request, selected_source=source, selected_backend=backend),
    )


@router.get("/setup/versions", response_class=HTMLResponse)
async def setup_versions(request: Request,
                         variant: str = "",
                         hide: int = 0,
                         _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Fetch (or collapse) the installable version list for one variant and
    re-render the installed-variants partial. Manual (button-triggered) so
    GitHub/PyPI isn't hit on every page load.

    ``hide=1`` drops the variant from the cached version lists so the picker
    collapses back to the "Versions…" button — otherwise, once expanded, the
    cache entry would keep the picker open across every subsequent render.
    """
    from .llama_installer import list_versions
    parsed = parse_variant_id(variant) if variant else None
    cache: dict[str, dict] = getattr(request.app.state, "install_versions", None) or {}
    if parsed is not None:
        if hide:
            cache.pop(variant, None)
        else:
            source, backend = parsed
            loop = asyncio.get_running_loop()
            cache[variant] = await loop.run_in_executor(
                None, list_versions, source, backend)
        request.app.state.install_versions = cache
    return templates.TemplateResponse(
        request, "_installed_variants.html", _setup_ctx(request),
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
        parsed = parse_variant_id(variant)
        if cfg.llama_server_binary != state.installed_path:
            cfg.llama_server_binary = state.installed_path
            engine = engine_type_for(parsed[0]) if parsed else "llama"
            cfg.llama_server_engine = engine
            patch_config_binary(cfg.config_path, state.installed_path, engine=engine)
        # We just installed the latest build for this variant, so the cached
        # "update available" flag is now stale. Refresh it to the freshly
        # installed version + up-to-date, so the variant row stops offering
        # the build we just installed. (An *update* keeps the same path, so
        # this must run regardless of the binary-path check above.)
        if parsed is not None:
            meta = read_install_meta(parsed[0], parsed[1]) or {}
            ver = meta.get("version")
            updates = getattr(request.app.state, "install_updates", None)
            if isinstance(updates, dict) and ver:
                updates[variant] = {"latest": ver, "has_update": False,
                                    "current": ver}
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


def _set_auto_update_flag(request: Request, engine: str, enabled: bool) -> bool:
    """Validate + persist one engine's auto-update switch. Returns True on
    success, False for an unknown engine key (caller ignores the toggle)."""
    from .api_admin import _valid_auto_update_key
    from .config import update_auto_update
    if not _valid_auto_update_key(engine):
        return False
    cfg = request.app.state.cfg
    update_auto_update(cfg.config_path, engine=engine, enabled=enabled)
    cfg.auto_update_engines = dict(cfg.auto_update_engines or {})
    cfg.auto_update_engines[engine] = enabled
    request.app.state.db.log_event(
        "auto_update_toggled", {"engine": engine, "enabled": enabled})
    return True


@router.post("/setup/auto-update", response_class=HTMLResponse)
async def setup_auto_update_toggle(request: Request,
                                   engine: str = Form(...),
                                   enabled: str = Form(""),
                                   _: None = Depends(require_csrf)) -> Response:
    """Flip a llama variant's auto-update-when-idle switch (Setup page)."""
    _set_auto_update_flag(request, engine, enabled.strip().lower() == "on")
    return RedirectResponse("/ui/setup", status_code=303)


@router.post("/setup-diffusion/auto-update", response_class=HTMLResponse)
async def setup_diffusion_auto_update_toggle(request: Request,
                                             engine: str = Form(...),
                                             enabled: str = Form(""),
                                             _: None = Depends(require_csrf)) -> Response:
    """Flip a diffusion engine's auto-update-when-idle switch."""
    _set_auto_update_flag(request, engine, enabled.strip().lower() == "on")
    return RedirectResponse("/ui/setup-diffusion", status_code=303)


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


@router.get("/setup/_variants", response_class=HTMLResponse)
async def setup_variants(request: Request,
                         _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """Lightweight refresh of the installed-variants block (no network).

    The install-progress 'done' partial fires `hx-get /ui/setup/_variants` to
    show the newly-installed variant and clear its stale 'update available'
    flag. Without this route that fetch 404s — which renders as an
    'Error: Not Found' toast on every Setup-page load while an install state
    is 'done'."""
    return templates.TemplateResponse(
        request, "_installed_variants.html", _setup_ctx(request),
    )


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
    # Preserve runtime-only fields not persisted to config.toml (e.g. the
    # VRAM total detected once at startup, used by the memory guardrails).
    new_cfg.vram_total_gb = getattr(cfg, "vram_total_gb", None)
    request.app.state.cfg = new_cfg
    # Update registry to use potentially new models_dir
    request.app.state.registry.models_dir = new_cfg.models_dir


def _models_redirect(request: Request) -> Response:
    """Post-mutation response for /ui/models actions.

    For htmx (boosted) requests we send an ``HX-Redirect`` header, which makes
    htmx do a real client-side navigation (``location.href``) to the models
    page — a fresh full render, exactly like clicking the nav link. We do NOT
    return a 303 here: htmx would follow it with an XHR and swap the full page
    into ``<body>``, and that boosted whole-page swap proved persistently
    fragile for the models page (repeated "UI breaks on save", with and
    without a console error). A genuine navigation sidesteps the swap entirely.
    Non-htmx clients (rare) get the normal 303 so plain form posts still work.
    """
    if _wants_htmx(request):
        resp = Response(status_code=204)
        resp.headers["HX-Redirect"] = "/ui/models"
        return resp
    return RedirectResponse("/ui/models", status_code=303)


def _profile_saved_response(request: Request, model_id: str,
                            ctx_size: int | None,
                            mmproj: str = "") -> Response:
    """Post-save response for the profile editor. See ``_models_redirect``:
    a real navigation back to the models page, not a boosted body swap."""
    return _models_redirect(request)


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

    # Group profiles by their parent model for the JS-side filtering of
    # the profile dropdown. (Profiles are nested under models now.)
    profiles_by_model: dict[str, list[str]] = {}
    model_default_profiles: dict[str, str] = {}
    for mid, m in cfg.models.items():
        if m.profiles:
            profiles_by_model[mid] = sorted(m.profiles.keys())
        if m.default_profile:
            model_default_profiles[mid] = m.default_profile

    from . import exclusive as _exclusive
    last_sweep = _exclusive.last_result()
    return _ctx(
        request,
        status=sm.status(),
        model_ids=model_ids,
        profiles_by_model=profiles_by_model,
        global_profile_names=[],  # no more globals — kept for template compat
        model_default_profiles=model_default_profiles,
        default_model=cfg.default_model,
        default_profile="",
        autorestart=supervisor.enabled,
        autolaunch=cfg.autolaunch,
        max_restarts=cfg.max_restarts_in_window,
        log_tail=_read_log_tail(cfg),
        exclusive_mode=getattr(cfg, "exclusive_mode", "off"),
        exclusive_grace=getattr(cfg, "exclusive_grace_seconds", 5.0),
        exclusive_heartbeat=getattr(cfg, "exclusive_heartbeat_seconds", 120),
        exclusive_last=(last_sweep.to_dict() if last_sweep else None),
        exclusive_modes=list(_exclusive.VALID_MODES),
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


@router.post("/launch/exclusive", response_class=HTMLResponse)
async def launch_exclusive(request: Request,
                           mode: str = Form("off"),
                           _: None = Depends(require_csrf)) -> Response:
    from .config import VALID_EXCLUSIVE_MODES, load_config, update_exclusive_mode
    cfg = request.app.state.cfg
    m = (mode or "off").strip().lower()
    if m not in VALID_EXCLUSIVE_MODES:
        m = "off"
    try:
        update_exclusive_mode(cfg.path, mode=m)
    except (ValueError, OSError) as e:
        return _error_html(str(e), status_code=400)
    new_cfg = load_config(cfg.path)
    new_cfg.bind = cfg.bind
    new_cfg.port = cfg.port
    request.app.state.cfg = new_cfg
    return RedirectResponse("/ui/launch", status_code=303)


@router.post("/launch/exclusive-sweep", response_class=HTMLResponse)
async def launch_exclusive_sweep(request: Request,
                                 _: None = Depends(require_csrf)) -> Response:
    from . import exclusive as _exclusive
    cfg = request.app.state.cfg
    mode = (getattr(cfg, "exclusive_mode", "off") or "off").strip().lower()
    if mode == "off":
        # Show the operator what *would* be killed without doing anything.
        _exclusive.scan_and_record("warn")
    else:
        await _exclusive.sweep_and_record(
            mode,
            grace_seconds=float(getattr(cfg, "exclusive_grace_seconds", 5.0)),
        )
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
# Profiles always live as a child of a model. All mutating endpoints take
# (model_id, profile_name) and redirect back to /ui/models.


def _parse_optional_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"expected a number, got {s!r}")


def _parse_optional_int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"expected an integer, got {s!r}")


def _build_llm_profile_from_values(
    name: str,
    *,
    mmproj: str = "",
    ctx_size: int | None = None,
    vram_limit_gb: float | None = None,
    ram_spill_policy: str = "default",
    ram_spill_limit_gb: float | None = None,
    thinking: str = "",
    reasoning_budget: int | None = None,
    kv_cache_type: str = "",
    flash_attn: str = "",
    parallel: int | None = None,
    mtp: bool = False,
    mtp_n_max: int | None = None,
    args: dict[str, Any] | None = None,
) -> Profile:
    """Validate already-typed values and assemble an LLM ``Profile``.

    Shared by the UI form parser (which coerces strings first) and the
    admin JSON endpoints (which post typed values directly). Keeps a
    single source of truth for ram-spill / thinking validation so the
    CLI and the web UI reject the same junk.
    """
    policy = (ram_spill_policy or "default").strip().lower()
    if policy not in VALID_RAM_SPILL_POLICIES:
        raise ValueError(
            f"invalid ram_spill_policy {policy!r}; "
            f"must be one of {VALID_RAM_SPILL_POLICIES}"
        )
    if policy != "limited":
        ram_spill_limit_gb = None
    thinking_val = (thinking or "").strip().lower()
    if thinking_val not in VALID_THINKING:
        raise ValueError(
            f"invalid thinking {thinking_val!r}; must be '', 'on', or 'off'"
        )
    kv_val = (kv_cache_type or "").strip().lower()
    if kv_val not in VALID_KV_CACHE_TYPES:
        raise ValueError(
            f"invalid kv_cache_type {kv_val!r}; "
            f"must be one of {VALID_KV_CACHE_TYPES}"
        )
    fa_val = (flash_attn or "").strip().lower()
    if fa_val not in VALID_FLASH_ATTN:
        raise ValueError(
            f"invalid flash_attn {fa_val!r}; "
            f"must be one of {VALID_FLASH_ATTN}"
        )
    if reasoning_budget is not None and reasoning_budget < 0:
        raise ValueError(
            "reasoning_budget must be >= 0 (0 = no thinking; blank = unbounded)"
        )
    if parallel is not None and parallel < 1:
        raise ValueError(
            "parallel must be >= 1 (number of concurrent slots; blank = auto)"
        )
    if mtp_n_max is not None and mtp_n_max < 1:
        raise ValueError(
            "mtp_n_max must be >= 1 (drafted tokens per step; blank = 2)"
        )
    if mtp:
        # MTP self-speculation can't share the model with a vision projector
        # or run across multiple slots (llama.cpp restriction). The launcher
        # pins --parallel 1 when MTP is on; reject an explicit slot count > 1
        # so the saved profile matches what actually runs.
        if (mmproj or "").strip():
            raise ValueError(
                "MTP cannot be combined with an mmproj (vision projector); "
                "llama.cpp does not support --mmproj with --spec-type draft-mtp"
            )
        if parallel is not None and parallel > 1:
            raise ValueError(
                "MTP runs single-slot only; set parallel to 1 or leave it blank"
            )
    if not mtp:
        mtp_n_max = None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("args must be a JSON object / dict")
    return Profile(
        name=name,
        mmproj=(mmproj or "").strip(),
        ctx_size=ctx_size,
        vram_limit_gb=vram_limit_gb,
        ram_spill_policy=policy,
        ram_spill_limit_gb=ram_spill_limit_gb,
        thinking=thinking_val,
        reasoning_budget=reasoning_budget,
        kv_cache_type=kv_val,
        flash_attn=fa_val,
        parallel=parallel,
        mtp=bool(mtp),
        mtp_n_max=mtp_n_max,
        args=args,
    )


def _build_profile_from_form(
    name: str,
    *,
    mmproj: str,
    ctx_size: str,
    vram_limit_gb: str,
    vram_unlimited: str,
    ram_spill_policy: str,
    ram_spill_limit_gb: str,
    thinking: str,
    args_json: str,
    kv_cache_type: str = "",
    flash_attn: str = "",
    reasoning_budget: str = "",
    parallel: str = "",
    mtp: str = "",
    mtp_n_max: str = "",
) -> Profile:
    """String → typed wrapper around ``_build_llm_profile_from_values``.

    Used by the form-encoded UI endpoints (`/ui/models/profiles/*`).
    """
    try:
        ctx_size_val = _parse_optional_int(ctx_size)
        vram_val = (None if vram_unlimited == "on"
                    else _parse_optional_float(vram_limit_gb))
        ram_limit_val = _parse_optional_float(ram_spill_limit_gb)
        reasoning_budget_val = _parse_optional_int(reasoning_budget)
        parallel_val = _parse_optional_int(parallel)
        mtp_n_max_val = _parse_optional_int(mtp_n_max)
    except ValueError as e:
        raise ValueError(str(e))
    mtp_val = (mtp == "on")
    try:
        args = json.loads(args_json.strip() or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in args: {e}")
    return _build_llm_profile_from_values(
        name,
        mmproj=mmproj,
        ctx_size=ctx_size_val,
        vram_limit_gb=vram_val,
        ram_spill_policy=ram_spill_policy,
        ram_spill_limit_gb=ram_limit_val,
        thinking=thinking,
        reasoning_budget=reasoning_budget_val,
        kv_cache_type=kv_cache_type,
        flash_attn=flash_attn,
        parallel=parallel_val,
        mtp=mtp_val,
        mtp_n_max=mtp_n_max_val,
        args=args,
    )


@router.post("/models/profiles/create", response_class=HTMLResponse)
async def models_profile_create(request: Request,
                                model_id: str = Form(...),
                                name: str = Form(...),
                                mmproj: str = Form(""),
                                ctx_size: str = Form(""),
                                vram_limit_gb: str = Form(""),
                                vram_unlimited: str = Form(""),
                                ram_spill_policy: str = Form("default"),
                                ram_spill_limit_gb: str = Form(""),
                                thinking: str = Form(""),
                                reasoning_budget: str = Form(""),
                                kv_cache_type: str = Form(""),
                                flash_attn: str = Form(""),
                                parallel: str = Form(""),
                                mtp: str = Form(""),
                                mtp_n_max: str = Form(""),
                                args_json: str = Form("{}"),
                                make_default: str = Form(""),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    mid = model_id.strip()
    if not mid:
        return _error_html("model_id required", status_code=400)
    profile_name = name.strip().lower()
    existing = cfg.get_model(mid)
    if existing and profile_name in existing.profiles:
        return _error_html(
            f"profile '{profile_name}' already exists for model '{mid}'",
            status_code=409,
        )
    try:
        prof = _build_profile_from_form(
            profile_name,
            mmproj=mmproj, ctx_size=ctx_size,
            vram_limit_gb=vram_limit_gb, vram_unlimited=vram_unlimited,
            ram_spill_policy=ram_spill_policy,
            ram_spill_limit_gb=ram_spill_limit_gb,
            thinking=thinking,
            reasoning_budget=reasoning_budget,
            kv_cache_type=kv_cache_type,
            flash_attn=flash_attn,
            parallel=parallel,
            mtp=mtp,
            mtp_n_max=mtp_n_max,
            args_json=args_json,
        )
        save_profile(cfg.config_path, mid, profile_name, prof)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    if make_default == "on":
        set_model_default_profile(cfg.config_path, mid, profile_name)
    _reload_config(request)
    return _profile_saved_response(request, mid, prof.ctx_size, prof.mmproj)


@router.post("/models/profiles/{profile_name}/update", response_class=HTMLResponse)
async def models_profile_update(request: Request, profile_name: str,
                                model_id: str = Form(...),
                                new_name: str = Form(""),
                                mmproj: str = Form(""),
                                ctx_size: str = Form(""),
                                vram_limit_gb: str = Form(""),
                                vram_unlimited: str = Form(""),
                                ram_spill_policy: str = Form("default"),
                                ram_spill_limit_gb: str = Form(""),
                                thinking: str = Form(""),
                                reasoning_budget: str = Form(""),
                                kv_cache_type: str = Form(""),
                                flash_attn: str = Form(""),
                                parallel: str = Form(""),
                                mtp: str = Form(""),
                                mtp_n_max: str = Form(""),
                                args_json: str = Form("{}"),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    mid = model_id.strip()
    m = cfg.get_model(mid)
    if not m or profile_name not in m.profiles:
        return _error_html(
            f"profile '{profile_name}' not found for model '{mid}'",
            status_code=404,
        )
    target_name = (new_name.strip().lower() or profile_name)
    try:
        prof = _build_profile_from_form(
            target_name,
            mmproj=mmproj, ctx_size=ctx_size,
            vram_limit_gb=vram_limit_gb, vram_unlimited=vram_unlimited,
            ram_spill_policy=ram_spill_policy,
            ram_spill_limit_gb=ram_spill_limit_gb,
            thinking=thinking,
            reasoning_budget=reasoning_budget,
            kv_cache_type=kv_cache_type,
            flash_attn=flash_attn,
            parallel=parallel,
            mtp=mtp,
            mtp_n_max=mtp_n_max,
            args_json=args_json,
        )
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    if target_name != profile_name:
        if target_name in m.profiles:
            return _error_html(
                f"profile '{target_name}' already exists for model '{mid}'",
                status_code=409,
            )
        try:
            rename_profile(cfg.config_path, mid, profile_name, target_name)
        except ValueError as e:
            return _error_html(str(e), status_code=400)
    try:
        save_profile(cfg.config_path, mid, target_name, prof)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    _reload_config(request)
    return _profile_saved_response(request, mid, prof.ctx_size, prof.mmproj)


@router.post("/models/profiles/{profile_name}/delete", response_class=HTMLResponse)
async def models_profile_delete(request: Request, profile_name: str,
                                model_id: str = Form(...),
                                _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    mid = model_id.strip()
    delete_profile(cfg.config_path, mid, profile_name)
    _reload_config(request)
    return _models_redirect(request)


@router.post("/models/profiles/set-model-default", response_class=HTMLResponse)
async def models_set_model_default(request: Request,
                                   model_id: str = Form(...),
                                   profile_name: str = Form(""),
                                   _: None = Depends(require_csrf)) -> Response:
    """Set/clear the default profile for a specific model."""
    cfg = request.app.state.cfg
    mid = model_id.strip()
    pname = profile_name.strip()
    if not mid:
        return _error_html("model_id required", status_code=400)
    if pname:
        if cfg.get_profile(mid, pname) is None:
            return _error_html(
                f"unknown profile '{pname}' for model '{mid}'",
                status_code=400,
            )
    set_model_default_profile(cfg.config_path, mid, pname)
    _reload_config(request)
    return _models_redirect(request)


@router.post("/models/default-args/save", response_class=HTMLResponse)
async def models_default_args_save(request: Request,
                                   engine: str = Form(...),
                                   args_json: str = Form("{}"),
                                   _: None = Depends(require_csrf)) -> Response:
    """Replace the engine-keyed default-args bucket (the "minimum defaults"
    inherited by every model of that engine)."""
    cfg = request.app.state.cfg
    eng = engine.strip().lower()
    if eng not in ("llama", "mlx"):
        return _error_html(f"unknown engine: {eng}", status_code=400)
    try:
        args = json.loads(args_json.strip() or "{}")
    except json.JSONDecodeError as e:
        return _error_html(f"invalid JSON in args: {e}", status_code=400)
    if not isinstance(args, dict):
        return _error_html("args must be a JSON object", status_code=400)
    set_default_args(cfg.config_path, eng, args)
    _reload_config(request)
    return _models_redirect(request)


@router.post("/models/set-default", response_class=HTMLResponse)
async def models_set_default(request: Request,
                             model_id: str = Form(""),
                             _: None = Depends(require_csrf)) -> Response:
    """Set the configured default model. Empty model_id clears it."""
    cfg = request.app.state.cfg
    update_defaults(cfg.config_path, default_model=model_id.strip())
    _reload_config(request)
    return _models_redirect(request)


# ---- Top-bar (sticky model selector) routes ----

def _topbar_redirect(request: Request) -> RedirectResponse:
    """Send the operator back to the page they triggered the top bar from
    so the selector doesn't yank them to the dashboard on every action.

    We read HTMX's ``HX-Current-URL`` header first: the app sets
    ``Referrer-Policy: no-referrer``, so the browser never sends a
    ``Referer`` and the old referer-based logic *always* fell through to
    ``/ui/`` — every topbar load/unload/star silently bounced the user to
    the dashboard. ``HX-Current-URL`` is sent by htmx on every boosted
    request and is immune to the referrer policy. We only honour the
    same-origin path (never an absolute cross-origin URL)."""
    target = "/ui/"
    current = (request.headers.get("HX-Current-URL")
               or request.headers.get("referer") or "")
    if current:
        try:
            parsed = urlparse(current)
        except Exception:
            parsed = None
        if parsed is not None and (not parsed.netloc
                                   or parsed.netloc == request.url.netloc):
            path = parsed.path or "/ui/"
            if path.startswith("/ui"):
                # Preserve query string (e.g. ?model=… on chat) so the
                # operator lands exactly where they were.
                target = path + (("?" + parsed.query) if parsed.query else "")
    return RedirectResponse(target, status_code=303)


@router.post("/topbar/llm/load", response_class=HTMLResponse)
async def topbar_llm_load(request: Request,
                          model_id: str = Form(...),
                          profile: str = Form(""),
                          _: None = Depends(require_csrf)) -> Response:
    """Load (or swap to) the requested LLM model + optional profile."""
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    try:
        spec = resolve_spec(cfg, model=model_id.strip(),
                            profile=profile.strip() or None)
        if sm.is_running:
            await sm.swap(spec)
        else:
            await sm.start(spec)
    except (ServerError, ValueError) as e:
        return _error_html(f"load failed: {e}", status_code=400)
    return _topbar_redirect(request)


@router.post("/topbar/llm/unload", response_class=HTMLResponse)
async def topbar_llm_unload(request: Request,
                            _: None = Depends(require_csrf)) -> Response:
    """Stop the currently loaded LLM (if any)."""
    sm: ServerManager = request.app.state.sm
    if sm.is_running:
        try:
            await sm.stop()
        except ServerError as e:
            return _error_html(f"unload failed: {e}", status_code=400)
    return _topbar_redirect(request)


@router.post("/topbar/llm/set-default", response_class=HTMLResponse)
async def topbar_llm_set_default(request: Request,
                                 model_id: str = Form(""),
                                 _: None = Depends(require_csrf)) -> Response:
    """Persist ``default_model`` so the LLM is auto-loaded on startup
    when ``autolaunch`` is enabled, and used by requests that omit
    ``model``."""
    cfg = request.app.state.cfg
    update_defaults(cfg.config_path, default_model=model_id.strip())
    _reload_config(request)
    return _topbar_redirect(request)


@router.post("/topbar/image/set-default", response_class=HTMLResponse)
async def topbar_image_set_default(request: Request,
                                   model_id: str = Form(""),
                                   profile: str = Form(""),
                                   _: None = Depends(require_csrf)) -> Response:
    """Persist the default image model + profile. /v1/images/generations
    uses these when the request omits ``model``."""
    cfg = request.app.state.cfg
    update_defaults(
        cfg.config_path,
        default_image_model=model_id.strip(),
        default_image_profile=profile.strip(),
    )
    _reload_config(request)
    return _topbar_redirect(request)


# ---------- Diffusion models page ------------------------------------------
#
# CRUD for per-image-model profiles. Driven by each engine's
# ``profile_schema()`` so adding a new engine surfaces here with no
# template changes. The catalog (``diffusion_catalog.CATALOG``) drives the
# "not installed yet" rows that link to the Diffusion engines page.

def _serialize_profile_field(field) -> dict[str, Any]:
    """Marshal an engines._base.ProfileField for template iteration."""
    return {
        "key": field.key,
        "label": field.label,
        "kind": field.kind,
        "default": field.default,
        "options": list(field.options or []),
        "help": field.help,
    }


def _profile_field_value(prof: Profile | None, key: str) -> str:
    """Read a Profile attribute by key for form pre-fill. Empty string for
    None / missing attrs so HTML inputs render cleanly."""
    if prof is None:
        return ""
    v = getattr(prof, key, None)
    if v is None or v == "":
        return ""
    return str(v)


def _build_image_profile_from_form(name: str, engine_module,
                                   form: dict[str, str]) -> Profile:
    """Walk the engine's ``profile_schema()`` and assemble a Profile.

    Field values arrive as ``field_<key>`` form entries. Unset/empty fields
    leave the corresponding Profile attribute at its dataclass default
    (empty string / None), which the engine's adapter then resolves at
    request time against any built-in baseline.
    """
    schema = engine_module.profile_schema()
    kwargs: dict[str, Any] = {"name": name}
    for f in schema:
        raw = (form.get(f"field_{f.key}", "") or "").strip()
        if not raw:
            continue
        try:
            if f.kind == "int":
                kwargs[f.key] = int(raw)
            elif f.kind == "float":
                kwargs[f.key] = float(raw)
            elif f.kind == "select":
                if f.options and raw not in f.options:
                    raise ValueError(
                        f"invalid value for {f.label}: {raw!r}; "
                        f"must be one of {f.options}"
                    )
                kwargs[f.key] = raw
            else:  # 'text'
                kwargs[f.key] = raw
        except ValueError as e:
            # Re-raise with field context so the error helper surfaces
            # something a human can act on.
            raise ValueError(f"{f.label}: {e}") from None
    return Profile(**kwargs)


def _diffusion_models_ctx(request: Request) -> dict[str, Any]:
    """Build the per-engine catalog + installed + profile context.

    Returns a structure shaped for direct template iteration:

      engines = [
        {
          'id': 'hidream',
          'label': 'HiDream-O1-Image',
          'configured': True/False,  # paths set in [image] config
          'setup_link': '/ui/setup-diffusion',
          'schema': [serialized ProfileField, ...],
          'default_profiles': [{'name': ..., 'fields': {...}}, ...],
          'rows': [
            {
              'catalog': <CatalogEntry or None>,
              'installed': True/False,
              'model_id': 'HiDream-O1-Image',
              'is_active_default': True/False,
              'profiles': [...],
              'default_profile': '',
            },
            ...
          ],
        },
        ...
      ]
    """
    cfg = request.app.state.cfg
    reg: Registry = request.app.state.registry

    # 1) Index on-disk image models by canonical model_id.
    on_disk: dict[str, dict[str, Any]] = {}
    for entry in reg.list():
        engine = detect_engine_for_id(entry.model_id, cfg.models_dir)
        if ENGINE_FAMILY.get(engine, "text") != "image":
            continue
        on_disk[entry.model_id] = {
            "model_id": entry.model_id,
            "engine": engine,
            "size_bytes": entry.size,
            "path": str(entry.path),
        }

    # 2) For each image engine in declaration order, build:
    #    - catalog rows (from diffusion_catalog) joined with on-disk state
    #    - any *extra* on-disk models the catalog doesn't know about
    engines_view: list[dict[str, Any]] = []
    for eng_id, eng_mod in image_engines.ADAPTERS.items():
        # Configured = engine has the path it needs set in [image] config.
        configured = {
            "hidream": bool(cfg.hidream_python and cfg.hidream_repo),
            "z_image": bool(cfg.z_image_python),
            "krea":    bool(cfg.z_image_python),
            "ideogram4": bool(cfg.ideogram4_python),
            "flux2":   bool(cfg.flux2_sd_cli),
        }.get(eng_id, False)

        # Pretty label for the engine section heading.
        label = {
            "hidream": "HiDream-O1-Image",
            "z_image": "Z-Image (Tongyi-MAI / Z-Anime)",
            "krea":    "Krea 2 Turbo",
            "ideogram4": "Ideogram 4",
            "flux2":   "FLUX 2 Dev",
        }.get(eng_id, eng_id)

        schema = [_serialize_profile_field(f) for f in eng_mod.profile_schema()]
        try:
            default_profiles_dict = eng_mod.default_profiles()
        except Exception:
            default_profiles_dict = {}
        default_profiles_view = [
            {"name": n, "fields": d}
            for n, d in default_profiles_dict.items()
        ]

        # Helper to build a row for one model_id (catalog entry or on-disk).
        def _row_for(catalog_entry, model_id: str | None) -> dict[str, Any]:
            mid = model_id or (catalog_entry.canonical_id if catalog_entry else "")
            installed = mid in on_disk
            model_cfg = cfg.get_model(mid) if mid else None
            profs: list[dict[str, Any]] = []
            if model_cfg:
                for p in sorted(model_cfg.profiles.values(),
                                key=lambda x: x.name):
                    profs.append({
                        "name": p.name,
                        "fields": {
                            s["key"]: _profile_field_value(p, s["key"])
                            for s in schema
                        },
                    })
            return {
                "catalog": catalog_entry,
                "installed": installed,
                "model_id": mid,
                "size_bytes": on_disk.get(mid, {}).get("size_bytes", 0),
                "is_active_default": (mid == cfg.default_image_model),
                "profiles": profs,
                "default_profile": (model_cfg.default_profile if model_cfg
                                    else ""),
            }

        # Start with catalog rows for this engine.
        rows: list[dict[str, Any]] = []
        catalog_ids: set[str] = set()
        for cat in diffusion_catalog.for_engine(eng_id):
            rows.append(_row_for(cat, cat.canonical_id))
            catalog_ids.add(cat.canonical_id)
        # Append any on-disk models that aren't in the catalog (operator
        # downloaded a non-canonical name, or we don't track this one yet).
        for mid, meta in on_disk.items():
            if meta["engine"] != eng_id or mid in catalog_ids:
                continue
            rows.append(_row_for(None, mid))

        engines_view.append({
            "id": eng_id,
            "label": label,
            "configured": configured,
            "schema": schema,
            "default_profiles": default_profiles_view,
            "rows": rows,
        })

    ctx = _ctx(
        request,
        active="diffusion-models",
        engines=engines_view,
        default_image_model=cfg.default_image_model or "",
        default_image_profile=cfg.default_image_profile or "",
        setup_diffusion_link="/ui/setup-diffusion",
    )
    return ctx


@router.get("/diffusion-models", response_class=HTMLResponse)
async def diffusion_models_view(request: Request,
                                _: Origin = Depends(require_admin_ui)
                                ) -> HTMLResponse:
    """List supported diffusion models, show which are installed, and
    expose CRUD for per-model profiles."""
    return templates.TemplateResponse(
        request, "diffusion_models.html", _diffusion_models_ctx(request),
    )


@router.post("/diffusion-models/activate", response_class=HTMLResponse)
async def diffusion_models_activate(request: Request,
                                    model_id: str = Form(...),
                                    _: None = Depends(require_csrf)) -> Response:
    """Set this image model as the dashboard / API default."""
    cfg = request.app.state.cfg
    mid = model_id.strip()
    if not mid:
        return _error_html("model_id required", status_code=400)
    update_defaults(cfg.config_path, default_image_model=mid)
    _reload_config(request)
    return RedirectResponse("/ui/diffusion-models", status_code=303)


def _engine_module_or_400(engine: str):
    try:
        return image_engines.get(engine)
    except KeyError:
        return None


def _models_page_for(request: Request, model_id: str, engine: str = "") -> str:
    """Return the models page a profile-CRUD redirect should land on.

    Profile endpoints are shared across the image and audio families; this
    sends the operator back to whichever page they came from (ASR models vs
    Diffusion models) based on the model's engine family."""
    eng = engine or detect_engine_for_id(model_id, request.app.state.cfg.models_dir)
    if ENGINE_FAMILY.get(eng, "text") == "audio":
        return "/ui/asr-models"
    return "/ui/diffusion-models"


@router.post("/diffusion-models/profiles/create", response_class=HTMLResponse)
async def diffusion_profile_create(request: Request,
                                   _: None = Depends(require_csrf)) -> Response:
    form = await request.form()
    cfg = request.app.state.cfg
    model_id = (form.get("model_id", "") or "").strip()
    engine = (form.get("engine", "") or "").strip()
    name = (form.get("name", "") or "").strip().lower()
    make_default = (form.get("make_default", "") or "") == "on"
    if not model_id or not engine or not name:
        return _error_html("model_id, engine, and name are required",
                           status_code=400)
    eng_mod = _engine_module_or_400(engine)
    if eng_mod is None:
        return _error_html(f"unknown engine {engine!r}", status_code=400)
    existing = cfg.get_model(model_id)
    if existing and name in existing.profiles:
        return _error_html(
            f"profile {name!r} already exists for model {model_id!r}",
            status_code=409,
        )
    try:
        prof = _build_image_profile_from_form(name, eng_mod, form)
        save_profile(cfg.config_path, model_id, name, prof)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    if make_default:
        set_model_default_profile(cfg.config_path, model_id, name)
    _reload_config(request)
    return RedirectResponse(_models_page_for(request, model_id, engine),
                            status_code=303)


@router.post("/diffusion-models/profiles/{profile_name}/update",
             response_class=HTMLResponse)
async def diffusion_profile_update(request: Request, profile_name: str,
                                   _: None = Depends(require_csrf)) -> Response:
    form = await request.form()
    cfg = request.app.state.cfg
    model_id = (form.get("model_id", "") or "").strip()
    engine = (form.get("engine", "") or "").strip()
    new_name = (form.get("new_name", "") or "").strip().lower()
    if not model_id or not engine:
        return _error_html("model_id and engine are required",
                           status_code=400)
    eng_mod = _engine_module_or_400(engine)
    if eng_mod is None:
        return _error_html(f"unknown engine {engine!r}", status_code=400)
    m = cfg.get_model(model_id)
    if not m or profile_name not in m.profiles:
        return _error_html(
            f"profile {profile_name!r} not found for model {model_id!r}",
            status_code=404,
        )
    target = new_name or profile_name
    try:
        prof = _build_image_profile_from_form(target, eng_mod, form)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    if target != profile_name:
        if target in m.profiles:
            return _error_html(
                f"profile {target!r} already exists for model {model_id!r}",
                status_code=409,
            )
        try:
            rename_profile(cfg.config_path, model_id, profile_name, target)
        except ValueError as e:
            return _error_html(str(e), status_code=400)
    try:
        save_profile(cfg.config_path, model_id, target, prof)
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    _reload_config(request)
    return RedirectResponse(_models_page_for(request, model_id, engine),
                            status_code=303)


@router.post("/diffusion-models/profiles/{profile_name}/delete",
             response_class=HTMLResponse)
async def diffusion_profile_delete(request: Request, profile_name: str,
                                   model_id: str = Form(...),
                                   _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    mid = model_id.strip()
    delete_profile(cfg.config_path, mid, profile_name)
    _reload_config(request)
    return RedirectResponse(_models_page_for(request, mid), status_code=303)


@router.post("/diffusion-models/profiles/{profile_name}/clone",
             response_class=HTMLResponse)
async def diffusion_profile_clone(request: Request, profile_name: str,
                                  model_id: str = Form(...),
                                  new_name: str = Form(...),
                                  _: None = Depends(require_csrf)) -> Response:
    """Duplicate a profile under a new name. Field values copy verbatim;
    the per-model default pointer is not touched."""
    cfg = request.app.state.cfg
    mid = model_id.strip()
    new = new_name.strip().lower()
    if not new:
        return _error_html("new_name required", status_code=400)
    m = cfg.get_model(mid)
    if not m or profile_name not in m.profiles:
        return _error_html(
            f"profile {profile_name!r} not found for model {mid!r}",
            status_code=404,
        )
    if new in m.profiles:
        return _error_html(
            f"profile {new!r} already exists for model {mid!r}",
            status_code=409,
        )
    src = m.profiles[profile_name]
    # Copy every Profile field except ``name``.
    from dataclasses import fields as _fields
    clone_kwargs: dict[str, Any] = {"name": new}
    for f in _fields(src):
        if f.name == "name":
            continue
        clone_kwargs[f.name] = getattr(src, f.name)
    try:
        save_profile(cfg.config_path, mid, new, Profile(**clone_kwargs))
    except ValueError as e:
        return _error_html(str(e), status_code=400)
    _reload_config(request)
    return RedirectResponse(_models_page_for(request, mid), status_code=303)


@router.post("/diffusion-models/profiles/set-model-default",
             response_class=HTMLResponse)
async def diffusion_profiles_set_model_default(
        request: Request,
        model_id: str = Form(...),
        profile_name: str = Form(""),
        _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    mid = model_id.strip()
    pname = profile_name.strip()
    if not mid:
        return _error_html("model_id required", status_code=400)
    if pname and cfg.get_profile(mid, pname) is None:
        return _error_html(
            f"unknown profile {pname!r} for model {mid!r}",
            status_code=400,
        )
    set_model_default_profile(cfg.config_path, mid, pname)
    _reload_config(request)
    return RedirectResponse(_models_page_for(request, mid), status_code=303)


@router.post("/diffusion-models/profiles/materialize-defaults",
             response_class=HTMLResponse)
async def diffusion_profiles_materialize_defaults(
        request: Request,
        model_id: str = Form(...),
        engine: str = Form(...),
        _: None = Depends(require_csrf)) -> Response:
    """Persist the engine's built-in default profiles into config.toml
    so the operator can edit them. Skips any whose name already exists."""
    cfg = request.app.state.cfg
    mid = model_id.strip()
    eng_mod = _engine_module_or_400(engine)
    if eng_mod is None or not mid:
        return _error_html("model_id and a known engine are required",
                           status_code=400)
    try:
        builtins = eng_mod.default_profiles()
    except Exception as e:
        return _error_html(f"engine {engine!r} has no default profiles: {e}",
                           status_code=400)
    existing = cfg.get_model(mid)
    existing_names = set(existing.profiles.keys()) if existing else set()
    materialized: list[str] = []
    for prof_name, field_dict in builtins.items():
        if prof_name in existing_names:
            continue
        try:
            prof = Profile(name=prof_name, **field_dict)
            save_profile(cfg.config_path, mid, prof_name, prof)
            materialized.append(prof_name)
        except (TypeError, ValueError) as e:
            return _error_html(
                f"failed to materialize {prof_name!r}: {e}",
                status_code=400,
            )
    _reload_config(request)
    return RedirectResponse(_models_page_for(request, mid, engine),
                            status_code=303)


# ---- ASR (speech-to-text) models page ---------------------------------

def _asr_models_ctx(request: Request) -> dict[str, Any]:
    """Context for the ASR models page: installed audio models + profiles,
    plus the audio-engine install/setup state.

    Mirrors ``_diffusion_models_ctx`` but filters to the *audio* family and
    has no catalog (audio models are pulled by the operator). Profile CRUD
    reuses the shared /ui/diffusion-models/profiles/* endpoints, which
    redirect back here via ``_models_page_for``."""
    cfg = request.app.state.cfg
    installer = request.app.state.engine_installer
    from .engine_installer import ENGINE_PLANS
    from .audio_runner import scan_asr_models

    # Scan the (possibly dedicated) ASR models directory, not the Registry —
    # ASR models may live in their own folder outside the LLM models_dir.
    on_disk: dict[str, dict[str, Any]] = {}
    for m in scan_asr_models(cfg):
        on_disk[m["model_id"]] = {
            "model_id": m["model_id"], "engine": "asr",
            "size_bytes": m["size_bytes"], "path": m["path"],
        }

    engines_view: list[dict[str, Any]] = []
    for eng_id, eng_mod in image_engines.ADAPTERS.items():
        if ENGINE_FAMILY.get(eng_id, "text") != "audio":
            continue
        configured = bool(cfg.asr_python) if eng_id == "asr" else False
        label = {"asr": "Whisper (transformers)"}.get(eng_id, eng_id)
        schema = [_serialize_profile_field(f) for f in eng_mod.profile_schema()]
        try:
            default_profiles_dict = eng_mod.default_profiles()
        except Exception:
            default_profiles_dict = {}
        default_profiles_view = [
            {"name": n, "fields": d} for n, d in default_profiles_dict.items()
        ]

        rows: list[dict[str, Any]] = []
        for mid, meta in on_disk.items():
            if meta["engine"] != eng_id:
                continue
            model_cfg = cfg.get_model(mid)
            profs: list[dict[str, Any]] = []
            if model_cfg:
                for p in sorted(model_cfg.profiles.values(), key=lambda x: x.name):
                    profs.append({
                        "name": p.name,
                        "fields": {s["key"]: _profile_field_value(p, s["key"])
                                   for s in schema},
                    })
            rows.append({
                "catalog": None, "installed": True, "model_id": mid,
                "size_bytes": meta.get("size_bytes", 0),
                "is_active_default": False,
                "profiles": profs,
                "default_profile": (model_cfg.default_profile if model_cfg else ""),
            })

        # Install state + plan for the engine card.
        active = installer.active_for_engine(eng_id)
        if not active:
            recent = installer.list_for_engine(eng_id, limit=1)
            active = recent[0] if recent else None
        plan = ENGINE_PLANS.get(eng_id)
        engines_view.append({
            "id": eng_id, "label": label, "configured": configured,
            "schema": schema, "default_profiles": default_profiles_view,
            "rows": rows, "python_path": cfg.asr_python if eng_id == "asr" else "",
            "install": active,
            "plan": {"packages": plan.packages, "notes": plan.notes} if plan else None,
        })

    from .config import asr_max_concurrent
    return _ctx(
        request, active="asr-models", engines=engines_view,
        asr_models_dir=str(cfg.asr_models_dir),
        asr_models_dir_is_default=(cfg.asr_models_dir_override is None),
        asr_defaults={
            "vram_budget_gb": cfg.asr_vram_budget_gb,
            "coexist": cfg.asr_coexist,
            "idle_timeout_s": cfg.asr_idle_timeout_s,
            "decode_interval_s": cfg.asr_decode_interval_s,
            "max_concurrent": asr_max_concurrent(cfg),
        },
    )


@router.get("/asr-models", response_class=HTMLResponse)
async def asr_models_view(request: Request,
                          _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    """List installed speech-to-text models, manage profiles, and install /
    point the ASR engine at a Python environment."""
    return templates.TemplateResponse(
        request, "asr_models.html", _asr_models_ctx(request),
    )


@router.post("/asr-models/install-deps", response_class=HTMLResponse)
async def asr_install_deps(request: Request,
                           torch_backend: str = Form("auto"),
                           _: None = Depends(require_csrf)) -> Response:
    """Install (or reuse a diffusion venv for) the ASR engine deps."""
    from .engine_installer import TORCH_BACKENDS
    installer = request.app.state.engine_installer
    options: dict[str, Any] = {}
    tb = (torch_backend or "auto").strip().lower()
    if tb in TORCH_BACKENDS and tb != "auto":
        options["torch_backend"] = tb
    try:
        installer.start("asr", options=options)
    except (ValueError, RuntimeError) as e:
        return _error_html(f"install failed: {e}", status_code=400)
    return RedirectResponse("/ui/asr-models", status_code=303)


@router.post("/asr-models/install-deps/{install_id}/cancel",
             response_class=HTMLResponse)
async def asr_cancel_install(request: Request, install_id: str,
                             _: None = Depends(require_csrf)) -> Response:
    request.app.state.engine_installer.cancel(install_id)
    return RedirectResponse("/ui/asr-models", status_code=303)


@router.post("/setup/audio/asr", response_class=HTMLResponse)
async def setup_asr(request: Request, asr_python: str = Form(""),
                    _: None = Depends(require_csrf)) -> Response:
    """Point the ASR engine at an existing Python interpreter (one with
    torch + transformers, e.g. the z_image venv)."""
    cfg = request.app.state.cfg
    cfg.asr_python = asr_python.strip()
    update_image_config(cfg.config_path, asr_python=cfg.asr_python)
    return RedirectResponse("/ui/asr-models", status_code=303)


@router.post("/setup/audio/asr-defaults", response_class=HTMLResponse)
async def setup_asr_defaults(request: Request,
                            asr_vram_budget_gb: str = Form(""),
                            asr_coexist: str = Form(""),
                            asr_idle_timeout_s: str = Form(""),
                            asr_decode_interval_s: str = Form(""),
                            _: None = Depends(require_csrf)) -> Response:
    """Set the ASR GPU budget + coexistence + idle/decode defaults."""
    from .config import update_image_config
    cfg = request.app.state.cfg
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    def _i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    budget = _f(asr_vram_budget_gb)
    idle = _i(asr_idle_timeout_s)
    interval = _f(asr_decode_interval_s)
    coexist = (asr_coexist or "") == "on"
    update_image_config(cfg.config_path, asr_vram_budget_gb=budget,
                        asr_coexist=coexist, asr_idle_timeout_s=idle,
                        asr_decode_interval_s=interval)
    # Mutate the shared cfg in place so the queue + worker manager pick it up.
    if budget is not None:
        cfg.asr_vram_budget_gb = budget
    cfg.asr_coexist = coexist
    if idle is not None:
        cfg.asr_idle_timeout_s = idle
    if interval is not None:
        cfg.asr_decode_interval_s = interval
    return RedirectResponse("/ui/asr-models", status_code=303)


@router.post("/setup/audio/asr-models-dir", response_class=HTMLResponse)
async def setup_asr_models_dir(request: Request, asr_models_dir: str = Form(""),
                              _: None = Depends(require_csrf)) -> Response:
    """Set (or clear, when blank) the dedicated folder ASR models are scanned
    from. Blank reverts to the shared LLM models directory."""
    cfg = request.app.state.cfg
    raw = asr_models_dir.strip()
    if raw:
        new_dir = Path(raw).expanduser().resolve()
        new_dir.mkdir(parents=True, exist_ok=True)
        cfg.asr_models_dir_override = new_dir
        update_image_config(cfg.config_path, asr_models_dir=str(new_dir))
    else:
        cfg.asr_models_dir_override = None
        update_image_config(cfg.config_path, asr_models_dir="")
    return RedirectResponse("/ui/asr-models", status_code=303)


# ============================================================ #
# Multi-slot LLM (beta) UI                                     #
# ============================================================ #


def _slots_ui_ctx(request: Request) -> dict[str, Any]:
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    reg: Registry = request.app.state.registry
    text_model_ids: list[str] = []
    for entry in reg.list():
        try:
            engine = detect_engine_for_id(entry.model_id, cfg.models_dir)
        except Exception:
            engine = "llama"
        if ENGINE_FAMILY.get(engine, "text") == "text":
            text_model_ids.append(entry.model_id)
    slot_views: list[dict[str, Any]] = []
    if hasattr(sm, "slots"):
        slot_views = [sv.to_dict() for sv in sm.slots()]
    return _ctx(
        request,
        slots_enabled=bool(getattr(cfg, "multi_slot_enabled", False)),
        slots_max=int(getattr(cfg, "multi_slot_max", 4)),
        slots_base_port=int(getattr(cfg, "multi_slot_base_port", 7201)),
        slots_allow_diffusion=bool(
            getattr(cfg, "allow_diffusion_with_slots", False)),
        slots=slot_views,
        text_model_ids=sorted(text_model_ids),
        exclusive_mode_now=(getattr(cfg, "exclusive_mode", "off") or "off"),
    )


@router.get("/slots", response_class=HTMLResponse)
async def slots_view(request: Request,
                     _: Origin = Depends(require_admin_ui)) -> HTMLResponse:
    return templates.TemplateResponse(request, "slots.html",
                                       _slots_ui_ctx(request))


@router.post("/slots/toggle", response_class=HTMLResponse)
async def slots_toggle(request: Request,
                       enabled: str = Form("off"),
                       _: None = Depends(require_csrf)) -> Response:
    """Master switch. Mirrors POST /admin/slots/enable for the UI cookie path."""
    from .config import (load_config, update_exclusive_mode,
                         update_multi_slot)
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    want_on = (enabled == "on")
    if want_on:
        if (getattr(cfg, "exclusive_mode", "off") or "off") != "off":
            try:
                update_exclusive_mode(cfg.path, mode="off")
                request.app.state.db.log_event(
                    "slots_force_exclusive_off", {})
            except (ValueError, OSError) as e:
                return _error_html(
                    f"could not force exclusive off: {e}", status_code=500)
        try:
            update_multi_slot(cfg.path, enabled=True)
        except (ValueError, OSError) as e:
            return _error_html(str(e), status_code=400)
        new_cfg = load_config(cfg.path)
        new_cfg.bind = cfg.bind
        new_cfg.port = cfg.port
        request.app.state.cfg = new_cfg
        try:
            await sm.boot_slots(supervisor=request.app.state.supervisor)
        except Exception:  # noqa: BLE001
            log.exception("slots: boot after enable failed")
    else:
        if hasattr(sm, "slots"):
            for sv in sm.slots():
                if sv.id == 0:
                    continue
                try:
                    await sm.stop_slot(sv.id)
                except Exception:  # noqa: BLE001
                    log.exception("slots: stop_slot %d failed", sv.id)
        try:
            update_multi_slot(cfg.path, enabled=False)
        except (ValueError, OSError) as e:
            return _error_html(str(e), status_code=400)
        new_cfg = load_config(cfg.path)
        new_cfg.bind = cfg.bind
        new_cfg.port = cfg.port
        request.app.state.cfg = new_cfg
    return RedirectResponse("/ui/slots", status_code=303)


@router.post("/slots/add", response_class=HTMLResponse)
async def slots_ui_add(request: Request,
                       _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    if not getattr(cfg, "multi_slot_enabled", False):
        return _error_html("multi-slot is not enabled", status_code=409)
    try:
        await sm.add_slot(supervisor=request.app.state.supervisor)
    except Exception as e:  # noqa: BLE001
        return _error_html(str(e), status_code=409)
    return RedirectResponse("/ui/slots", status_code=303)


@router.post("/slots/{slot_id}/remove", response_class=HTMLResponse)
async def slots_ui_remove(request: Request, slot_id: int,
                          _: None = Depends(require_csrf)) -> Response:
    sm = request.app.state.sm
    try:
        await sm.remove_slot(slot_id)
    except Exception as e:  # noqa: BLE001
        return _error_html(str(e), status_code=400)
    return RedirectResponse("/ui/slots", status_code=303)


@router.post("/slots/{slot_id}/load", response_class=HTMLResponse)
async def slots_ui_load(request: Request, slot_id: int,
                        model: str = Form(...),
                        profile: str = Form(""),
                        _: None = Depends(require_csrf)) -> Response:
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    try:
        spec = resolve_spec(cfg, model=model.strip(),
                            profile=profile.strip() or None)
    except (ServerError, ValueError) as e:
        return _error_html(str(e), status_code=400)
    slot_sm = sm.slot(slot_id) if hasattr(sm, "slot") else None
    if slot_sm is None:
        return _error_html(f"no such slot {slot_id}", status_code=404)
    try:
        if slot_sm.is_running:
            await sm.swap_in(slot_id, spec)
        else:
            await sm.start_slot(slot_id, spec)
    except ServerError as e:
        return _error_html(str(e), status_code=400)
    return RedirectResponse("/ui/slots", status_code=303)


@router.post("/slots/{slot_id}/unload", response_class=HTMLResponse)
async def slots_ui_unload(request: Request, slot_id: int,
                          _: None = Depends(require_csrf)) -> Response:
    sm = request.app.state.sm
    try:
        await sm.stop_slot(slot_id)
    except Exception as e:  # noqa: BLE001
        return _error_html(str(e), status_code=400)
    return RedirectResponse("/ui/slots", status_code=303)


@router.post("/slots/coex", response_class=HTMLResponse)
async def slots_ui_coex(request: Request,
                        allow: str = Form("off"),
                        _: None = Depends(require_csrf)) -> Response:
    from .config import load_config, update_coexistence_policy
    cfg = request.app.state.cfg
    try:
        update_coexistence_policy(
            cfg.path, allow_diffusion_with_slots=(allow == "on"))
    except (ValueError, OSError) as e:
        return _error_html(str(e), status_code=400)
    new_cfg = load_config(cfg.path)
    new_cfg.bind = cfg.bind
    new_cfg.port = cfg.port
    request.app.state.cfg = new_cfg
    return RedirectResponse("/ui/slots", status_code=303)
