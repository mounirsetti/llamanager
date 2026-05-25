"""Admin endpoints — server lifecycle, queue, models, origins, logs, events."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .auth import AuthManager, Origin
from .events import list_events
from .queue_mgr import QueueManager
from .registry import Registry
from .server_manager import ServerError, ServerManager, resolve_spec

router = APIRouter(prefix="/admin", tags=["admin"])


async def admin_origin(request: Request) -> Origin:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    key = auth.split(" ", 1)[1].strip()
    am: AuthManager = request.app.state.auth
    origin = await am.verify(key)
    if not origin or not origin.is_admin:
        raise HTTPException(status_code=403, detail="admin scope required")
    return origin


# ---------- status ----------

@router.get("/status")
async def status(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    sm: ServerManager = request.app.state.sm
    qm: QueueManager = request.app.state.queue
    base = sm.status()
    snap = qm.snapshot()
    base.update({
        "queue_depth": snap["depth"],
        "in_flight": snap["in_flight"],
        "in_flight_count": len(snap["in_flight"]),
        "paused": snap["paused"],
    })
    return JSONResponse(base)


# ---------- server lifecycle ----------

class StartBody(BaseModel):
    profile: str | None = None
    model: str | None = None
    mmproj: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


@router.post("/server/start")
async def server_start(request: Request, body: StartBody,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    try:
        spec = resolve_spec(cfg, profile=body.profile, model=body.model,
                            mmproj=body.mmproj, args=body.args)
        pid = await sm.start(spec)
    except (ServerError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"pid": pid}, status_code=202)


@router.post("/server/stop")
async def server_stop(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    sm: ServerManager = request.app.state.sm
    await sm.stop()
    return JSONResponse({"ok": True})


@router.post("/server/restart")
async def server_restart(request: Request, body: StartBody | None = None,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    spec = None
    if body and (body.profile or body.model):
        # Profile-only restart falls back to the default model (or the
        # currently-loaded one if no default is configured).
        model_hint = body.model
        if body.profile and not model_hint:
            model_hint = sm.runtime.current_model or cfg.default_model or None
        try:
            spec = resolve_spec(cfg, profile=body.profile, model=model_hint,
                                mmproj=body.mmproj, args=body.args)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        pid = await sm.restart(spec)
    except ServerError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"pid": pid})


@router.post("/server/swap")
async def server_swap(request: Request, body: StartBody,
                      _: Origin = Depends(admin_origin)) -> JSONResponse:
    sm: ServerManager = request.app.state.sm
    cfg = request.app.state.cfg
    if not body.model:
        raise HTTPException(
            status_code=400,
            detail="model required (profile alone is no longer accepted)",
        )
    try:
        spec = resolve_spec(cfg, profile=body.profile, model=body.model,
                            mmproj=body.mmproj, args=body.args)
        pid = await sm.swap(spec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServerError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"pid": pid})


# ---------- queue ----------

@router.get("/queue")
async def queue_list(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    qm: QueueManager = request.app.state.queue
    return JSONResponse(qm.snapshot())


@router.delete("/queue/{request_id}")
async def queue_cancel(request: Request, request_id: str,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    qm: QueueManager = request.app.state.queue
    ok = qm.cancel(request_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"ok": True})


@router.post("/queue/{request_id}/cancel")
async def queue_cancel_in_flight(request: Request, request_id: str,
                                 _: Origin = Depends(admin_origin)) -> JSONResponse:
    qm: QueueManager = request.app.state.queue
    ok = qm.cancel(request_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"ok": True})


@router.post("/queue/pause")
async def queue_pause(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    qm: QueueManager = request.app.state.queue
    qm.pause()
    return JSONResponse({"ok": True})


@router.post("/queue/resume")
async def queue_resume(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    qm: QueueManager = request.app.state.queue
    await qm.resume()
    return JSONResponse({"ok": True})


@router.post("/queue/cancel-all")
async def queue_cancel_all(request: Request, origin: str | None = None,
                           _: Origin = Depends(admin_origin)) -> JSONResponse:
    qm: QueueManager = request.app.state.queue
    am: AuthManager = request.app.state.auth
    if origin:
        target = am.get_origin_by_name(origin)
        if not target:
            raise HTTPException(status_code=404, detail="origin not found")
        n = qm.cancel_all_for_origin(target.id)
    else:
        snap = qm.snapshot()
        n = 0
        for r in snap["pending"]:
            if qm.cancel(r["id"]):
                n += 1
    return JSONResponse({"cancelled": n})


# ---------- models ----------

class PullBody(BaseModel):
    source: str
    files: list[str] | None = None


@router.get("/models")
async def models_list(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    reg: Registry = request.app.state.registry
    return JSONResponse([m.to_dict() for m in reg.list()])


@router.post("/models/pull")
async def models_pull(request: Request, body: PullBody,
                      _: Origin = Depends(admin_origin)) -> JSONResponse:
    reg: Registry = request.app.state.registry
    try:
        did = reg.start_pull(source=body.source, files=body.files)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"download_id": did}, status_code=202)


@router.get("/downloads")
async def downloads_list(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    reg: Registry = request.app.state.registry
    return JSONResponse(reg.list_downloads())


@router.get("/downloads/{download_id}")
async def download_get(request: Request, download_id: str,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    reg: Registry = request.app.state.registry
    d = reg.get_download(download_id)
    if not d:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(d)


@router.delete("/downloads/{download_id}")
async def download_cancel(request: Request, download_id: str,
                          _: Origin = Depends(admin_origin)) -> JSONResponse:
    reg: Registry = request.app.state.registry
    ok = reg.cancel_pull(download_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found or not running")
    return JSONResponse({"ok": True})


@router.delete("/models/{model_id:path}")
async def model_delete(request: Request, model_id: str, force: bool = False,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    reg: Registry = request.app.state.registry
    sm: ServerManager = request.app.state.sm
    loaded = sm.runtime.current_model
    if loaded == model_id and force:
        await sm.stop()
        loaded = None
    ok, err = reg.delete(model_id, currently_loaded=loaded, force=force)
    if not ok:
        if err == "loaded":
            raise HTTPException(
                status_code=409,
                detail="model is currently loaded. Pass ?force=true to stop server first",
            )
        if err == "not_found":
            raise HTTPException(status_code=404, detail="model not found")
        raise HTTPException(status_code=400, detail=err or "failed")
    return JSONResponse({"ok": True})


# ---------- origins ----------

class CreateOriginBody(BaseModel):
    name: str
    priority: int | None = None
    allowed_models: list[str] | None = None
    is_admin: bool = False


class PatchOriginBody(BaseModel):
    priority: int | None = None
    allowed_models: list[str] | None = None
    is_admin: bool | None = None


@router.get("/origins")
async def origins_list(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    am: AuthManager = request.app.state.auth
    return JSONResponse([o.to_public() for o in am.list_origins()])


@router.post("/origins")
async def origins_create(request: Request, body: CreateOriginBody,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    am: AuthManager = request.app.state.auth
    if am.get_origin_by_name(body.name):
        raise HTTPException(status_code=409, detail="origin name already exists")
    origin, key = am.create_origin(
        name=body.name, priority=body.priority,
        allowed_models=body.allowed_models, is_admin=body.is_admin,
    )
    return JSONResponse({"origin": origin.to_public(), "api_key": key},
                        status_code=201)


@router.patch("/origins/{origin_id}")
async def origins_patch(request: Request, origin_id: int, body: PatchOriginBody,
                        _: Origin = Depends(admin_origin)) -> JSONResponse:
    am: AuthManager = request.app.state.auth
    origin = am.update_origin(origin_id, priority=body.priority,
                              allowed_models=body.allowed_models,
                              is_admin=body.is_admin)
    if not origin:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(origin.to_public())


@router.delete("/origins/{origin_id}")
async def origins_delete(request: Request, origin_id: int,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    am: AuthManager = request.app.state.auth
    if not am.delete_origin(origin_id):
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"ok": True})


@router.post("/origins/{origin_id}/rotate-key")
async def origins_rotate(request: Request, origin_id: int,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    am: AuthManager = request.app.state.auth
    key = am.rotate_key(origin_id)
    if not key:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"api_key": key})


# ---------- logs / events ----------

@router.get("/logs")
async def logs(request: Request, tail: int = 200, source: str = "activity",
               _: Origin = Depends(admin_origin)) -> PlainTextResponse:
    cfg = request.app.state.cfg
    if source == "activity":
        from .activity import build_activity, render_activity
        entries = build_activity(
            request.app.state.db, cfg.logs_dir, tail=tail,
        )
        return PlainTextResponse(render_activity(entries))
    logfile: Path
    if source == "llama-server":
        logfile = cfg.logs_dir / "llama-server.log"
    elif source == "llamanager":
        logfile = cfg.logs_dir / "llamanager.log"
    else:
        # Treat ``source`` as an engine name; the corresponding raw
        # log file is logs_dir/<engine>.log. Only accept it if the file
        # actually exists, to avoid arbitrary-path reads from the param.
        from .activity import discover_engine_logs
        candidate = {n: p for n, p in discover_engine_logs(cfg.logs_dir)}
        if source not in candidate:
            raise HTTPException(status_code=400, detail="unknown log source")
        logfile = candidate[source]
    if not logfile.exists():
        return PlainTextResponse("")
    # Tail N lines.
    try:
        with open(logfile, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 8192
            data = b""
            while size > 0 and data.count(b"\n") <= tail:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            lines = data.splitlines()[-tail:]
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return PlainTextResponse(b"\n".join(lines).decode("utf-8", errors="replace"))


@router.get("/events")
async def events(request: Request, limit: int = 200,
                 _: Origin = Depends(admin_origin)) -> JSONResponse:
    return JSONResponse(list_events(request.app.state.db, limit=limit))


# ---------- config ----------

@router.post("/reload")
async def reload_config(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import load_config
    cfg = load_config(request.app.state.cfg.path)
    # Replace in-memory config but keep paths/binding stable.
    old = request.app.state.cfg
    cfg.bind = old.bind
    cfg.port = old.port
    request.app.state.cfg = cfg
    request.app.state.db.log_event("config_reloaded", {})
    return JSONResponse({"ok": True})


# ---------- disk ----------

@router.get("/disk")
async def disk(request: Request, _: Origin = Depends(admin_origin)) -> JSONResponse:
    cfg = request.app.state.cfg
    usage = shutil.disk_usage(cfg.models_dir)
    return JSONResponse({
        "models_dir": str(cfg.models_dir),
        "free_gb": round(usage.free / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "max_disk_gb": cfg.max_disk_gb,
    })


# ---------- self-update ----------
#
# JSON wrappers around the UI's update flow so the CLI can call the
# same code path (factored helpers live in api_ui.py to keep the UI
# button and the CLI behaviour byte-identical).

@router.get("/update/check")
async def update_check(request: Request,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Report whether a newer release exists on GitHub. Same source of
    truth as the Check-for-updates button in /ui/about."""
    from .api_ui import _check_latest_release
    try:
        info = _check_latest_release()
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"update check failed: {e}")
    return JSONResponse(info)


@router.post("/update")
async def update_self(request: Request,
                      _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Pull + reinstall + restart. Same flow as the Update button in
    /ui/about. Blocks until the reinstall finishes; the restart fires
    via a deferred task so this response can complete first."""
    import asyncio as _asyncio
    from .api_ui import _run_self_update, _schedule_self_restart
    project_dir = Path(__file__).parent.parent
    loop = _asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _run_self_update, project_dir)
    if not res["ok"]:
        return JSONResponse(
            {"ok": False, "log": res["log"],
             "error": res.get("error", "update failed")},
            status_code=500,
        )
    sm: ServerManager = request.app.state.sm
    if sm.is_running:
        await sm.stop()
    await _schedule_self_restart()
    return JSONResponse({"ok": True, "log": res["log"],
                         "restarting": True})


# ---------- diffusion ----------
#
# Mirror the UI's diffusion-models page over JSON so the CLI can list
# catalog + installed models, kick installs, and edit profiles without
# scraping HTML. Heavy lifting (catalog join, schema serialisation,
# profile assembly) lives in api_ui helpers — these endpoints are thin
# pass-throughs that return JSON.

@router.get("/diffusion/engines")
async def diffusion_engines(request: Request,
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Per-engine state: configured paths, install status, target GPU,
    plus the most recent install row."""
    from . import engines as _engines
    from .engine_installer import resolve_plan
    from .gpu_detect import detect_gpu, render_group_ok
    cfg = request.app.state.cfg
    installer = request.app.state.engine_installer
    gpu = detect_gpu()
    out: list[dict[str, Any]] = []
    for eng_id in _engines.ADAPTERS.keys():
        configured = {
            "hidream": bool(cfg.hidream_python and cfg.hidream_repo),
            "z_image": bool(cfg.z_image_python),
            "flux2":   bool(cfg.flux2_sd_cli),
        }.get(eng_id, False)
        plan = resolve_plan(eng_id, gpu, emit=None, cfg=cfg)
        active = installer.active_for_engine(eng_id)
        if active:
            last = active
        else:
            recent = installer.list_for_engine(eng_id, limit=1)
            last = recent[0] if recent else None
        out.append({
            "engine": eng_id,
            "configured": configured,
            "target": plan.target if plan else None,
            "packages": (plan.packages if plan else []),
            "wheel_urls": (plan.wheel_urls if plan else []),
            "supports_flash_attn_patch": (
                plan.supports_flash_attn_patch if plan else False),
            "last_install": last,
        })
    return JSONResponse({
        "engines": out,
        "gpu": {
            "kind": gpu.kind,
            "rocm_arch": gpu.rocm_arch,
            "needs_render_group": gpu.needs_render_group,
            "render_group_ok": render_group_ok(gpu),
        },
    })


class DiffusionInstallBody(BaseModel):
    patch_flash_attn: bool = False


@router.post("/diffusion/engines/{engine}/install")
async def diffusion_install(request: Request, engine: str,
                            body: DiffusionInstallBody = DiffusionInstallBody(),
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    installer = request.app.state.engine_installer
    options: dict[str, Any] = {}
    if body.patch_flash_attn:
        options["patch_flash_attn"] = True
    try:
        install_id = installer.start(engine, options=options)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"id": install_id})


@router.post("/diffusion/engines/{engine}/cancel-install")
async def diffusion_cancel_install(request: Request, engine: str,
                                   _: Origin = Depends(admin_origin)) -> JSONResponse:
    installer = request.app.state.engine_installer
    active = installer.active_for_engine(engine)
    if not active:
        return JSONResponse({"ok": True, "cancelled": False})
    installer.cancel(active["id"])
    return JSONResponse({"ok": True, "cancelled": True})


@router.get("/diffusion/models")
async def diffusion_models(request: Request,
                           _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Installed image-family models + the catalog of installable ones."""
    from .config import ENGINE_FAMILY, detect_engine_for_id
    from . import diffusion_catalog
    cfg = request.app.state.cfg
    reg: Registry = request.app.state.registry
    on_disk: list[dict[str, Any]] = []
    for entry in reg.list():
        engine = detect_engine_for_id(entry.model_id, cfg.models_dir)
        if ENGINE_FAMILY.get(engine, "text") != "image":
            continue
        on_disk.append({
            "model_id": entry.model_id, "engine": engine,
            "size_bytes": entry.size, "path": str(entry.path),
            "is_default": entry.model_id == cfg.default_image_model,
        })
    catalog = [
        {"canonical_id": e.canonical_id, "engine": e.engine,
         "label": e.label, "hf_repo": e.hf_repo, "subfolder": e.subfolder,
         "approx_size_gb": e.approx_size_gb,
         "description": e.description, "homepage": e.homepage}
        for e in diffusion_catalog.CATALOG
    ]
    return JSONResponse({"installed": on_disk, "catalog": catalog,
                         "default_image_model": cfg.default_image_model or ""})


class DiffusionActivateBody(BaseModel):
    model_id: str


@router.post("/diffusion/models/activate")
async def diffusion_activate(request: Request,
                             body: DiffusionActivateBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import update_defaults, load_config
    cfg = request.app.state.cfg
    update_defaults(cfg.config_path, default_image_model=body.model_id.strip())
    new = load_config(cfg.path)
    new.bind = cfg.bind; new.port = cfg.port
    request.app.state.cfg = new
    return JSONResponse({"ok": True, "default_image_model": new.default_image_model})


@router.get("/diffusion/profiles")
async def diffusion_profiles_list(request: Request,
                                  model: str = "",
                                  _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Profiles attached to an image model, plus the engine's built-in
    defaults so callers can see what they'd materialise."""
    from . import engines as _engines
    from .config import detect_engine_for_id
    cfg = request.app.state.cfg
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    m = cfg.get_model(model)
    profiles: list[dict[str, Any]] = []
    if m:
        for p in sorted(m.profiles.values(), key=lambda x: x.name):
            fields = {
                k: getattr(p, k) for k in (
                    "image_model_type", "image_size", "image_steps",
                    "image_guidance", "image_seed", "image_editing_scheduler",
                ) if getattr(p, k) not in ("", None)
            }
            profiles.append({"name": p.name, "fields": fields})
    engine = detect_engine_for_id(model, cfg.models_dir)
    builtins: dict[str, Any] = {}
    try:
        builtins = _engines.get(engine).default_profiles()
    except Exception:
        builtins = {}
    return JSONResponse({
        "model": model, "engine": engine,
        "default_profile": (m.default_profile if m else ""),
        "profiles": profiles,
        "builtin_defaults": builtins,
    })


class DiffusionProfileBody(BaseModel):
    model_id: str
    name: str
    fields: dict[str, Any] = Field(default_factory=dict)
    make_default: bool = False


@router.post("/diffusion/profiles")
async def diffusion_profile_create(request: Request,
                                   body: DiffusionProfileBody,
                                   _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import Profile, save_profile, set_model_default_profile, detect_engine_for_id
    from . import engines as _engines
    cfg = request.app.state.cfg
    if not body.model_id or not body.name:
        raise HTTPException(status_code=400, detail="model_id and name are required")
    existing = cfg.get_model(body.model_id)
    name = body.name.strip().lower()
    if existing and name in existing.profiles:
        raise HTTPException(status_code=409,
                            detail=f"profile {name!r} already exists")
    engine = detect_engine_for_id(body.model_id, cfg.models_dir)
    try:
        adapter = _engines.get(engine)
    except KeyError:
        raise HTTPException(status_code=400,
                            detail=f"unknown engine for model {body.model_id!r}")
    kwargs: dict[str, Any] = {"name": name}
    schema_keys = {f.key for f in adapter.profile_schema()}
    for k, v in body.fields.items():
        if k not in schema_keys:
            continue
        if v in ("", None):
            continue
        kwargs[k] = v
    try:
        save_profile(cfg.config_path, body.model_id, name, Profile(**kwargs))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    if body.make_default:
        set_model_default_profile(cfg.config_path, body.model_id, name)
    return JSONResponse({"ok": True, "name": name})


class DiffusionProfileUpdateBody(BaseModel):
    model_id: str
    fields: dict[str, Any] = Field(default_factory=dict)
    new_name: str | None = None


@router.patch("/diffusion/profiles/{name}")
async def diffusion_profile_update(request: Request, name: str,
                                   body: DiffusionProfileUpdateBody,
                                   _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import (Profile, save_profile, rename_profile,
                         detect_engine_for_id)
    from . import engines as _engines
    cfg = request.app.state.cfg
    m = cfg.get_model(body.model_id)
    if not m or name not in m.profiles:
        raise HTTPException(status_code=404,
                            detail=f"profile {name!r} not found")
    target = (body.new_name or name).strip().lower()
    engine = detect_engine_for_id(body.model_id, cfg.models_dir)
    try:
        adapter = _engines.get(engine)
    except KeyError:
        raise HTTPException(status_code=400,
                            detail=f"unknown engine for model {body.model_id!r}")
    if target != name:
        if target in m.profiles:
            raise HTTPException(status_code=409,
                                detail=f"profile {target!r} already exists")
        try:
            rename_profile(cfg.config_path, body.model_id, name, target)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    kwargs: dict[str, Any] = {"name": target}
    schema_keys = {f.key for f in adapter.profile_schema()}
    for k, v in body.fields.items():
        if k not in schema_keys:
            continue
        if v in ("", None):
            continue
        kwargs[k] = v
    try:
        save_profile(cfg.config_path, body.model_id, target, Profile(**kwargs))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "name": target})


@router.delete("/diffusion/profiles/{name}")
async def diffusion_profile_delete(request: Request, name: str,
                                   model_id: str = "",
                                   _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import delete_profile
    cfg = request.app.state.cfg
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    delete_profile(cfg.config_path, model_id, name)
    return JSONResponse({"ok": True})


class DiffusionProfileCloneBody(BaseModel):
    model_id: str
    new_name: str


@router.post("/diffusion/profiles/{name}/clone")
async def diffusion_profile_clone(request: Request, name: str,
                                  body: DiffusionProfileCloneBody,
                                  _: Origin = Depends(admin_origin)) -> JSONResponse:
    from dataclasses import fields as _fields
    from .config import Profile, save_profile
    cfg = request.app.state.cfg
    new = body.new_name.strip().lower()
    if not new:
        raise HTTPException(status_code=400, detail="new_name required")
    m = cfg.get_model(body.model_id)
    if not m or name not in m.profiles:
        raise HTTPException(status_code=404,
                            detail=f"profile {name!r} not found")
    if new in m.profiles:
        raise HTTPException(status_code=409,
                            detail=f"profile {new!r} already exists")
    src = m.profiles[name]
    kwargs: dict[str, Any] = {"name": new}
    for f in _fields(src):
        if f.name == "name":
            continue
        kwargs[f.name] = getattr(src, f.name)
    try:
        save_profile(cfg.config_path, body.model_id, new, Profile(**kwargs))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "name": new})


class DiffusionSetDefaultBody(BaseModel):
    model_id: str
    profile_name: str = ""


@router.post("/diffusion/profiles/set-model-default")
async def diffusion_set_model_default(request: Request,
                                      body: DiffusionSetDefaultBody,
                                      _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import set_model_default_profile
    cfg = request.app.state.cfg
    if not body.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    if body.profile_name and cfg.get_profile(body.model_id, body.profile_name) is None:
        raise HTTPException(status_code=400, detail="unknown profile")
    set_model_default_profile(cfg.config_path, body.model_id, body.profile_name)
    return JSONResponse({"ok": True})


class DiffusionMaterializeBody(BaseModel):
    model_id: str
    engine: str


@router.post("/diffusion/profiles/materialize-defaults")
async def diffusion_materialize_defaults(request: Request,
                                         body: DiffusionMaterializeBody,
                                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import Profile, save_profile
    from . import engines as _engines
    cfg = request.app.state.cfg
    try:
        adapter = _engines.get(body.engine)
    except KeyError:
        raise HTTPException(status_code=400, detail="unknown engine")
    try:
        builtins = adapter.default_profiles()
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"engine has no default profiles: {e}")
    existing = cfg.get_model(body.model_id)
    existing_names = set(existing.profiles.keys()) if existing else set()
    written: list[str] = []
    for prof_name, fields in builtins.items():
        if prof_name in existing_names:
            continue
        try:
            save_profile(cfg.config_path, body.model_id, prof_name,
                         Profile(name=prof_name, **fields))
            written.append(prof_name)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "materialized": written})
