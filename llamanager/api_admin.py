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
        try:
            spec = resolve_spec(cfg, profile=body.profile, model=body.model,
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
    if not (body.profile or body.model):
        raise HTTPException(status_code=400, detail="profile or model required")
    try:
        spec = resolve_spec(cfg, profile=body.profile, model=body.model,
                            mmproj=body.mmproj, args=body.args)
        pid = await sm.swap(spec)
    except (ServerError, ValueError) as e:
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
async def logs(request: Request, tail: int = 200, source: str = "llama-server",
               _: Origin = Depends(admin_origin)) -> PlainTextResponse:
    cfg = request.app.state.cfg
    logfile: Path
    if source == "llama-server":
        logfile = cfg.logs_dir / "llama-server.log"
    elif source == "llamanager":
        logfile = cfg.logs_dir / "llamanager.log"
    else:
        raise HTTPException(status_code=400, detail="unknown log source")
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
