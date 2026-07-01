"""Admin endpoints — server lifecycle, queue, models, origins, logs, events."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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


# ---------- exclusive mode ----------

class ExclusiveModeBody(BaseModel):
    mode: str | None = None
    grace_seconds: float | None = None
    heartbeat_seconds: int | None = None


@router.get("/exclusive")
async def exclusive_status(request: Request,
                           _: Origin = Depends(admin_origin)) -> JSONResponse:
    from . import exclusive as _exclusive
    cfg = request.app.state.cfg
    last = _exclusive.last_result()
    return JSONResponse({
        "mode": getattr(cfg, "exclusive_mode", "off"),
        "grace_seconds": getattr(cfg, "exclusive_grace_seconds", 5.0),
        "heartbeat_seconds": getattr(cfg, "exclusive_heartbeat_seconds", 120),
        "valid_modes": list(_exclusive.VALID_MODES),
        "last_sweep": last.to_dict() if last else None,
    })


@router.post("/exclusive")
async def exclusive_set(request: Request, body: ExclusiveModeBody,
                        _: Origin = Depends(admin_origin)) -> JSONResponse:
    from . import exclusive as _exclusive
    from .config import (VALID_EXCLUSIVE_MODES, load_config,
                         update_exclusive_mode)
    cfg = request.app.state.cfg
    if body.mode is not None:
        m = body.mode.strip().lower()
        if m not in VALID_EXCLUSIVE_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"invalid mode {body.mode!r}; "
                       f"must be one of {VALID_EXCLUSIVE_MODES}",
            )
    # Pre-check the mutex with multi-slot so we can return a 409
    # (Conflict) instead of a generic 400 — clients can branch on it.
    if (body.mode is not None and body.mode.strip().lower() != "off"
            and bool(getattr(cfg, "multi_slot_enabled", False))):
        raise HTTPException(
            status_code=409,
            detail="cannot enable exclusive mode while multi-slot is on; "
                   "disable multi-slot first.",
        )
    try:
        update_exclusive_mode(
            cfg.path,
            mode=body.mode,
            grace_seconds=body.grace_seconds,
            heartbeat_seconds=body.heartbeat_seconds,
        )
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Hot-reload so the new mode applies immediately.
    new_cfg = load_config(cfg.path)
    new_cfg.bind = cfg.bind
    new_cfg.port = cfg.port
    request.app.state.cfg = new_cfg
    return JSONResponse({
        "mode": new_cfg.exclusive_mode,
        "grace_seconds": new_cfg.exclusive_grace_seconds,
        "heartbeat_seconds": new_cfg.exclusive_heartbeat_seconds,
    })


@router.post("/exclusive/sweep")
async def exclusive_sweep_now(request: Request,
                              _: Origin = Depends(admin_origin)) -> JSONResponse:
    from . import exclusive as _exclusive
    cfg = request.app.state.cfg
    mode = getattr(cfg, "exclusive_mode", "off") or "off"
    if mode == "off":
        # Allow a one-shot scan even when disabled so operators can see
        # what *would* be killed before flipping the switch.
        scan_result = _exclusive.scan_and_record("warn")
        return JSONResponse({
            "ran": False,
            "reason": "exclusive_mode is off — returned a warn-mode scan instead",
            "result": scan_result.to_dict(),
        })
    result = await _exclusive.sweep_and_record(
        mode,
        grace_seconds=float(getattr(cfg, "exclusive_grace_seconds", 5.0)),
    )
    return JSONResponse({"ran": True, "result": result.to_dict()})


# ---------- multi-slot LLM (beta) ----------

class SlotsEnableBody(BaseModel):
    enabled: bool


class SlotLoadBody(BaseModel):
    model: str
    profile: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    force: bool = False


class SlotsCoexBody(BaseModel):
    allow: bool


def _slots_payload(request: Request) -> dict[str, Any]:
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    slot_views: list[dict[str, Any]] = []
    total_loaded_gb = 0.0
    if hasattr(sm, "slots"):
        slot_views = [sv.to_dict() for sv in sm.slots()]
    if hasattr(sm, "total_loaded_size_gb"):
        total_loaded_gb = sm.total_loaded_size_gb()
    return {
        "enabled": bool(getattr(cfg, "multi_slot_enabled", False)),
        "max_slots": int(getattr(cfg, "multi_slot_max", 4)),
        "base_port": int(getattr(cfg, "multi_slot_base_port", 7201)),
        "allow_diffusion_with_slots": bool(
            getattr(cfg, "allow_diffusion_with_slots", False)
        ),
        "total_loaded_size_gb": total_loaded_gb,
        "slots": slot_views,
    }


@router.get("/slots")
async def slots_status(request: Request,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Snapshot of the slot fleet — used by the UI dashboard + CLI."""
    return JSONResponse(_slots_payload(request))


@router.post("/slots/enable")
async def slots_enable(request: Request, body: SlotsEnableBody,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Master switch. Force-disables exclusive_mode on the way ON; on
    the way OFF, drains slots 1..N (up to 30s) then force-stops."""
    from .config import (load_config, update_exclusive_mode,
                         update_multi_slot)
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    if body.enabled:
        # Mutex with exclusive mode: force exclusive=off first so
        # update_multi_slot doesn't refuse.
        if (getattr(cfg, "exclusive_mode", "off") or "off") != "off":
            try:
                update_exclusive_mode(cfg.path, mode="off")
            except (ValueError, OSError) as e:
                raise HTTPException(status_code=500,
                                    detail=f"could not force exclusive off: {e}")
            request.app.state.db.log_event("slots_force_exclusive_off", {})
        try:
            update_multi_slot(cfg.path, enabled=True)
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        new_cfg = load_config(cfg.path)
        new_cfg.bind = cfg.bind
        new_cfg.port = cfg.port
        request.app.state.cfg = new_cfg
        # Restore any slots 1..N that exist in the manifest from a
        # previous enable. Boot is best-effort.
        try:
            await sm.boot_slots(supervisor=request.app.state.supervisor)
        except Exception as e:  # noqa: BLE001
            log.warning("slots: boot after enable failed: %s", e)
    else:
        # Drain + stop slots 1..N. Per user decision: wait up to 30s
        # then force.
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
            raise HTTPException(status_code=400, detail=str(e))
        new_cfg = load_config(cfg.path)
        new_cfg.bind = cfg.bind
        new_cfg.port = cfg.port
        request.app.state.cfg = new_cfg
    return JSONResponse(_slots_payload(request))


@router.post("/slots")
async def slots_add(request: Request,
                    _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Allocate the next free slot id + port."""
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    if not getattr(cfg, "multi_slot_enabled", False):
        raise HTTPException(status_code=409,
                            detail="multi-slot is not enabled")
    try:
        slot_id = await sm.add_slot(supervisor=request.app.state.supervisor)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=409, detail=str(e))
    return JSONResponse({"id": slot_id, **_slots_payload(request)})


@router.delete("/slots/{slot_id}")
async def slots_remove(request: Request, slot_id: int,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Stop + remove a slot. Slot 0 is unremovable."""
    sm = request.app.state.sm
    try:
        result = await sm.remove_slot(slot_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"result": result, **_slots_payload(request)})


@router.post("/slots/{slot_id}/load")
async def slots_load(request: Request, slot_id: int, body: SlotLoadBody,
                     _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Load (or swap) a model into a specific slot.

    v1 has no automatic VRAM admission — the dashboard surfaces total
    loaded file size as a proxy for VRAM pressure (``size_gb`` per
    slot in the response payload). ``body.force`` is accepted for
    forward compatibility and currently ignored.
    """
    cfg = request.app.state.cfg
    sm = request.app.state.sm
    try:
        spec = resolve_spec(cfg, model=body.model,
                            profile=body.profile, args=body.args or {})
    except (ServerError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    slot_sm = sm.slot(slot_id) if hasattr(sm, "slot") else None
    if slot_sm is None:
        raise HTTPException(status_code=404, detail=f"no such slot {slot_id}")
    try:
        if slot_sm.is_running:
            await sm.swap_in(slot_id, spec)
        else:
            await sm.start_slot(slot_id, spec)
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(_slots_payload(request))


@router.post("/slots/{slot_id}/unload")
async def slots_unload(request: Request, slot_id: int,
                       _: Origin = Depends(admin_origin)) -> JSONResponse:
    sm = request.app.state.sm
    try:
        await sm.stop_slot(slot_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(_slots_payload(request))


@router.post("/slots/diffusion-coexistence")
async def slots_diffusion_coex(request: Request, body: SlotsCoexBody,
                               _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Toggle the ``allow_diffusion_with_slots`` policy.

    When on, dispatching an image task does NOT unload the LLM slots;
    operator accepts the VRAM contention risk.
    """
    from .config import load_config, update_coexistence_policy
    cfg = request.app.state.cfg
    try:
        update_coexistence_policy(cfg.path, allow_diffusion_with_slots=body.allow)
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    new_cfg = load_config(cfg.path)
    new_cfg.bind = cfg.bind
    new_cfg.port = cfg.port
    request.app.state.cfg = new_cfg
    return JSONResponse(_slots_payload(request))


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
    """Upgrade llamanager + restart. Same flow as the Update button in
    /ui/about: runs ``pip install --upgrade llamanager`` against the
    daemon's own venv, then SIGTERMs so the supervisor reloads the new
    code. Editable installs are refused with a 409 so the CLI / button
    can render the manual-update instructions instead."""
    import asyncio as _asyncio
    from .api_ui import _run_self_update, _schedule_self_restart
    loop = _asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _run_self_update)
    if not res["ok"]:
        # 409 (Conflict) for editable mode — the install state itself
        # blocks the operation. 500 for everything else (pip failure,
        # network, etc.).
        status = 409 if res.get("mode") == "editable" else 500
        return JSONResponse(
            {"ok": False, "log": res["log"],
             "error": res.get("error", "update failed"),
             "mode": res.get("mode", "unknown")},
            status_code=status,
        )
    sm: ServerManager = request.app.state.sm
    if sm.is_running:
        await sm.stop()
    await _schedule_self_restart()
    return JSONResponse({"ok": True, "log": res["log"],
                         "mode": res.get("mode", "pypi"),
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
    # Optional explicit diffusers version (upgrade/downgrade). Persists as an
    # override so auto-update converges to it instead of the shipped pin.
    diffusers_version: str = ""
    # Clear any existing override and reinstall the shipped DIFFUSERS_PIN.
    reset_diffusers: bool = False


@router.get("/diffusion/engines/{engine}/versions")
async def diffusion_versions(request: Request, engine: str,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    """List installable diffusers versions (newest first) plus the engine's
    installed version and current target (override or shipped pin)."""
    import asyncio as _asyncio
    from .engine_installer import (
        list_diffusers_versions, diffusion_update_info,
        diffusion_target_version, _plan_diffusers_pin,
    )
    cfg = request.app.state.cfg
    loop = _asyncio.get_running_loop()
    listing = await loop.run_in_executor(None, list_diffusers_versions)
    # diffusion_update_info reads the installed version + computes has_update
    # against the target (override or shipped pin) — the diffusion equivalent
    # of "check for updates".
    info = await loop.run_in_executor(None, diffusion_update_info, cfg, engine)
    return JSONResponse({
        "engine": engine,
        "installed": info["installed"],
        "target": info["target"],
        "has_update": info["has_update"],
        "pin": _plan_diffusers_pin(engine),
        **listing,
    })


@router.post("/diffusion/engines/{engine}/install")
async def diffusion_install(request: Request, engine: str,
                            body: DiffusionInstallBody = DiffusionInstallBody(),
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import set_diffusers_override
    cfg = request.app.state.cfg
    installer = request.app.state.engine_installer
    options: dict[str, Any] = {}
    if body.patch_flash_attn:
        options["patch_flash_attn"] = True

    chosen = (body.diffusers_version or "").strip()
    if body.reset_diffusers:
        # Clear the override and reinstall the shipped pin.
        set_diffusers_override(cfg.config_path, engine, None)
        cfg.image_diffusers_version = {
            k: v for k, v in (cfg.image_diffusers_version or {}).items()
            if k != engine
        }
    elif chosen:
        options["diffusers_version"] = chosen
        # Persist as the override so a deliberate pin isn't re-bumped by
        # auto-update.
        set_diffusers_override(cfg.config_path, engine, chosen)
        cfg.image_diffusers_version = dict(cfg.image_diffusers_version or {})
        cfg.image_diffusers_version[engine] = chosen

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


# ---------- LLM profiles ----------
#
# Mirrors `/ui/models/profiles/*` over JSON so the CLI can manage LLM
# profiles (mmproj, ctx_size, vram_limit_gb, ram-spill policy, kv_cache_type,
# thinking, reasoning_budget, parallel, MTP, args) without going through the
# browser. Uses the shared validator in api_ui so UI and CLI reject the same
# junk.

@router.get("/profiles")
async def llm_profiles_list(request: Request,
                            model: str = "",
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    """List LLM profiles attached to a model, plus the model's default."""
    from .config import detect_engine_for_id, ENGINE_FAMILY
    cfg = request.app.state.cfg
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    m = cfg.get_model(model)
    profiles: list[dict[str, Any]] = []
    if m:
        for p in sorted(m.profiles.values(), key=lambda x: x.name):
            profiles.append({
                "name": p.name,
                "mmproj": p.mmproj or "",
                "ctx_size": p.ctx_size,
                "vram_limit_gb": p.vram_limit_gb,
                "ram_spill_policy": p.ram_spill_policy or "default",
                "ram_spill_limit_gb": p.ram_spill_limit_gb,
                "kv_cache_type": p.kv_cache_type or "",
                "flash_attn": p.flash_attn or "",
                "thinking": p.thinking or "",
                "reasoning_budget": p.reasoning_budget,
                "parallel": p.parallel,
                "mtp": p.mtp,
                "mtp_n_max": p.mtp_n_max,
                "args": p.args or {},
            })
    engine = detect_engine_for_id(model, cfg.models_dir)
    return JSONResponse({
        "model": model, "engine": engine,
        "family": ENGINE_FAMILY.get(engine, "text"),
        "default_profile": (m.default_profile if m else ""),
        "profiles": profiles,
    })


class LlmProfileBody(BaseModel):
    """Shape posted by AdminClient / CLI for LLM profile create/update.

    Mirrors the fields the UI form takes but in JSON-native types — no
    ``vram_unlimited`` checkbox (the JSON caller passes ``vram_limit_gb:
    null`` for unlimited)."""
    model_id: str
    name: str
    mmproj: str = ""
    ctx_size: int | None = None
    vram_limit_gb: float | None = None
    ram_spill_policy: str = "default"
    ram_spill_limit_gb: float | None = None
    kv_cache_type: str = ""
    flash_attn: str = ""
    thinking: str = ""
    reasoning_budget: int | None = None
    parallel: int | None = None
    mtp: bool = False
    mtp_n_max: int | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    make_default: bool = False


@router.post("/profiles")
async def llm_profile_create(request: Request,
                             body: LlmProfileBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .api_ui import _build_llm_profile_from_values
    from .config import save_profile, set_model_default_profile
    cfg = request.app.state.cfg
    if not body.model_id or not body.name:
        raise HTTPException(status_code=400,
                            detail="model_id and name are required")
    name = body.name.strip().lower()
    existing = cfg.get_model(body.model_id)
    if existing and name in existing.profiles:
        raise HTTPException(status_code=409,
                            detail=f"profile {name!r} already exists")
    try:
        prof = _build_llm_profile_from_values(
            name, mmproj=body.mmproj, ctx_size=body.ctx_size,
            vram_limit_gb=body.vram_limit_gb,
            ram_spill_policy=body.ram_spill_policy,
            ram_spill_limit_gb=body.ram_spill_limit_gb,
            kv_cache_type=body.kv_cache_type,
            flash_attn=body.flash_attn,
            thinking=body.thinking,
            reasoning_budget=body.reasoning_budget,
            parallel=body.parallel,
            mtp=body.mtp,
            mtp_n_max=body.mtp_n_max,
            args=body.args,
        )
        save_profile(cfg.config_path, body.model_id, name, prof)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    if body.make_default:
        set_model_default_profile(cfg.config_path, body.model_id, name)
    return JSONResponse({"ok": True, "name": name})


class LlmProfileUpdateBody(BaseModel):
    """Update body. Every field is optional — only non-None values are
    persisted to the new profile. ``new_name`` (optional) renames in the
    same call. Pass ``args = {}`` to wipe the args bucket explicitly;
    omit ``args`` (None default) to leave them as they are.
    """
    model_id: str
    mmproj: str | None = None
    ctx_size: int | None = None
    vram_limit_gb: float | None = None
    ram_spill_policy: str | None = None
    ram_spill_limit_gb: float | None = None
    kv_cache_type: str | None = None
    flash_attn: str | None = None
    thinking: str | None = None
    reasoning_budget: int | None = None
    parallel: int | None = None
    mtp: bool | None = None
    mtp_n_max: int | None = None
    args: dict[str, Any] | None = None
    new_name: str | None = None


@router.patch("/profiles/{name}")
async def llm_profile_update(request: Request, name: str,
                             body: LlmProfileUpdateBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .api_ui import _build_llm_profile_from_values
    from .config import save_profile, rename_profile
    cfg = request.app.state.cfg
    m = cfg.get_model(body.model_id)
    if not m or name not in m.profiles:
        raise HTTPException(status_code=404,
                            detail=f"profile {name!r} not found")
    target = (body.new_name or name).strip().lower()
    if target != name:
        if target in m.profiles:
            raise HTTPException(status_code=409,
                                detail=f"profile {target!r} already exists")
        try:
            rename_profile(cfg.config_path, body.model_id, name, target)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    src = m.profiles[name]
    try:
        prof = _build_llm_profile_from_values(
            target,
            mmproj=(body.mmproj if body.mmproj is not None else src.mmproj),
            ctx_size=(body.ctx_size if body.ctx_size is not None
                      else src.ctx_size),
            vram_limit_gb=(body.vram_limit_gb if body.vram_limit_gb is not None
                           else src.vram_limit_gb),
            ram_spill_policy=(body.ram_spill_policy
                              if body.ram_spill_policy is not None
                              else src.ram_spill_policy),
            ram_spill_limit_gb=(body.ram_spill_limit_gb
                                if body.ram_spill_limit_gb is not None
                                else src.ram_spill_limit_gb),
            kv_cache_type=(body.kv_cache_type if body.kv_cache_type is not None
                           else src.kv_cache_type),
            flash_attn=(body.flash_attn if body.flash_attn is not None
                        else src.flash_attn),
            thinking=(body.thinking if body.thinking is not None
                      else src.thinking),
            reasoning_budget=(body.reasoning_budget
                              if body.reasoning_budget is not None
                              else src.reasoning_budget),
            parallel=(body.parallel if body.parallel is not None
                      else src.parallel),
            mtp=(body.mtp if body.mtp is not None else src.mtp),
            mtp_n_max=(body.mtp_n_max if body.mtp_n_max is not None
                       else src.mtp_n_max),
            args=(body.args if body.args is not None else src.args),
        )
        save_profile(cfg.config_path, body.model_id, target, prof)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "name": target})


@router.delete("/profiles/{name}")
async def llm_profile_delete(request: Request, name: str,
                             model_id: str = "",
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import delete_profile
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    delete_profile(request.app.state.cfg.config_path, model_id, name)
    return JSONResponse({"ok": True})


class LlmProfileCloneBody(BaseModel):
    model_id: str
    new_name: str


@router.post("/profiles/{name}/clone")
async def llm_profile_clone(request: Request, name: str,
                            body: LlmProfileCloneBody,
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    from dataclasses import fields as _fields
    from .config import Profile as _Profile, save_profile
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
        save_profile(cfg.config_path, body.model_id, new, _Profile(**kwargs))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "name": new})


class LlmSetModelDefaultBody(BaseModel):
    model_id: str
    profile_name: str = ""


@router.post("/profiles/set-model-default")
async def llm_set_model_default(request: Request,
                                body: LlmSetModelDefaultBody,
                                _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import set_model_default_profile
    cfg = request.app.state.cfg
    if not body.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    if body.profile_name and cfg.get_profile(body.model_id,
                                             body.profile_name) is None:
        raise HTTPException(status_code=400, detail="unknown profile")
    set_model_default_profile(cfg.config_path, body.model_id,
                              body.profile_name)
    return JSONResponse({"ok": True})


# ---------- models housekeeping ----------

class SetDefaultModelBody(BaseModel):
    model_id: str = ""  # empty clears it


@router.post("/models/set-default")
async def set_default_text_model(request: Request,
                                 body: SetDefaultModelBody,
                                 _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Persist the configured default text model (used by /v1/chat/completions
    when the request omits ``model``)."""
    from .config import update_defaults, load_config
    cfg = request.app.state.cfg
    update_defaults(cfg.config_path, default_model=body.model_id.strip())
    new = load_config(cfg.path)
    new.bind = cfg.bind; new.port = cfg.port
    request.app.state.cfg = new
    return JSONResponse({"ok": True, "default_model": new.default_model})


class AddExistingModelBody(BaseModel):
    file_path: str


@router.post("/models/add-existing")
async def add_existing_model(request: Request,
                             body: AddExistingModelBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Register a pre-downloaded GGUF by symlinking (or copying) it into
    the configured ``models_dir``. The registry picks it up on next scan."""
    cfg = request.app.state.cfg
    src = Path(body.file_path.strip()).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=400, detail=f"not a file: {src}")
    if not src.name.lower().endswith(".gguf"):
        raise HTTPException(status_code=400,
                            detail="only .gguf files are supported")
    dest = cfg.models_dir / src.name
    if dest.exists():
        raise HTTPException(status_code=409,
                            detail=f"a model named {src.name} already exists")
    try:
        dest.symlink_to(src)
    except OSError:
        import shutil as _shutil
        _shutil.copy2(src, dest)
    return JSONResponse({"ok": True, "added": str(dest)})


class SetModelsDirBody(BaseModel):
    models_dir: str


@router.post("/models/set-dir")
async def set_models_dir(request: Request,
                         body: SetModelsDirBody,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Change ``models_dir`` at runtime + persist it to config.toml."""
    import re as _re
    import json as _json
    cfg = request.app.state.cfg
    new_dir = Path(body.models_dir.strip()).expanduser().resolve()
    new_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir_override = new_dir
    request.app.state.registry.models_dir = new_dir
    config_path = cfg.config_path
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    new_line = f'models_dir = {_json.dumps(str(new_dir))}'
    text, n = _re.subn(r'^models_dir\s*=\s*.*$', new_line, text,
                       flags=_re.MULTILINE)
    if n == 0:
        server_match = _re.search(r'^\[server\]', text, flags=_re.MULTILINE)
        if server_match:
            text = (text[:server_match.end()] + f"\n{new_line}"
                    + text[server_match.end():])
        else:
            text = text.rstrip("\n") + f"\n\n[server]\n{new_line}\n"
    config_path.write_text(text, encoding="utf-8")
    return JSONResponse({"ok": True, "models_dir": str(new_dir)})


# ---------- setup / config ----------

class SetupBinaryBody(BaseModel):
    binary_path: str


@router.post("/setup/llama-binary")
async def setup_llama_binary(request: Request,
                             body: SetupBinaryBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Set the llama-server binary path the daemon uses."""
    from .llama_installer import patch_config_binary
    cfg = request.app.state.cfg
    cfg.llama_server_binary = body.binary_path.strip()
    patch_config_binary(cfg.config_path, cfg.llama_server_binary)
    return JSONResponse({"ok": True,
                         "llama_server_binary": cfg.llama_server_binary})


class SetupHidreamBody(BaseModel):
    hidream_python: str | None = None
    hidream_repo: str | None = None


@router.post("/setup/hidream")
async def setup_hidream_paths(request: Request,
                              body: SetupHidreamBody,
                              _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Set hidream_python / hidream_repo. Omitted fields are left alone."""
    from .config import update_image_config
    cfg = request.app.state.cfg
    if body.hidream_python is not None:
        cfg.hidream_python = body.hidream_python.strip()
    if body.hidream_repo is not None:
        cfg.hidream_repo = body.hidream_repo.strip()
    update_image_config(cfg.config_path,
                        hidream_python=cfg.hidream_python,
                        hidream_repo=cfg.hidream_repo)
    return JSONResponse({"ok": True,
                         "hidream_python": cfg.hidream_python,
                         "hidream_repo": cfg.hidream_repo})


class SetupZImageBody(BaseModel):
    z_image_python: str


@router.post("/setup/z-image")
async def setup_z_image_paths(request: Request,
                              body: SetupZImageBody,
                              _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import update_image_config
    cfg = request.app.state.cfg
    cfg.z_image_python = body.z_image_python.strip()
    update_image_config(cfg.config_path, z_image_python=cfg.z_image_python)
    return JSONResponse({"ok": True, "z_image_python": cfg.z_image_python})


class SetupFlux2Body(BaseModel):
    flux2_sd_cli: str | None = None
    flux2_device_index: int | None = None
    clear_device_index: bool = False


@router.post("/setup/flux2")
async def setup_flux2_paths(request: Request,
                            body: SetupFlux2Body,
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import update_image_config
    cfg = request.app.state.cfg
    if body.flux2_sd_cli is not None:
        cfg.flux2_sd_cli = body.flux2_sd_cli.strip()
    if body.clear_device_index:
        cfg.flux2_device_index = None
    elif body.flux2_device_index is not None:
        cfg.flux2_device_index = body.flux2_device_index
    update_image_config(
        cfg.config_path,
        flux2_sd_cli=cfg.flux2_sd_cli if body.flux2_sd_cli is not None else None,
        flux2_device_index=cfg.flux2_device_index,
        clear_flux2_device_index=body.clear_device_index,
    )
    return JSONResponse({
        "ok": True,
        "flux2_sd_cli": cfg.flux2_sd_cli,
        "flux2_device_index": cfg.flux2_device_index,
    })


class CoexistenceBody(BaseModel):
    unload_text_on_arrival: bool | None = None
    restart_text_after_image: bool | None = None
    allow_concurrent: bool | None = None


@router.post("/setup/coexistence")
async def setup_coexistence_admin(request: Request,
                                  body: CoexistenceBody,
                                  _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import update_coexistence_policy
    cfg = request.app.state.cfg
    if body.unload_text_on_arrival is not None:
        cfg.unload_text_on_arrival = body.unload_text_on_arrival
    if body.restart_text_after_image is not None:
        cfg.restart_text_after_image = body.restart_text_after_image
    if body.allow_concurrent is not None:
        cfg.allow_concurrent = body.allow_concurrent
    update_coexistence_policy(
        cfg.config_path,
        unload_text_on_arrival=cfg.unload_text_on_arrival,
        restart_text_after_image=cfg.restart_text_after_image,
        allow_concurrent=cfg.allow_concurrent,
    )
    return JSONResponse({
        "ok": True,
        "unload_text_on_arrival": cfg.unload_text_on_arrival,
        "restart_text_after_image": cfg.restart_text_after_image,
        "allow_concurrent": cfg.allow_concurrent,
    })


class DefaultArgsBody(BaseModel):
    engine: str
    args: dict[str, Any] = Field(default_factory=dict)


@router.post("/setup/default-args")
async def setup_default_args(request: Request,
                             body: DefaultArgsBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Replace the engine-keyed default-args bucket (``[default_args.<engine>]``
    in config.toml). Affects llama / mlx."""
    from .config import set_default_args, load_config
    cfg = request.app.state.cfg
    eng = body.engine.strip().lower()
    if eng not in ("llama", "mlx"):
        raise HTTPException(status_code=400,
                            detail=f"unknown engine: {eng}")
    set_default_args(cfg.config_path, eng, body.args)
    new = load_config(cfg.path)
    new.bind = cfg.bind; new.port = cfg.port
    request.app.state.cfg = new
    return JSONResponse({"ok": True, "engine": eng, "args": body.args})


class ToggleBody(BaseModel):
    enabled: bool


@router.post("/setup/autolaunch")
async def setup_autolaunch(request: Request,
                           body: ToggleBody,
                           _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Toggle whether the daemon auto-starts the configured default LLM
    on its own startup. Persisted to [defaults].autolaunch."""
    from .config import update_defaults
    cfg = request.app.state.cfg
    cfg.autolaunch = body.enabled
    update_defaults(cfg.config_path, autolaunch=body.enabled)
    return JSONResponse({"ok": True, "autolaunch": cfg.autolaunch})


@router.post("/setup/autorestart")
async def setup_autorestart(request: Request,
                            body: ToggleBody,
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Toggle the supervisor's crash auto-restart. In-memory only — the
    setting resets when the daemon restarts (matches the UI behaviour)."""
    request.app.state.supervisor.enabled = body.enabled
    return JSONResponse({"ok": True,
                         "autorestart": request.app.state.supervisor.enabled})


# ---- llama-server installer (text-side engine) ----

class InstallLlamaBody(BaseModel):
    source: str = "llama.cpp"
    backend: str = ""
    # Optional explicit upstream version (a GitHub release tag for llama
    # sources, a mlx-lm PyPI version for MLX). Empty = install the latest.
    version: str = ""


@router.post("/setup/install-llama-server")
async def install_llama_server(request: Request,
                               body: InstallLlamaBody,
                               _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Kick off the llama-server variant install in the background.

    Returns the variant id immediately; poll
    ``GET /admin/setup/install-llama-server/status?variant=<id>`` to
    watch progress. ``version`` pins a specific build (upgrade or downgrade);
    omit it to install the latest."""
    import asyncio as _asyncio
    from .llama_installer import (
        InstallState, install_variant, variant_id,
    )
    from .api_ui import _resolve_variant
    source, backend = _resolve_variant(body.source, body.backend)
    vid = variant_id(source, backend)
    version = (body.version or "").strip() or None
    states: dict[str, InstallState] = request.app.state.install_states
    state = states.setdefault(vid, InstallState())
    if state.status == "running":
        return JSONResponse({"id": vid, "status": "running",
                             "note": "install already in progress"})
    state.status = "running"
    state.lines = []
    state.error = None
    state.installed_path = None
    _asyncio.create_task(install_variant(state, source, backend, version=version))
    return JSONResponse({"id": vid, "status": "running",
                         "source": source, "backend": backend,
                         "version": version})


@router.get("/setup/engine-versions")
async def setup_engine_versions(request: Request,
                                variant: str,
                                _: Origin = Depends(admin_origin)) -> JSONResponse:
    """List installable upstream versions for an LLM variant (newest first)."""
    import asyncio as _asyncio
    from .llama_installer import list_versions, parse_variant_id
    parsed = parse_variant_id(variant)
    if parsed is None:
        raise HTTPException(status_code=400,
                            detail=f"invalid variant id: {variant!r}")
    source, backend = parsed
    loop = _asyncio.get_running_loop()
    result = await loop.run_in_executor(None, list_versions, source, backend)
    return JSONResponse({"variant": variant, **result})


@router.get("/setup/check-updates")
async def setup_check_updates(request: Request,
                              variant: str = "",
                              _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Check upstream for a newer build of one variant (``variant=<id>``) or
    every installed variant (omit ``variant``). JSON mirror of the UI's
    *Check for updates* button — returns ``{variant_id: {installed, latest,
    has_update, error}}``."""
    import asyncio as _asyncio
    from .llama_installer import (
        check_for_update, detect_variant_binary, list_variants, parse_variant_id,
    )
    loop = _asyncio.get_running_loop()
    updates: dict[str, Any] = {}
    if variant:
        parsed = parse_variant_id(variant)
        if parsed is None:
            raise HTTPException(status_code=400,
                                detail=f"invalid variant id: {variant!r}")
        info = await loop.run_in_executor(None, check_for_update, *parsed)
        updates[variant] = info.to_dict()
    else:
        for v in list_variants():
            if detect_variant_binary(v["source"], v["backend"]) is None:
                continue
            info = await loop.run_in_executor(
                None, check_for_update, v["source"], v["backend"])
            updates[v["id"]] = info.to_dict()
    # Cache so the UI's variant cards reflect the same result.
    request.app.state.install_updates = updates
    return JSONResponse({"updates": updates})


@router.get("/setup/install-llama-server/status")
async def install_llama_server_status(
        request: Request,
        variant: str,
        _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .llama_installer import InstallState
    states: dict[str, InstallState] = request.app.state.install_states
    state = states.get(variant)
    if state is None:
        raise HTTPException(status_code=404,
                            detail=f"no install for {variant!r}")
    return JSONResponse({"id": variant, **state.to_dict()})


class SwitchVariantBody(BaseModel):
    variant: str


@router.post("/setup/switch-variant")
async def switch_variant(request: Request,
                         body: SwitchVariantBody,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Switch the active text engine to a previously-installed variant.

    Equivalent of clicking a different variant card on /ui/setup."""
    from .llama_installer import (
        engine_type_for, parse_variant_id, patch_config_binary,
        variant_install_path,
    )
    cfg = request.app.state.cfg
    parsed = parse_variant_id(body.variant)
    if parsed is None:
        raise HTTPException(status_code=400,
                            detail=f"invalid variant id: {body.variant!r}")
    source, backend = parsed
    path = variant_install_path(source, backend)
    if not path.exists():
        raise HTTPException(status_code=400,
                            detail=f"variant not installed at {path}")
    engine = engine_type_for(source)
    cfg.llama_server_binary = str(path)
    cfg.llama_server_engine = engine
    patch_config_binary(cfg.config_path, str(path), engine=engine)
    return JSONResponse({
        "ok": True,
        "llama_server_binary": cfg.llama_server_binary,
        "llama_server_engine": cfg.llama_server_engine,
    })


# ---- auto-update-when-idle ----

def _valid_auto_update_key(key: str) -> bool:
    """Accept a llama variant id, a diffusion engine with an install plan,
    or the reserved ``"llamanager"`` self-update key."""
    from .auto_update import SELF_KEY
    from .engine_installer import ENGINE_PLANS
    from .llama_installer import parse_variant_id
    if key == SELF_KEY:
        return True
    if parse_variant_id(key) is not None:
        return True
    return key in ENGINE_PLANS


class AutoUpdateToggleBody(BaseModel):
    engine: str
    enabled: bool


class AutoUpdateSettingsBody(BaseModel):
    idle_seconds: int | None = None
    check_interval_seconds: int | None = None


@router.get("/setup/auto-update")
async def get_auto_update(request: Request,
                          _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Report the per-engine auto-update map + the idle/check tuning knobs."""
    cfg = request.app.state.cfg
    return JSONResponse({
        "engines": dict(cfg.auto_update_engines or {}),
        "idle_seconds": cfg.auto_update_idle_seconds,
        "check_interval_seconds": cfg.auto_update_check_interval_seconds,
    })


@router.post("/setup/auto-update")
async def set_auto_update(request: Request,
                          body: AutoUpdateToggleBody,
                          _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Flip one engine's auto-update-when-idle switch and persist it."""
    from .config import update_auto_update
    cfg = request.app.state.cfg
    if not _valid_auto_update_key(body.engine):
        raise HTTPException(
            status_code=400,
            detail=(f"unknown engine key {body.engine!r}: expected a llama "
                    f"variant id, a diffusion engine name, or 'llamanager'"))
    update_auto_update(cfg.config_path, engine=body.engine, enabled=body.enabled)
    cfg.auto_update_engines = dict(cfg.auto_update_engines or {})
    cfg.auto_update_engines[body.engine] = body.enabled
    request.app.state.db.log_event(
        "auto_update_toggled", {"engine": body.engine, "enabled": body.enabled})
    return JSONResponse({"ok": True, "engines": dict(cfg.auto_update_engines)})


@router.post("/setup/auto-update/settings")
async def set_auto_update_settings(request: Request,
                                   body: AutoUpdateSettingsBody,
                                   _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Tune the idle window and upstream-check cadence."""
    from .config import update_auto_update
    cfg = request.app.state.cfg
    if body.idle_seconds is not None and body.idle_seconds < 0:
        raise HTTPException(status_code=400, detail="idle_seconds must be >= 0")
    if (body.check_interval_seconds is not None
            and body.check_interval_seconds < 60):
        raise HTTPException(status_code=400,
                            detail="check_interval_seconds must be >= 60")
    update_auto_update(cfg.config_path,
                       idle_seconds=body.idle_seconds,
                       check_interval_seconds=body.check_interval_seconds)
    if body.idle_seconds is not None:
        cfg.auto_update_idle_seconds = body.idle_seconds
    if body.check_interval_seconds is not None:
        cfg.auto_update_check_interval_seconds = body.check_interval_seconds
    return JSONResponse({
        "ok": True,
        "idle_seconds": cfg.auto_update_idle_seconds,
        "check_interval_seconds": cfg.auto_update_check_interval_seconds,
    })


# ---------- ASR (speech-to-text) ----------
#
# Mirror the ASR models page over JSON so the CLI can list, install, profile,
# and transcribe without the web UI. Audio-family only (engine "asr").

def _reload_cfg_inplace(request: Request) -> None:
    """Re-read config.toml into app.state.cfg after a write, keeping the
    runtime bind/port stable — so a profile created over the admin API is
    immediately visible to listing and usable for transcription."""
    from .config import load_config
    old = request.app.state.cfg
    fresh = load_config(old.path)
    fresh.bind = old.bind
    fresh.port = old.port
    request.app.state.cfg = fresh

@router.get("/asr/engines")
async def asr_engines(request: Request,
                      _: Origin = Depends(admin_origin)) -> JSONResponse:
    """ASR engine(s) state: configured python path, install status, plan."""
    from . import engines as _engines
    from .config import ENGINE_FAMILY
    from .engine_installer import ENGINE_PLANS
    cfg = request.app.state.cfg
    installer = request.app.state.engine_installer
    out: list[dict[str, Any]] = []
    for eng_id in _engines.ADAPTERS.keys():
        if ENGINE_FAMILY.get(eng_id, "text") != "audio":
            continue
        configured = bool(cfg.asr_python) if eng_id == "asr" else False
        active = installer.active_for_engine(eng_id)
        last = active or (installer.list_for_engine(eng_id, limit=1) or [None])[0]
        plan = ENGINE_PLANS.get(eng_id)
        out.append({
            "engine": eng_id,
            "label": getattr(_engines.get(eng_id), "LABEL", eng_id),
            "configured": configured,
            "python": cfg.asr_python if eng_id == "asr" else "",
            "packages": (plan.packages if plan else []),
            "reuse_from": list(plan.reuse_from) if plan else [],
            "notes": (plan.notes if plan else ""),
            "last_install": last,
        })
    return JSONResponse({"engines": out})


class AsrInstallBody(BaseModel):
    torch_backend: str = "auto"


@router.post("/asr/install")
async def asr_install(request: Request, body: AsrInstallBody | None = None,
                      _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Install (or reuse a diffusion venv for) the ASR engine dependencies."""
    from .engine_installer import TORCH_BACKENDS
    installer = request.app.state.engine_installer
    options: dict[str, Any] = {}
    tb = ((body.torch_backend if body else "auto") or "auto").strip().lower()
    if tb in TORCH_BACKENDS and tb != "auto":
        options["torch_backend"] = tb
    try:
        install_id = installer.start("asr", options=options)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "install_id": install_id}, status_code=202)


@router.post("/asr/cancel-install")
async def asr_cancel_install(request: Request,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    installer = request.app.state.engine_installer
    active = installer.active_for_engine("asr")
    if active:
        installer.cancel(active["id"])
        return JSONResponse({"ok": True, "cancelled": active["id"]})
    return JSONResponse({"ok": True, "cancelled": None})


class AsrSetupBody(BaseModel):
    python: str


@router.post("/asr/setup")
async def asr_setup(request: Request, body: AsrSetupBody,
                    _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Point the ASR engine at a Python interpreter (torch + transformers)."""
    from .config import update_image_config
    cfg = request.app.state.cfg
    cfg.asr_python = body.python.strip()
    update_image_config(cfg.config_path, asr_python=cfg.asr_python)
    return JSONResponse({"ok": True, "asr_python": cfg.asr_python})


@router.get("/asr/models")
async def asr_models(request: Request,
                     _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Installed speech-to-text models under the ASR models directory."""
    from .audio_runner import scan_asr_models
    cfg = request.app.state.cfg
    on_disk: list[dict[str, Any]] = []
    for m in scan_asr_models(cfg):
        mc = cfg.get_model(m["model_id"])
        on_disk.append({
            "model_id": m["model_id"], "engine": "asr",
            "size_bytes": m["size_bytes"], "path": m["path"],
            "default_profile": (mc.default_profile if mc else ""),
            "profiles": sorted(mc.profiles.keys()) if mc else [],
        })
    return JSONResponse({
        "installed": on_disk,
        "configured": bool(cfg.asr_python),
        "asr_models_dir": str(cfg.asr_models_dir),
        "asr_models_dir_is_default": cfg.asr_models_dir_override is None,
    })


class AsrModelsDirBody(BaseModel):
    path: str = ""


@router.post("/asr/models-dir")
async def asr_models_dir(request: Request, body: AsrModelsDirBody,
                         _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Set (blank clears → shared models dir) the dedicated ASR models folder."""
    from .config import update_image_config
    cfg = request.app.state.cfg
    raw = body.path.strip()
    if raw:
        new_dir = Path(raw).expanduser().resolve()
        new_dir.mkdir(parents=True, exist_ok=True)
        cfg.asr_models_dir_override = new_dir
        update_image_config(cfg.config_path, asr_models_dir=str(new_dir))
    else:
        cfg.asr_models_dir_override = None
        update_image_config(cfg.config_path, asr_models_dir="")
    return JSONResponse({"ok": True, "asr_models_dir": str(cfg.asr_models_dir)})


@router.get("/asr/profiles")
async def asr_profiles_list(request: Request, model: str = "",
                            _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Profiles attached to an audio model + the engine's built-in defaults."""
    from . import engines as _engines
    from .config import detect_engine_for_id
    cfg = request.app.state.cfg
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    m = cfg.get_model(model)
    profiles: list[dict[str, Any]] = []
    if m:
        for p in sorted(m.profiles.values(), key=lambda x: x.name):
            fields = {k: getattr(p, k) for k in ("audio_language", "audio_task")
                      if getattr(p, k) not in ("", None)}
            profiles.append({"name": p.name, "fields": fields})
    engine = detect_engine_for_id(model, cfg.models_dir)
    try:
        builtins = _engines.get(engine).default_profiles()
    except Exception:
        builtins = {}
    return JSONResponse({
        "model": model, "engine": engine,
        "default_profile": (m.default_profile if m else ""),
        "profiles": profiles, "builtin_defaults": builtins,
    })


class AsrProfileBody(BaseModel):
    model_id: str
    name: str
    fields: dict[str, Any] = Field(default_factory=dict)
    make_default: bool = False


@router.post("/asr/profiles")
async def asr_profile_create(request: Request, body: AsrProfileBody,
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import (Profile, save_profile, set_model_default_profile,
                         detect_engine_for_id)
    from . import engines as _engines
    cfg = request.app.state.cfg
    if not body.model_id or not body.name:
        raise HTTPException(status_code=400, detail="model_id and name are required")
    name = body.name.strip().lower()
    existing = cfg.get_model(body.model_id)
    if existing and name in existing.profiles:
        raise HTTPException(status_code=409, detail=f"profile {name!r} already exists")
    engine = detect_engine_for_id(body.model_id, cfg.models_dir)
    try:
        adapter = _engines.get(engine)
    except KeyError:
        raise HTTPException(status_code=400,
                            detail=f"unknown engine for model {body.model_id!r}")
    kwargs: dict[str, Any] = {"name": name}
    schema_keys = {f.key for f in adapter.profile_schema()}
    for k, v in body.fields.items():
        if k in schema_keys and v not in ("", None):
            kwargs[k] = v
    try:
        save_profile(cfg.config_path, body.model_id, name, Profile(**kwargs))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    if body.make_default:
        set_model_default_profile(cfg.config_path, body.model_id, name)
    _reload_cfg_inplace(request)
    return JSONResponse({"ok": True, "name": name})


@router.delete("/asr/profiles/{name}")
async def asr_profile_delete(request: Request, name: str, model_id: str = "",
                             _: Origin = Depends(admin_origin)) -> JSONResponse:
    from .config import delete_profile
    cfg = request.app.state.cfg
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    delete_profile(cfg.config_path, model_id.strip(), name)
    _reload_cfg_inplace(request)
    return JSONResponse({"ok": True})


class AsrSetDefaultBody(BaseModel):
    model_id: str
    profile_name: str = ""


@router.post("/asr/profiles/set-default")
async def asr_profile_set_default(request: Request, body: AsrSetDefaultBody,
                                  _: Origin = Depends(admin_origin)) -> JSONResponse:
    """Set (or clear, with a blank name) a model's default audio profile."""
    from .config import set_model_default_profile
    cfg = request.app.state.cfg
    mid = body.model_id.strip()
    pname = body.profile_name.strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model_id is required")
    if pname and cfg.get_profile(mid, pname) is None:
        raise HTTPException(status_code=400,
                            detail=f"unknown profile {pname!r} for model {mid!r}")
    set_model_default_profile(cfg.config_path, mid, pname)
    _reload_cfg_inplace(request)
    return JSONResponse({"ok": True, "default_profile": pname})


class AsrTranscribeBody(BaseModel):
    file: str
    model: str
    language: str = ""
    task: str = "transcribe"
    profile: str = ""


@router.post("/asr/transcribe")
async def asr_transcribe(request: Request, body: AsrTranscribeBody,
                         origin: Origin = Depends(admin_origin)) -> JSONResponse:
    """Transcribe a server-local audio file through the normal queue.

    The admin/CLI caller and the daemon share a host, so we pass a path
    rather than uploading bytes. Routing still goes through the queue so the
    1-slot ceiling and text/image coexistence are honoured.
    """
    from .audio_runner import (AudioError, AudioTaskRunner,
                               execute_transcription, resolve_audio_engine)
    from .engines._base import AudioRequest
    from .queue_mgr import Cancelled, QueueFull
    cfg = request.app.state.cfg
    qm: QueueManager = request.app.state.queue
    runner: AudioTaskRunner = request.app.state.audio_runner

    audio_path = Path(body.file).expanduser()
    if not audio_path.is_file():
        raise HTTPException(status_code=400, detail=f"file not found: {audio_path}")
    try:
        engine = resolve_audio_engine(cfg, body.model)
    except AudioError as e:
        raise HTTPException(status_code=400, detail=str(e))

    profile_obj = None
    profile_required = body.profile.strip() or None
    if profile_required:
        profile_obj = cfg.get_profile(body.model, profile_required)
        if profile_obj is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown profile {profile_required!r} for model {body.model!r}")
    try:
        qr = await qm.enqueue(origin=origin, model_required=body.model,
                              profile_required=profile_required, task_type="audio")
    except QueueFull:
        raise HTTPException(status_code=503, detail="queue full")

    audio_req = AudioRequest(audio_path=audio_path,
                             language=(body.language.strip() or None),
                             task=(body.task or "transcribe"))
    try:
        result = await execute_transcription(
            qm, runner, qr=qr, engine=engine, model_id=body.model,
            profile_obj=profile_obj, audio_req=audio_req)
    except AudioError as e:
        code = 499 if str(e) == "cancelled" else 502
        raise HTTPException(status_code=code, detail=str(e))
    except Cancelled:
        raise HTTPException(status_code=499, detail="cancelled")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(e))
    return JSONResponse({
        "text": result.text, "language": result.language,
        "duration_s": result.duration_s, "segments": result.segments,
        "request_id": qr.request_id,
    })
