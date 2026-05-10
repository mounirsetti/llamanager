"""OpenAI-compatible inference proxy.

Behavior:
- Resolve origin from Authorization: Bearer <key>.
- Resolve requested model:
    1. X-Llamanager-Model header (subject to origin.allowed_models)
    2. otherwise leave model_required=None — dispatcher will use whatever's loaded
       and cold-start the default if nothing is.
- Enqueue, then either:
    - non-streaming: wait for slot, forward request, return JSON.
    - streaming: open SSE response immediately and emit ":queued" / ":swapping"
      keepalive comments while we wait. Once the slot is ours, proxy bytes
      from llama-server.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthManager, Origin
from .queue_mgr import Cancelled, QueueManager, QueueFull, QueuedRequest
from .registry import Registry
from .server_manager import ServerManager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])

KEEPALIVE_INTERVAL_S = 10.0


async def _origin_from_request(req: Request) -> Origin:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    key = auth.split(" ", 1)[1].strip()
    am: AuthManager = req.app.state.auth
    origin = await am.verify(key)
    if not origin:
        raise HTTPException(status_code=401, detail="invalid api key")
    return origin


def _check_model_allowed(origin: Origin, model_id: str) -> None:
    allowed = origin.allowed_models
    if "*" in allowed:
        return
    if model_id == "default" and "default" in allowed:
        return
    if model_id in allowed:
        return
    raise HTTPException(
        status_code=403,
        detail=f"origin '{origin.name}' is not allowed to use model '{model_id}'",
    )


def _model_required(req: Request, origin: Origin) -> str | None:
    hdr = req.headers.get("x-llamanager-model")
    if not hdr:
        return None
    _check_model_allowed(origin, hdr)
    return hdr


def _is_streaming(body: dict[str, Any]) -> bool:
    return bool(body.get("stream", False))


async def _proxy_non_streaming(
    qr: QueuedRequest, sm: ServerManager, path: str, body: dict[str, Any]
) -> Response:
    url = f"{sm.upstream_base}{path}"
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            r = await client.post(url, json=body)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    headers = {
        "x-llamanager-request-id": qr.request_id,
    }
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=headers,
        media_type=r.headers.get("content-type", "application/json"),
    )


async def _stream_with_keepalives(
    qm: QueueManager,
    qr: QueuedRequest,
    sm: ServerManager,
    path: str,
    body: dict[str, Any],
    client_disconnected: asyncio.Event,
) -> AsyncIterator[bytes]:
    """SSE generator. Emits keepalive comments while the request waits in
    the queue or during a model swap, then proxies upstream chunks once
    the dispatcher releases our slot.

    SSE comments (lines starting with `:`) are ignored by clients but keep
    proxies and connection-tracking middleboxes from timing out.
    """
    cancelled_flag = {"value": False}

    async def watch_disconnect() -> None:
        await client_disconnected.wait()
        qr.cancel.set()

    disconnect_task = asyncio.create_task(watch_disconnect())

    try:
        # Phase 1: wait for slot, emit keepalives.
        ready_task = asyncio.create_task(qr.ready.wait())
        last_status_emitted = ""
        while not ready_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(ready_task),
                                       timeout=KEEPALIVE_INTERVAL_S)
            except asyncio.TimeoutError:
                if qr.cancel.is_set():
                    cancelled_flag["value"] = True
                    return
                status = qr.status
                if status != last_status_emitted:
                    last_status_emitted = status
                    yield f": status={status}\n\n".encode("utf-8")
                else:
                    yield b": keepalive\n\n"

        if qr.cancel.is_set():
            cancelled_flag["value"] = True
            return
        if qr.error:
            payload = {"error": {"message": qr.error,
                                 "type": "llamanager_error"}}
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
            return

        # Phase 2: proxy stream.
        url = f"{sm.upstream_base}{path}"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=body) as r:
                async for chunk in r.aiter_bytes():
                    if qr.cancel.is_set():
                        cancelled_flag["value"] = True
                        return
                    if not chunk:
                        continue
                    yield chunk
    finally:
        disconnect_task.cancel()
        # Bookkeeping happens in the wrapping handler.
        qr._stream_cancelled = cancelled_flag["value"]  # type: ignore[attr-defined]


async def _handle_inference(
    request: Request, path: str
) -> Response:
    origin = await _origin_from_request(request)
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    qm: QueueManager = request.app.state.queue
    sm: ServerManager = request.app.state.sm

    model_required = _model_required(request, origin)
    try:
        qr = await qm.enqueue(origin=origin, model_required=model_required)
    except QueueFull:
        raise HTTPException(status_code=503, detail="queue full")

    streaming = _is_streaming(body)

    if streaming:
        client_disconnected = asyncio.Event()

        async def watcher() -> None:
            try:
                while True:
                    if await request.is_disconnected():
                        client_disconnected.set()
                        return
                    await asyncio.sleep(1.0)
            except Exception:
                client_disconnected.set()

        watch_task = asyncio.create_task(watcher())

        async def gen() -> AsyncIterator[bytes]:
            error: str | None = None
            usage: dict[str, Any] = {}
            buffer = ""
            try:
                async for chunk in _stream_with_keepalives(
                    qm, qr, sm, path, body, client_disconnected
                ):
                    yield chunk
                    # Parse SSE chunks to extract usage from the final event
                    try:
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                parsed = json.loads(line[6:])
                                u = parsed.get("usage")
                                if u:
                                    usage = u
                    except Exception:
                        pass
            except Exception as e:
                error = str(e)
                log.exception("stream error for %s", qr.request_id)
                payload = {"error": {"message": error, "type": "llamanager_error"}}
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            finally:
                watch_task.cancel()
                cancelled = bool(getattr(qr, "_stream_cancelled", False))
                qm.mark_in_flight_done(
                    qr,
                    error=error,
                    cancelled=cancelled,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )

        headers = {
            "x-llamanager-request-id": qr.request_id,
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        }
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers=headers)

    # non-streaming
    error: str | None = None
    try:
        await qm.wait_for_slot(qr)
        resp = await _proxy_non_streaming(qr, sm, path, body)
        # Try to extract token usage from the response body for stats.
        prompt_tokens = completion_tokens = None
        try:
            if resp.media_type and "json" in resp.media_type:
                parsed = json.loads(resp.body)
                usage = parsed.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
        except Exception:
            pass
        return resp
    except Cancelled:
        error = "cancelled"
        return JSONResponse(
            status_code=499,
            content={"error": {"message": "request cancelled",
                               "type": "llamanager_error"}},
        )
    except HTTPException:
        raise
    except Exception as e:
        error = str(e)
        log.exception("non-streaming proxy failed for %s", qr.request_id)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        qm.mark_in_flight_done(
            qr, error=error, cancelled=(error == "cancelled"),
            prompt_tokens=locals().get("prompt_tokens"),
            completion_tokens=locals().get("completion_tokens"),
        )


@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _handle_inference(request, "/v1/chat/completions")


@router.post("/completions")
async def completions(request: Request) -> Response:
    return await _handle_inference(request, "/v1/completions")


@router.get("/models")
async def list_models(request: Request) -> Response:
    origin = await _origin_from_request(request)
    sm: ServerManager = request.app.state.sm
    reg: Registry = request.app.state.registry

    visible: list[str] = []
    if "*" in origin.allowed_models:
        visible = [m.model_id for m in reg.list()]
    else:
        registered = {m.model_id for m in reg.list()}
        for a in origin.allowed_models:
            if a == "default":
                continue
            if a in registered:
                visible.append(a)
    if sm.runtime.current_model and sm.runtime.current_model not in visible:
        visible.insert(0, sm.runtime.current_model)

    now = int(time.time())
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": now, "owned_by": "llamanager"}
            for m in visible
        ],
    })
