"""OpenAI-compatible inference proxy.

Behavior:
- Resolve origin from Authorization: Bearer <key>.
- Resolve requested model:
    1. X-Llamanager-Model header (bare model id, subject to origin.allowed_models)
    2. X-Llamanager-Profile header (optional) — selects a profile bound to the
       model above (or the global default profile). Profile without a model
       is rejected (400).
    3. otherwise leave model_required=None — dispatcher will use whatever's
       loaded and cold-start the default if nothing is.
- Enqueue, then either:
    - non-streaming: wait for slot, forward request, return JSON.
    - streaming: open SSE response immediately and emit ":queued" / ":swapping"
      keepalive comments while we wait. Once the slot is ours, proxy bytes
      from llama-server.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthManager, Origin
from .caller import describe_caller
from .config import ENGINE_FAMILY, Config, detect_engine_for_id
from .audio_runner import (
    AudioError, AudioTaskRunner, execute_transcription, resolve_audio_engine,
)
from .engines._base import AudioRequest, ImageRequest
from .image_runner import ImageError, ImageTaskRunner, resolve_image_engine
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
    _require_origin_enabled(origin)
    return origin


def _require_origin_enabled(origin: Origin) -> None:
    """Reject a disabled origin from submitting work. Auth still succeeds (so
    the caller gets a clear 403, not a misleading 401); the operator toggle
    lives on the origin row (see auth.set_enabled)."""
    if not origin.enabled:
        raise HTTPException(
            status_code=403,
            detail=(f"origin '{origin.name}' is disabled and may not submit "
                    f"requests; ask an administrator to re-enable it."),
        )


def _model_allowed(origin: Origin, model_id: str) -> bool:
    """Whether ``origin`` may use ``model_id`` per its allowlist.

    ``["*"]`` allows everything; the special token ``"default"`` allows the
    operator's configured default model; otherwise the id must be listed.
    """
    allowed = origin.allowed_models
    if "*" in allowed:
        return True
    if model_id == "default" and "default" in allowed:
        return True
    return model_id in allowed


def _check_model_allowed(origin: Origin, model_id: str) -> None:
    """Raising form of :func:`_model_allowed`.

    Used where a hard 403 is the right answer — the image endpoints, which
    require an explicit model id and have no graceful default to fall back to.
    The text inference path uses the non-raising predicate instead and degrades
    to the default model.
    """
    if not _model_allowed(origin, model_id):
        raise HTTPException(
            status_code=403,
            detail=f"origin '{origin.name}' is not allowed to use model '{model_id}'",
        )


def _model_known(cfg: Config, sm: ServerManager, model_id: str) -> bool:
    """Whether ``model_id`` names a model llamanager can serve: a configured
    model, or one currently loaded in a slot. Kept cheap (no disk scan) for the
    request hot path; mirrors what ``/v1/models`` surfaces from config + slots.
    """
    if cfg.get_model(model_id) is not None:
        return True
    if hasattr(sm, "slots"):
        return any(getattr(sv, "model", None) == model_id for sv in sm.slots())
    return getattr(sm.runtime, "current_model", None) == model_id


def _model_required(req: Request, origin: Origin) -> tuple[str | None, str | None]:
    """Return (model_id, profile_name) from the request headers.

    No allowlist enforcement happens here — :func:`_resolve_request_model`
    applies the allow/known checks with a graceful fall back to the default
    model. The legacy ``X-Llamanager-Model: profile:foo`` shorthand is still
    rejected with a 400 pointing callers at the two-header contract.
    """
    model = req.headers.get("x-llamanager-model")
    profile = req.headers.get("x-llamanager-profile")
    if model and model.startswith("profile:"):
        raise HTTPException(
            status_code=400,
            detail=(
                "the 'profile:<name>' shorthand on X-Llamanager-Model is no "
                "longer supported. Send X-Llamanager-Model: <model-id> and "
                "X-Llamanager-Profile: <name> as separate headers."
            ),
        )
    return (model or None), (profile or None)


def _model_from_body(body: dict[str, Any]) -> str | None:
    """The OpenAI ``model`` field from the request body as a routing hint.

    Stock OpenAI clients (Continue, the ``openai`` SDK, …) name the model in
    the body, not the ``X-Llamanager-Model`` header. Returns the trimmed id,
    or ``None`` for empty / ``"default"`` / non-string values. Existence and
    allowlist checks happen in :func:`_resolve_request_model`.
    """
    m = body.get("model")
    if not isinstance(m, str):
        return None
    m = m.strip()
    if not m or m == "default":
        return None
    return m


def _resolve_request_model(
    req: Request, origin: Origin, body: dict[str, Any],
    cfg: Config, sm: ServerManager,
) -> tuple[str | None, str | None, str | None]:
    """Resolve ``(model_id, profile, fallback_reason)`` for an inference request.

    Precedence: the ``X-Llamanager-Model`` header wins; otherwise the OpenAI
    ``model`` field in the body. The resolved model is honoured only when it is
    *known* (a configured model or one currently loaded) **and** *allowed* for
    the origin. When nothing usable is requested — no model named, an unknown
    id, or one the origin isn't permitted to use — we fall back to the
    configured default model (returned as ``None`` so the dispatcher picks
    ``cfg.default_model``) and report a human-readable reason. Model choice
    never raises 403 here; it degrades to the default.
    """
    model, profile = _model_required(req, origin)  # header only (may 400 on shorthand)
    if model is None:
        model = _model_from_body(body)
    if not model or model == "default":
        return None, profile, None
    if not _model_known(cfg, sm, model):
        return None, profile, (
            f"requested model '{model}' is not installed; using the default model"
        )
    if not _model_allowed(origin, model):
        return None, profile, (
            f"origin '{origin.name}' is not permitted to use model '{model}'; "
            "using the default model"
        )
    return model, profile, None


def _is_streaming(body: dict[str, Any]) -> bool:
    return bool(body.get("stream", False))


# --------------------------------------------------------------------------
# Prompt / response capture for the UI's request-detail view.
# --------------------------------------------------------------------------

def _render_content(content: Any) -> str:
    """Flatten one OpenAI message ``content`` into readable text.

    Multimodal parts (image_url, input_audio, …) collapse to a short
    ``[image]`` / ``[audio]`` placeholder so we never persist megabytes of
    base64 in the requests table.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("type")
                if t == "text":
                    parts.append(str(p.get("text") or ""))
                elif t in ("image_url", "input_image", "image"):
                    parts.append("[image]")
                elif t in ("input_audio", "audio"):
                    parts.append("[audio]")
                else:
                    parts.append(f"[{t or 'part'}]")
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def _extract_prompt_text(body: dict[str, Any]) -> str:
    """Best-effort human-readable rendering of the request prompt."""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        lines: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "?")
            lines.append(f"{role}: {_render_content(m.get('content'))}")
        return "\n\n".join(lines)
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(p) for p in prompt)
    return ""


def _extract_response_text(parsed: dict[str, Any]) -> str:
    """Pull the assistant text out of a non-streaming completion body."""
    try:
        choices = parsed.get("choices") or []
        if not choices:
            return ""
        c0 = choices[0] or {}
        msg = c0.get("message")
        if isinstance(msg, dict):
            return _render_content(msg.get("content"))
        if c0.get("text") is not None:
            return str(c0.get("text") or "")
    except Exception:
        return ""
    return ""


def _parse_sse_data(event: bytes) -> dict[str, Any] | None:
    """Parse the JSON object from a single SSE event's ``data:`` line.

    Returns None for comments (``:`` keepalives), the ``[DONE]`` sentinel,
    and anything that doesn't parse as a JSON object.
    """
    try:
        text = event.decode("utf-8", errors="replace")
    except Exception:
        return None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    obj = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    return None
                return obj if isinstance(obj, dict) else None
    return None


async def _proxy_and_capture(
    upstream: AsyncIterator[bytes],
    *,
    strip_usage_only: bool,
    response_parts: list[str],
    usage_holder: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Forward an upstream SSE byte stream one complete event at a time,
    capturing the assistant text into ``response_parts`` and token usage
    into ``usage_holder`` (both owned by the caller, mutated in place so the
    caller sees the latest values even if iteration stops early).

    ``strip_usage_only`` drops the synthetic usage-only tail chunk (empty
    ``choices`` + a ``usage`` object) that llama.cpp emits because we asked
    for ``include_usage`` — keeping the bytes the client receives identical
    to a stream we never touched. Buffering raw bytes (never re-encoding)
    keeps multibyte UTF-8 content intact across chunk boundaries.
    """
    pending = b""

    def _absorb(obj: dict[str, Any]) -> bool:
        """Record usage/text from one parsed event; return True to drop it."""
        u = obj.get("usage")
        if u:
            usage_holder.clear()
            usage_holder.update(u)
        choices = obj.get("choices") or []
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta")
            if isinstance(delta, dict) and delta.get("content"):
                response_parts.append(str(delta["content"]))
            elif ch.get("text"):
                response_parts.append(str(ch["text"]))
        return bool(strip_usage_only and u and not choices)

    async for chunk in upstream:
        pending += chunk
        while b"\n\n" in pending:
            event, pending = pending.split(b"\n\n", 1)
            obj = _parse_sse_data(event)
            if obj is not None and _absorb(obj):
                continue
            yield event + b"\n\n"
    # Flush any trailing bytes not terminated by a blank line.
    if pending:
        obj = _parse_sse_data(pending)
        if not (obj is not None and _absorb(obj)):
            yield pending


def _thinking_from_header(request: Request) -> str:
    """Read the per-request reasoning override. Empty when unset.

    Raises 400 on an unrecognised value so callers see the typo instead of
    silently inheriting the profile default.
    """
    raw = request.headers.get("x-llamanager-thinking", "")
    val = raw.strip().lower()
    if not val:
        return ""
    if val not in ("on", "off"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid X-Llamanager-Thinking value {raw!r}; "
                f"expected 'on' or 'off'"
            ),
        )
    return val


def _profile_thinking(cfg: Config, model: str | None, profile: str | None) -> str:
    """Resolve the profile's ``thinking`` field for a request.

    Walks the same model/profile fallback chain that ``resolve_spec`` uses
    so the body-merge default tracks the profile the dispatcher will pick
    when no header override is supplied.
    """
    mid = (model or cfg.default_model or "").strip()
    if not mid:
        return ""
    m = cfg.get_model(mid)
    if not m:
        return ""
    pname = (profile or m.default_profile or "").strip()
    if not pname:
        return ""
    prof = m.profiles.get(pname)
    return prof.thinking if prof else ""


def _apply_thinking_to_body(body: dict[str, Any], thinking: str,
                            *, forced: bool) -> None:
    """Merge ``chat_template_kwargs.enable_thinking`` into ``body``.

    ``forced=True`` (the header override) wins over any caller-supplied
    value. ``forced=False`` (the profile default) defers to whatever the
    caller already set, so explicit per-call control still works.
    """
    if thinking not in ("on", "off"):
        return
    enable = (thinking == "on")
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
        body["chat_template_kwargs"] = kwargs
    if forced or "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = enable


async def _cancelable_post(
    client: httpx.AsyncClient, url: str, body: dict[str, Any], qr: QueuedRequest
) -> httpx.Response:
    """``client.post`` that aborts the moment ``qr.cancel`` fires.

    A bare ``await client.post(..., timeout=None)`` blocks until the upstream
    returns the *entire* completion, so a cancel arriving mid-generation is
    ignored and llama-server keeps generating to the end. We race the POST
    against the cancel event; if cancel wins we abort the request task — which
    closes the underlying connection so the engine stops — and raise
    ``Cancelled``. Upstream transport errors surface unchanged for the caller's
    ``httpx.HTTPError`` handling.
    """
    post_task = asyncio.ensure_future(client.post(url, json=body))
    cancel_task = asyncio.ensure_future(qr.cancel.wait())
    try:
        await asyncio.wait(
            {post_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        # Cancel wins ties: if the flag is set, treat as cancelled even if the
        # POST also happened to finish in the same loop step.
        if qr.cancel.is_set():
            raise Cancelled()
        return post_task.result()
    finally:
        for t in (post_task, cancel_task):
            if not t.done():
                t.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await post_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await cancel_task


async def _iter_until_cancel(
    aiter: AsyncIterator[bytes], cancel: asyncio.Event
) -> AsyncIterator[bytes]:
    """Yield from ``aiter`` but stop promptly when ``cancel`` fires.

    The upstream byte iterator blocks in ``__anext__`` while the engine
    prefills a long prompt — no bytes flow yet, so a per-chunk cancel check
    never gets a turn. Racing each fetch against ``cancel`` makes prefill
    interruptible: when cancel wins we abort the pending read (closing the
    connection) and stop iterating. Genuine upstream errors propagate.
    """
    it = aiter.__aiter__()
    cancel_task = asyncio.ensure_future(cancel.wait())
    try:
        while True:
            nxt = asyncio.ensure_future(it.__anext__())
            await asyncio.wait(
                {nxt, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancel.is_set():
                if not nxt.done():
                    nxt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await nxt
                return
            try:
                yield nxt.result()
            except StopAsyncIteration:
                return
    finally:
        if not cancel_task.done():
            cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await cancel_task


async def _proxy_non_streaming(
    qr: QueuedRequest, sm: ServerManager, path: str, body: dict[str, Any]
) -> Response:
    from ._routing import upstream_base as _upstream
    url = f"{_upstream(sm, qr)}{path}"
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            r = await _cancelable_post(client, url, body, qr)
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
        last_emit: tuple[str, int] = ("", -2)
        while not ready_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(ready_task),
                                       timeout=KEEPALIVE_INTERVAL_S)
            except asyncio.TimeoutError:
                if qr.cancel.is_set():
                    cancelled_flag["value"] = True
                    return
                status = qr.status
                # position is only meaningful while still pending; once
                # the dispatcher pulls us off the heap, we're no longer
                # countable. Emit -1 to mean "not in pending anymore"
                # so the client knows to drop the position label.
                pos = qm.position_for(qr) if status == "queued" else -1
                emit_key = (status, pos)
                if emit_key != last_emit:
                    last_emit = emit_key
                    chunk = f": status={status}\n"
                    if pos >= 0:
                        chunk += f": queue_pos={pos}\n"
                    chunk += "\n"
                    yield chunk.encode("utf-8")
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
        from ._routing import upstream_base as _upstream
        slot_base = _upstream(sm, qr)
        url = f"{slot_base}{path}"
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=body) as r:
                    if r.status_code >= 400:
                        # Upstream returned an error before any tokens.
                        # Surface it as a structured SSE event so the
                        # client gets a usable message instead of raw
                        # upstream bytes (which may not even be SSE).
                        err_body = await r.aread()
                        detail = (
                            err_body.decode("utf-8", errors="replace").strip()
                            or f"HTTP {r.status_code}"
                        )
                        payload = {"error": {
                            "message": f"upstream returned {r.status_code}: {detail}",
                            "type": "llamanager_upstream_error",
                            "status": r.status_code,
                        }}
                        yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                        return
                    async for chunk in _iter_until_cancel(
                            r.aiter_bytes(), qr.cancel):
                        if qr.cancel.is_set():
                            cancelled_flag["value"] = True
                            return
                        if not chunk:
                            continue
                        yield chunk
                    # _iter_until_cancel returns (rather than raising) when
                    # cancel fires during a prefill stall — catch that here.
                    if qr.cancel.is_set():
                        cancelled_flag["value"] = True
                        return
        except httpx.RequestError as e:
            # Connect / read / write failure talking to the upstream
            # engine — most commonly: the engine crashed or exited between
            # the readiness gate and the proxy POST. Emit a clean SSE
            # error and [DONE] so the client closes the stream gracefully.
            log.warning(
                "upstream connect/transport error for %s at %s: %s",
                qr.request_id, url, e,
            )
            payload = {"error": {
                "message": (
                    f"could not reach inference engine at {slot_base}: "
                    f"{e}. The engine may have crashed or stopped — check "
                    f"its status."
                ),
                "type": "llamanager_upstream_unreachable",
            }}
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
            return
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
    cfg: Config = request.app.state.cfg

    # Resolve the model from the X-Llamanager-Model header or the OpenAI
    # ``model`` field in the body. A request for an unknown or not-permitted
    # model degrades to the configured default rather than erroring.
    model_required, profile_required, model_fallback = _resolve_request_model(
        request, origin, body, cfg, sm
    )
    if model_fallback:
        log.info("origin %s: %s", origin.name, model_fallback)

    # Apply reasoning default/override. Header wins over profile; both
    # only touch /v1/chat/completions (the bare /completions endpoint
    # doesn't render the chat template that consumes the kwarg).
    if path == "/v1/chat/completions":
        header_thinking = _thinking_from_header(request)
        if header_thinking:
            _apply_thinking_to_body(body, header_thinking, forced=True)
        else:
            profile_thinking = _profile_thinking(
                cfg, model_required, profile_required,
            )
            if profile_thinking:
                _apply_thinking_to_body(body, profile_thinking, forced=False)

    try:
        qr = await qm.enqueue(
            origin=origin,
            model_required=model_required,
            profile_required=profile_required,
            caller=await describe_caller(request),
        )
    except QueueFull:
        raise HTTPException(status_code=503, detail="queue full")

    # Snapshot the prompt for the request-detail view before we forward.
    # When retention is on, hang it (and a live response buffer) off the
    # queued request so the detail view can show the conversation *while it
    # is still running*, not only once it finishes.
    prompt_text = _extract_prompt_text(body)
    retain = getattr(request.app.state.cfg, "conversation_retention_days", 0) > 0
    if retain:
        qr.prompt_text = prompt_text

    streaming = _is_streaming(body)

    # llama.cpp omits ``usage`` from the streaming SSE tail unless asked, so
    # streaming requests would otherwise never show token counts or tok/s in
    # the dashboard. Ask for it — but only when the caller didn't already, so
    # we know to strip the extra usage-only chunk back out and keep the bytes
    # the client receives identical to a vanilla stream.
    injected_usage = False
    if streaming:
        so = body.get("stream_options")
        client_wants_usage = isinstance(so, dict) and so.get("include_usage") is True
        if not client_wants_usage:
            body["stream_options"] = {
                **(so if isinstance(so, dict) else {}),
                "include_usage": True,
            }
            injected_usage = True

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
            # Accumulate into the queued request's live buffer (when
            # retaining) so the detail view streams the partial response.
            response_parts: list[str] = qr.response_parts if retain else []
            try:
                upstream = _stream_with_keepalives(
                    qm, qr, sm, path, body, client_disconnected
                )
                async for chunk in _proxy_and_capture(
                    upstream,
                    strip_usage_only=injected_usage,
                    response_parts=response_parts,
                    usage_holder=usage,
                ):
                    yield chunk
            except Exception as e:
                error = str(e)
                log.exception("stream error for %s", qr.request_id)
                payload = {"error": {"message": error, "type": "llamanager_error"}}
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            finally:
                watch_task.cancel()
                cancelled = bool(getattr(qr, "_stream_cancelled", False))
                qm.mark_in_flight_done(
                    qr,
                    error=error,
                    cancelled=cancelled,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    prompt_text=prompt_text,
                    response_text="".join(response_parts) or None,
                )

        headers = {
            "x-llamanager-request-id": qr.request_id,
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        }
        if model_fallback:
            headers["x-llamanager-model-fallback"] = model_fallback
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers=headers)

    # non-streaming
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_text: str | None = None
    try:
        await qm.wait_for_slot(qr)
        resp = await _proxy_non_streaming(qr, sm, path, body)
        # Try to extract token usage + the response text from the body.
        try:
            if resp.media_type and "json" in resp.media_type:
                parsed = json.loads(resp.body)
                usage = parsed.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                response_text = _extract_response_text(parsed)
        except Exception:
            pass
        if model_fallback:
            resp.headers["x-llamanager-model-fallback"] = model_fallback
        return resp
    except Cancelled:
        error = "cancelled"
        return JSONResponse(
            status_code=499,
            content={"error": {"message": "request cancelled",
                               "type": "llamanager_error"}},
        )
    except asyncio.TimeoutError:
        error = "queue timeout"
        qm.cancel(qr.request_id)
        raise HTTPException(
            status_code=504,
            detail="request timed out waiting in queue",
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
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_text=prompt_text,
            response_text=response_text,
        )


@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _handle_inference(request, "/v1/chat/completions")


@router.post("/completions")
async def completions(request: Request) -> Response:
    return await _handle_inference(request, "/v1/completions")


def _parse_size(size: str | None) -> tuple[int, int]:
    """Parse an OpenAI-style ``WxH`` size; default 0,0 lets the adapter
    pick from its profile/defaults."""
    if not size:
        return 0, 0
    s = str(size).lower().strip().replace("×", "x")
    if "x" not in s:
        return 0, 0
    w, h = s.split("x", 1)
    try:
        return int(w), int(h)
    except ValueError:
        return 0, 0


# Reference-image constraints. Per-image and per-request caps protect the
# daemon from OOM and disk-bomb shapes; format whitelist matches what the
# downstream engines actually accept.
_REF_IMAGE_MAX_BYTES = 20 * 1024 * 1024     # 20 MiB per image, decoded
_REF_IMAGE_MAX_COUNT = 8                     # mirrors n's upper bound
_REF_IMAGE_MAGIC = {
    b"\x89PNG\r\n\x1a\n":          ("png",  "image/png"),
    b"\xff\xd8\xff":                ("jpg",  "image/jpeg"),
    b"RIFF":                        ("webp", "image/webp"),   # checked further below
}


def _decode_ref_image(raw: str, index: int) -> tuple[bytes, str]:
    """Decode one base64 reference-image string from the request body.

    Accepts either a bare base64 payload or a ``data:image/...;base64,...``
    URL. Returns ``(bytes, file_extension)``. Raises ``HTTPException`` with
    a 400 for any decoding or validation failure.
    """
    import binascii
    s = (raw or "").strip()
    if not s:
        raise HTTPException(status_code=400,
                            detail=f"image[{index}] is empty")
    if s.startswith("data:"):
        # data URL — header is "data:<mime>;base64,<payload>"
        try:
            header, payload = s.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400,
                                detail=f"image[{index}] is not a valid data URL")
        if ";base64" not in header.lower():
            raise HTTPException(
                status_code=400,
                detail=f"image[{index}] data URL must be base64-encoded",
            )
        s = payload
    try:
        blob = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"image[{index}] is not valid base64: {e}",
        )
    if not blob:
        raise HTTPException(status_code=400,
                            detail=f"image[{index}] decoded to 0 bytes")
    if len(blob) > _REF_IMAGE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(f"image[{index}] is {len(blob)} bytes; max is "
                    f"{_REF_IMAGE_MAX_BYTES}"),
        )
    # Sniff format from the magic bytes — never trust the data URL's mime.
    ext: str | None = None
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        ext = "png"
    elif blob.startswith(b"\xff\xd8\xff"):
        ext = "jpg"
    elif blob.startswith(b"RIFF") and len(blob) >= 12 and blob[8:12] == b"WEBP":
        ext = "webp"
    if ext is None:
        raise HTTPException(
            status_code=400,
            detail=(f"image[{index}] is not a recognised PNG / JPEG / WebP "
                    f"payload (sniffed from the decoded bytes)"),
        )
    return blob, ext


def _stage_ref_images(
    cfg, request_id: str, payloads: list[str]
) -> list:
    """Decode + persist a list of base64 reference images.

    Files land under ``<data_dir>/refs/<request_id>/`` as ``ref_NN.<ext>``.
    Returns the list of absolute paths. Caller (ImageTaskRunner) is
    responsible for removing the parent directory after use.
    """
    from pathlib import Path as _Path
    if not payloads:
        return []
    if len(payloads) > _REF_IMAGE_MAX_COUNT:
        raise HTTPException(
            status_code=400,
            detail=(f"too many reference images ({len(payloads)}); max is "
                    f"{_REF_IMAGE_MAX_COUNT}"),
        )
    ref_root = _Path(cfg.data_dir) / "refs" / request_id
    ref_root.mkdir(parents=True, exist_ok=True)
    paths: list[_Path] = []
    for i, raw in enumerate(payloads):
        if not isinstance(raw, str):
            raise HTTPException(
                status_code=400,
                detail=f"image[{i}] must be a base64 string or data URL",
            )
        blob, ext = _decode_ref_image(raw, i)
        p = ref_root / f"ref_{i:02d}.{ext}"
        p.write_bytes(blob)
        paths.append(p)
    return paths


@router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request) -> Response:
    """Transcribe an audio file (OpenAI-compatible).

    multipart/form-data fields:
        file:             the audio file (required; wav/mp3/m4a/ogg/…).
        model:            bare model id (audio-family); falls back to the
                          X-Llamanager-Model header if omitted.
        language:         ISO hint (e.g. "ar"), or "auto"/omitted to detect.
        response_format:  "json" (default) | "text" | "verbose_json".
        profile:          llamanager extension — selects an audio profile.
    """
    origin = await _origin_from_request(request)
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="expected multipart/form-data")

    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="'file' (audio upload) is required")

    qm: QueueManager = request.app.state.queue
    cfg = request.app.state.cfg
    runner: AudioTaskRunner = request.app.state.audio_runner

    model_required = (form.get("model")
                      or request.headers.get("x-llamanager-model"))
    if not model_required:
        raise HTTPException(
            status_code=400,
            detail="audio requests require 'model' (an audio-family model id)",
        )
    _check_model_allowed(origin, model_required)
    try:
        engine = resolve_audio_engine(cfg, model_required)
    except AudioError as e:
        raise HTTPException(status_code=400, detail=str(e))

    profile_required = (form.get("profile")
                        or request.headers.get("x-llamanager-profile"))
    language = (form.get("language") or "").strip() or None
    task = (form.get("task") or "transcribe").strip().lower()
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400,
                            detail="task must be 'transcribe' or 'translate'")
    response_format = (form.get("response_format") or "json").strip()

    try:
        qr = await qm.enqueue(
            origin=origin,
            model_required=model_required,
            profile_required=profile_required,
            task_type="audio",
            caller=await describe_caller(request),
        )
    except QueueFull:
        raise HTTPException(status_code=503, detail="queue full")

    # Stage the uploaded audio under <data_dir>/audio_in/<request_id>/ after
    # enqueue (stable request_id) but before the runner spawns.
    stage_dir = cfg.data_dir / "audio_in" / qr.request_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(getattr(upload, "filename", "") or "audio").suffix or ".bin"
    audio_path = stage_dir / f"input{suffix}"
    try:
        audio_path.write_bytes(await upload.read())
    except Exception as e:
        qm.cancel(qr.request_id)
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"could not read upload: {e}")

    profile_obj = None
    if profile_required:
        profile_obj = cfg.get_profile(model_required, profile_required)
        if profile_obj is None:
            qm.cancel(qr.request_id)
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"unknown profile {profile_required!r} for model {model_required!r}",
            )

    audio_req = AudioRequest(audio_path=audio_path, language=language,
                             task=task)
    try:
        result = await execute_transcription(
            qm, runner, qr=qr, engine=engine, model_id=model_required,
            profile_obj=profile_obj, audio_req=audio_req,
        )
        headers = {"x-llamanager-request-id": qr.request_id}
        if response_format == "text":
            return Response(content=result.text, media_type="text/plain",
                            headers=headers)
        if response_format == "verbose_json":
            return JSONResponse(content=result.raw, headers=headers)
        return JSONResponse(content={"text": result.text}, headers=headers)
    except AudioError as e:
        if str(e) == "cancelled":
            return JSONResponse(
                status_code=499,
                content={"error": {"message": "request cancelled",
                                   "type": "llamanager_error"}},
            )
        raise HTTPException(status_code=502, detail=str(e))
    except Cancelled:
        return JSONResponse(
            status_code=499,
            content={"error": {"message": "request cancelled",
                               "type": "llamanager_error"}},
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504,
                            detail="request timed out waiting in queue")
    except Exception as e:
        log.exception("transcription failed for %s", qr.request_id)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


@router.post("/images/generations")
async def images_generations(request: Request) -> Response:
    """Generate one or more images.

    OpenAI-compatible request body:
        prompt:                str (required)
        model:                 str — bare model id (image-family); if omitted, falls
                               back to the X-Llamanager-Model header.
        n:                     int (default 1)
        size:                  "WxH" (default per adapter)
        response_format:       "b64_json" (default) | "url"
        stream:                bool — when true, returns SSE progress with a final
                               data event carrying the result.
        seed:                  int (optional)
        profile:               llamanager-specific — selects an image profile by name.

    Reference-image fields (llamanager extension):
        image:                 str — single base64-encoded image (data URL or
                               raw base64). Shorthand for ``images: [image]``.
        images:                list[str] — up to 8 base64 reference images.
                               HiDream accepts multiple (composition / multi-
                               subject); Flux2 accepts exactly one (img2img).
        keep_original_aspect:  bool — HiDream only. With exactly one ref,
                               resize it to max 2048 on the long side and use
                               those dimensions for the output (bypasses the
                               2048-bucket snap).
        layout_bboxes:         str — HiDream only. JSON forwarded verbatim to
                               --layout_bboxes (e.g. ``"[[0.1,0.4,0.2,0.6]]"``).
        strength:              float — Flux2 img2img denoise strength (0..1).
    """
    origin = await _origin_from_request(request)
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    prompt = body.get("prompt") or ""
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    qm: QueueManager = request.app.state.queue
    cfg = request.app.state.cfg
    runner: ImageTaskRunner = request.app.state.image_runner

    # Resolve model id: body `model`, X-Llamanager-Model header, or the
    # operator-set default image model.
    model_required = (body.get("model")
                      or request.headers.get("x-llamanager-model")
                      or cfg.default_image_model)
    if not model_required:
        raise HTTPException(
            status_code=400,
            detail=("image requests require 'model' (an image-family model id) "
                    "or a default image model set in the UI top bar"),
        )
    _check_model_allowed(origin, model_required)
    # Verify it's an image-family model before queueing.
    try:
        engine = resolve_image_engine(cfg, model_required)
    except ImageError as e:
        raise HTTPException(status_code=400, detail=str(e))

    profile_required = (body.get("profile")
                        or request.headers.get("x-llamanager-profile"))
    # When the caller didn't specify a profile but the default-image model
    # is in play, fall back to the default profile saved for that model.
    if (not profile_required
            and model_required == cfg.default_image_model
            and cfg.default_image_profile):
        profile_required = cfg.default_image_profile

    n = int(body.get("n", 1) or 1)
    if n < 1 or n > 8:
        raise HTTPException(status_code=400, detail="n must be between 1 and 8")
    width, height = _parse_size(body.get("size"))
    response_format = str(body.get("response_format") or "b64_json").lower()
    if response_format not in ("b64_json", "url"):
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'b64_json' or 'url'",
        )
    streaming = bool(body.get("stream", False))
    seed = body.get("seed")
    seed_int: int | None
    if seed is None:
        seed_int = None
    else:
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="seed must be an integer")

    # ---- reference-image inputs (llamanager extension) -------------------
    # Accept ``image`` (single) or ``images`` (list); the OpenAI Images
    # API uses both spellings depending on the route. We unify them here.
    raw_ref_payloads: list[str] = []
    single_ref = body.get("image")
    multi_refs = body.get("images")
    if multi_refs is not None:
        if not isinstance(multi_refs, list):
            raise HTTPException(
                status_code=400,
                detail="'images' must be an array of base64 strings",
            )
        raw_ref_payloads = list(multi_refs)
    if single_ref is not None:
        if not isinstance(single_ref, str):
            raise HTTPException(
                status_code=400,
                detail="'image' must be a base64 string or data URL",
            )
        # If both are sent, prepend the single-image field for stability —
        # callers that send both probably meant the singular as primary.
        raw_ref_payloads = [single_ref] + raw_ref_payloads

    # Engine-specific arity guard. Adapters re-check too, but we want a
    # clear 400 before we burn a queue slot or decode bytes.
    if engine == "flux2" and len(raw_ref_payloads) > 1:
        raise HTTPException(
            status_code=400,
            detail=("flux2 supports at most one reference image (img2img); "
                    f"got {len(raw_ref_payloads)}"),
        )

    keep_original_aspect = bool(body.get("keep_original_aspect", False))
    layout_bboxes_raw = body.get("layout_bboxes")
    layout_bboxes: str | None
    if layout_bboxes_raw is None or layout_bboxes_raw == "":
        layout_bboxes = None
    elif isinstance(layout_bboxes_raw, str):
        layout_bboxes = layout_bboxes_raw
    else:
        # Allow callers to send a JSON value directly — re-serialize it.
        try:
            layout_bboxes = json.dumps(layout_bboxes_raw)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="layout_bboxes must be a JSON string or JSON-serialisable value",
            )

    strength_raw = body.get("strength")
    strength: float | None
    if strength_raw is None:
        strength = None
    else:
        try:
            strength = float(strength_raw)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail="strength must be a number")
        if not 0.0 <= strength <= 1.0:
            raise HTTPException(
                status_code=400, detail="strength must be in [0, 1]")

    try:
        qr = await qm.enqueue(
            origin=origin,
            model_required=model_required,
            profile_required=profile_required,
            task_type="image",
            caller=await describe_caller(request),
        )
    except QueueFull:
        raise HTTPException(status_code=503, detail="queue full")

    if cfg.conversation_retention_days > 0:
        qr.prompt_text = prompt  # visible in the detail view while generating

    # Stage refs to disk *after* enqueue so we have a stable request_id to
    # namespace them by, but *before* the runner kicks off so the adapter
    # sees real paths. Any decode failure unwinds the queue entry via
    # cancel() — using mark_in_flight_done here would incorrectly free a
    # slot the dispatcher never acquired (the request is still queued).
    try:
        ref_paths = _stage_ref_images(cfg, qr.request_id, raw_ref_payloads)
    except HTTPException:
        qm.cancel(qr.request_id)
        raise

    # Build the typed request now so adapter errors fail fast.
    image_req = ImageRequest(
        prompt=prompt,
        width=width,
        height=height,
        steps=None,    # adapter pulls from profile/defaults
        seed=seed_int,
        n=n,
        ref_images=ref_paths,
        keep_original_aspect=keep_original_aspect,
        layout_bboxes=layout_bboxes,
        strength=strength,
    )

    # Resolve the profile object (may be None — runner accepts that).
    profile_obj = None
    if profile_required:
        profile_obj = cfg.get_profile(model_required, profile_required)
        if profile_obj is None:
            qm.cancel(qr.request_id)
            raise HTTPException(
                status_code=400,
                detail=f"unknown profile {profile_required!r} for model {model_required!r}",
            )

    if streaming:
        return await _images_stream(qm, qr, runner, request, image_req,
                                     model_required, engine, profile_obj,
                                     response_format)
    return await _images_blocking(qm, qr, runner, request, image_req,
                                    model_required, engine, profile_obj,
                                    response_format)


async def _images_blocking(
    qm: QueueManager,
    qr: QueuedRequest,
    runner: ImageTaskRunner,
    request: Request,
    image_req: ImageRequest,
    model_required: str,
    engine: str,
    profile_obj,
    response_format: str,
) -> Response:
    error: str | None = None
    try:
        await qm.wait_for_slot(qr)
        try:
            result = await runner.run(
                model_id=model_required,
                engine=engine,
                profile=profile_obj,
                req=image_req,
                request_id=qr.request_id,
                origin_name=qr.origin.name,
                cancel_event=qr.cancel,
            )
        except ImageError as e:
            error = str(e)
            raise HTTPException(status_code=502, detail=error)
        payload = _build_image_response(result, response_format, request)
        headers = {"x-llamanager-request-id": qr.request_id}
        return JSONResponse(content=payload, headers=headers)
    except Cancelled:
        error = "cancelled"
        return JSONResponse(
            status_code=499,
            content={"error": {"message": "request cancelled",
                               "type": "llamanager_error"}},
        )
    except asyncio.TimeoutError:
        error = "queue timeout"
        qm.cancel(qr.request_id)
        raise HTTPException(status_code=504,
                            detail="request timed out waiting in queue")
    except HTTPException:
        raise
    except Exception as e:
        error = str(e)
        log.exception("image generation failed for %s", qr.request_id)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        qm.mark_in_flight_done(
            qr, error=error,
            cancelled=(error == "cancelled"),
            prompt_tokens=None, completion_tokens=None,
            prompt_text=image_req.prompt,
        )


async def _images_stream(
    qm: QueueManager,
    qr: QueuedRequest,
    runner: ImageTaskRunner,
    request: Request,
    image_req: ImageRequest,
    model_required: str,
    engine: str,
    profile_obj,
    response_format: str,
) -> StreamingResponse:
    """SSE response. Emits ``: status=...`` while queued/swapping,
    ``: step=N/M`` while generating, then a final ``data:`` event with
    the OpenAI-shaped payload."""
    progress_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def _on_progress(ev) -> None:
        try:
            progress_queue.put_nowait(("step", (ev.step, ev.total)))
        except asyncio.QueueFull:
            pass

    async def gen() -> AsyncIterator[bytes]:
        error: str | None = None
        try:
            # Phase 1: wait for slot. ImageTaskRunner is invoked AFTER the
            # slot is acquired; until then emit keepalives.
            ready_task = asyncio.create_task(qr.ready.wait())
            last_emit: tuple[str, int] = ("", -2)
            while not ready_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(ready_task),
                                            timeout=KEEPALIVE_INTERVAL_S)
                except asyncio.TimeoutError:
                    if qr.cancel.is_set():
                        return
                    status = qr.status
                    pos = qm.position_for(qr) if status == "queued" else -1
                    emit_key = (status, pos)
                    if emit_key != last_emit:
                        last_emit = emit_key
                        chunk = f": status={status}\n"
                        if pos >= 0:
                            chunk += f": queue_pos={pos}\n"
                        chunk += "\n"
                        yield chunk.encode("utf-8")
                    else:
                        yield b": keepalive\n\n"
            if qr.error:
                payload = {"error": {"message": qr.error,
                                      "type": "llamanager_error"}}
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            # Phase 2: run the engine. We launch the actual task as a
            # background task so we can interleave progress events.
            run_task = asyncio.create_task(
                runner.run(
                    model_id=model_required,
                    engine=engine,
                    profile=profile_obj,
                    req=image_req,
                    request_id=qr.request_id,
                    origin_name=qr.origin.name,
                    progress_cb=_on_progress,
                    cancel_event=qr.cancel,
                )
            )
            # Poll the progress queue often (so per-step events reach the
            # client live) but only emit a keepalive after a real lull —
            # otherwise the bound to KEEPALIVE_INTERVAL_S would batch
            # progress 10s at a time and a fast run would show none at all.
            last_out = time.monotonic()
            while not run_task.done():
                done, _ = await asyncio.wait(
                    {run_task},
                    timeout=0.4,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                emitted = False
                while not progress_queue.empty():
                    kind, payload = progress_queue.get_nowait()
                    if kind == "step":
                        step, total = payload
                        yield f": step={step}/{total}\n\n".encode("utf-8")
                        emitted = True
                now = time.monotonic()
                if emitted:
                    last_out = now
                elif not done and (now - last_out) >= KEEPALIVE_INTERVAL_S:
                    yield b": keepalive\n\n"
                    last_out = now

            try:
                result = run_task.result()
            except ImageError as e:
                error = str(e)
                payload = {"error": {"message": error,
                                      "type": "llamanager_error"}}
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            final_payload = _build_image_response(result, response_format, request)
            yield f"data: {json.dumps(final_payload)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        except Exception as e:
            error = str(e)
            log.exception("image stream failed for %s", qr.request_id)
            yield f"data: {json.dumps({'error': {'message': error}})}\n\n".encode("utf-8")
        finally:
            qm.mark_in_flight_done(
                qr, error=error,
                cancelled=(error == "cancelled"),
                prompt_tokens=None, completion_tokens=None,
                prompt_text=image_req.prompt,
            )

    headers = {
        "x-llamanager-request-id": qr.request_id,
        "cache-control": "no-cache",
        "x-accel-buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers=headers)


def _build_image_response(result, response_format: str, request: Request) -> dict[str, Any]:
    """Translate an ImageResult into an OpenAI-shaped JSON payload."""
    out_path = result.output_path
    items: list[dict[str, Any]] = []
    paths: list = [out_path] + [
        type(out_path)(p) for p in result.sidecar.get("batch", [])
        if p and p != str(out_path)
    ]
    for p in paths:
        item: dict[str, Any] = {}
        if response_format == "b64_json":
            try:
                item["b64_json"] = base64.b64encode(p.read_bytes()).decode("ascii")
            except OSError:
                continue
        else:
            # Authenticated UI route, gated by the same session as the rest.
            item["url"] = f"/ui/images/file/{p.parent.parent.name}/{p.parent.name}/{p.name}"
        item["revised_prompt"] = result.sidecar.get("prompt")
        items.append(item)
    return {
        "created": int(time.time()),
        "data": items,
        "llamanager": {
            "engine": result.engine,
            "model": result.model_id,
            "profile": result.profile_name,
            "seed": result.seed,
            "duration_s": round(result.duration_s, 3),
        },
    }


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
    # Surface every loaded slot's model so consumers see them all when
    # multi-slot is on. ``sm.slots()`` exists on ServerPool; in legacy
    # single-instance mode (or under tests with a bare ServerManager) we
    # fall back to the single ``runtime.current_model`` field.
    loaded_models: list[str] = []
    if hasattr(sm, "slots"):
        for sv in sm.slots():
            if sv.model and sv.model not in loaded_models:
                loaded_models.append(sv.model)
    elif sm.runtime.current_model:
        loaded_models.append(sm.runtime.current_model)
    for m in loaded_models:
        if m not in visible:
            visible.insert(0, m)

    now = int(time.time())
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": now, "owned_by": "llamanager"}
            for m in visible
        ],
    })
