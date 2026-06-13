"""Anthropic-compatible inference proxy.

Mounted under ``/anthropic/v1`` so it sits beside the OpenAI surface in
:mod:`api_v1` without colliding on shapes (``/v1/models`` returns a
different envelope in each spec).

The actual inference engine (llama-server) speaks OpenAI chat
completions; this module is a translation layer. The flow:

  1. Authenticate the caller (``x-api-key`` or ``Authorization: Bearer``).
  2. Translate the Anthropic Messages request into an OpenAI chat
     completions body.
  3. Reuse the queue/dispatcher from :mod:`api_v1` to acquire a slot
     and either:
       - non-streaming: forward, translate the JSON back to Anthropic shape.
       - streaming: wrap the upstream OpenAI SSE bytes in a generator that
         re-emits them as Anthropic ``message_start`` /
         ``content_block_delta`` / ``message_stop`` events.

Tool-use is translated both directions: Anthropic ``tool_use`` /
``tool_result`` blocks ↔ OpenAI ``tool_calls`` / ``role=tool`` messages.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .api_v1 import (
    KEEPALIVE_INTERVAL_S,
    _apply_thinking_to_body,
    _cancelable_post,
    _extract_prompt_text,
    _extract_response_text,
    _model_allowed,
    _model_known,
    _profile_thinking,
    _stream_with_keepalives,
    _thinking_from_header,
)
from .auth import AuthManager, Origin
from .caller import describe_caller
from .config import Config
from .queue_mgr import Cancelled, QueueFull, QueueManager
from .registry import Registry
from .server_manager import ServerManager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/anthropic/v1", tags=["anthropic"])


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

async def _origin_from_request(req: Request) -> Origin:
    """Resolve the caller's origin from either Anthropic's ``x-api-key``
    or the OpenAI-style ``Authorization: Bearer`` header. Anthropic SDK
    clients send the former; ``Bearer`` is accepted so existing
    llamanager keys work without reconfiguring the SDK's auth slot."""
    key = req.headers.get("x-api-key", "").strip()
    if not key:
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth.split(" ", 1)[1].strip()
    if not key:
        raise HTTPException(
            status_code=401,
            detail="missing api key (x-api-key or Authorization: Bearer)",
        )
    am: AuthManager = req.app.state.auth
    origin = await am.verify(key)
    if not origin:
        raise HTTPException(status_code=401, detail="invalid api key")
    return origin


def _model_and_profile(
    req: Request, origin: Origin, body_model: str | None,
    cfg: Config, sm: ServerManager,
) -> tuple[str | None, str | None, str | None]:
    """Pick ``(model, profile, fallback_reason)`` for an Anthropic request.

    Precedence: body ``model`` field → ``X-Llamanager-Model`` header.
    Profile is taken from ``X-Llamanager-Profile`` only (Anthropic has no
    profile concept). The resolved model is honoured only when it is *known*
    and *allowed* for the origin; otherwise we degrade to the configured
    default model (``None``) and report a reason — same graceful behaviour as
    the OpenAI surface, no 403 for model choice. ``None`` means "let the
    dispatcher use the default / whatever is loaded".
    """
    model = (body_model or req.headers.get("x-llamanager-model") or "").strip()
    profile = (req.headers.get("x-llamanager-profile") or "").strip()
    if model and model.startswith("profile:"):
        raise HTTPException(
            status_code=400,
            detail=("the 'profile:<name>' shorthand is not supported here; "
                    "send X-Llamanager-Profile separately"),
        )
    if not model or model == "default":
        return None, (profile or None), None
    if not _model_known(cfg, sm, model):
        return None, (profile or None), (
            f"requested model '{model}' is not installed; using the default model"
        )
    if not _model_allowed(origin, model):
        return None, (profile or None), (
            f"origin '{origin.name}' is not permitted to use model '{model}'; "
            "using the default model"
        )
    return model, (profile or None), None


# --------------------------------------------------------------------------
# Anthropic Messages -> OpenAI chat completions
# --------------------------------------------------------------------------

def _flatten_system(system: Any) -> str | None:
    """Anthropic accepts ``system`` as either a string or a list of
    ``{"type":"text","text":...}`` blocks. Collapse to a single string;
    OpenAI's system role doesn't support multimodal content."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts) if parts else None
    raise HTTPException(
        status_code=400,
        detail="'system' must be a string or a list of text blocks",
    )


def _image_block_to_openai(src: dict[str, Any], idx: int) -> dict[str, Any]:
    """Turn one Anthropic image block ``source`` into an OpenAI
    ``image_url`` content part. Base64 sources become a data URL; URL
    sources pass through. The decode is left to the upstream — we only
    validate the envelope so a malformed block fails fast with a 400."""
    if not isinstance(src, dict):
        raise HTTPException(
            status_code=400,
            detail=f"image[{idx}].source must be an object",
        )
    stype = src.get("type")
    if stype == "base64":
        media_type = src.get("media_type") or "image/png"
        data = src.get("data")
        if not isinstance(data, str) or not data:
            raise HTTPException(
                status_code=400,
                detail=f"image[{idx}] base64 source is missing 'data'",
            )
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if stype == "url":
        url = src.get("url")
        if not isinstance(url, str) or not url:
            raise HTTPException(
                status_code=400,
                detail=f"image[{idx}] url source is missing 'url'",
            )
        return {"type": "image_url", "image_url": {"url": url}}
    raise HTTPException(
        status_code=400,
        detail=f"image[{idx}] source.type must be 'base64' or 'url'",
    )


def _translate_user_blocks(
    blocks: list[Any], msg_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate a user-message ``content`` array.

    Anthropic mixes ``tool_result`` blocks into the user message, but
    OpenAI represents tool results as separate ``role=tool`` messages.
    So this returns *two* lists:

      - ``parts``: the multimodal content kept on the user message
        (text + image_url entries). May be empty.
      - ``tool_msgs``: a list of synthesised ``role=tool`` messages, one
        per ``tool_result`` block, that the caller should emit BEFORE
        the user message in the OpenAI ``messages`` array.

    Tool results convert to a single string per OpenAI's contract: if
    the Anthropic block carries content blocks (text/image), we flatten
    text and drop images (OpenAI tool messages are text-only).
    """
    parts: list[dict[str, Any]] = []
    tool_msgs: list[dict[str, Any]] = []
    img_idx = 0
    for block in blocks:
        if not isinstance(block, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{msg_index}] content blocks must be objects",
            )
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
        elif btype == "image":
            parts.append(_image_block_to_openai(block.get("source"), img_idx))
            img_idx += 1
        elif btype == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise HTTPException(
                    status_code=400,
                    detail=(f"messages[{msg_index}] tool_result is missing "
                            f"'tool_use_id'"),
                )
            raw_content = block.get("content")
            if isinstance(raw_content, str):
                text_out = raw_content
            elif isinstance(raw_content, list):
                pieces: list[str] = []
                for sub in raw_content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        t = sub.get("text")
                        if isinstance(t, str):
                            pieces.append(t)
                text_out = "\n".join(pieces)
            else:
                text_out = ""
            if block.get("is_error"):
                text_out = f"[tool error] {text_out}"
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": text_out,
            })
        elif btype == "document":
            raise HTTPException(
                status_code=400,
                detail=(f"messages[{msg_index}] document blocks are not "
                        f"supported by the llamanager Anthropic facade"),
            )
        else:
            # thinking / server_tool_use / unknown — silently drop. These
            # don't have an OpenAI equivalent and shouldn't break the call.
            continue
    return parts, tool_msgs


def _translate_assistant_blocks(
    blocks: list[Any], msg_index: int,
) -> dict[str, Any]:
    """Translate an assistant-message ``content`` array into a single
    OpenAI message dict. Concatenates text blocks; converts ``tool_use``
    blocks into ``tool_calls`` entries. Drops ``thinking`` blocks — they
    don't round-trip through llama-server."""
    text_pieces: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{msg_index}] content blocks must be objects",
            )
        btype = block.get("type")
        if btype == "text":
            t = block.get("text")
            if isinstance(t, str):
                text_pieces.append(t)
        elif btype == "tool_use":
            tu_id = block.get("id")
            tu_name = block.get("name")
            tu_input = block.get("input")
            if not isinstance(tu_id, str) or not isinstance(tu_name, str):
                raise HTTPException(
                    status_code=400,
                    detail=(f"messages[{msg_index}] tool_use requires "
                            f"string 'id' and 'name'"),
                )
            tool_calls.append({
                "id": tu_id,
                "type": "function",
                "function": {
                    "name": tu_name,
                    "arguments": json.dumps(tu_input or {}),
                },
            })
        # thinking / other: drop.
    msg: dict[str, Any] = {"role": "assistant"}
    if text_pieces:
        msg["content"] = "".join(text_pieces)
    else:
        # OpenAI requires the field present when there are tool_calls;
        # null is the canonical "no text" marker.
        msg["content"] = None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _translate_messages(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{i}] must be an object",
            )
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=(f"messages[{i}].role must be 'user' or 'assistant' "
                        f"(got {role!r})"),
            )
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise HTTPException(
                status_code=400,
                detail=(f"messages[{i}].content must be a string or list "
                        f"of content blocks"),
            )
        if role == "user":
            parts, tool_msgs = _translate_user_blocks(content, i)
            # OpenAI wants tool result messages before the user message
            # they're a response to.
            out.extend(tool_msgs)
            if parts:
                # If every part is text, collapse to a single string
                # so plain text→text round-trips identically to the
                # OpenAI surface.
                if all(p.get("type") == "text" for p in parts):
                    out.append({
                        "role": "user",
                        "content": "".join(p["text"] for p in parts),
                    })
                else:
                    out.append({"role": "user", "content": parts})
        else:
            out.append(_translate_assistant_blocks(content, i))
    return out


def _translate_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="'tools' must be a list")
    out: list[dict[str, Any]] = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise HTTPException(
                status_code=400,
                detail=f"tools[{i}] must be an object",
            )
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise HTTPException(
                status_code=400,
                detail=f"tools[{i}] is missing 'name'",
            )
        # Anthropic server-side tools (web_search, computer_use, ...) carry
        # a ``type`` field and no ``input_schema``. We can't execute those
        # — reject so the caller sees the mismatch instead of getting a
        # silently-disabled tool.
        if "input_schema" not in t and t.get("type"):
            raise HTTPException(
                status_code=400,
                detail=(f"tools[{i}] ({t.get('type')!r}) is a server-side "
                        f"tool which the llamanager facade does not "
                        f"provide; pass user-defined tools with "
                        f"'input_schema' instead"),
            )
        schema = t.get("input_schema") or {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description") or "",
                "parameters": schema,
            },
        })
    return out


def _translate_tool_choice(tc: Any) -> Any:
    if tc is None:
        return None
    if not isinstance(tc, dict):
        raise HTTPException(
            status_code=400,
            detail="'tool_choice' must be an object",
        )
    ttype = tc.get("type")
    if ttype == "auto":
        return "auto"
    if ttype == "any":
        return "required"
    if ttype == "none":
        return "none"
    if ttype == "tool":
        name = tc.get("name")
        if not isinstance(name, str) or not name:
            raise HTTPException(
                status_code=400,
                detail="tool_choice.type='tool' requires 'name'",
            )
        return {"type": "function", "function": {"name": name}}
    raise HTTPException(
        status_code=400,
        detail=f"unknown tool_choice.type {ttype!r}",
    )


def _build_openai_body(body: dict[str, Any]) -> dict[str, Any]:
    """Top-level translator: Anthropic Messages request → OpenAI chat
    completions request body. Doesn't include the bare ``model`` field —
    the caller resolves the model id itself and lets the queue dispatcher
    pick the loaded one."""
    messages_in = body.get("messages")
    if not isinstance(messages_in, list) or not messages_in:
        raise HTTPException(
            status_code=400,
            detail="'messages' is required and must be a non-empty list",
        )
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="'max_tokens' is required and must be a positive integer",
        )

    openai_messages: list[dict[str, Any]] = []
    system = _flatten_system(body.get("system"))
    if system:
        openai_messages.append({"role": "system", "content": system})
    openai_messages.extend(_translate_messages(messages_in))

    out: dict[str, Any] = {
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": bool(body.get("stream", False)),
    }
    for src, dst in (("temperature", "temperature"),
                     ("top_p", "top_p"),
                     ("top_k", "top_k")):
        if src in body and body[src] is not None:
            out[dst] = body[src]
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        out["stop"] = stop

    tools = body.get("tools")
    if tools is not None:
        out["tools"] = _translate_tools(tools)
        tc = _translate_tool_choice(body.get("tool_choice"))
        if tc is not None:
            out["tool_choice"] = tc

    if body.get("stream") and bool(body.get("stream", False)):
        # llama.cpp's chat completions endpoint omits usage from the
        # final SSE chunk unless asked. We always want it so we can emit
        # the Anthropic ``message_delta`` event with output_tokens.
        out["stream_options"] = {"include_usage": True}
    return out


# --------------------------------------------------------------------------
# OpenAI -> Anthropic Messages (non-streaming)
# --------------------------------------------------------------------------

_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _anthropic_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _openai_to_anthropic_response(
    openai_body: dict[str, Any], *, model_id: str,
) -> dict[str, Any]:
    choices = openai_body.get("choices") or []
    if not choices:
        raise HTTPException(
            status_code=502,
            detail="upstream returned no choices",
        )
    choice = choices[0]
    msg = choice.get("message") or {}
    text = msg.get("content")
    tool_calls = msg.get("tool_calls") or []

    content_blocks: list[dict[str, Any]] = []
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        # Some llama.cpp builds echo content back as parts; flatten text.
        flat = "".join(
            p.get("text", "") for p in text
            if isinstance(p, dict) and p.get("type") == "text"
        )
        if flat:
            content_blocks.append({"type": "text", "text": flat})
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        args_raw = fn.get("arguments") or "{}"
        try:
            args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            # Forward as a string under a synthetic key so the caller
            # still sees what came back instead of getting a 502.
            args_obj = {"_raw": args_raw}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": fn.get("name") or "",
            "input": args_obj,
        })

    finish = choice.get("finish_reason") or "stop"
    # If the model emitted any tool_use blocks, normalize the stop reason
    # — llama.cpp sometimes reports "stop" alongside tool calls.
    if any(b["type"] == "tool_use" for b in content_blocks):
        stop_reason = "tool_use"
    else:
        stop_reason = _STOP_REASON.get(finish, "end_turn")

    usage = openai_body.get("usage") or {}
    return {
        "id": _anthropic_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# --------------------------------------------------------------------------
# SSE translator: OpenAI chunks -> Anthropic events
# --------------------------------------------------------------------------

def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Encode one Anthropic-shaped SSE event. Anthropic always sends
    BOTH an ``event:`` header and a ``data:`` JSON body — its SDK parses
    the event name, so omitting it would route every event through the
    generic ``data`` handler."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


class _AnthropicStreamTranslator:
    """Stateful translator that consumes OpenAI SSE chunks and emits
    Anthropic SSE events. Handles interleaved text and tool_use blocks,
    incrementally streaming partial JSON for tool arguments via
    ``input_json_delta`` events.

    The OpenAI stream uses one ``choices[0].delta`` per chunk that may
    contain ``content`` (text), ``tool_calls`` (each with ``index`` + a
    partial ``function.arguments`` string), and eventually
    ``finish_reason``. The Anthropic stream is block-oriented: each text
    or tool_use block has its own ``content_block_start`` /
    ``content_block_delta`` / ``content_block_stop`` lifecycle. We track
    one slot per logical block and emit the right transitions as the
    deltas arrive."""

    def __init__(self, *, model_id: str):
        self.model_id = model_id
        self.msg_id = _anthropic_msg_id()
        self.started = False
        self.message_stopped = False
        # Each entry: {"type": "text"|"tool_use", "index": int,
        #              "tc_index": int|None, "name": str|None}
        # ``tc_index`` is the OpenAI tool_calls[].index this block came
        # from, so subsequent deltas land in the right block. ``index``
        # is the Anthropic content-block index emitted to the client.
        self.blocks: list[dict[str, Any]] = []
        self._next_block_index = 0
        # The currently-open block index (last started, not yet stopped).
        # Text and tool_use blocks may interleave in principle, but in
        # practice OpenAI emits text first then tool_calls — we close
        # whichever is open before opening a new one of a different kind.
        self._open_text_block: int | None = None
        self._tc_index_to_block: dict[int, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.finish_reason: str | None = None
        self.stop_sequence: str | None = None
        # Accumulated assistant text, for the UI's request-detail view.
        self.text_parts: list[str] = []
        # SSE chunk-parser state
        self._buf = ""
        self.upstream_error: dict[str, Any] | None = None

    @property
    def response_text(self) -> str:
        return "".join(self.text_parts)

    # ---- output helpers --------------------------------------------------

    def _emit_message_start(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        payload = {
            "type": "message_start",
            "message": {
                "id": self.msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model_id,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": 0,
                },
            },
        }
        return [_sse_event("message_start", payload)]

    def _close_open_text(self) -> list[bytes]:
        if self._open_text_block is None:
            return []
        idx = self._open_text_block
        self._open_text_block = None
        return [_sse_event("content_block_stop",
                           {"type": "content_block_stop", "index": idx})]

    def _open_text(self) -> tuple[int, list[bytes]]:
        if self._open_text_block is not None:
            return self._open_text_block, []
        idx = self._next_block_index
        self._next_block_index += 1
        self._open_text_block = idx
        self.blocks.append({"type": "text", "index": idx})
        ev = _sse_event("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""},
        })
        return idx, [ev]

    def _open_tool_use(self, tc_index: int, tool_id: str,
                       name: str) -> tuple[int, list[bytes]]:
        # Close any open text block — Anthropic blocks don't overlap.
        out = self._close_open_text()
        idx = self._next_block_index
        self._next_block_index += 1
        self._tc_index_to_block[tc_index] = idx
        self.blocks.append({
            "type": "tool_use",
            "index": idx,
            "tc_index": tc_index,
            "name": name,
        })
        out.append(_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        }))
        return idx, out

    def _stop_all_blocks(self) -> list[bytes]:
        """Emit ``content_block_stop`` for every block that's still open."""
        out: list[bytes] = []
        out.extend(self._close_open_text())
        for b in self.blocks:
            if b["type"] == "tool_use" and not b.get("_stopped"):
                b["_stopped"] = True
                out.append(_sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": b["index"],
                }))
        return out

    # ---- ingest ---------------------------------------------------------

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume one upstream byte chunk; return Anthropic events to
        emit downstream. SSE comments (``:keepalive``) and blank lines
        are silently absorbed — the wrapping handler emits its own
        Anthropic-style ``ping`` events on a timer."""
        out: list[bytes] = []
        self._buf += chunk.decode("utf-8", errors="replace")
        while True:
            sep = self._buf.find("\n")
            if sep < 0:
                break
            line = self._buf[:sep].rstrip("\r")
            self._buf = self._buf[sep + 1:]
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            out.extend(self._consume(obj))
        return out

    def _consume(self, obj: dict[str, Any]) -> list[bytes]:
        out: list[bytes] = []

        # Some llama.cpp builds emit a final usage-only object after
        # ``finish_reason`` instead of inlining it. Capture either form.
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if "prompt_tokens" in usage:
                self.input_tokens = int(usage.get("prompt_tokens") or 0)
            if "completion_tokens" in usage:
                self.output_tokens = int(usage.get("completion_tokens") or 0)

        # Surface upstream-side errors as a translator-level error so the
        # caller emits message_stop cleanly instead of stalling.
        if isinstance(obj.get("error"), dict):
            self.upstream_error = obj["error"]
            return out

        # Anthropic's message_start carries input_tokens. llama.cpp tends
        # to put usage in the final chunk only, so we emit message_start
        # eagerly on the first content delta and let the final
        # ``message_delta`` carry the real output_tokens.
        choices = obj.get("choices") or []
        if not choices:
            return out
        choice = choices[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        if not self.started:
            out.extend(self._emit_message_start())

        # ---- text delta -----
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.text_parts.append(content)
            idx, opening = self._open_text()
            out.extend(opening)
            out.append(_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": content},
            }))
        elif isinstance(content, list):
            # Some servers emit content as parts even mid-stream.
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str) and t:
                        self.text_parts.append(t)
                        idx, opening = self._open_text()
                        out.extend(opening)
                        out.append(_sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": t},
                        }))

        # ---- tool-call deltas -----
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_idx = tc.get("index")
            if not isinstance(tc_idx, int):
                # Without an index we can't multiplex multiple calls;
                # fall back to a synthetic 0 so single-tool callers still
                # work.
                tc_idx = 0
            fn = tc.get("function") or {}
            existing_block_idx = self._tc_index_to_block.get(tc_idx)
            if existing_block_idx is None:
                # First time we see this tool_call: open a tool_use block.
                tool_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"
                name = fn.get("name") or ""
                existing_block_idx, opening = self._open_tool_use(
                    tc_idx, tool_id, name,
                )
                out.extend(opening)
            args = fn.get("arguments")
            if isinstance(args, str) and args:
                out.append(_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": existing_block_idx,
                    "delta": {"type": "input_json_delta",
                              "partial_json": args},
                }))

        if finish:
            self.finish_reason = finish
        return out

    # ---- finalisation ---------------------------------------------------

    def finalise(self) -> list[bytes]:
        """Emit the closing ``content_block_stop`` for every block plus
        ``message_delta`` (stop reason + final usage) and ``message_stop``."""
        if self.message_stopped:
            return []
        self.message_stopped = True
        out: list[bytes] = []
        if not self.started:
            # The upstream returned nothing usable — still emit a valid
            # (empty) Anthropic message so SDK clients close cleanly.
            out.extend(self._emit_message_start())
        out.extend(self._stop_all_blocks())
        stop_reason = "end_turn"
        if any(b["type"] == "tool_use" for b in self.blocks):
            stop_reason = "tool_use"
        elif self.finish_reason:
            stop_reason = _STOP_REASON.get(self.finish_reason, "end_turn")
        out.append(_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": self.stop_sequence,
            },
            "usage": {"output_tokens": self.output_tokens},
        }))
        out.append(_sse_event("message_stop", {"type": "message_stop"}))
        return out

    def finalise_with_error(self, err: dict[str, Any]) -> list[bytes]:
        """Emit a translator-level error event followed by a clean stop
        so SDK clients see the failure as an event and close the stream."""
        out: list[bytes] = []
        out.append(_sse_event("error", {
            "type": "error",
            "error": {
                "type": err.get("type") or "api_error",
                "message": err.get("message") or "upstream error",
            },
        }))
        # Don't tack on message_stop after error — that's how Anthropic
        # frames it: an ``error`` event terminates the stream.
        self.message_stopped = True
        return out


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

@router.post("/messages")
async def messages(request: Request) -> Response:
    origin = await _origin_from_request(request)
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    cfg: Config = request.app.state.cfg
    qm: QueueManager = request.app.state.queue
    sm: ServerManager = request.app.state.sm

    requested_model = body.get("model") if isinstance(body.get("model"), str) else None
    model_required, profile_required, model_fallback = _model_and_profile(
        request, origin, requested_model, cfg, sm,
    )
    if model_fallback:
        log.info("origin %s: %s", origin.name, model_fallback)
    openai_body = _build_openai_body(body)

    # Apply reasoning default/override the same way the OpenAI handler does
    # — chat/completions is the only place ``enable_thinking`` is consumed.
    header_thinking = _thinking_from_header(request)
    if header_thinking:
        _apply_thinking_to_body(openai_body, header_thinking, forced=True)
    else:
        prof_thinking = _profile_thinking(cfg, model_required, profile_required)
        if prof_thinking:
            _apply_thinking_to_body(openai_body, prof_thinking, forced=False)

    try:
        qr = await qm.enqueue(
            origin=origin,
            model_required=model_required,
            profile_required=profile_required,
            caller=await describe_caller(request),
        )
    except QueueFull:
        raise HTTPException(status_code=503, detail="queue full")

    # The Anthropic response uses whatever the caller sent as the model
    # name (or the resolved bare id) — never the OpenAI ``id`` echoed back.
    response_model_name = (requested_model or model_required
                           or sm.runtime.current_model or "llamanager")
    streaming = openai_body.get("stream", False)

    if streaming:
        return await _stream_messages(request, qm, qr, sm,
                                        openai_body, response_model_name)
    return await _blocking_messages(request, qm, qr, sm,
                                      openai_body, response_model_name)


async def _blocking_messages(request: Request, qm: QueueManager, qr,
                              sm: ServerManager, openai_body: dict[str, Any],
                              response_model_name: str) -> Response:
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_text = _extract_prompt_text(openai_body)
    response_text: str | None = None
    if getattr(request.app.state.cfg, "conversation_retention_days", 0) > 0:
        qr.prompt_text = prompt_text
    try:
        await qm.wait_for_slot(qr)
        from ._routing import upstream_base as _upstream
        url = f"{_upstream(sm, qr)}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                r = await _cancelable_post(client, url, openai_body, qr)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"upstream error: {e}")
        if r.status_code >= 400:
            detail = r.text or f"HTTP {r.status_code}"
            raise HTTPException(
                status_code=r.status_code if r.status_code < 600 else 502,
                detail=f"upstream returned {r.status_code}: {detail}",
            )
        try:
            upstream = r.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=502,
                                detail="upstream returned non-JSON body")
        response_text = _extract_response_text(upstream)
        anth = _openai_to_anthropic_response(upstream, model_id=response_model_name)
        prompt_tokens = anth["usage"]["input_tokens"] or None
        completion_tokens = anth["usage"]["output_tokens"] or None
        headers = {"x-llamanager-request-id": qr.request_id}
        return JSONResponse(content=anth, headers=headers)
    except Cancelled:
        error = "cancelled"
        return JSONResponse(
            status_code=499,
            content={"type": "error",
                     "error": {"type": "request_cancelled",
                               "message": "request cancelled"}},
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
        log.exception("anthropic non-stream proxy failed for %s", qr.request_id)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        qm.mark_in_flight_done(
            qr, error=error, cancelled=(error == "cancelled"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_text=prompt_text,
            response_text=response_text,
        )


async def _stream_messages(request: Request, qm: QueueManager, qr,
                            sm: ServerManager, openai_body: dict[str, Any],
                            response_model_name: str) -> StreamingResponse:
    prompt_text = _extract_prompt_text(openai_body)
    retain = getattr(request.app.state.cfg, "conversation_retention_days", 0) > 0
    if retain:
        qr.prompt_text = prompt_text
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
        translator = _AnthropicStreamTranslator(model_id=response_model_name)
        # Share the translator's text buffer with the queued request so the
        # detail view can stream the partial response while it generates.
        if retain:
            qr.response_parts = translator.text_parts
        upstream_iter = _stream_with_keepalives(
            qm, qr, sm, "/v1/chat/completions", openai_body, client_disconnected,
        )
        last_ping = time.monotonic()
        try:
            async for chunk in upstream_iter:
                # Pass through SSE keepalives as Anthropic ``ping`` events
                # so middleboxes see traffic without polluting the stream
                # with unknown event types.
                if chunk.startswith(b":"):
                    now = time.monotonic()
                    if now - last_ping > KEEPALIVE_INTERVAL_S / 2:
                        last_ping = now
                        yield _sse_event("ping", {"type": "ping"})
                    continue
                events = translator.feed(chunk)
                if translator.upstream_error:
                    for e in translator.finalise_with_error(
                            translator.upstream_error):
                        yield e
                    return
                for e in events:
                    yield e
            for e in translator.finalise():
                yield e
            # Capture token usage for the queue's accounting.
            qr_tokens = (translator.input_tokens or None,
                         translator.output_tokens or None)
            # Stash on the request so the finally block can pick it up.
            request.state._anth_tokens = qr_tokens  # type: ignore[attr-defined]
        except Exception as e:
            error = str(e)
            log.exception("anthropic stream failed for %s", qr.request_id)
            for ev in translator.finalise_with_error({
                "type": "api_error", "message": error,
            }):
                yield ev
        finally:
            watch_task.cancel()
            cancelled = client_disconnected.is_set() or qr.cancel.is_set()
            tokens = getattr(request.state, "_anth_tokens", (None, None))
            qm.mark_in_flight_done(
                qr, error=error, cancelled=cancelled,
                prompt_tokens=tokens[0], completion_tokens=tokens[1],
                prompt_text=prompt_text,
                response_text=translator.response_text,
            )

    headers = {
        "x-llamanager-request-id": qr.request_id,
        "cache-control": "no-cache",
        "x-accel-buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers=headers)


# --------------------------------------------------------------------------
# count_tokens
# --------------------------------------------------------------------------

@router.post("/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
    """Approximate token count for an Anthropic Messages request.

    We translate the request to OpenAI shape, render it into a single
    string (system + each message's flattened text), and call
    llama-server's ``/tokenize`` endpoint. Tool definitions and image
    blocks are excluded from the count — a deliberate undercount that
    matches what local inference will actually tokenize."""
    origin = await _origin_from_request(request)
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    requested_model = body.get("model") if isinstance(body.get("model"), str) else None
    # Validates the profile-shorthand contract; model choice itself degrades
    # to the default rather than erroring, so the result is unused here.
    _ = _model_and_profile(
        request, origin, requested_model,
        request.app.state.cfg, request.app.state.sm,
    )

    # Render to a single string so /tokenize sees something representative.
    parts: list[str] = []
    system = _flatten_system(body.get("system"))
    if system:
        parts.append(system)
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        raise HTTPException(
            status_code=400,
            detail="'messages' is required and must be a list",
        )
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{i}] must be an object",
            )
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
                elif block.get("type") == "tool_use":
                    name = block.get("name") or ""
                    inp = block.get("input")
                    parts.append(f"{name}({json.dumps(inp or {})})")
                elif block.get("type") == "tool_result":
                    raw = block.get("content")
                    if isinstance(raw, str):
                        parts.append(raw)
                    elif isinstance(raw, list):
                        for sub in raw:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                t = sub.get("text")
                                if isinstance(t, str):
                                    parts.append(t)
    rendered = "\n".join(parts)

    sm: ServerManager = request.app.state.sm
    if not sm.is_running:
        # Without a live engine we can't tokenize; return a rough byte/4
        # estimate so SDK callers that budget against the result still
        # get a usable number.
        approx = max(1, len(rendered) // 4)
        return JSONResponse({"input_tokens": approx})

    # Advisory tokenize: hit slot 0 (the pool's default forwarder). Even
    # with multi-slot on, we don't know which slot the caller intends to
    # eventually talk to, so we use the default slot's tokenizer. The
    # estimate is good enough for SDK budgeting.
    url = f"{sm.upstream_base}/tokenize"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json={"content": rendered})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"upstream tokenize failed: {e}")
    if r.status_code >= 400:
        # Fall back to the rough estimate rather than surfacing an opaque
        # error — count_tokens is advisory.
        approx = max(1, len(rendered) // 4)
        return JSONResponse({"input_tokens": approx})
    try:
        payload = r.json()
    except json.JSONDecodeError:
        approx = max(1, len(rendered) // 4)
        return JSONResponse({"input_tokens": approx})
    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        return JSONResponse({"input_tokens": len(tokens)})
    approx = max(1, len(rendered) // 4)
    return JSONResponse({"input_tokens": approx})


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

@router.get("/models")
async def list_models(request: Request) -> Response:
    """List models in Anthropic's shape.

    Anthropic returns ``{"data": [{"id", "type": "model", "display_name",
    "created_at"}], "has_more": false, "first_id": ..., "last_id": ...}``
    — distinct from the OpenAI envelope at ``/v1/models``."""
    origin = await _origin_from_request(request)
    sm: ServerManager = request.app.state.sm
    reg: Registry = request.app.state.registry

    if "*" in origin.allowed_models:
        visible = [m.model_id for m in reg.list()]
    else:
        registered = {m.model_id for m in reg.list()}
        visible = [a for a in origin.allowed_models
                   if a != "default" and a in registered]
    # Union loaded slots so multi-slot users see every warm model.
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

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data = [{
        "id": m,
        "type": "model",
        "display_name": m,
        "created_at": now_iso,
    } for m in visible]
    return JSONResponse({
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    })
