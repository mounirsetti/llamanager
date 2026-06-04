"""Prompt/response capture, streaming usage handling, and the request
detail view.

Covers the three dashboard fixes:
  * tok/s shows for streaming requests (we ask llama.cpp for usage and
    capture it, stripping the synthetic chunk back out);
  * the Recent origin column stays a name on the polled partial (the JOIN
    fix), with the id column replaced by a local time;
  * clicking a request surfaces its stored prompt + response.
"""
from __future__ import annotations

import asyncio
import re
import time

from fastapi.testclient import TestClient

from llamanager.api_ui import COOKIE_NAME, _localdt
from llamanager.api_v1 import (
    _extract_prompt_text,
    _extract_response_text,
    _parse_sse_data,
    _proxy_and_capture,
    _render_content,
)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_extract_prompt_text_chat_messages():
    body = {"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]}
    assert _extract_prompt_text(body) == "system: be brief\n\nuser: hi"


def test_extract_prompt_text_completions():
    assert _extract_prompt_text({"prompt": "once upon"}) == "once upon"


def test_extract_prompt_text_multimodal_collapses_image():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    out = _extract_prompt_text(body)
    assert "what is this?" in out
    assert "[image]" in out
    assert "AAAA" not in out  # base64 never persisted


def test_render_content_plain_and_none():
    assert _render_content("hello") == "hello"
    assert _render_content(None) == ""


def test_extract_response_text_chat_and_completion():
    chat = {"choices": [{"message": {"role": "assistant", "content": "yo"}}]}
    assert _extract_response_text(chat) == "yo"
    comp = {"choices": [{"text": "tail"}]}
    assert _extract_response_text(comp) == "tail"
    assert _extract_response_text({"choices": []}) == ""


def test_parse_sse_data():
    assert _parse_sse_data(b"data: {\"a\": 1}") == {"a": 1}
    assert _parse_sse_data(b": keepalive") is None
    assert _parse_sse_data(b"data: [DONE]") is None
    assert _parse_sse_data(b"data: not json") is None


# --------------------------------------------------------------------------
# Streaming capture + usage-chunk filtering
# --------------------------------------------------------------------------

async def _drain(chunks, *, strip):
    parts: list[str] = []
    usage: dict = {}

    async def upstream():
        for c in chunks:
            yield c

    out = b""
    async for b in _proxy_and_capture(
        upstream(), strip_usage_only=strip,
        response_parts=parts, usage_holder=usage,
    ):
        out += b
    return out, parts, usage


_STREAM = [
    b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
    b'data: [DONE]\n\n',
]


def test_proxy_strips_injected_usage_chunk_and_captures():
    out, parts, usage = asyncio.run(_drain(_STREAM, strip=True))
    # Content forwarded, response captured, usage captured.
    assert "".join(parts) == "Hello world"
    assert usage == {"prompt_tokens": 5, "completion_tokens": 2}
    # The usage-only event is gone from the client stream...
    assert b'"usage"' not in out
    # ...but the real content and [DONE] are intact.
    assert b'Hello' in out and b' world' in out
    assert out.rstrip().endswith(b"data: [DONE]")


def test_proxy_keeps_usage_chunk_when_client_asked():
    out, parts, usage = asyncio.run(_drain(_STREAM, strip=False))
    assert usage == {"prompt_tokens": 5, "completion_tokens": 2}
    # Client requested usage themselves -> forward it unchanged.
    assert b'"usage"' in out


def test_proxy_preserves_multibyte_split_across_chunks():
    # The two UTF-8 bytes of 'é' (0xC3 0xA9) land in different chunks.
    a = 'data: {"choices":[{"delta":{"content":"hé'.encode("utf-8")[:-1]
    tail = 'éllo"}}]}\n\n'.encode("utf-8")[1:]
    chunks = [a, tail, b'data: [DONE]\n\n']
    out, parts, _ = asyncio.run(_drain(chunks, strip=True))
    assert "".join(parts) == "héllo"
    assert "héllo" in out.decode("utf-8")


def test_proxy_buffers_event_split_across_chunks():
    chunks = [
        b'data: {"choices":[{"delta":{"con',
        b'tent":"hi"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    out, parts, _ = asyncio.run(_drain(chunks, strip=True))
    assert "".join(parts) == "hi"
    assert b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n' in out


# --------------------------------------------------------------------------
# Persistence: mark_in_flight_done stores + clips
# --------------------------------------------------------------------------

def test_mark_in_flight_done_persists_prompt_and_response(cfg):
    from llamanager.auth import Origin
    from llamanager.db import DB
    from llamanager.queue_mgr import QueueManager, QueuedRequest, _MAX_STORED_TEXT
    from llamanager.server_manager import ServerManager

    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)
        origin = Origin(id=1, name="t", priority=50,
                        allowed_models=["*"], is_admin=False, created_at=0.0)
        qr = QueuedRequest(request_id="rid-1", origin=origin, priority=50,
                           model_required="m", enqueued_at=0.0, seq=0)
        db.insert_request(request_id="rid-1", origin_id=None,
                          model="m", priority=50)
        qm._in_flight["rid-1"] = qr
        qm.mark_in_flight_done(
            qr, error=None, cancelled=False,
            prompt_tokens=3, completion_tokens=4,
            prompt_text="the prompt", response_text="x" * (_MAX_STORED_TEXT + 50),
        )
        row = db.query_one(
            "SELECT prompt_text, response_text, status FROM requests WHERE id=?",
            ("rid-1",),
        )
        assert row["status"] == "done"
        assert row["prompt_text"] == "the prompt"
        # Response was clipped to the cap + a truncation marker.
        assert len(row["response_text"]) <= _MAX_STORED_TEXT + 64
        assert "truncated" in row["response_text"]
    finally:
        db.close()


# --------------------------------------------------------------------------
# localdt filter
# --------------------------------------------------------------------------

def test_localdt_filter():
    assert _localdt(0) == ""        # falsy -> empty
    assert _localdt(None) == ""
    assert _localdt("nope") == ""   # invalid -> empty
    # A real timestamp renders HH:MM somewhere.
    assert re.search(r"\d{2}:\d{2}", _localdt(1_700_000_000))


# --------------------------------------------------------------------------
# Routes (dashboard origin name + request detail)
# --------------------------------------------------------------------------

def _admin_client(app) -> TestClient:
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    key = am.rotate_key(boot.id)
    client = TestClient(app)
    r = client.post("/ui/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303 and COOKIE_NAME in r.headers.get("set-cookie", "")
    return client


def _insert_completed_request(app, **fields):
    db = app.state.db
    boot = app.state.auth.get_origin_by_name("bootstrap")
    rid = fields.get("id", "req-test-1")
    db.insert_request(request_id=rid, origin_id=boot.id,
                      model="test/model.gguf", priority=50)
    db.update_request_status(
        rid, "done",
        started_at=1_700_000_000.0, finished_at=1_700_000_002.0,
        prompt_tokens=fields.get("prompt_tokens", 10),
        completion_tokens=fields.get("completion_tokens", 20),
        prompt_text=fields.get("prompt_text", "user: hello there"),
        response_text=fields.get("response_text", "general kenobi"),
    )
    return rid, boot.name


def test_dashboard_partial_shows_origin_name_not_id(app):
    with _admin_client(app) as client:
        _, name = _insert_completed_request(app)
        body = client.get("/ui/_partials/dashboard").text
        assert name in body                 # the readable origin name
        assert "#%s" % app.state.auth.get_origin_by_name(name).id not in body
        # tok/s column renders a number now that completion_tokens is present.
        assert "tok/s" in body


def test_request_detail_shows_prompt_and_response(app):
    with _admin_client(app) as client:
        rid, _ = _insert_completed_request(
            app, prompt_text="explain recursion", response_text="see recursion")
        body = client.get(f"/ui/requests/{rid}").text
        assert "explain recursion" in body
        assert "see recursion" in body
        assert "Prompt" in body and "Response" in body


def test_request_detail_unknown_id(app):
    with _admin_client(app) as client:
        body = client.get("/ui/requests/does-not-exist").text
        assert "no longer available" in body


def test_request_detail_requires_auth(app):
    client = TestClient(app)
    r = client.get("/ui/requests/whatever", follow_redirects=False)
    assert r.status_code in (302, 303, 401, 403)


# --------------------------------------------------------------------------
# Live (in-flight) view — prompt + partial response while still running
# --------------------------------------------------------------------------

def _live_request(app, rid="live-1", *, partial=("Par", "tial ", "answer")):
    from llamanager.auth import Origin
    from llamanager.queue_mgr import QueuedRequest
    qm = app.state.queue
    boot = app.state.auth.get_origin_by_name("bootstrap")
    origin = Origin(id=boot.id, name=boot.name, priority=50,
                    allowed_models=["*"], is_admin=True, created_at=0.0)
    qr = QueuedRequest(request_id=rid, origin=origin, priority=50,
                       model_required="test/model.gguf",
                       enqueued_at=time.time(), seq=0)
    qr.status = "running"
    qr.started_at = time.time()
    qr.prompt_text = "user: stream please"
    qr.response_parts.extend(partial)
    qm._by_id[rid] = qr
    return rid


def test_request_detail_live_streams_partial(app):
    rid = _live_request(app)
    with _admin_client(app) as client:
        body = client.get(f"/ui/requests/{rid}").text
        assert "user: stream please" in body      # prompt visible mid-flight
        assert "Partial answer" in body            # partial response so far
        assert "streaming" in body                 # live indicator
        assert 'hx-trigger="every 1500ms"' in body  # self-refresh wired


def test_request_detail_live_respects_zero_retention(app):
    rid = _live_request(app, rid="live-2")
    app.state.cfg.conversation_retention_days = 0
    with _admin_client(app) as client:
        body = client.get(f"/ui/requests/{rid}").text
        assert "stream please" not in body         # content suppressed
        assert "hx-trigger" not in body            # no polling either
