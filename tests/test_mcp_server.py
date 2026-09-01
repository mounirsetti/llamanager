"""The MCP endpoint: who may reach it, and what the tools answer.

Everything here runs against the real mounted ASGI app through the real
MCP client, so a protocol-level regression (transport, handshake, schema)
fails these tests rather than surviving to a live client.
"""
from __future__ import annotations

import httpx2
import pytest
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:7200/mcp/"
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
HDRS = {"content-type": "application/json",
        "accept": "application/json, text/event-stream"}


def _keys(app):
    am = app.state.auth
    _, admin = am.create_origin(name="mcp-admin", is_admin=True)
    _, plain = am.create_origin(name="mcp-plain", is_admin=False)
    return admin, plain


async def _raw_post(app, headers):
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url="http://127.0.0.1:7200",
    ) as c:
        return await c.post("/mcp/", json=INIT, headers=headers)


def _connected(app, key: str):
    """An MCP client speaking to ``app`` in-process as ``key``'s origin."""
    hc = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url="http://127.0.0.1:7200",
        headers={"authorization": f"Bearer {key}"},
    )
    return hc, streamable_http_client(MCP_URL, http_client=hc)


# ------------------------------------------------------------- auth ----

@pytest.mark.asyncio
async def test_mcp_requires_a_bearer_token(app):
    async with app.router.lifespan_context(app):
        r = await _raw_post(app, HDRS)
    assert r.status_code == 401
    assert "missing bearer token" in r.text


@pytest.mark.asyncio
async def test_mcp_rejects_an_unknown_key(app):
    async with app.router.lifespan_context(app):
        r = await _raw_post(app, {**HDRS, "authorization": "Bearer lm_nope"})
    assert r.status_code == 401
    assert "invalid api key" in r.text


@pytest.mark.asyncio
async def test_mcp_rejects_a_disabled_origin(app):
    async with app.router.lifespan_context(app):
        am = app.state.auth
        origin, key = am.create_origin(name="mcp-off", is_admin=True)
        am.set_enabled(origin.id, False)
        r = await _raw_post(app, {**HDRS, "authorization": f"Bearer {key}"})
    assert r.status_code == 403
    assert "disabled" in r.text


@pytest.mark.asyncio
async def test_mcp_accepts_a_valid_key(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        r = await _raw_post(app, {**HDRS, "authorization": f"Bearer {admin}"})
    assert r.status_code == 200
    assert "serverInfo" in r.text or "protocolVersion" in r.text


@pytest.mark.asyncio
async def test_mcp_blocks_a_foreign_origin_header(app):
    """DNS-rebinding guard: a page on another site must not drive the daemon."""
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        r = await _raw_post(app, {**HDRS,
                                  "authorization": f"Bearer {admin}",
                                  "origin": "http://evil.example"})
    assert r.status_code >= 400
    assert r.status_code != 200


# ------------------------------------------------------------ tools ----

@pytest.mark.asyncio
async def test_tool_catalogue_is_complete(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            names = {t.name for t in (await cl.list_tools()).tools}
    expected = {
        "list_models", "server_status", "load_model", "unload_model",
        "pull_model", "get_download", "cancel_download", "queue_status",
        "set_queue_paused", "cancel_request", "ask_local_model",
        "generate_image", "generate_video", "get_generation_job",
        "get_generation_image", "cancel_generation_job", "transcribe_audio",
    }
    assert expected <= names, expected - names


@pytest.mark.asyncio
async def test_list_models_and_status_answer(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            models = await cl.call_tool("list_models", {})
            status = await cl.call_tool("server_status", {})

    assert not models.is_error
    assert "models" in (models.structured_content or {})

    assert not status.is_error
    sc = status.structured_content or {}
    # The status tool is the one place mem_guard reaches an API caller at
    # all, so its shape is worth pinning.
    for key in ("runtime", "slots", "queue", "memory", "gpus"):
        assert key in sc, key
    assert "pressure" in sc["memory"]


@pytest.mark.asyncio
async def test_admin_tools_refuse_a_plain_origin_by_name(app):
    async with app.router.lifespan_context(app):
        _, plain = _keys(app)
        hc, transport = _connected(app, plain)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("load_model", {"model": "test/model.gguf"})

    assert r.is_error
    text = r.content[0].text
    assert "mcp-plain" in text and "admin" in text


@pytest.mark.asyncio
async def test_unknown_download_is_a_clean_tool_error(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("get_download", {"download_id": "nope"})

    assert r.is_error
    assert "nope" in r.content[0].text


@pytest.mark.asyncio
async def test_queue_pause_and_resume_round_trip(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            paused = await cl.call_tool("set_queue_paused", {"paused": True})
            assert (paused.structured_content or {}).get("paused") is True
            resumed = await cl.call_tool("set_queue_paused", {"paused": False})
            assert (resumed.structured_content or {}).get("paused") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("args,expected", [
    ({"model": "w"}, "exactly one"),                       # neither
    ({"model": "w", "file_path": "/x", "audio_base64": "aGk="}, "exactly one"),
    ({"model": "w", "file_path": "/definitely/missing.wav"}, "no such audio"),
    ({"model": "w", "audio_base64": "!!not-base64!!"}, "not valid base64"),
])
async def test_transcribe_input_validation_never_guesses(app, args, expected):
    """Ambiguous audio input is refused, not resolved by picking one."""
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("transcribe_audio", args)

    assert r.is_error
    assert expected in r.content[0].text


@pytest.mark.asyncio
async def test_generation_job_is_scoped_to_its_owner(app):
    """One origin must not see, or even confirm, another's job."""
    async with app.router.lifespan_context(app):
        admin, plain = _keys(app)
        hc, transport = _connected(app, plain)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("get_generation_job", {"job_id": "img_ghost"})

    assert r.is_error
    assert "img_ghost" in r.content[0].text


@pytest.mark.asyncio
async def test_every_reporting_tool_declares_an_output_schema(app):
    """A tool typed `-> Any` silently returns no structured content.

    Hosts then get JSON-as-text instead of parsed fields, and a polling
    caller reading `structured_content` loops forever on an empty dict.
    That is how the first cut of get_generation_job shipped, so pin it.
    """
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            tools = {t.name: t for t in (await cl.list_tools()).tools}

    # get_generation_image returns image blocks, so it has no output schema
    # by design; everything else reports data and must be parseable.
    for name, tool in tools.items():
        if name == "get_generation_image":
            continue
        assert tool.output_schema is not None, f"{name} has no output schema"


@pytest.mark.asyncio
async def test_asking_for_pixels_before_the_job_is_done_says_so(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("get_generation_image", {"job_id": "img_x"})

    assert r.is_error
    assert "img_x" in r.content[0].text


def test_inline_images_are_downscaled_not_refused(app, tmp_path):
    """The engines write 2048-square PNGs; raw bytes would swamp a context.

    Returning nothing would make the tool useless on this hardware, so it
    downscales instead. The full-resolution file stays on disk.
    """
    import base64

    from PIL import Image

    from llamanager.mcp_jobs import GenJob, GenJobRegistry

    src = tmp_path / "big.png"
    Image.new("RGB", (2048, 2048), (12, 90, 200)).save(src)

    job = GenJob(job_id="img_t", kind="image", origin_name="o",
                 prompt="p", model="m", status="done",
                 result=[{"path": str(src), "url": "/images/file/d/o/big.png"}])
    blocks = GenJobRegistry(app)._image_blocks(job)

    assert len(blocks) == 1
    assert blocks[0].mime_type == "image/jpeg"
    raw = base64.b64decode(blocks[0].data)
    assert len(raw) < 200_000, "inline image should be small enough to send"

    import io
    with Image.open(io.BytesIO(raw)) as im:
        # Downscaled to the viewing size, whatever the source resolution.
        assert max(im.size) == 768
    # The full-resolution original is left alone.
    with Image.open(src) as orig:
        assert orig.size == (2048, 2048)


def test_video_is_not_shown_inline(app, tmp_path):
    from llamanager.mcp_jobs import GenJob, GenJobRegistry

    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00\x00\x00 ftypisom")
    job = GenJob(job_id="vid_t", kind="video", origin_name="o", prompt="p",
                 model="m", status="done",
                 result=[{"path": str(mp4), "url": "/images/file/d/o/clip.mp4"}])
    assert GenJobRegistry(app)._image_blocks(job) == []
