"""The MCP endpoint: who may reach it, and what the tools answer.

Everything here runs against the real mounted ASGI app through the real
MCP client, so a protocol-level regression (transport, handshake, schema)
fails these tests rather than surviving to a live client.
"""
from __future__ import annotations

import asyncio

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


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"source": "openai/whisper-tiny"},                       # no mode
    {"source": "o/r", "whole_repo": True, "files": ["a.gguf"]},  # two modes
    {"source": "o/r", "whole_repo": True, "subfolder": "d"},     # two modes
])
async def test_pull_model_requires_exactly_one_mode(app, args):
    """Guessing costs either the wrong file or hundreds of gigabytes.

    The registry rejects a bare HF source, so a tool that only offered
    `files` could never pull a diffusers pipeline at all.
    """
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("pull_model", args)

    assert r.is_error
    assert "exactly one of files, whole_repo or subfolder" in r.content[0].text


@pytest.mark.asyncio
async def test_pull_model_passes_each_mode_through(app):
    """Each mode reaches Registry.start_pull as the mode the caller chose."""
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        seen = []
        app.state.registry.start_pull = lambda **kw: (seen.append(kw), "dl1")[1]

        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            await cl.call_tool("pull_model",
                               {"source": "org/repo", "files": ["m.gguf"]})
            await cl.call_tool("pull_model",
                               {"source": "org/repo", "whole_repo": True,
                                "family": "image"})
            await cl.call_tool("pull_model",
                               {"source": "org/repo", "subfolder": "diffusers"})

    assert seen[0]["files"] == ["m.gguf"] and not seen[0]["whole_repo"]
    assert seen[1]["whole_repo"] is True and seen[1]["family"] == "image"
    assert seen[2]["subfolder"] == "diffusers"
    # A bare name is normalised to an hf:// source, as the admin route does.
    assert all(k["source"] == "hf://org/repo" for k in seen)


@pytest.mark.asyncio
async def test_a_cancelled_job_is_not_relabelled_failed(app, monkeypatch):
    """The engine's teardown error must not overwrite the user's cancel.

    Cancelling tears the engine down mid-step, so the route can answer 502
    rather than 499. Reporting that as "failed" sends the caller hunting a
    bug that is really their own cancel.
    """
    import httpx

    from llamanager import mcp_server
    from llamanager.mcp_jobs import GenJob, GenJobRegistry

    reg = GenJobRegistry(app)
    job = GenJob(job_id="img_c", kind="image", origin_name="o",
                 prompt="p", model="m")

    async def _cancelled_mid_flight(*a, **kw):
        # What really happens: the run is in flight, the user cancels, and
        # only then does the route answer.
        job.status = "cancelled"
        return httpx.Response(502, json={"detail": "engine torn down"})

    monkeypatch.setattr(mcp_server, "call_v1", _cancelled_mid_flight)

    await reg._run(job, "/v1/images/generations", {}, "key")
    assert job.status == "cancelled", "a late error overwrote the cancel"


@pytest.mark.asyncio
async def test_a_cancel_landing_before_the_task_starts_also_holds(app, monkeypatch):
    """A cancel between submit and the task's first slice must stick too."""
    import httpx

    from llamanager import mcp_server
    from llamanager.mcp_jobs import GenJob, GenJobRegistry

    async def _late_502(*a, **kw):
        return httpx.Response(502, json={"detail": "engine torn down"})

    monkeypatch.setattr(mcp_server, "call_v1", _late_502)

    reg = GenJobRegistry(app)
    job = GenJob(job_id="img_c2", kind="image", origin_name="o",
                 prompt="p", model="m", status="cancelled")
    await reg._run(job, "/v1/images/generations", {}, "key")
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_a_genuine_failure_is_still_reported(app, monkeypatch):
    """The guard above must not swallow real failures."""
    import httpx

    from llamanager import mcp_server
    from llamanager.mcp_jobs import GenJob, GenJobRegistry

    async def _fails(*a, **kw):
        return httpx.Response(502, json={"detail": "no such model"})

    monkeypatch.setattr(mcp_server, "call_v1", _fails)

    reg = GenJobRegistry(app)
    job = GenJob(job_id="img_f", kind="image", origin_name="o",
                 prompt="p", model="m")
    await reg._run(job, "/v1/images/generations", {}, "key")
    assert job.status == "failed"
    assert "no such model" in (job.error or "")


@pytest.mark.asyncio
async def test_video_rejects_two_opening_frames(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("generate_video",
                                   {"prompt": "p", "image_path": "/a",
                                    "image_base64": "aGk="})
    assert r.is_error
    assert "at most one" in r.content[0].text


@pytest.mark.asyncio
async def test_video_reports_a_missing_opening_frame_file(app):
    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("generate_video",
                                   {"prompt": "p",
                                    "image_path": "/definitely/missing.png"})
    assert r.is_error
    assert "no such image file" in r.content[0].text


@pytest.mark.asyncio
async def test_video_sends_the_opening_frame_as_base64(app, tmp_path):
    """The only installed video model here is image-to-video, so a tool
    that cannot carry an opening frame cannot drive it at all."""
    import base64

    from PIL import Image

    from llamanager import mcp_server

    frame = tmp_path / "first.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(frame)

    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        sent = {}

        async def _capture(app_, key, path, *, json_body=None, **kw):
            sent.update(path=path, body=json_body)
            import httpx
            return httpx.Response(200, json={"data": []},
                                  headers={"x-llamanager-request-id": "r1"})

        mcp_server.call_v1, original = _capture, mcp_server.call_v1
        try:
            hc, transport = _connected(app, admin)
            async with hc, Client(transport) as cl:
                r = await cl.call_tool("generate_video",
                                       {"prompt": "p",
                                        "image_path": str(frame)})
                assert not r.is_error
                await asyncio.sleep(0.2)   # let the job task run
        finally:
            mcp_server.call_v1 = original

    assert sent["path"] == "/v1/videos/generations"
    assert sent["body"]["image"] == base64.b64encode(
        frame.read_bytes()).decode("ascii")


@pytest.mark.asyncio
async def test_an_all_thinking_answer_is_flagged_not_returned_blank(app, monkeypatch):
    """A reasoning model can burn the whole cap thinking and answer nothing.

    Returning "" reads as 'the model had nothing to say', which sends the
    caller down the wrong path; say what actually happened.
    """
    import httpx

    from llamanager import mcp_server

    async def _capped(app_, key, path, *, json_body=None, **kw):
        return httpx.Response(200, json={
            "model": "m", "usage": {},
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        })

    monkeypatch.setattr(mcp_server, "call_v1", _capped)

    async with app.router.lifespan_context(app):
        admin, _ = _keys(app)
        hc, transport = _connected(app, admin)
        async with hc, Client(transport) as cl:
            r = await cl.call_tool("ask_local_model",
                                   {"prompt": "hi", "max_tokens": 24})

    sc = r.structured_content or {}
    assert sc.get("text") == ""
    assert "24-token cap" in (sc.get("warning") or "")
