"""MCP (Model Context Protocol) surface for llamanager.

Why this exists: an agent that can *talk* to a local model still cannot
*drive the box* — load a model, watch VRAM, pull weights, start a
generation. That is what llamanager knows how to do, and MCP is the
protocol every host (Claude Code, Claude Desktop, Cursor, VS Code, and
ChatGPT through OpenAI's tunnel) already speaks.

The server is mounted on the daemon's own FastAPI app at ``/mcp`` over
Streamable HTTP, so there is exactly one process holding the GPU, the
queue and the DB. The ``mcp-stdio`` CLI verb is a thin protocol proxy to
this endpoint, not a second copy of the app (see ``mcp_stdio.py``).

Two rules shape everything below:

* **Every request is authenticated.** There is no anonymous mode and no
  "disabled" state that silently opens the door. The ASGI wrapper in
  :func:`mount_mcp` rejects anything without a valid origin bearer key
  before the MCP machinery sees it, and each tool re-resolves the caller
  so admin-only verbs can name the origin they refused.
* **Nothing here reimplements a route.** Management tools call the same
  internals ``/admin`` calls; work-submitting tools (inference, image,
  video, transcription) go back through the real ``/v1`` routes over an
  in-process ASGI transport so every gate — intake, model allowlists,
  mem_guard, queue accounting, gallery attribution — applies exactly
  once and exactly as it does for an HTTP caller.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from .auth import AuthManager, Origin
from .intake import is_accepting

log = logging.getLogger("llamanager.mcp")

# The daemon talks to itself over this base URL; the host is arbitrary
# (ASGITransport never opens a socket) but must be a valid absolute URL.
_INTERNAL_BASE = "http://llamanager.internal"

# Generations can legitimately run for twenty minutes (the Krea 2 cap), so
# the in-process client must not impose a shorter deadline than the engine.
_NO_TIMEOUT = httpx.Timeout(None)


# ---------------------------------------------------------------- auth ----

def _bearer_from_headers(headers: dict[str, str] | None) -> str | None:
    if not headers:
        return None
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    if not raw.lower().startswith("bearer "):
        return None
    return raw.split(" ", 1)[1].strip() or None


async def _caller(app, ctx: Context) -> tuple[Origin, str]:
    """Resolve the verified origin behind this tool call, plus its raw key.

    The ASGI wrapper already rejected unauthenticated requests, so this is
    a cache hit in practice (``AuthManager.verify`` keeps a positive cache
    keyed on the raw key). Reading the header here rather than threading a
    ContextVar through the session manager keeps the identity correct no
    matter which task the transport dispatches the call on.
    """
    key = _bearer_from_headers(ctx.headers)
    if not key:
        raise ToolError(
            "no bearer token on this MCP request; llamanager requires an "
            "origin API key (mint one at /ui/connect)"
        )
    am: AuthManager = app.state.auth
    origin = await am.verify(key)
    if origin is None:
        raise ToolError("invalid api key")
    return origin, key


def _require_admin(origin: Origin, tool: str) -> None:
    if not origin.is_admin:
        raise ToolError(
            f"tool '{tool}' needs an admin origin; '{origin.name}' is not one. "
            f"Mint an admin key at /ui/connect and reconnect."
        )


# ------------------------------------------------------- ASGI back-door ----

def _client_for(app) -> httpx.AsyncClient:
    """A cached in-process client that speaks to this very app.

    Not a network hop: ``ASGITransport`` calls the ASGI callable directly.
    It exists so the work-submitting tools reuse the real ``/v1`` handlers
    (and every gate inside them) instead of a parallel implementation.
    """
    existing = getattr(app.state, "mcp_http", None)
    if existing is not None:
        return existing
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=_INTERNAL_BASE,
        timeout=_NO_TIMEOUT,
    )
    app.state.mcp_http = client
    return client


async def call_v1(app, key: str, path: str, *, json_body: dict[str, Any] | None = None,
                  files: Any = None, data: Any = None) -> httpx.Response:
    """POST to one of this app's own ``/v1`` routes as the calling origin."""
    client = _client_for(app)
    headers = {"authorization": f"Bearer {key}"}
    return await client.post(path, headers=headers, json=json_body,
                             files=files, data=data)


def _detail_of(resp: httpx.Response) -> str:
    """Best-effort human message from a FastAPI error response."""
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 — non-JSON error bodies are still useful
        return resp.text[:400]
    if isinstance(body, dict):
        if isinstance(body.get("detail"), str):
            return body["detail"]
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
    return json.dumps(body)[:400]


def raise_for_v1(resp: httpx.Response, what: str) -> None:
    if resp.status_code >= 400:
        raise ToolError(f"{what} failed ({resp.status_code}): {_detail_of(resp)}")


# ------------------------------------------------------------- server ----

def build_mcp(app) -> MCPServer:
    """Construct the MCP server and register every tool against ``app``."""
    mcp = MCPServer(
        name="llamanager",
        title="llamanager",
        instructions=(
            "Drive a local llamanager daemon: inspect and load models, watch "
            "GPU and RAM pressure, pull weights, run local text inference, "
            "generate images and video, and transcribe audio. Image and video "
            "generation are long-running: submit returns a job id, then poll "
            "get_generation_job. Management tools require an admin origin key."
        ),
    )
    _register_management_tools(mcp, app)
    _register_generation_tools(mcp, app)
    _register_inference_tools(mcp, app)
    return mcp


def _register_management_tools(mcp: MCPServer, app) -> None:
    from .config import ENGINE_FAMILY, detect_engine_for_id, model_role
    from . import mem_guard as mg

    @mcp.tool(
        name="list_models",
        description=(
            "List every model in the llamanager registry with its family "
            "(text/image/video/audio), engine, role and size. Use this before "
            "load_model or any generation tool to find valid model ids."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True,
                                    open_world_hint=False),
    )
    async def list_models(ctx: Context, family: str | None = None) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "list_models")
        reg = app.state.registry
        cfg = app.state.cfg
        out = []
        for m in reg.list():
            d = m.to_dict()
            engine = detect_engine_for_id(m.model_id, cfg.models_dir)
            d["engine"] = engine
            d["family"] = ENGINE_FAMILY.get(engine, "text")
            d["role"] = model_role(m.model_id)
            if family and d["family"] != family:
                continue
            out.append(d)
        return {"models": out, "count": len(out)}

    @mcp.tool(
        name="server_status",
        description=(
            "Snapshot of the whole daemon: running model slots, queue depth "
            "and in-flight requests, whether intake is accepting work, plus "
            "host RAM/swap, GPU VRAM per card and the current memory-pressure "
            "class. Call this before loading a large model."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def server_status(ctx: Context) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "server_status")
        from .api_admin import _active_downloads, _active_installs, slots_payload_for_app

        sm = app.state.sm
        qm = app.state.queue
        cfg = app.state.cfg

        base = sm.status()
        snap = qm.snapshot()
        state = mg.read_mem_state()
        pressure = mg.classify_pressure(state, mg.MemThresholds.from_cfg(cfg))
        gpus = [
            {
                "card": g.card,
                "vram_used_gb": round(g.vram_used, 2),
                "vram_total_gb": round(g.vram_total, 2),
                "gtt_used_gb": round(g.gtt_used, 2),
                "gtt_total_gb": round(g.gtt_total, 2),
            }
            for g in mg.read_gpu_mem()
        ]
        return {
            "runtime": base,
            "slots": slots_payload_for_app(app),
            "queue": {
                "depth": snap["depth"],
                "in_flight": snap["in_flight"],
                "in_flight_count": len(snap["in_flight"]),
                "paused": snap["paused"],
            },
            "accepting_requests": is_accepting(app),
            "active_downloads": _active_downloads(app),
            "active_installs": _active_installs(app),
            "memory": {
                "ram_total_gb": round(state.ram_total_gb, 2),
                "ram_available_gb": round(state.ram_available_gb, 2),
                "swap_total_gb": round(state.swap_total_gb, 2),
                "swap_used_gb": round(state.swap_used_gb, 2),
                "swap_io_mb_s": round(state.swap_io_mb_s, 2),
                "pressure": pressure.name,
            },
            "gpus": gpus,
        }

    @mcp.tool(
        name="load_model",
        description=(
            "Load (or hot-swap) a model into a llamanager slot. Slot 0 is the "
            "default and is the only slot when multi-slot is off. Check "
            "server_status first: loading a model larger than free VRAM will "
            "spill to host RAM or be refused."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                    idempotent_hint=True, open_world_hint=False),
    )
    async def load_model(ctx: Context, model: str | None = None,
                         profile: str | None = None,
                         slot_id: int = 0) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "load_model")
        from .api_admin import slots_payload_for_app
        from .server_manager import ServerError, resolve_spec

        cfg = app.state.cfg
        sm = app.state.sm
        try:
            spec = resolve_spec(cfg, model=model, profile=profile, args={})
        except (ServerError, ValueError) as e:
            raise ToolError(f"cannot resolve model spec: {e}")
        slot_sm = sm.slot(slot_id) if hasattr(sm, "slot") else None
        if slot_sm is None:
            raise ToolError(f"no such slot {slot_id}")
        try:
            if slot_sm.is_running:
                await sm.swap_in(slot_id, spec)
            else:
                await sm.start_slot(slot_id, spec)
        except ServerError as e:
            raise ToolError(f"load failed: {e}")
        return slots_payload_for_app(app)

    @mcp.tool(
        name="unload_model",
        description=(
            "Stop the model in a slot and free its VRAM. The model stays on "
            "disk and can be loaded again with load_model."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                    idempotent_hint=True, open_world_hint=False),
    )
    async def unload_model(ctx: Context, slot_id: int = 0) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "unload_model")
        from .api_admin import slots_payload_for_app
        try:
            await app.state.sm.stop_slot(slot_id)
        except Exception as e:  # noqa: BLE001 — surfaced as a tool error
            raise ToolError(f"unload failed: {e}")
        return slots_payload_for_app(app)

    @mcp.tool(
        name="pull_model",
        description=(
            "Start downloading model weights from Hugging Face. Accepts "
            "'org/name', 'hf://org/name' or a full URL. Choose exactly one "
            "of: files (specific filenames — the usual choice for GGUF), "
            "whole_repo (every file, for diffusers-style models where the "
            "layout matters), or subfolder (only that subtree). Returns a "
            "download_id immediately — poll get_download for progress. A "
            "daemon restart cancels an in-flight pull."
        ),
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=True),
    )
    async def pull_model(ctx: Context, source: str,
                         files: list[str] | None = None,
                         whole_repo: bool = False,
                         subfolder: str | None = None,
                         family: str = "text",
                         target_dir: str | None = None) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "pull_model")
        reg = app.state.registry
        src = source.strip()
        if not src.startswith(("http://", "https://", "hf://")):
            src = "hf://" + src.removeprefix("hf:")
        # Picking a mode for the caller would either fetch one file when they
        # wanted a pipeline, or pull hundreds of gigabytes when they wanted
        # one GGUF. Make them say which.
        chosen = [bool(files), bool(whole_repo), bool(subfolder)]
        if sum(chosen) != 1:
            raise ToolError(
                "choose exactly one of files, whole_repo or subfolder "
                f"(got files={files!r}, whole_repo={whole_repo}, "
                f"subfolder={subfolder!r}). Use files=['model.gguf'] for a "
                "single GGUF, whole_repo=true for a diffusers pipeline.")
        try:
            did = reg.start_pull(
                source=src, files=files, whole_repo=whole_repo,
                subfolder=(subfolder or "").strip().strip("/") or None,
                family=family,
                target_dir=(target_dir or "").strip().strip("/") or None)
        except Exception as e:  # noqa: BLE001 — bad source is a user error
            raise ToolError(f"pull failed: {e}")
        return {"download_id": did,
                "note": "poll get_download with this id for progress"}

    @mcp.tool(
        name="get_download",
        description=(
            "Progress of a weights download started by pull_model: status is "
            "pending, running, done, cancelled or failed."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True,
                                    open_world_hint=False),
    )
    async def get_download(ctx: Context, download_id: str) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "get_download")
        d = app.state.registry.get_download(download_id)
        if not d:
            raise ToolError(f"no download {download_id}")
        return d

    @mcp.tool(
        name="cancel_download",
        description="Cancel a running weights download. Partial files are kept.",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                    open_world_hint=False),
    )
    async def cancel_download(ctx: Context, download_id: str) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "cancel_download")
        if not app.state.registry.cancel_pull(download_id):
            raise ToolError(f"download {download_id} is not running")
        return {"ok": True, "download_id": download_id}

    @mcp.tool(
        name="queue_status",
        description=(
            "Pending and in-flight requests across all families, and whether "
            "the queue is paused."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def queue_status(ctx: Context) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "queue_status")
        return app.state.queue.snapshot()

    @mcp.tool(
        name="set_queue_paused",
        description=(
            "Pause or resume queue dispatch. Pausing lets in-flight work "
            "finish but holds everything queued behind it."
        ),
        annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True,
                                    open_world_hint=False),
    )
    async def set_queue_paused(ctx: Context, paused: bool) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "set_queue_paused")
        qm = app.state.queue
        if paused:
            qm.pause()
        else:
            await qm.resume()
        return {"paused": qm.snapshot()["paused"]}

    @mcp.tool(
        name="cancel_request",
        description=(
            "Cancel a queued or in-flight request by its request id (as shown "
            "by queue_status)."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                    open_world_hint=False),
    )
    async def cancel_request(ctx: Context, request_id: str) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        _require_admin(origin, "cancel_request")
        ok = app.state.queue.cancel(request_id)
        if not ok:
            raise ToolError(f"no cancellable request {request_id}")
        return {"ok": True, "request_id": request_id}


def _register_generation_tools(mcp: MCPServer, app) -> None:
    from .mcp_jobs import GenJobRegistry

    def _jobs() -> GenJobRegistry:
        return app.state.mcp_jobs

    @mcp.tool(
        name="generate_image",
        description=(
            "Start a local image generation. Returns a job_id immediately — "
            "generation can take minutes, so poll get_generation_job rather "
            "than waiting. Jobs are lost if the daemon restarts, but finished "
            "images remain in the gallery."
        ),
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
    )
    async def generate_image(ctx: Context, prompt: str, model: str | None = None,
                             size: str | None = None, n: int = 1,
                             seed: int | None = None,
                             profile: str | None = None) -> dict[str, Any]:
        origin, key = await _caller(app, ctx)
        body: dict[str, Any] = {"prompt": prompt, "n": n}
        if model:
            body["model"] = model
        if size:
            body["size"] = size
        if seed is not None:
            body["seed"] = seed
        if profile:
            body["profile"] = profile
        job = await _jobs().submit("image", body, origin=origin, key=key)
        return {"job_id": job.job_id, "status": job.status,
                "note": "poll get_generation_job with this job_id"}

    @mcp.tool(
        name="generate_video",
        description=(
            "Start a local video generation. Returns a job_id immediately; "
            "poll get_generation_job. Video runs are long (many minutes). "
            "Some models are image-to-video and REQUIRE an opening frame — "
            "pass one as image_path (a file on this machine) or image_base64; "
            "a text-only call to such a model is refused. Frame count and fps "
            "come from the model's profile, not from here."
        ),
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
    )
    async def generate_video(ctx: Context, prompt: str, model: str | None = None,
                             image_path: str | None = None,
                             image_base64: str | None = None,
                             size: str | None = None,
                             seed: int | None = None,
                             profile: str | None = None) -> dict[str, Any]:
        import base64 as _b64
        from pathlib import Path as _Path

        origin, key = await _caller(app, ctx)
        body: dict[str, Any] = {"prompt": prompt}
        if model:
            body["model"] = model
        if size:
            body["size"] = size
        if seed is not None:
            body["seed"] = seed
        if profile:
            body["profile"] = profile
        if image_path and image_base64:
            raise ToolError("give at most one of image_path or image_base64")
        if image_path:
            src = _Path(image_path).expanduser()
            if not src.is_file():
                raise ToolError(f"no such image file: {src}")
            body["image"] = _b64.b64encode(src.read_bytes()).decode("ascii")
        elif image_base64:
            body["image"] = image_base64
        job = await _jobs().submit("video", body, origin=origin, key=key)
        return {"job_id": job.job_id, "status": job.status,
                "note": "poll get_generation_job with this job_id"}

    # Two tools rather than one with a flag that changes the return type:
    # a tool whose shape depends on an argument has no output schema, so the
    # poll result would come back as unstructured text.
    @mcp.tool(
        name="get_generation_job",
        description=(
            "Status and result of an image or video job. While running it "
            "reports queue position and the current diffusion step. When done, "
            "result carries the file path and a URL for each output. Poll this "
            "after generate_image or generate_video."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def get_generation_job(ctx: Context, job_id: str) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        return await _jobs().describe(job_id, origin=origin)

    @mcp.tool(
        name="get_generation_image",
        description=(
            "The actual pixels of a finished image job, inline, so you can "
            "look at what was generated. Downscaled for viewing; the "
            "full-resolution file stays on disk at the path reported by "
            "get_generation_job. Stills only — video stays on disk."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def get_generation_image(ctx: Context, job_id: str) -> list[Any]:
        origin, _ = await _caller(app, ctx)
        return await _jobs().image_content(job_id, origin=origin)

    @mcp.tool(
        name="cancel_generation_job",
        description="Cancel a running image or video job and free the GPU.",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                    open_world_hint=False),
    )
    async def cancel_generation_job(ctx: Context, job_id: str) -> dict[str, Any]:
        origin, _ = await _caller(app, ctx)
        return await _jobs().cancel(job_id, origin=origin)


def _register_inference_tools(mcp: MCPServer, app) -> None:

    @mcp.tool(
        name="ask_local_model",
        description=(
            "Run a prompt through a model on this machine and return the "
            "completion. Use it to delegate work to the local model — "
            "private data, bulk work, or a model with capabilities you lack. "
            "If no model is loaded, llamanager loads the default one first, "
            "which can take a while."
        ),
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
    )
    async def ask_local_model(ctx: Context, prompt: str, model: str | None = None,
                              system: str | None = None,
                              max_tokens: int = 1024,
                              temperature: float | None = None) -> dict[str, Any]:
        origin, key = await _caller(app, ctx)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        await ctx.report_progress(0.0, None, "queued on llamanager")
        resp = await call_v1(app, key, "/v1/chat/completions", json_body=body)
        raise_for_v1(resp, "local inference")
        payload = resp.json()
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        out: dict[str, Any] = {
            "text": text,
            "model": payload.get("model"),
            "usage": payload.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        }
        note = resp.headers.get("x-llamanager-model-fallback")
        if note:
            # The route served a different model than asked for; say so rather
            # than letting the caller believe it got what it requested.
            out["model_substituted"] = note
        if not text.strip() and choice.get("finish_reason") == "length":
            # A reasoning model can spend the whole budget thinking and return
            # no visible answer. An empty string reads as "it had nothing to
            # say", which would send the caller down the wrong path.
            out["warning"] = (
                f"the model hit the {max_tokens}-token cap before producing "
                f"any answer text (reasoning models spend budget thinking "
                f"first) — retry with a larger max_tokens")
        return out

    @mcp.tool(
        name="transcribe_audio",
        description=(
            "Transcribe an audio file with a local ASR model. Give exactly one "
            "of file_path (a path on this machine) or audio_base64. The model "
            "id is required — call list_models with family='audio' to find one."
        ),
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
    )
    async def transcribe_audio(ctx: Context, model: str,
                               file_path: str | None = None,
                               audio_base64: str | None = None,
                               filename: str | None = None,
                               language: str | None = None,
                               response_format: str = "json") -> dict[str, Any]:
        import base64
        from pathlib import Path

        origin, key = await _caller(app, ctx)
        if bool(file_path) == bool(audio_base64):
            # Guessing here would either read a file the caller did not mean
            # or silently ignore the bytes they sent.
            raise ToolError(
                "give exactly one of file_path or audio_base64 "
                f"(got file_path={'set' if file_path else 'unset'}, "
                f"audio_base64={'set' if audio_base64 else 'unset'})")
        if file_path:
            p = Path(file_path).expanduser()
            if not p.is_file():
                raise ToolError(f"no such audio file: {p}")
            raw = p.read_bytes()
            name = p.name
        else:
            try:
                raw = base64.b64decode(audio_base64, validate=True)
            except Exception as e:  # noqa: BLE001 — malformed input
                raise ToolError(f"audio_base64 is not valid base64: {e}")
            if not filename:
                raise ToolError("filename is required with audio_base64 "
                                "(the ASR route needs the extension)")
            name = filename
        form = {"model": model, "response_format": response_format}
        if language:
            form["language"] = language
        resp = await call_v1(app, key, "/v1/audio/transcriptions",
                             files={"file": (name, raw)}, data=form)
        raise_for_v1(resp, "transcription")
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 — text/plain response formats
            return {"text": resp.text}


# -------------------------------------------------------------- mount ----

def _loopback_hosts(cfg) -> list[str]:
    """Hosts the MCP endpoint will answer to, for DNS-rebinding protection."""
    port = int(getattr(cfg, "port", 7200))
    hosts = ["127.0.0.1", "localhost", "[::1]"]
    out: list[str] = []
    for h in hosts:
        out.extend([h, f"{h}:{port}"])
    bind = str(getattr(cfg, "bind", "") or "")
    if bind and bind not in ("127.0.0.1", "0.0.0.0", "::"):
        out.extend([bind, f"{bind}:{port}"])
    return out


def mount_mcp(app) -> MCPServer:
    """Build the MCP server, guard it, and mount it at ``/mcp``.

    The guard is a plain ASGI callable rather than FastAPI middleware so it
    covers this sub-app and nothing else. It answers with JSON errors that
    match the rest of the API (``detail``) so a misconfigured client sees a
    sentence, not an empty 500.
    """
    from .mcp_jobs import GenJobRegistry

    cfg = app.state.cfg
    mcp = build_mcp(app)
    app.state.mcp = mcp
    app.state.mcp_jobs = GenJobRegistry(app)

    hosts = _loopback_hosts(cfg)
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[f"http://{h}" for h in hosts] + [f"https://{h}" for h in hosts],
    )
    inner = mcp.streamable_http_app(
        streamable_http_path="/",
        # Stateless: a daemon restart must not strand a client holding a
        # session id, and several hosts may connect at once.
        stateless_http=True,
        transport_security=security,
    )

    async def guarded(scope, receive, send):
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        key = _bearer_from_headers(headers)
        if not key:
            await _json_error(send, 401, "missing bearer token")
            return
        am: AuthManager = app.state.auth
        origin = await am.verify(key)
        if origin is None:
            log.warning("mcp: rejected an unknown api key from %s",
                        (scope.get("client") or ("?",))[0])
            await _json_error(send, 401, "invalid api key")
            return
        if not origin.enabled:
            await _json_error(
                send, 403,
                f"origin '{origin.name}' is disabled and may not submit "
                f"requests; ask an administrator to re-enable it.")
            return
        from .api_v1 import require_local_origin_is_local
        from fastapi import HTTPException
        peer = scope.get("client")
        try:
            require_local_origin_is_local(origin, peer[0] if peer else None)
        except HTTPException as e:
            await _json_error(send, e.status_code, str(e.detail))
            return
        await inner(scope, receive, send)

    app.mount("/mcp", guarded)
    return mcp


async def _json_error(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii"))],
    })
    await send({"type": "http.response.body", "body": body})
