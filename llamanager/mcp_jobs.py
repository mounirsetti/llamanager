"""Submit-and-poll layer for image and video generation over MCP.

The ``/v1`` generation routes are synchronous by design: they hold the
connection for the whole run, which is right for an HTTP client that can
wait. An MCP host cannot — a Krea 2 run is capped at twenty minutes and
every host will have given up on the tool call long before that.

So the MCP generation tools hand the work to this registry, which owns
the long-lived request and returns a job id immediately. The shape is
deliberately the same as llamanager's weights downloads (start returns an
id, poll for ``pending|running|done|cancelled|failed``), because that is
the create-and-poll pattern the rest of the daemon already uses.

Jobs live in memory only. A daemon restart loses them, exactly as it
loses in-flight downloads; finished images are already in the gallery and
survive regardless. The tool descriptions say so.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .auth import Origin

log = logging.getLogger("llamanager.mcp.jobs")

# Keep finished jobs around long enough for a client to poll the result,
# without growing without bound in a daemon that runs for weeks.
_MAX_FINISHED = 200

#: Long edge of an inline image handed back by ``get_generation_image``.
#: Big enough to judge composition and detail, small enough that a couple
#: of them do not dominate the caller's context.
_VIEW_PX = 768

#: Only these can be shown inline; video stays on disk.
_STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class GenJob:
    job_id: str
    kind: str                      # "image" | "video"
    origin_name: str
    prompt: str
    model: str | None
    status: str = "pending"        # pending|running|done|cancelled|failed
    request_id: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)   # engine/seed/duration
    error: str | None = None
    task: asyncio.Task | None = None

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "prompt": self.prompt,
            "model": self.model,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "info": self.info,
            "error": self.error,
        }


class GenJobRegistry:
    """Tracks MCP-submitted generations for one daemon."""

    def __init__(self, app) -> None:
        self._app = app
        self._jobs: dict[str, GenJob] = {}
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------- submit ----

    async def submit(self, kind: str, body: dict[str, Any], *,
                     origin: Origin, key: str) -> GenJob:
        if kind not in ("image", "video"):
            raise ValueError(f"unknown generation kind {kind!r}")
        job_id = f"{'img' if kind == 'image' else 'vid'}_{uuid.uuid4().hex[:16]}"
        job = GenJob(
            job_id=job_id,
            kind=kind,
            origin_name=origin.name,
            prompt=str(body.get("prompt") or ""),
            model=body.get("model"),
        )
        async with self._lock:
            self._evict_locked()
            self._jobs[job_id] = job

        # The route needs a URL it can hand back and a correlation id we can
        # match against the queue while the run is still in flight.
        payload = dict(body)
        payload["response_format"] = "url"
        payload["stream"] = False
        payload["client_ref"] = job_id

        path = ("/v1/images/generations" if kind == "image"
                else "/v1/videos/generations")
        job.task = asyncio.create_task(self._run(job, path, payload, key))
        return job

    async def _run(self, job: GenJob, path: str, payload: dict[str, Any],
                   key: str) -> None:
        from .mcp_server import call_v1, _detail_of

        # A cancel can land before this task gets its first slice; don't
        # walk it back to "running".
        if job.status != "cancelled":
            job.status = "running"
        try:
            resp = await call_v1(self._app, key, path, json_body=payload)
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.finished_at = time.time()
            raise
        except Exception as e:  # noqa: BLE001 — recorded on the job
            job.status = "failed"
            job.error = str(e)
            job.finished_at = time.time()
            log.exception("mcp %s job %s failed", job.kind, job.job_id)
            return

        job.request_id = resp.headers.get("x-llamanager-request-id") or job.request_id
        job.finished_at = time.time()
        if job.status == "cancelled":
            # Someone cancelled this job while it was running. Whatever the
            # route answered on the way out — a 499, or a 502 because the
            # engine was torn down mid-step — the job was cancelled, and
            # relabelling it "failed" here would send the caller looking for
            # a bug that is really their own cancel.
            return
        if resp.status_code == 499:
            job.status = "cancelled"
            return
        if resp.status_code >= 400:
            job.status = "failed"
            job.error = f"{resp.status_code}: {_detail_of(resp)}"
            return
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = f"unparseable response: {e}"
            return
        job.result = [self._describe_output(item)
                      for item in (data.get("data") or [])]
        job.info = data.get("llamanager") or {}
        job.status = "done"

    def _describe_output(self, item: dict[str, Any]) -> dict[str, Any]:
        """Turn one response entry into something a local client can open.

        The route answers with a gallery URL (``/images/file/day/origin/name``)
        because that is what an HTTP caller can fetch. An MCP client is on
        this machine, so the file path is usually the more useful handle —
        give both, and never invent a path the URL doesn't describe.
        """
        out: dict[str, Any] = {"url": item.get("url"),
                               "revised_prompt": item.get("revised_prompt")}
        url = item.get("url") or ""
        parts = [p for p in url.split("/") if p]
        if len(parts) == 5 and parts[:2] == ["images", "file"]:
            day, origin_dir, name = parts[2], parts[3], parts[4]
            out["path"] = str(self._app.state.cfg.images_dir / day / origin_dir / name)
        return out

    # ----------------------------------------------------------- query ----

    def _visible(self, job: GenJob, origin: Origin) -> bool:
        """Admins see every job; everyone else sees only their own."""
        return origin.is_admin or job.origin_name == origin.name

    async def _get(self, job_id: str, origin: Origin) -> GenJob:
        from mcp.server.mcpserver.exceptions import ToolError
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None or not self._visible(job, origin):
            # Same answer either way: an origin must not be able to probe
            # for the existence of another origin's jobs.
            raise ToolError(f"no generation job {job_id}")
        return job

    async def describe(self, job_id: str, *, origin: Origin) -> dict[str, Any]:
        job = await self._get(job_id, origin)
        out = job.public()
        out["queue"] = self._queue_view(job)
        if job.status == "running":
            out["progress"] = self._progress_for(job)
        return out

    async def image_content(self, job_id: str, *, origin: Origin) -> list[Any]:
        from mcp.server.mcpserver.exceptions import ToolError

        job = await self._get(job_id, origin)
        if job.status != "done":
            raise ToolError(
                f"job {job_id} is {job.status}, not done; poll "
                f"get_generation_job until it finishes")
        blocks = self._image_blocks(job)
        if not blocks:
            paths = [r.get("path") for r in job.result]
            raise ToolError(
                f"job {job_id} produced no still image to show inline "
                f"(video stays on disk). Files: {paths}")
        return blocks

    def _queue_view(self, job: GenJob) -> dict[str, Any] | None:
        """Where this job sits in the queue, matched by its own client_ref."""
        qm = self._app.state.queue
        snap = qm.snapshot()
        for bucket in ("in_flight", "pending"):
            for entry in snap.get(bucket) or []:
                if entry.get("client_ref") == job.job_id:
                    if job.request_id is None:
                        job.request_id = entry.get("id")
                    return {"state": bucket, "request": entry}
        return None

    def _progress_for(self, job: GenJob) -> dict[str, Any] | None:
        """Live diffusion step, but only when the runner is on *this* job.

        The image runner reports one in-flight generation; attributing its
        steps to a job that is merely queued behind it would be a lie.
        """
        runner = getattr(self._app.state, "image_runner", None)
        if runner is None or job.request_id is None:
            return None
        try:
            st = runner.status()
        except Exception:  # noqa: BLE001 — progress is cosmetic
            return None
        if st.get("request_id") != job.request_id:
            return None
        return {"status": st.get("status"), "step": st.get("step"),
                "total_steps": st.get("total_steps"),
                "model_id": st.get("model_id")}

    def _image_blocks(self, job: GenJob) -> list[Any]:
        """Finished stills as inline content, downscaled to a viewable size.

        Returning the original bytes is not an option: the engines here
        write 2048-square PNGs (~2 MB, ~2.7 MB once base64-encoded), and a
        few of those would swamp the caller's context. A long-edge
        ``_VIEW_PX`` JPEG is enough to see what was generated, and the
        full-resolution file is on disk either way — every result carries
        its path. Same Pillow treatment the gallery thumbnails get.
        """
        import base64
        import io
        from pathlib import Path

        from mcp.types import ImageContent

        blocks: list[Any] = []
        for item in job.result:
            path = item.get("path")
            if not path:
                continue
            src = Path(path)
            if not src.is_file() or src.suffix.lower() not in _STILL_SUFFIXES:
                continue
            try:
                from PIL import Image

                with Image.open(src) as im:
                    im.draft("RGB", (_VIEW_PX, _VIEW_PX))
                    im = im.convert("RGB")
                    im.thumbnail((_VIEW_PX, _VIEW_PX))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=82, optimize=True)
            except Exception:  # noqa: BLE001 — a bad file is skipped, not fatal
                log.exception("mcp: cannot render %s for inline viewing", src)
                continue
            blocks.append(ImageContent(
                type="image",
                data=base64.b64encode(buf.getvalue()).decode("ascii"),
                mime_type="image/jpeg",
            ))
        return blocks

    # ---------------------------------------------------------- cancel ----

    async def cancel(self, job_id: str, *, origin: Origin) -> dict[str, Any]:
        from mcp.server.mcpserver.exceptions import ToolError

        job = await self._get(job_id, origin)
        if job.status in ("done", "failed", "cancelled"):
            raise ToolError(f"job {job_id} already finished ({job.status})")
        # Resolve the request id if the run has not returned yet.
        if job.request_id is None:
            self._queue_view(job)
        if job.request_id is None:
            raise ToolError(
                f"job {job_id} has not reached the queue yet; try again shortly")
        # Cancel through the queue, not by dropping our HTTP call: the route
        # deliberately shields the engine task from client disconnects, so
        # only the queue-level signal actually stops the subprocess.
        if not self._app.state.queue.cancel(job.request_id):
            raise ToolError(f"job {job_id} is no longer cancellable")
        job.status = "cancelled"
        job.finished_at = time.time()
        return {"ok": True, "job_id": job_id, "request_id": job.request_id}

    # ----------------------------------------------------------- admin ----

    def _evict_locked(self) -> None:
        finished = [j for j in self._jobs.values()
                    if j.status in ("done", "failed", "cancelled")]
        if len(finished) <= _MAX_FINISHED:
            return
        finished.sort(key=lambda j: j.finished_at or j.created_at)
        for old in finished[: len(finished) - _MAX_FINISHED]:
            self._jobs.pop(old.job_id, None)

    async def shutdown(self) -> None:
        """Cancel every in-flight job task on daemon shutdown."""
        async with self._lock:
            tasks = [j.task for j in self._jobs.values()
                     if j.task is not None and not j.task.done()]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
