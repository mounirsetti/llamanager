"""Image task runner.

One-shot subprocess lifecycle for image-generation engines: spawn,
stream stderr/stdout into a per-engine log + parse progress along the
way, wait for exit, and return the produced PNG.

The runner serialises requests (one in-flight image task at a time —
mutual exclusion within the image family). Cross-family coordination
with the running text server happens through
``ServerManager.yield_to_image()`` (see server_manager.py).

Designed as a peer of ``ServerManager``, not as a replacement: text
engines stay long-running HTTP servers, image engines are tasks.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import os
import secrets
import shutil
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import engines, exclusive as _exclusive, runtime_state as rt
from .auth import validate_origin_name
from .config import Config, Profile, ENGINE_FAMILY, detect_engine_for_path
from .db import DB
from .engines._base import ImageRequest, ProgressEvent
from .server_manager import _safe_under, _validate_model_id

log = logging.getLogger(__name__)


PROGRESS_THROTTLE_S = 0.5      # don't spam runtime.json — update at most twice/sec
GENERATION_HARD_TIMEOUT_S = 30 * 60   # 30 min — covers FLUX 2 quality runs
# Video generation is far heavier (a 5B model, dozens of frames, VAE decode of
# a whole clip) and can legitimately exceed the image budget on a single card.
VIDEO_GENERATION_HARD_TIMEOUT_S = 90 * 60   # 90 min — Wan 720p / 121-frame runs


class ImageError(Exception):
    """Recoverable failure during image generation."""


@dataclass
class ImageResult:
    """The end-state of one finished image task."""
    request_id: str
    engine: str
    model_id: str
    profile_name: str | None
    output_path: Path
    seed: int | None
    duration_s: float
    sidecar: dict[str, Any] = field(default_factory=dict)


def _gallery_dir(cfg: Config, origin: str) -> Path:
    """Per-day, per-origin output directory under cfg.images_dir.

    The origin name is used verbatim. It is validated at creation
    (``auth.validate_origin_name``) so that exactly this mapping is injective;
    sanitising here instead silently merged distinct origins into one
    directory, and the gallery routes then let each read the other's history.
    An origin predating that validation fails loudly here rather than writing
    into somebody else's folder.
    """
    validate_origin_name(origin)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = cfg.images_dir / day / origin
    d.mkdir(parents=True, exist_ok=True)
    return d


def _enforce_disk_cap(cfg: Config) -> None:
    """Trim the gallery oldest-first when it exceeds ``images_max_disk_gb``."""
    cap_bytes = int(cfg.images_max_disk_gb) * 1024 ** 3
    if cap_bytes <= 0 or not cfg.images_dir.exists():
        return
    entries: list[tuple[float, Path]] = []
    total = 0
    for root, _, files in os.walk(cfg.images_dir):
        for fn in files:
            p = Path(root) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, p))
            total += st.st_size
    if total <= cap_bytes:
        return
    # Drop oldest files first until we're back under the cap.
    entries.sort(key=lambda x: x[0])
    for _, p in entries:
        try:
            size = p.stat().st_size
            p.unlink()
            sidecar = p.with_suffix(p.suffix + ".json")
            with contextlib.suppress(OSError):
                sidecar.unlink()
            total -= size
        except OSError:
            continue
        if total <= cap_bytes:
            break


#: Long side of the reference thumbnails stored in a sidecar. Big enough to
#: recognise the picture in the details panel, small enough that eight of them
#: (the per-request cap) stay well under a megabyte of JSON.
_REF_THUMB_PX = 384


def _ref_thumbnails(paths: list[Path]) -> list[str]:
    """Downscaled data URIs for the reference images used by one generation.

    The staged originals live under ``data_dir/refs/<request_id>`` and are
    deleted as soon as the run finishes, so a sidecar that merely named them
    would point at nothing. Embedding a thumbnail keeps the answer to "what
    did I feed this?" attached to the output forever, survives the gallery's
    disk-cap GC, needs no new route, and cannot leave orphans behind.

    Best-effort by design: this is metadata, and a decode failure must never
    cost the operator a finished image.
    """
    out: list[str] = []
    for p in paths:
        try:
            from PIL import Image

            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((_REF_THUMB_PX, _REF_THUMB_PX))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=72, optimize=True)
            out.append("data:image/jpeg;base64,"
                       + base64.b64encode(buf.getvalue()).decode("ascii"))
        except Exception:  # noqa: BLE001 — metadata only
            log.warning("could not thumbnail reference image %s", p,
                        exc_info=True)
    return out


class ImageTaskRunner:
    """Single-slot dispatcher for image-generation tasks.

    Holds no model in VRAM between requests — that's the point of the
    one-shot design. State persists only through ``runtime.json`` and
    sidecar metadata on disk.
    """

    def __init__(self, cfg: Config, db: DB, sm=None) -> None:
        self.cfg = cfg
        self.db = db
        self.sm = sm   # ServerManager — used for yield_to_image coordination
        self._lock = asyncio.Lock()
        # Crash-policy state: rolling window of recent failure timestamps,
        # mirroring Supervisor's policy for text engines.
        self._failures: deque[float] = deque()
        self._cooldown_until: float = 0.0
        # Optional callback hooks for streaming progress to UI/SSE.
        self._progress_listeners: list[Callable[[ProgressEvent], Any]] = []
        # Fired (synchronously) once a run has released the GPU. Wired to
        # the queue dispatcher so it re-evaluates eligibility — see
        # ``is_busy``.
        self.on_idle: Callable[[], None] | None = None

    # ---- lifecycle ----
    @property
    def is_busy(self) -> bool:
        """True while a generation actually owns the GPU.

        The run lock is the authority here, not ``runtime.json``: it is
        held for exactly as long as an engine is resident (including the
        yield_to_image window around it), it needs no disk read, and it
        cannot lag behind reality. The queue consults this as a second,
        independent invariant alongside its own in-flight counter — the
        counter tracks *requests*, and a request can end while its engine
        has not (client disconnects, handler unwinds). Trusting the
        counter alone is what let llama-server start on top of a live
        diffusion engine.
        """
        return self._lock.locked()

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def status(self) -> dict[str, Any]:
        state = rt.load(self.cfg.runtime_path).image
        return {
            "status": state.status,
            "engine": state.engine,
            "model_id": state.model_id,
            "profile": state.profile,
            "request_id": state.request_id,
            "step": state.step,
            "total_steps": state.total_steps,
            "started_at": state.started_at,
        }

    # ---- main entry point ----
    async def run(
        self,
        *,
        model_id: str,
        engine: str,
        profile: Profile | None,
        req: ImageRequest,
        request_id: str,
        origin_name: str,
        progress_cb: Callable[[ProgressEvent], Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ImageResult:
        """Generate ``req.n`` images for ``model_id``.

        Holds the per-runner lock for the full generation: this enforces
        single-task mutual exclusion in the image family.

        ``cancel_event`` lets the queue-level cancel signal reach the
        running subprocess. When set, the in-flight image generation is
        terminated and an ``ImageError("cancelled")`` is raised so the
        handler can surface a 499.
        """
        if self.in_cooldown:
            wait = int(self._cooldown_until - time.time())
            raise ImageError(
                f"image engine in cooldown for {wait}s after repeated failures"
            )
        if engine not in engines.ADAPTERS:
            raise ImageError(f"unknown image engine: {engine!r}")
        adapter = engines.get(engine)

        # Validate model path inside the sandboxed models_dir.
        _validate_model_id(model_id)
        model_path = _safe_under(self.cfg.models_dir, self.cfg.models_dir / model_id)
        if not model_path.exists():
            raise ImageError(f"model not found: {model_path}")
        # adapter.detect() is best-effort; we don't hard-fail on shape
        # mismatch because the operator may point at a sibling layout.
        if not adapter.detect(model_path):
            log.debug("adapter %s.detect() returned False for %s",
                      engine, model_path)

        profile = profile or Profile(name="(none)")

        # Produced-file extension is engine-declared (video engines → "mp4",
        # everything else → the "png" default). The gallery/serving layers
        # treat both uniformly by suffix.
        output_ext = str(
            engines.capabilities(engine).get("output_ext") or "png").lstrip(".")

        gallery = _gallery_dir(self.cfg, origin_name)
        outputs: list[Path] = []
        seed_used: int | None = None
        n = max(1, int(req.n))
        sidecar_base: dict[str, Any] = {}
        t0 = time.time()

        # Reference-image staging directory (populated by the API layer at
        # ``<data_dir>/refs/<request_id>``). We don't strictly require it
        # to be a child of that path — just clean up the parent of the
        # first ref if all refs live in the same directory.
        ref_tempdir: Path | None = None
        if req.ref_images:
            parents = {p.parent for p in req.ref_images}
            if len(parents) == 1:
                only_parent = next(iter(parents))
                # Sanity-guard: only auto-clean dirs that live under our
                # data_dir/refs/, never paths the operator might have set
                # by hand.
                try:
                    only_parent.relative_to(self.cfg.data_dir / "refs")
                    ref_tempdir = only_parent
                except ValueError:
                    pass

        try:
            async with self._lock:
                # Coordinate with the text engine if running.
                yield_cm = (
                    self.sm.yield_to_image(reason=f"image:{engine}")
                    if self.sm is not None
                    else _nullcontext()
                )
                async with yield_cm:
                    for i in range(n):
                        # Different seed per image when not pinned. Ref images
                        # and the other reference-input knobs are shared across
                        # the whole batch — they describe the input, not the
                        # individual sample.
                        # replace(), not a hand-listed copy: this used to
                        # enumerate every field, so each new one added to
                        # ImageRequest was silently dropped on its way to the
                        # adapter — which is how model_type/guidance/
                        # editing_scheduler arrived from the API and never
                        # reached the engine. Only per-sample values differ.
                        per_req = replace(
                            req,
                            seed=(req.seed if req.seed is not None
                                  else _new_seed()),
                            n=1,
                            ref_images=list(req.ref_images),
                        )
                        seed_used = per_req.seed
                        out_path = gallery / _new_image_filename(
                            engine, gallery, ext=output_ext)
                        argv, env = adapter.build_command(
                            self.cfg, model_path, profile, per_req, out_path,
                        )
                        sidecar_base = {
                            "engine": engine,
                            "model_id": model_id,
                            "profile": profile.name,
                            "prompt": per_req.prompt,
                            "width": per_req.width,
                            "height": per_req.height,
                            "steps": per_req.steps,
                            "seed": per_req.seed,
                        }
                        # Which references went in. Kept as thumbnails
                        # because the staged files are deleted below.
                        if per_req.ref_images:
                            thumbs = _ref_thumbnails(per_req.ref_images)
                            if thumbs:
                                sidecar_base["ref_thumbs"] = thumbs
                            sidecar_base["ref_count"] = len(per_req.ref_images)
                        # What the request *asked for* is not always what
                        # runs: HiDream's dev recipe hardwires its step count
                        # and disables guidance, so a 28-step image could be
                        # filed as "steps: 100". Adapters that can resolve
                        # their own knobs report them here and win.
                        resolve = getattr(adapter, "effective_params", None)
                        if resolve is not None:
                            try:
                                sidecar_base.update(
                                    resolve(self.cfg, model_path, profile,
                                            per_req))
                            except Exception:  # noqa: BLE001 — metadata only
                                log.exception(
                                    "%s: effective_params failed; recording "
                                    "the requested values", engine)
                        try:
                            await self._run_one(
                                engine=engine,
                                model_id=model_id,
                                profile_name=profile.name,
                                request_id=request_id,
                                argv=argv,
                                env=env,
                                out_path=out_path,
                                adapter=adapter,
                                progress_cb=progress_cb,
                                sidecar=sidecar_base,
                                cancel_event=cancel_event,
                            )
                        except Exception:
                            self._record_failure()
                            raise
                        outputs.append(out_path)
                        # Honour cancellation between samples of an n>1
                        # batch so we don't keep burning compute after
                        # the client gave up.
                        if cancel_event is not None and cancel_event.is_set():
                            raise ImageError("cancelled")
                # Recovery: window resets after a successful run.
                self._failures.clear()

            _enforce_disk_cap(self.cfg)
            duration = time.time() - t0
            # When n > 1 we return the *first* image as the canonical result
            # alongside a manifest of all paths inside sidecar.
            if not outputs:
                raise ImageError("image generation produced no output")
            primary = outputs[0]
            sidecar = {**sidecar_base, "duration_s": round(duration, 3)}
            if len(outputs) > 1:
                sidecar["batch"] = [str(p) for p in outputs]
            return ImageResult(
                request_id=request_id,
                engine=engine,
                model_id=model_id,
                profile_name=profile.name,
                output_path=primary,
                seed=seed_used,
                duration_s=duration,
                sidecar=sidecar,
            )
        finally:
            # The run lock is released by now on every exit path, so the
            # GPU is genuinely free. Poke the queue: a text request may be
            # parked purely because ``is_busy`` was True, and nothing else
            # would wake the dispatcher for a lock release.
            if self.on_idle is not None:
                try:
                    self.on_idle()
                except Exception:  # noqa: BLE001 — never break a run on this
                    log.debug("on_idle hook raised", exc_info=True)
            # Drop the staged reference-image directory regardless of how
            # we exited — success, engine crash, queue cancel, timeout.
            # ``shutil.rmtree`` with ignore_errors is safe even if the
            # adapter never touched the dir; we already vetted ``ref_tempdir``
            # is under ``data_dir/refs/`` above, so this can't escape.
            if ref_tempdir is not None:
                shutil.rmtree(ref_tempdir, ignore_errors=True)

    # ---- internals ----
    async def _run_one(
        self,
        *,
        engine: str,
        model_id: str,
        profile_name: str,
        request_id: str,
        argv: list[str],
        env: dict[str, str],
        out_path: Path,
        adapter,
        progress_cb: Callable[[ProgressEvent], Any] | None,
        sidecar: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        log_path = self.cfg.logs_dir / f"{engine}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate the engine log on every new request so the dashboard
        # log tail only shows lines from *this* run. Without this, stale
        # tracebacks from prior installs / model swaps / failures linger
        # in the file and bleed through into the UI, making it look like
        # the current run is failing for old reasons. The DB activity
        # feed remains the system-of-record for cross-run history.
        try:
            with open(log_path, "w", encoding="utf-8") as fp:
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                fp.write(
                    f"# {engine} request {request_id} — {model_id} "
                    f"(profile={profile_name}) — {ts}\n"
                )
        except OSError:
            pass
        # Update runtime state — generating.
        state = rt.load(self.cfg.runtime_path)
        state.image = rt.ImageRuntimeState(
            status="generating",
            engine=engine,
            model_id=model_id,
            profile=profile_name,
            request_id=request_id,
            step=None,
            total_steps=None,
            started_at=time.time(),
            last_event_at=time.time(),
        )
        rt.save(self.cfg.runtime_path, state)
        self.db.log_event("image_generate_begin", {
            "request_id": request_id, "engine": engine,
            "model": model_id, "profile": profile_name,
        })

        # Defence in depth: if exclusive mode is on, sweep one more time
        # right before we spawn the engine worker. ServerManager.start
        # and yield_to_image already swept, but this catches workers that
        # appeared in the gap (a heartbeat tick that just fired, etc.).
        mode = (getattr(self.cfg, "exclusive_mode", "off") or "off").lower()
        if mode not in ("off", ""):
            try:
                await _exclusive.sweep_and_record(
                    mode,
                    grace_seconds=float(getattr(
                        self.cfg, "exclusive_grace_seconds", 5.0) or 5.0),
                )
            except Exception:  # noqa: BLE001
                log.exception("exclusive: pre-image-spawn sweep failed")

        full_env = {**os.environ, **env}
        # Wall clock for this one image, recorded in its sidecar so the
        # gallery can show what it cost. Starts at the spawn, so it counts
        # weight loading — the dominant term for most engines — but not the
        # queue wait, which is not a property of the image.
        spawn_t0 = time.time()
        log.info("launching %s engine: %s", engine,
                 " ".join(map(str, argv[:4])) + " ...")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=full_env,
            )
        except FileNotFoundError as e:
            self._reset_runtime_state(failed=True)
            raise ImageError(f"engine binary not found: {argv[0]}") from e

        last_progress_update = 0.0
        last_event: ProgressEvent | None = None

        async def _handle_line(line: str) -> None:
            """Parse one output line for progress and (throttled) publish it."""
            nonlocal last_progress_update, last_event
            ev = adapter.parse_progress(line)
            if ev is None:
                return
            last_event = ev
            now = time.time()
            if now - last_progress_update < PROGRESS_THROTTLE_S:
                return
            last_progress_update = now
            state.image.step = ev.step
            state.image.total_steps = ev.total
            state.image.last_event_at = now
            rt.save(self.cfg.runtime_path, state)
            if progress_cb is not None:
                try:
                    res = progress_cb(ev)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    log.debug("progress_cb raised", exc_info=True)

        async def _drain(stream, prefix: str) -> None:
            """Pump one subprocess pipe → engine log + parse progress.

            diffusers/tqdm refresh the progress line *in place* with
            carriage returns (``\\r``) and only emit a newline when the bar
            finishes, so a plain ``readline()`` would block until the end
            and the UI would show no live progress. We read in chunks and
            treat both ``\\r`` and ``\\n`` as line breaks, so every tqdm
            refresh ("N/M") is parsed as it happens.
            """
            buf = ""
            with open(log_path, "ab") as log_fp:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        rest = buf.strip("\r\n")
                        if rest:
                            await _handle_line(rest)
                        return
                    log_fp.write(chunk)
                    # Flush per chunk: the handle stays open for the whole
                    # run, so without this the engine log reaches disk only
                    # when the process exits — the dashboard's log tail
                    # showed an empty file for the entire generation, which
                    # is exactly when an operator wants to read it.
                    log_fp.flush()
                    try:
                        buf += chunk.decode("utf-8", errors="replace")
                    except Exception:
                        buf = ""
                        continue
                    # Treat CR as a line break so in-place tqdm refreshes
                    # parse live; keep the trailing partial for next chunk.
                    segments = buf.replace("\r", "\n").split("\n")
                    buf = segments.pop()
                    for seg in segments:
                        if seg:
                            await _handle_line(seg)

        drain_out = asyncio.create_task(_drain(proc.stdout, "out"))
        drain_err = asyncio.create_task(_drain(proc.stderr, "err"))

        # If the caller hands us a cancel event, spawn a watcher that
        # SIGTERMs the subprocess as soon as the queue-level cancel
        # fires. Without this, a "cancelled" image request would still
        # burn its full generation budget in the background.
        cancel_watcher: asyncio.Task[None] | None = None
        cancelled = False

        async def _watch_cancel() -> None:
            nonlocal cancelled
            assert cancel_event is not None
            await cancel_event.wait()
            cancelled = True
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            # If the child doesn't exit promptly, escalate to KILL.
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

        if cancel_event is not None:
            cancel_watcher = asyncio.create_task(_watch_cancel())

        from .config import engine_family
        hard_timeout = (VIDEO_GENERATION_HARD_TIMEOUT_S
                        if engine_family(engine) == "video"
                        else GENERATION_HARD_TIMEOUT_S)
        try:
            rc = await asyncio.wait_for(
                proc.wait(), timeout=hard_timeout,
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await asyncio.gather(drain_out, drain_err, return_exceptions=True)
            self._reset_runtime_state(failed=True)
            raise ImageError("image generation timed out")
        finally:
            if cancel_watcher is not None:
                cancel_watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_watcher
            await asyncio.gather(drain_out, drain_err, return_exceptions=True)

        if cancelled:
            self._reset_runtime_state(failed=False)
            self.db.log_event("image_generate_cancelled", {
                "request_id": request_id, "engine": engine,
            })
            raise ImageError("cancelled")

        if rc != 0:
            self._reset_runtime_state(failed=True)
            self.db.log_event("image_generate_failed", {
                "request_id": request_id, "engine": engine, "rc": rc,
            })
            raise ImageError(f"engine {engine} exited with rc={rc}")

        if not out_path.exists():
            self._reset_runtime_state(failed=True)
            raise ImageError(f"engine reported success but no output file at {out_path}")

        # Write sidecar metadata.
        sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
        try:
            import json
            sidecar_path.write_text(
                json.dumps({**sidecar,
                            "generation_s": round(time.time() - spawn_t0, 2)},
                           indent=2),
                encoding="utf-8",
            )
        except OSError:
            log.warning("sidecar write failed for %s", sidecar_path)

        # Reset state to idle.
        self._reset_runtime_state(failed=False)
        self.db.log_event("image_generate_done", {
            "request_id": request_id, "engine": engine,
            "model": model_id, "output": str(out_path),
        })

    def _reset_runtime_state(self, *, failed: bool) -> None:
        state = rt.load(self.cfg.runtime_path)
        state.image = rt.ImageRuntimeState(
            status="failed" if failed else "idle",
            last_event_at=time.time(),
        )
        rt.save(self.cfg.runtime_path, state)

    def _record_failure(self) -> None:
        now = time.time()
        self._failures.append(now)
        cutoff = now - self.cfg.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        if len(self._failures) >= self.cfg.max_restarts_in_window:
            # Same 3-in-5min policy as text engines; cooldown == window.
            self._cooldown_until = now + self.cfg.window_seconds
            self.db.log_event("image_engine_degraded", {
                "failures": len(self._failures),
                "window_s": self.cfg.window_seconds,
            })
            log.error(
                "image runner: %d failures in %ds, entering %ds cooldown",
                len(self._failures), self.cfg.window_seconds,
                self.cfg.window_seconds,
            )


def _new_seed() -> int:
    return int.from_bytes(secrets.token_bytes(4), "big") & 0x7FFFFFFF


def _new_image_filename(engine: str, gallery: Path, ext: str = "png") -> str:
    """Build an output filename of the form ``<eng><hhmm>[-NN].<ext>``.

    ``<eng>`` is the first three letters of the engine name (hidream → hid,
    flux2 → flu) — a short, eyeball-friendly tag rather than an opaque
    hash. ``<hhmm>`` is the local wall-clock time at filename allocation,
    which is what an operator will be looking for when they ask "where's
    the image I just generated at 14:23". An incrementing ``-NN`` suffix
    breaks ties when more than one output is produced in the same minute
    (n>1 requests, or back-to-back single requests). Falls back to a
    random hex suffix after 99 same-minute collisions so the function
    cannot get stuck. ``ext`` is the produced-file extension ("png" for
    images, "mp4" for video engines).
    """
    prefix = (engine or "img")[:3].lower()
    ext = (ext or "png").lstrip(".")
    hhmm = datetime.now().strftime("%H%M")
    base = f"{prefix}{hhmm}"
    candidate = gallery / f"{base}.{ext}"
    if not candidate.exists():
        return candidate.name
    for i in range(2, 100):
        candidate = gallery / f"{base}-{i}.{ext}"
        if not candidate.exists():
            return candidate.name
    return f"{base}-{secrets.token_hex(2)}.{ext}"


@contextlib.asynccontextmanager
async def _nullcontext():
    """Async no-op context manager (Python 3.10+ has contextlib.nullcontext
    but it's sync-only)."""
    yield


def resolve_image_engine(cfg: Config, model_id: str) -> str:
    """Resolve which image engine to use for ``model_id``. Raises
    ``ImageError`` if the model isn't an image-family model."""
    _validate_model_id(model_id)
    model_path = _safe_under(cfg.models_dir, cfg.models_dir / model_id)
    if not model_path.exists():
        raise ImageError(f"model not found: {model_id}")
    engine = detect_engine_for_path(model_path)
    if ENGINE_FAMILY.get(engine, "text") != "image":
        raise ImageError(
            f"model {model_id!r} is not an image-generation model "
            f"(detected engine: {engine})"
        )
    return engine


def resolve_video_engine(cfg: Config, model_id: str) -> str:
    """Resolve which video engine to use for ``model_id``. Raises
    ``ImageError`` if the model isn't a video-family model."""
    _validate_model_id(model_id)
    model_path = _safe_under(cfg.models_dir, cfg.models_dir / model_id)
    if not model_path.exists():
        raise ImageError(f"model not found: {model_id}")
    engine = detect_engine_for_path(model_path)
    if ENGINE_FAMILY.get(engine, "text") != "video":
        raise ImageError(
            f"model {model_id!r} is not a video-generation model "
            f"(detected engine: {engine})"
        )
    return engine
