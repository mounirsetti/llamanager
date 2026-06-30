"""Audio task runner (ASR / speech-to-text).

One-shot subprocess lifecycle for speech-to-text engines: spawn the
transcription runner, stream stderr into a per-engine log + parse ``chunk
i/N`` progress, wait for exit, and return the parsed JSON transcript.

A trimmed peer of ``ImageTaskRunner`` — same single-slot mutual exclusion
(one in-flight audio task at a time) and the same cross-family coordination
with the running text server via ``ServerManager.yield_to_image()`` (the
name is image-historical; it just pauses the text engine while a one-shot
GPU task runs). Audio engines hold no model in VRAM between requests.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import engines, exclusive as _exclusive, runtime_state as rt
from .config import Config, Profile, ENGINE_FAMILY, detect_engine_for_path
from .db import DB
from .engines._base import AudioRequest, ProgressEvent
from .server_manager import _safe_under, _validate_model_id

log = logging.getLogger(__name__)

TRANSCRIBE_HARD_TIMEOUT_S = 30 * 60   # 30 min — covers long recordings
PROGRESS_THROTTLE_S = 0.5


class AudioError(Exception):
    """Recoverable failure during transcription."""


@dataclass
class AudioResult:
    """The end-state of one finished transcription task."""
    request_id: str
    engine: str
    model_id: str
    profile_name: str | None
    text: str
    language: str | None
    duration_s: float
    segments: list[dict] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def resolve_audio_engine(cfg: Config, model_id: str) -> str:
    """Resolve which audio engine to use for ``model_id``. Raises
    ``AudioError`` if the model isn't an audio-family model."""
    _validate_model_id(model_id)
    model_path = _safe_under(cfg.models_dir, cfg.models_dir / model_id)
    if not model_path.exists():
        raise AudioError(f"model not found: {model_id}")
    engine = detect_engine_for_path(model_path)
    if ENGINE_FAMILY.get(engine, "text") != "audio":
        raise AudioError(
            f"model {model_id!r} is not a speech-to-text model "
            f"(detected engine: {engine})"
        )
    return engine


async def execute_transcription(qm, runner: "AudioTaskRunner", *, qr,
                                engine: str, model_id: str,
                                profile_obj, audio_req: AudioRequest) -> AudioResult:
    """Run one queued transcription to completion, always releasing the slot.

    ``qr`` must already be enqueued with ``task_type="audio"`` and the audio
    file staged at ``audio_req.audio_path``. Waits for the queue slot, invokes
    the runner, and releases the in-flight slot in a ``finally`` (so the
    queue's family accounting stays correct on every exit path).

    Re-raises ``AudioError`` / queue ``Cancelled`` / ``asyncio.TimeoutError``
    so the caller (HTTP or admin/CLI) can map them to a response. Shared by
    the ``/v1/audio/transcriptions`` and ``/admin/asr/transcribe`` endpoints.
    """
    from .queue_mgr import Cancelled
    error: str | None = None
    try:
        await qm.wait_for_slot(qr)
        return await runner.run(
            model_id=model_id, engine=engine, profile=profile_obj,
            req=audio_req, request_id=qr.request_id,
            origin_name=qr.origin.name, cancel_event=qr.cancel,
        )
    except AudioError as e:
        error = str(e)
        raise
    except Cancelled:
        error = "cancelled"
        raise
    except asyncio.TimeoutError:
        error = "queue timeout"
        raise
    except Exception as e:  # noqa: BLE001
        error = str(e)
        raise
    finally:
        qm.mark_in_flight_done(
            qr, error=error, cancelled=(error == "cancelled"),
            prompt_tokens=None, completion_tokens=None,
        )


@contextlib.asynccontextmanager
async def _nullcontext():
    yield


class AudioTaskRunner:
    """Single-slot dispatcher for speech-to-text tasks."""

    def __init__(self, cfg: Config, db: DB, sm=None) -> None:
        self.cfg = cfg
        self.db = db
        self.sm = sm   # ServerManager — used for yield_to_image coordination
        self._lock = asyncio.Lock()
        self._failures: deque[float] = deque()
        self._cooldown_until: float = 0.0

    # ---- lifecycle ----
    @property
    def is_busy(self) -> bool:
        return (self.cfg.runtime_path.exists()
                and rt.load(self.cfg.runtime_path).audio.status == "transcribing")

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def status(self) -> dict[str, Any]:
        state = rt.load(self.cfg.runtime_path).audio
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
        req: AudioRequest,
        request_id: str,
        origin_name: str,
        progress_cb: Callable[[ProgressEvent], Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AudioResult:
        """Transcribe ``req.audio_path`` for ``model_id``.

        Holds the per-runner lock for the full transcription: single-task
        mutual exclusion in the audio family.
        """
        if self.in_cooldown:
            wait = int(self._cooldown_until - time.time())
            raise AudioError(
                f"audio engine in cooldown for {wait}s after repeated failures"
            )
        if engine not in engines.ADAPTERS:
            raise AudioError(f"unknown audio engine: {engine!r}")
        adapter = engines.get(engine)

        _validate_model_id(model_id)
        model_path = _safe_under(self.cfg.models_dir, self.cfg.models_dir / model_id)
        if not model_path.exists():
            raise AudioError(f"model not found: {model_path}")
        if not adapter.detect(model_path):
            log.debug("adapter %s.detect() returned False for %s",
                      engine, model_path)
        if not req.audio_path.exists():
            raise AudioError(f"audio file not found: {req.audio_path}")

        profile = profile or Profile(name="(none)")
        out_dir = self.cfg.data_dir / "transcripts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request_id}.json"
        t0 = time.time()

        async with self._lock:
            yield_cm = (
                self.sm.yield_to_image(reason=f"audio:{engine}")
                if self.sm is not None
                else _nullcontext()
            )
            try:
                async with yield_cm:
                    argv, env = adapter.build_command(
                        self.cfg, model_path, profile, req, out_path,
                    )
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
                        cancel_event=cancel_event,
                    )
                self._failures.clear()
            except Exception:
                self._record_failure()
                raise

        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise AudioError(f"could not read transcript output: {e}") from e
        finally:
            with contextlib.suppress(OSError):
                out_path.unlink()

        return AudioResult(
            request_id=request_id,
            engine=engine,
            model_id=model_id,
            profile_name=profile.name,
            text=str(data.get("text", "")),
            language=data.get("language"),
            duration_s=float(data.get("duration_s") or (time.time() - t0)),
            segments=data.get("segments") or [],
            raw=data,
        )

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
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        log_path = self.cfg.logs_dir / f"{engine}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w", encoding="utf-8") as fp:
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                fp.write(
                    f"# {engine} request {request_id} — {model_id} "
                    f"(profile={profile_name}) — {ts}\n"
                )
        except OSError:
            pass

        state = rt.load(self.cfg.runtime_path)
        state.audio = rt.AudioRuntimeState(
            status="transcribing",
            engine=engine,
            model_id=model_id,
            profile=profile_name,
            request_id=request_id,
            started_at=time.time(),
            last_event_at=time.time(),
        )
        rt.save(self.cfg.runtime_path, state)
        self.db.log_event("audio_transcribe_begin", {
            "request_id": request_id, "engine": engine,
            "model": model_id, "profile": profile_name,
        })

        # Defence in depth: sweep exclusive-mode workers right before spawn.
        mode = (getattr(self.cfg, "exclusive_mode", "off") or "off").lower()
        if mode not in ("off", ""):
            try:
                await _exclusive.sweep_and_record(
                    mode,
                    grace_seconds=float(getattr(
                        self.cfg, "exclusive_grace_seconds", 5.0) or 5.0),
                )
            except Exception:  # noqa: BLE001
                log.exception("exclusive: pre-audio-spawn sweep failed")

        full_env = {**os.environ, **env}
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
            raise AudioError(f"engine binary not found: {argv[0]}") from e

        last_progress_update = 0.0

        async def _handle_line(line: str) -> None:
            nonlocal last_progress_update
            ev = adapter.parse_progress(line)
            if ev is None:
                return
            now = time.time()
            if now - last_progress_update < PROGRESS_THROTTLE_S:
                return
            last_progress_update = now
            state.audio.step = ev.step
            state.audio.total_steps = ev.total
            state.audio.last_event_at = now
            rt.save(self.cfg.runtime_path, state)
            if progress_cb is not None:
                try:
                    res = progress_cb(ev)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    log.debug("progress_cb raised", exc_info=True)

        async def _drain(stream) -> None:
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
                    try:
                        buf += chunk.decode("utf-8", errors="replace")
                    except Exception:
                        buf = ""
                        continue
                    segments = buf.replace("\r", "\n").split("\n")
                    buf = segments.pop()
                    for seg in segments:
                        if seg:
                            await _handle_line(seg)

        drain_out = asyncio.create_task(_drain(proc.stdout))
        drain_err = asyncio.create_task(_drain(proc.stderr))

        cancel_watcher: asyncio.Task[None] | None = None
        cancelled = False

        async def _watch_cancel() -> None:
            nonlocal cancelled
            assert cancel_event is not None
            await cancel_event.wait()
            cancelled = True
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

        if cancel_event is not None:
            cancel_watcher = asyncio.create_task(_watch_cancel())

        try:
            rc = await asyncio.wait_for(
                proc.wait(), timeout=TRANSCRIBE_HARD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await asyncio.gather(drain_out, drain_err, return_exceptions=True)
            self._reset_runtime_state(failed=True)
            raise AudioError("transcription timed out")
        finally:
            if cancel_watcher is not None:
                cancel_watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_watcher
            await asyncio.gather(drain_out, drain_err, return_exceptions=True)

        if cancelled:
            self._reset_runtime_state(failed=False)
            self.db.log_event("audio_transcribe_cancelled", {
                "request_id": request_id, "engine": engine,
            })
            raise AudioError("cancelled")

        if rc != 0:
            self._reset_runtime_state(failed=True)
            self.db.log_event("audio_transcribe_failed", {
                "request_id": request_id, "engine": engine, "rc": rc,
            })
            raise AudioError(f"engine {engine} exited with rc={rc}")

        if not out_path.exists():
            self._reset_runtime_state(failed=True)
            raise AudioError(
                f"engine reported success but no output file at {out_path}")

        self._reset_runtime_state(failed=False)
        self.db.log_event("audio_transcribe_done", {
            "request_id": request_id, "engine": engine, "model": model_id,
        })

    def _reset_runtime_state(self, *, failed: bool) -> None:
        state = rt.load(self.cfg.runtime_path)
        state.audio = rt.AudioRuntimeState(
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
            self._cooldown_until = now + self.cfg.window_seconds
            self.db.log_event("audio_engine_degraded", {
                "failures": len(self._failures),
                "window_s": self.cfg.window_seconds,
            })
            log.error(
                "audio runner: %d failures in %ds, entering %ds cooldown",
                len(self._failures), self.cfg.window_seconds,
                self.cfg.window_seconds,
            )
