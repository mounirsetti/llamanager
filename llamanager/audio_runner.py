"""Audio task runner — manages a persistent, warm ASR worker.

ASR is a multi-tenant GPU service: a **warm worker** (`engines/_asr_worker.py`)
keeps Whisper loaded and serves **many concurrent** transcriptions over a
loopback HTTP port, the same way `llama-server` serves text. This manager owns
the worker's lifecycle (lazy start, health, idle-stop to reclaim VRAM), proxies
requests to it, and applies the coexistence policy:

  * ``asr_coexist = True``  → the worker runs alongside a loaded LLM, bounded by
    ``asr_vram_budget_gb`` (→ ``config.asr_max_concurrent``). No unloading.
  * ``asr_coexist = False`` → the worker's lifetime holds ``yield_to_image``
    (the text server is unloaded while the worker is warm, restored when it
    idle-stops), and audio is mutually exclusive with text in the queue.

Concurrency is bounded by the queue's admission (see ``queue_mgr``), not a
per-runner lock — so the class name is historical; it no longer serialises.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from . import engines, runtime_state as rt
from .config import (Config, Profile, ENGINE_FAMILY, asr_max_concurrent,
                     detect_audio_engine_for_path)
from .db import DB
from .engines._base import AudioRequest, ProgressEvent
from .server_manager import _safe_under, _validate_model_id

log = logging.getLogger(__name__)

WORKER_START_TIMEOUT_S = 180.0     # model load can be slow (cold cache)
TRANSCRIBE_TIMEOUT_S = 30 * 60
IDLE_CHECK_INTERVAL_S = 20.0


class AudioError(Exception):
    """Recoverable failure during transcription."""


@dataclass
class AudioResult:
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
    """Resolve the audio engine for ``model_id`` (under ``asr_models_dir``).
    Raises ``AudioError`` unless it's an audio-family model."""
    _validate_model_id(model_id)
    base = cfg.asr_models_dir
    model_path = _safe_under(base, base / model_id)
    if not model_path.exists():
        raise AudioError(f"model not found: {model_id}")
    engine = detect_audio_engine_for_path(model_path)
    if engine is None or ENGINE_FAMILY.get(engine, "text") != "audio":
        raise AudioError(
            f"model {model_id!r} is not a speech-to-text model "
            f"(detected engine: {engine})")
    return engine


def scan_asr_models(cfg: Config) -> list[dict[str, Any]]:
    """List audio model folders under ``cfg.asr_models_dir``, tagged by engine.

    Registry-independent scan across all three audio-engine shapes: transformers
    Whisper (``config.json``), whisper.cpp GGML (``ggml-*.bin``), and sherpa-onnx
    (``tokens.txt``). Symlinked dirs are skipped to match the ``_safe_under``
    sandbox the transcription path enforces."""
    base = cfg.asr_models_dir
    out: list[dict[str, Any]] = []
    if not base.is_dir():
        return out
    # Collect candidate model dirs from each engine's marker file, then detect.
    seen: set[Path] = set()
    marker_globs = ("config.json", "ggml-*.bin", "*.gguf", "*tokens.txt")
    candidates: set[Path] = set()
    for pattern in marker_globs:
        for marker in base.rglob(pattern):
            if marker.is_file():
                candidates.add(marker.parent)
    for d in candidates:
        if d in seen:
            continue
        engine = detect_audio_engine_for_path(d)
        if engine is None:
            continue
        seen.add(d)
        try:
            rel = d.relative_to(base).as_posix()
        except ValueError:
            continue
        try:
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        except OSError:
            size = 0
        out.append({"model_id": rel, "path": str(d), "size_bytes": size,
                    "engine": engine})
    out.sort(key=lambda m: m["model_id"])
    return out


async def execute_transcription(qm, runner: "AudioTaskRunner", *, qr,
                                engine: str, model_id: str,
                                profile_obj, audio_req: AudioRequest) -> AudioResult:
    """Run one queued transcription to completion, always releasing the slot.

    ``qr`` must already be enqueued with ``task_type="audio"``. Waits for the
    queue slot (admission bounds concurrency by the VRAM budget), invokes the
    worker via ``runner.run``, and releases the in-flight slot in a ``finally``.
    Re-raises ``AudioError`` / queue ``Cancelled`` / ``asyncio.TimeoutError``."""
    from .queue_mgr import Cancelled
    error: str | None = None
    try:
        await qm.wait_for_slot(qr)
        return await runner.run(
            model_id=model_id, engine=engine, profile=profile_obj,
            req=audio_req, request_id=qr.request_id,
            origin_name=qr.origin.name, cancel_event=qr.cancel)
    except AudioError as e:
        error = str(e); raise
    except Cancelled:
        error = "cancelled"; raise
    except asyncio.TimeoutError:
        error = "queue timeout"; raise
    except Exception as e:  # noqa: BLE001
        error = str(e); raise
    finally:
        qm.mark_in_flight_done(
            qr, error=error, cancelled=(error == "cancelled"),
            prompt_tokens=None, completion_tokens=None)


@contextlib.asynccontextmanager
async def _nullcontext():
    yield


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class AudioTaskRunner:
    """Manager for the warm ASR worker (see module docstring)."""

    def __init__(self, cfg: Config, db: DB, sm=None) -> None:
        self.cfg = cfg
        self.db = db
        self.sm = sm
        self._start_lock = asyncio.Lock()     # serialises worker start/stop only
        self._proc: asyncio.subprocess.Process | None = None
        self._worker_model: str | None = None
        self._worker_engine: str | None = None
        self._base_url: str = ""
        self._client: httpx.AsyncClient | None = None
        self._active = 0                       # in-flight transcriptions
        self._last_used = 0.0
        self._idle_task: asyncio.Task | None = None
        self._yield_cm = None                  # held yield_to_image (coexist=off)
        # Refcount for the coexist=off LLM/diffusion yield: the other engine is
        # unloaded once while ANY audio task is in flight and restored when the
        # last one finishes, so concurrent transcriptions share a single unload
        # (not a per-task unload/restore thrash).
        self._yield_refs = 0
        self._yield_lock = asyncio.Lock()
        self._failures: deque[float] = deque()
        self._cooldown_until = 0.0

    # ---- status ----
    @property
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    @property
    def is_busy(self) -> bool:
        return self._active > 0

    def _coexist_feasible(self) -> bool:
        """Is it safe to run the ASR worker *alongside* the loaded LLM right now?

        The concurrent path only makes sense when there's real headroom. Host
        RAM is the signal that matters — the OOM incident was host-RAM
        exhaustion (a big LLM partially offloaded to CPU + a coexisting Whisper
        worker), not VRAM (GPU OOM there was caught and returned an error). If
        host memory is already tight (WARN or worse), refuse coexistence so the
        caller unloads the LLM for the task instead of stacking on top."""
        try:
            from .mem_guard import (read_mem_state, classify_pressure,
                                    MemThresholds, Pressure)
            th = MemThresholds.from_cfg(self.cfg)
            if classify_pressure(read_mem_state(), th) >= Pressure.WARN:
                return False
        except Exception:  # noqa: BLE001 — never let a probe error block ASR
            pass
        return True

    def status(self) -> dict[str, Any]:
        state = rt.load(self.cfg.runtime_path).audio
        return {
            "status": state.status, "engine": state.engine,
            "model_id": self._worker_model, "active": self._active,
            "worker_up": self._proc is not None,
            "max_concurrent": asr_max_concurrent(self.cfg),
            "coexist": bool(self.cfg.asr_coexist),
        }

    # ---- main entry ----
    async def run(self, *, model_id: str, engine: str, profile: Profile | None,
                  req: AudioRequest, request_id: str, origin_name: str,
                  progress_cb: Callable[[ProgressEvent], Any] | None = None,
                  cancel_event: asyncio.Event | None = None) -> AudioResult:
        if self.in_cooldown:
            raise AudioError(
                f"audio engine in cooldown for "
                f"{int(self._cooldown_until - time.time())}s after failures")
        if engine not in engines.ADAPTERS:
            raise AudioError(f"unknown audio engine: {engine!r}")
        adapter = engines.get(engine)
        _validate_model_id(model_id)
        base = self.cfg.asr_models_dir
        model_path = _safe_under(base, base / model_id)
        if not model_path.exists():
            raise AudioError(f"model not found: {model_path}")
        if not req.audio_path.exists():
            raise AudioError(f"audio file not found: {req.audio_path}")
        profile = profile or Profile(name="(none)")

        # Resolve word-timestamps: per-request wins, else the profile toggle.
        word_ts = bool(req.word_timestamps) or (
            (profile.audio_word_timestamps or "").lower() == "on")

        # Coexist alongside the LLM only when configured AND currently feasible.
        # The feasibility check is what prevents the OOM that took the box down:
        # if host memory is already tight, running Whisper next to a big LLM
        # will thrash — so we fall through to the exclusive path (unload the LLM
        # for this task) instead of stacking on top of it.
        if self.cfg.asr_coexist and self._coexist_feasible():
            # Warm, concurrent: keep the worker loaded alongside the LLM.
            await self._ensure_worker(model_id, model_path, engine)
            return await self._proxy(model_id, engine, profile, req, request_id, word_ts)
        if self.cfg.asr_coexist:
            log.info("asr: coexist requested but host memory is tight — "
                     "unloading the LLM for this transcription instead")
        # coexist=off (or not feasible): ASR owns the GPU — the other engine is
        # unloaded (refcounted, so concurrent transcriptions share one unload)
        # and the warm worker serves them all. The worker + yield are released
        # when the last concurrent task finishes (or the worker idle-stops).
        await self._acquire_llm_yield()
        try:
            await self._ensure_worker(model_id, model_path, engine)
            return await self._proxy(model_id, engine, profile, req,
                                     request_id, word_ts)
        finally:
            await self._release_llm_yield()

    async def _proxy(self, model_id, engine, profile, req, request_id,
                     word_ts) -> AudioResult:
        self._active += 1
        self._last_used = time.time()
        self._set_state("transcribing", engine, model_id, profile.name, request_id)
        try:
            assert self._client is not None
            resp = await self._client.post(
                f"{self._base_url}/transcribe",
                json={"path": str(req.audio_path),
                      "language": req.language, "task": req.task,
                      "word_timestamps": word_ts},
                timeout=TRANSCRIBE_TIMEOUT_S)
            env = resp.json()
            if resp.status_code != 200:
                raise AudioError(env.get("error") or f"worker {resp.status_code}")
            self._failures.clear()
            self.db.log_event("audio_transcribe_done", {
                "request_id": request_id, "engine": engine, "model": model_id})
            return AudioResult(
                request_id=request_id, engine=engine, model_id=model_id,
                profile_name=profile.name, text=str(env.get("text", "")),
                language=env.get("language"),
                duration_s=float(env.get("audio_ms") or 0) / 1000.0,
                segments=env.get("segments") or [], raw=env)
        except AudioError:
            self._record_failure(); raise
        except Exception as e:  # noqa: BLE001
            self._record_failure()
            log.exception("transcription proxy failed for %s", request_id)
            raise AudioError(str(e)) from e
        finally:
            self._active = max(0, self._active - 1)
            self._last_used = time.time()
            if self._active == 0:
                self._set_state("idle", None, None, None, None)

    # ---- streaming helpers (Phase 2) ----
    async def ensure_worker(self, model_id: str, model_path: Path,
                            engine: str = "asr") -> None:
        """Public: make sure the warm worker for ``model_id`` is up. Used by a
        streaming session, which owns its own queue slot + coexistence."""
        await self._ensure_worker(model_id, model_path, engine)

    async def transcribe_pcm(self, pcm: bytes, *, language: str | None,
                             task: str, word_timestamps: bool = True) -> dict:
        """Transcribe a raw float32 16 kHz mono PCM buffer via the warm worker
        (no temp file / ffmpeg). The caller must have ensured the worker."""
        assert self._client is not None and self._base_url
        from urllib.parse import urlencode
        qs = urlencode({"language": language or "", "task": task,
                        "word_timestamps": "1" if word_timestamps else "0"})
        self._last_used = time.time()
        resp = await self._client.post(
            f"{self._base_url}/transcribe_pcm?{qs}", content=pcm,
            headers={"Content-Type": "application/octet-stream"},
            timeout=TRANSCRIBE_TIMEOUT_S)
        env = resp.json()
        if resp.status_code != 200:
            raise AudioError(env.get("error") or f"worker {resp.status_code}")
        return env

    async def stream_pcm(self, pcm: bytes, *, sid: str, final: bool,
                         language: str | None, task: str,
                         word_timestamps: bool = True) -> dict:
        """Feed the *new* PCM for session ``sid`` into the worker's stateful
        native-streaming endpoint (``/stream_pcm``). Only valid for engines that
        advertise ``native_streaming`` (sherpa-onnx). The caller must have
        ensured the worker."""
        assert self._client is not None and self._base_url
        from urllib.parse import urlencode
        qs = urlencode({"sid": sid, "final": "1" if final else "0",
                        "language": language or "", "task": task,
                        "word_timestamps": "1" if word_timestamps else "0"})
        self._last_used = time.time()
        resp = await self._client.post(
            f"{self._base_url}/stream_pcm?{qs}", content=pcm,
            headers={"Content-Type": "application/octet-stream"},
            timeout=TRANSCRIBE_TIMEOUT_S)
        env = resp.json()
        if resp.status_code != 200:
            raise AudioError(env.get("error") or f"worker {resp.status_code}")
        return env

    # ---- worker lifecycle ----
    async def _ensure_worker(self, model_id: str, model_path: Path,
                             engine: str = "asr") -> None:
        if (self._proc is not None and self._worker_model == model_id
                and self._worker_engine == engine and await self._healthy()):
            return
        async with self._start_lock:
            if (self._proc is not None and self._worker_model == model_id
                    and self._worker_engine == engine and await self._healthy()):
                return
            await self._stop_worker_locked()
            if self._client is None:
                self._client = httpx.AsyncClient()
            port = _free_port()
            argv, env = engines.get(engine).build_worker_command(
                self.cfg, model_path, port, asr_max_concurrent(self.cfg))
            log_path = self.cfg.logs_dir / "asr.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(log_path, "a", encoding="utf-8")
            self._proc = await asyncio.create_subprocess_exec(
                *argv, stdout=logf, stderr=logf,
                stdin=asyncio.subprocess.DEVNULL,
                env={**__import__("os").environ, **env})
            self._base_url = f"http://127.0.0.1:{port}"
            self._worker_model = model_id
            self._worker_engine = engine
            log.info("asr worker (%s) starting for %s on %s",
                     engine, model_id, self._base_url)
            try:
                await self._wait_healthy(WORKER_START_TIMEOUT_S)
            except Exception:
                await self._stop_worker_locked()
                self._record_failure()
                raise AudioError("ASR worker failed to become healthy in time")
            if self._idle_task is None or self._idle_task.done():
                self._idle_task = asyncio.create_task(self._idle_monitor())

    async def _healthy(self) -> bool:
        if self._client is None or not self._base_url:
            return False
        try:
            r = await self._client.get(f"{self._base_url}/healthz", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def _wait_healthy(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.returncode is not None:
                raise AudioError(f"worker exited early rc={self._proc.returncode}")
            if await self._healthy():
                return
            await asyncio.sleep(1.0)
        raise AudioError("worker health timeout")

    async def _idle_monitor(self) -> None:
        idle = int(getattr(self.cfg, "asr_idle_timeout_s", 0) or 0)
        if idle <= 0:
            return
        while True:
            await asyncio.sleep(IDLE_CHECK_INTERVAL_S)
            if self._proc is None:
                return
            if self._active == 0 and (time.time() - self._last_used) > idle:
                async with self._start_lock:
                    if self._active == 0 and (time.time() - self._last_used) > idle:
                        log.info("asr worker idle %ds — stopping to free VRAM", idle)
                        await self._stop_worker_locked()
                        return

    async def _acquire_llm_yield(self) -> None:
        """Ensure the other engine (LLM / diffusion) is unloaded while audio is
        in flight. Refcounted: the first concurrent task performs the unload;
        later ones just bump the count. No-op when there's no ServerManager."""
        async with self._yield_lock:
            self._yield_refs += 1
            if self._yield_refs == 1 and self.sm is not None and self._yield_cm is None:
                cm = self.sm.yield_to_image(reason="asr")
                await cm.__aenter__()
                self._yield_cm = cm

    async def _release_llm_yield(self) -> None:
        """Drop one audio in-flight ref. When the last finishes, stop the warm
        worker (free its VRAM) and restore the yielded engine."""
        async with self._yield_lock:
            self._yield_refs = max(0, self._yield_refs - 1)
            if self._yield_refs == 0:
                async with self._start_lock:
                    await self._stop_worker_locked()   # free the worker's VRAM
                await self._exit_yield()               # restore the other engine

    async def _exit_yield(self) -> None:
        """Restore the yielded LLM/diffusion engine (coexist=off). Separate from
        worker teardown so ``_ensure_worker``'s stop-then-start doesn't
        accidentally restore the engine mid-transcription."""
        if self._yield_cm is not None:
            with contextlib.suppress(Exception):
                await self._yield_cm.__aexit__(None, None, None)
            self._yield_cm = None

    async def _stop_worker_locked(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
            self._proc = None
        self._worker_model = None
        self._worker_engine = None
        self._base_url = ""

    async def stop(self) -> None:
        """Shutdown hook: stop the worker + restore any yielded engine."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_task
            self._idle_task = None
        async with self._start_lock:
            await self._stop_worker_locked()
        await self._exit_yield()
        self._yield_refs = 0
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- state / failure policy ----
    def _set_state(self, status: str, engine, model_id, profile, request_id) -> None:
        state = rt.load(self.cfg.runtime_path)
        state.audio = rt.AudioRuntimeState(
            status=status, engine=engine, model_id=model_id, profile=profile,
            request_id=request_id, started_at=time.time(),
            last_event_at=time.time())
        rt.save(self.cfg.runtime_path, state)

    def _record_failure(self) -> None:
        now = time.time()
        self._failures.append(now)
        cutoff = now - self.cfg.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        if len(self._failures) >= self.cfg.max_restarts_in_window:
            self._cooldown_until = now + self.cfg.window_seconds
            self.db.log_event("audio_engine_degraded",
                              {"failures": len(self._failures)})
            log.error("audio runner: %d failures in %ds → %ds cooldown",
                      len(self._failures), self.cfg.window_seconds,
                      self.cfg.window_seconds)
