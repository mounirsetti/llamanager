"""Live streaming ASR session (Phase 2).

The daemon owns the WebSocket and the streaming loop; the warm worker stays a
simple transcribe service. A session:

  1. spawns ``ffmpeg`` to decode the client's incoming audio (any container —
     browser webm/opus, CLI-piped wav/raw) into float32 16 kHz mono PCM,
  2. accumulates PCM in a bounded window (committing + sliding a stable prefix
     for long audio),
  3. every ``decode_interval_s`` re-transcribes the window via the worker's
     ``/transcribe_pcm`` and pushes a revised partial
     ``{type:"transcript", rev, final:false, audio_ms, words:[{w,t0,t1,p}]}``,
  4. on stop / disconnect emits a ``final:true`` transcript.

Revisions are whole-hypothesis (re-decode) — the standard pseudo-streaming
approach for a non-streaming model. A committed prefix keeps compute bounded
and stops already-spoken words from jittering.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 4                 # float32
WINDOW_MAX_MS = 28_000              # re-decode window ceiling
COMMIT_TAIL_MS = 3_000             # keep this much tail uncommitted (unstable)


def _bytes_to_ms(n: int) -> int:
    return int(1000 * (n // BYTES_PER_SAMPLE) / SAMPLE_RATE)


def _ms_to_bytes(ms: int) -> int:
    return int(ms / 1000 * SAMPLE_RATE) * BYTES_PER_SAMPLE


class AsrStreamSession:
    def __init__(self, *, cfg, runner, sm, ws, model_id: str, model_path: Path,
                 language: str | None, task: str, decode_interval_s: float,
                 prebuffer: bytes = b""):
        self.cfg = cfg
        self.runner = runner
        self.sm = sm
        self.ws = ws
        self.model_id = model_id
        self.model_path = model_path
        self.language = language
        self.task = task
        self.interval = max(0.2, float(decode_interval_s or 1.0))
        self._buf = bytearray()          # uncommitted PCM (float32 bytes)
        self._committed_words: list[dict] = []
        self._committed_ms = 0
        self._rev = 0
        self._eof = asyncio.Event()
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._prebuffer = prebuffer

    async def run(self) -> None:
        # coexist=off → hold the text server yielded for the whole session.
        cm = (self.sm.yield_to_image(reason="asr-stream")
              if (self.sm is not None and not self.cfg.asr_coexist)
              else _nullcontext())
        async with cm:
            await self.runner.ensure_worker(self.model_id, self.model_path)
            self._ffmpeg = await asyncio.create_subprocess_exec(
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                # low-latency input: don't buffer/probe ahead, emit PCM as
                # audio arrives so partials track the live stream. probesize
                # 32k is enough to detect webm/opus (browser MediaRecorder) and
                # wav (CLI) without waiting on a long analyze pass.
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-probesize", "32k", "-analyzeduration", "0",
                "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar",
                str(SAMPLE_RATE), "-flush_packets", "1", "pipe:1",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            if self._prebuffer and self._ffmpeg.stdin:
                self._ffmpeg.stdin.write(self._prebuffer)
                with contextlib.suppress(Exception):
                    await self._ffmpeg.stdin.drain()
            await self._send({"type": "ready", "model": self.model_id,
                              "decode_interval_s": self.interval})
            reader = asyncio.create_task(self._pump_ws())
            collector = asyncio.create_task(self._collect_pcm())
            try:
                await self._decode_loop()
            finally:
                for t in (reader, collector):
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
                with contextlib.suppress(ProcessLookupError):
                    if self._ffmpeg and self._ffmpeg.returncode is None:
                        self._ffmpeg.kill()

    # ---- client audio in ----
    async def _pump_ws(self) -> None:
        """WS binary frames → ffmpeg stdin. Text ``{type:stop}`` or a
        disconnect closes stdin (EOF)."""
        from starlette.websockets import WebSocketDisconnect
        assert self._ffmpeg and self._ffmpeg.stdin
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    self._ffmpeg.stdin.write(data)
                    with contextlib.suppress(Exception):
                        await self._ffmpeg.stdin.drain()
                    continue
                text = msg.get("text")
                if text:
                    with contextlib.suppress(Exception):
                        if json.loads(text).get("type") == "stop":
                            break
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("asr-stream ws pump ended: %s", e)
        finally:
            with contextlib.suppress(Exception):
                self._ffmpeg.stdin.close()

    async def _collect_pcm(self) -> None:
        """ffmpeg stdout PCM → the rolling window buffer. Sets EOF on end."""
        assert self._ffmpeg and self._ffmpeg.stdout
        try:
            while True:
                chunk = await self._ffmpeg.stdout.read(16384)
                if not chunk:
                    break
                self._buf += chunk
        finally:
            self._eof.set()

    # ---- decode loop ----
    async def _decode_loop(self) -> None:
        while True:
            done = self._eof.is_set()
            if len(self._buf) >= BYTES_PER_SAMPLE:
                await self._decode_and_emit(final=done)
                self._maybe_commit()
            if done:
                # one last decode already emitted final above (or below)
                if len(self._buf) < BYTES_PER_SAMPLE and self._rev == 0:
                    await self._send({"type": "transcript", "rev": 1,
                                      "final": True, "audio_ms": 0, "words": []})
                break
            try:
                await asyncio.wait_for(self._eof.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
        await self._send({"type": "done"})

    async def _decode_and_emit(self, *, final: bool) -> None:
        window = bytes(self._buf)
        if not window:
            return
        try:
            env = await self.runner.transcribe_pcm(
                window, language=self.language, task=self.task,
                word_timestamps=True)
        except Exception as e:  # noqa: BLE001
            await self._send({"type": "error", "error": str(e)})
            return
        words = []
        for w in env.get("words", []):
            words.append({"w": w.get("w", ""),
                          "t0": self._committed_ms + int(w.get("t0", 0)),
                          "t1": self._committed_ms + int(w.get("t1", 0)),
                          **({"p": w["p"]} if "p" in w else {})})
        self._rev += 1
        self._last_hyp = words
        audio_ms = self._committed_ms + _bytes_to_ms(len(window))
        await self._send({
            "type": "transcript", "rev": self._rev, "final": final,
            "audio_ms": audio_ms, "text": env.get("text", ""),
            "words": self._committed_words + words})

    def _maybe_commit(self) -> None:
        """When the window grows past the ceiling, commit the stable prefix
        (words older than the unstable tail) and drop that audio."""
        win_ms = _bytes_to_ms(len(self._buf))
        if win_ms <= WINDOW_MAX_MS:
            return
        cutoff_rel_ms = win_ms - COMMIT_TAIL_MS
        stable, rest = [], []
        for w in getattr(self, "_last_hyp", []):
            rel_t1 = w["t1"] - self._committed_ms
            (stable if rel_t1 <= cutoff_rel_ms else rest).append(w)
        if not stable:
            return
        self._committed_words += stable
        drop_bytes = min(len(self._buf), _ms_to_bytes(cutoff_rel_ms))
        del self._buf[:drop_bytes]
        self._committed_ms += _bytes_to_ms(drop_bytes)

    async def _send(self, obj: dict) -> None:
        with contextlib.suppress(Exception):
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))


@contextlib.asynccontextmanager
async def _nullcontext():
    yield
