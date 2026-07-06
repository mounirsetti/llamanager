"""Persistent whisper.cpp worker — a warm GGML ASR server (stdlib only).

Launched and managed by ``audio_runner.AudioTaskRunner`` exactly like the
transformers ``_asr_worker.py``, and speaking the **same loopback protocol** so
everything downstream (proxy, streaming, word timestamps, UI) is engine-
agnostic:

  * ``GET  /healthz``        → ``{"ok": true, "model": …}``
  * ``POST /transcribe``     → body ``{"path","language","task","word_timestamps"}``
  * ``POST /transcribe_pcm`` → raw float32 mono 16 kHz PCM body; query
      ``language`` / ``task`` / ``word_timestamps``
  → the transcript envelope
    ``{"type":"transcript","rev":1,"final":true,"audio_ms":N,"text":…,
       "language":…,"task":…, "segments"|"words": …}``

Unlike the transformers worker this needs **no torch / numpy** — it shells out
to the native ``whisper-cli`` (built with Vulkan) and parses its JSON output.
Audio for ``/transcribe`` is normalised to 16 kHz mono WAV via the system
ffmpeg; ``/transcribe_pcm`` writes the incoming float32 PCM straight to a WAV
(int16) with the stdlib ``wave`` module. Concurrency is bounded by a semaphore
sized from the daemon's VRAM budget.
"""
from __future__ import annotations

import argparse
import array
import json
import subprocess
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLE_RATE = 16000

_whisper_cli = ""
_model = ""
_model_name = ""
_sem: threading.Semaphore | None = None


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _decode_to_wav(src: Path, dst: Path) -> None:
    """Normalise any input container to 16 kHz mono 16-bit WAV via ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-threads", "1", "-y", "-i", str(src),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", "-sample_fmt",
           "s16", str(dst)]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH") from e
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"ffmpeg failed to decode {src.name}:\n{tail}") from e


def _pcm_f32_to_wav(raw: bytes, dst: Path) -> int:
    """Write raw float32 mono 16 kHz PCM to a 16-bit WAV. Returns duration ms."""
    floats = array.array("f")
    floats.frombytes(raw)
    ints = array.array("h",
                       (max(-32768, min(32767, int(s * 32767.0))) for s in floats))
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(ints.tobytes())
    return int(round(1000 * len(floats) / SAMPLE_RATE))


def _run_whisper_cli(wav: Path, language: str | None, task: str,
                     word_timestamps: bool) -> dict:
    """Invoke whisper-cli, returning its parsed ``-oj`` JSON document."""
    with tempfile.TemporaryDirectory() as td:
        of = Path(td) / "out"
        argv = [_whisper_cli, "-m", _model, "-f", str(wav),
                "-oj", "-of", str(of), "-nt"]
        lang = (language or "").strip().lower()
        argv += ["-l", (lang if lang and lang != "auto" else "auto")]
        if task == "translate":
            argv.append("-tr")
        if word_timestamps:
            # one token per segment → word-level segments with offsets
            argv += ["-ml", "1"]
        assert _sem is not None
        with _sem:
            try:
                subprocess.run(argv, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                tail = (e.stderr or b"").decode("utf-8", "replace")[-1500:]
                raise RuntimeError(f"whisper-cli failed:\n{tail}") from e
        js = of.with_suffix(".json")
        if not js.is_file():
            raise RuntimeError("whisper-cli produced no JSON output")
        return json.loads(js.read_text(encoding="utf-8"))


def _envelope(doc: dict, language: str | None, task: str,
              word_timestamps: bool, audio_ms: int) -> dict:
    """Map whisper-cli's JSON to the shared transcript envelope."""
    entries = doc.get("transcription") or []
    text = "".join(e.get("text", "") for e in entries).strip()
    env: dict = {"type": "transcript", "rev": 1, "final": True,
                 "audio_ms": audio_ms, "text": text,
                 "language": (language or None), "task": task}
    if word_timestamps:
        words = []
        for e in entries:
            w = (e.get("text") or "").strip()
            if not w:
                continue
            off = e.get("offsets") or {}
            words.append({"w": w, "t0": int(off.get("from", 0)),
                          "t1": int(off.get("to", 0))})
        env["words"] = words
    else:
        segs = []
        for e in entries:
            off = e.get("offsets") or {}
            segs.append({"start": round(off.get("from", 0) / 1000.0, 2),
                         "end": round(off.get("to", 0) / 1000.0, 2),
                         "text": (e.get("text") or "").strip()})
        env["segments"] = segs
    return env


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence default access logging
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/healthz":
            self._send(200, {"ok": True, "model": _model_name})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route == "/transcribe":
            self._do_transcribe_file()
        elif route == "/transcribe_pcm":
            self._do_transcribe_pcm()
        else:
            self._send(404, {"error": "not found"})

    def _do_transcribe_file(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": f"bad request: {e}"})
            return
        path = Path(str(req.get("path", "")))
        if not path.is_file():
            self._send(400, {"error": f"file not found: {path}"})
            return
        task = (req.get("task") or "transcribe")
        wts = bool(req.get("word_timestamps"))
        try:
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / "in.wav"
                _decode_to_wav(path, wav)
                audio_ms = _wav_ms(wav)
                doc = _run_whisper_cli(wav, req.get("language"), task, wts)
            self._send(200, _envelope(doc, req.get("language"), task, wts, audio_ms))
        except Exception as e:  # noqa: BLE001
            _log(f"[whispercpp-worker] transcribe error: {e}")
            self._send(500, {"error": str(e)})

    def _do_transcribe_pcm(self):
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not raw:
                self._send(400, {"error": "empty pcm"})
                return
            lang = (q.get("language", [""])[0] or None)
            task = (q.get("task", ["transcribe"])[0] or "transcribe")
            wts = (q.get("word_timestamps", ["1"])[0] in ("1", "true", "on", "yes"))
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / "in.wav"
                audio_ms = _pcm_f32_to_wav(raw, wav)
                doc = _run_whisper_cli(wav, lang, task, wts)
            self._send(200, _envelope(doc, lang, task, wts, audio_ms))
        except Exception as e:  # noqa: BLE001
            _log(f"[whispercpp-worker] transcribe_pcm error: {e}")
            self._send(500, {"error": str(e)})


def _wav_ms(wav: Path) -> int:
    with wave.open(str(wav), "rb") as w:
        return int(round(1000 * w.getnframes() / (w.getframerate() or SAMPLE_RATE)))


def main() -> int:
    global _whisper_cli, _model, _model_name, _sem
    p = argparse.ArgumentParser(description="Persistent whisper.cpp worker")
    p.add_argument("--whisper-cli", required=True)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--max-concurrent", type=int, default=2)
    args = p.parse_args()

    _whisper_cli = args.whisper_cli
    _model = str(args.model)
    _model_name = args.model.name
    _sem = threading.Semaphore(max(1, args.max_concurrent))
    _log(f"[whispercpp-worker] {_whisper_cli} model={_model} "
         f"(max_concurrent={args.max_concurrent})")

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.daemon_threads = True
    _log(f"[whispercpp-worker] ready on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
