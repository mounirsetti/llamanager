"""Persistent sherpa-onnx worker — a warm streaming/offline ASR server.

Launched and managed by ``audio_runner.AudioTaskRunner`` like the other audio
workers, and speaking the same loopback protocol so the batch path is engine-
agnostic:

  * ``GET  /healthz``        → ``{"ok": true, "model": …, "online": bool}``
  * ``POST /transcribe``     → body ``{"path","language","task","word_timestamps"}``
  * ``POST /transcribe_pcm`` → raw float32 mono 16 kHz PCM body → transcript envelope

Plus a **native streaming** endpoint used only by ``asr_stream`` when the engine
advertises ``native_streaming`` (see engines/sherpa.py ``capabilities``):

  * ``POST /stream_pcm?sid=<id>&final=0|1`` → body = the *new* float32 PCM since
    the previous call for this session. The worker keeps a stateful online
    stream per ``sid``, feeds only the new audio, and returns the running
    hypothesis. ``final=1`` flushes and drops the session.

Runs in a small pip venv (``sherpa-onnx`` + ``numpy`` + onnxruntime — no torch).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLE_RATE = 16000

_rec = None            # sherpa_onnx recognizer (online or offline)
_online = False        # True → OnlineRecognizer (native streaming)
_model_name = ""
_sem: threading.Semaphore | None = None
_sessions: dict = {}   # sid → {"stream": ..., "committed_ms": int}
_sessions_lock = threading.Lock()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _decode_audio(path: Path):
    """Decode any input to a float32 mono 16 kHz numpy array via ffmpeg."""
    import numpy as np
    cmd = ["ffmpeg", "-nostdin", "-threads", "1", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH") from e
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"ffmpeg failed to decode {path.name}:\n{tail}") from e
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"decoded zero audio samples from {path.name}")
    return audio


# ---- recognizer construction -------------------------------------------

def _find(model_dir: Path, *needles: str) -> str | None:
    """First ``*.onnx`` whose name contains all ``needles``. When a repo ships
    multiple precisions (e.g. both ``…onnx`` and ``…int8.onnx``), prefer the
    full-precision file so the three transducer graphs stay consistent."""
    matches = [p for p in model_dir.iterdir()
               if p.is_file() and p.name.lower().endswith(".onnx")
               and all(x in p.name.lower() for x in needles)]
    if not matches:
        return None

    def rank(p: Path):
        n = p.name.lower()
        quantized = (".int8." in n) or (".fp16." in n) or (".quant." in n)
        return (1 if quantized else 0, n)

    matches.sort(key=rank)
    return str(matches[0])


def _build_recognizer(model_dir: Path, num_threads: int):
    """Build the right sherpa-onnx recognizer for the model on disk. Returns
    (recognizer, is_online). Streaming transducers → online (native streaming);
    Whisper-ONNX / Paraformer / CTC → offline."""
    import sherpa_onnx
    tokens = str(model_dir / "tokens.txt")
    enc = _find(model_dir, "encoder")
    dec = _find(model_dir, "decoder")
    joiner = _find(model_dir, "joiner")
    common = dict(tokens=tokens, num_threads=num_threads,
                  sample_rate=SAMPLE_RATE, feature_dim=80)
    if enc and dec and joiner:
        # Prefer a streaming recognizer; fall back to offline if the graphs are
        # not streaming-capable (from_transducer raises).
        try:
            rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=enc, decoder=dec, joiner=joiner,
                decoding_method="greedy_search", **common)
            return rec, True
        except Exception as e:  # noqa: BLE001
            _log(f"[sherpa-worker] not a streaming transducer ({e}); offline")
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=enc, decoder=dec, joiner=joiner,
                decoding_method="greedy_search", **common)
            return rec, False
    if enc and dec and not joiner:
        # Whisper-ONNX (encoder + decoder, no joiner) → offline whisper.
        _log("[sherpa-worker] Whisper-ONNX layout → OfflineRecognizer.from_whisper")
        rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=enc, decoder=dec, tokens=tokens, num_threads=num_threads)
        return rec, False
    # Paraformer / CTC single-graph → offline.
    para = _find(model_dir, "model") or _find(model_dir)
    if para:
        try:
            rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=para, **common)
            return rec, False
        except Exception:  # noqa: BLE001
            rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=para, **common)
            return rec, False
    raise RuntimeError(f"no recognisable sherpa-onnx graphs in {model_dir}")


# ---- result mapping -----------------------------------------------------

def _result_of(stream) -> tuple[str, list, list]:
    """Return (text, tokens, timestamps_seconds) from a decoded stream, tolerant
    of sherpa-onnx API drift across versions."""
    r = getattr(stream, "result", None)
    if r is None and _online:
        r = _rec.get_result(stream)
    if isinstance(r, str):
        return r, [], []
    text = getattr(r, "text", "") if r is not None else ""
    tokens = list(getattr(r, "tokens", []) or [])
    ts = list(getattr(r, "timestamps", []) or [])
    return text, tokens, ts


def _words_from(tokens: list, ts: list, offset_ms: int) -> list[dict]:
    """Group sentencepiece/BPE tokens into words with {w,t0,t1} (ms)."""
    words: list[dict] = []
    cur = None
    for i, tok in enumerate(tokens):
        piece = str(tok)
        t = int(round((ts[i] if i < len(ts) else 0.0) * 1000)) + offset_ms
        starts = piece.startswith("▁") or piece.startswith(" ") or cur is None
        clean = piece.replace("▁", " ")
        if starts and cur is not None and cur["w"].strip():
            words.append({"w": cur["w"].strip(), "t0": cur["t0"], "t1": cur["t1"]})
            cur = None
        if cur is None:
            cur = {"w": clean, "t0": t, "t1": t}
        else:
            cur["w"] += clean
            cur["t1"] = t
    if cur is not None and cur["w"].strip():
        words.append({"w": cur["w"].strip(), "t0": cur["t0"], "t1": cur["t1"]})
    return words


def _decode_full(audio, word_timestamps: bool, offset_ms: int = 0) -> dict:
    """Decode a complete float32 buffer (offline, or online run to completion)."""
    assert _rec is not None and _sem is not None
    with _sem:
        stream = _rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        if _online:
            import numpy as np
            stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.5 * SAMPLE_RATE),
                                                         dtype="float32"))
            stream.input_finished()
            while _rec.is_ready(stream):
                _rec.decode_stream(stream)
        else:
            _rec.decode_stream(stream)
    text, tokens, ts = _result_of(stream)
    dur_ms = int(round(1000 * len(audio) / SAMPLE_RATE))
    env: dict = {"type": "transcript", "rev": 1, "final": True,
                 "audio_ms": dur_ms, "text": text.strip()}
    if word_timestamps:
        env["words"] = _words_from(tokens, ts, offset_ms)
    else:
        env["segments"] = [{"start": 0.0, "end": round(dur_ms / 1000.0, 2),
                            "text": text.strip()}]
    return env


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
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
            self._send(200, {"ok": True, "model": _model_name, "online": _online})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route == "/transcribe":
            self._do_transcribe_file()
        elif route == "/transcribe_pcm":
            self._do_transcribe_pcm()
        elif route == "/stream_pcm":
            self._do_stream_pcm()
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
        try:
            audio = _decode_audio(path)
            env = _decode_full(audio, bool(req.get("word_timestamps")))
            env["language"] = req.get("language")
            env["task"] = req.get("task") or "transcribe"
            self._send(200, env)
        except Exception as e:  # noqa: BLE001
            _log(f"[sherpa-worker] transcribe error: {e}")
            self._send(500, {"error": str(e)})

    def _read_pcm(self):
        import numpy as np
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return np.frombuffer(raw, dtype=np.float32)

    def _do_transcribe_pcm(self):
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        try:
            audio = self._read_pcm()
            if audio.size == 0:
                self._send(400, {"error": "empty pcm"})
                return
            wts = (q.get("word_timestamps", ["1"])[0] in ("1", "true", "on", "yes"))
            env = _decode_full(audio, wts)
            env["language"] = q.get("language", [""])[0] or None
            env["task"] = q.get("task", ["transcribe"])[0] or "transcribe"
            self._send(200, env)
        except Exception as e:  # noqa: BLE001
            _log(f"[sherpa-worker] transcribe_pcm error: {e}")
            self._send(500, {"error": str(e)})

    def _do_stream_pcm(self):
        """Stateful streaming: feed only the new audio for this session."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        sid = q.get("sid", ["default"])[0]
        final = q.get("final", ["0"])[0] in ("1", "true", "on", "yes")
        if not _online:
            # No native streaming for offline models — behave like a re-decode.
            return self._do_transcribe_pcm()
        try:
            import numpy as np
            audio = self._read_pcm()
            with _sessions_lock:
                sess = _sessions.get(sid)
                if sess is None:
                    sess = {"stream": _rec.create_stream(), "committed_ms": 0}
                    _sessions[sid] = sess
                stream = sess["stream"]
            if audio.size:
                stream.accept_waveform(SAMPLE_RATE, audio)
                sess["committed_ms"] += int(round(1000 * audio.size / SAMPLE_RATE))
            assert _sem is not None
            with _sem:
                if final:
                    stream.accept_waveform(
                        SAMPLE_RATE, np.zeros(int(0.5 * SAMPLE_RATE), dtype="float32"))
                    stream.input_finished()
                while _rec.is_ready(stream):
                    _rec.decode_stream(stream)
            text, tokens, ts = _result_of(stream)
            env = {"type": "transcript", "rev": 1, "final": final,
                   "audio_ms": sess["committed_ms"], "text": text.strip(),
                   "words": _words_from(tokens, ts, 0)}
            if final:
                with _sessions_lock:
                    _sessions.pop(sid, None)
            self._send(200, env)
        except Exception as e:  # noqa: BLE001
            _log(f"[sherpa-worker] stream_pcm error: {e}")
            self._send(500, {"error": str(e)})


def main() -> int:
    global _rec, _online, _model_name, _sem
    p = argparse.ArgumentParser(description="Persistent sherpa-onnx worker")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--max-concurrent", type=int, default=2)
    args = p.parse_args()

    _model_name = args.model_path.name
    _sem = threading.Semaphore(max(1, args.max_concurrent))
    _log(f"[sherpa-worker] loading {args.model_path} "
         f"(max_concurrent={args.max_concurrent})")
    _rec, _online = _build_recognizer(args.model_path, max(1, args.max_concurrent))
    _log(f"[sherpa-worker] ready (online={_online})")

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.daemon_threads = True
    _log(f"[sherpa-worker] serving on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
