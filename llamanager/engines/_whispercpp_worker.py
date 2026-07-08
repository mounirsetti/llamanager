"""Persistent whisper.cpp worker — a warm GGML ASR server (stdlib only).

Launched and managed by ``audio_runner.AudioTaskRunner`` and speaking the shared
loopback protocol (``/healthz``, ``/transcribe``, ``/transcribe_pcm``) so
everything downstream is engine-agnostic.

Speed: this shim runs the native **``whisper-server``** once — the model + the
Vulkan context load a single time and stay warm — and proxies each request to
its ``/inference`` endpoint. (The earlier design shelled ``whisper-cli`` per
request, which reloaded the model + re-initialised Vulkan on *every* decode: the
5 s streaming latency.) The AMD card is pinned via ``GGML_VK_VISIBLE_DEVICES``
resolved from the operator's GPU pin, so it doesn't land on a slow iGPU.

Output shape matches the transformers engine: whole **words** with
``{w, t0, t1, p}``. ``whisper-server`` with ``max_len=1 & split_on_word`` returns
one segment per word; ``p`` = ``exp(avg_logprob)``. Word timestamps from
fine-tuned models can collapse (DTW alignment drift), so we detect broken timing
and fall back to a proportional spread — same as the transformers worker.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLE_RATE = 16000

_server_url = ""          # http://127.0.0.1:<port> of the native whisper-server
_proc: subprocess.Popen | None = None
_model_name = ""
_sem: threading.Semaphore | None = None


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- Vulkan device pin -------------------------------------------------

def _resolve_vk_index(whisper_cli: str, model: str, device_name: str,
                      env: dict) -> str | None:
    """Return the ggml Vulkan device index (as a string) to pin: the operator's
    named GPU if it matches, else a discrete GPU (avoiding the iGPU). Delegates
    to the shared, cached resolver in gpu_detect, which enumerates via the ggml
    binary's own device list and kills it before the (slow) model load — so this
    is ~1 s, not a full model load. None → let ggml auto-pick.

    Runs under llamanager's own interpreter, so importing the package is safe."""
    from llamanager.gpu_detect import resolve_ggml_vulkan_index
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(_silent_wav_bytes())
        wav = f.name
    try:
        idx = resolve_ggml_vulkan_index(
            whisper_cli, ["-m", model, "-f", wav], device_name, env=env)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(wav)
    return str(idx) if idx is not None else None


def _silent_wav_bytes() -> bytes:
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * 1600)   # 0.1 s of silence
    return buf.getvalue()


# ---- audio → wav -------------------------------------------------------

def _decode_to_wav(src: Path, dst: Path) -> None:
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


def _pcm_f32_to_wav_bytes(raw: bytes) -> tuple[bytes, int]:
    """Wrap raw float32 mono 16 kHz PCM as a 16-bit WAV. Returns (bytes, ms)."""
    import io
    floats = array.array("f"); floats.frombytes(raw)
    ints = array.array("h", (max(-32768, min(32767, int(s * 32767.0)))
                             for s in floats))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(ints.tobytes())
    return buf.getvalue(), int(round(1000 * len(floats) / SAMPLE_RATE))


# ---- whisper-server /inference client ----------------------------------

def _infer(wav_bytes: bytes, language: str | None, task: str,
           word_mode: bool) -> dict:
    """POST a WAV to the warm whisper-server /inference and return its parsed
    verbose_json. Multipart built by hand (stdlib only)."""
    boundary = uuid.uuid4().hex
    fields = {"response_format": "verbose_json",
              "temperature": "0.0"}
    lang = (language or "").strip().lower()
    fields["language"] = lang if (lang and lang != "auto") else "auto"
    if task == "translate":
        fields["translate"] = "true"
    if word_mode:
        fields["max_len"] = "1"
        fields["split_on_word"] = "true"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="file"; filename="a.wav"\r\n'
                 "Content-Type: audio/wav\r\n\r\n".encode())
    parts.append(wav_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{_server_url}/inference", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(body))})
    assert _sem is not None
    with _sem:
        with urllib.request.urlopen(req, timeout=30 * 60) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))


def _p_from_logprob(seg: dict) -> float | None:
    lp = seg.get("avg_logprob")
    if lp is None:
        return None
    try:
        return round(min(1.0, max(0.0, math.exp(float(lp)))), 3)
    except (ValueError, OverflowError):
        return None


def _envelope(doc: dict, language: str | None, task: str,
              word_mode: bool, audio_ms: int) -> dict:
    segs = doc.get("segments") or []
    text = (doc.get("text") or "").strip()
    env: dict = {"type": "transcript", "rev": 1, "final": True,
                 "audio_ms": audio_ms, "text": text,
                 "language": (language or None), "task": task}
    if word_mode:
        # With max_len=1 & split_on_word, each segment is one whole word.
        words = []
        for s in segs:
            w = (s.get("text") or "").strip()
            if not w:
                continue
            words.append({"w": w,
                          "t0": int(round(float(s.get("start", 0.0)) * 1000)),
                          "t1": int(round(float(s.get("end", 0.0)) * 1000)),
                          **({"p": _p_from_logprob(s)}
                             if _p_from_logprob(s) is not None else {})})
        env["words"] = _sanitize_word_timings(words, audio_ms)
    else:
        env["segments"] = [{"start": round(float(s.get("start", 0.0)), 2),
                            "end": round(float(s.get("end", 0.0)), 2),
                            "text": (s.get("text") or "").strip()}
                           for s in segs]
    return env


def _sanitize_word_timings(words: list[dict], audio_ms: int) -> list[dict]:
    """Fine-tuned models' word timestamps can collapse (all pinned to one time,
    or the last jumping to the clip end). When the timing is clearly broken,
    fall back to a proportional spread across [0, audio_ms] by word length —
    rough but ordered and bounded. ``w``/``p`` are untouched. Mirrors the
    transformers worker so both engines behave identically on-device."""
    if not words or audio_ms <= 0:
        return words
    t0s = [w["t0"] for w in words]
    t1s = [w["t1"] for w in words]
    collapsed = len(set(t0s)) <= max(1, len(words) // 2)
    broken = (collapsed or max(t1s) > audio_ms * 1.25 or min(t0s) < -200
              or any(t1s[i] < t0s[i] for i in range(len(words))))
    if not broken:
        return words
    total = sum(max(1, len(w["w"])) for w in words)
    acc = 0
    for w in words:
        span = max(1, len(w["w"]))
        w["t0"] = int(round(audio_ms * acc / total))
        acc += span
        w["t1"] = int(round(audio_ms * acc / total))
        w["approx_timing"] = True
    return words


# ---- HTTP shim (our stable protocol) -----------------------------------

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
            self._send(200, {"ok": _server_ready(), "model": _model_name})
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
            self._send(400, {"error": f"bad request: {e}"}); return
        path = Path(str(req.get("path", "")))
        if not path.is_file():
            self._send(400, {"error": f"file not found: {path}"}); return
        task = (req.get("task") or "transcribe")
        wts = bool(req.get("word_timestamps"))
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / "in.wav"
                _decode_to_wav(path, wav)
                data = wav.read_bytes()
                audio_ms = _wav_ms(wav)
            doc = _infer(data, req.get("language"), task, wts)
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
                self._send(400, {"error": "empty pcm"}); return
            lang = (q.get("language", [""])[0] or None)
            task = (q.get("task", ["transcribe"])[0] or "transcribe")
            wts = (q.get("word_timestamps", ["1"])[0] in ("1", "true", "on", "yes"))
            wav_bytes, audio_ms = _pcm_f32_to_wav_bytes(raw)
            doc = _infer(wav_bytes, lang, task, wts)
            self._send(200, _envelope(doc, lang, task, wts, audio_ms))
        except Exception as e:  # noqa: BLE001
            _log(f"[whispercpp-worker] transcribe_pcm error: {e}")
            self._send(500, {"error": str(e)})


def _wav_ms(wav: Path) -> int:
    with wave.open(str(wav), "rb") as w:
        return int(round(1000 * w.getnframes() / (w.getframerate() or SAMPLE_RATE)))


def _server_ready() -> bool:
    if not _server_url:
        return False
    try:
        with urllib.request.urlopen(f"{_server_url}/", timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    global _server_url, _proc, _model_name, _sem
    p = argparse.ArgumentParser(description="Persistent whisper.cpp worker")
    p.add_argument("--whisper-server", required=True)
    p.add_argument("--whisper-cli", required=True)  # for device enumeration
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--max-concurrent", type=int, default=2)
    p.add_argument("--device-name", default="")
    args = p.parse_args()

    _model_name = args.model.name
    _sem = threading.Semaphore(max(1, args.max_concurrent))

    # Pin the requested GPU (resolve its Vulkan index from the operator's name).
    child_env = dict(os.environ)
    idx = _resolve_vk_index(args.whisper_cli, str(args.model),
                            args.device_name, child_env)
    if idx is not None:
        child_env["GGML_VK_VISIBLE_DEVICES"] = idx

    # Launch the native whisper-server on an internal port; it loads the model
    # + Vulkan context ONCE and stays warm.
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); inner = s.getsockname()[1]; s.close()
    _server_url = f"http://127.0.0.1:{inner}"
    argv = [args.whisper_server, "-m", str(args.model),
            "--host", "127.0.0.1", "--port", str(inner),
            "-t", str(max(1, args.max_concurrent))]
    _log(f"[whispercpp-worker] starting whisper-server: {' '.join(argv)}")
    _proc = subprocess.Popen(argv, env=child_env,
                             stdout=subprocess.DEVNULL, stderr=sys.stderr)
    deadline = time.time() + 180
    while time.time() < deadline:
        if _proc.poll() is not None:
            _log(f"[whispercpp-worker] whisper-server exited rc={_proc.returncode}")
            return 1
        if _server_ready():
            break
        time.sleep(0.5)
    _log(f"[whispercpp-worker] whisper-server warm on {_server_url}")

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.daemon_threads = True

    # The manager stops us with SIGTERM (or a group signal). Default SIGTERM
    # would kill this process before the finally-block below runs, orphaning the
    # native whisper-server (and leaking its model mmap + Vulkan context). Handle
    # it so we shut the server down and reap the child. serve_forever() runs in a
    # background thread because BaseServer.shutdown() deadlocks if called from the
    # thread running serve_forever() — and signal handlers run in the main thread.
    stop = threading.Event()

    def _shutdown(signum, _frame):
        _log(f"[whispercpp-worker] received signal {signum} — shutting down")
        stop.set()
    for _sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(_sig, _shutdown)

    serve = threading.Thread(target=httpd.serve_forever, daemon=True)
    serve.start()
    _log(f"[whispercpp-worker] ready on http://{args.host}:{args.port}")
    try:
        while not stop.is_set():
            # Exit promptly if the native server dies on its own, too.
            if _proc and _proc.poll() is not None:
                _log(f"[whispercpp-worker] whisper-server exited rc={_proc.returncode}")
                break
            stop.wait(1.0)
    finally:
        httpd.shutdown()
        if _proc and _proc.poll() is None:
            _proc.terminate()
            with contextlib.suppress(Exception):
                _proc.wait(timeout=10)
            if _proc.poll() is None:
                with contextlib.suppress(Exception):
                    _proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
