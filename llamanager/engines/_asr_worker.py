"""Persistent ASR worker — a warm Whisper inference server.

Launched and managed by ``audio_runner.AsrWorker``. Loads the model **once**
and serves many concurrent transcriptions over a loopback HTTP endpoint —
the same role ``llama-server`` plays for text. Runs inside the ASR venv
(torch + transformers).

Endpoints (loopback only):
  * ``GET  /healthz``    → ``{"ok": true, "model": …}``
  * ``POST /transcribe`` → body ``{"path","language","task","word_timestamps"}``
      (the audio file is on the shared host, so we pass a path, not bytes)
      → the transcript envelope
        ``{"type":"transcript","rev":1,"final":true,"audio_ms":N,
           "text":…,"language":…,"words":[{"w","t0","t1","p"}]}``  (words only
      when requested; otherwise ``segments`` as before).

Concurrency: a bounded semaphore (``--max-concurrent``, handed down from the
daemon's VRAM budget). Each request builds its **own** ``GenerationConfig``
copy — the model's ``generation_config`` is never mutated per request, so
concurrent decodes with different languages don't race.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
MAX_NEW_TOKENS = 440

_model = None
_processor = None
_device = "cpu"
_dtype = None
_sem: threading.Semaphore | None = None
_model_name = ""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _decode_audio(path: Path):
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


def _select_device() -> tuple[str, str]:
    import torch
    if torch.cuda.is_available():
        return "cuda", "float16"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


# Published cross-attention alignment heads (OpenAI whisper) needed for
# accurate word timestamps. Keyed by decoder-layer count. large-v3-turbo (4
# decoder layers) is the common case; fine-tunes of it strip the value from
# their generation_config, so we restore it here. Heuristic fallback otherwise.
_ALIGNMENT_HEADS_BY_LAYERS = {
    4: [[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]],  # large-v3-turbo
}


def _inject_alignment_heads() -> None:
    """Word timestamps need ``generation_config.alignment_heads``. Restore the
    published heads for known architectures (fine-tunes strip them); fall back
    to a rough heuristic (last two decoder layers) for unknown ones."""
    gc = _model.generation_config
    if getattr(gc, "alignment_heads", None):
        return
    c = _model.config
    L, H = c.decoder_layers, c.decoder_attention_heads
    known = _ALIGNMENT_HEADS_BY_LAYERS.get(L)
    if known:
        gc.alignment_heads = known
        _log(f"[asr-worker] restored published alignment_heads for {L}-layer decoder")
        return
    gc.alignment_heads = [[l, h] for l in range(max(0, L - 2), L) for h in range(H)]
    _log(f"[asr-worker] no alignment_heads in config; heuristic "
         f"({len(gc.alignment_heads)} heads over last 2 of {L} layers) — "
         "word timings will be approximate")


def _forced_ids(language: str | None, task: str):
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        return None
    try:
        return _processor.get_decoder_prompt_ids(language=lang, task=task)
    except (KeyError, ValueError):
        return None


def _field(out, key):
    """Whisper's word-timestamp generate returns a plain dict in some
    transformers versions and a ModelOutput in others — read either."""
    if isinstance(out, dict):
        return out.get(key)
    return getattr(out, key, None)


def _words_from(out, offset_ms: int) -> list[dict]:
    """Group generated tokens into words with {w, t0, t1, p} (ms, probability).

    Best-effort: leading-space token boundaries; per-word ``p`` = exp(mean
    token log-prob). Any failure degrades to no words rather than erroring."""
    try:
        sequences = _field(out, "sequences")
        seq = sequences[0]
        tts = _field(out, "token_timestamps")[0].tolist()
        scores = _field(out, "scores")
        n_gen = len(scores) if scores else 0
        prompt_len = len(seq) - n_gen if n_gen else 0
        logprobs = None
        if n_gen:
            try:
                tr = _model.compute_transition_scores(
                    sequences, scores, normalize_logits=True)
                logprobs = tr[0].float().tolist()
            except Exception:  # noqa: BLE001
                logprobs = None
        tok = _processor.tokenizer
        special = set(tok.all_special_ids)
        words: list[dict] = []
        cur = None
        for j in range(prompt_len, len(seq)):
            tid = int(seq[j])
            if tid in special:
                continue
            piece = tok.decode([tid])
            t = float(tts[j]) if j < len(tts) else 0.0
            lp = logprobs[j - prompt_len] if (logprobs and j - prompt_len < len(logprobs)) else None
            starts_word = piece.startswith(" ") or piece.startswith("‏") or cur is None
            if starts_word and cur is not None and cur["_w"].strip():
                words.append(_finish_word(cur, offset_ms))
                cur = None
            if cur is None:
                cur = {"_w": piece, "_t0": t, "_t1": t, "_lps": []}
            else:
                cur["_w"] += piece
                cur["_t1"] = t
            if lp is not None:
                cur["_lps"].append(lp)
        if cur is not None and cur["_w"].strip():
            words.append(_finish_word(cur, offset_ms))
        return words
    except Exception as e:  # noqa: BLE001
        _log(f"[asr-worker] word grouping failed: {e}")
        return []


def _finish_word(cur: dict, offset_ms: int) -> dict:
    import math
    w = {"w": cur["_w"].strip(),
         "t0": offset_ms + int(round(cur["_t0"] * 1000)),
         "t1": offset_ms + int(round(cur["_t1"] * 1000))}
    if cur["_lps"]:
        w["p"] = round(math.exp(sum(cur["_lps"]) / len(cur["_lps"])), 3)
    return w


def transcribe(path: Path, language: str | None, task: str,
               word_timestamps: bool) -> dict:
    return transcribe_audio(_decode_audio(path), language, task, word_timestamps)


def transcribe_audio(audio, language: str | None, task: str,
                     word_timestamps: bool) -> dict:
    """Core transcription over a float32 mono 16 kHz numpy array. Shared by the
    file endpoint and the streaming raw-PCM endpoint."""
    import torch
    dur_ms = int(round(1000 * audio.size / SAMPLE_RATE))
    n_chunks = max(1, (audio.size + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES)
    fid = _forced_ids(language, task)
    texts: list[str] = []
    words: list[dict] = []
    segments: list[dict] = []
    for i in range(n_chunks):
        start = i * CHUNK_SAMPLES
        win = audio[start:start + CHUNK_SAMPLES]
        feats = _processor(win, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt").input_features.to(_device, _dtype)
        gc = copy.deepcopy(_model.generation_config)
        gc.forced_decoder_ids = fid
        assert _sem is not None
        with _sem:
            with torch.no_grad():
                if word_timestamps:
                    # num_frames = actual (unpadded) mel frames so token
                    # timestamps scale to the real audio, not the 30 s pad.
                    # Whisper hop length is 160 samples (100 frames/s @ 16 kHz).
                    num_frames = max(1, int(win.size // 160))
                    out = _model.generate(
                        feats, generation_config=gc,
                        return_token_timestamps=True, num_frames=num_frames,
                        return_dict_in_generate=True, output_scores=True,
                        max_new_tokens=MAX_NEW_TOKENS)
                    seq = _field(out, "sequences")[0]
                else:
                    out = _model.generate(feats, generation_config=gc,
                                          max_new_tokens=MAX_NEW_TOKENS)
                    seq = out[0]
        text = _processor.decode(seq, skip_special_tokens=True).strip()
        texts.append(text)
        segments.append({
            "start": round(start / SAMPLE_RATE, 2),
            "end": round(min(start + CHUNK_SAMPLES, audio.size) / SAMPLE_RATE, 2),
            "text": text,
        })
        if word_timestamps:
            words += _words_from(out, offset_ms=int(round(1000 * start / SAMPLE_RATE)))
    env: dict = {
        "type": "transcript", "rev": 1, "final": True, "audio_ms": dur_ms,
        "text": " ".join(t for t in texts if t).strip(),
        "language": (language or None), "task": task,
    }
    if word_timestamps:
        env["words"] = _sanitize_word_timings(words, dur_ms)
    else:
        env["segments"] = segments
    return env


def _sanitize_word_timings(words: list[dict], audio_ms: int) -> list[dict]:
    """Cross-attention DTW word timing is only reliable when the model's
    alignment heads match its attention. Heavily fine-tuned models drift and
    can place words outside the audio (seen with the Quran fine-tune). When the
    DTW timings are clearly broken (exceed the audio, or non-monotonic), fall
    back to a proportional spread across [0, audio_ms] by word length — rough
    but bounded and ordered. ``w`` and ``p`` are untouched (always reliable)."""
    if not words or audio_ms <= 0:
        return words
    t1s = [w.get("t1", 0) for w in words]
    t0s = [w.get("t0", 0) for w in words]
    broken = (max(t1s) > audio_ms * 1.25 or min(t0s) < -200
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
        try:
            env = transcribe(path, req.get("language"),
                             (req.get("task") or "transcribe"),
                             bool(req.get("word_timestamps")))
            self._send(200, env)
        except Exception as e:  # noqa: BLE001
            _log(f"[asr-worker] transcribe error: {e}")
            self._send(500, {"error": str(e)})

    def _do_transcribe_pcm(self):
        """Body: raw float32 mono 16 kHz PCM. Query: language, task,
        word_timestamps. Used by the daemon's streaming loop — no ffmpeg /
        temp file per interval."""
        import numpy as np
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            audio = np.frombuffer(raw, dtype=np.float32)
            if audio.size == 0:
                self._send(400, {"error": "empty pcm"})
                return
            lang = (q.get("language", [""])[0] or None)
            task = (q.get("task", ["transcribe"])[0] or "transcribe")
            wts = (q.get("word_timestamps", ["1"])[0] in ("1", "true", "on", "yes"))
            env = transcribe_audio(audio, lang, task, wts)
            self._send(200, env)
        except Exception as e:  # noqa: BLE001
            _log(f"[asr-worker] transcribe_pcm error: {e}")
            self._send(500, {"error": str(e)})


def main() -> int:
    global _model, _processor, _device, _dtype, _sem, _model_name
    p = argparse.ArgumentParser(description="Persistent ASR worker")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--max-concurrent", type=int, default=2)
    p.add_argument("--device", default="")
    args = p.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    _device, dtype_name = _select_device()
    if args.device:
        _device = args.device
        dtype_name = "float16" if _device != "cpu" else "float32"
    _dtype = getattr(torch, dtype_name)
    _model_name = args.model_path.name
    _sem = threading.Semaphore(max(1, args.max_concurrent))
    _log(f"[asr-worker] loading {args.model_path} on {_device}/{dtype_name} "
         f"(max_concurrent={args.max_concurrent})")
    _processor = WhisperProcessor.from_pretrained(str(args.model_path))
    _model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model_path), torch_dtype=_dtype).to(_device)
    _model.eval()
    _inject_alignment_heads()

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    # daemon threads so a stuck request can't block shutdown
    httpd.daemon_threads = True
    _log(f"[asr-worker] ready on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
