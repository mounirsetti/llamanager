"""Whisper transcription runner — invoked by ``asr.py`` as a subprocess.

This script ships with llamanager and runs inside the user-configured Python
environment (the one with ``torch`` + ``transformers`` installed — typically
the shared z_image venv). It loads a Hugging Face
``WhisperForConditionalGeneration`` checkpoint, transcribes one audio file,
and writes a JSON result to disk.

Audio decoding goes through the system ``ffmpeg`` (decoded to 16 kHz mono
float32), so no audio Python packages are required in the venv. Long inputs
are processed in 30-second windows; we emit ``chunk i/N`` to stderr so the
parent adapter's ``parse_progress`` can drive the UI progress bar.

Language / task selection uses ``forced_decoder_ids`` from the processor
rather than ``generate(language=...)``: some fine-tunes (e.g. the Quran
Whisper) ship a ``generation_config`` without the lang/task token maps, which
makes the ``language=`` kwarg raise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Whisper operates on 30-second windows at 16 kHz mono.
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _decode_audio(path: Path):
    """Decode ``path`` to a mono float32 numpy array at 16 kHz via ffmpeg."""
    import numpy as np

    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH — required to decode audio") from e
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed to decode {path.name}:\n{tail}") from e
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"decoded zero audio samples from {path.name}")
    return audio


def _select_device() -> tuple[str, str]:
    """Return (device, dtype_name). fp16 on GPU, fp32 on CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda", "float16"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


def main() -> int:
    p = argparse.ArgumentParser(description="Whisper transcription runner")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default="", help="ISO code, or empty to auto-detect")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--device", default="", help="override device (cuda/cpu/mps)")
    args = p.parse_args()

    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device, dtype_name = _select_device()
    if args.device:
        device = args.device
        dtype_name = "float16" if device != "cpu" else "float32"
    torch_dtype = getattr(torch, dtype_name)
    _log(f"[asr] device={device} dtype={dtype_name} model={args.model_path}")

    processor = WhisperProcessor.from_pretrained(str(args.model_path))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model_path), torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    # Force language + task by writing decoder-prompt ids onto the generation
    # config. transformers 5.x dropped ``forced_decoder_ids`` as a generate()
    # kwarg, and the ``language=`` kwarg path needs a ``lang_to_id`` map this
    # fine-tune's generation_config lacks — so the generation_config attribute
    # is the one route that works here (verified on the Quran Whisper).
    lang = (args.language or "").strip().lower()
    if lang and lang != "auto":
        try:
            model.generation_config.forced_decoder_ids = (
                processor.get_decoder_prompt_ids(language=lang, task=args.task)
            )
        except (KeyError, ValueError) as e:
            _log(f"[asr] language hint {lang!r} rejected, auto-detecting: {e}")
            model.generation_config.forced_decoder_ids = None

    audio = _decode_audio(args.audio)
    duration_s = float(audio.size) / SAMPLE_RATE
    n_chunks = max(1, (audio.size + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES)
    _log(f"[asr] decoded {duration_s:.1f}s of audio → {n_chunks} chunk(s)")

    texts: list[str] = []
    segments: list[dict] = []
    for i in range(n_chunks):
        start = i * CHUNK_SAMPLES
        window = audio[start:start + CHUNK_SAMPLES]
        feats = processor(
            window, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        ).input_features.to(device, torch_dtype)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=440)
        chunk_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        texts.append(chunk_text)
        segments.append({
            "start": round(start / SAMPLE_RATE, 2),
            "end": round(min(start + CHUNK_SAMPLES, audio.size) / SAMPLE_RATE, 2),
            "text": chunk_text,
        })
        # Progress line consumed by asr.parse_progress.
        _log(f"[asr] chunk {i + 1}/{n_chunks}")

    result = {
        "text": " ".join(t for t in texts if t).strip(),
        "language": lang or None,
        "task": args.task,
        "duration_s": round(duration_s, 2),
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    _log(f"[asr] wrote {args.output} ({len(result['text'])} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
