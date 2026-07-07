"""Curated catalog of speech-to-text models, keyed by ASR engine.

Mirrors ``diffusion_catalog``: each entry says "this is a model we know how to
run", the on-disk ``canonical_id`` folder it produces under ``asr_models_dir``,
which engine detects it, and where to pull it from on Hugging Face. The ASR
models page joins this catalog against what's actually on disk to show
"Installed" vs "Download".

Two download shapes:
  * ``hf_file`` set  → a single-file download (whisper.cpp GGML ``ggml-*.bin``)
    into ``asr_models_dir/<canonical_id>/<hf_file>``.
  * ``hf_file`` empty → a snapshot of ``hf_repo`` (optionally ``subfolder``)
    into ``asr_models_dir/<canonical_id>/`` (transformers checkpoints, sherpa
    model folders).

Keeping the catalog in a Python module (vs config.toml) means new model support
ships with a code release — one new entry here, no operator action.
"""
from __future__ import annotations

from dataclasses import dataclass

# Official pre-converted GGML Whisper models (see whisper.cpp's
# models/download-ggml-model.sh). Files are named ``ggml-<model>.bin``.
_GGML_REPO = "ggerganov/whisper.cpp"


@dataclass(frozen=True)
class AsrCatalogEntry:
    """One known speech-to-text model.

    ``canonical_id`` is the directory name created under ``asr_models_dir`` by
    the download; the engine detector (``detect_audio_engine_for_path``) then
    binds the folder to ``engine``. ``language`` is a human display string used
    for the sherpa language autocomplete and shown on every row.
    """
    canonical_id: str
    engine: str            # 'asr' | 'whispercpp' | 'sherpa'
    label: str
    language: str          # display, e.g. "Multilingual", "Arabic", "English"
    hf_repo: str
    hf_file: str = ""      # single-file download (GGML); empty → snapshot repo
    subfolder: str = ""    # optional HF subfolder for a snapshot
    approx_size_gb: float = 0.0
    description: str = ""
    # Individual languages a model covers, for the sherpa language autocomplete
    # + row filter. Empty → not surfaced in the language picker (whisper models
    # are multilingual by architecture, not filtered by language).
    langs: tuple[str, ...] = ()


def _ggml(model: str, label: str, size_gb: float, desc: str) -> AsrCatalogEntry:
    return AsrCatalogEntry(
        canonical_id=f"ggml-{model}", engine="whispercpp",
        label=label, language=("English" if ".en" in model else "Multilingual"),
        hf_repo=_GGML_REPO, hf_file=f"ggml-{model}.bin",
        approx_size_gb=size_gb, description=desc)


CATALOG: list[AsrCatalogEntry] = [
    # ---- whisper.cpp (GGML / Vulkan) — official pre-converted models ----
    _ggml("large-v3-turbo", "Whisper large-v3-turbo", 1.6,
          "Fastest large model; best default for GPU transcription."),
    _ggml("large-v3-turbo-q5_0", "Whisper large-v3-turbo (q5_0)", 0.55,
          "Quantized large-v3-turbo — ~3x smaller, near-lossless, great on VRAM."),
    _ggml("large-v3-turbo-q8_0", "Whisper large-v3-turbo (q8_0)", 0.87,
          "Quantized large-v3-turbo — half size, effectively lossless."),
    _ggml("large-v3", "Whisper large-v3", 3.1,
          "Full large-v3 accuracy (slower than turbo)."),
    _ggml("large-v3-q5_0", "Whisper large-v3 (q5_0)", 1.1,
          "Quantized large-v3 — much smaller, minor quality cost."),
    _ggml("medium", "Whisper medium", 1.5,
          "Good multilingual accuracy at a moderate footprint."),
    _ggml("small", "Whisper small", 0.49,
          "Fast multilingual model for lower-resource machines."),
    _ggml("base", "Whisper base", 0.15,
          "Small and quick; fine for clean audio / drafts."),
    _ggml("tiny", "Whisper tiny", 0.078,
          "Smallest model; fastest, lowest accuracy."),

    # ---- asr (Hugging Face transformers) — the models already in use ----
    AsrCatalogEntry(
        canonical_id="whisper-large-v3-turbo", engine="asr",
        label="Whisper large-v3-turbo (transformers)", language="Multilingual",
        hf_repo="openai/whisper-large-v3-turbo", approx_size_gb=1.6,
        description="OpenAI large-v3-turbo in HF transformers format."),
    AsrCatalogEntry(
        canonical_id="whisper-large-v3-turbo-ar-quran", engine="asr",
        label="Whisper large-v3-turbo — Arabic Quran", language="Arabic",
        hf_repo="naazimsnh02/whisper-large-v3-turbo-ar-quran",
        approx_size_gb=1.6,
        description="Arabic Quran-recitation fine-tune of large-v3-turbo."),

    # ---- sherpa-onnx — streaming (online) transducers, per language ----
    # Verified k2-fsa/csukuangfj HF repos; each is a full model folder
    # (tokens.txt + encoder/decoder/joiner ONNX) → snapshot download.
    AsrCatalogEntry(
        canonical_id="sherpa-quran-ar-whisper-tiny", engine="sherpa",
        label="Whisper-tiny — Quran (Arabic)", language="Arabic (Quran)",
        langs=("Arabic",),
        hf_repo="kasimyazan/sherpa-tarteel-whisper-tiny-ar-quran",
        approx_size_gb=0.05,
        description="Quran-recitation Whisper-tiny fine-tune (tarteel) exported "
                    "for sherpa-onnx — offline decode. Small and fast; for "
                    "true streaming Arabic use the multilingual model below."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-multi-ar", engine="sherpa",
        label="Streaming Zipformer — Arabic + 7 languages",
        language="Multilingual (ar, en, id, ja, ru, th, vi, zh)",
        langs=("Arabic", "English", "Indonesian", "Japanese", "Russian",
               "Thai", "Vietnamese", "Chinese"),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10",
        approx_size_gb=0.35,
        description="Low-latency streaming model covering Arabic and 7 more "
                    "languages — the best sherpa fit for Arabic live mic."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-en", engine="sherpa",
        label="Streaming Zipformer — English", language="English",
        langs=("English",),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        approx_size_gb=0.34,
        description="Well-tested English streaming transducer."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-zh-en", engine="sherpa",
        label="Streaming Zipformer — Chinese + English", language="Chinese, English",
        langs=("Chinese", "English"),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        approx_size_gb=0.35,
        description="Bilingual Mandarin/English streaming transducer."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-zh", engine="sherpa",
        label="Streaming Zipformer — Chinese", language="Chinese",
        langs=("Chinese",),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-13",
        approx_size_gb=0.35,
        description="Mandarin (multi-domain) streaming transducer."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-fr", engine="sherpa",
        label="Streaming Zipformer — French", language="French",
        langs=("French",),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06",
        approx_size_gb=0.2,
        description="French streaming transducer (Kroko)."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-de", engine="sherpa",
        label="Streaming Zipformer — German", language="German",
        langs=("German",),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06",
        approx_size_gb=0.2,
        description="German streaming transducer (Kroko)."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-es", engine="sherpa",
        label="Streaming Zipformer — Spanish", language="Spanish",
        langs=("Spanish",),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06",
        approx_size_gb=0.2,
        description="Spanish streaming transducer (Kroko)."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-ko", engine="sherpa",
        label="Streaming Zipformer — Korean", language="Korean",
        langs=("Korean",),
        hf_repo="k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16",
        approx_size_gb=0.35,
        description="Korean streaming transducer."),
    AsrCatalogEntry(
        canonical_id="sherpa-streaming-ru", engine="sherpa",
        label="Streaming Zipformer — Russian", language="Russian",
        langs=("Russian",),
        hf_repo="csukuangfj/sherpa-onnx-streaming-zipformer-small-ru-vosk-2025-08-16",
        approx_size_gb=0.2,
        description="Russian streaming transducer (Vosk)."),
]


def catalog_for(engine: str) -> list[AsrCatalogEntry]:
    """Entries for one engine, in catalog order."""
    return [e for e in CATALOG if e.engine == engine]


def get(canonical_id: str) -> AsrCatalogEntry | None:
    """Look up an entry by its on-disk ``canonical_id``."""
    for e in CATALOG:
        if e.canonical_id == canonical_id:
            return e
    return None


def languages() -> list[str]:
    """Sorted unique individual languages across the catalog — the source for
    the sherpa language autocomplete."""
    seen: set[str] = set()
    for e in CATALOG:
        seen.update(e.langs)
    return sorted(seen)
