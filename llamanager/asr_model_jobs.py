"""ASR model-acquisition jobs: downloads and cross-engine conversions.

A small DB-backed job manager (table ``asr_model_jobs``) modeled on
``EngineInstaller`` — async task + ``asyncio.Event`` cancel + a streamed log via
an ``emit`` closure — but keyed by the target model, since many can run at once.

Job kinds:
  * ``download``       — pull a catalog / free-form model into ``asr_models_dir``
                         (single-file GGML via ``hf_hub_download`` or a repo
                         snapshot via ``snapshot_download``).
  * ``convert_ggml``   — transformers Whisper → whisper.cpp GGML (+ optional
                         ``whisper-quantize``), so an existing model runs on the
                         Vulkan engine.
  * ``convert_sherpa`` — transformers Whisper → sherpa-onnx Whisper-ONNX.

Produced folders are engine-detected by ``detect_audio_engine_for_path`` — no
registration needed; they light up on the ASR models page under the right engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from . import asr_catalog
from .config import Config
from .db import DB
from .server_manager import _safe_under, _validate_model_id

log = logging.getLogger(__name__)

MAX_LOG_BYTES = 200_000
_VALID_QUANT = ("none", "q5_0", "q8_0")


class AsrJobError(Exception):
    """A user-facing job setup error (bad engine, missing config, etc.)."""


class AsrModelJobs:
    """Owns background download / conversion tasks for ASR models."""

    def __init__(self, cfg: Config, db: DB) -> None:
        self.cfg = cfg
        self.db = db
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel: dict[str, asyncio.Event] = {}

    # ---- public surface ----
    def start_download(self, *, canonical_id: str | None = None,
                       repo: str = "", file: str = "", subfolder: str = "",
                       name: str = "") -> str:
        """Queue a download. Either ``canonical_id`` (a catalog entry) or a
        free-form ``repo`` (+ optional ``file`` / ``subfolder`` / ``name``)."""
        if canonical_id:
            entry = asr_catalog.get(canonical_id)
            if entry is None:
                raise AsrJobError(f"unknown catalog model: {canonical_id!r}")
            spec = {"engine": entry.engine, "model_id": entry.canonical_id,
                    "repo": entry.hf_repo, "file": entry.hf_file,
                    "subfolder": entry.subfolder,
                    "approx_gb": entry.approx_size_gb}
        else:
            repo = (repo or "").strip()
            if not repo:
                raise AsrJobError("a Hugging Face repo is required")
            model_id = (name or file or repo.split("/")[-1]).strip()
            if file:
                # single-file: land it in a folder named after the file stem
                model_id = (name or Path(file).stem).strip()
            spec = {"engine": "", "model_id": model_id, "repo": repo,
                    "file": (file or "").strip(),
                    "subfolder": (subfolder or "").strip(), "approx_gb": 0.0}
        _validate_model_id(spec["model_id"])
        return self._spawn("download", spec)

    def start_convert(self, model_id: str, to_engine: str, *,
                      quantize: str = "none") -> str:
        """Queue a conversion of the installed transformers model ``model_id``
        to ``to_engine`` (``whispercpp`` or ``sherpa``)."""
        _validate_model_id(model_id)
        if to_engine == "sherpa":
            # sherpa's exporter loads openai-whisper checkpoints, not HF
            # transformers fine-tunes; a blind weight remap is unreliable. Use
            # whisper.cpp for reuse, or follow the manual recipe in the README.
            raise AsrJobError(
                "auto-conversion to sherpa-onnx isn't supported — sherpa's "
                "Whisper exporter needs an openai-whisper checkpoint, not a HF "
                "fine-tune. Use whisper.cpp for reuse, or the manual export "
                "recipe in the README (sherpa runs the resulting ONNX fine).")
        if to_engine != "whispercpp":
            raise AsrJobError(f"cannot convert to engine {to_engine!r}")
        if quantize not in _VALID_QUANT:
            raise AsrJobError(f"invalid quantize {quantize!r}")
        base = self.cfg.asr_models_dir
        src = _safe_under(base, base / model_id)
        if not src.exists():
            raise AsrJobError(f"model not found: {model_id}")
        out_id = f"{model_id}-ggml"
        return self._spawn("convert_ggml", {
            "engine": "whispercpp", "model_id": out_id,
            "src_id": model_id, "quantize": quantize})

    def cancel(self, job_id: str) -> bool:
        ev = self._cancel.get(job_id)
        if not ev:
            return False
        ev.set()
        return True

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM asr_model_jobs WHERE id=?", (job_id,))
        return _row(row) if row else None

    def list_jobs(self, *, limit: int = 25) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM asr_model_jobs ORDER BY started_at DESC LIMIT ?",
            (limit,))
        return [_row(r) for r in rows]

    def active_jobs(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM asr_model_jobs WHERE status IN ('pending','running') "
            "ORDER BY started_at DESC")
        return [_row(r) for r in rows]

    def job_for_model(self, model_id: str) -> dict[str, Any] | None:
        """Most recent job that targets ``model_id`` (for per-row status)."""
        row = self.db.query_one(
            "SELECT * FROM asr_model_jobs WHERE model_id=? "
            "ORDER BY started_at DESC LIMIT 1", (model_id,))
        return _row(row) if row else None

    async def shutdown(self) -> None:
        for ev in self._cancel.values():
            ev.set()
        for t in list(self._tasks.values()):
            t.cancel()
            with __import__("contextlib").suppress(Exception):
                await t

    # ---- internals ----
    def _spawn(self, kind: str, spec: dict[str, Any]) -> str:
        job_id = secrets.token_urlsafe(8)
        self.db.execute(
            "INSERT INTO asr_model_jobs(id,kind,engine,model_id,status,message,"
            "started_at,options_json) VALUES (?,?,?,?,?,?,?,?)",
            (job_id, kind, spec.get("engine", ""), spec["model_id"], "pending",
             "queued", time.time(), json.dumps(spec, sort_keys=True)))
        cancel = asyncio.Event()
        self._cancel[job_id] = cancel
        runner = {"download": self._run_download,
                  "convert_ggml": self._run_convert_ggml}[kind]
        self._tasks[job_id] = asyncio.create_task(runner(job_id, spec, cancel))
        self.db.log_event("asr_model_job_started",
                          {"id": job_id, "kind": kind,
                           "model_id": spec["model_id"]})
        return job_id

    def _set(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE asr_model_jobs SET {cols} WHERE id=?",
                        (*fields.values(), job_id))

    def _emitter(self, job_id: str):
        buf: list[str] = []

        def emit(line: str) -> None:
            buf.append(line)
            joined = "\n".join(buf)
            if len(joined) > MAX_LOG_BYTES:
                joined = joined[-MAX_LOG_BYTES:]
                buf[:] = joined.splitlines()
            self._set(job_id, log=joined)

        def progress(pct: int, message: str) -> None:
            self._set(job_id, progress_pct=int(pct), message=message)
            emit(f"[{pct:3d}%] {message}")

        return emit, progress

    def _finish_ok(self, job_id: str, model_id: str, kind: str) -> None:
        self._set(job_id, status="done", progress_pct=100,
                  finished_at=time.time())
        self.db.log_event("asr_model_job_done",
                          {"id": job_id, "kind": kind, "model_id": model_id})

    def _finish_err(self, job_id: str, emit, e: Exception) -> None:
        emit(f"[error] {e}")
        self._set(job_id, status="failed", finished_at=time.time(), error=str(e))
        self.db.log_event("asr_model_job_failed", {"id": job_id, "error": str(e)})

    def _cleanup(self, job_id: str) -> None:
        self._cancel.pop(job_id, None)
        self._tasks.pop(job_id, None)

    def _target_dir(self, model_id: str) -> Path:
        base = self.cfg.asr_models_dir
        base.mkdir(parents=True, exist_ok=True)
        target = _safe_under(base, base / model_id)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _hf_token(self) -> str | None:
        env = getattr(self.cfg, "hf_token_env", "HF_TOKEN") or "HF_TOKEN"
        return os.environ.get(env) or None

    # ---- download ----
    async def _run_download(self, job_id: str, spec: dict[str, Any],
                            cancel: asyncio.Event) -> None:
        emit, progress = self._emitter(job_id)
        try:
            self._set(job_id, status="running", message="Preparing download")
            from huggingface_hub import hf_hub_download, snapshot_download
            target = self._target_dir(spec["model_id"])
            token = self._hf_token()
            repo, file, subfolder = spec["repo"], spec["file"], spec["subfolder"]
            approx = float(spec.get("approx_gb") or 0.0)
            progress(2, f"Downloading {repo}" + (f" · {file}" if file else ""))

            def do_download() -> None:
                if file:
                    hf_hub_download(repo_id=repo, filename=file,
                                    local_dir=str(target), token=token)
                else:
                    allow = None
                    if subfolder:
                        allow = [f"{subfolder}/*", f"{subfolder}/**/*"]
                    snapshot_download(repo_id=repo, local_dir=str(target),
                                      allow_patterns=allow, token=token)

            dl = asyncio.create_task(asyncio.to_thread(do_download))
            # Best-effort moving progress from on-disk size vs the catalog's
            # approx (HF's own progress isn't surfaced through the API here).
            while not dl.done():
                if cancel.is_set():
                    emit("[cancelled] download stopped (files already fetched "
                         "are kept and resumable)")
                    self._set(job_id, status="cancelled", finished_at=time.time())
                    self._cleanup(job_id)
                    return
                if approx > 0:
                    gb = _dir_size_gb(target)
                    progress(min(99, int(gb / approx * 100)),
                             f"Downloading… {gb:.2f} / ~{approx:.2f} GB")
                await asyncio.wait({dl}, timeout=1.5)
            await dl  # surface any exception
            progress(100, f"Downloaded to {target}")
            self._finish_ok(job_id, spec["model_id"], "download")
        except asyncio.CancelledError:
            self._set(job_id, status="cancelled", finished_at=time.time())
        except Exception as e:  # noqa: BLE001
            log.exception("asr download %s failed", job_id)
            self._finish_err(job_id, emit, e)
        finally:
            self._cleanup(job_id)

    # ---- convert: transformers → whisper.cpp GGML ----
    async def _run_convert_ggml(self, job_id: str, spec: dict[str, Any],
                                cancel: asyncio.Event) -> None:
        emit, progress = self._emitter(job_id)
        try:
            self._set(job_id, status="running", message="Preparing conversion")
            py, repo = self._require_ggml_toolchain()
            script = repo / "models" / "convert-h5-to-ggml.py"
            if not script.is_file():
                raise AsrJobError(f"convert script missing: {script}")
            base = self.cfg.asr_models_dir
            src = _safe_under(base, base / spec["src_id"])
            if not src.exists():
                raise AsrJobError(f"source model not found: {spec['src_id']}")
            out = self._target_dir(spec["model_id"])

            progress(10, "Ensuring mel-filter assets (openai-whisper)")
            mel_dir = await self._ensure_whisper_assets(py, cancel, emit)

            progress(30, "Converting Hugging Face checkpoint → GGML")
            rc = await _run_subprocess(
                [str(py), str(script), str(src), str(mel_dir), str(out)],
                cancel, emit)
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc != 0:
                raise AsrJobError(f"convert-h5-to-ggml.py failed (exit {rc})")
            produced = out / "ggml-model.bin"
            if not produced.is_file():
                raise AsrJobError("conversion produced no ggml-model.bin")
            final = out / f"ggml-{spec['model_id']}.bin"
            produced.replace(final)

            quant = spec.get("quantize", "none")
            if quant != "none":
                progress(80, f"Quantizing → {quant}")
                qbin = repo / "build" / "bin" / "whisper-quantize"
                if not qbin.exists():
                    emit(f"[warn] {qbin} missing — keeping f16 model unquantized")
                else:
                    qout = out / f"ggml-{spec['model_id']}-{quant}.bin"
                    rc = await _run_subprocess(
                        [str(qbin), str(final), str(qout), quant], cancel, emit)
                    if rc == 0 and qout.is_file():
                        final.unlink(missing_ok=True)  # keep only the quantized
                    else:
                        emit(f"[warn] quantize failed (exit {rc}); keeping f16")
            progress(100, f"Built {out}")
            self._finish_ok(job_id, spec["model_id"], "convert_ggml")
        except asyncio.CancelledError:
            self._set(job_id, status="cancelled", finished_at=time.time())
        except Exception as e:  # noqa: BLE001
            log.exception("asr convert_ggml %s failed", job_id)
            self._finish_err(job_id, emit, e)
        finally:
            self._cleanup(job_id)

    # ---- convert helpers ----
    def _require_ggml_toolchain(self) -> tuple[Path, Path]:
        """Return (asr_python, whispercpp_repo). The converter loads the HF
        checkpoint with torch+transformers (``asr_python``); the repo ships the
        convert script + quantize binary (derived from ``whispercpp_binary``)."""
        if not self.cfg.asr_python:
            raise AsrJobError(
                "the transformers ASR engine (image.asr_python) must be "
                "installed to convert — it provides torch + transformers.")
        py = Path(self.cfg.asr_python).expanduser()
        if not py.exists():
            raise AsrJobError(f"asr python not found: {py}")
        if not self.cfg.whispercpp_binary:
            raise AsrJobError(
                "build whisper.cpp first (its repo ships the GGML converter).")
        binp = Path(self.cfg.whispercpp_binary).expanduser()
        # .../whispercpp/build/bin/whisper-cli → repo = .../whispercpp
        repo = binp.parent.parent.parent
        if not (repo / "models" / "convert-h5-to-ggml.py").is_file():
            raise AsrJobError(
                f"whisper.cpp repo not found next to {binp} (looked in {repo}).")
        return py, repo

    async def _ensure_whisper_assets(self, py: Path, cancel: asyncio.Event,
                                     emit) -> Path:
        """Return a dir containing ``whisper/assets/mel_filters.npz`` (needed by
        the converter). Uses openai-whisper's package assets; installs it into
        the asr venv on first use."""
        probe = ("import os,whisper;"
                 "print(os.path.dirname(os.path.dirname(whisper.__file__)))")
        rc, outs = await _run_subprocess_capture([str(py), "-c", probe], cancel)
        site = outs.strip().splitlines()[-1] if rc == 0 and outs.strip() else ""
        if not site or not (Path(site) / "whisper" / "assets" / "mel_filters.npz").is_file():
            emit("[assets] installing openai-whisper for mel filters")
            rc = await _run_subprocess(
                [str(py), "-m", "pip", "install", "--upgrade",
                 "--progress-bar", "off", "openai-whisper"], cancel, emit)
            if rc != 0:
                raise AsrJobError(f"failed to install openai-whisper (exit {rc})")
            rc, outs = await _run_subprocess_capture([str(py), "-c", probe], cancel)
            site = outs.strip().splitlines()[-1] if rc == 0 and outs.strip() else ""
        mel = Path(site) / "whisper" / "assets" / "mel_filters.npz"
        if not mel.is_file():
            raise AsrJobError("could not locate whisper/assets/mel_filters.npz")
        return Path(site)


# ---- module helpers ----

def _row(row) -> dict[str, Any]:
    return {
        "id": row["id"], "kind": row["kind"], "engine": row["engine"],
        "model_id": row["model_id"], "status": row["status"],
        "progress_pct": row["progress_pct"], "message": row["message"] or "",
        "log": row["log"] or "", "started_at": row["started_at"],
        "finished_at": row["finished_at"], "error": row["error"],
    }


def _dir_size_gb(d: Path) -> float:
    total = 0
    try:
        for p in d.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
    except OSError:
        pass
    return total / (1024 ** 3)


async def _run_subprocess(argv: list[str], cancel: asyncio.Event, emit) -> int:
    """Run a subprocess, streaming stdout/stderr lines into the job log. Honours
    ``cancel`` by terminating the child. Returns the exit code."""
    emit(f"$ {' '.join(argv)}")
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    async def reader() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            emit(line.decode("utf-8", "replace").rstrip())

    rt = asyncio.create_task(reader())
    try:
        while True:
            if cancel.is_set():
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                break
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break
            except asyncio.TimeoutError:
                continue
    finally:
        await rt
    return proc.returncode if proc.returncode is not None else -1


async def _run_subprocess_capture(argv: list[str],
                                  cancel: asyncio.Event) -> tuple[int, str]:
    """Run a short subprocess and capture stdout (for probes)."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    return (proc.returncode if proc.returncode is not None else -1,
            out.decode("utf-8", "replace"))
