"""Opinionated dependency installer for diffusion engines.

For each image engine, we know what Python packages or external binaries
are needed. This module creates a per-engine virtual environment under
``{data_dir}/venvs/{engine}/`` and pip-installs the right packages. Once
installed, the engine's "Python executable" field is auto-populated with
the venv's interpreter path.

Note on opinionation: each engine's dependency list is the minimum that
gets a known-good baseline running on a reasonably mainstream box (CUDA
or CPU). For ROCm or specialised hardware setups, the operator should
build the venv themselves and point llamanager at it — the auto-install
path will install a generic torch wheel that may not match their GPU.
The setup card surfaces this caveat in plain language.

The installer streams stdout/stderr lines from ``python -m venv`` and
``pip install`` into the ``engine_installs.log`` column so the UI can
show live output. Progress is reported as percentage milestones rather
than parsing pip's bar (which is unreliable across pip versions).
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .db import DB

log = logging.getLogger(__name__)

MAX_LOG_BYTES = 200_000  # cap stored log size so the DB doesn't bloat


@dataclass(frozen=True)
class EnginePackages:
    """Pip install plan for one diffusion engine."""
    engine: str
    label: str
    packages: list[str]   # what pip-install actually receives
    extra_index_url: str | None = None
    space_mb: int = 0     # rough disk footprint estimate of the installed venv
    notes: str = ""       # surfaced in the UI under the install button


# Per-engine install plans. Kept declarative so the setup page can render
# the same info as a checklist before the user clicks Install.
ENGINE_PLANS: dict[str, EnginePackages] = {
    "z_image": EnginePackages(
        engine="z_image",
        label="Z-Image (Tongyi-MAI / Z-Anime)",
        # Z-Image needs bleeding-edge diffusers — install from main.
        packages=[
            "torch", "transformers", "accelerate", "huggingface_hub",
            "safetensors", "Pillow", "sentencepiece",
            "git+https://github.com/huggingface/diffusers",
        ],
        space_mb=8500,
        notes=(
            "Installs the latest diffusers (from git) plus torch and the "
            "Z-Image tokenizer/text-encoder deps. On NVIDIA, the default "
            "torch wheel is CUDA 12.x; on AMD/ROCm or Apple Silicon you "
            "should build the venv yourself and skip this button."
        ),
    ),
    "hidream": EnginePackages(
        engine="hidream",
        label="HiDream-O1-Image",
        # Minimum to run upstream inference.py — these are the packages
        # the HiDream README pins. Operators on ROCm should use the
        # vendor wheels directly instead of this auto-install.
        packages=[
            "torch", "transformers", "accelerate", "diffusers",
            "huggingface_hub", "safetensors", "Pillow",
            "einops", "sentencepiece",
        ],
        space_mb=7500,
        notes=(
            "Installs a generic torch wheel. HiDream is verified on "
            "ROCm 7.2+/gfx1201 with the official AMD wheels — if you "
            "want that, follow docs/hidream.md and skip this button."
        ),
    ),
}


def venv_root(cfg: Config) -> Path:
    """Where per-engine venvs live."""
    return cfg.data_dir / "venvs"


def venv_python(cfg: Config, engine: str) -> Path:
    """Predict the python interpreter path for a given engine's venv.
    Returned path may not exist until an install completes."""
    root = venv_root(cfg) / engine
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


class EngineInstaller:
    """Owns the background installer tasks for diffusion engines."""

    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel_flags: dict[str, asyncio.Event] = {}

    # ---- public surface ----
    def start(self, engine: str) -> str:
        plan = ENGINE_PLANS.get(engine)
        if plan is None:
            raise ValueError(f"no install plan for engine {engine!r}")
        # Reject if one is already running for this engine.
        active = self.active_for_engine(engine)
        if active:
            raise RuntimeError(
                f"an install for {engine} is already running "
                f"(id={active['id']})"
            )
        install_id = secrets.token_urlsafe(8)
        self.db.execute(
            "INSERT INTO engine_installs(id, engine, kind, status, "
            "message, started_at) VALUES (?, ?, 'pip', 'pending', ?, ?)",
            (install_id, engine, "queued", time.time()),
        )
        cancel = asyncio.Event()
        self._cancel_flags[install_id] = cancel
        task = asyncio.create_task(self._run(install_id, plan, cancel))
        self._tasks[install_id] = task
        self.db.log_event("install_started",
                          {"id": install_id, "engine": engine})
        return install_id

    def cancel(self, install_id: str) -> bool:
        ev = self._cancel_flags.get(install_id)
        if not ev:
            return False
        ev.set()
        return True

    def get(self, install_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM engine_installs WHERE id=?", (install_id,),
        )
        if not row:
            return None
        return _row_to_dict(row)

    def list_for_engine(self, engine: str, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM engine_installs WHERE engine=? "
            "ORDER BY started_at DESC LIMIT ?", (engine, limit),
        )
        return [_row_to_dict(r) for r in rows]

    def active_for_engine(self, engine: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM engine_installs WHERE engine=? "
            "AND status IN ('pending', 'running') "
            "ORDER BY started_at DESC LIMIT 1",
            (engine,),
        )
        return _row_to_dict(row) if row else None

    # ---- internals ----
    async def _run(self, install_id: str, plan: EnginePackages,
                   cancel: asyncio.Event) -> None:
        log_buf: list[str] = []

        def emit(line: str) -> None:
            log_buf.append(line)
            joined = "\n".join(log_buf)
            if len(joined) > MAX_LOG_BYTES:
                # Keep the tail when log overflows the budget.
                joined = joined[-MAX_LOG_BYTES:]
                log_buf[:] = joined.splitlines()
            self._set(install_id, log=joined)

        def set_progress(pct: int, message: str) -> None:
            self._set(install_id, progress_pct=int(pct), message=message)
            emit(f"[{pct:3d}%] {message}")

        try:
            self._set(install_id, status="running",
                      message="Creating virtual environment")
            set_progress(5, f"Creating venv at {venv_root(self.cfg) / plan.engine}")

            venv_dir = venv_root(self.cfg) / plan.engine
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
            python_path = venv_python(self.cfg, plan.engine)

            if not python_path.exists():
                rc = await self._run_subprocess(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    cancel, emit,
                )
                if cancel.is_set():
                    raise asyncio.CancelledError()
                if rc != 0:
                    raise RuntimeError(f"venv create failed with exit {rc}")
            else:
                emit(f"[ok] venv already exists at {venv_dir}")

            set_progress(15, "Upgrading pip in the new environment")
            rc = await self._run_subprocess(
                [str(python_path), "-m", "pip", "install",
                 "--upgrade", "pip", "wheel", "setuptools"],
                cancel, emit,
            )
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc != 0:
                raise RuntimeError(f"pip self-upgrade failed (exit {rc})")

            set_progress(25, f"Installing {len(plan.packages)} packages")
            argv = [
                str(python_path), "-m", "pip", "install", "--upgrade",
                "--progress-bar", "off",
            ]
            if plan.extra_index_url:
                argv += ["--extra-index-url", plan.extra_index_url]
            argv += plan.packages

            rc = await self._run_subprocess(argv, cancel, emit)
            if cancel.is_set():
                raise asyncio.CancelledError()
            if rc != 0:
                raise RuntimeError(f"pip install failed (exit {rc})")

            set_progress(95, "Verifying torch imports")
            rc = await self._run_subprocess(
                [str(python_path), "-c", "import torch; print(torch.__version__)"],
                cancel, emit,
            )
            if rc != 0:
                emit("[warn] torch import check failed — installed packages "
                     "may not be runnable on this machine; pointing the "
                     "engine at the new venv anyway.")

            # Persist the new venv path into the engine's config field.
            self._persist_engine_python(plan.engine, str(python_path))

            set_progress(100, f"Installed at {python_path}")
            self._set(install_id, status="done", finished_at=time.time())
            self.db.log_event("install_done", {
                "id": install_id, "engine": plan.engine,
                "python": str(python_path),
            })
        except asyncio.CancelledError:
            emit("[cancelled] install stopped")
            self._set(install_id, status="cancelled",
                      finished_at=time.time())
        except Exception as e:
            log.exception("install %s failed", install_id)
            emit(f"[error] {e}")
            self._set(install_id, status="failed",
                      finished_at=time.time(), error=str(e))
            self.db.log_event("install_failed",
                              {"id": install_id, "error": str(e)})
        finally:
            self._cancel_flags.pop(install_id, None)
            self._tasks.pop(install_id, None)

    async def _run_subprocess(self, argv: list[str], cancel: asyncio.Event,
                              emit) -> int:
        """Run a subprocess, streaming each stdout/stderr line into the log.
        Honours ``cancel`` by terminating the child. Returns the exit code."""
        emit(f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def reader() -> None:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                emit(line.decode("utf-8", errors="replace").rstrip())

        reader_task = asyncio.create_task(reader())
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
            await reader_task
        return proc.returncode if proc.returncode is not None else -1

    def _set(self, install_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(
            f"UPDATE engine_installs SET {cols} WHERE id=?",
            (*fields.values(), install_id),
        )

    def _persist_engine_python(self, engine: str, python_path: str) -> None:
        """Save the venv's python path back to config.toml so the engine
        is ready to use without the operator having to copy/paste the path."""
        from .config import update_image_config
        kwargs: dict[str, Any] = {}
        if engine == "z_image":
            kwargs["z_image_python"] = python_path
            self.cfg.z_image_python = python_path
        elif engine == "hidream":
            kwargs["hidream_python"] = python_path
            self.cfg.hidream_python = python_path
        else:
            return
        try:
            update_image_config(self.cfg.config_path, **kwargs)
        except Exception:
            log.exception("failed to persist %s python path to config", engine)


def _row_to_dict(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "engine": row["engine"],
        "kind": row["kind"],
        "status": row["status"],
        "progress_pct": row["progress_pct"],
        "message": row["message"] or "",
        "log": row["log"] or "",
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
    }
