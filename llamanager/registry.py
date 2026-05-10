"""Model registry: list / pull / delete GGUFs in ~/.llamanager/models/.

Spec §4.4. Pulls run as background asyncio tasks; the API returns a
download_id immediately and clients poll /admin/downloads/{id}.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from .config import Config
from .db import DB

log = logging.getLogger(__name__)

SAFETY_MARGIN_BYTES = 2 * 1024 ** 3  # 2 GB

# Acceptable model filename: letters, digits, dot, underscore, dash, plus.
# Specifically forbids slashes and backslashes so an attacker can't escape
# the target directory via a hostile HTTP filename or HF entry.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._+\-]+$")
_ALLOWED_DOWNLOAD_EXTS = (".gguf", ".safetensors", ".bin")
# HF repo ids look like "org/name". Allow only conservative chars, two segs.
_SAFE_HF_REPO_RE = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")
# HF files may live in subdirs; allow forward-slash separated segments.
_SAFE_HF_FILE_SEG_RE = re.compile(r"^[A-Za-z0-9._+\-]+$")


def _sanitize_download_filename(name: str) -> str:
    """Return a safe basename or raise ValueError.

    Rejects anything that could escape the target directory: paths with
    separators, dotfiles, parent refs, NUL bytes, and unexpected extensions.
    """
    if not name or "\x00" in name:
        raise ValueError("download filename is empty or contains NUL")
    base = os.path.basename(name.replace("\\", "/"))
    if base != name or not _SAFE_FILENAME_RE.match(base):
        raise ValueError(f"unsafe download filename: {name!r}")
    if base.startswith(".") or base in ("..", "."):
        raise ValueError(f"unsafe download filename: {name!r}")
    if not base.lower().endswith(_ALLOWED_DOWNLOAD_EXTS):
        raise ValueError(
            f"download filename must end with one of "
            f"{_ALLOWED_DOWNLOAD_EXTS}: {name!r}"
        )
    return base


def _sanitize_hf_repo(repo: str) -> str:
    if not repo or "\x00" in repo:
        raise ValueError("HF repo id is empty or invalid")
    if not _SAFE_HF_REPO_RE.match(repo):
        raise ValueError(f"unsafe HF repo id: {repo!r}")
    if ".." in repo.split("/"):
        raise ValueError(f"unsafe HF repo id: {repo!r}")
    return repo


def _sanitize_hf_file(path: str) -> str:
    """HF file paths may be nested ('subdir/foo.gguf'). Validate each segment."""
    if not path or "\x00" in path or "\\" in path:
        raise ValueError(f"unsafe HF file path: {path!r}")
    posix = PurePosixPath(path)
    if posix.is_absolute():
        raise ValueError(f"HF file path must be relative: {path!r}")
    parts = posix.parts
    if not parts or any(seg in ("", "..", ".") for seg in parts):
        raise ValueError(f"HF file path contains forbidden segment: {path!r}")
    for seg in parts[:-1]:
        if not _SAFE_HF_FILE_SEG_RE.match(seg):
            raise ValueError(f"unsafe HF file path segment: {seg!r}")
    # Final segment must look like a model file.
    _sanitize_download_filename(parts[-1])
    return path


def _ensure_under(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"download target escapes models_dir: {candidate} -> {resolved}"
        )
    return resolved


@dataclass
class ModelEntry:
    model_id: str        # path relative to models_dir, e.g. "repo/file.gguf"
    path: Path
    size: int
    sha256: str | None
    source: str | None
    pulled_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
            "source": self.source,
            "pulled_at": self.pulled_at,
        }


def _meta_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".meta.json")


def _read_meta(p: Path) -> dict[str, Any]:
    mp = _meta_path(p)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(p: Path, data: dict[str, Any]) -> None:
    mp = _meta_path(p)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


class Registry:
    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self.models_dir = cfg.models_dir
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel_flags: dict[str, asyncio.Event] = {}

    # ---- list ----
    def list(self) -> list[ModelEntry]:
        out: list[ModelEntry] = []
        if not self.models_dir.exists():
            return out
        for p in self.models_dir.rglob("*.gguf"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.models_dir).as_posix()
            meta = _read_meta(p)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            out.append(ModelEntry(
                model_id=rel,
                path=p,
                size=size,
                sha256=meta.get("sha256"),
                source=meta.get("source"),
                pulled_at=meta.get("pulled_at"),
            ))
        out.sort(key=lambda m: m.model_id)
        return out

    def get(self, model_id: str) -> ModelEntry | None:
        for m in self.list():
            if m.model_id == model_id:
                return m
        return None

    # ---- delete ----
    def delete(self, model_id: str, *, currently_loaded: str | None,
               force: bool = False) -> tuple[bool, str | None]:
        m = self.get(model_id)
        if not m:
            return False, "not_found"
        if currently_loaded == model_id and not force:
            return False, "loaded"
        try:
            m.path.unlink()
        except FileNotFoundError:
            pass
        mp = _meta_path(m.path)
        with contextlib.suppress(FileNotFoundError):
            mp.unlink()
        # Clean empty parent dirs back up to models_dir.
        parent = m.path.parent
        while parent != self.models_dir and parent.exists():
            try:
                parent.rmdir()
                parent = parent.parent
            except OSError:
                break
        self.db.log_event("model_deleted", {"model_id": model_id, "force": force})
        return True, None

    # ---- pull ----
    def start_pull(self, *, source: str, files: list[str] | None) -> str:
        download_id = secrets.token_urlsafe(8)
        self.db.execute(
            "INSERT INTO downloads(id, source, files_json, status, started_at)"
            " VALUES (?, ?, ?, 'pending', ?)",
            (download_id, source, json.dumps(files or []), time.time()),
        )
        cancel = asyncio.Event()
        self._cancel_flags[download_id] = cancel
        task = asyncio.create_task(self._run_pull(download_id, source, files or [], cancel))
        self._tasks[download_id] = task
        self.db.log_event("download_started",
                          {"id": download_id, "source": source, "files": files or []})
        return download_id

    def cancel_pull(self, download_id: str) -> bool:
        ev = self._cancel_flags.get(download_id)
        if not ev:
            return False
        ev.set()
        return True

    def delete_download(self, download_id: str) -> bool:
        """Remove a download record and clean up any partial files."""
        row = self.db.query_one("SELECT * FROM downloads WHERE id=?", (download_id,))
        if not row:
            return False
        # Cancel if still running
        self.cancel_pull(download_id)
        # Clean up partial files in _direct/
        files_json = row.get("files_json", "[]")
        try:
            files = json.loads(files_json) if files_json else []
        except (json.JSONDecodeError, TypeError):
            files = []
        direct_dir = self.models_dir / "_direct"
        # Try to infer filename from source or files list
        source = row.get("source", "")
        candidates: list[str] = list(files)
        if not candidates:
            tail = source.rsplit("/", 1)[-1].split("?", 1)[0]
            if tail:
                candidates.append(tail)
        for fn in candidates:
            base = os.path.basename(fn.replace("\\", "/"))
            if not base:
                continue
            for suffix in ("", ".part"):
                p = direct_dir / (base + suffix)
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            mp = direct_dir / (base + ".meta.json")
            if mp.exists():
                try:
                    mp.unlink()
                except OSError:
                    pass
        # Also clean HF-style downloads: models_dir/<repo>/
        if source.startswith("hf://") or source.startswith("hf:"):
            repo = source.removeprefix("hf://").removeprefix("hf:")
            repo_dir = self.models_dir / repo
            if repo_dir.exists():
                for fn in candidates:
                    base = os.path.basename(fn.replace("\\", "/"))
                    for p in repo_dir.rglob(base + "*"):
                        try:
                            p.unlink()
                        except OSError:
                            pass
        self.db.execute("DELETE FROM downloads WHERE id=?", (download_id,))
        return True

    def get_download(self, download_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM downloads WHERE id=?", (download_id,))
        if not row:
            return None
        return {
            "id": row["id"],
            "source": row["source"],
            "files": json.loads(row["files_json"]),
            "status": row["status"],
            "bytes_done": row["bytes_done"],
            "bytes_total": row["bytes_total"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
        }

    def list_downloads(self) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT id FROM downloads ORDER BY started_at DESC")
        return [d for d in (self.get_download(r["id"]) for r in rows) if d]

    # ---- internals ----
    def _check_disk(self, estimate_bytes: int) -> str | None:
        free = _free_bytes(self.models_dir)
        if estimate_bytes + SAFETY_MARGIN_BYTES > free:
            return f"insufficient disk: need {estimate_bytes + SAFETY_MARGIN_BYTES}B, free {free}B"
        if self.cfg.max_disk_gb:
            cap = self.cfg.max_disk_gb * 1024 ** 3
            current = _dir_size(self.models_dir)
            if current + estimate_bytes > cap:
                return (f"would exceed max_disk_gb={self.cfg.max_disk_gb}: "
                        f"{current + estimate_bytes} > {cap}")
        return None

    def _set_download(self, did: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(
            f"UPDATE downloads SET {cols} WHERE id=?",
            (*fields.values(), did),
        )

    @staticmethod
    def _normalise_hf_url(source: str, files: list[str]
                          ) -> tuple[str, list[str]]:
        """Convert a HuggingFace web URL into an hf:// source + files list.

        Handles patterns like:
          https://huggingface.co/org/repo/blob/main/file.gguf
          https://huggingface.co/org/repo
        """
        m = re.match(
            r'https?://huggingface\.co/([^/]+/[^/]+)(?:/(?:blob|resolve)/[^/]+/(.+))?$',
            source,
        )
        if m:
            repo = m.group(1)
            filename = m.group(2)
            source = f"hf://{repo}"
            if filename and filename not in files:
                files = [filename] + files
        return source, files

    async def _run_pull(self, did: str, source: str, files: list[str],
                        cancel: asyncio.Event) -> None:
        try:
            self._set_download(did, status="running")
            # Convert HF web URLs to hf:// sources automatically
            source, files = self._normalise_hf_url(source, files)
            if source.startswith("hf://") or source.startswith("hf:"):
                await self._pull_hf(did, source, files, cancel)
            elif source.startswith("http://") or source.startswith("https://"):
                await self._pull_http(did, source, files[0] if files else None, cancel)
            else:
                raise ValueError(f"unsupported source: {source}")
            if cancel.is_set():
                self._set_download(did, status="cancelled", finished_at=time.time())
            else:
                self._set_download(did, status="done", finished_at=time.time())
                self.db.log_event("download_done", {"id": did, "source": source})
        except asyncio.CancelledError:
            self._set_download(did, status="cancelled", finished_at=time.time())
            raise
        except Exception as e:
            log.exception("download %s failed", did)
            self._set_download(did, status="failed", finished_at=time.time(),
                               error=str(e))
            self.db.log_event("download_failed", {"id": did, "error": str(e)})
        finally:
            self._cancel_flags.pop(did, None)
            self._tasks.pop(did, None)

    async def _pull_hf(self, did: str, source: str, files: list[str],
                       cancel: asyncio.Event) -> None:
        """Use huggingface_hub.hf_hub_download in a thread, one file at a time
        (so we can update progress)."""
        from huggingface_hub import hf_hub_download  # type: ignore

        # source: hf://repo/id or hf:repo/id
        repo = _sanitize_hf_repo(source.removeprefix("hf://").removeprefix("hf:"))
        if not files:
            raise ValueError("HF source requires at least one file in `files`")
        # Validate every requested file before kicking off any IO.
        files = [_sanitize_hf_file(f) for f in files]

        token = os.environ.get(self.cfg.hf_token_env or "", None) or None
        target_root = _ensure_under(self.models_dir, self.models_dir / repo)
        target_root.mkdir(parents=True, exist_ok=True)

        # We don't know exact sizes a priori. Skip the disk-cap pre-check here
        # and rely on the post-download size + max_disk_gb after each file.
        bytes_done = 0
        for fn in files:
            if cancel.is_set():
                return
            self._set_download(did, bytes_done=bytes_done)
            log.info("pull[%s] downloading %s/%s", did, repo, fn)
            local = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo,
                filename=fn,
                local_dir=str(target_root),
                token=token,
                resume_download=True,
            )
            local_path = Path(local)
            # Defense in depth: even though we sanitized fn, confirm
            # huggingface_hub didn't write outside the sandbox.
            _ensure_under(self.models_dir, local_path)
            try:
                bytes_done += local_path.stat().st_size
            except OSError:
                pass
            sha = await asyncio.to_thread(_sha256_file, local_path)
            _write_meta(local_path, {
                "source": f"hf://{repo}",
                "filename": fn,
                "sha256": sha,
                "pulled_at": time.time(),
            })
            self._set_download(did, bytes_done=bytes_done, bytes_total=bytes_done)

            # Enforce cap after each file.
            err = self._check_disk(0)
            if err:
                raise RuntimeError(err)

    async def _pull_http(self, did: str, url: str, filename: str | None,
                         cancel: asyncio.Event) -> None:
        # HF blob URLs return an HTML page — rewrite to the resolve URL
        # which returns the actual file content.
        if "huggingface.co/" in url and "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        if not filename:
            filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
            if not filename:
                raise ValueError("could not infer filename from URL")
        # Sanitize: server-supplied filenames or URL tails could contain
        # path traversal segments. Strip to a safe basename.
        filename = _sanitize_download_filename(filename)
        direct_dir = _ensure_under(self.models_dir, self.models_dir / "_direct")
        direct_dir.mkdir(parents=True, exist_ok=True)
        target = _ensure_under(self.models_dir, direct_dir / filename)
        tmp = target.with_suffix(target.suffix + ".part")
        _ensure_under(self.models_dir, tmp)
        existing = tmp.stat().st_size if tmp.exists() else 0

        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            headers = {}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            async with client.stream("GET", url, headers=headers) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"GET {url} returned {r.status_code}")
                content_length = int(r.headers.get("content-length", "0") or 0)
                total = existing + content_length
                err = self._check_disk(total - existing)
                if err:
                    raise RuntimeError(err)
                self._set_download(did, bytes_total=total)
                bytes_done = existing
                with open(tmp, "ab") as f:
                    async for chunk in r.aiter_bytes(chunk_size=1024 * 64):
                        if cancel.is_set():
                            return
                        f.write(chunk)
                        bytes_done += len(chunk)
                        self._set_download(did, bytes_done=bytes_done)
        os.replace(tmp, target)
        sha = await asyncio.to_thread(_sha256_file, target)
        _write_meta(target, {
            "source": url, "filename": filename,
            "sha256": sha, "pulled_at": time.time(),
        })


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
