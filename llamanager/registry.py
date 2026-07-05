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

from .config import (
    Config, ENGINE_FAMILY, Profile, delete_model_entry,
    detect_engine_for_path, save_profile, set_model_default_profile,
    update_defaults,
)
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


def _decode_files_json(raw: str) -> tuple[list[str], str, bool]:
    """Read a downloads.files_json column in either the legacy (list of
    filenames) or new (object with ``files`` / ``subfolder`` /
    ``whole_repo``) format. Returns ``(files, subfolder, whole_repo)``.
    Safe against malformed JSON — returns sensible empty defaults."""
    if not raw:
        return [], "", False
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], "", False
    if isinstance(decoded, list):
        return [str(x) for x in decoded if isinstance(x, str)], "", False
    if isinstance(decoded, dict):
        files = decoded.get("files") or []
        if not isinstance(files, list):
            files = []
        return (
            [str(x) for x in files if isinstance(x, str)],
            str(decoded.get("subfolder") or ""),
            bool(decoded.get("whole_repo")),
        )
    return [], "", False


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

        # Pass 1: discover image-engine directories. Each candidate dir
        # is identified by a marker file (tokenizer_config / config.json
        # / model_index.json for Diffusers pipelines) or by containing a
        # flux*.gguf alongside a VAE. We collect their paths so the
        # GGUF and MLX passes can skip files inside them.
        image_dirs: set[Path] = set()
        for marker in ("tokenizer_config.json", "config.json", "model_index.json"):
            for p in self.models_dir.rglob(marker):
                d = p.parent
                if d == self.models_dir:
                    continue
                engine = detect_engine_for_path(d)
                if ENGINE_FAMILY.get(engine, "text") == "image":
                    image_dirs.add(d)
        # Flux2 dirs may not have any marker config — detect via file shape.
        for p in self.models_dir.rglob("*.gguf"):
            d = p.parent
            if d == self.models_dir or d in image_dirs:
                continue
            engine = detect_engine_for_path(d)
            if ENGINE_FAMILY.get(engine, "text") == "image":
                image_dirs.add(d)

        def _is_inside_image_dir(p: Path) -> bool:
            return any(p == d or d in p.parents for d in image_dirs)

        # Pass 2: GGUF single-file models (llama.cpp engines). Skip files
        # that live inside an image-engine dir (those are referenced from
        # the parent image model, not as standalone models).
        for p in self.models_dir.rglob("*.gguf"):
            if not p.is_file():
                continue
            if _is_inside_image_dir(p):
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

        # Pass 3: directory-style models (mlx, hidream, flux2). MLX dirs
        # use config.json; HiDream uses tokenizer_config + preprocessor;
        # flux2 uses file shape.
        seen: set[Path] = set(image_dirs)
        # Emit image-engine dirs first.
        for d in image_dirs:
            rel = d.relative_to(self.models_dir).as_posix()
            meta = _read_meta(d)
            size = _dir_size(d)
            out.append(ModelEntry(
                model_id=rel,
                path=d,
                size=size,
                sha256=meta.get("sha256"),
                source=meta.get("source"),
                pulled_at=meta.get("pulled_at"),
            ))
        # Then non-image directory models.
        for cfg_path in self.models_dir.rglob("config.json"):
            d = cfg_path.parent
            if d in seen or d == self.models_dir:
                continue
            # A Diffusers pipeline contains many subdirs that each have
            # their own config.json (text_encoder/, vae/, transformer/…).
            # Skip anything that lives inside an already-registered
            # image dir — those are pipeline components, not models.
            if _is_inside_image_dir(d):
                continue
            engine = detect_engine_for_path(d)
            if engine == "llama":
                # Loose dir without a clear shape — skip.
                continue
            if ENGINE_FAMILY.get(engine, "text") == "image":
                # Already handled above.
                continue
            has_weights = any(d.glob("*.safetensors")) or any(d.glob("*.npz"))
            if not has_weights:
                continue
            seen.add(d)
            rel = d.relative_to(self.models_dir).as_posix()
            meta = _read_meta(d)
            size = _dir_size(d)
            out.append(ModelEntry(
                model_id=rel,
                path=d,
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
            if m.path.is_dir():
                shutil.rmtree(m.path)
            else:
                m.path.unlink()
        except FileNotFoundError:
            pass
        if m.path.is_file():
            mp = _meta_path(m.path)
            with contextlib.suppress(FileNotFoundError):
                mp.unlink()
        # Clean empty parent dirs back up to models_dir.
        parent = m.path.parent if m.path.is_file() or not m.path.exists() else m.path.parent
        while parent != self.models_dir and parent.exists():
            try:
                parent.rmdir()
                parent = parent.parent
            except OSError:
                break
        # Cascade: drop the model's entire entry (profiles + default
        # pointer). In-memory cfg is also updated so the rest of the
        # request observes a consistent picture.
        removed_profiles = delete_model_entry(self.cfg.config_path, model_id)
        self.cfg.models.pop(model_id, None)
        self.db.log_event("model_deleted", {
            "model_id": model_id, "force": force,
            "removed_profiles": removed_profiles,
        })
        return True, None

    # ---- pull ----
    def start_pull(self, *, source: str, files: list[str] | None,
                   subfolder: str | None = None,
                   whole_repo: bool = False,
                   bytes_total: int = 0) -> str:
        """Kick off a background download.

        Modes:
          - ``files=[...]`` — fetch specific filenames inside an HF repo
            or a direct URL. The original pull behaviour.
          - ``whole_repo=True`` — fetch every file in the HF repo via
            ``snapshot_download``. Used for Diffusers pipelines and any
            non-GGUF model where the on-disk layout matters.
          - ``subfolder="dir"`` — fetch only files under ``dir/``
            (e.g. Z-Anime's ``diffusers/`` subdirectory). Implies
            ``whole_repo`` for that subtree.

        ``bytes_total`` lets the caller seed the progress bar when the
        size is known up-front (e.g. computed via :meth:`estimate_repo_size`).
        """
        download_id = secrets.token_urlsafe(8)
        # Stash mode metadata inside files_json so the run loop can
        # decide which fetch path to take without a separate column.
        meta = {
            "files": files or [],
            "subfolder": subfolder or "",
            "whole_repo": bool(whole_repo) or bool(subfolder),
        }
        self.db.execute(
            "INSERT INTO downloads(id, source, files_json, status, "
            "bytes_total, started_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (download_id, source, json.dumps(meta),
             int(bytes_total or 0), time.time()),
        )
        cancel = asyncio.Event()
        self._cancel_flags[download_id] = cancel
        task = asyncio.create_task(
            self._run_pull(download_id, source, files or [],
                           subfolder or None,
                           bool(whole_repo) or bool(subfolder),
                           cancel),
        )
        self._tasks[download_id] = task
        self.db.log_event("download_started", {
            "id": download_id, "source": source,
            "files": files or [], "subfolder": subfolder or "",
            "whole_repo": bool(whole_repo) or bool(subfolder),
        })
        return download_id

    async def estimate_repo_size(self, repo: str,
                                 subfolder: str | None = None,
                                 files: list[str] | None = None) -> int:
        """Sum the byte sizes of files in an HF repo (or a single subfolder).
        Cheap — one metadata API call. Returns 0 if the call fails so the
        UI can still proceed with an unknown estimate.

        When ``files`` is given, only those specific filenames are summed
        (used to seed the progress total for a single-file GGUF pull so the
        download strip can show a percentage, not just bytes-so-far)."""
        try:
            from huggingface_hub import HfApi  # type: ignore
        except Exception:
            return 0
        try:
            repo = _sanitize_hf_repo(repo)
        except ValueError:
            return 0
        token = os.environ.get(self.cfg.hf_token_env or "", None) or None
        api = HfApi(token=token)
        try:
            info = await asyncio.to_thread(
                api.model_info, repo, files_metadata=True,
            )
        except Exception as exc:
            log.warning("estimate_repo_size(%s) failed: %s", repo, exc)
            return 0
        total = 0
        prefix = (subfolder.strip("/") + "/") if subfolder else ""
        want = {f.strip() for f in files if f.strip()} if files else None
        for sib in getattr(info, "siblings", []):
            name = getattr(sib, "rfilename", "") or ""
            if want is not None:
                # Match by exact name or basename (callers pass either the
                # repo-relative path or just the file name).
                if name not in want and name.rsplit("/", 1)[-1] not in want:
                    continue
            elif prefix and not name.startswith(prefix):
                continue
            size = getattr(sib, "size", None)
            if isinstance(size, int) and size > 0:
                total += size
        return total

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
        files_json = row["files_json"] if "files_json" in row.keys() else "[]"
        files, _subfolder, _whole = _decode_files_json(files_json)
        direct_dir = self.models_dir / "_direct"
        # Try to infer filename from source or files list
        source = row["source"] if "source" in row.keys() else ""
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
                if _whole:
                    # Whole-repo / subfolder pull: nuke the (sub)tree that
                    # got downloaded so partial bytes don't litter the
                    # models_dir.
                    target = repo_dir / _subfolder if _subfolder else repo_dir
                    if target.exists() and target != self.models_dir:
                        try:
                            shutil.rmtree(target, ignore_errors=True)
                        except OSError:
                            pass
                else:
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
        files, subfolder, whole_repo = _decode_files_json(row["files_json"])
        return {
            "id": row["id"],
            "source": row["source"],
            "files": files,
            "subfolder": subfolder,
            "whole_repo": whole_repo,
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
    @staticmethod
    def _scan_dir_size(path: Path) -> int:
        """Total size of all files in a directory (non-recursive, fast)."""
        total = 0
        try:
            for f in path.iterdir():
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def _check_disk(self, estimate_bytes: int) -> str | None:
        gb = 1024 ** 3
        free = _free_bytes(self.models_dir)
        if estimate_bytes + SAFETY_MARGIN_BYTES > free:
            return (
                f"insufficient disk in {self.models_dir}: need "
                f"{(estimate_bytes + SAFETY_MARGIN_BYTES) / gb:.1f} GB "
                f"(model + 2 GB safety margin), only "
                f"{free / gb:.1f} GB free"
            )
        if self.cfg.max_disk_gb:
            cap = self.cfg.max_disk_gb * gb
            current = _dir_size(self.models_dir)
            if current + estimate_bytes > cap:
                return (
                    f"would exceed the configured max_disk_gb cap of "
                    f"{self.cfg.max_disk_gb} GB "
                    f"(models dir is {current / gb:.1f} GB, "
                    f"this download needs {estimate_bytes / gb:.1f} GB). "
                    f"Raise or set max_disk_gb=0 in config.toml to disable."
                )
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

    def _maybe_create_profile(self, source: str) -> None:
        """After a successful pull, create a default profile if none exists
        yet for the downloaded model.

        Scans the repo directory for a main GGUF file and an optional mmproj
        file. Skips if no main model is found (e.g. user only pulled an
        mmproj) or if a profile already references this model.

        For image-engine model directories, seeds the adapter's
        ``default_profiles()`` instead of the LLM-shaped default.
        """
        # Only handle HF pulls where files land in a repo subdirectory
        if not (source.startswith("hf://") or source.startswith("hf:")):
            return
        repo = source.removeprefix("hf://").removeprefix("hf:")
        repo_dir = self.models_dir / repo
        if not repo_dir.is_dir():
            return

        # First: is this an image-engine model? If so, register the dir
        # itself as the model_id and seed adapter-specific profiles.
        engine = detect_engine_for_path(repo_dir)
        if ENGINE_FAMILY.get(engine, "text") == "image":
            self._seed_image_profiles(repo, repo_dir, engine)
            return

        # Collect GGUF files, separate main models from mmproj files
        main_models: list[str] = []
        mmproj_files: list[str] = []
        for p in sorted(repo_dir.rglob("*.gguf")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.models_dir).as_posix()
            name_lower = p.name.lower()
            if "mmproj" in name_lower:
                mmproj_files.append(rel)
            else:
                main_models.append(rel)

        if not main_models:
            return  # only mmproj files — nothing to launch

        # Pick the first main model (alphabetically — usually the best quant)
        model_id = main_models[0]

        # Skip if this model already has any profile.
        existing_model = self.cfg.get_model(model_id)
        if existing_model and existing_model.profiles:
            return

        # Build a profile name from the repo + filename
        # e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF/Q4_K_M.gguf"
        #   -> "llama-3-2-1b-instruct-q4-k-m"
        stem = Path(model_id).stem  # "Q4_K_M"
        repo_name = repo.split("/")[-1]  # "Llama-3.2-1B-Instruct-GGUF"
        # Strip common suffixes from repo name
        for suffix in ("-GGUF", "-gguf", "_GGUF", "_gguf"):
            if repo_name.endswith(suffix):
                repo_name = repo_name[: -len(suffix)]
                break
        raw_name = f"{repo_name}-{stem}"
        # Normalise to lowercase, replace non-alnum with hyphens, collapse
        profile_name = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
        # Ensure uniqueness within this model.
        base_name = profile_name
        counter = 2
        if existing_model:
            while profile_name in existing_model.profiles:
                profile_name = f"{base_name}-{counter}"
                counter += 1

        # Pick the first mmproj if one exists in the same repo
        mmproj = mmproj_files[0] if mmproj_files else ""

        prof = Profile(
            name=profile_name,
            mmproj=mmproj,
            ctx_size=4096,
            args={"temp": 0.7},
        )

        try:
            save_profile(self.cfg.config_path, model_id, profile_name, prof)
            # Hot-reload into the running config so the profile is immediately
            # visible without a restart.
            m = self.cfg.ensure_model(model_id)
            m.profiles[profile_name] = prof
            # Register this as the model's default profile if the model
            # doesn't already have one. Also seed cfg.default_model if no
            # default model is set, but never overwrite an existing one.
            if not m.default_profile:
                m.default_profile = profile_name
                set_model_default_profile(
                    self.cfg.config_path, model_id, profile_name,
                )
            if not self.cfg.default_model:
                self.cfg.default_model = model_id
                update_defaults(
                    self.cfg.config_path,
                    default_model=model_id,
                )
            log.info("auto-created profile %r for %s", profile_name, model_id)
            self.db.log_event("profile_auto_created", {
                "profile": profile_name, "model": model_id, "mmproj": mmproj,
            })
        except Exception:
            log.exception("failed to auto-create profile for %s", model_id)

    def _seed_image_profiles(self, repo: str, repo_dir: Path, engine: str) -> None:
        """Auto-create the adapter's default profiles on an image-model pull."""
        from . import engines as _engines_pkg
        try:
            adapter = _engines_pkg.get(engine)
        except KeyError:
            log.warning("no adapter for engine %r; skipping default profiles", engine)
            return
        model_id = repo  # directory-style id
        existing_model = self.cfg.get_model(model_id)
        defaults = {}
        if hasattr(adapter, "default_profiles"):
            try:
                defaults = adapter.default_profiles(repo_dir)
            except TypeError:
                defaults = adapter.default_profiles()
        for pname, body in defaults.items():
            if existing_model and pname in existing_model.profiles:
                continue
            prof = Profile(name=pname, **body)
            try:
                save_profile(self.cfg.config_path, model_id, pname, prof)
                m = self.cfg.ensure_model(model_id)
                m.profiles[pname] = prof
                if not m.default_profile:
                    m.default_profile = pname
                    set_model_default_profile(self.cfg.config_path, model_id, pname)
                log.info("auto-created %s profile %r for %s",
                         engine, pname, model_id)
                self.db.log_event("profile_auto_created", {
                    "engine": engine, "profile": pname, "model": model_id,
                })
            except Exception:
                log.exception("failed to auto-create image profile %r for %s",
                              pname, model_id)

    async def _run_pull(self, did: str, source: str, files: list[str],
                        subfolder: str | None,
                        whole_repo: bool,
                        cancel: asyncio.Event) -> None:
        try:
            self._set_download(did, status="running")
            # Convert HF web URLs to hf:// sources automatically
            source, files = self._normalise_hf_url(source, files)
            if source.startswith("hf://") or source.startswith("hf:"):
                if whole_repo or subfolder:
                    await self._pull_hf_snapshot(did, source, subfolder, cancel)
                else:
                    await self._pull_hf(did, source, files, cancel)
            elif source.startswith("http://") or source.startswith("https://"):
                await self._pull_http(did, source, files[0] if files else None, cancel)
            else:
                raise ValueError(f"unsupported source: {source}")
            if cancel.is_set():
                self._set_download(did, status="cancelled", finished_at=time.time())
            else:
                self._maybe_create_profile(source)
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

    async def _pull_hf_snapshot(self, did: str, source: str,
                                 subfolder: str | None,
                                 cancel: asyncio.Event) -> None:
        """Fetch an entire HF repo (or a single subfolder) via
        ``snapshot_download``. Used for Diffusers pipelines like
        Z-Image and Z-Anime where the model is a tree of files."""
        from huggingface_hub import snapshot_download  # type: ignore

        repo = _sanitize_hf_repo(source.removeprefix("hf://").removeprefix("hf:"))
        token = os.environ.get(self.cfg.hf_token_env or "", None) or None
        target_root = _ensure_under(self.models_dir, self.models_dir / repo)
        target_root.mkdir(parents=True, exist_ok=True)

        allow_patterns: list[str] | None = None
        if subfolder:
            sf = subfolder.strip("/")
            if not sf or ".." in sf.split("/"):
                raise ValueError(f"unsafe subfolder: {subfolder!r}")
            allow_patterns = [f"{sf}/*", f"{sf}/**/*"]

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=repo,
                local_dir=str(target_root),
                token=token,
                allow_patterns=allow_patterns,
                resume_download=True,
                # tqdm would spam stderr; we poll dir size instead.
                tqdm_class=None,
            ),
        )

        # Poll for progress via on-disk byte count while the snapshot
        # download runs. We compare against ``bytes_total`` (seeded by
        # ``estimate_repo_size`` before start_pull was called); if that
        # was zero, the UI shows raw bytes-done only.
        scan_root = target_root / subfolder if subfolder else target_root
        while not future.done():
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break
            try:
                cur = _dir_size(scan_root)
            except OSError:
                cur = 0
            self._set_download(did, bytes_done=cur)
            if cancel.is_set():
                future.cancel()
                return

        # Surface any exception from the executor.
        future.result()

        # Final size reconciliation: prefer the actual on-disk total over
        # the API-estimated bytes_total so the bar lands at 100%.
        final = _dir_size(scan_root)
        self._set_download(did, bytes_done=final, bytes_total=final)
        _ensure_under(self.models_dir, scan_root)
        _write_meta(target_root, {
            "source": f"hf://{repo}",
            "subfolder": subfolder or "",
            "whole_repo": True,
            "pulled_at": time.time(),
        })
        # Cap enforcement after the whole pull lands.
        err = self._check_disk(0)
        if err:
            raise RuntimeError(err)

    async def _pull_hf(self, did: str, source: str, files: list[str],
                       cancel: asyncio.Event) -> None:
        """Use huggingface_hub.hf_hub_download in a thread, one file at a time.

        Monitors incomplete download files to report progress while the
        blocking hf_hub_download runs in a thread.
        """
        from huggingface_hub import hf_hub_download  # type: ignore

        # source: hf://repo/id or hf:repo/id
        repo = _sanitize_hf_repo(source.removeprefix("hf://").removeprefix("hf:"))
        if not files:
            raise ValueError("HF source requires at least one file in `files`")
        files = [_sanitize_hf_file(f) for f in files]

        token = os.environ.get(self.cfg.hf_token_env or "", None) or None
        target_root = _ensure_under(self.models_dir, self.models_dir / repo)
        target_root.mkdir(parents=True, exist_ok=True)

        bytes_done = 0
        for fn in files:
            if cancel.is_set():
                return
            self._set_download(did, bytes_done=bytes_done)
            log.info("pull[%s] downloading %s/%s", did, repo, fn)

            # Start the blocking download in a thread
            import concurrent.futures
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=repo,
                    filename=fn,
                    local_dir=str(target_root),
                    token=token,
                    resume_download=True,
                ),
            )

            # Poll for progress while the download runs by checking
            # incomplete files (.incomplete) in the HF cache or local_dir
            while not future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    break
                # Scan for .incomplete or partial files to estimate progress
                cur = self._scan_dir_size(target_root)
                self._set_download(did, bytes_done=bytes_done + cur)
                if cancel.is_set():
                    future.cancel()
                    return

            local = future.result()
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
