"""Gallery thumbnails.

The gallery grid used to download every full-size PNG (≈1–5 MB each) or
whole MP4 to draw a ~260px tile, sixty at a time. This module gives each
gallery file a small JPEG derivative that is generated once, cached on disk
next to the gallery, and served by its own route.

Layout: ``<images_dir>/.thumbs/<day>/<origin>/<name>.jpg`` — the leading dot
keeps the cache out of the gallery walk (``_list_gallery`` skips dot-dirs)
and out of the disk-cap accounting of the originals.

* PNG → Pillow ``thumbnail`` to :data:`THUMB_PX` on the long edge, JPEG.
* MP4 → one frame via ``ffmpeg`` (a real dependency for video posters; a
  missing binary is reported as :class:`ThumbError`, not papered over).

Thumbnails are regenerated when the source is newer than the cached file
(a re-written original never shows a stale tile). Concurrent requests for
the same thumbnail share one generation via a per-path lock.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

log = logging.getLogger("llamanager.thumbs")

#: Long edge of a gallery thumbnail. Tiles are ~260px wide on desktop and
#: half a phone screen on mobile; 512 covers 2× displays for both.
THUMB_PX = 512
#: JPEG quality — visually clean for photographic diffusion output while
#: keeping a portrait 512px tile around 40–60 KB.
THUMB_JPEG_QUALITY = 82
THUMBS_DIRNAME = ".thumbs"
THUMB_SUFFIX = ".jpg"
#: How long ffmpeg may take on one poster frame before we give up.
FFMPEG_TIMEOUT_S = 30
#: Portable per-file locking so two concurrent gallery loads that both miss
#: the cache for the same item generate it once. Bounded: keys are the
#: thumbnail path and entries are removed when the last waiter leaves.
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


class ThumbError(RuntimeError):
    """The thumbnail could not be produced. The message names the cause."""


def thumb_path(images_dir: Path, day: str, origin: str, name: str) -> Path:
    """Where the thumbnail for ``<images_dir>/<day>/<origin>/<name>`` lives."""
    return images_dir / THUMBS_DIRNAME / day / origin / (name + THUMB_SUFFIX)


def _lock_for(p: Path) -> threading.Lock:
    with _locks_guard:
        lk = _locks.get(p)
        if lk is None:
            lk = threading.Lock()
            _locks[p] = lk
        return lk


def _release_lock(p: Path, lk: threading.Lock) -> None:
    lk.release()
    with _locks_guard:
        # Drop the entry when nobody else is holding/waiting; a stray
        # re-creation on the next request is harmless.
        if not lk.locked():
            _locks.pop(p, None)


def _render_png(src: Path, dst: Path) -> None:
    from PIL import Image

    with Image.open(src) as im:
        # draft() lets Pillow decode JPEG sources at reduced size; PNG
        # ignores it, which is fine — thumbnail() does the real work.
        im.draft("RGB", (THUMB_PX, THUMB_PX))
        im = im.convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=THUMB_JPEG_QUALITY, optimize=True)
    _atomic_write(dst, buf.getvalue())


def _render_mp4(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ThumbError(
            "ffmpeg is not installed — it is required to build video "
            "posters for the gallery (install it, e.g. `apt install ffmpeg`)")
    tmp = dst.with_name(dst.name + f".{os.getpid()}.part")
    # First frame, scaled to the long-edge cap; "-2" keeps the other side
    # even, which the JPEG encoder wants.
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-frames:v", "1",
        "-vf", (f"scale='if(gt(iw,ih),min({THUMB_PX},iw),-2)':"
                f"'if(gt(iw,ih),-2,min({THUMB_PX},ih))'"),
        "-q:v", "3",
        "-f", "image2", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        raise ThumbError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s on {src.name}")
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        err = (r.stderr or "").strip().splitlines()
        detail = err[-1] if err else f"rc={r.returncode}"
        raise ThumbError(f"ffmpeg could not extract a poster from {src.name}: {detail}")
    os.replace(tmp, dst)


def _atomic_write(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(dst.name + f".{os.getpid()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def ensure_thumbnail(src: Path, dst: Path) -> Path:
    """Return ``dst``, generating it from ``src`` if missing or stale.

    Blocking; call from a worker thread in async code (see
    :func:`ensure_thumbnail_async`). Raises :class:`ThumbError` (or
    ``FileNotFoundError`` for a vanished source) — never returns a
    placeholder.
    """
    if not src.is_file():
        raise FileNotFoundError(str(src))
    src_mtime = src.stat().st_mtime
    try:
        st = dst.stat()
        if st.st_size > 0 and st.st_mtime >= src_mtime:
            return dst
    except FileNotFoundError:
        pass
    lk = _lock_for(dst)
    lk.acquire()
    try:
        # Re-check under the lock: another thread may have just built it.
        try:
            st = dst.stat()
            if st.st_size > 0 and st.st_mtime >= src_mtime:
                return dst
        except FileNotFoundError:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        low = src.suffix.lower()
        if low == ".png":
            _render_png(src, dst)
        elif low == ".mp4":
            _render_mp4(src, dst)
        else:
            raise ThumbError(f"no thumbnailer for {src.suffix!r}")
        # Stamp the thumb at least as new as its source so the staleness
        # test above is monotonic even on coarse filesystem clocks.
        try:
            os.utime(dst, (src_mtime + 1, src_mtime + 1))
        except OSError:
            pass
        return dst
    finally:
        _release_lock(dst, lk)


async def ensure_thumbnail_async(src: Path, dst: Path) -> Path:
    """Non-blocking wrapper: image decode / ffmpeg run in a worker thread."""
    return await asyncio.to_thread(ensure_thumbnail, src, dst)


def warm_thumbnail(images_dir: Path, out_path: Path) -> None:
    """Build the thumbnail for a freshly written gallery file.

    Called from the image runner right after a generation lands so the
    first gallery view after a run doesn't pay the decode. Failures are
    logged, not raised: the on-demand route regenerates on the next view,
    and a thumbnail problem must never cost the operator a finished image.
    """
    try:
        rel = out_path.resolve().relative_to(images_dir.resolve())
    except ValueError:
        log.warning("thumbnail warm skipped: %s is not under %s",
                    out_path, images_dir)
        return
    parts = rel.parts
    if len(parts) != 3:
        log.warning("thumbnail warm skipped: unexpected gallery layout %s", rel)
        return
    dst = thumb_path(images_dir, *parts)
    try:
        ensure_thumbnail(out_path, dst)
    except Exception:  # noqa: BLE001 — side effect of a finished run
        log.warning("thumbnail warm failed for %s", out_path, exc_info=True)


def drop_thumbnail(images_dir: Path, gallery_file: Path) -> None:
    """Remove the cached thumbnail of a gallery file that is being deleted."""
    try:
        rel = gallery_file.resolve().relative_to(images_dir.resolve())
    except ValueError:
        return
    if len(rel.parts) != 3:
        return
    p = thumb_path(images_dir, *rel.parts)
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        log.debug("could not remove thumbnail %s", p, exc_info=True)
