"""Unified, natural-language activity stream.

Merges three classes of raw log source into one human-readable feed:

* lifecycle events recorded in the ``events`` table by llamanager itself
  (server_starting, server_swap_*, image_generate_*, download_*, …);
* the tail of ``llama-server.log`` — llama.cpp's own stdout/stderr;
* per-engine logs for each diffusion runner (``hidream.log``,
  ``flux2.log``, ``z_image.log``, …), which capture the subprocess'
  stdout/stderr from its Python script.

Each line in llama.cpp's log starts with ``MM.SS.ms.us`` *relative to
the server process start* — i.e. it resets to 0 every time we relaunch
the server and the file is append-only across multiple sessions. We
detect those session boundaries (timestamp jumping backwards) and
anchor only the **most recent** session to wall-clock time, using the
latest ``server_starting`` event in the DB (or file mtime as a
fallback).

Image-engine logs have no per-line timestamps at all — they're raw
stdout from a one-shot Python invocation. We stamp each line near the
file's mtime in stable, monotonic order, so the entries cluster at the
wall-clock time of the most recent generation and interleave naturally
with the surrounding DB events.

In every case the parser keeps only the lines that carry real signal —
request boundaries, final timings, errors, lifecycle messages — and
discards per-step progress, prompt-cache state dumps, tqdm bars, and
other diagnostic chatter. Lines matching ``error|failed|fatal|panic|
out of memory`` are passed through verbatim with a warning level.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .db import DB
from .events import list_events


# Sources whose raw files live in ``logs_dir`` but are *not* per-engine
# logs and should never appear as image-engine entries. Bootstrap logs
# come from the Windows service wrapper, not from an engine subprocess.
_RESERVED_LOG_STEMS = {
    "llama-server",
    "llamanager",
    "llamanager-bootstrap",
}

# Chip name shown in the activity feed for the llama.cpp text-server.
_LLAMA_CPP_CHIP = "llama.cpp"


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    ts: float           # wall-clock seconds since epoch (best-effort)
    level: str          # "info" | "warn" | "error"
    source: str         # chip label: "llamanager" | "llama.cpp" | "<engine>"
    text: str           # one-line, natural-language

    def fmt(self) -> str:
        when = datetime.fromtimestamp(self.ts).strftime("%H:%M:%S")
        chip = "!!" if self.level == "error" else ("! " if self.level == "warn" else "  ")
        return f"{when} {chip} [{self.source}] {self.text}"


# ---------------------------------------------------------------------------
# DB event -> natural language
# ---------------------------------------------------------------------------

def _fmt_event(kind: str, payload: dict[str, Any]) -> tuple[str, str] | None:
    """Return (level, text) for one event, or None to drop it.

    Keep this map exhaustive over kinds emitted by db.log_event(). Unknown
    kinds fall through to a generic renderer so we never silently drop
    something new.
    """
    p = payload or {}
    model = p.get("model") or p.get("model_id") or ""
    profile = p.get("profile") or ""

    def with_model(base: str) -> str:
        bits = [base]
        if model:
            bits.append(f"`{model}`")
        if profile:
            bits.append(f"(profile {profile})")
        return " ".join(bits)

    match kind:
        case "server_starting":
            return ("info", with_model("Starting llama-server for"))
        case "server_running":
            pid = p.get("pid")
            tail = f" (pid {pid})" if pid else ""
            return ("info", with_model("llama-server is up — serving") + tail)
        case "server_stopped":
            return ("info", "llama-server stopped")
        case "server_crashed":
            rc = p.get("returncode")
            return ("error", f"llama-server crashed (exit code {rc})")
        case "server_swap_begin":
            frm = p.get("from") or p.get("from_model") or "?"
            to = p.get("to") or p.get("to_model") or "?"
            return ("info", f"Swapping model: {frm} → {to}")
        case "server_swap_ok":
            return ("info", with_model("Model swap complete —"))
        case "server_swap_fail":
            err = p.get("error") or "unknown error"
            return ("error", f"Model swap failed: {err}")
        case "server_degraded":
            return ("warn", f"Server degraded: {p.get('reason') or 'unknown'}")
        case "server_supervisor_giveup":
            return ("error", f"Supervisor gave up restarting: {p.get('reason')}")
        case "text_yield_to_image_begin":
            return ("info", "Pausing text engine to free GPU for image generation")
        case "text_yield_to_image_restored":
            return ("info", with_model("Text engine restored —"))
        case "text_yield_to_image_restore_failed":
            return ("warn", f"Text engine restore failed: {p.get('error')}")
        case "text_yield_to_image_kept_unloaded":
            return ("info", "Text engine kept unloaded after image task")
        case "dispatch_swap_required":
            req = p.get("model") or "?"
            return ("info", f"Queued request needs a model swap to `{req}`")
        case "dispatch_swap_done":
            return ("info", with_model("Swap done for queued request —"))
        case "request_cancelled":
            return ("info", f"Request {p.get('id')} cancelled")
        case "image_generate_begin":
            eng = p.get("engine") or "image"
            return ("info", f"{eng}: generation started")
        case "image_generate_done":
            eng = p.get("engine") or "image"
            secs = p.get("duration_s") or p.get("elapsed_s")
            tail = f" in {secs:.1f}s" if isinstance(secs, (int, float)) else ""
            return ("info", f"{eng}: image generated{tail}")
        case "image_generate_cancelled":
            eng = p.get("engine") or "image"
            return ("info", f"{eng}: generation cancelled")
        case "image_generate_failed":
            eng = p.get("engine") or "image"
            reason = p.get("error") or (
                f"exit code {p['rc']}" if p.get("rc") is not None else "unknown"
            )
            return ("error", f"{eng}: generation failed — {reason}")
        case "image_engine_degraded":
            eng = p.get("engine") or "image"
            return ("warn", f"{eng}: engine degraded — {p.get('reason')}")
        case "install_started":
            return ("info", f"Install started: {p.get('engine') or p.get('what')}")
        case "install_done":
            return ("info", f"Install finished: {p.get('engine') or p.get('what')}")
        case "install_failed":
            return ("error", f"Install failed: {p.get('error')}")
        case "download_started":
            return ("info", f"Download started: {p.get('source') or p.get('id')}")
        case "download_done":
            return ("info", f"Download finished: {p.get('source') or p.get('id')}")
        case "download_failed":
            return ("error", f"Download failed: {p.get('error')}")
        case "model_deleted":
            return ("info", f"Model deleted: {p.get('path') or p.get('model')}")
        case "profile_auto_created":
            return ("info", f"Auto-created profile for `{p.get('model')}`")
        case "config_reloaded":
            via = p.get("via")
            tail = f" (via {via})" if via else ""
            return ("info", f"Config reloaded{tail}")
        case "origin_created":
            return ("info", f"Origin created: {p.get('name')}")
        case "origin_updated":
            return ("info", f"Origin updated (id {p.get('id')})")
        case "origin_deleted":
            return ("info", f"Origin deleted (id {p.get('id')})")
        case "origin_key_rotated":
            return ("info", f"Origin key rotated (id {p.get('id')})")
        case "bootstrap_created":
            return ("info", "Bootstrap origin created")
        case _:
            # Unknown kind — render a compact fallback so it isn't silently lost
            return ("info", f"event: {kind} {payload!r}")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _read_tail(path: Path, max_bytes: int = 1_000_000) -> str:
    """Read the last ``max_bytes`` bytes of ``path`` as text, best-effort."""
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            buf = f.read()
        return buf.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _split_tqdm_runs(text: str) -> str:
    """Image-engine logs often write tqdm progress bars in carriage-return
    overwrite mode, so a "single line" in the file can contain dozens of
    bar updates separated by ``\\r``. Newer Python tqdm flushes them to a
    file with all updates joined by ``\\r`` and a final ``\\n``.

    We split on either ``\\r`` or ``\\n`` so each bar update can be
    filtered individually — the final "100%" update is the one we keep.
    """
    return re.sub(r"\r+", "\n", text)


_ERROR_HINT = re.compile(r"\b(error|failed|fatal|panic|out of memory|oom|abort)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# llama.cpp log parsing
# ---------------------------------------------------------------------------

# Line format: "MM.SS.ms.us L TAG: message"
# The first column resets to 0 each time the server restarts.
_LLAMA_LINE_RE = re.compile(
    r"^\s*(?P<min>\d+)\.(?P<sec>\d+)\.(?P<ms>\d+)\.(?P<us>\d+)\s+(?P<lvl>[IWED])\s+(?P<rest>.*)$"
)


@dataclass
class _RawLLM:
    offset: float       # seconds since this server session started
    level: str          # "info" | "warn" | "error"
    rest: str           # everything after the timestamp + level


def _parse_llama_line(line: str) -> _RawLLM | None:
    m = _LLAMA_LINE_RE.match(line)
    if not m:
        return None
    minute = int(m["min"])
    sec = int(m["sec"])
    ms = int(m["ms"])
    us = int(m["us"])
    offset = minute * 60 + sec + ms / 1000.0 + us / 1_000_000.0
    raw_lvl = m["lvl"]
    level = {"E": "error", "W": "warn"}.get(raw_lvl, "info")
    rest = m["rest"]
    if _ERROR_HINT.search(rest):
        level = "error" if level == "info" else level
    return _RawLLM(offset=offset, level=level, rest=rest)


def _split_llama_sessions(lines: Iterable[_RawLLM]) -> list[list[_RawLLM]]:
    """Group raw llama.cpp lines into contiguous server sessions.

    A new session starts whenever the offset jumps backwards relative to
    the previous line (llama-server reset its clock). Sub-second jitter
    is tolerated; a real reset shrinks the offset by more than a few
    seconds.
    """
    sessions: list[list[_RawLLM]] = []
    current: list[_RawLLM] = []
    prev: float | None = None
    for entry in lines:
        if prev is not None and entry.offset + 5.0 < prev:
            if current:
                sessions.append(current)
            current = []
        current.append(entry)
        prev = entry.offset
    if current:
        sessions.append(current)
    return sessions


def _shape_llama(raw: _RawLLM) -> tuple[str, str] | None:
    s = raw.rest
    lvl = raw.level

    # --- request lifecycle ---------------------------------------------------
    m = re.match(r"slot\s+launch_slot_:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+processing task", s)
    if m:
        return ("info", f"Slot {m[1]}: started request (task {m[2]})")

    m = re.match(r"slot\s+release:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+stop processing:\s+(.*)$", s)
    if m:
        return ("info", f"Slot {m[1]}: finished request (task {m[2]}) — {m[3]}")

    # Collapse the three-line print_timing summary into one. We anchor on
    # the "total time" line and drop prompt-eval / eval-time / graphs-reused
    # / n_decoded progress.
    m = re.match(r"slot\s+print_timing:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+total time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens", s)
    if m:
        total_ms = float(m[3])
        tokens = int(m[4])
        tps = tokens * 1000.0 / total_ms if total_ms > 0 else 0.0
        return ("info", f"Slot {m[1]} (task {m[2]}) timings: {tokens} tokens in {total_ms/1000:.2f}s ({tps:.1f} tok/s overall)")

    if re.match(r"slot\s+print_timing:.*\b(n_decoded|prompt eval time|eval time|graphs reused)\b", s):
        return None

    # --- prompt-cache / slot-internals chatter -------------------------------
    if re.match(r"srv\s+(get_availabl|update|load):", s):
        return None
    if re.match(r"srv\s+prompt_save:", s):
        return None
    m = re.match(r"srv\s+stop:\s+cancel task,\s+id_task\s*=\s*(\d+)", s)
    if m:
        return ("info", f"Cancel requested for task {m[1]}")
    if re.match(r"slot\s+(get_availabl|update_slots):", s):
        if "restored context checkpoint" in s:
            return ("info", "Slot context checkpoint restored from prompt cache")
        if "forcing full prompt re-processing" in s:
            return ("info", "Slot reprocessing full prompt (no usable cache)")
        if "erased invalidated" in s:
            return None
        return None
    if "all slots are idle" in s:
        return None
    if re.match(r"slot\s+load_model:", s):
        return None
    if re.match(r"init:\s+chat template", s):
        return None
    if re.match(r"common_speculative_init:", s):
        return None
    if re.match(r"load_hparams:", s):
        return ("info", s.split("(", 1)[0].strip())

    # --- startup lines worth surfacing --------------------------------------
    m = re.match(r"srv\s+load_model:\s+loading model '(.+)'", s)
    if m:
        return ("info", f"Loading model: {m[1]}")
    if re.match(r"srv\s+llama_server:\s+model loaded", s):
        return ("info", "Model loaded into VRAM")
    m = re.match(r"srv\s+llama_server:\s+server is listening on (.+)", s)
    if m:
        return ("info", f"llama-server listening on {m[1]}")
    if re.match(r"common_init_from_params:\s+warming up", s):
        return ("info", "Warming up model with empty run")
    m = re.match(r"srv\s+load_model:\s+initializing slots,\s+n_slots = (\d+)", s)
    if m:
        return ("info", f"Initialising {m[1]} slots")

    # --- shutdown ------------------------------------------------------------
    if "cleaning up before exit" in s:
        return ("info", "llama-server shutting down")

    # --- errors / warnings always pass through verbatim ----------------------
    if lvl in ("warn", "error"):
        return (lvl, s.strip())

    return None


def _llama_entries(log_path: Path, anchor_ts: float | None) -> list[Entry]:
    """Parse the tail of llama.cpp's log and return entries from the latest session."""
    text = _read_tail(log_path)
    if not text:
        return []
    raw: list[_RawLLM] = []
    for line in text.splitlines():
        r = _parse_llama_line(line)
        if r is not None:
            raw.append(r)
    if not raw:
        return []
    sessions = _split_llama_sessions(raw)
    latest = sessions[-1]
    if not latest:
        return []
    if anchor_ts is None:
        try:
            mtime = log_path.stat().st_mtime
            anchor_ts = mtime - latest[-1].offset
        except OSError:
            anchor_ts = time.time() - latest[-1].offset

    out: list[Entry] = []
    for r in latest:
        shaped = _shape_llama(r)
        if shaped is None:
            continue
        lvl, txt = shaped
        out.append(Entry(
            ts=anchor_ts + r.offset,
            level=lvl,
            source=_LLAMA_CPP_CHIP,
            text=txt,
        ))
    return out


# ---------------------------------------------------------------------------
# Image-engine log parsing
# ---------------------------------------------------------------------------

# tqdm progress bars look like:
#   "Generating:  72%|███████▏  | 36/50 [03:54<01:30,  6.45s/it]"
# The bar character class includes the block-drawing glyphs tqdm renders.
_TQDM_RE = re.compile(
    r"^(?P<label>[A-Za-z ][A-Za-z _-]*?)\s*:\s+"
    r"(?P<pct>\d+)\s*%\|[^|]*\|\s+"
    r"(?P<step>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:?]+),\s*(?P<rate>[\d.]+)\s*(?P<unit>\S+)\]"
)


def _shape_image_engine(line: str) -> tuple[str, str] | None:
    """Parse one line of an image-engine subprocess log."""
    s = line.strip()
    if not s:
        return None

    # Strip a leading tqdm bar fragment off lines like
    #   "Generating: 0%|...|0/28 [00:00<?, ?it/s]C:\…\foo.py:96: UserWarning: …"
    # where a warning printed without a leading newline got concatenated
    # behind the bar. The bar itself isn't interesting (pct<100); we
    # want to evaluate whatever comes after.
    tqdm_strip = re.match(r"^[A-Za-z ][A-Za-z _-]*?\s*:\s+\d+\s*%\|[^|]*\|\s+\d+\s*/\s*\d+\s+\[[^\]]+\](.*)$", s)
    if tqdm_strip:
        tail = tqdm_strip[1].strip()
        if tail:
            sub = _shape_image_engine(tail)
            if sub is not None:
                return sub
            # No useful payload in the tail; fall through to handle the
            # bar itself.

    # tqdm progress: only keep the final "100%" tick of any given label, which
    # carries the total wall-clock time. Skip intermediate updates.
    m = _TQDM_RE.match(s)
    if m:
        if int(m["pct"]) < 100:
            return None
        label = m["label"].strip()
        elapsed = m["elapsed"]
        return ("info", f"{label} complete — {m['total']} steps in {elapsed}")

    # Engine-emitted tagged lines (HiDream / Flux2 wrappers).
    m = re.match(r"\[inference\]\s+Loading processor and model from\s+(.+)$", s, re.IGNORECASE)
    if m:
        return ("info", f"Loading model from {m[1]}")
    m = re.match(r"\[inference\]\s+Saved\s+->\s+(.+)$", s, re.IGNORECASE)
    if m:
        return ("info", f"Saved image: {Path(m[1]).name}")
    m = re.match(r"\[inference\]\s+(.+)$", s, re.IGNORECASE)
    if m:
        return ("info", m[1])
    m = re.match(r"\[warning\]\s+(.+)$", s, re.IGNORECASE)
    if m:
        return ("warn", m[1])
    m = re.match(r"\[error\]\s+(.+)$", s, re.IGNORECASE)
    if m:
        return ("error", m[1])

    # HiDream's "🚨 X is part of … signature, but not documented" repeats
    # on every load and is an upstream-library hygiene notice, not
    # something the operator can act on. Drop it.
    if s.startswith("🚨") and "is part of" in s and "signature, but not documented" in s:
        return None
    m = re.match(r"🚨\s+(.+)$", s)
    if m:
        return ("warn", m[1])

    # Python tracebacks — the "Traceback" / "Error:" line is enough.
    if s.startswith("Traceback"):
        return ("error", s)
    if re.match(r"\w*(Error|Exception):\s+", s):
        return ("error", s)
    # UserWarnings often come with a "file:line:" prefix. Surface just
    # the message so the chip + text reads cleanly.
    m = re.search(r"\bUserWarning:\s+(.+)$", s)
    if m:
        return ("warn", f"UserWarning: {m[1]}")

    # Generic errors / warnings sniffed by keyword (don't double-fire on
    # tqdm lines because those returned earlier). Cap the noise on
    # deprecation notices too.
    if "DeprecationWarning" in s or "is deprecated" in s.lower():
        return None
    if _ERROR_HINT.search(s):
        return ("error", s)

    return None


def _image_engine_entries(engine: str, log_path: Path,
                          anchor_ts: float | None) -> list[Entry]:
    """Parse the tail of an image engine's log.

    Image-engine logs have no per-line timestamps. We stamp each line near
    the file's mtime in monotonic order so they sort correctly against
    DB events and don't pile up at exactly one timestamp.

    Optional ``anchor_ts`` overrides mtime — typically the most recent
    matching ``image_generate_begin`` event for this engine.
    """
    text = _read_tail(log_path)
    if not text:
        return []
    text = _split_tqdm_runs(text)
    lines = text.splitlines()

    shaped: list[tuple[str, str]] = []
    for line in lines:
        ent = _shape_image_engine(line)
        if ent is None:
            continue
        # Collapse consecutive duplicates — tqdm typically prints its
        # "100% complete" tick twice in a row when it closes.
        if shaped and shaped[-1] == ent:
            continue
        shaped.append(ent)

    if not shaped:
        return []

    if anchor_ts is None:
        try:
            anchor_ts = log_path.stat().st_mtime
        except OSError:
            anchor_ts = time.time()

    # Spread the entries backwards from the anchor in stable order. We
    # tuck them all inside a one-millisecond window so they cluster
    # tightly with the surrounding DB events at the same wall-clock time
    # but still sort deterministically among themselves.
    n = len(shaped)
    out: list[Entry] = []
    for i, (lvl, txt) in enumerate(shaped):
        out.append(Entry(
            ts=anchor_ts - (n - 1 - i) * 1e-3,
            level=lvl,
            source=engine,
            text=txt,
        ))
    return out


# ---------------------------------------------------------------------------
# source discovery & public API
# ---------------------------------------------------------------------------

def discover_engine_logs(logs_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(engine_name, log_path)`` for each ``<engine>.log`` file in
    ``logs_dir``, excluding the reserved llamanager/llama-server logs.

    The engine name is the filename stem; the dropdown uses it both as the
    chip label and the ``source`` query parameter, so it matches whatever
    ``image_runner`` writes (which in turn is whatever ``ADAPTERS`` is
    keyed on plus any future engine that lands new log files).
    """
    if not logs_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    try:
        for p in sorted(logs_dir.glob("*.log")):
            stem = p.stem
            if stem in _RESERVED_LOG_STEMS:
                continue
            out.append((stem, p))
    except OSError:
        pass
    return out


def build_activity(db: DB, logs_dir: Path, *, tail: int = 200) -> list[Entry]:
    """Return the most recent ``tail`` activity entries, newest last.

    Pulls roughly ``tail`` DB events plus the latest llama.cpp session and
    the tail of every per-engine log in ``logs_dir``. Merges by timestamp
    and trims to the last ``tail`` lines.
    """
    events = list_events(db, limit=max(tail * 2, 200))

    db_entries: list[Entry] = []
    for ev in events:
        shaped = _fmt_event(ev["kind"], ev.get("payload") or {})
        if shaped is None:
            continue
        lvl, txt = shaped
        db_entries.append(Entry(
            ts=ev["ts"],
            level=lvl,
            source="llamanager",
            text=txt,
        ))

    # Anchor llama.cpp's latest session to the most recent server_starting
    # event if we have one (most accurate); otherwise fall back to file
    # mtime inside ``_llama_entries``.
    llama_anchor: float | None = None
    for ev in events:
        if ev["kind"] == "server_starting":
            llama_anchor = ev["ts"]
            break

    llama_path = logs_dir / "llama-server.log"
    llm_entries = _llama_entries(llama_path, llama_anchor)

    # Image engines: anchor each to the most recent matching
    # ``image_generate_begin`` event for that engine, if any.
    engine_anchors: dict[str, float] = {}
    for ev in events:
        if ev["kind"] == "image_generate_begin":
            eng = (ev.get("payload") or {}).get("engine")
            if eng and eng not in engine_anchors:
                engine_anchors[eng] = ev["ts"]

    engine_entries: list[Entry] = []
    for engine, path in discover_engine_logs(logs_dir):
        engine_entries.extend(_image_engine_entries(
            engine, path, engine_anchors.get(engine),
        ))

    merged = sorted(db_entries + llm_entries + engine_entries, key=lambda e: e.ts)
    return merged[-tail:]


def render_activity(entries: list[Entry]) -> str:
    """Render entries as a plain-text block suitable for the existing <pre>."""
    return "\n".join(e.fmt() for e in entries)
