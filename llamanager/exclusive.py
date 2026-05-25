"""Exclusive-mode process sweeper.

When enabled, llamanager claims the host's ML accelerators: any
``llama-server`` (or known image-engine worker, or in aggressive mode any
foreign ML runtime like ComfyUI / A1111 / vLLM / Ollama) that is not in
our own process tree is SIGTERM-ed (then SIGKILL-ed after a grace
window). This frees VRAM/RAM for the model llamanager is about to load.

Modes (config ``[server].exclusive_mode``):

- ``off``        — disabled. ``sweep()`` is a no-op.
- ``warn``       — scan + log what would be killed, kill nothing.
- ``exclusive``  — kill llama-server binaries and our own orphaned
                   image-engine worker python procs.
- ``aggressive`` — also kill foreign ML runtimes (ComfyUI, sd.webui,
                   kohya_ss, text-generation-webui, vLLM, Ollama).

Trigger points wired by callers:

- ``ServerManager.start()`` pre-spawn (text engine claims the slot)
- ``ServerManager.yield_to_image()`` pre-yield (image worker claims it)
- ``ImageTaskRunner._run_one()`` pre-spawn (defence in depth)
- Daemon startup, once
- A 120 s heartbeat (configurable) running for the lifetime of the app
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VALID_MODES = ("off", "warn", "exclusive", "aggressive")

# Process-name allowlist (basename match on /proc/<pid>/exe).
NARROW_EXE_NAMES: frozenset[str] = frozenset({
    "llama-server", "llama-cli", "mtmd-cli", "sd-cli",
})

# Substrings on the python cmdline that mark a process as one of our
# image-engine workers. A worker that we spawned ourselves will be inside
# our process tree (caught by the pgid/tree check) and therefore spared;
# the cmdline match only fires for orphans of a previous llamanager
# crash, which is exactly what we want to clean up.
NARROW_CMDLINE_MARKERS: tuple[str, ...] = (
    "llamanager.engines._z_image_runner",
    "llamanager/engines/_z_image_runner",
    "inference.py",  # HiDream-O1-Image
    "hidream_inference",
    "z_image_runner",
)

# Aggressive-mode additions: substrings (case-insensitive) on the cmdline
# that mark a foreign ML runtime. Kept narrow on purpose — generic
# matches like "python" or "torch" would catch unrelated user work.
AGGRESSIVE_CMDLINE_MARKERS: tuple[str, ...] = (
    "comfyui",
    "comfy.cli",
    "stable-diffusion-webui",
    "webui.py",
    "sd-webui",
    "kohya_ss",
    "kohya-ss",
    "text-generation-webui",
    "ooba",
    "vllm.entrypoints",
    "vllm serve",
    "ollama serve",
    "lmstudio",
    "lm-studio",
    "koboldcpp",
    "tabbyapi",
)


@dataclass
class Victim:
    pid: int
    kind: str       # "exe" | "engine-orphan" | "foreign-ml"
    exe: str
    cmdline: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "kind": self.kind,
            "exe": self.exe,
            "cmdline": self.cmdline[:240],
        }


@dataclass
class SweepResult:
    mode: str
    at: float = field(default_factory=time.time)
    scanned: int = 0
    victims: list[Victim] = field(default_factory=list)
    terminated: list[int] = field(default_factory=list)
    killed: list[int] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "at": self.at,
            "scanned": self.scanned,
            "victims": [v.to_dict() for v in self.victims],
            "terminated": list(self.terminated),
            "killed": list(self.killed),
            "error": self.error,
        }


def _iter_pids() -> list[int]:
    try:
        return sorted(int(e) for e in os.listdir("/proc") if e.isdigit())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def _read_proc(pid: int) -> tuple[str, str, int] | None:
    """Return (exe_basename, cmdline_str, ppid) for ``pid`` or None."""
    try:
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
        except (PermissionError, FileNotFoundError):
            exe = ""
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, PermissionError):
            raw = b""
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        ppid = 0
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError):
            return None
        return exe, cmdline, ppid
    except ProcessLookupError:
        return None


def _collect_tree(root_pid: int) -> set[int]:
    """Walk ``/proc`` building the set of all descendants of ``root_pid``.

    Two passes catch grandchildren that show up later in the listing.
    """
    tree = {root_pid}
    snapshots: list[tuple[int, int]] = []  # (pid, ppid)
    for pid in _iter_pids():
        r = _read_proc(pid)
        if r is None:
            continue
        snapshots.append((pid, r[2]))
    changed = True
    while changed:
        changed = False
        for pid, ppid in snapshots:
            if pid not in tree and ppid in tree:
                tree.add(pid)
                changed = True
    return tree


def _classify(exe: str, cmdline: str, *, aggressive: bool) -> str | None:
    exe_name = os.path.basename(exe) if exe else ""
    if exe_name in NARROW_EXE_NAMES:
        return "exe"
    # cmdline lowercase for marker matching
    cl = cmdline.lower()
    for marker in NARROW_CMDLINE_MARKERS:
        if marker.lower() in cl:
            return "engine-orphan"
    if aggressive:
        for marker in AGGRESSIVE_CMDLINE_MARKERS:
            if marker.lower() in cl:
                return "foreign-ml"
    return None


def scan(mode: str) -> SweepResult:
    """Synchronous scan of /proc; no signals sent. Used by ``sweep`` and
    by callers who only want to know what's out there (UI status)."""
    mode = (mode or "off").strip().lower()
    if mode not in VALID_MODES:
        mode = "off"
    result = SweepResult(mode=mode)
    if mode == "off":
        return result
    aggressive = (mode == "aggressive")
    own_pid = os.getpid()
    own_tree = _collect_tree(own_pid)
    for pid in _iter_pids():
        if pid == own_pid or pid in own_tree:
            continue
        r = _read_proc(pid)
        if r is None:
            continue
        exe, cmdline, _ppid = r
        result.scanned += 1
        kind = _classify(exe, cmdline, aggressive=aggressive)
        if kind:
            result.victims.append(Victim(pid=pid, kind=kind, exe=exe, cmdline=cmdline))
    return result


async def sweep(mode: str, grace_seconds: float = 5.0) -> SweepResult:
    """Scan and (unless mode=='off' or 'warn') terminate matched processes.

    SIGTERM → ``grace_seconds`` → SIGKILL stragglers. Returns the full
    result so callers can log or expose it via the admin/UI layer.
    """
    result = scan(mode)
    if result.mode in ("off", "warn"):
        if result.mode == "warn" and result.victims:
            for v in result.victims:
                log.warning(
                    "exclusive (warn): would kill pid=%d kind=%s exe=%s cmd=%s",
                    v.pid, v.kind, v.exe, v.cmdline[:160],
                )
        return result
    if not result.victims:
        return result

    for v in result.victims:
        try:
            os.kill(v.pid, signal.SIGTERM)
            result.terminated.append(v.pid)
            log.warning(
                "exclusive (%s): SIGTERM pid=%d kind=%s exe=%s",
                result.mode, v.pid, v.kind, v.exe,
            )
        except ProcessLookupError:
            pass
        except PermissionError as e:
            log.warning(
                "exclusive: cannot signal pid=%d (%s) — different user?",
                v.pid, e,
            )

    if result.terminated:
        await asyncio.sleep(max(0.0, grace_seconds))

    for pid in result.terminated:
        try:
            os.kill(pid, 0)  # liveness probe
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            result.killed.append(pid)
            log.warning("exclusive: SIGKILL pid=%d (grace expired)", pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    return result


# ---------------------------------------------------------------------------
# Singleton state — last sweep result, exposed to the UI/admin layer.
# ---------------------------------------------------------------------------

_LAST: SweepResult | None = None


def last_result() -> SweepResult | None:
    return _LAST


async def sweep_and_record(mode: str, grace_seconds: float = 5.0) -> SweepResult:
    """Run ``sweep`` and stash the result for later inspection."""
    global _LAST
    try:
        _LAST = await sweep(mode, grace_seconds=grace_seconds)
    except Exception as e:  # noqa: BLE001 — last-ditch guard
        log.exception("exclusive: sweep failed")
        _LAST = SweepResult(mode=mode, error=str(e))
    return _LAST


def scan_and_record(mode: str) -> SweepResult:
    """Run a sync scan and stash it as the last result. Used when the
    operator hits "Sweep now" with mode=off — they get a warn-style
    preview of what *would* be killed without anything being touched.
    """
    global _LAST
    _LAST = scan(mode)
    return _LAST


async def heartbeat(get_mode, get_grace, get_interval, *,
                    stop_event: asyncio.Event | None = None) -> None:
    """Background task: re-run ``sweep_and_record`` every interval seconds.

    The mode/grace/interval are read fresh on each tick via callables so
    config hot-reload takes effect without restarting the task.
    """
    while True:
        interval = max(30, int(get_interval() or 120))
        try:
            if stop_event is not None:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return  # stop_event fired
            else:
                await asyncio.sleep(interval)
        except asyncio.TimeoutError:
            pass  # interval elapsed without stop
        except asyncio.CancelledError:
            return
        try:
            mode = get_mode() or "off"
            if mode == "off":
                continue
            grace = float(get_grace() or 5.0)
            await sweep_and_record(mode, grace_seconds=grace)
        except Exception:  # noqa: BLE001 — heartbeat must never die
            log.exception("exclusive: heartbeat tick failed")
