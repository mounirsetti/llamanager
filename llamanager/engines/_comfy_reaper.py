"""Shuts down a warm ComfyUI server once it has been idle.

A warm server exists so the next request skips rebuilding the text encoder
(measured at 719 of 740 seconds for Krea 2 Turbo). It holds VRAM for as long
as it lives, and the request that started it exits immediately, so something
detached has to end it — that is this.

Run as::

    _comfy_reaper.py --pid PID --beat PATH --idle SECONDS --state PATH

It watches a heartbeat file that every request touches, and kills the
server's process GROUP once the file has gone stale. Group, not process:
ComfyUI spawns helpers, and signalling only the parent leaves grandchildren
holding the GPU.

Kept as a file rather than an inline string so it can be read and tested.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

POLL_SECONDS = 5.0
# How long to let the group exit on SIGTERM before escalating. SIGKILL of a
# process holding a KFD context has leaked GPU memory on this hardware, so it
# gets a real chance to shut down cleanly first.
TERM_GRACE_SECONDS = 30.0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def idle_for(beat: Path, now: float) -> float | None:
    """Seconds since the last request, or None if that cannot be determined.

    A missing heartbeat returns None rather than infinity: treating "no file"
    as "infinitely idle" kills a server that has only just been recorded,
    which is exactly the race that broke the first version of this.
    """
    try:
        return now - beat.stat().st_mtime
    except OSError:
        return None


def owns(state: Path, pid: int) -> str:
    """Whether ``state`` still names ``pid`` as the warm server.

    Returns "mine", "superseded" (the record now names a different process —
    ours has been replaced and nothing else will ever stop it), or "gone"
    (no readable record — ours is untracked, so this reaper is the only
    thing left that knows it exists).

    Both non-"mine" answers were reachable in production and neither was
    handled: the heartbeat file is shared per model, so a replacement
    server's traffic kept an orphan's reaper believing its own server was
    busy, forever.
    """
    try:
        recorded = int(json.loads(state.read_text())["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return "gone"
    return "mine" if recorded == pid else "superseded"


def reap(pid: int, beat: Path, idle: float, state: Path,
         poll: float = POLL_SECONDS) -> str:
    """Block until the server is gone. Returns why it stopped watching."""
    reason = "server exited on its own"
    # When the record disappears, idleness can no longer be read from the
    # shared heartbeat — it may be another server's now, or absent. Fall back
    # to the last moment this reaper KNEW its server was wanted.
    last_known_wanted = time.time()
    while True:
        time.sleep(poll)
        if not alive(pid):
            break
        held = owns(state, pid)
        if held == "mine":
            quiet = idle_for(beat, time.time())
            if quiet is None:
                # Recorded, but the heartbeat is missing — measure from the
                # last moment we could confirm the server was wanted.
                quiet = time.time() - last_known_wanted
            else:
                last_known_wanted = time.time() - quiet   # the beat's mtime
            if quiet < idle:
                continue
            why = f"idle for {quiet:.0f}s"
        elif held == "superseded":
            # The record names someone else. Our server can never be adopted
            # again and no other reaper is watching it: it is an orphan the
            # moment we notice, and waiting out an idle window only means
            # more of the machine spent on work nobody will collect.
            why = "superseded by a newer server for the same model"
        else:
            # Untracked. No request can reach it, so its heartbeat will never
            # be touched again — but the beat file may still be updated by a
            # DIFFERENT server for this model, so measure from our own clock.
            quiet = time.time() - last_known_wanted
            if quiet < idle:
                continue
            why = f"untracked and idle for {quiet:.0f}s"
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            break
        deadline = time.time() + TERM_GRACE_SECONDS
        while time.time() < deadline and alive(pid):
            time.sleep(1.0)
        if alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                pass
        reason = why
        break
    # Only our own record. Deleting unconditionally used to erase the state
    # and heartbeat of whichever server had replaced us, untracking IT.
    if owns(state, pid) in ("mine", "gone"):
        for p in (beat, state):
            try:
                p.unlink()
            except OSError:
                pass
    return reason


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--beat", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--idle", type=float, required=True)
    args = ap.parse_args()
    reap(args.pid, args.beat, args.idle, args.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
