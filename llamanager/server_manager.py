"""llama-server subprocess lifecycle.

Spec §4.3 + §6.4. Owns one child process at a time. Start/stop/restart/
swap. Crash supervisor lives in supervisor.py and listens for unexpected
exits via the shared `wait_event`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import runtime_state as rt
from .config import Config, Profile
from .db import DB

log = logging.getLogger(__name__)

START_TIMEOUT_S = 60.0
STOP_GRACE_S = 10.0
HEALTH_POLL_INTERVAL_S = 0.5


# Args that don't translate cleanly to mlx_lm.server. mlx-lm has a much
# narrower flag surface than llama-server; we pass through what it accepts
# and silently drop the rest (they're llama.cpp-specific).
_MLX_SUPPORTED_ARGS = {
    "model", "host", "port", "trust-remote-code", "log-level",
    "chat-template", "use-default-chat-template",
    "temp", "top-p", "top-k", "min-p",
    "max-tokens",  # default; per-request override comes from the API
}


@dataclass
class StartSpec:
    """Resolved set of args to launch the inference engine with.

    ``cmdline`` adapts to either llama-server (the llama.cpp CLI) or
    ``mlx_lm.server`` based on the *engine* argument.
    """
    model_path: Path
    mmproj_path: Path | None
    extra_args: dict[str, Any]
    profile_name: str | None
    model_id: str  # the requested logical id (bare model path relative to models_dir)

    def cmdline(self, binary: str, port: int, engine: str = "llama") -> list[str]:
        if engine == "mlx":
            return self._mlx_cmdline(binary, port)
        return self._llama_cmdline(binary, port)

    def _llama_cmdline(self, binary: str, port: int) -> list[str]:
        cmd = [binary, "-m", str(self.model_path)]
        if self.mmproj_path:
            cmd += ["--mmproj", str(self.mmproj_path)]
        cmd += ["--host", "127.0.0.1", "--port", str(port)]
        for k, v in self.extra_args.items():
            if k in ("host", "port"):
                continue
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]
        return cmd

    def _mlx_cmdline(self, python: str, port: int) -> list[str]:
        # binary is the venv Python; mlx-lm is invoked as a module.
        cmd = [python, "-m", "mlx_lm", "server",
               "--model", str(self.model_path),
               "--host", "127.0.0.1", "--port", str(port)]
        for k, v in self.extra_args.items():
            if k in ("host", "port", "mmproj"):
                continue
            if k not in _MLX_SUPPORTED_ARGS:
                log.debug("dropping arg %r — not supported by mlx-lm", k)
                continue
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]
        return cmd


def _safe_under(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and require it to live under `base`.

    Rejects absolute escapes, .., symlinks pointing outside, and any other
    traversal trick. Returns the resolved (absolute) path on success.
    """
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"path escapes models_dir: {candidate} -> {resolved}"
        )
    return resolved


def _validate_model_id(model_id: str) -> None:
    """Reject obviously unsafe model identifiers before they hit the FS."""
    if not model_id or not isinstance(model_id, str):
        raise ValueError("model id is empty")
    if "\x00" in model_id:
        raise ValueError("model id contains NUL byte")
    # Backslashes are not legitimate in model ids (we always use forward
    # slashes); on POSIX they would survive Path() unsplit, so block here.
    if "\\" in model_id:
        raise ValueError(f"model id may not contain backslashes: {model_id!r}")
    p = Path(model_id)
    if p.is_absolute() or p.drive or p.root:
        raise ValueError(f"model id must be relative: {model_id!r}")
    parts = p.parts
    if any(seg in ("", "..", ".") for seg in parts):
        raise ValueError(f"model id contains forbidden segment: {model_id!r}")


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
        return True
    except OSError:
        return False


def resolve_default(cfg: Config) -> tuple[str, str | None]:
    """Return the (model_id, profile_name) pair used when a request specifies
    neither. Profile is None when no model-default and no global-default exist.

    Returns ("", None) when no default_model is configured — callers should
    treat that as "nothing to start" rather than crashing.
    """
    model = cfg.default_model
    if not model:
        return "", None
    profile = cfg.model_default_profiles.get(model) or cfg.default_profile or None
    return model, profile


def resolve_spec(cfg: Config, *, profile: str | None = None,
                 model: str | None = None,
                 mmproj: str | None = None,
                 args: dict[str, Any] | None = None) -> StartSpec:
    """Turn a request into a concrete StartSpec under the three-scenario rule:

    1. Nothing specified           → cfg.default_model + its default profile
    2. Only model specified        → that model + its default profile
    3. Both specified              → that exact pair (profile must be bound to
                                     that model, or be the global default)

    Profile-without-model is rejected.

    The ``profile:foo`` shorthand on ``model`` is no longer accepted; callers
    that used it must migrate to passing ``model`` and ``profile`` separately.
    """
    args = dict(args or {})
    model = (model or "").strip() or None
    profile = (profile or "").strip() or None

    if profile and not model:
        raise ValueError(
            "profile requires a model: pass model + profile, not profile alone"
        )

    if not model:
        # Scenarios 1/2: fall through to the default. Note: scenario 2 (model
        # only) is handled below — we only reach this branch when *neither*
        # was given, so use cfg.default_model.
        model, default_prof = resolve_default(cfg)
        if not model:
            raise ValueError("no model specified and no default configured")
        if profile is None:
            profile = default_prof
    elif profile is None:
        # Scenario 2: pick the model's default profile, then the global one.
        profile = cfg.model_default_profiles.get(model) or cfg.default_profile or None

    prof: Profile | None = None
    if profile:
        prof = cfg.profiles.get(profile)
        if not prof:
            raise ValueError(f"unknown profile: {profile}")
        if prof.model and prof.model != model:
            raise ValueError(
                f"profile {profile!r} is bound to model {prof.model!r}, "
                f"not {model!r}"
            )

    chosen_mmproj = mmproj if mmproj is not None else (prof.mmproj if prof else "")
    merged_args = dict(prof.args) if prof else {}
    merged_args.update(args)

    _validate_model_id(model)
    model_path = _safe_under(cfg.models_dir, cfg.models_dir / model)
    mmproj_path: Path | None
    if chosen_mmproj:
        _validate_model_id(chosen_mmproj)
        mmproj_path = _safe_under(cfg.models_dir, cfg.models_dir / chosen_mmproj)
    else:
        mmproj_path = None

    return StartSpec(
        model_path=model_path,
        mmproj_path=mmproj_path,
        extra_args=merged_args,
        profile_name=profile,
        model_id=model,
    )


class ServerError(Exception):
    pass


class ServerManager:
    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self.proc: asyncio.subprocess.Process | None = None
        self.spec: StartSpec | None = None
        self.runtime: rt.RuntimeState = rt.load(cfg.runtime_path)
        self._wait_task: asyncio.Task[int] | None = None
        self._lock = asyncio.Lock()
        self._log_fp = None  # type: ignore[assignment]
        self._intentional_stop = False
        # subscribers to unexpected-exit events
        self._exit_listeners: list[asyncio.Queue[int]] = []

    # ---- lifecycle hooks ----
    def add_exit_listener(self, q: asyncio.Queue[int]) -> None:
        self._exit_listeners.append(q)

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def upstream_base(self) -> str:
        return f"http://127.0.0.1:{self.cfg.llama_server_port}"

    # ---- start ----
    async def start(self, spec: StartSpec) -> int:
        async with self._lock:
            if self.is_running:
                raise ServerError("inference engine already running")
            engine = getattr(self.cfg, "llama_server_engine", "llama") or "llama"
            if not spec.model_path.exists():
                raise ServerError(f"model not found: {spec.model_path}")
            if spec.mmproj_path and not spec.mmproj_path.exists():
                raise ServerError(f"mmproj file not found: {spec.mmproj_path}")
            port = self.cfg.llama_server_port
            if not _port_free(port):
                raise ServerError(f"port {port} is busy on 127.0.0.1")

            cmd = spec.cmdline(self.cfg.llama_server_binary, port, engine=engine)
            log_path = self.cfg.logs_dir / "llama-server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = open(log_path, "ab", buffering=0)
            log.info("launching %s engine: %s", engine, shlex.join(cmd))
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=self._log_fp,
                    stderr=self._log_fp,
                    stdin=asyncio.subprocess.DEVNULL,
                )
            except FileNotFoundError as e:
                self._close_log()
                raise ServerError(
                    f"engine binary not found: {self.cfg.llama_server_binary}"
                ) from e

            self.spec = spec
            self._intentional_stop = False
            self.runtime = rt.RuntimeState(
                state="starting", pid=self.proc.pid,
                current_model=spec.model_id,
                current_profile=spec.profile_name,
                current_args=spec.extra_args,
                started_at=time.time(),
                last_event_at=time.time(),
            )
            rt.save(self.cfg.runtime_path, self.runtime)
            self.db.log_event("server_starting", {"pid": self.proc.pid,
                                                  "cmd": cmd,
                                                  "model": spec.model_id,
                                                  "profile": spec.profile_name})
            self._wait_task = asyncio.create_task(self._watch_proc())

        # health-poll outside the lock so /admin/status can answer
        ok = await self._await_ready()
        if not ok:
            await self.stop()
            raise ServerError("llama-server failed to become healthy in time")

        self.runtime.state = "running"
        self.runtime.last_event_at = time.time()
        rt.save(self.cfg.runtime_path, self.runtime)
        self.db.log_event("server_running", {"pid": self.proc.pid if self.proc else None,
                                             "model": spec.model_id})
        return self.proc.pid if self.proc else -1

    async def _await_ready(self) -> bool:
        url = f"{self.upstream_base}/health"
        deadline = time.time() + START_TIMEOUT_S
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if not self.is_running:
                    return False
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
        return False

    # ---- stop ----
    async def stop(self) -> None:
        async with self._lock:
            self._intentional_stop = True
            await self._terminate()
            self.runtime.state = "stopped"
            self.runtime.pid = None
            self.runtime.current_model = None
            self.runtime.current_profile = None
            self.runtime.last_event_at = time.time()
            rt.save(self.cfg.runtime_path, self.runtime)
            self.db.log_event("server_stopped", {})

    async def _terminate(self) -> None:
        if not self.proc:
            return
        if self.proc.returncode is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=STOP_GRACE_S)
            except asyncio.TimeoutError:
                log.warning("llama-server did not exit on SIGTERM, escalating to SIGKILL")
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    log.error("llama-server still alive after SIGKILL — leaking")
        self._close_log()
        self.proc = None
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
        self._wait_task = None

    def _close_log(self) -> None:
        try:
            if self._log_fp:
                self._log_fp.close()
        except Exception:
            pass
        self._log_fp = None

    # ---- restart / swap ----
    async def restart(self, spec: StartSpec | None = None) -> int:
        target = spec or self.spec
        if not target:
            raise ServerError("no previous spec to restart with")
        await self.stop()
        return await self.start(target)

    async def swap(self, new_spec: StartSpec) -> int:
        """Stop current, start new. On failure roll back to previous spec."""
        old_spec = self.spec
        self.runtime.state = "swapping"
        self.runtime.last_event_at = time.time()
        rt.save(self.cfg.runtime_path, self.runtime)
        self.db.log_event("server_swap_begin",
                          {"from": old_spec.model_id if old_spec else None,
                           "to": new_spec.model_id})
        await self.stop()
        try:
            pid = await self.start(new_spec)
            self.db.log_event("server_swap_ok",
                              {"from": old_spec.model_id if old_spec else None,
                               "to": new_spec.model_id})
            return pid
        except Exception as e:
            self.db.log_event("server_swap_fail",
                              {"to": new_spec.model_id, "error": str(e)})
            if old_spec:
                try:
                    return await self.start(old_spec)
                except Exception as e2:
                    self.runtime.state = "degraded"
                    self.runtime.last_event_at = time.time()
                    rt.save(self.cfg.runtime_path, self.runtime)
                    self.db.log_event("server_degraded",
                                      {"reason": "rollback_failed", "error": str(e2)})
                    raise ServerError(
                        f"swap failed and rollback also failed: {e2}"
                    ) from e2
            raise

    # ---- crash detection ----
    async def _watch_proc(self) -> int:
        proc = self.proc
        assert proc
        rc = await proc.wait()
        # If we initiated the stop, this is benign.
        if self._intentional_stop:
            return rc
        log.warning("llama-server exited unexpectedly with code %s", rc)
        self.db.log_event("server_crashed", {"returncode": rc})
        self.runtime.state = "crashed"
        self.runtime.pid = None
        self.runtime.last_event_at = time.time()
        rt.save(self.cfg.runtime_path, self.runtime)
        for q in list(self._exit_listeners):
            try:
                q.put_nowait(rc)
            except asyncio.QueueFull:
                pass
        return rc

    # ---- diagnostics ----
    async def health(self) -> bool:
        if not self.is_running:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.upstream_base}/health")
                return r.status_code == 200
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        try:
            import psutil
            mem_rss_mb: float | None = None
            if self.proc and self.proc.pid:
                try:
                    mem_rss_mb = psutil.Process(self.proc.pid).memory_info().rss / (1024 * 1024)
                except Exception:
                    mem_rss_mb = None
        except ImportError:
            mem_rss_mb = None
        uptime = (time.time() - self.runtime.started_at) if self.runtime.started_at else None
        return {
            "state": self.runtime.state,
            "current_model": self.runtime.current_model,
            "current_profile": self.runtime.current_profile,
            "pid": self.runtime.pid,
            "started_at": self.runtime.started_at,
            "uptime_s": uptime,
            "mem_rss_mb": mem_rss_mb,
        }

    # ---- supervisor coordination ----
    def mark_degraded(self, reason: str) -> None:
        self.runtime.state = "crashed"
        self.runtime.last_event_at = time.time()
        rt.save(self.cfg.runtime_path, self.runtime)
        self.db.log_event("server_supervisor_giveup", {"reason": reason})
