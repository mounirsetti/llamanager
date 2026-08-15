"""One-shot ComfyUI driver — invoked by the comfy-family adapters.

Runs inside the ``comfy`` venv (``cfg.comfyui_python``). One invocation ==
one generation: it starts a private headless ComfyUI on a free loopback port,
submits a single frozen workflow, copies the result to ``--output``, and shuts
the server down. Nothing stays resident in VRAM between requests, which is the
same contract every other image/video engine in llamanager honours.

CHILD-PROCESS SAFETY. ComfyUI is started in its own session
(``start_new_session=True``) and this script installs SIGTERM/SIGINT handlers
that ``killpg`` that session. That is not incidental: ``image_runner.py``
terminates its child with a plain ``terminate()``, so a SIGTERM arriving here
would otherwise leave an orphaned ComfyUI holding tens of GB of VRAM. This
project has already had one production incident from exactly that shape (a
whisper shim that did not reap its native child leaked 56 orphans and OOMed
the machine), so the server is reaped on every exit path, including crashes.

PROGRESS. ComfyUI reports sampling progress over its websocket. We translate
it to ``N/M`` lines on stderr, which is the format every llamanager adapter's
``parse_progress`` already understands — no new progress channel.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# The server has to load and JIT a lot before it answers; on a cold page cache
# this has been observed to take over a minute. Generation itself is bounded
# by --timeout instead.
STARTUP_TIMEOUT_S = 300


def log(msg: str) -> None:
    print(f"[comfy] {msg}", file=sys.stderr, flush=True)


def _free_port() -> int:
    """Ask the OS for an unused loopback port.

    Binding to port 0 and reading it back leaves a small race between close
    and ComfyUI's own bind, but it is far safer than a fixed port: a fixed
    port would collide with an operator's own ComfyUI and silently submit this
    workflow to *their* server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ComfyServer:
    """A private ComfyUI process, reaped on every exit path."""

    def __init__(self, python: Path, repo: Path, port: int,
                 output_dir: Path, input_dir: Path, temp_dir: Path,
                 extra_paths: Path, log_file: Path,
                 extra_args: list[str] | None = None):
        self.python, self.repo, self.port = python, repo, port
        self.output_dir, self.input_dir, self.temp_dir = (
            output_dir, input_dir, temp_dir)
        self.extra_paths, self.log_file = extra_paths, log_file
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        argv = [
            str(self.python), "main.py",
            "--listen", "127.0.0.1", "--port", str(self.port),
            "--disable-auto-launch",
            "--output-directory", str(self.output_dir),
            "--input-directory", str(self.input_dir),
            "--temp-directory", str(self.temp_dir),
            "--extra-model-paths-config", str(self.extra_paths),
            # Latent previews would decode every few steps for a UI nobody is
            # watching; on a video model that is a large, pure waste.
            "--preview-method", "none",
            *self.extra_args,
        ]
        log(f"starting server on port {self.port}; log -> {self.log_file}")
        # ComfyUI's own logging is verbose and continuous. It goes to a file,
        # never to our stderr, so it cannot flood the caller's log capture.
        self._log_fh = open(self.log_file, "wb")
        self.proc = subprocess.Popen(
            argv, cwd=str(self.repo),
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float = STARTUP_TIMEOUT_S) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited with code {self.proc.returncode} before "
                    f"becoming ready. Last lines:\n{self.tail_log()}")
            try:
                with urllib.request.urlopen(
                        f"{self.base}/system_stats", timeout=5) as r:
                    stats = json.loads(r.read())
                devs = stats.get("devices") or [{}]
                d = devs[0]
                log(f"ready: device={d.get('name')} "
                    f"type={d.get('type')} "
                    f"vram_total={(d.get('vram_total') or 0) / 2**30:.1f} GiB")
                return
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(1.0)
        raise TimeoutError(
            f"ComfyUI did not become ready within {timeout:.0f}s. "
            f"Last lines:\n{self.tail_log()}")

    def tail_log(self, limit: int = 4000) -> str:
        try:
            data = self.log_file.read_bytes()
        except OSError:
            return "(no log)"
        return data[-limit:].decode("utf-8", errors="replace")

    def stop(self) -> None:
        """Terminate the whole process group, then verify it is gone.

        ComfyUI spawns helpers, so signalling only the direct child can leave
        grandchildren holding VRAM. SIGKILL of a process holding a KFD context
        has previously leaked GPU memory on this hardware, so the group gets a
        real chance to exit on SIGTERM first.
        """
        if self.proc is None or self.proc.poll() is not None:
            self._close_log()
            return
        pgid = os.getpgid(self.proc.pid)
        log("stopping server")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._close_log()
            return
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log("server ignored SIGTERM; sending SIGKILL to the group")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log("WARNING: server process did not reap")
        self._close_log()

    def _close_log(self) -> None:
        fh = getattr(self, "_log_fh", None)
        if fh is not None and not fh.closed:
            fh.close()

    # ---- HTTP helpers -------------------------------------------------

    def post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ComfyUI rejected {path} ({e.code}): {detail[:4000]}") from None

    def get_json(self, path: str, timeout: float = 30) -> dict:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=timeout) as r:
            return json.loads(r.read())

    def upload_image(self, path: Path) -> str:
        """POST an image to /upload/image; returns the server-side name."""
        boundary = uuid.uuid4().hex
        data = path.read_bytes()
        parts = [
            f"--{boundary}\r\n".encode(),
            (f'Content-Disposition: form-data; name="image"; '
             f'filename="{path.name}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode(),
            data, b"\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode(),
        ]
        req = urllib.request.Request(
            f"{self.base}/upload/image", data=b"".join(parts),
            headers={"Content-Type":
                     f"multipart/form-data; boundary={boundary}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            info = json.loads(r.read())
        name = info["name"]
        if info.get("subfolder"):
            name = f"{info['subfolder']}/{name}"
        log(f"uploaded init image as {name}")
        return name


def run_prompt(server: ComfyServer, graph: dict, client_id: str,
               timeout: float) -> dict:
    """Submit the graph and follow it to completion. Returns its history entry.

    Progress comes over the websocket when ``websocket-client`` is available
    and falls back to polling ``/history`` otherwise, so a missing optional
    dependency costs progress reporting rather than the whole generation.
    """
    resp = server.post_json("/prompt", {"prompt": graph, "client_id": client_id})
    prompt_id = resp["prompt_id"]
    log(f"queued prompt {prompt_id}")

    try:
        import websocket  # type: ignore
    except ImportError:
        websocket = None
        log("websocket-client not installed; falling back to history polling")

    deadline = time.time() + timeout
    if websocket is not None:
        ws = websocket.WebSocket()
        ws.connect(f"ws://127.0.0.1:{server.port}/ws?clientId={client_id}",
                   timeout=30)
        ws.settimeout(30)
        try:
            while time.time() < deadline:
                try:
                    raw = ws.recv()
                except Exception:  # timeout or transient close
                    if _history_done(server, prompt_id):
                        break
                    continue
                if not isinstance(raw, str):
                    continue  # binary frames are latent previews
                msg = json.loads(raw)
                mtype, data = msg.get("type"), msg.get("data") or {}
                if mtype == "progress":
                    # The format every adapter's parse_progress already reads.
                    print(f"{data.get('value')}/{data.get('max')}",
                          file=sys.stderr, flush=True)
                elif mtype == "execution_error":
                    raise RuntimeError(
                        f"node {data.get('node_type')} failed: "
                        f"{data.get('exception_message')}\n"
                        f"{''.join(data.get('traceback') or [])[:4000]}")
                elif mtype == "executing" and data.get("node") is None \
                        and data.get("prompt_id") == prompt_id:
                    break
                elif mtype == "status":
                    pass
        finally:
            try:
                ws.close()
            except Exception:
                pass
    else:
        while time.time() < deadline:
            if _history_done(server, prompt_id):
                break
            time.sleep(2.0)

    hist = server.get_json(f"/history/{prompt_id}", timeout=60).get(prompt_id)
    if not hist:
        raise TimeoutError(
            f"prompt {prompt_id} did not finish within {timeout:.0f}s")
    status = hist.get("status") or {}
    if status.get("status_str") == "error":
        raise RuntimeError(
            f"workflow failed: {json.dumps(status)[:4000]}\n"
            f"server log:\n{server.tail_log()}")
    return hist


def _history_done(server: ComfyServer, prompt_id: str) -> bool:
    try:
        hist = server.get_json(f"/history/{prompt_id}", timeout=15)
    except Exception:
        return False
    entry = hist.get(prompt_id)
    return bool(entry and (entry.get("status") or {}).get("completed"))


def collect_output(hist: dict, output_dir: Path, dest: Path) -> Path:
    """Copy the workflow's single produced file to ``dest``.

    ComfyUI groups outputs by node and by kind ('images', 'gifs', 'video',
    'audio'). We take the newest real file rather than assuming a key, so the
    same runner serves both the image and the video adapters.
    """
    candidates: list[Path] = []
    for node_out in (hist.get("outputs") or {}).values():
        for items in node_out.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                if item.get("type") == "temp":
                    continue
                p = output_dir / (item.get("subfolder") or "") / item["filename"]
                if p.is_file():
                    candidates.append(p)
    if not candidates:
        raise RuntimeError(
            f"workflow produced no output files. history keys: "
            f"{sorted((hist.get('outputs') or {}).keys())}")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, dest)
    log(f"wrote {dest} ({dest.stat().st_size / 2**20:.1f} MiB)")
    return dest


def parse_set(pairs: list[str]) -> dict[str, object]:
    """Turn ``--set key=value`` pairs into typed workflow token values.

    Values are JSON-decoded when possible so numbers arrive as numbers, with a
    plain-string fallback for the common case of unquoted text (prompts,
    filenames) that is not valid JSON.
    """
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _load_backend():
    """Import ``comfy_backend`` by file path, not as ``llamanager.engines.*``.

    This script runs inside the ComfyUI venv, which has torch and ComfyUI's
    stack but not llamanager's — importing through the package would pull in
    fastapi and the rest of the daemon's dependencies and fail. The backend
    module itself is pure stdlib, so loading just that file is enough.
    """
    import importlib.util
    path = Path(__file__).with_name("comfy_backend.py")
    spec = importlib.util.spec_from_file_location("comfy_backend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-repo", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path,
                   help="model directory holding diffusion_models/, vae/, ...")
    p.add_argument("--workflow", required=True, type=Path,
                   help="frozen API-format workflow template")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--set", action="append", default=[], dest="sets",
                   help="workflow token, e.g. --set WIDTH=1344")
    p.add_argument("--init-image", type=Path, default=None,
                   help="uploaded and bound to the INIT_IMAGE token")
    p.add_argument("--bypass", action="append", default=[],
                   metavar="NODE_ID:INPUT",
                   help="drop a node, rewiring consumers to that input")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--comfy-arg", action="append", default=[],
                   help="extra flag passed through to ComfyUI's main.py")
    p.add_argument("--keep-server-log", action="store_true",
                   help="keep the ComfyUI log file instead of the temp dir")
    args = p.parse_args()

    cb = _load_backend()

    if not args.model_path.is_dir():
        log(f"model path does not exist: {args.model_path}")
        return 2
    if not args.workflow.is_file():
        log(f"workflow template not found: {args.workflow}")
        return 2

    values = parse_set(args.sets)
    work = Path(tempfile.mkdtemp(prefix="llamanager-comfy-"))
    server: ComfyServer | None = None

    def _terminate(signum, _frame):
        # Reap the child before dying, or it keeps the GPU.
        log(f"received signal {signum}; shutting down")
        if server is not None:
            server.stop()
        shutil.rmtree(work, ignore_errors=True)
        os._exit(143)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)

    try:
        out_dir, in_dir, tmp_dir = work / "out", work / "in", work / "tmp"
        for d in (out_dir, in_dir, tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

        paths_yaml = work / "extra_model_paths.yaml"
        paths_yaml.write_text(cb.extra_model_paths_yaml(args.model_path))

        server = ComfyServer(
            python=Path(sys.executable), repo=args.comfy_repo,
            port=_free_port(), output_dir=out_dir, input_dir=in_dir,
            temp_dir=tmp_dir, extra_paths=paths_yaml,
            log_file=work / "comfyui.log", extra_args=args.comfy_arg)
        server.start()
        server.wait_ready()

        if args.init_image is not None:
            if not args.init_image.is_file():
                log(f"init image not found: {args.init_image}")
                return 2
            values["INIT_IMAGE"] = server.upload_image(args.init_image)

        graph = cb.render_workflow(args.workflow.read_text(), values)
        for spec in args.bypass:
            node_id, _, input_name = spec.partition(":")
            cb.bypass_node(graph, node_id, input_name or "model")

        t0 = time.time()
        hist = run_prompt(server, graph, uuid.uuid4().hex, args.timeout)
        log(f"sampling finished in {time.time() - t0:.1f}s")
        collect_output(hist, out_dir, args.output)
        return 0
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        if server is not None:
            log("server log tail:\n" + server.tail_log())
        return 1
    finally:
        if server is not None:
            server.stop()
        if args.keep_server_log and server is not None:
            kept = args.output.with_suffix(".comfyui.log")
            try:
                shutil.copy2(server.log_file, kept)
                log(f"kept server log at {kept}")
            except OSError:
                pass
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
