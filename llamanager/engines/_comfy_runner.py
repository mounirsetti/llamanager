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
import threading
import time
import urllib.error
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
        # A server kept warm for the next request is "adopted": this run talks
        # to it but must not reap it on exit. The idle reaper owns its life.
        self.adopted = False

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timestamped: bool = True) -> None:
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
        #
        # Each line is stamped with seconds since launch. ComfyUI prints no
        # timestamps of its own, which makes "where did the twelve minutes
        # go" unanswerable from its log — the stamps turn a wall of INFO
        # lines into a phase profile.
        self._log_fh = open(self.log_file, "w", encoding="utf-8")
        if not timestamped:
            # A server that outlives this process must NOT write into a pipe
            # we own: when this runner exits the read end closes and ComfyUI
            # dies on EPIPE the moment it next logs. Hand it the file directly
            # and give up the timestamps, which are a debugging aid rather
            # than something a warm server needs.
            self.proc = subprocess.Popen(
                argv, cwd=str(self.repo),
                stdout=self._log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return
        self.proc = subprocess.Popen(
            argv, cwd=str(self.repo),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, text=True, errors="replace", bufsize=1,
        )
        t0 = time.time()

        def _stamp() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                self._log_fh.write(f"[{time.time() - t0:7.1f}s] {line}")
                self._log_fh.flush()

        self._log_thread = threading.Thread(target=_stamp, daemon=True)
        self._log_thread.start()

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
            return self.log_file.read_text(encoding="utf-8",
                                           errors="replace")[-limit:]
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        """Terminate the whole process group, then verify it is gone.

        ComfyUI spawns helpers, so signalling only the direct child can leave
        grandchildren holding VRAM. SIGKILL of a process holding a KFD context
        has previously leaked GPU memory on this hardware, so the group gets a
        real chance to exit on SIGTERM first.
        """
        if self.adopted:
            log("leaving the warm server running for the next request")
            self._close_log()
            return
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

    def interrupt(self) -> bool:
        """Ask the server to abandon whatever it is executing. True if sent.

        A warm server outlives the run that submitted the prompt, so when
        that run is cancelled the server carries on regardless — nobody is
        left to collect the result, and it keeps the GPU and the CPU for as
        long as the work takes. One cancelled MiniMax-H3 clip burned 6h47m of
        CPU and 46 GB of RSS this way, with the machine in swap, hours after
        its request was cancelled.

        Best-effort and short-timeout on purpose: this runs from a signal
        handler, where blocking is the worse failure.
        """
        req = urllib.request.Request(f"{self.base}/interrupt", data=b"{}",
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:  # noqa: BLE001 — a dying run must not raise here
            return False

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
        log(f"uploaded {path.name} as {name}")
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


def parse_set(pairs: list[str], *, as_text: bool = False) -> dict[str, object]:
    """Turn ``key=value`` pairs into workflow token values.

    ``--set`` JSON-decodes each value so numbers arrive as numbers, falling
    back to a plain string when the value is not valid JSON.

    ``--set-str`` (``as_text``) never decodes. That distinction matters for
    caller-supplied text: a prompt of "2024" or "true" is valid JSON, so
    decoding it would put a number or a boolean where the graph needs a
    string, and ComfyUI would reject the workflow for a reason that has
    nothing to do with what the user typed.
    """
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            flag = "--set-str" if as_text else "--set"
            raise SystemExit(f"{flag} expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        if as_text:
            out[key.strip()] = raw
            continue
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


class _Heartbeat:
    """Keeps the warm server's heartbeat fresh while a prompt is running.

    The reaper measures idleness from the heartbeat file's age and nothing
    else, so a single touch before submitting only covers requests shorter
    than the idle window. A longer one — the first LoRA request on a GGUF
    transformer, measured at 448 s, or any video clip — went quiet, was
    SIGTERMed mid-generation by its own reaper, and left the runner waiting
    on a dead server until its hour-long timeout.

    Generating IS activity. This says so for as long as it lasts.
    """

    def __init__(self, model_path: Path, cb, period: float = 10.0) -> None:
        self._model_path = model_path
        self._cb = cb
        self._period = period
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Heartbeat":
        self._cb.touch_heartbeat(self._model_path)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            self._cb.touch_heartbeat(self._model_path)

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._period)
        # One last touch: the idle window should start when the work ended,
        # not up to `period` seconds before it.
        self._cb.touch_heartbeat(self._model_path)


def _spawn_reaper(state: Path, beat: Path, pid: int,
                  idle_seconds: float) -> None:
    """Start the detached process that will eventually stop the warm server.

    It cannot be this run: this run is about to exit. See _comfy_reaper.py.
    """
    reaper = Path(__file__).with_name("_comfy_reaper.py")
    subprocess.Popen(
        [sys.executable, str(reaper), "--pid", str(pid),
         "--beat", str(beat), "--state", str(state),
         "--idle", str(idle_seconds)],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"warm server will be reaped after {idle_seconds:.0f}s idle")


def _safetensors_key_count(path: Path) -> int:
    """Number of tensors in a .safetensors file, from its header alone.

    The header is a length-prefixed JSON dict at the start of the file, so
    this costs one small read rather than loading 1.8 GB of weights.
    """
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header = json.loads(fh.read(int.from_bytes(raw, "little")))
    return len([k for k in header if k != "__metadata__"])


# ComfyUI's two ways of saying "this LoRA key matched nothing in the model".
# Neither is an error there: it warns and samples on, which is why a LoRA for
# the wrong architecture produces a normal-looking image and no complaint.
_LORA_MISS_MARKERS = ("lora key not loaded:", "NOT LOADED")


def check_lora_applied(log_file: Path, from_pos: int, lora: Path) -> bool:
    """Report how much of ``lora`` ComfyUI actually bound. False = none of it.

    Reads only the part of the server log this request appended. A warm
    server keeps its log inside the process that started it, so there may be
    nothing to read — that is reported as unchecked, never as success.
    """
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(from_pos)
            fresh = fh.read()
    except OSError:
        fresh = ""
    if not fresh.strip():
        log(f"lora {lora.name}: could not verify (no server log for this "
            "request — a reused warm server keeps its own). Sampling "
            "continued.")
        return True
    missed = sum(1 for line in fresh.splitlines()
                 if any(m in line for m in _LORA_MISS_MARKERS))
    try:
        total = _safetensors_key_count(lora)
    except (OSError, ValueError) as e:
        log(f"lora {lora.name}: {missed} unmatched keys; could not read its "
            f"header to say how many that is ({e})")
        return True
    if missed == 0:
        log(f"lora {lora.name}: applied, {total} keys, none unmatched")
        return True
    if missed >= total:
        log(f"lora {lora.name}: NONE of its {total} keys matched this model. "
            "The image was generated as if no LoRA were selected — this is a "
            "LoRA for a different architecture, or in a key layout ComfyUI "
            "cannot map onto Krea 2.")
        return False
    log(f"lora {lora.name}: {missed} of {total} keys unmatched — it applied "
        "only partially, so the effect will be weaker than intended")
    return True


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
                   help="workflow token, JSON-decoded, e.g. --set WIDTH=1344")
    p.add_argument("--set-str", action="append", default=[], dest="set_strs",
                   help="workflow token kept verbatim as a string; use this "
                        "for prompts and any caller-supplied text")
    p.add_argument("--image", action="append", default=[], dest="images",
                   metavar="TOKEN=PATH",
                   help="upload an image and bind the server-side name to "
                        "TOKEN, e.g. --image REF_IMAGE=/tmp/a.png. Repeatable: "
                        "an edit graph takes one image per reference slot.")
    p.add_argument("--lora-file", type=Path, default=None,
                   help="the LoRA the graph loads. Given one, the runner "
                        "checks afterwards how many of its keys ComfyUI "
                        "actually bound, and fails the request if none did.")
    p.add_argument("--bypass", action="append", default=[],
                   metavar="NODE_ID:INPUT",
                   help="drop a node, rewiring consumers to that input")
    p.add_argument("--drop-node", action="append", default=[],
                   dest="drop_nodes", metavar="NODE_ID",
                   help="delete a node and every link into it. For unused "
                        "OPTIONAL inputs (an unfilled reference slot); a "
                        "required one then fails ComfyUI validation by name.")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--comfy-arg", action="append", default=[],
                   help="extra flag passed through to ComfyUI's main.py")
    p.add_argument("--keep-server-log", action="store_true",
                   help="keep the ComfyUI log file instead of the temp dir")
    p.add_argument("--keep-warm", type=float, default=0.0, metavar="SECONDS",
                   help="leave the server running for this many idle seconds "
                        "so the next request reuses its loaded models. 0 "
                        "(the default) keeps strict one-shot behaviour.")
    args = p.parse_args()

    cb = _load_backend()

    if not args.model_path.is_dir():
        log(f"model path does not exist: {args.model_path}")
        return 2
    if not args.workflow.is_file():
        log(f"workflow template not found: {args.workflow}")
        return 2

    values = parse_set(args.sets)
    values.update(parse_set(args.set_strs, as_text=True))
    work = Path(tempfile.mkdtemp(prefix="llamanager-comfy-"))
    server: ComfyServer | None = None

    def _terminate(signum, _frame):
        # Reap the child before dying, or it keeps the GPU. A warm server is
        # exempt: the reaper owns it, and killing it here would throw away the
        # loaded models the next request is meant to reuse.
        log(f"received signal {signum}; shutting down")
        if server is not None:
            # Whatever it is computing is for THIS run, and this run is over.
            # Leaving a warm server is not the same as leaving it working:
            # without this the abandoned prompt runs to completion (or for
            # hours) with nobody to collect it.
            if getattr(server, "adopted", False):
                if server.interrupt():
                    log("asked the warm server to interrupt the prompt")
            server.stop()
        if server is None or not getattr(server, "adopted", False):
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

        t_start = time.time()
        warm = cb.read_live_server(args.model_path) if args.keep_warm else None
        if warm:
            # Reuse: the expensive part (constructing the text encoder) has
            # already happened inside that process, and ComfyUI keeps it
            # cached between prompts. Its directories are its own, so adopt
            # them rather than the ones prepared above.
            out_dir = Path(warm["output_dir"])
            in_dir = Path(warm["input_dir"])
            # The ORIGINAL process's log, not a fresh work-dir one: the
            # warm server keeps appending there, and the LoRA-binding check
            # reads what this request appended to it.
            warm_log = warm.get("log_file")
            server = ComfyServer(
                python=Path(sys.executable), repo=args.comfy_repo,
                port=int(warm["port"]), output_dir=out_dir, input_dir=in_dir,
                temp_dir=tmp_dir, extra_paths=paths_yaml,
                log_file=(Path(warm_log) if warm_log
                          else work / "comfyui.log"),
                extra_args=args.comfy_arg)
            server.adopted = True
            log(f"reusing warm server pid={warm['pid']} port={warm['port']}")
        else:
            if args.keep_warm:
                # A record we could not adopt means a server that is alive but
                # unusable — unreachable, or wedged on a prompt whose caller
                # is long gone. Starting a second one for the same model would
                # put two copies of the weights on one card AND overwrite the
                # record below, leaving the first with no state file and no
                # reaper that can see it. That is exactly the 2026-08-27 leak.
                orphan = cb.stop_recorded_server(args.model_path)
                if orphan:
                    log(f"stopped an unusable recorded server (pid={orphan}) "
                        "before starting a fresh one")
            server = ComfyServer(
                python=Path(sys.executable), repo=args.comfy_repo,
                port=_free_port(), output_dir=out_dir, input_dir=in_dir,
                temp_dir=tmp_dir, extra_paths=paths_yaml,
                log_file=work / "comfyui.log", extra_args=args.comfy_arg)
            server.start(timestamped=not args.keep_warm)
            server.wait_ready()
            if args.keep_warm:
                server.adopted = True   # leave it running for the next request
                cb.write_server_state(args.model_path, server.proc.pid,
                                      server.port, out_dir, in_dir,
                                      log_file=server.log_file)
                st = cb.server_state_path(args.model_path)
                _spawn_reaper(st, cb.heartbeat_path(st), server.proc.pid,
                              args.keep_warm)
        if args.keep_warm:
            cb.touch_heartbeat(args.model_path)
        t_ready = time.time()

        for spec in args.images:
            token, sep, raw = spec.partition("=")
            if not sep or not token.strip():
                log(f"malformed --image (want TOKEN=PATH): {spec!r}")
                return 2
            path = Path(raw)
            if not path.is_file():
                log(f"image not found: {path}")
                return 2
            values[token.strip()] = server.upload_image(path)

        graph = cb.render_workflow(args.workflow.read_text(), values)
        for spec in args.bypass:
            node_id, _, input_name = spec.partition(":")
            cb.bypass_node(graph, node_id, input_name or "model")
        for node_id in args.drop_nodes:
            cb.drop_node(graph, node_id)

        t_submit = time.time()
        try:
            log_pos = server.log_file.stat().st_size
        except OSError:
            log_pos = 0
        if args.keep_warm:
            with _Heartbeat(args.model_path, cb):
                hist = run_prompt(server, graph, uuid.uuid4().hex,
                                  args.timeout)
        else:
            hist = run_prompt(server, graph, uuid.uuid4().hex, args.timeout)
        t_done = time.time()
        collect_output(hist, out_dir, args.output)

        if args.lora_file is not None:
            # After collect_output on purpose: the image is already saved, so
            # a failure here reports a real result the operator can look at
            # rather than throwing it away.
            if not check_lora_applied(server.log_file, log_pos,
                                      args.lora_file):
                return 3

        # A phase breakdown, because "it took 766 seconds" does not say
        # whether the fix is a faster sampler or a warm server. ComfyUI
        # reports its own execution time, which covers loading the weights
        # plus sampling plus decoding; the difference between that and our
        # wall clock is queue and transport.
        comfy_exec = ""
        for line in server.tail_log(20000).splitlines():
            if "Prompt executed in" in line:
                comfy_exec = line.split("Prompt executed in", 1)[1].strip()
        log(f"TIMING startup={t_ready - t_start:.1f}s "
            f"execute={t_done - t_submit:.1f}s "
            f"collect={time.time() - t_done:.1f}s "
            f"total={time.time() - t_start:.1f}s"
            + (f" comfy_reported={comfy_exec}" if comfy_exec else ""))
        return 0
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        if server is not None:
            # Same reasoning as the signal handler: a timeout or a transport
            # failure ends OUR interest in the prompt, but a warm server keeps
            # executing it — the run that timed out at an hour left a server
            # computing for six more.
            if getattr(server, "adopted", False) and server.interrupt():
                log("asked the warm server to interrupt the prompt")
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
        # A warm server's output and input directories live inside this run's
        # work dir, so it can only be removed when nothing is left running in
        # it. The reaper deletes the state file; the directory is small and
        # lands under TMPDIR, so leaving it is the safe trade.
        if not getattr(server, "adopted", False):
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
