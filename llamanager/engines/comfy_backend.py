"""Shared plumbing for engines that run on a headless ComfyUI.

WHY THIS EXISTS. diffusers cannot open the memory-efficient weights the
community actually publishes for large video models. ``MiniMaxH3Transformer3DModel``
has no ``from_single_file`` and no GGUF loader, so on a 32 GB card the
diffusers path has to quantise a bf16 checkpoint on the way to the GPU —
measured on gfx1201 at 5.8 s/tensor and ~50 GB of host RAM, i.e. over an hour
of loading before a single frame is sampled. Krea 2 Turbo's GGUF quants are
blocked by exactly the same gap. ComfyUI reads GGUF and single-file
safetensors natively, so the weights arrive already quantised.

WHAT IS SHARED HERE. Everything that is about ComfyUI rather than about a
particular model: where the checkout and its model folders live, how a model
directory is described to ComfyUI, and how a frozen workflow template is
turned into a concrete API-format graph. A per-model adapter contributes only
its template plus a mapping from llamanager's request fields to that
template's tokens — which is what keeps the family open to further models.

The workflow templates are frozen API-format JSON (see ``comfy_workflows/``),
not graphs assembled in code. That is deliberate: ComfyUI's node signatures
move between releases, and a frozen graph exported from a known-good template
fails loudly on a mismatch instead of silently sampling something else.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Subfolders of a model directory, in ComfyUI's own vocabulary. A model dir
# holds one model assembled from several uploaders (transformer, text encoder,
# VAEs, LoRA each ship separately), which is why the registry grew a
# target_dir option.
MODEL_SUBDIRS = ("diffusion_models", "text_encoders", "vae", "loras",
                 "clip_vision", "upscale_models")


def workflow_path(name: str) -> Path:
    """Absolute path of a frozen workflow template shipped with llamanager."""
    return Path(__file__).with_name("comfy_workflows") / f"{name}.json"


def extra_model_paths_yaml(model_dir: Path) -> str:
    """A ComfyUI ``extra_model_paths.yaml`` pointing at one model directory.

    Written per request rather than installed into the checkout so that two
    models never see each other's files: a loader combo that lists exactly one
    candidate cannot pick the wrong one, and a missing file fails with a clear
    "value not in list" instead of silently loading a neighbour.
    """
    lines = ["llamanager:", f"  base_path: {model_dir}"]
    for sub in MODEL_SUBDIRS:
        lines.append(f"  {sub}: {sub}")
    return "\n".join(lines) + "\n"


def render_workflow(template_text: str, values: dict[str, Any]) -> dict:
    """Substitute ``"__TOKEN__"`` placeholders and parse the result.

    Each token is replaced *including its surrounding quotes* with the JSON
    encoding of its value, so a string stays quoted, a number loses its quotes
    and becomes a real JSON number, and a prompt containing quotes or newlines
    is escaped correctly rather than corrupting the document.

    Raises KeyError if the template contains a token nobody supplied — an
    unsubstituted placeholder would otherwise reach ComfyUI as the literal
    string "__WIDTH__" and fail much further from the cause.
    """
    text = template_text
    for key, value in values.items():
        text = text.replace(f'"__{key}__"', json.dumps(value))

    graph = json.loads(text)
    leftover = sorted(_find_tokens(graph))
    if leftover:
        raise KeyError(
            f"workflow template has unsubstituted tokens: {leftover}")
    # Documentation lives in the template alongside the nodes it explains.
    # ComfyUI would reject a top-level "_comment" as a node with no
    # class_type, and a per-node "_note" as an unknown field, so both are
    # stripped here — keeping the explanation next to the thing it explains
    # without it ever reaching the server.
    for key in [k for k in graph if k.startswith("_")]:
        del graph[key]
    for node in graph.values():
        for key in [k for k in node if k.startswith("_")]:
            del node[key]
    return graph


def _find_tokens(obj: Any) -> set[str]:
    """Collect any remaining ``__TOKEN__`` strings anywhere in the graph."""
    found: set[str] = set()
    if isinstance(obj, str):
        if len(obj) > 4 and obj.startswith("__") and obj.endswith("__"):
            found.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _find_tokens(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _find_tokens(v)
    return found


def bypass_node(graph: dict, node_id: str, passthrough_input: str) -> dict:
    """Remove a node, rewiring its consumers to its own upstream input.

    Used for genuinely optional stages — chiefly the Turbo LoRA, which some
    profiles want and others (the full-step reference profile) do not. Setting
    a LoRA's strength to zero would still read 1.8 GB off disk and load it, so
    the node is dropped from the graph instead.

    ``passthrough_input`` names the input carrying the value that consumers
    should fall back to (for ``LoraLoaderModelOnly`` that is ``model``). Only
    output slot 0 is rewired, which is all a passthrough node has.
    """
    node = graph.get(node_id)
    if node is None:
        return graph
    upstream = node["inputs"][passthrough_input]
    if not isinstance(upstream, list):
        raise ValueError(
            f"cannot bypass node {node_id}: its {passthrough_input!r} input is "
            f"a literal ({upstream!r}), not a link to another node")
    del graph[node_id]
    for other in graph.values():
        for name, value in list(other.get("inputs", {}).items()):
            if isinstance(value, list) and len(value) == 2 \
                    and str(value[0]) == str(node_id):
                other["inputs"][name] = upstream
    return graph


def drop_node(graph: dict, node_id: str) -> dict:
    """Remove a node and every link that pointed at it.

    The counterpart to ``bypass_node`` for nodes that have nothing to pass
    through: a ``LoadImage`` feeding an optional reference slot has no
    upstream input to rewire to, so an unused one must be deleted outright
    and the consumer's input key removed with it.

    This is only correct for OPTIONAL inputs. Dropping a node that feeds a
    required one leaves the consumer missing that key, which ComfyUI rejects
    at validation with the input's name — loud, and at the right layer.
    """
    if node_id not in graph:
        raise KeyError(f"cannot drop node {node_id}: not in the graph")
    del graph[node_id]
    for other in graph.values():
        for name, value in list(other.get("inputs", {}).items()):
            if isinstance(value, list) and len(value) == 2 \
                    and str(value[0]) == str(node_id):
                del other["inputs"][name]
    return graph


def missing_files(model_dir: Path, required: dict[str, str]) -> list[str]:
    """Names from ``{subdir: filename}`` that are not present on disk.

    Checked before the server is started so a half-downloaded model reports
    which component is missing, rather than surfacing as a ComfyUI validation
    error about a combo value.
    """
    return [f"{sub}/{name}"
            for sub, name in required.items()
            if name and not (model_dir / sub / name).is_file()]


# ---------------------------------------------------------------- warm server
#
# Measured on this box: a Krea 2 Turbo request spends 719 of 740 seconds
# building the Qwen3-VL text encoder. The GGUF transformer loads in 1.1s and
# sampling takes 12s. The encoder cost is the same whether the checkpoint is
# fp8_scaled or bf16 (719s vs 684s), so no choice of weights avoids it — the
# only way not to pay it is to not do it again.
#
# ComfyUI caches loaded models between prompts, so a server that outlives one
# request serves the next one with the encoder already built. These helpers
# let a runner find an existing server for the same model directory, and let
# a reaper shut it down once it has been idle, because a warm server holds
# VRAM that nothing else on this machine can use.


def server_state_path(model_dir: Path) -> Path:
    """Where the warm server for ``model_dir`` records itself.

    Keyed by the model directory, not globally: two models must never share a
    server, or the second request would silently sample with the first
    model's weights still resident and its own loaders pointed elsewhere.
    """
    key = hashlib.sha256(str(model_dir.resolve()).encode()).hexdigest()[:16]
    root = Path(os.environ.get("TMPDIR", "/tmp"))
    return root / f"llamanager-comfy-warm-{key}.json"


def heartbeat_path(state: Path) -> Path:
    """Touched by every request; the reaper measures idleness from its mtime."""
    return state.with_suffix(".beat")


def read_live_server(model_dir: Path) -> dict[str, Any] | None:
    """Return the recorded server if it is genuinely alive and serving.

    Every field is re-verified rather than trusted: a stale state file from a
    killed server would otherwise send a request into a closed port, and a
    recycled PID would be even worse.
    """
    state = server_state_path(model_dir)
    try:
        info = json.loads(state.read_text())
    except (OSError, ValueError):
        return None
    pid, port = info.get("pid"), info.get("port")
    if not pid or not port:
        return None
    try:
        os.kill(int(pid), 0)          # alive? (no signal sent)
    except (OSError, ValueError):
        return None
    if str(info.get("model_dir")) != str(model_dir.resolve()):
        return None
    # Its directories must still exist, or results would land nowhere.
    for key in ("output_dir", "input_dir"):
        if not info.get(key) or not Path(info[key]).is_dir():
            return None
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/system_stats", timeout=5) as r:
            json.loads(r.read())
    except Exception:  # noqa: BLE001 — unreachable means not usable
        return None
    return info


def write_server_state(model_dir: Path, pid: int, port: int,
                       output_dir: Path, input_dir: Path) -> Path:
    """Record a warm server. The directories matter as much as the port: a
    reusing run must collect results from, and upload images to, the ones the
    server was actually started with."""
    state = server_state_path(model_dir)
    state.write_text(json.dumps({
        "pid": pid, "port": port,
        "model_dir": str(model_dir.resolve()),
        "output_dir": str(output_dir), "input_dir": str(input_dir),
        "started": time.time(),
    }))
    heartbeat_path(state).write_text(str(time.time()))
    return state


def touch_heartbeat(model_dir: Path) -> None:
    try:
        heartbeat_path(server_state_path(model_dir)).write_text(str(time.time()))
    except OSError:
        pass


def warm_servers() -> list[dict[str, Any]]:
    """Every live warm ComfyUI server on this machine.

    Each entry is the recorded state plus the ``state`` path it came from.
    Records whose process is gone are skipped rather than reported, so a
    caller never acts on a server that has already exited.
    """
    root = Path(os.environ.get("TMPDIR", "/tmp"))
    out: list[dict[str, Any]] = []
    for state in sorted(root.glob("llamanager-comfy-warm-*.json")):
        try:
            info = json.loads(state.read_text())
            pid = int(info["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        info["state"] = str(state)
        out.append(info)
    return out


def stop_warm_servers(grace_seconds: float = 20.0) -> list[int]:
    """Stop every warm server and return the pids that were signalled.

    A warm server holds its weights — 16 GB for Krea 2, 11 GB for
    MiniMax-H3 — so it has to go before anything else claims the card. The
    text engine restarting after an image task is exactly that moment: two
    resident models do not fit on a 32 GB card, and the LLM would fail to
    start or spill to host RAM.

    SIGTERM to the process GROUP, because ComfyUI spawns helpers that would
    otherwise keep the GPU. SIGKILL only after the grace period: killing a
    process that holds a KFD context has leaked GPU memory on this hardware.
    """
    stopped: list[int] = []
    for info in warm_servers():
        pid = int(info["pid"])
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped.append(pid)
        except OSError:
            continue
    deadline = time.time() + grace_seconds
    for pid in stopped:
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.5)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                pass
    # Clear the records of servers that are now gone, so a later request
    # does not try to adopt one. Survivors keep theirs.
    for state in Path(os.environ.get("TMPDIR", "/tmp")).glob(
            "llamanager-comfy-warm-*.json"):
        try:
            info = json.loads(state.read_text())
            os.kill(int(info["pid"]), 0)
        except (OSError, ValueError, KeyError, TypeError):
            for p in (heartbeat_path(state), state):
                try:
                    p.unlink()
                except OSError:
                    pass
    return stopped
