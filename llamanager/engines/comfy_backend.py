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

import json
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


def missing_files(model_dir: Path, required: dict[str, str]) -> list[str]:
    """Names from ``{subdir: filename}`` that are not present on disk.

    Checked before the server is started so a half-downloaded model reports
    which component is missing, rather than surfacing as a ComfyUI validation
    error about a combo value.
    """
    return [f"{sub}/{name}"
            for sub, name in required.items()
            if name and not (model_dir / sub / name).is_file()]
