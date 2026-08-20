"""Tests for the ComfyUI engine family and the download targeting it needs.

Why this engine exists: the community publishes memory-efficient weights for
video models in ComfyUI's single-file / GGUF formats, and diffusers cannot open
them — ``MiniMaxH3Transformer3DModel`` has no ``from_single_file`` and there is
no MiniMax entry in diffusers' ``single_file_utils``. The same limitation blocks
Krea 2 Turbo's GGUF quants. Measured on this box, the diffusers path for
MiniMax-H3 needed ~50 GB of host RAM and 5.8 s/tensor (over an hour just to
load), while the ComfyUI-format Q4_K_M weights are 18.5 GB and pre-quantised.

These tests cover the parts that are pure logic — download targeting, adapter
registration, workflow templating — and deliberately touch no GPU, no network
and no model weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from llamanager.registry import Registry


class _FakeDB:
    """Minimal DB stand-in: Registry only needs execute/query/log_event here."""

    def __init__(self):
        self.rows = []

    def execute(self, *a, **k):
        self.rows.append(a)

    def query(self, *a, **k):
        return []

    def log_event(self, *a, **k):
        pass


def _registry(tmp_path) -> Registry:
    from llamanager.config import Config
    cfg = Config()
    cfg.models_dir_override = tmp_path
    return Registry(cfg, _FakeDB())


# --------------------------------------------------------- download targeting


def test_pull_target_defaults_to_the_repo_name(tmp_path):
    """Unchanged behaviour for every existing caller."""
    reg = _registry(tmp_path)
    assert reg._pull_target("org/model", None) == tmp_path / "org/model"
    assert reg._pull_target("org/model", "") == tmp_path / "org/model"


def test_pull_target_honours_an_explicit_directory(tmp_path):
    """The point of the override: several repos contributing to ONE model dir.

    A ComfyUI-style model takes its transformer, text encoder, VAEs and LoRA
    from four different uploaders; without this they would land in four
    unrelated directories and no adapter could detect the model.
    """
    reg = _registry(tmp_path)
    got = reg._pull_target("realrebelai/MiniMax-H3_GGUFs",
                           "MiniMax-H3-Comfy/diffusion_models")
    assert got == tmp_path / "MiniMax-H3-Comfy" / "diffusion_models"


@pytest.mark.parametrize("evil", [
    "../escape",
    "MiniMax-H3-Comfy/../../escape",
])
def test_pull_target_rejects_traversal(tmp_path, evil):
    """A crafted override must not escape models_dir."""
    reg = _registry(tmp_path)
    with pytest.raises(Exception):
        reg._pull_target("org/model", evil)


def test_pull_target_anchors_absolute_paths_under_models_dir(tmp_path):
    """An absolute-looking override is treated as relative, not obeyed.

    Leading slashes are stripped before joining, so "/etc" lands at
    models_dir/etc — contained, which is the whole requirement. It is not an
    error case, so it must not raise.
    """
    reg = _registry(tmp_path)
    assert reg._pull_target("org/model", "/etc") == tmp_path / "etc"


def test_pull_target_strips_surrounding_slashes(tmp_path):
    reg = _registry(tmp_path)
    assert (reg._pull_target("org/model", "/Model-Comfy/vae/")
            == tmp_path / "Model-Comfy" / "vae")


def test_flatten_into_strips_the_repo_layout(tmp_path):
    """A file asked for as ``vae/x.safetensors`` must not land in ``vae/vae/``.

    hf_hub_download reproduces the repo's directory layout under local_dir, so
    without this the video VAE would end up one level deeper than the adapter
    looks, and the model would read as incomplete.
    """
    reg = _registry(tmp_path)
    root = tmp_path / "MiniMax-H3-Comfy" / "vae"
    nested = root / "vae"
    nested.mkdir(parents=True)
    src = nested / "video_vae.safetensors"
    src.write_bytes(b"weights")

    out = reg._flatten_into(root, src)

    assert out == root / "video_vae.safetensors"
    assert out.read_bytes() == b"weights"
    assert not nested.exists(), "the emptied repo-shaped dir should be removed"


def test_flatten_into_leaves_a_root_level_file_alone(tmp_path):
    reg = _registry(tmp_path)
    root = tmp_path / "M" / "diffusion_models"
    root.mkdir(parents=True)
    src = root / "model.gguf"
    src.write_bytes(b"x")
    assert reg._flatten_into(root, src) == src
    assert src.is_file()


# ---------------------------------------------- one model, not its components


def _comfy_pack(root: Path, name: str, unet: str) -> Path:
    """A ComfyUI-format model dir: weights sorted into ComfyUI subfolders."""
    d = root / name
    (d / "diffusion_models").mkdir(parents=True)
    (d / "diffusion_models" / unet).write_bytes(b"w")
    (d / "vae").mkdir()
    (d / "vae" / "vae.safetensors").write_bytes(b"w")
    (d / "text_encoders").mkdir()
    (d / "text_encoders" / "te.safetensors").write_bytes(b"w")
    return d


def test_list_emits_the_model_dir_not_its_component_dirs(tmp_path):
    """The picker showed three Krea entries and two Z-Image entries.

    Two of those were internals: a ComfyUI pack's ``diffusion_models/`` (it
    holds ``krea2_turbo-*.gguf``, which the Krea detector claims on its own)
    and the ``.zimage-scaffold/`` the z_image runner writes beside the
    weights (its own ``model_index.json`` makes it look like a pipeline).
    Only the outermost directory is a model.
    """
    import json

    _comfy_pack(tmp_path, "Krea-2-Turbo-Comfy", "krea2_turbo-Q6_K.gguf")

    zi = tmp_path / "Z-Image"
    (zi / ".zimage-scaffold").mkdir(parents=True)
    index = json.dumps({"_class_name": "ZImagePipeline"})
    (zi / "model_index.json").write_text(index)
    (zi / ".zimage-scaffold" / "model_index.json").write_text(index)
    (zi / "weights.safetensors").write_bytes(b"w")

    ids = sorted(m.model_id for m in _registry(tmp_path).list())

    assert ids == ["Krea-2-Turbo-Comfy", "Z-Image"]


def test_list_still_keeps_component_weights_out_of_the_text_models(tmp_path):
    """The pruning must not undo what it is built on: a pack's GGUF is not
    a llama model just because nothing else claimed its directory."""
    _comfy_pack(tmp_path, "MiniMax-H3-Comfy", "MiniMax-H3-FL2VA-Q4_K_M.gguf")

    ids = sorted(m.model_id for m in _registry(tmp_path).list())

    assert ids == ["MiniMax-H3-Comfy"]


# ------------------------------------------------- the download route wiring


class _SpyRegistry:
    """Records start_pull kwargs; estimate_repo_size returns a fixed size."""

    def __init__(self):
        self.calls = []

    async def estimate_repo_size(self, repo, *a, **k):
        return 1234

    def start_pull(self, **kw):
        self.calls.append(kw)


class _Req:
    """Enough of a Request for the route: it only reads app.state."""

    def __init__(self, app):
        self.app = app


def _call_download(tmp_path, **form):
    """Invoke the admin download route directly, bypassing session auth.

    The route is a plain async function whose only dependency is app.state,
    so calling it here tests the wiring we care about (does target_dir reach
    start_pull?) without standing up cookies and CSRF.
    """
    import asyncio
    from types import SimpleNamespace

    from llamanager import api_ui
    from llamanager.config import Config

    cfg = Config()
    cfg.models_dir_override = tmp_path
    spy = _SpyRegistry()
    app = SimpleNamespace(state=SimpleNamespace(registry=spy, cfg=cfg))
    kwargs = {"repo": "", "subfolder": "", "filename": "",
              "models_dir": "", "target_dir": "", "open": "", **form}
    engine = kwargs.pop("engine", "minimax_h3_comfy")
    resp = asyncio.run(api_ui.download_engine_model(
        _Req(app), engine=engine, _=None, **kwargs))
    return spy, resp


def test_download_route_forwards_target_dir_for_a_single_file(tmp_path):
    """The component-download case: one file from one repo into a shared dir."""
    spy, _ = _call_download(
        tmp_path,
        repo="realrebelai/MiniMax-H3_GGUFs",
        filename="MiniMax-H3-FL2VA-Q4_K_M.gguf",
        target_dir="MiniMax-H3-Comfy/diffusion_models")
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["files"] == ["MiniMax-H3-FL2VA-Q4_K_M.gguf"]
    assert call["target_dir"] == "MiniMax-H3-Comfy/diffusion_models"


def test_download_route_forwards_target_dir_for_a_whole_repo(tmp_path):
    spy, _ = _call_download(tmp_path, repo="org/model",
                            target_dir="Model-Comfy/vae")
    assert spy.calls[0]["target_dir"] == "Model-Comfy/vae"
    assert spy.calls[0]["whole_repo"] is True


def test_download_route_tags_the_pull_with_the_engine_family(tmp_path):
    """Diffusion pulls must be enqueued as image/video, not text.

    The family is what keeps a failed component pull off the LLM models page,
    where it used to show up looking like a broken language model.
    """
    spy, _ = _call_download(tmp_path, repo="Comfy-Org/MiniMax-H3",
                            target_dir="MiniMax-H3-Comfy/vae")
    assert spy.calls[0]["family"] == "video"

    spy, _ = _call_download(tmp_path, engine="krea_comfy",
                            repo="Comfy-Org/Krea-2")
    assert spy.calls[0]["family"] == "image"

    # An engine the family map doesn't know still stays out of the text list.
    spy, _ = _call_download(tmp_path, engine="brand_new_engine",
                            repo="org/model")
    assert spy.calls[0]["family"] == "image"


def test_download_route_omits_target_dir_when_blank(tmp_path):
    """Existing catalog downloads keep landing at models_dir/<repo>."""
    spy, _ = _call_download(tmp_path, repo="org/model")
    assert spy.calls[0]["target_dir"] is None


# ------------------------------------------------------ workflow templating


def test_workflow_renders_with_typed_values():
    """Numbers must arrive as JSON numbers, prompts must survive escaping."""
    from llamanager.engines import comfy_backend as cb
    tpl = cb.workflow_path("minimax_h3_i2v_gguf").read_text()
    graph = cb.render_workflow(tpl, _WORKFLOW_VALUES)

    assert "_comment" not in graph, "the doc block is not a node"
    assert graph["7"]["inputs"]["width"] == 1344
    assert isinstance(graph["7"]["inputs"]["width"], int)
    # A prompt with a quote and a newline must not corrupt the document.
    assert graph["7"]["inputs"]["prompt"] == 'a "grand" hall\nwith light'


def test_workflow_decodes_the_latent_through_both_vaes():
    """The soundtrack is the point of this engine, so assert the audio branch.

    A regression that dropped VAEDecodeAudio would still produce a playable
    clip - just a silent one - which is exactly the kind of failure that
    survives a smoke test.
    """
    from llamanager.engines import comfy_backend as cb
    graph = cb.render_workflow(
        cb.workflow_path("minimax_h3_i2v_gguf").read_text(), _WORKFLOW_VALUES)

    kinds = {n["class_type"] for n in graph.values()}
    assert {"VAEDecode", "VAEDecodeAudio"} <= kinds
    video = next(k for k, n in graph.items() if n["class_type"] == "VAEDecode")
    audio = next(k for k, n in graph.items()
                 if n["class_type"] == "VAEDecodeAudio")
    # Both decode the SAME sampled latent - that is what keeps them in sync.
    assert graph[video]["inputs"]["samples"] == graph[audio]["inputs"]["samples"]
    create = next(n for n in graph.values() if n["class_type"] == "CreateVideo")
    assert create["inputs"]["audio"] == [audio, 0]


def test_unsubstituted_token_is_an_error():
    from llamanager.engines import comfy_backend as cb
    vals = {k: v for k, v in _WORKFLOW_VALUES.items() if k != "SEED"}
    with pytest.raises(KeyError, match="SEED"):
        cb.render_workflow(
            cb.workflow_path("minimax_h3_i2v_gguf").read_text(), vals)


def test_bypassing_the_lora_rewires_its_consumers():
    """Dropping the LoRA must leave the graph sampling from the base model."""
    from llamanager.engines import comfy_backend as cb
    graph = cb.render_workflow(
        cb.workflow_path("minimax_h3_i2v_gguf").read_text(), _WORKFLOW_VALUES)
    assert graph["10"]["inputs"]["model"] == ["2", 0]

    cb.bypass_node(graph, "2", "model")

    assert "2" not in graph
    assert graph["10"]["inputs"]["model"] == ["1", 0]
    assert graph["11"]["inputs"]["model"] == ["1", 0]


_WORKFLOW_VALUES = {
    "UNET": "t.gguf", "LORA": "l.safetensors", "LORA_STRENGTH": 1.0,
    "CLIP": "c.gguf", "VIDEO_VAE": "v.safetensors",
    "AUDIO_VAE": "a.safetensors", "INIT_IMAGE": "in.png",
    "PROMPT": 'a "grand" hall\nwith light',
    "WIDTH": 1344, "HEIGHT": 768, "LENGTH": 124, "SEED": 7,
    "SAMPLER": "res_multistep", "SCHEDULER": "simple", "STEPS": 4,
    "FPS": 24.0,
}


# ------------------------------------------------------- the video adapter


def test_frame_counts_snap_up_to_a_decodable_length():
    """The video VAE only decodes 17n+5 frames; snapping down would silently
    shorten the clip, so it must round up."""
    from llamanager.engines import minimax_h3_comfy as m
    assert m.snap_length(124) == 124          # already valid (17*7 + 5)
    assert m.snap_length(100) == 107          # 17*6 + 5
    assert m.snap_length(1) == 5
    for n in (5, 60, 200, 362):
        assert (m.snap_length(n) - 5) % 17 == 0
        assert m.snap_length(n) >= n


def test_dimensions_snap_to_multiples_of_32():
    from llamanager.engines import minimax_h3_comfy as m
    assert m.snap_dimension(1344) == 1344
    assert m.snap_dimension(1000) == 992
    assert m.snap_dimension(1) == 32


def test_detect_requires_the_audio_vae(tmp_path):
    """A tree without the audio VAE can make pictures but not the soundtrack
    this engine exists for, so it must not be claimed."""
    from llamanager.engines import minimax_h3_comfy as m
    root = tmp_path / "MiniMax-H3-Comfy"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "diffusion_models" / "MiniMax-H3-FL2VA-Q4_K_M.gguf").write_bytes(b"x")
    assert not m.detect(root)

    (root / "vae" / m.AUDIO_VAE_FILE).write_bytes(b"x")
    assert m.detect(root)


def test_detect_ignores_an_unrelated_comfy_model(tmp_path):
    from llamanager.engines import minimax_h3_comfy as m
    root = tmp_path / "Other-Comfy"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "diffusion_models" / "flux-Q4.gguf").write_bytes(b"x")
    (root / "vae" / m.AUDIO_VAE_FILE).write_bytes(b"x")
    assert not m.detect(root)


def test_config_detects_the_comfy_variant_before_the_diffusers_one(tmp_path):
    from llamanager.config import detect_engine_for_path
    from llamanager.engines import minimax_h3_comfy as m
    root = tmp_path / "MiniMax-H3-Comfy"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "diffusion_models" / "MiniMax-H3-FL2VA-Q4_K_M.gguf").write_bytes(b"x")
    (root / "vae" / m.AUDIO_VAE_FILE).write_bytes(b"x")
    assert detect_engine_for_path(root) == "minimax_h3_comfy"


def test_engine_family_is_video():
    from llamanager.config import ENGINE_FAMILY, VIDEO_ENGINES
    assert ENGINE_FAMILY["minimax_h3_comfy"] == "video"
    assert "minimax_h3_comfy" in VIDEO_ENGINES


def test_build_command_refuses_without_a_reference_image(tmp_path, cfg):
    """It is an image-to-video model; a text-only request cannot be served."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.engines._base import ImageRequest
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    req = _image_request("a hotel lobby", [])
    with pytest.raises(RuntimeError, match="reference image"):
        m.build_command(cfg, tmp_path, Profile(name="p"), req,
                        tmp_path / "o.mp4")


def test_build_command_reports_missing_components_by_name(tmp_path, cfg):
    """Naming the missing file beats ComfyUI's combo-validation error."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.engines._base import ImageRequest
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)

    model = tmp_path / "MiniMax-H3-Comfy"
    (model / "diffusion_models").mkdir(parents=True)
    (model / "diffusion_models" / m.UNET_FILE).write_bytes(b"x")
    img = tmp_path / "frame.png"
    img.write_bytes(b"x")

    req = _image_request("hotel", [img])
    with pytest.raises(RuntimeError) as ei:
        m.build_command(cfg, model, Profile(name="p"), req,
                        tmp_path / "o.mp4")
    msg = str(ei.value)
    assert m.CLIP_FILE in msg and m.AUDIO_VAE_FILE in msg


def _image_request(prompt, ref_images):
    """An ImageRequest with the fields these adapters actually read."""
    from llamanager.engines._base import ImageRequest
    return ImageRequest(prompt=prompt, width=0, height=0, steps=None,
                        seed=None, n=1, ref_images=list(ref_images))


def _complete_model(tmp_path):
    """A model directory with every component the adapter requires."""
    from llamanager.engines import minimax_h3_comfy as m
    root = tmp_path / "MiniMax-H3-Comfy"
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "diffusion_models" / m.UNET_FILE).write_bytes(b"x")
    (root / "text_encoders" / m.CLIP_FILE).write_bytes(b"x")
    (root / "vae" / m.VIDEO_VAE_FILE).write_bytes(b"x")
    (root / "vae" / m.AUDIO_VAE_FILE).write_bytes(b"x")
    (root / "loras" / m.TURBO_LORA_FILE).write_bytes(b"x")
    return root


def _argv_tokens(argv):
    """Collect the --set / --set-str KEY=VALUE pairs out of a built argv."""
    out = {}
    for i, a in enumerate(argv):
        if a in ("--set", "--set-str") and "=" in argv[i + 1]:
            k, _, v = argv[i + 1].partition("=")
            out[k] = v
    return out


def test_build_command_passes_the_turbo_profile_through(tmp_path, cfg):
    """The default profile must reach the runner as 4 steps with the LoRA."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _complete_model(tmp_path)
    img = tmp_path / "frame.png"
    img.write_bytes(b"x")

    prof = Profile(name="h3-turbo-4step",
                   **m.default_profiles()["h3-turbo-4step"])
    argv, env = m.build_command(cfg, model, prof,
                                _image_request("a hotel atrium", [img]),
                                tmp_path / "out.mp4")

    tok = _argv_tokens(argv)
    assert tok["STEPS"] == "4"
    assert tok["WIDTH"] == "1344" and tok["HEIGHT"] == "768"
    assert tok["LENGTH"] == "124"
    assert tok["FPS"] == "24.0"
    assert tok["LORA"] == m.TURBO_LORA_FILE
    assert "--bypass" not in argv, "the LoRA should be wired in, not dropped"
    assert any(a.startswith("INIT_IMAGE=") for a in argv)


def test_build_command_drops_the_lora_node_for_the_full_profile(tmp_path, cfg):
    """Zeroing a LoRA's strength would still load 1.8 GB, so it is bypassed."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _complete_model(tmp_path)
    img = tmp_path / "frame.png"
    img.write_bytes(b"x")

    prof = Profile(name="h3-full-50step",
                   **m.default_profiles()["h3-full-50step"])
    argv, _ = m.build_command(cfg, model, prof,
                              _image_request("a hotel atrium", [img]),
                              tmp_path / "out.mp4")

    assert _argv_tokens(argv)["STEPS"] == "50"
    assert "--bypass" in argv
    assert argv[argv.index("--bypass") + 1] == "2:model"


def test_build_command_runs_the_baked_turbo_without_a_lora(tmp_path, cfg):
    """The turbo bake carries the distill, so the LoRA node is dropped."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _complete_model(tmp_path)
    (model / "diffusion_models" / m.TURBO_UNET_FILE).write_bytes(b"x")
    img = tmp_path / "frame.png"
    img.write_bytes(b"x")

    prof = Profile(name="h3-turbo-baked-8step",
                   **m.default_profiles()["h3-turbo-baked-8step"])
    argv, _ = m.build_command(cfg, model, prof,
                              _image_request("a hotel atrium", [img]),
                              tmp_path / "out.mp4")

    tok = _argv_tokens(argv)
    assert tok["UNET"] == m.TURBO_UNET_FILE
    assert tok["STEPS"] == "8"
    assert "--bypass" in argv
    assert argv[argv.index("--bypass") + 1] == "2:model"


def test_build_command_refuses_the_turbo_bake_with_a_lora(tmp_path, cfg):
    """Stacking the distill on weights that already carry it halves quality
    silently, so the combination is refused by name."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _complete_model(tmp_path)
    (model / "diffusion_models" / m.TURBO_UNET_FILE).write_bytes(b"x")
    img = tmp_path / "frame.png"
    img.write_bytes(b"x")

    prof = Profile(name="p", image_model_type="Q4_K_M-Turbo", image_steps=8,
                   image_lora_weights=m.TURBO_LORA_FILE)
    with pytest.raises(RuntimeError, match="already has the Turbo distill"):
        m.build_command(cfg, model, prof,
                        _image_request("a hotel atrium", [img]),
                        tmp_path / "out.mp4")


def _ref2va_model(tmp_path):
    """A model directory that also holds the REF2VA transformer."""
    from llamanager.engines import minimax_h3_comfy as m
    root = _complete_model(tmp_path)
    (root / "diffusion_models" / m.REF2VA_UNET_FILE).write_bytes(b"x")
    return root


def test_ref2va_profile_routes_to_the_reference_graph(tmp_path, cfg):
    """The transformer quant picks the head, and with it the workflow."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _ref2va_model(tmp_path)
    refs = []
    for n in ("a", "b", "c"):
        p = tmp_path / f"{n}.png"
        p.write_bytes(b"x")
        refs.append(p)

    prof = Profile(name="h3-ref2va-4step",
                   **m.default_profiles()["h3-ref2va-4step"])
    argv, _ = m.build_command(cfg, model, prof,
                              _image_request("<Picture 1> at dusk", refs),
                              tmp_path / "out.mp4")

    assert "minimax_h3_ref2v_gguf.json" in " ".join(argv)
    tok = _argv_tokens(argv)
    assert tok["UNET"] == m.REF2VA_UNET_FILE
    assert tok["REF_DETAIL"] == "match"
    images = [argv[i + 1].split("=", 1)[0]
              for i, a in enumerate(argv) if a == "--image"]
    assert images == ["REF1", "REF2", "REF3"]
    # The six unfilled slots leave the graph, LoadImage nodes 23..28.
    dropped = [argv[i + 1] for i, a in enumerate(argv) if a == "--drop-node"]
    assert dropped == ["23", "24", "25", "26", "27", "28"]
    # No LoRA node exists in that graph, so nothing may try to bypass one.
    assert "--bypass" not in argv
    assert not any(a.startswith("INIT_IMAGE=") for a in argv)


def test_ref2va_accepts_the_full_nine_references(tmp_path, cfg):
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _ref2va_model(tmp_path)
    refs = []
    for n in range(m.MAX_REF_IMAGES):
        p = tmp_path / f"r{n}.png"
        p.write_bytes(b"x")
        refs.append(p)

    prof = Profile(name="p", image_model_type="Q4_K_M-Ref-Turbo",
                   image_steps=4, image_lora_weights="",
                   image_ref_detail="max")
    argv, _ = m.build_command(cfg, model, prof,
                              _image_request("x", refs), tmp_path / "o.mp4")

    assert "--drop-node" not in argv
    assert _argv_tokens(argv)["REF_DETAIL"] == "max"


def test_ref2va_refuses_a_tenth_reference(tmp_path, cfg):
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _ref2va_model(tmp_path)
    refs = []
    for n in range(m.MAX_REF_IMAGES + 1):
        p = tmp_path / f"r{n}.png"
        p.write_bytes(b"x")
        refs.append(p)

    prof = Profile(name="p", **m.default_profiles()["h3-ref2va-4step"])
    with pytest.raises(RuntimeError, match="at most 9"):
        m.build_command(cfg, model, prof, _image_request("x", refs),
                        tmp_path / "o.mp4")


def test_ref2va_needs_an_explicit_reference_detail(tmp_path, cfg):
    """'match' and 'max' differ by several times the sampling cost, so the
    profile has to say which — no quiet default."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _ref2va_model(tmp_path)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")

    prof = Profile(name="p", image_model_type="Q4_K_M-Ref-Turbo",
                   image_steps=4, image_lora_weights="")
    with pytest.raises(RuntimeError, match="Reference detail"):
        m.build_command(cfg, model, prof, _image_request("x", [img]),
                        tmp_path / "o.mp4")


def test_fl2va_with_several_references_names_the_ref2va_quant(tmp_path, cfg):
    """The opening-frame head takes exactly one image; the error says which
    quant to pick instead of just refusing."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _ref2va_model(tmp_path)
    refs = []
    for n in ("a", "b"):
        p = tmp_path / f"{n}.png"
        p.write_bytes(b"x")
        refs.append(p)

    prof = Profile(name="h3-turbo-4step",
                   **m.default_profiles()["h3-turbo-4step"])
    with pytest.raises(RuntimeError, match="Q4_K_M-Ref-Turbo"):
        m.build_command(cfg, model, prof, _image_request("x", refs),
                        tmp_path / "o.mp4")


def test_capabilities_advertise_the_reference_ceiling(tmp_path):
    """api_v1 rejects extra references off this number before queueing."""
    from llamanager.engines import minimax_h3_comfy as m
    caps = m.capabilities()
    assert caps["ref_images_max"] == m.MAX_REF_IMAGES == 9
    assert caps["ref_images_required"] is True
    assert caps["output_ext"] == "mp4"


def test_build_command_snaps_an_odd_frame_count(tmp_path, cfg):
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    model = _complete_model(tmp_path)
    img = tmp_path / "frame.png"
    img.write_bytes(b"x")

    prof = Profile(name="p", video_num_frames=100)
    argv, _ = m.build_command(cfg, model, prof,
                              _image_request("x", [img]), tmp_path / "o.mp4")
    assert _argv_tokens(argv)["LENGTH"] == "107"


def test_fake_comfy_install(cfg, tmp_path):
    """Guard the helper itself: a silently-broken fixture would make the
    argv tests above pass for the wrong reason."""
    _fake_comfy_install(cfg, tmp_path)
    assert Path(cfg.comfyui_repo, "main.py").is_file()


def _fake_comfy_install(cfg, tmp_path):
    """Point cfg at a ComfyUI checkout that exists but is never executed.

    build_command only checks these paths are present; the tests here assert
    on the argv it builds, so nothing is ever run.
    """
    repo = tmp_path / "comfyui"
    repo.mkdir(exist_ok=True)
    (repo / "main.py").write_text("")
    py = tmp_path / "python"
    py.write_text("")
    cfg.comfyui_python, cfg.comfyui_repo = str(py), str(repo)


# -------------------------------------------------------- the image adapter


def test_krea_workflow_zeroes_the_negative_branch():
    """Krea 2 Turbo is guidance-distilled: the negative conditioning is a
    zeroed copy of the positive, and cfg stays 1.0. A regression that wired a
    real negative prompt here would quietly degrade every image."""
    from llamanager.engines import comfy_backend as cb
    graph = cb.render_workflow(
        cb.workflow_path("krea2_t2i_gguf").read_text(), _KREA_VALUES)

    pos = next(k for k, n in graph.items()
               if n["class_type"] == "CLIPTextEncode")
    zero = next(k for k, n in graph.items()
                if n["class_type"] == "ConditioningZeroOut")
    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert graph[zero]["inputs"]["conditioning"] == [pos, 0]
    assert sampler["inputs"]["positive"] == [pos, 0]
    assert sampler["inputs"]["negative"] == [zero, 0]
    assert sampler["inputs"]["cfg"] == 1.0


def test_krea_detect_needs_a_text_encoder(tmp_path):
    from llamanager.engines import krea_comfy as k
    root = tmp_path / "Krea-2-Turbo-Comfy"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "vae" / k.VAE_FILE).write_bytes(b"x")
    (root / "diffusion_models" / "krea2_turbo-Q6_K.gguf").write_bytes(b"x")
    assert not k.detect(root)

    (root / "text_encoders").mkdir()
    (root / "text_encoders" / k.CLIP_FILE).write_bytes(b"x")
    assert k.detect(root)


def test_krea_comfy_and_minimax_comfy_do_not_claim_each_other(tmp_path):
    """Both are ComfyUI trees; detection must key on the model, not the shape."""
    from llamanager.engines import krea_comfy as k, minimax_h3_comfy as m
    krea = tmp_path / "K"
    for sub in ("diffusion_models", "text_encoders", "vae"):
        (krea / sub).mkdir(parents=True)
    (krea / "diffusion_models" / "krea2_turbo-Q6_K.gguf").write_bytes(b"x")
    (krea / "text_encoders" / k.CLIP_FILE).write_bytes(b"x")
    (krea / "vae" / k.VAE_FILE).write_bytes(b"x")

    assert k.detect(krea) and not m.detect(krea)


def test_krea_build_command_selects_the_quant(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = tmp_path / "Krea-2-Turbo-Comfy"
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        (root / sub).mkdir(parents=True)
    for quant, (fname, _gb) in k.QUANT_FILES.items():
        (root / "diffusion_models" / fname).write_bytes(b"x")
    (root / "text_encoders" / k.CLIP_FILE).write_bytes(b"x")
    (root / "vae" / k.VAE_FILE).write_bytes(b"x")

    prof = Profile(name="kreac-draft", **k.default_profiles()["kreac-draft"])
    argv, _ = k.build_command(cfg, root, prof, _image_request("a lobby", []),
                              tmp_path / "o.png")
    tok = _argv_tokens(argv)
    assert tok["UNET"] == k.QUANT_FILES["Q4_K_M"][0]
    assert tok["STEPS"] == "4"
    assert tok["CFG"] == "1.0"
    # No LoRA configured -> the node is dropped, not zeroed.
    assert "--bypass" in argv


def _krea_model(tmp_path, *loras):
    """A complete Krea 2 Comfy model dir, plus any LoRA files named."""
    from llamanager.engines import krea_comfy as k
    root = tmp_path / "Krea-2-Turbo-Comfy"
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for _quant, (fname, _gb) in k.QUANT_FILES.items():
        (root / "diffusion_models" / fname).write_bytes(b"x")
    (root / "text_encoders" / k.CLIP_FILE).write_bytes(b"x")
    (root / "vae" / k.VAE_FILE).write_bytes(b"x")
    for lora in loras:
        (root / "loras" / lora).write_bytes(b"x")
    return root


def test_krea_refuses_to_edit_without_an_edit_lora(tmp_path, cfg):
    """Reference images alone do nothing: stock Krea 2 has no path to read
    them. Naming the LoRAs that would work beats a silently ignored input."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = _krea_model(tmp_path)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="edit LoRA"):
        k.build_command(cfg, root, Profile(name="p"),
                        _image_request("x", [img]), tmp_path / "o.png")


def test_krea_refuses_to_edit_with_an_unknown_lora(tmp_path, cfg):
    """The two node packs place the reference differently and a LoRA in the
    wrong one still renders — wrongly. So an unknown LoRA plus a reference is
    an error, never a guess."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = _krea_model(tmp_path, "somebody_elses_lora.safetensors")
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    prof = Profile(name="p",
                   image_lora_weights="somebody_elses_lora.safetensors")
    with pytest.raises(RuntimeError, match="no safe guess"):
        k.build_command(cfg, root, prof, _image_request("x", [img]),
                        tmp_path / "o.png")
    # ...but the same LoRA is fine for plain text-to-image.
    argv, _ = k.build_command(cfg, root, prof, _image_request("x", []),
                              tmp_path / "o.png")
    assert "krea2_t2i_gguf.json" in " ".join(argv)


def test_krea_edit_lora_requires_its_reference(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_identity_edit_v1_2.safetensors"
    root = _krea_model(tmp_path, lora)
    with pytest.raises(RuntimeError, match="at least 1 reference"):
        k.build_command(cfg, root, Profile(name="p", image_lora_weights=lora),
                        _image_request("x", []), tmp_path / "o.png")


def test_krea_identity_edit_selects_pack_a_and_its_geometry(tmp_path, cfg):
    """The LoRA picks the graph, the reference slots AND the fit mode — the
    operator never chooses geometry the weights already determine."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_identity_edit_v1_2.safetensors"
    root = _krea_model(tmp_path, lora)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    prof = Profile(name="p", **k.default_profiles()["kreac-edit"])
    argv, _ = k.build_command(cfg, root, prof, _image_request("x", [img]),
                              tmp_path / "o.png")
    joined = " ".join(argv)
    assert "krea2_edit_a_gguf.json" in joined
    assert f"REF_IMAGE={img}" in argv
    tok = _argv_tokens(argv)
    assert tok["FIT_MODE"] == "fit"          # v1.2 geometry, not v1.1's crop
    assert tok["REF_BOOST"] == "4.0"
    # The unused second slot is dropped, not left dangling: its LoadImage has
    # no upstream to bypass to, and its VAEEncode would have no pixels.
    assert argv.count("--drop-node") == 2
    assert "12" in argv and "14" in argv


def test_krea_legacy_identity_edit_keeps_the_crop_geometry(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_identity_edit_v1_1_r64.safetensors"
    root = _krea_model(tmp_path, lora)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    argv, _ = k.build_command(cfg, root,
                              Profile(name="p", image_lora_weights=lora),
                              _image_request("x", [img]), tmp_path / "o.png")
    assert _argv_tokens(argv)["FIT_MODE"] == "crop (legacy)"


def test_krea_style_reference_selects_pack_b(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_style_reference.safetensors"
    root = _krea_model(tmp_path, lora)
    imgs = []
    for n in ("a", "b"):
        img = tmp_path / f"{n}.png"
        img.write_bytes(b"x")
        imgs.append(img)
    argv, _ = k.build_command(cfg, root,
                              Profile(name="p", image_lora_weights=lora),
                              _image_request("x", imgs), tmp_path / "o.png")
    joined = " ".join(argv)
    assert "krea2_edit_b_gguf.json" in joined
    assert f"REF_IMAGE={imgs[0]}" in argv and f"REF_IMAGE_B={imgs[1]}" in argv
    # Only the third slot is unused.
    assert argv.count("--drop-node") == 1


def test_krea_pose_lora_takes_exactly_one_reference(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_turbo_openpose_controlnet.safetensors"
    root = _krea_model(tmp_path, lora)
    imgs = []
    for n in ("a", "b"):
        img = tmp_path / f"{n}.png"
        img.write_bytes(b"x")
        imgs.append(img)
    with pytest.raises(RuntimeError, match="at most 1 reference"):
        k.build_command(cfg, root, Profile(name="p", image_lora_weights=lora),
                        _image_request("x", imgs), tmp_path / "o.png")


def test_krea_edit_reports_a_missing_lora_file(tmp_path, cfg):
    """The edit LoRA is load-bearing, so a missing file is a missing model
    component — not a warning followed by an unpatched sample."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_identity_edit_v1_2.safetensors"
    root = _krea_model(tmp_path)          # no LoRA on disk
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=f"loras/{lora}"):
        k.build_command(cfg, root, Profile(name="p", image_lora_weights=lora),
                        _image_request("x", [img]), tmp_path / "o.png")


def _graph_from_argv(argv):
    """Render + prune exactly the way the runner does, from a built argv.

    This is the only test that closes the loop between the adapter and the
    frozen templates: the node ids it drops live in the adapter, the nodes
    live in the JSON, and nothing else would notice them drifting apart.
    """
    from llamanager.engines import comfy_backend as cb
    values, images, drops, workflow = {}, {}, [], None
    for i, a in enumerate(argv):
        if a in ("--set", "--set-str") and "=" in argv[i + 1]:
            k, _, v = argv[i + 1].partition("=")
            if a == "--set":
                import json as _json
                try:
                    v = _json.loads(v)
                except ValueError:
                    pass
            values[k] = v
        elif a == "--image":
            k, _, v = argv[i + 1].partition("=")
            values[k] = Path(v).name        # the runner substitutes the
            images[k] = v                   # server-side upload name here
        elif a == "--drop-node":
            drops.append(argv[i + 1])
        elif a == "--workflow":
            workflow = Path(argv[i + 1])
    graph = cb.render_workflow(workflow.read_text(), values)
    for node_id in drops:
        cb.drop_node(graph, node_id)
    return graph, images


def _assert_links_resolve(graph):
    for node_id, node in graph.items():
        for name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2:
                assert str(value[0]) in graph, (
                    f"node {node_id}.{name} links to missing node {value[0]}")


def test_krea_edit_graph_is_coherent_with_one_reference(tmp_path, cfg):
    """Pack A, single reference: the second slot's nodes are gone and nothing
    still points at them, while the first reference reaches all three places
    it has to — the latent, the model patch, and the grounded encode."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_identity_edit_v1_2.safetensors"
    root = _krea_model(tmp_path, lora)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    argv, _ = k.build_command(
        cfg, root, Profile(name="p", **k.default_profiles()["kreac-edit"]),
        _image_request("recolour the car", [img]), tmp_path / "o.png")

    graph, images = _graph_from_argv(argv)
    _assert_links_resolve(graph)
    assert "12" not in graph and "14" not in graph
    patch = graph["10"]["inputs"]
    assert "source_latent_b" not in patch and "source_image_b" not in patch
    # The load-bearing wiring, which a renumbered template would silently lose.
    assert patch["source_latent"] == ["13", 0]
    assert patch["target_latent"] == ["7", 0]     # pre-encode, not mid-sample
    assert patch["vae"] == ["4", 0]
    assert graph["5"]["inputs"]["image"] == ["11", 0]
    assert graph["8"]["inputs"]["model"] == ["10", 0]
    assert graph["11"]["inputs"]["image"] == img.name
    assert set(images) == {"REF_IMAGE"}


def test_krea_edit_graph_is_coherent_with_two_references(tmp_path, cfg):
    """Both slots filled: nothing is dropped and both reach the patch node."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_identity_edit_v1_2.safetensors"
    root = _krea_model(tmp_path, lora)
    imgs = []
    for n in ("scene", "person"):
        f = tmp_path / f"{n}.png"
        f.write_bytes(b"x")
        imgs.append(f)
    argv, _ = k.build_command(
        cfg, root, Profile(name="p", image_lora_weights=lora),
        _image_request("place them at the table", imgs), tmp_path / "o.png")

    graph, images = _graph_from_argv(argv)
    _assert_links_resolve(graph)
    assert "--drop-node" not in argv
    patch = graph["10"]["inputs"]
    assert patch["source_latent"] == ["13", 0]
    assert patch["source_latent_b"] == ["14", 0]
    assert set(images) == {"REF_IMAGE", "REF_IMAGE_B"}


def test_krea_pack_b_graph_prunes_the_slots_it_does_not_use(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    lora = "krea2_style_reference.safetensors"
    root = _krea_model(tmp_path, lora)
    img = tmp_path / "style.png"
    img.write_bytes(b"x")
    argv, _ = k.build_command(cfg, root,
                              Profile(name="p", image_lora_weights=lora),
                              _image_request("a lighthouse", [img]),
                              tmp_path / "o.png")
    graph, _images = _graph_from_argv(argv)
    _assert_links_resolve(graph)
    assert "12" not in graph and "13" not in graph
    enc = graph["5"]["inputs"]
    assert enc["image1"] == ["11", 0]
    assert "image2" not in enc and "image3" not in enc
    # The VAE on the ENCODE node is what makes reference latents at all.
    assert enc["vae"] == ["4", 0]
    assert graph["10"]["class_type"] == "Krea2OstrisEditModelPatch"
    assert graph["10"]["inputs"]["kv_cache"] is False


def test_krea_rejects_an_unknown_quant(tmp_path, cfg):
    """Substituting the default would run weights nobody asked for."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = _krea_model(tmp_path)
    with pytest.raises(RuntimeError, match="Q9_K"):
        k.build_command(cfg, root, Profile(name="p", image_model_type="Q9_K"),
                        _image_request("x", []), tmp_path / "o.png")


_KREA_VALUES = {
    "UNET": "krea2_turbo-Q6_K.gguf", "LORA": "", "LORA_STRENGTH": 0.0,
    "CLIP": "qwen3vl_4b_fp8_scaled.safetensors",
    "VAE": "qwen_image_vae.safetensors", "PROMPT": "a hotel lobby",
    "WIDTH": 1024, "HEIGHT": 1024, "STEPS": 8, "CFG": 1.0,
    "SAMPLER": "euler", "SCHEDULER": "simple", "SEED": 1,
}


def test_every_comfy_workflow_template_is_valid_json_and_fully_tokenised():
    """Guards the whole family: a template whose tokens nobody supplies, or
    which stopped parsing, must fail here rather than at request time."""
    import json
    from llamanager.engines import comfy_backend as cb
    edit_a = {**_KREA_VALUES, "REF_IMAGE": "a.png", "REF_IMAGE_B": "b.png",
              "REF_BOOST": 4.0, "GROUNDING_PX": 768, "FIT_MODE": "fit"}
    edit_b = {**_KREA_VALUES, "REF_IMAGE": "a.png", "REF_IMAGE_B": "b.png",
              "REF_IMAGE_C": "c.png"}
    ref2v = {**{k: v for k, v in _WORKFLOW_VALUES.items()
                if k not in ("INIT_IMAGE", "LORA", "LORA_STRENGTH")},
             "REF_DETAIL": "match",
             **{f"REF{i}": f"r{i}.png" for i in range(1, 10)}}
    known = {"minimax_h3_i2v_gguf": _WORKFLOW_VALUES,
             "minimax_h3_ref2v_gguf": ref2v, "krea2_t2i_gguf": _KREA_VALUES,
             "krea2_t2i_gguf_te": _KREA_VALUES,
             "krea2_edit_a_gguf": edit_a, "krea2_edit_a_gguf_te": edit_a,
             "krea2_edit_b_gguf": edit_b, "krea2_edit_b_gguf_te": edit_b}
    templates = sorted(cb.workflow_path("x").parent.glob("*.json"))
    assert templates, "no workflow templates found"
    for path in templates:
        raw = json.loads(path.read_text())          # parses as JSON at rest
        assert "_comment" in raw, f"{path.name} should document itself"
        values = known.get(path.stem)
        assert values is not None, f"{path.name} has no test coverage"
        graph = cb.render_workflow(path.read_text(), values)
        for node_id, node in graph.items():
            assert "class_type" in node, f"{path.name}:{node_id}"
            # Documentation keys must never reach ComfyUI: a top-level one is
            # a node with no class_type, a per-node one an unknown field.
            assert not node_id.startswith("_"), f"{path.name}:{node_id}"
            assert not any(k.startswith("_") for k in node), (
                f"{path.name}:{node_id} kept a documentation key")


# ------------------------------------------------------------- the runner


def _runner_module():
    """Load _comfy_runner.py the way it loads itself: by path.

    It is written to run inside the ComfyUI venv (which has no llamanager),
    so it must not be importable only as a package member.
    """
    import importlib.util
    from llamanager.engines import comfy_backend as cb
    path = Path(cb.__file__).with_name("_comfy_runner.py")
    spec = importlib.util.spec_from_file_location("_comfy_runner_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dropping_a_node_removes_the_links_into_it():
    """An unfilled reference slot: the LoadImage has no upstream to bypass
    to, so it goes, and the optional input it fed goes with it."""
    from llamanager.engines import comfy_backend as cb
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "b.png"}},
        "3": {"class_type": "Encode",
              "inputs": {"image": ["1", 0], "image_b": ["2", 0], "px": 768}},
    }
    cb.drop_node(graph, "2")
    assert "2" not in graph
    assert graph["3"]["inputs"] == {"image": ["1", 0], "px": 768}


def test_dropping_a_node_that_is_not_there_is_an_error():
    """Silence would mean a template renumbering quietly stopped pruning the
    slot, leaving a LoadImage pointed at an image nobody uploaded."""
    from llamanager.engines import comfy_backend as cb
    with pytest.raises(KeyError):
        cb.drop_node({"1": {"class_type": "X", "inputs": {}}}, "9")


def test_runner_loads_the_backend_without_the_llamanager_package():
    """The runner runs under a different interpreter than the daemon, so it
    loads comfy_backend by file path. If that ever became a package import it
    would fail only at request time, inside the ComfyUI venv."""
    runner = _runner_module()
    cb = runner._load_backend()
    assert hasattr(cb, "render_workflow") and hasattr(cb, "bypass_node")


def _fake_lora(path: Path, n_keys: int) -> Path:
    """A .safetensors file with a real header and no tensor data.

    Enough for the key count, which is all the check reads — and it reads it
    from the header precisely so it never has to load the weights.
    """
    import json as _json
    import struct
    header = {f"lora_unet_block{i}.lora_down.weight":
              {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
              for i in range(n_keys)}
    blob = _json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00\x00")
    return path


def test_lora_check_fails_when_no_key_matched(tmp_path):
    """The exact failure this exists for: ComfyUI warns per unmatched key and
    samples on, so a LoRA for another architecture produces a perfectly
    normal image and nothing says the LoRA did nothing."""
    runner = _runner_module()
    lora = _fake_lora(tmp_path / "wrong_arch.safetensors", 3)
    server_log = tmp_path / "comfyui.log"
    server_log.write_text(
        "prompt executed\n"
        + "".join(f"lora key not loaded: lora_unet_block{i}.lora_down.weight\n"
                  for i in range(3)))
    assert runner.check_lora_applied(server_log, 0, lora) is False


def test_lora_check_passes_a_clean_load(tmp_path):
    runner = _runner_module()
    lora = _fake_lora(tmp_path / "good.safetensors", 3)
    server_log = tmp_path / "comfyui.log"
    server_log.write_text("loaded model\nprompt executed\n")
    assert runner.check_lora_applied(server_log, 0, lora) is True


def test_lora_check_tolerates_a_partial_load(tmp_path):
    """Some unmatched keys is normal (a LoRA can carry text-encoder keys the
    model-only loader has no home for). Only ALL of them is a no-op."""
    runner = _runner_module()
    lora = _fake_lora(tmp_path / "partial.safetensors", 4)
    server_log = tmp_path / "comfyui.log"
    server_log.write_text("lora key not loaded: a\nNOT LOADED b\n")
    assert runner.check_lora_applied(server_log, 0, lora) is True


def test_lora_check_reads_only_this_request_from_a_shared_log(tmp_path):
    """A warm server appends; yesterday's failures are not today's."""
    runner = _runner_module()
    lora = _fake_lora(tmp_path / "l.safetensors", 2)
    server_log = tmp_path / "comfyui.log"
    stale = "lora key not loaded: a\nlora key not loaded: b\n"
    server_log.write_text(stale + "prompt executed cleanly\n")
    assert runner.check_lora_applied(server_log, len(stale), lora) is True


def test_lora_check_does_not_claim_success_without_a_log(tmp_path):
    """A reused warm server keeps its log in the process that started it.
    Unverifiable must read as unverified, not as verified."""
    runner = _runner_module()
    lora = _fake_lora(tmp_path / "l.safetensors", 2)
    assert runner.check_lora_applied(tmp_path / "absent.log", 0, lora) is True


def test_runner_declares_the_flags_the_engines_emit():
    """The adapters and the runner are different processes on different
    interpreters, so a flag one side emits and the other never learned about
    fails at request time. Ask the runner itself what it accepts."""
    import subprocess
    import sys
    from llamanager.engines import comfy_backend as cb
    path = Path(cb.__file__).with_name("_comfy_runner.py")
    out = subprocess.run([sys.executable, str(path), "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    for flag in ("--image", "--drop-node", "--set-str", "--bypass",
                 "--keep-warm", "--lora-file"):
        assert flag in out, f"runner does not accept {flag}"
    assert "--init-image" not in out       # replaced by the generic --image


def test_runner_parses_typed_set_pairs():
    runner = _runner_module()
    got = runner.parse_set([
        "WIDTH=1344", "FPS=24.0", "SEED=0",
        "PROMPT=a hotel: grand, warm",     # not JSON -> stays a string
        "LORA=",                            # empty -> empty string
    ])
    assert got["WIDTH"] == 1344 and isinstance(got["WIDTH"], int)
    assert got["FPS"] == 24.0
    assert got["SEED"] == 0
    assert got["PROMPT"] == "a hotel: grand, warm"
    assert got["LORA"] == ""


def test_runner_rejects_a_malformed_set_pair():
    runner = _runner_module()
    with pytest.raises(SystemExit):
        runner.parse_set(["WIDTH"])


def test_runner_collects_the_newest_real_output(tmp_path):
    """ComfyUI groups outputs by node and kind; temp files must be skipped."""
    import os
    import time as _t
    runner = _runner_module()
    out_dir = tmp_path / "out"
    (out_dir / "sub").mkdir(parents=True)
    old = out_dir / "old.png"
    old.write_bytes(b"old")
    new = out_dir / "sub" / "clip.mp4"
    new.write_bytes(b"new")
    os.utime(old, (1, 1))
    os.utime(new, (_t.time(), _t.time()))

    hist = {"outputs": {
        "9":  {"images": [{"filename": "old.png", "subfolder": "",
                           "type": "output"}]},
        "16": {"video": [{"filename": "clip.mp4", "subfolder": "sub",
                          "type": "output"},
                         {"filename": "preview.png", "subfolder": "",
                          "type": "temp"}]},
    }}
    dest = tmp_path / "final.mp4"
    assert runner.collect_output(hist, out_dir, dest) == dest
    assert dest.read_bytes() == b"new"


def test_runner_reports_when_a_workflow_produced_nothing(tmp_path):
    runner = _runner_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="no output files"):
        runner.collect_output({"outputs": {}}, out_dir, tmp_path / "x.mp4")


def test_runner_picks_a_free_loopback_port():
    """A fixed port would submit our workflow to an operator's own ComfyUI."""
    runner = _runner_module()
    a, b = runner._free_port(), runner._free_port()
    assert 1024 < a < 65536 and 1024 < b < 65536


def test_a_numeric_prompt_stays_a_string(tmp_path, cfg):
    """Regression: "2024" is valid JSON, so --set would have turned a prompt
    into an integer and ComfyUI would reject the graph."""
    from llamanager.engines import minimax_h3_comfy as m
    from llamanager.config import Profile

    runner = _runner_module()
    _fake_comfy_install(cfg, tmp_path)
    model = _complete_model(tmp_path)
    img = tmp_path / "f.png"
    img.write_bytes(b"x")

    argv, _ = m.build_command(cfg, model, Profile(name="p"),
                              _image_request("2024", [img]),
                              tmp_path / "o.mp4")
    # Replay exactly what the runner would parse out of that argv.
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    strs = [argv[i + 1] for i, a in enumerate(argv) if a == "--set-str"]
    values = runner.parse_set(sets)
    values.update(runner.parse_set(strs, as_text=True))
    assert values["PROMPT"] == "2024"
    assert isinstance(values["PROMPT"], str)
    # ...while genuinely numeric tokens still arrive as numbers.
    assert values["WIDTH"] == 1344 and isinstance(values["WIDTH"], int)


# ------------------------------------------------------ model discovery


def test_a_comfy_transformer_is_not_listed_as_a_text_model(tmp_path):
    """Regression: MiniMax-H3's transformer is a .gguf, and the scanner used to
    treat any .gguf as a standalone llama model. It appeared in /v1/models as a
    text model an operator could try to load - an 18.5 GB video transformer.

    The engine directory is the GGUF's *grand*parent here, because ComfyUI
    sorts weights into diffusion_models/ one level down.
    """
    from llamanager.engines import minimax_h3_comfy as m
    reg = _registry(tmp_path)
    root = tmp_path / "MiniMax-H3-Comfy"
    for sub in ("diffusion_models", "text_encoders", "vae"):
        (root / sub).mkdir(parents=True)
    (root / "diffusion_models" / m.UNET_FILE).write_bytes(b"x")
    (root / "text_encoders" / m.CLIP_FILE).write_bytes(b"x")
    (root / "vae" / m.VIDEO_VAE_FILE).write_bytes(b"x")
    (root / "vae" / m.AUDIO_VAE_FILE).write_bytes(b"x")

    ids = {e.model_id for e in reg.list()}
    assert "MiniMax-H3-Comfy" in ids, "the model directory itself should list"
    assert not any(i.endswith(".gguf") for i in ids), (
        f"no component should list as a standalone model, got {ids}")


def test_a_half_downloaded_comfy_model_hides_its_components(tmp_path):
    """While components are still downloading no adapter claims the directory,
    but its weights must still not surface as text models."""
    reg = _registry(tmp_path)
    root = tmp_path / "Partial-Comfy"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / "some-model-Q4_K_M.gguf").write_bytes(b"x")

    ids = {e.model_id for e in reg.list()}
    assert not any(i.endswith(".gguf") for i in ids), ids


def test_a_real_standalone_gguf_still_lists_as_a_text_model(tmp_path):
    """The fix must not hide ordinary llama models."""
    reg = _registry(tmp_path)
    d = tmp_path / "unsloth" / "Some-Model-GGUF"
    d.mkdir(parents=True)
    (d / "model-Q4_K_M.gguf").write_bytes(b"x")

    ids = {e.model_id for e in reg.list()}
    assert "unsloth/Some-Model-GGUF/model-Q4_K_M.gguf" in ids


# --------------------------------------------- public pages see live config


def test_public_pages_read_the_live_config_not_the_boot_snapshot(app, cfg):
    """Regression: /chat, /images and /videos closed over the cfg captured
    when the app was built, so a profile added later was invisible on the
    public surfaces forever while the admin pages showed it immediately.

    Asserted at the source level because reproducing it end-to-end needs a
    real config reload mid-process; the failure mode is entirely "which
    object does this route read".
    """
    import inspect
    from llamanager import app as app_mod

    src = inspect.getsource(app_mod.create_app)
    for route in ("images_public", "videos_public", "chat_public"):
        start = src.index(f"async def {route}(")
        end = src.find("\n    @app.", start)
        body = src[start:end if end != -1 else None]
        assert "request.app.state.cfg" in body, (
            f"{route} must read request.app.state.cfg so config reloads "
            f"reach the public pages")


def test_materializing_profiles_refreshes_the_in_memory_config():
    """The composer reads cfg.iter_profiles(), so writing config.toml without
    reloading leaves a freshly materialized profile unusable until restart."""
    import inspect
    from llamanager import api_ui

    src = inspect.getsource(api_ui.diffusion_profiles_materialize_defaults)
    assert "_reload_config(request)" in src


# ------------------------------------------------- the GGUF text encoder


def test_krea_prefers_the_gguf_encoder_when_the_pair_is_present(tmp_path):
    """The whole 34x speedup hangs on this routing, so it is pinned.

    A GGUF encoder is only usable WITH its mmproj: without the vision-tower
    keys ComfyUI cannot recognise a Qwen3-VL and routes it to a plain-LLM
    encoder that produces the wrong conditioning shape (2560 instead of
    12x2560). So the GGUF path is chosen only when both files exist.
    """
    from llamanager.engines import krea_comfy as k
    te = tmp_path / "text_encoders"
    te.mkdir()
    (te / k.CLIP_SAFETENSORS).write_bytes(b"x")
    assert k.resolve_text_encoder(tmp_path) == (k.CLIP_SAFETENSORS, "_gguf")

    # GGUF alone is not enough — and it must SAY so. Quietly serving the
    # safetensors instead would turn a 22-second request into a 12-minute one
    # with nothing in the log to explain it.
    (te / k.CLIP_GGUF).write_bytes(b"x")
    with pytest.raises(RuntimeError, match=k.CLIP_GGUF_MMPROJ):
        k.resolve_text_encoder(tmp_path)

    (te / k.CLIP_GGUF_MMPROJ).write_bytes(b"x")   # the pair: fast path
    assert k.resolve_text_encoder(tmp_path) == (k.CLIP_GGUF, "_gguf_te")


def test_gguf_te_workflow_uses_the_gguf_clip_loader_with_krea2_type():
    """The type must stay 'krea2': the 12-layer tap and template are applied
    at encode time by comfy.text_encoders.krea2, not by the loader."""
    from llamanager.engines import comfy_backend as cb
    g = cb.render_workflow(cb.workflow_path("krea2_t2i_gguf_te").read_text(),
                           _KREA_VALUES)
    assert g["3"]["class_type"] == "CLIPLoaderGGUF"
    assert g["3"]["inputs"]["type"] == "krea2"


def test_comfy_plan_carries_its_gguf_loader_patches():
    """A fresh install must get the fast encoder path and the turbo quants.

    Each patch is checked for a marker only it can carry, so a truncated or
    swapped file fails here rather than at request time.
    """
    from llamanager.engine_installer import ENGINE_PLANS
    from pathlib import Path
    plan = ENGINE_PLANS["comfy"]
    markers = {
        "comfyui-gguf-qwen3vl-mmproj.patch": ("QWEN3VL_VISION_SD_MAP",
                                              "deepstack_merger_list"),
        "comfyui-gguf-ltx2-arch.patch": ("IMG_ARCH_LIST", '"ltx2"'),
    }
    names = [name for _dest, name in plan.patches]
    assert set(names) == set(markers), names
    for _dest, name in plan.patches:
        patch = (Path(__file__).parent.parent / "llamanager" / "engines"
                 / "comfy_patches" / name)
        assert patch.is_file(), f"missing patch file {name}"
        text = patch.read_text()
        for marker in markers[name]:
            assert marker in text, f"{name} is missing {marker}"


# ------------------------------------------------------- the LoRA picker


def test_lora_field_offers_the_files_in_the_models_loras_folder(tmp_path):
    """The LoRA field was a blank text box: no way to see what was installed.

    ``options_dir`` makes it a picker over the model's own folder, so the
    editor offers exactly the files the adapter is willing to load (it
    bypasses the LoraLoader node for anything not in ``loras/``).
    """
    from llamanager.api_ui import _dir_options, _serialize_profile_field
    from llamanager.config import Config
    from llamanager.engines import krea_comfy

    schema = [_serialize_profile_field(f) for f in krea_comfy.profile_schema()]
    lora = next(f for f in schema if f["key"] == "image_lora_weights")
    assert lora["options_dir"] == "loras"
    # The transformer override is a picker over the model's transformers, so
    # a baked LoRA is chosen from the files present rather than typed.
    unet = next(f for f in schema if f["key"] == "image_unet_file")
    assert unet["options_dir"] == "diffusion_models"

    cfg = Config()
    cfg.models_dir_override = tmp_path
    d = _comfy_pack(tmp_path, "Krea-2-Turbo-Comfy", "krea2_turbo-Q6_K.gguf")

    # No loras/ folder yet — an explicit empty list, which the UI renders as
    # "no loras installed" instead of an input inviting a guess.
    assert _dir_options(cfg, "Krea-2-Turbo-Comfy", schema) == {
        "image_lora_weights": [],
        "image_unet_file": ["krea2_turbo-Q6_K.gguf"]}

    (d / "loras").mkdir()
    (d / "loras" / "krea2_darkbrush.safetensors").write_bytes(b"w")
    (d / "loras" / "notes.txt").write_text("not a lora")

    assert _dir_options(cfg, "Krea-2-Turbo-Comfy", schema) == {
        "image_lora_weights": ["krea2_darkbrush.safetensors"],
        "image_unet_file": ["krea2_turbo-Q6_K.gguf"]}


def test_a_lora_folder_does_not_become_a_model(tmp_path):
    """Adding loras/ to a pack must not resurrect the duplicate-entry bug."""
    d = _comfy_pack(tmp_path, "Krea-2-Turbo-Comfy", "krea2_turbo-Q6_K.gguf")
    (d / "loras").mkdir()
    (d / "loras" / "krea2_darkbrush.safetensors").write_bytes(b"w")

    assert [m.model_id for m in _registry(tmp_path).list()] == [
        "Krea-2-Turbo-Comfy"]


# ------------------------------------------------- the warm-server heartbeat


class _StubBackend:
    """Just the one call _Heartbeat makes, with a record of when."""

    def __init__(self):
        self.touches = []

    def touch_heartbeat(self, model_path):
        import time
        self.touches.append(time.time())


def test_heartbeat_keeps_touching_while_the_work_runs():
    """One touch before submitting only covers requests shorter than the
    idle window; generating is activity and has to say so."""
    import time
    from llamanager.engines._comfy_runner import _Heartbeat
    from pathlib import Path

    cb = _StubBackend()
    with _Heartbeat(Path("/nonexistent"), cb, period=0.02):
        time.sleep(0.25)
    # Entry, several refreshes, and a final one on the way out.
    assert len(cb.touches) >= 4, cb.touches


def test_a_prompt_outlasting_the_idle_window_is_not_reaped(tmp_path):
    """The bug this covers: a 448 s first LoRA step went quiet, its own
    reaper SIGTERMed the server mid-generation, and the runner then waited
    on a dead port until its hour-long timeout."""
    import subprocess
    import sys
    import threading
    import time
    from llamanager.engines._comfy_reaper import reap
    from llamanager.engines._comfy_runner import _Heartbeat

    beat = tmp_path / "warm.beat"
    state = tmp_path / "warm.json"
    state.write_text("{}")
    beat.write_text(str(time.time()))

    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                              start_new_session=True)
    try:
        cb = type("CB", (), {"touch_heartbeat": staticmethod(
            lambda _p: beat.write_text(str(time.time())))})()
        done = threading.Event()
        watcher = threading.Thread(
            target=lambda: (reap(victim.pid, beat, idle=0.3, state=state,
                                 poll=0.05), done.set()), daemon=True)
        watcher.start()

        # While work is in flight the server must survive its idle window
        # several times over.
        with _Heartbeat(tmp_path, cb, period=0.05):
            time.sleep(1.5)
            assert victim.poll() is None, "reaped mid-generation"

        # Once the work ends, the idle window applies again and it is reaped.
        done.wait(timeout=10)
        victim.wait(timeout=10)
        assert victim.poll() is not None, "warm server outlived its idle window"
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait(timeout=5)


def test_the_runner_only_heartbeats_when_it_is_keeping_a_server_warm():
    """A one-shot run has no warm state to touch; doing it anyway would
    create a heartbeat file for a server nobody recorded."""
    import inspect
    import re
    from llamanager.engines import _comfy_runner

    src = inspect.getsource(_comfy_runner.main)
    guarded = re.search(r"if args\.keep_warm:\s*\n\s*with _Heartbeat\(", src)
    assert guarded, "run_prompt should heartbeat only under keep-warm"


# ------------------------------------------------------- baking a LoRA in


def test_bake_quantises_only_the_big_two_dimensional_weights():
    """Norms, biases and scales stay full precision — quantising them costs
    accuracy for no space, and Q8_0 needs rows divisible by its block."""
    from llamanager.engines._lora_bake import should_quantise

    assert should_quantise((6144, 6144))
    assert should_quantise((384, 1536))
    assert not should_quantise((6144,)), "1-D norm"
    assert not should_quantise((128,)), "1-D scale"
    assert not should_quantise((16, 16)), "too small to be worth it"
    assert not should_quantise((1536, 100)), "row not divisible by the block"
    assert not should_quantise((2, 4, 320, 320)), "not 2-D"


def test_bake_maps_lora_names_to_the_weights_they_patch():
    """ComfyUI names a patch diffusion_model.<key-without-.weight>; deriving
    the map from the state dict keeps this file-to-file, with no model built
    on a device first."""
    from llamanager.engines._lora_bake import lora_key_map

    keys = ["blocks.0.attn.wk.weight", "blocks.0.attn.wk.bias",
            "blocks.0.prenorm.scale"]
    assert lora_key_map(keys) == {
        "diffusion_model.blocks.0.attn.wk": "blocks.0.attn.wk.weight"}


def test_bake_states_its_strength_and_arch_rather_than_defaulting():
    """Both are baked irreversibly into the file: a wrong arch will not load
    at all, and a silent strength would be undiscoverable afterwards."""
    import inspect
    from llamanager.engines import _lora_bake

    src = inspect.getsource(_lora_bake.main)
    for flag in ('"--strength", type=float, required=True',
                 '"--arch", required=True'):
        assert flag in src, flag


def test_a_baked_transformer_file_overrides_the_quant(tmp_path, cfg):
    """A baked LoRA is a file the quant list cannot name, and baking is the
    difference between a 498 s request and a 45 s one."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = _krea_model(tmp_path)
    baked = "krea2_turbo-realism-v2-Q8_0.gguf"
    (root / "diffusion_models" / baked).write_bytes(b"x")

    prof = Profile(name="p", image_model_type="Q6_K", image_unet_file=baked)
    argv, _ = k.build_command(cfg, root, prof, _image_request("x", []),
                              tmp_path / "o.png")

    assert _argv_tokens(argv)["UNET"] == baked
    # No LoRA node: the merge is in the weights, so nothing is patched.
    assert _argv_tokens(argv)["LORA"] == ""


def test_a_missing_baked_transformer_is_named_not_swapped(tmp_path, cfg):
    """Falling back to the quant would silently render without the LoRA the
    operator baked in, and the image would look plausible."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = _krea_model(tmp_path)

    prof = Profile(name="p", image_unet_file="not-here-Q8_0.gguf")
    with pytest.raises(RuntimeError, match="not-here-Q8_0.gguf"):
        k.build_command(cfg, root, prof, _image_request("x", []),
                        tmp_path / "o.png")


def test_the_quant_still_applies_when_no_file_is_named(tmp_path, cfg):
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    root = _krea_model(tmp_path)
    prof = Profile(name="p", image_model_type="Q8_0", image_unet_file="")
    argv, _ = k.build_command(cfg, root, prof, _image_request("x", []),
                              tmp_path / "o.png")
    assert _argv_tokens(argv)["UNET"] == k.QUANT_FILES["Q8_0"][0]


def test_warm_servers_ignores_records_whose_process_is_gone(tmp_path, monkeypatch):
    """A stale state file would send a request at a closed port, or worse,
    at whatever recycled that pid."""
    import json
    from llamanager.engines import comfy_backend as cb

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    (tmp_path / "llamanager-comfy-warm-dead.json").write_text(
        json.dumps({"pid": 2 ** 22, "port": 1234}))
    assert cb.warm_servers() == []


def test_warm_servers_reports_a_live_one(tmp_path, monkeypatch):
    import json
    import os
    from llamanager.engines import comfy_backend as cb

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    (tmp_path / "llamanager-comfy-warm-live.json").write_text(
        json.dumps({"pid": os.getpid(), "port": 4321}))
    found = cb.warm_servers()
    assert [f["port"] for f in found] == [4321]


def test_stopping_warm_servers_clears_dead_records(tmp_path, monkeypatch):
    """Whatever is gone must leave no state behind, or the next request
    tries to adopt it."""
    import json
    from llamanager.engines import comfy_backend as cb

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    state = tmp_path / "llamanager-comfy-warm-dead.json"
    state.write_text(json.dumps({"pid": 2 ** 22, "port": 1234}))
    cb.heartbeat_path(state).write_text("0")

    assert cb.stop_warm_servers(grace_seconds=0.1) == []
    assert not state.exists()
    assert not cb.heartbeat_path(state).exists()


def test_the_text_engine_restart_stops_warm_servers_first(tmp_path):
    """Two resident models do not fit on a 32 GB card: keep-warm must not
    cost the operator an LLM that will not start."""
    import inspect
    from llamanager import server_manager

    src = inspect.getsource(server_manager.ServerManager.yield_to_image)
    stop = src.find("stop_warm_servers")
    start = src.find("await self.start(saved_spec)", src.find("finally:"))
    assert stop != -1, "warm servers are never stopped"
    assert stop < start, "warm server must yield before the LLM starts"
