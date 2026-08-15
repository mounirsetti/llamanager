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
              "models_dir": "", "target_dir": "", **form}
    resp = asyncio.run(api_ui.download_engine_model(
        _Req(app), engine="minimax_h3_comfy", _=None, **kwargs))
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
    """Collect the --set KEY=VALUE pairs out of a built argv."""
    out = {}
    for i, a in enumerate(argv):
        if a == "--set" and "=" in argv[i + 1]:
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
    assert "--init-image" in argv


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


def test_krea_rejects_reference_images(tmp_path, cfg):
    """img2img stays on the diffusers engine; say so instead of ignoring it."""
    from llamanager.engines import krea_comfy as k
    from llamanager.config import Profile

    _fake_comfy_install(cfg, tmp_path)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="text-to-image"):
        k.build_command(cfg, tmp_path, Profile(name="p"),
                        _image_request("x", [img]), tmp_path / "o.png")


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
    known = {"minimax_h3_i2v_gguf": _WORKFLOW_VALUES, "krea2_t2i_gguf": _KREA_VALUES}
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


def test_runner_loads_the_backend_without_the_llamanager_package():
    """The runner runs under a different interpreter than the daemon, so it
    loads comfy_backend by file path. If that ever became a package import it
    would fail only at request time, inside the ComfyUI venv."""
    runner = _runner_module()
    cb = runner._load_backend()
    assert hasattr(cb, "render_workflow") and hasattr(cb, "bypass_node")


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
