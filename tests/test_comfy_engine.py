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
