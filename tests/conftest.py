"""Pytest fixtures: build a Config + app rooted in a tmpdir so we never
touch the user's real ~/.llamanager directory."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "llamanager"
    d.mkdir(parents=True)
    (d / "models").mkdir()
    (d / "logs").mkdir()
    return d


@pytest.fixture
def cfg(data_dir: Path):
    from llamanager.config import Config
    c = Config(data_dir=data_dir)
    return c


@pytest.fixture
def app(data_dir: Path):
    """Build a FastAPI app with a fresh data dir.

    We patch the home dir lookup to point at the tmp dir so config and
    DB land there and don't pollute ~/.llamanager.
    """
    cfg_path = data_dir / "config.toml"
    # Use bundled defaults but override data_dir.
    cfg_path.write_text(f"""
[server]
bind = "127.0.0.1"
port = 7200
llama_server_binary = "true"
llama_server_port = 7201
data_dir = "{data_dir.as_posix()}"

[defaults]
model = "test/model.gguf"
profile = "test"
origin_priority = 50

[restart_policy]
max_restarts_in_window = 3
window_seconds = 300
success_run_seconds = 300

[downloads]
max_disk_gb = 80
hf_token_env = "HF_TOKEN"

[queue]
max_concurrent = 1
max_queue_depth = 200

[profiles.test]
model = "test/model.gguf"
mmproj = ""
args = {{ ctx-size = 1024 }}
""", encoding="utf-8")
    from llamanager.app import create_app
    return create_app(cfg_path, print_bootstrap=False)
