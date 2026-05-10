"""Static config loaded from ~/.llamanager/config.toml.

The schema mirrors spec §7. Anything missing falls back to defaults so
that a brand-new install with an empty config still boots.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_TOML = """\
[server]
# Bind to loopback by default. Bearer tokens are sent in cleartext over HTTP,
# so any non-loopback exposure SHOULD be fronted by a TLS-terminating reverse
# proxy (Caddy, nginx, Tailscale serve, etc.). To bind on a LAN interface,
# set bind = "0.0.0.0" explicitly and accept the documented risk.
bind = "127.0.0.1"
port = 7200
llama_server_binary = "llama-server"
llama_server_port = 7201
data_dir = "~/.llamanager"

[defaults]
model = "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"
profile = "qwen35-4b-default"
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

[profiles.qwen35-4b-default]
model = "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"
mmproj = ""
args = { ctx-size = 16384, host = "127.0.0.1", port = 7201, temp = 0.7, top-p = 0.8, top-k = 20, min-p = 0.0, presence-penalty = 1.5, alias = "qwen3.5-4b" }

[profiles.qwen35-4b-vision]
model = "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"
mmproj = "unsloth/Qwen3.5-4B-GGUF/mmproj-BF16.gguf"
args = { ctx-size = 8192, host = "127.0.0.1", port = 7201, temp = 0.7, top-p = 0.8, alias = "qwen3.5-4b-vl" }
"""


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


@dataclass
class Profile:
    name: str
    model: str
    mmproj: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    bind: str = "127.0.0.1"
    port: int = 7200
    llama_server_binary: str = "llama-server"
    llama_server_port: int = 7201
    data_dir: Path = field(default_factory=lambda: expand("~/.llamanager"))

    default_model: str = "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"
    default_profile: str = "qwen35-4b-default"
    default_origin_priority: int = 50

    max_restarts_in_window: int = 3
    window_seconds: int = 300
    success_run_seconds: int = 300

    max_disk_gb: int = 80
    hf_token_env: str = "HF_TOKEN"

    max_concurrent: int = 1
    max_queue_depth: int = 200

    profiles: dict[str, Profile] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # ---- derived paths ----
    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def runtime_path(self) -> Path:
        return self.data_dir / "runtime.json"

    @property
    def config_path(self) -> Path:
        return self.path or (self.data_dir / "config.toml")

    @property
    def session_secret_path(self) -> Path:
        return self.data_dir / ".session-secret"


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return tomllib.loads(DEFAULT_CONFIG_TOML)
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(path: Path | None = None) -> Config:
    """Load config from `path` (or default location). Missing keys use
    safe defaults; missing file uses bundled defaults entirely."""
    cfg_path = expand(str(path)) if path else expand("~/.llamanager/config.toml")
    raw = _load_toml(cfg_path)

    server = raw.get("server", {})
    defaults = raw.get("defaults", {})
    rp = raw.get("restart_policy", {})
    dl = raw.get("downloads", {})
    q = raw.get("queue", {})

    cfg = Config(
        bind=server.get("bind", "127.0.0.1"),
        port=int(server.get("port", 7200)),
        llama_server_binary=server.get("llama_server_binary", "llama-server"),
        llama_server_port=int(server.get("llama_server_port", 7201)),
        data_dir=expand(server.get("data_dir", "~/.llamanager")),
        default_model=defaults.get("model", "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"),
        default_profile=defaults.get("profile", "qwen35-4b-default"),
        default_origin_priority=int(defaults.get("origin_priority", 50)),
        max_restarts_in_window=int(rp.get("max_restarts_in_window", 3)),
        window_seconds=int(rp.get("window_seconds", 300)),
        success_run_seconds=int(rp.get("success_run_seconds", 300)),
        max_disk_gb=int(dl.get("max_disk_gb", 80)),
        hf_token_env=dl.get("hf_token_env", "HF_TOKEN"),
        max_concurrent=int(q.get("max_concurrent", 1)),
        max_queue_depth=int(q.get("max_queue_depth", 200)),
        raw=raw,
        path=cfg_path,
    )

    for name, body in (raw.get("profiles") or {}).items():
        cfg.profiles[name] = Profile(
            name=name,
            model=body.get("model", cfg.default_model),
            mmproj=body.get("mmproj", "") or "",
            args=dict(body.get("args") or {}),
        )

    # Ensure required dirs exist.
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    return cfg


def write_default_config(path: Path) -> Path:
    path = expand(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Profile CRUD — round-trip editing of config.toml via tomlkit
# ---------------------------------------------------------------------------

import re as _re

_PROFILE_NAME_RE = _re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def _load_tomlkit(path: Path):
    """Load a TOML file with tomlkit (preserves comments/formatting)."""
    import tomlkit
    if not path.exists():
        return tomlkit.parse(DEFAULT_CONFIG_TOML)
    return tomlkit.load(path.open("rb"))


def _save_tomlkit(path: Path, doc) -> None:
    import tomlkit
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _validate_profile_name(name: str) -> str:
    name = name.strip().lower()
    if not _PROFILE_NAME_RE.match(name):
        raise ValueError(
            f"profile name must be lowercase alphanumeric with hyphens: {name!r}"
        )
    return name


def save_profile(cfg_path: Path, name: str, model: str,
                 mmproj: str, args: dict[str, Any]) -> None:
    """Create or overwrite a profile in config.toml."""
    import tomlkit
    name = _validate_profile_name(name)
    doc = _load_tomlkit(cfg_path)
    if "profiles" not in doc:
        doc.add("profiles", tomlkit.table(is_super_table=True))
    prof = tomlkit.table()
    prof.add("model", model)
    if mmproj:
        prof.add("mmproj", mmproj)
    if args:
        prof.add("args", args)
    doc["profiles"][name] = prof
    _save_tomlkit(cfg_path, doc)


def delete_profile(cfg_path: Path, name: str) -> None:
    """Remove a profile from config.toml."""
    doc = _load_tomlkit(cfg_path)
    profiles = doc.get("profiles")
    if profiles and name in profiles:
        del profiles[name]
        _save_tomlkit(cfg_path, doc)


def update_defaults(cfg_path: Path, *,
                    default_model: str | None = None,
                    default_profile: str | None = None) -> None:
    """Update the [defaults] section in config.toml."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "defaults" not in doc:
        doc.add("defaults", tomlkit.table())
    if default_model is not None:
        doc["defaults"]["model"] = default_model
    if default_profile is not None:
        doc["defaults"]["profile"] = default_profile
    _save_tomlkit(cfg_path, doc)
