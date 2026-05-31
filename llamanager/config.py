"""Static config loaded from ~/.llamanager/config.toml.

Schema:

  [models."repo/file.gguf"]
  default_profile = "fast"

  [models."repo/file.gguf".profiles.fast]
  mmproj = ""
  ctx_size = 4096
  vram_limit_gb = 8.0
  ram_spill_policy = "limited"   # "default" | "unlimited" | "limited" | "none"
  ram_spill_limit_gb = 4.0
  [models."repo/file.gguf".profiles.fast.args]
  temp = 0.7

  [default_args.llama]
  temp = 0.7

  [default_args.mlx]
  temp = 0.6

Profiles always live under a model. Engine-keyed `default_args` provide
minimum defaults that apply to any model of a given engine when its
profile (or the request) doesn't override them.
"""
from __future__ import annotations

import logging
import os
import re as _re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)


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

# Exclusive mode: when llamanager is running, prevent foreign
# llama-server / image-engine workers from competing for the GPU.
#   off        — disabled
#   warn       — scan + log, kill nothing
#   exclusive  — kill llama-server binaries and orphaned image workers
#   aggressive — also kill foreign ML runtimes (ComfyUI, vLLM, Ollama, ...)
exclusive_mode = "off"
exclusive_grace_seconds = 5
exclusive_heartbeat_seconds = 120

# Multi-slot LLM (beta). When false, llamanager runs one llama-server at a
# time (legacy behaviour). When true, multiple slots can each hold a
# different model on its own port; routing is by `model` id in the request.
# Mutually exclusive with `exclusive_mode` — enabling slots force-disables
# exclusive sweeps.
multi_slot_enabled = false
multi_slot_base_port = 7201
multi_slot_max = 4


[defaults]
model = ""
origin_priority = 50

[default_args.llama]
ctx_size = 4096
temp = 0.7

[default_args.mlx]
temp = 0.7

[restart_policy]
max_restarts_in_window = 3
window_seconds = 300
success_run_seconds = 300

[downloads]
# 0 = no cap (only the actual partition free-space check applies).
# Set to a non-zero GB value to cap the cumulative size of the models dir.
max_disk_gb = 0
hf_token_env = "HF_TOKEN"

[queue]
max_concurrent = 1
max_queue_depth = 200
max_wait_s = 300
queue_timeout_s = 300

[image]
# Image-engine paths. Both stacks have hardware-pinned dependency chains
# (see docs/hidream.md and docs/flux2.md) and are NOT auto-installed —
# set these to point at an existing install.
# hidream_python    = "/path/to/.venv-hidream/bin/python"
# hidream_repo      = "/path/to/HiDream-O1-Image"
# flux2_sd_cli      = "/path/to/sd-cli"
# flux2_device_index = 1     # GGML_VK_VISIBLE_DEVICES for the AMD card
# images_dir        = "~/.llamanager/images"
max_disk_gb = 10               # cap for the on-disk image gallery

[coexistence]
# Default: when an image task is dispatched and a text engine is running,
# snapshot the text spec, stop it, run the image task, then restart text.
# This keeps the single-slot dashboard invariant.
unload_text_on_arrival = true
restart_text_after_image = true
# Concurrent mode: keep both loaded at once. Risks VRAM OOM on cards with
# less than ~48 GiB. Recommended only when you know both fit.
allow_concurrent = false
# Multi-slot only: when true AND `[server].multi_slot_enabled = true`,
# dispatching an image task does NOT unload the LLM slots. VRAM headroom
# is the operator's responsibility.
allow_diffusion_with_slots = false

[auto_update]
# Per-engine "auto-update when idle". When an engine is enabled below,
# llamanager checks upstream for a newer build on a fixed cadence and, once
# the daemon has been idle (no in-flight or pending requests) for
# `idle_seconds`, applies the update automatically. Off for every engine by
# default — opt in per engine from the UI switch or `llamanager setup
# auto-update <engine> on`.
idle_seconds = 300              # quiet window required before an update fires
check_interval_seconds = 21600  # how often each enabled engine checks upstream (6h)

# Engine keys: a llama.cpp/MLX variant id ("llama.cpp-cuda", "atomic-vulkan",
# "mlx-apple-silicon"), a diffusion engine name ("hidream", "z_image"), or the
# reserved key "llamanager" for the daemon's own self-update.
[auto_update.engines]
# "llama.cpp-cuda" = true
# "z_image"        = false
# "llamanager"     = false
"""


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

# Engine name → family. Text engines run as long-running HTTP servers
# managed by ServerManager. Image engines are one-shot CLI invocations
# managed by ImageTaskRunner.
ENGINE_FAMILY: dict[str, str] = {
    "llama":   "text",
    "mlx":     "text",
    "hidream": "image",
    "flux2":   "image",
    "z_image": "image",
}

# Text engines that are still managed by the existing HTTP-server path.
TEXT_ENGINES = frozenset(e for e, f in ENGINE_FAMILY.items() if f == "text")
IMAGE_ENGINES = frozenset(e for e, f in ENGINE_FAMILY.items() if f == "image")


def engine_family(engine: str) -> str:
    """Return ``"text"`` or ``"image"`` (defaults to ``"text"`` for unknown
    engines so legacy configs keep working)."""
    return ENGINE_FAMILY.get(engine, "text")


def _looks_like_hidream(d: Path) -> bool:
    """HiDream-O1-Image checkpoint directory shape:
    ``tokenizer_config.json`` + ``preprocessor_config.json`` + safetensors
    shards. See docs/hidream.md."""
    return (
        (d / "tokenizer_config.json").exists()
        and (d / "preprocessor_config.json").exists()
        and any(d.glob("*.safetensors"))
    )


def _looks_like_flux2(d: Path) -> bool:
    """FLUX 2 / sd.cpp checkpoint directory shape: a FLUX diffusion GGUF
    plus a VAE safetensors and (typically) a text-encoder GGUF in the
    same folder. See docs/flux2.md."""
    names = {p.name.lower() for p in d.iterdir() if p.is_file()} \
            if d.is_dir() else set()
    has_diffusion = any("flux" in n and n.endswith(".gguf") for n in names)
    has_vae = any(n == "ae.safetensors"
                  or n.endswith("flux2-vae.safetensors")
                  or (n.startswith("ae") and n.endswith(".safetensors"))
                  for n in names)
    return has_diffusion and has_vae


def _looks_like_z_image(d: Path) -> bool:
    """Z-Image / Z-Anime checkpoint directory shape: a Diffusers
    ``model_index.json`` whose ``_class_name`` is ``ZImagePipeline``.
    Both Tongyi-MAI/Z-Image and SeeSee21/Z-Anime ship this layout.

    Reading the JSON is the *only* way to disambiguate Z-Image from
    other Diffusers pipelines that share the same on-disk shape (FLUX,
    SD3, etc.). The file is small (~500 bytes) so the read is cheap.
    """
    if not d.is_dir():
        return False
    mi = d / "model_index.json"
    if not mi.is_file():
        return False
    try:
        import json as _json
        data = _json.loads(mi.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (data.get("_class_name") or "").strip() == "ZImagePipeline"


def detect_engine_for_path(model_path: Path) -> str:
    """Return the engine name for a model on disk.

    Order matters: image engines are checked before the generic MLX
    directory shape because a HiDream dir technically also contains
    safetensors + (some) tokenizer config.
    """
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return "llama"
    if model_path.is_dir():
        if _looks_like_z_image(model_path):
            return "z_image"
        if _looks_like_hidream(model_path):
            return "hidream"
        if _looks_like_flux2(model_path):
            return "flux2"
        # Z-Anime ships its Diffusers pipeline in a ``diffusers/``
        # subfolder; check one level down too so the parent folder is
        # surfaced as a Z-Image model.
        if _looks_like_z_image(model_path / "diffusers"):
            return "z_image"
        if (model_path / "config.json").exists():
            if any(model_path.glob("*.safetensors")) or any(model_path.glob("*.npz")):
                return "mlx"
    return "llama"


def detect_engine_for_id(model_id: str, models_dir: Path) -> str:
    """Same as ``detect_engine_for_path`` but takes a logical model id."""
    if model_id.lower().endswith(".gguf"):
        return "llama"
    p = models_dir / model_id
    return detect_engine_for_path(p)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

VALID_RAM_SPILL_POLICIES = ("default", "unlimited", "limited", "none")
VALID_THINKING = ("", "on", "off")
VALID_EXCLUSIVE_MODES = ("off", "warn", "exclusive", "aggressive")


@dataclass
class Profile:
    """A named launch preset for a model.

    For text-family engines: ``ctx_size`` / ``vram_limit_gb`` /
    ``ram_spill_policy`` / ``ram_spill_limit_gb`` are the basic UI knobs.
    They map to engine flags (mostly ``--ctx-size`` + a heuristic
    ``--n-gpu-layers``).

    For image-family engines: ``image_model_type`` / ``image_steps`` /
    ``image_guidance`` / ``image_size`` / ``image_seed`` are surfaced in
    the Images page UI. Each image engine adapter decides which of these
    are meaningful for its CLI.

    ``args`` is the raw advanced override map for either family; it wins
    over basic-derived args during launch.
    """
    name: str
    # text-family knobs
    mmproj: str = ""
    ctx_size: int | None = None
    vram_limit_gb: float | None = None
    ram_spill_policy: str = "default"
    ram_spill_limit_gb: float | None = None
    # image-family knobs (ignored by text engines)
    image_model_type: str = ""        # e.g. "dev" | "full" (HiDream)
    image_steps: int | None = None
    image_guidance: float | None = None
    image_size: str = ""              # "WxH" — engine adapters validate/snap
    image_seed: int | None = None
    # Reference-image knobs. ``image_editing_scheduler`` is HiDream's
    # --editing_scheduler flag ("flow_match" | "flash"); only meaningful
    # when exactly one reference image is passed with --model_type dev.
    # ``image_strength`` is Flux2's img2img --strength (0..1); ignored by
    # HiDream. Both can be overridden per-request.
    image_editing_scheduler: str = ""
    image_strength: float | None = None
    # Chat reasoning default. "on" / "off" inject
    # ``chat_template_kwargs.enable_thinking`` into the upstream body for
    # /v1/chat/completions; "" leaves the model/template default alone.
    thinking: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Config block for one model id (the on-disk model path is implicit)."""
    id: str
    profiles: dict[str, Profile] = field(default_factory=dict)
    default_profile: str = ""


@dataclass
class Config:
    bind: str = "127.0.0.1"
    port: int = 7200
    llama_server_binary: str = "llama-server"
    # Default engine used when a model's engine can't be auto-detected
    # (e.g. when only an id is available without a file on disk).
    llama_server_engine: str = "llama"
    llama_server_port: int = 7201
    data_dir: Path = field(default_factory=lambda: expand("~/.llamanager"))

    default_model: str = ""
    # Image-family default — used by /v1/images/generations when the
    # request body omits ``model``. Profile defaults to the model's own
    # default_profile when blank.
    default_image_model: str = ""
    default_image_profile: str = ""
    default_origin_priority: int = 50
    autolaunch: bool = False

    max_restarts_in_window: int = 3
    window_seconds: int = 300
    success_run_seconds: int = 300

    # 0 = no cap; the actual free-disk check in registry.py still guards
    # against running out of space. A non-zero value caps the cumulative
    # size of the models directory.
    max_disk_gb: int = 0
    hf_token_env: str = "HF_TOKEN"

    max_concurrent: int = 1
    max_queue_depth: int = 200
    max_wait_s: int = 300
    queue_timeout_s: int = 300

    # New: profiles nest under their parent model.
    models: dict[str, ModelConfig] = field(default_factory=dict)
    # Engine-keyed minimum defaults ("llama" / "mlx" / "hidream" / "flux2"
    # → args dict).
    default_args: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Set by the operator via /ui/models/set-dir or [server].models_dir in
    # config.toml. When None, models_dir falls back to data_dir/models.
    models_dir_override: Path | None = None

    # ---- image engines ----
    # Per-engine paths the operator sets via /ui/setup (no auto-install:
    # both stacks have hardware-pinned dependency chains documented in
    # docs/hidream.md and docs/flux2.md).
    hidream_python: str = ""         # path to the .venv-hidream Python
    hidream_repo: str = ""           # path to the HiDream-O1-Image source folder
    # Pin the AMD ROCm release the hidream auto-installer targets. Empty
    # string → engine_installer.AMD_ROCM_REL (curated default). Set to
    # e.g. "rocm-rel-7.2.1" to override (matches a directory under
    # https://repo.radeon.com/rocm/manylinux/). Operators on AMD who
    # need stability over freshness can pin this; otherwise the
    # scraper picks the newest paired set in the curated release.
    hidream_target_rocm_release: str = ""
    flux2_sd_cli: str = ""           # path to sd-cli (or sd-cli.exe)
    flux2_device_index: int | None = None  # GGML_VK_VISIBLE_DEVICES value
    # Z-Image only needs a Python interpreter — the runner script ships
    # with llamanager (engines/_z_image_runner.py), so there's no
    # separate source folder to clone.
    z_image_python: str = ""
    # Per-engine diffusers version override (engine name → version), set when
    # the operator installs a specific diffusers version from the UI/CLI
    # version picker. While set it overrides engine_installer.DIFFUSERS_PIN as
    # the auto-update target so a deliberate downgrade isn't re-bumped. Stored
    # under [image].diffusers_version.<engine>.
    image_diffusers_version: dict[str, str] = field(default_factory=dict)

    # ---- coexistence policy ----
    # Default: when an image task arrives, snapshot the running text spec,
    # stop the text engine, run the image task, restart text. Set
    # ``unload_text_on_arrival = false`` to keep text running (concurrent
    # mode — risks VRAM OOM).
    unload_text_on_arrival: bool = True
    restart_text_after_image: bool = True
    allow_concurrent: bool = False
    # Multi-slot only: when true AND ``multi_slot_enabled`` is true, image
    # dispatch leaves LLM slots loaded. See server_pool.ServerPool and
    # image_runner.ImageTaskRunner.run for the gating logic.
    allow_diffusion_with_slots: bool = False

    # Where image outputs land. None → data_dir/images/.
    images_dir_override: Path | None = None
    images_max_disk_gb: int = 10

    # ---- exclusive mode ----
    # See [server] section in DEFAULT_CONFIG_TOML for the mode catalog.
    # Persisted under [server] so the surface stays small.
    exclusive_mode: str = "off"
    exclusive_grace_seconds: float = 5.0
    exclusive_heartbeat_seconds: int = 120

    # ---- multi-slot LLM (beta) ----
    # Master switch + capacity caps. See server_pool.ServerPool.
    multi_slot_enabled: bool = False
    multi_slot_base_port: int = 7201
    multi_slot_max: int = 4

    # ---- auto-update when idle ----
    # See [auto_update] in DEFAULT_CONFIG_TOML and auto_update.AutoUpdater.
    # ``auto_update_engines`` maps an engine key (llama variant id, diffusion
    # engine name, or "llamanager") → enabled bool. The two *_seconds knobs
    # are optional tuning parameters with documented defaults.
    auto_update_idle_seconds: int = 300
    auto_update_check_interval_seconds: int = 21600
    auto_update_engines: dict[str, bool] = field(default_factory=dict)

    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # ---- derived paths ----
    @property
    def models_dir(self) -> Path:
        if self.models_dir_override is not None:
            return self.models_dir_override
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def images_dir(self) -> Path:
        if self.images_dir_override is not None:
            return self.images_dir_override
        return self.data_dir / "images"

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

    # ---- profile accessors ----
    def get_model(self, model_id: str) -> ModelConfig | None:
        return self.models.get(model_id)

    def ensure_model(self, model_id: str) -> ModelConfig:
        m = self.models.get(model_id)
        if m is None:
            m = ModelConfig(id=model_id)
            self.models[model_id] = m
        return m

    def get_profile(self, model_id: str, name: str) -> Profile | None:
        m = self.models.get(model_id)
        return m.profiles.get(name) if m else None

    def iter_profiles(self) -> Iterator[tuple[str, Profile]]:
        """Yield (model_id, Profile) for every configured profile."""
        for mid, m in self.models.items():
            for p in m.profiles.values():
                yield mid, p

    def engine_for(self, model_id: str) -> str:
        """Detect the engine used by ``model_id``, defaulting to the
        configured global engine."""
        return detect_engine_for_id(model_id, self.models_dir) or self.llama_server_engine


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return tomllib.loads(DEFAULT_CONFIG_TOML)
    with path.open("rb") as f:
        return tomllib.load(f)


def _coerce_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_profile(name: str, body: dict[str, Any]) -> Profile:
    policy = str(body.get("ram_spill_policy", "default") or "default")
    if policy not in VALID_RAM_SPILL_POLICIES:
        policy = "default"
    thinking = str(body.get("thinking", "") or "").strip().lower()
    if thinking not in VALID_THINKING:
        thinking = ""
    return Profile(
        name=name,
        mmproj=str(body.get("mmproj", "") or ""),
        ctx_size=_coerce_int(body.get("ctx_size")),
        vram_limit_gb=_coerce_float(body.get("vram_limit_gb")),
        ram_spill_policy=policy,
        ram_spill_limit_gb=_coerce_float(body.get("ram_spill_limit_gb")),
        image_model_type=str(body.get("image_model_type", "") or ""),
        image_steps=_coerce_int(body.get("image_steps")),
        image_guidance=_coerce_float(body.get("image_guidance")),
        image_size=str(body.get("image_size", "") or ""),
        image_seed=_coerce_int(body.get("image_seed")),
        image_editing_scheduler=str(body.get("image_editing_scheduler", "") or ""),
        image_strength=_coerce_float(body.get("image_strength")),
        thinking=thinking,
        args=dict(body.get("args") or {}),
    )


def load_config(path: Path | None = None) -> Config:
    """Load config from ``path`` (or default location). Missing keys use
    safe defaults; missing file uses bundled defaults entirely. Migrates
    legacy flat ``[profiles.X]`` layouts on the fly."""
    cfg_path = expand(str(path)) if path else expand("~/.llamanager/config.toml")
    raw = _load_toml(cfg_path)

    server = raw.get("server", {})
    defaults = raw.get("defaults", {})
    rp = raw.get("restart_policy", {})
    dl = raw.get("downloads", {})
    q = raw.get("queue", {})
    image_cfg = raw.get("image", {}) if isinstance(raw.get("image"), dict) else {}
    coex_cfg = raw.get("coexistence", {}) if isinstance(raw.get("coexistence"), dict) else {}

    cfg = Config(
        bind=server.get("bind", "127.0.0.1"),
        port=int(server.get("port", 7200)),
        llama_server_binary=server.get("llama_server_binary", "llama-server"),
        llama_server_engine=str(server.get("llama_server_engine", "llama")),
        llama_server_port=int(server.get("llama_server_port", 7201)),
        data_dir=expand(server.get("data_dir", "~/.llamanager")),
        default_model=defaults.get("model", ""),
        default_image_model=defaults.get("image_model", ""),
        default_image_profile=defaults.get("image_profile", ""),
        default_origin_priority=int(defaults.get("origin_priority", 50)),
        autolaunch=bool(defaults.get("autolaunch", False)),
        max_restarts_in_window=int(rp.get("max_restarts_in_window", 3)),
        window_seconds=int(rp.get("window_seconds", 300)),
        success_run_seconds=int(rp.get("success_run_seconds", 300)),
        max_disk_gb=int(dl.get("max_disk_gb", 0)),
        hf_token_env=dl.get("hf_token_env", "HF_TOKEN"),
        max_concurrent=int(q.get("max_concurrent", 1)),
        max_queue_depth=int(q.get("max_queue_depth", 200)),
        max_wait_s=int(q.get("max_wait_s", 300)),
        queue_timeout_s=int(q.get("queue_timeout_s", 300)),
        hidream_python=str(image_cfg.get("hidream_python", "") or ""),
        hidream_repo=str(image_cfg.get("hidream_repo", "") or ""),
        hidream_target_rocm_release=str(
            image_cfg.get("hidream_target_rocm_release", "") or ""),
        flux2_sd_cli=str(image_cfg.get("flux2_sd_cli", "") or ""),
        flux2_device_index=_coerce_int(image_cfg.get("flux2_device_index")),
        z_image_python=str(image_cfg.get("z_image_python", "") or ""),
        images_max_disk_gb=int(image_cfg.get("max_disk_gb", 10)),
        unload_text_on_arrival=bool(coex_cfg.get("unload_text_on_arrival", True)),
        restart_text_after_image=bool(coex_cfg.get("restart_text_after_image", True)),
        allow_concurrent=bool(coex_cfg.get("allow_concurrent", False)),
        allow_diffusion_with_slots=bool(
            coex_cfg.get("allow_diffusion_with_slots", False)
        ),
        exclusive_mode=(
            str(server.get("exclusive_mode", "off") or "off").strip().lower()
            if str(server.get("exclusive_mode", "off") or "off").strip().lower()
            in VALID_EXCLUSIVE_MODES else "off"
        ),
        exclusive_grace_seconds=float(server.get("exclusive_grace_seconds", 5.0) or 5.0),
        exclusive_heartbeat_seconds=int(server.get("exclusive_heartbeat_seconds", 120) or 120),
        multi_slot_enabled=bool(server.get("multi_slot_enabled", False)),
        multi_slot_base_port=int(server.get("multi_slot_base_port", 7201) or 7201),
        multi_slot_max=max(1, int(server.get("multi_slot_max", 4) or 4)),
        raw=raw,
        path=cfg_path,
    )
    if "images_dir" in image_cfg:
        cfg.images_dir_override = expand(str(image_cfg["images_dir"]))

    dv = image_cfg.get("diffusers_version")
    if isinstance(dv, dict):
        cfg.image_diffusers_version = {
            str(k): str(v) for k, v in dv.items() if v
        }

    if "models_dir" in server:
        cfg.models_dir_override = expand(str(server["models_dir"]))

    # ---- new-format models section ----
    for model_id, body in (raw.get("models") or {}).items():
        if not isinstance(body, dict):
            continue
        m = ModelConfig(id=model_id)
        m.default_profile = str(body.get("default_profile", "") or "")
        for pname, pbody in (body.get("profiles") or {}).items():
            if isinstance(pbody, dict):
                m.profiles[pname] = _parse_profile(pname, pbody)
        cfg.models[model_id] = m

    # ---- engine-keyed default_args ----
    da = raw.get("default_args") or {}
    if isinstance(da, dict):
        for engine, body in da.items():
            if isinstance(body, dict):
                cfg.default_args[str(engine)] = dict(body)

    # ---- auto-update when idle ----
    au = raw.get("auto_update") or {}
    if isinstance(au, dict):
        cfg.auto_update_idle_seconds = int(au.get("idle_seconds", 300) or 300)
        cfg.auto_update_check_interval_seconds = int(
            au.get("check_interval_seconds", 21600) or 21600)
        engines = au.get("engines") or {}
        if isinstance(engines, dict):
            cfg.auto_update_engines = {
                str(k): bool(v) for k, v in engines.items()
            }

    # ---- legacy migration ----
    migrated = _migrate_legacy_inplace(raw, cfg)

    # If anything got migrated, persist immediately so the file reflects
    # the new layout (and the next load is a no-op).
    if migrated:
        try:
            _rewrite_after_migration(cfg_path, cfg)
        except Exception:
            log.exception("legacy config migration save failed for %s", cfg_path)

    # Ensure required dirs exist.
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    return cfg


def _migrate_legacy_inplace(raw: dict[str, Any], cfg: Config) -> bool:
    """Migrate old flat [profiles.X] / [model_defaults] / [defaults].profile
    into the new nested cfg.models + cfg.default_args layout. Returns True
    if anything was migrated (signalling a file rewrite is wanted)."""
    legacy_profiles = raw.get("profiles") or {}
    legacy_model_defaults = raw.get("model_defaults") or {}
    legacy_default_profile = (raw.get("defaults") or {}).get("profile") or ""

    if not legacy_profiles and not legacy_model_defaults and not legacy_default_profile:
        return False

    engine_fallback = cfg.llama_server_engine or "llama"
    # Per-profile migration.
    for name, body in legacy_profiles.items():
        if not isinstance(body, dict):
            continue
        mid = body.get("model", "") or ""
        # Translate the args dict, splitting structured fields out.
        args = dict(body.get("args") or {})
        ctx_size = _coerce_int(args.pop("ctx-size", None) or args.pop("ctx_size", None))
        prof = Profile(
            name=name,
            mmproj=str(body.get("mmproj", "") or ""),
            ctx_size=ctx_size,
            vram_limit_gb=None,
            ram_spill_policy="default",
            ram_spill_limit_gb=None,
            args=args,
        )
        if mid:
            m = cfg.ensure_model(mid)
            m.profiles.setdefault(name, prof)
        else:
            # Empty-model profile: its args (and ctx) merge into the engine
            # default bucket. Conservative: use the global engine fallback
            # since we can't infer per-model intent.
            bucket = cfg.default_args.setdefault(engine_fallback, {})
            if ctx_size is not None and "ctx_size" not in bucket and "ctx-size" not in bucket:
                bucket["ctx_size"] = ctx_size
            for k, v in args.items():
                bucket.setdefault(k, v)

    # Per-model default pointer.
    for mid, pname in legacy_model_defaults.items():
        if not isinstance(pname, str) or not pname:
            continue
        m = cfg.ensure_model(mid)
        if not m.default_profile:
            m.default_profile = pname

    # legacy_default_profile pointing at a model-bound profile is captured
    # by [model_defaults] in well-formed configs; if not, find it and lift.
    if legacy_default_profile:
        body = legacy_profiles.get(legacy_default_profile)
        if isinstance(body, dict):
            mid = body.get("model", "") or ""
            if mid:
                m = cfg.ensure_model(mid)
                if not m.default_profile:
                    m.default_profile = legacy_default_profile

    return True


def _rewrite_after_migration(cfg_path: Path, cfg: Config) -> None:
    """Re-serialise cfg into the new TOML layout and drop legacy sections."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    # Drop legacy sections.
    for legacy in ("profiles", "model_defaults"):
        if legacy in doc:
            del doc[legacy]
    defaults = doc.get("defaults")
    if isinstance(defaults, dict) and "profile" in defaults:
        del defaults["profile"]

    # Rewrite models section.
    if "models" in doc:
        del doc["models"]
    if cfg.models:
        models_tbl = tomlkit.table(is_super_table=True)
        for mid, m in cfg.models.items():
            mtbl = tomlkit.table()
            if m.default_profile:
                mtbl.add("default_profile", m.default_profile)
            if m.profiles:
                profs = tomlkit.table(is_super_table=True)
                for pname, prof in m.profiles.items():
                    profs[pname] = _profile_to_tomlkit(prof)
                mtbl.add("profiles", profs)
            models_tbl[mid] = mtbl
        doc["models"] = models_tbl

    # Rewrite default_args.
    if "default_args" in doc:
        del doc["default_args"]
    if cfg.default_args:
        da_tbl = tomlkit.table(is_super_table=True)
        for engine, args in cfg.default_args.items():
            da_tbl[engine] = _dict_to_tomlkit(args)
        doc["default_args"] = da_tbl

    _save_tomlkit(cfg_path, doc)


def write_default_config(path: Path) -> Path:
    path = expand(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Profile / model CRUD — round-trip editing of config.toml via tomlkit
# ---------------------------------------------------------------------------

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
    # write_bytes avoids the implicit \n → \r\n translation that text-mode
    # writes perform on Windows; newer tomlkit refuses CRLF on re-read.
    path.write_bytes(tomlkit.dumps(doc).encode("utf-8"))


def _validate_profile_name(name: str) -> str:
    name = name.strip().lower()
    if not _PROFILE_NAME_RE.match(name):
        raise ValueError(
            f"invalid profile name {name!r}: only lowercase letters, "
            f"digits and hyphens are allowed, and the first character "
            f"must be a letter or digit (e.g. 'fast', 'long-ctx', "
            f"'q4-balanced')"
        )
    return name


def _dict_to_tomlkit(d: dict[str, Any]):
    import tomlkit
    t = tomlkit.table()
    for k, v in d.items():
        t[k] = v
    return t


def _profile_to_tomlkit(prof: Profile):
    """Serialise a Profile to a tomlkit table with the structured fields
    inlined and only the raw args left in a sub-table."""
    import tomlkit
    tbl = tomlkit.table()
    if prof.mmproj:
        tbl.add("mmproj", prof.mmproj)
    if prof.ctx_size is not None:
        tbl.add("ctx_size", prof.ctx_size)
    if prof.vram_limit_gb is not None:
        tbl.add("vram_limit_gb", prof.vram_limit_gb)
    if prof.ram_spill_policy and prof.ram_spill_policy != "default":
        tbl.add("ram_spill_policy", prof.ram_spill_policy)
    if prof.ram_spill_limit_gb is not None:
        tbl.add("ram_spill_limit_gb", prof.ram_spill_limit_gb)
    if prof.image_model_type:
        tbl.add("image_model_type", prof.image_model_type)
    if prof.image_steps is not None:
        tbl.add("image_steps", prof.image_steps)
    if prof.image_guidance is not None:
        tbl.add("image_guidance", prof.image_guidance)
    if prof.image_size:
        tbl.add("image_size", prof.image_size)
    if prof.image_seed is not None:
        tbl.add("image_seed", prof.image_seed)
    if prof.image_editing_scheduler:
        tbl.add("image_editing_scheduler", prof.image_editing_scheduler)
    if prof.image_strength is not None:
        tbl.add("image_strength", prof.image_strength)
    if prof.thinking:
        tbl.add("thinking", prof.thinking)
    if prof.args:
        tbl.add("args", _dict_to_tomlkit(prof.args))
    return tbl


def _ensure_model_table(doc, model_id: str):
    """Ensure ``[models."<id>"]`` exists in the tomlkit doc; return it."""
    import tomlkit
    if "models" not in doc:
        doc.add("models", tomlkit.table(is_super_table=True))
    models = doc["models"]
    if model_id not in models:
        models[model_id] = tomlkit.table()
    return models[model_id]


def save_profile(cfg_path: Path, model_id: str, name: str, prof: Profile) -> None:
    """Create or overwrite a profile under its parent model in config.toml."""
    if not model_id:
        raise ValueError("model_id is required (profiles are nested under models)")
    name = _validate_profile_name(name)
    if prof.ram_spill_policy not in VALID_RAM_SPILL_POLICIES:
        raise ValueError(
            f"invalid ram_spill_policy: {prof.ram_spill_policy!r}; "
            f"must be one of {VALID_RAM_SPILL_POLICIES}"
        )
    doc = _load_tomlkit(cfg_path)
    m_tbl = _ensure_model_table(doc, model_id)
    if "profiles" not in m_tbl:
        import tomlkit
        m_tbl.add("profiles", tomlkit.table(is_super_table=True))
    m_tbl["profiles"][name] = _profile_to_tomlkit(prof)
    _save_tomlkit(cfg_path, doc)


def delete_profile(cfg_path: Path, model_id: str, name: str) -> None:
    """Remove a profile from its parent model in config.toml. Also clear
    the model's ``default_profile`` if it pointed at this name."""
    doc = _load_tomlkit(cfg_path)
    models = doc.get("models")
    if not models or model_id not in models:
        return
    m_tbl = models[model_id]
    profs = m_tbl.get("profiles")
    if profs and name in profs:
        del profs[name]
    if m_tbl.get("default_profile") == name:
        m_tbl["default_profile"] = ""
    _save_tomlkit(cfg_path, doc)


def rename_profile(cfg_path: Path, model_id: str, old: str, new: str) -> None:
    """Rename a profile in-place. Preserves the default-profile pointer."""
    if old == new:
        return
    new = _validate_profile_name(new)
    doc = _load_tomlkit(cfg_path)
    models = doc.get("models")
    if not models or model_id not in models:
        raise ValueError(f"model {model_id!r} not in config")
    m_tbl = models[model_id]
    profs = m_tbl.get("profiles")
    if not profs or old not in profs:
        raise ValueError(f"profile {old!r} not found for model {model_id!r}")
    if new in profs:
        raise ValueError(f"profile {new!r} already exists for model {model_id!r}")
    profs[new] = profs[old]
    del profs[old]
    if m_tbl.get("default_profile") == old:
        m_tbl["default_profile"] = new
    _save_tomlkit(cfg_path, doc)


def set_model_default_profile(cfg_path: Path, model_id: str, profile_name: str) -> None:
    """Write the per-model default profile pointer. Empty string clears it."""
    doc = _load_tomlkit(cfg_path)
    m_tbl = _ensure_model_table(doc, model_id)
    m_tbl["default_profile"] = profile_name
    _save_tomlkit(cfg_path, doc)


def clear_model_default_profile(cfg_path: Path, model_id: str) -> None:
    """Equivalent to ``set_model_default_profile(..., "")``."""
    set_model_default_profile(cfg_path, model_id, "")


def delete_model_entry(cfg_path: Path, model_id: str) -> list[str]:
    """Remove a whole model block (its profiles + default pointer) from
    config.toml. Returns the list of profile names that were removed."""
    doc = _load_tomlkit(cfg_path)
    models = doc.get("models")
    removed: list[str] = []
    if models and model_id in models:
        m_tbl = models[model_id]
        profs = m_tbl.get("profiles")
        if profs:
            removed = list(profs.keys())
        del models[model_id]
        _save_tomlkit(cfg_path, doc)
    return removed


def set_default_args(cfg_path: Path, engine: str, args: dict[str, Any]) -> None:
    """Replace the ``[default_args.<engine>]`` block."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "default_args" not in doc:
        doc.add("default_args", tomlkit.table(is_super_table=True))
    doc["default_args"][engine] = _dict_to_tomlkit(args)
    _save_tomlkit(cfg_path, doc)


def update_defaults(cfg_path: Path, *,
                    default_model: str | None = None,
                    default_image_model: str | None = None,
                    default_image_profile: str | None = None,
                    autolaunch: bool | None = None) -> None:
    """Update the [defaults] section in config.toml. Pass an empty string
    for ``default_image_model`` / ``default_image_profile`` to clear the
    saved value."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "defaults" not in doc:
        doc.add("defaults", tomlkit.table())
    if default_model is not None:
        doc["defaults"]["model"] = default_model
    if default_image_model is not None:
        doc["defaults"]["image_model"] = default_image_model
    if default_image_profile is not None:
        doc["defaults"]["image_profile"] = default_image_profile
    if autolaunch is not None:
        doc["defaults"]["autolaunch"] = autolaunch
    _save_tomlkit(cfg_path, doc)


def update_image_config(cfg_path: Path, *,
                        hidream_python: str | None = None,
                        hidream_repo: str | None = None,
                        hidream_target_rocm_release: str | None = None,
                        flux2_sd_cli: str | None = None,
                        flux2_device_index: int | None = None,
                        clear_flux2_device_index: bool = False,
                        z_image_python: str | None = None,
                        images_dir: str | None = None,
                        max_disk_gb: int | None = None) -> None:
    """Update the [image] section in config.toml. Each kwarg is persisted
    only when non-None. Pass ``clear_flux2_device_index=True`` to remove
    the key entirely."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "image" not in doc:
        doc.add("image", tomlkit.table())
    img = doc["image"]
    if hidream_python is not None:
        img["hidream_python"] = hidream_python
    if hidream_repo is not None:
        img["hidream_repo"] = hidream_repo
    if hidream_target_rocm_release is not None:
        img["hidream_target_rocm_release"] = hidream_target_rocm_release
    if flux2_sd_cli is not None:
        img["flux2_sd_cli"] = flux2_sd_cli
    if clear_flux2_device_index:
        if "flux2_device_index" in img:
            del img["flux2_device_index"]
    elif flux2_device_index is not None:
        img["flux2_device_index"] = int(flux2_device_index)
    if z_image_python is not None:
        img["z_image_python"] = z_image_python
    if images_dir is not None:
        img["images_dir"] = images_dir
    if max_disk_gb is not None:
        img["max_disk_gb"] = int(max_disk_gb)
    _save_tomlkit(cfg_path, doc)


def set_diffusers_override(cfg_path: Path, engine: str,
                           version: str | None) -> None:
    """Set or clear the per-engine diffusers version override under
    ``[image].diffusers_version.<engine>``.

    Pass a version string to pin it (a deliberate upgrade/downgrade); pass
    ``None`` or an empty string to clear it and fall back to the shipped
    ``DIFFUSERS_PIN``.
    """
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "image" not in doc:
        doc.add("image", tomlkit.table())
    img = doc["image"]
    if "diffusers_version" not in img:
        if not version:
            return
        img["diffusers_version"] = tomlkit.table()
    table = img["diffusers_version"]
    if version:
        table[engine] = version
    elif engine in table:
        del table[engine]
    _save_tomlkit(cfg_path, doc)


def update_exclusive_mode(cfg_path: Path, *,
                          mode: str | None = None,
                          grace_seconds: float | None = None,
                          heartbeat_seconds: int | None = None) -> None:
    """Update the exclusive-mode knobs under [server] in config.toml.

    Refuses to set a non-``off`` mode while ``multi_slot_enabled`` is on
    — the two features are mutually exclusive. Caller (admin handler)
    should either disable multi-slot first or surface the error.
    """
    import tomlkit
    if mode is not None:
        m = mode.strip().lower()
        if m not in VALID_EXCLUSIVE_MODES:
            raise ValueError(
                f"invalid exclusive_mode {mode!r}; "
                f"must be one of {VALID_EXCLUSIVE_MODES}"
            )
        mode = m
    doc = _load_tomlkit(cfg_path)
    if "server" not in doc:
        doc.add("server", tomlkit.table())
    srv = doc["server"]
    if mode is not None and mode != "off":
        if bool(srv.get("multi_slot_enabled", False)):
            raise ValueError(
                "cannot enable exclusive mode while multi-slot is on; "
                "disable multi-slot first (or set exclusive_mode = \"off\")."
            )
    if mode is not None:
        srv["exclusive_mode"] = mode
    if grace_seconds is not None:
        srv["exclusive_grace_seconds"] = float(grace_seconds)
    if heartbeat_seconds is not None:
        srv["exclusive_heartbeat_seconds"] = int(heartbeat_seconds)
    _save_tomlkit(cfg_path, doc)


def update_coexistence_policy(cfg_path: Path, *,
                              unload_text_on_arrival: bool | None = None,
                              restart_text_after_image: bool | None = None,
                              allow_concurrent: bool | None = None,
                              allow_diffusion_with_slots: bool | None = None) -> None:
    """Update the [coexistence] section in config.toml."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "coexistence" not in doc:
        doc.add("coexistence", tomlkit.table())
    sect = doc["coexistence"]
    if unload_text_on_arrival is not None:
        sect["unload_text_on_arrival"] = bool(unload_text_on_arrival)
    if restart_text_after_image is not None:
        sect["restart_text_after_image"] = bool(restart_text_after_image)
    if allow_concurrent is not None:
        sect["allow_concurrent"] = bool(allow_concurrent)
    if allow_diffusion_with_slots is not None:
        sect["allow_diffusion_with_slots"] = bool(allow_diffusion_with_slots)
    _save_tomlkit(cfg_path, doc)


def update_auto_update(cfg_path: Path, *,
                       engine: str | None = None,
                       enabled: bool | None = None,
                       idle_seconds: int | None = None,
                       check_interval_seconds: int | None = None) -> None:
    """Update the [auto_update] section in config.toml.

    Pass ``engine`` + ``enabled`` to flip one engine's switch under
    ``[auto_update.engines]``; pass ``idle_seconds`` / ``check_interval_seconds``
    to tune the global cadence. Engine keys are llama variant ids, diffusion
    engine names, or the reserved ``"llamanager"`` self-update key.
    """
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "auto_update" not in doc:
        doc.add("auto_update", tomlkit.table())
    sect = doc["auto_update"]
    if idle_seconds is not None:
        sect["idle_seconds"] = int(idle_seconds)
    if check_interval_seconds is not None:
        sect["check_interval_seconds"] = int(check_interval_seconds)
    if engine is not None and enabled is not None:
        if "engines" not in sect:
            sect["engines"] = tomlkit.table()
        sect["engines"][engine] = bool(enabled)
    _save_tomlkit(cfg_path, doc)


def update_multi_slot(cfg_path: Path, *,
                      enabled: bool | None = None,
                      base_port: int | None = None,
                      max_slots: int | None = None) -> None:
    """Update the multi-slot knobs under [server] in config.toml.

    Refuses to enable multi-slot while ``exclusive_mode != off`` — caller
    must clear exclusive first (or the admin handler does the choreography
    of writing both keys in one shot).
    """
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "server" not in doc:
        doc.add("server", tomlkit.table())
    srv = doc["server"]
    if enabled is True:
        existing_excl = str(srv.get("exclusive_mode", "off") or "off")
        if existing_excl.strip().lower() not in ("", "off"):
            raise ValueError(
                "cannot enable multi-slot while exclusive_mode is "
                f"{existing_excl!r}; set exclusive_mode = \"off\" first."
            )
    if enabled is not None:
        srv["multi_slot_enabled"] = bool(enabled)
    if base_port is not None:
        srv["multi_slot_base_port"] = int(base_port)
    if max_slots is not None:
        srv["multi_slot_max"] = max(1, int(max_slots))
    _save_tomlkit(cfg_path, doc)
