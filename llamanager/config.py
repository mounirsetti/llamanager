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
max_disk_gb = 80
hf_token_env = "HF_TOKEN"

[queue]
max_concurrent = 1
max_queue_depth = 200
max_wait_s = 300
queue_timeout_s = 300
"""


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

def detect_engine_for_path(model_path: Path) -> str:
    """Return ``"llama"`` for GGUF files, ``"mlx"`` for MLX/HF directory
    layouts, and fall back to ``"llama"`` when ambiguous."""
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return "llama"
    if model_path.is_dir() and (model_path / "config.json").exists():
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


@dataclass
class Profile:
    """A named launch preset for a model.

    ``ctx_size`` / ``vram_limit_gb`` / ``ram_spill_policy`` /
    ``ram_spill_limit_gb`` are the "basic" UI knobs — they map to engine
    flags (mostly ``--ctx-size`` + a heuristic ``--n-gpu-layers``).
    ``args`` is the raw advanced override map; it wins over basic-derived
    args during launch.
    """
    name: str
    mmproj: str = ""
    ctx_size: int | None = None
    vram_limit_gb: float | None = None
    ram_spill_policy: str = "default"
    ram_spill_limit_gb: float | None = None
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
    default_origin_priority: int = 50
    autolaunch: bool = False

    max_restarts_in_window: int = 3
    window_seconds: int = 300
    success_run_seconds: int = 300

    max_disk_gb: int = 80
    hf_token_env: str = "HF_TOKEN"

    max_concurrent: int = 1
    max_queue_depth: int = 200
    max_wait_s: int = 300
    queue_timeout_s: int = 300

    # New: profiles nest under their parent model.
    models: dict[str, ModelConfig] = field(default_factory=dict)
    # Engine-keyed minimum defaults ("llama" / "mlx" → args dict).
    default_args: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Set by the operator via /ui/models/set-dir or [server].models_dir in
    # config.toml. When None, models_dir falls back to data_dir/models.
    models_dir_override: Path | None = None

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
    return Profile(
        name=name,
        mmproj=str(body.get("mmproj", "") or ""),
        ctx_size=_coerce_int(body.get("ctx_size")),
        vram_limit_gb=_coerce_float(body.get("vram_limit_gb")),
        ram_spill_policy=policy,
        ram_spill_limit_gb=_coerce_float(body.get("ram_spill_limit_gb")),
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

    cfg = Config(
        bind=server.get("bind", "127.0.0.1"),
        port=int(server.get("port", 7200)),
        llama_server_binary=server.get("llama_server_binary", "llama-server"),
        llama_server_engine=str(server.get("llama_server_engine", "llama")),
        llama_server_port=int(server.get("llama_server_port", 7201)),
        data_dir=expand(server.get("data_dir", "~/.llamanager")),
        default_model=defaults.get("model", ""),
        default_origin_priority=int(defaults.get("origin_priority", 50)),
        autolaunch=bool(defaults.get("autolaunch", False)),
        max_restarts_in_window=int(rp.get("max_restarts_in_window", 3)),
        window_seconds=int(rp.get("window_seconds", 300)),
        success_run_seconds=int(rp.get("success_run_seconds", 300)),
        max_disk_gb=int(dl.get("max_disk_gb", 80)),
        hf_token_env=dl.get("hf_token_env", "HF_TOKEN"),
        max_concurrent=int(q.get("max_concurrent", 1)),
        max_queue_depth=int(q.get("max_queue_depth", 200)),
        max_wait_s=int(q.get("max_wait_s", 300)),
        queue_timeout_s=int(q.get("queue_timeout_s", 300)),
        raw=raw,
        path=cfg_path,
    )

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
            f"profile name must be lowercase alphanumeric with hyphens: {name!r}"
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
                    autolaunch: bool | None = None) -> None:
    """Update the [defaults] section in config.toml."""
    import tomlkit
    doc = _load_tomlkit(cfg_path)
    if "defaults" not in doc:
        doc.add("defaults", tomlkit.table())
    if default_model is not None:
        doc["defaults"]["model"] = default_model
    if autolaunch is not None:
        doc["defaults"]["autolaunch"] = autolaunch
    _save_tomlkit(cfg_path, doc)
