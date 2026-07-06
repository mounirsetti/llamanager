"""Thin HTTP client around the /admin/* control plane.

Used by the CLI verbs (`llamanager models ...`, `llamanager server ...`, etc.)
so agents and shell scripts can drive a running daemon without speaking the
full FastAPI surface. Inference itself is already covered by the OpenAI-
compatible /v1/* endpoint, so this file deliberately stays admin-only.

Auth resolution order (first hit wins):
    1. explicit `admin_key` argument (e.g. from a `--admin-key` flag)
    2. LLAMANAGER_ADMIN_KEY env var
    3. `[cli].admin_key` in config.toml

Base URL resolution order:
    1. explicit `base_url` argument (e.g. from a `--url` flag)
    2. LLAMANAGER_URL env var
    3. derived from config (`http://{bind}:{port}`, with 0.0.0.0 rewritten
       to 127.0.0.1 since the CLI usually runs on the same host)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config


class AdminClientError(RuntimeError):
    """Raised for any non-2xx response or transport-level failure."""


def resolve_base_url(cfg: Config | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("LLAMANAGER_URL")
    if env:
        return env.rstrip("/")
    if cfg is None:
        return "http://127.0.0.1:7200"
    host = cfg.bind if cfg.bind not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{cfg.port}"


def resolve_admin_key(cfg: Config | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("LLAMANAGER_ADMIN_KEY")
    if env:
        return env
    if cfg is not None:
        cli_section = (cfg.raw or {}).get("cli") or {}
        if cli_section.get("admin_key"):
            return str(cli_section["admin_key"])
    raise AdminClientError(
        "no admin key found. Set LLAMANAGER_ADMIN_KEY, pass --admin-key, "
        "or add `admin_key = \"...\"` under a `[cli]` section in config.toml."
    )


@dataclass
class AdminClient:
    base_url: str
    admin_key: str
    timeout: float = 30.0
    # Optional injected httpx client. Tests pass one backed by
    # httpx.ASGITransport to talk to the FastAPI app in-process; production
    # callers leave it None and we fall back to httpx.request().
    client: httpx.Client | None = None

    @classmethod
    def from_config(cls, cfg: Config | None, *, admin_key: str | None = None,
                    base_url: str | None = None,
                    timeout: float = 30.0,
                    client: httpx.Client | None = None) -> "AdminClient":
        return cls(
            base_url=resolve_base_url(cfg, base_url),
            admin_key=resolve_admin_key(cfg, admin_key),
            timeout=timeout,
            client=client,
        )

    # ---- low-level ----

    def _request(self, method: str, path: str, *,
                 json_body: Any | None = None,
                 params: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        try:
            if self.client is not None:
                # Injected clients (e.g. FastAPI's TestClient) carry their
                # own timeout; passing it here triggers a deprecation warning.
                r = self.client.request(method, url, headers=headers,
                                         json=json_body, params=params)
            else:
                r = httpx.request(method, url, headers=headers,
                                  json=json_body, params=params,
                                  timeout=self.timeout)
        except httpx.HTTPError as e:
            raise AdminClientError(
                f"could not reach llamanager at {self.base_url}: {e}"
            ) from e
        if r.status_code >= 400:
            detail: Any
            try:
                detail = r.json().get("detail", r.text)
            except (ValueError, json.JSONDecodeError):
                detail = r.text
            raise AdminClientError(
                f"{method} {path} -> {r.status_code}: {detail}"
            )
        return r

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params or None).json()

    def _post(self, path: str, body: Any | None = None) -> Any:
        r = self._request("POST", path, json_body=body)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def _delete(self, path: str, **params: Any) -> Any:
        r = self._request("DELETE", path, params=params or None)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    # ---- status ----

    def status(self) -> dict[str, Any]:
        return self._get("/admin/status")

    def disk(self) -> dict[str, Any]:
        return self._get("/admin/disk")

    def reload(self) -> dict[str, Any]:
        return self._post("/admin/reload")

    def events(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._get("/admin/events", limit=limit)

    def logs(self, *, source: str = "llama-server", tail: int = 200) -> str:
        r = self._request("GET", "/admin/logs",
                          params={"source": source, "tail": tail})
        return r.text

    # ---- exclusive mode ----

    def exclusive_status(self) -> dict[str, Any]:
        return self._get("/admin/exclusive")

    def exclusive_set(self, *, mode: str | None = None,
                      grace_seconds: float | None = None,
                      heartbeat_seconds: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if mode is not None:
            body["mode"] = mode
        if grace_seconds is not None:
            body["grace_seconds"] = grace_seconds
        if heartbeat_seconds is not None:
            body["heartbeat_seconds"] = heartbeat_seconds
        return self._post("/admin/exclusive", body)

    def exclusive_sweep(self) -> dict[str, Any]:
        return self._post("/admin/exclusive/sweep")

    # ---- multi-slot LLM (beta) ----

    def slots_status(self) -> dict[str, Any]:
        return self._get("/admin/slots")

    def slots_set_enabled(self, enabled: bool) -> dict[str, Any]:
        return self._post("/admin/slots/enable", {"enabled": bool(enabled)})

    def slots_add(self) -> dict[str, Any]:
        return self._post("/admin/slots")

    def slots_remove(self, slot_id: int) -> dict[str, Any]:
        return self._delete(f"/admin/slots/{slot_id}")

    def slots_load(self, slot_id: int, *, model: str,
                   profile: str | None = None,
                   args: dict[str, Any] | None = None,
                   force: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "force": bool(force)}
        if profile is not None:
            body["profile"] = profile
        if args:
            body["args"] = args
        return self._post(f"/admin/slots/{slot_id}/load", body)

    def slots_unload(self, slot_id: int) -> dict[str, Any]:
        return self._post(f"/admin/slots/{slot_id}/unload")

    def slots_diffusion_coex(self, allow: bool) -> dict[str, Any]:
        return self._post("/admin/slots/diffusion-coexistence",
                          {"allow": bool(allow)})

    # ---- server lifecycle ----

    def server_start(self, *, profile: str | None = None,
                     model: str | None = None,
                     mmproj: str | None = None,
                     args: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {"profile": profile, "model": model,
                "mmproj": mmproj, "args": args or {}}
        return self._post("/admin/server/start", body)

    def server_stop(self) -> dict[str, Any]:
        return self._post("/admin/server/stop")

    def server_restart(self, *, profile: str | None = None,
                       model: str | None = None,
                       mmproj: str | None = None,
                       args: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {"profile": profile, "model": model,
                "mmproj": mmproj, "args": args or {}}
        return self._post("/admin/server/restart", body)

    def server_swap(self, *, profile: str | None = None,
                    model: str | None = None,
                    mmproj: str | None = None,
                    args: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {"profile": profile, "model": model,
                "mmproj": mmproj, "args": args or {}}
        return self._post("/admin/server/swap", body)

    # ---- queue ----

    def queue_list(self) -> dict[str, Any]:
        return self._get("/admin/queue")

    def queue_cancel(self, request_id: str) -> dict[str, Any]:
        return self._delete(f"/admin/queue/{request_id}")

    def queue_pause(self) -> dict[str, Any]:
        return self._post("/admin/queue/pause")

    def queue_resume(self) -> dict[str, Any]:
        return self._post("/admin/queue/resume")

    # ---- models ----

    def models_list(self) -> list[dict[str, Any]]:
        return self._get("/admin/models")

    def models_pull(self, source: str,
                    files: list[str] | None = None) -> dict[str, Any]:
        return self._post("/admin/models/pull",
                          {"source": source, "files": files})

    def model_delete(self, model_id: str, *,
                     force: bool = False) -> dict[str, Any]:
        return self._delete(f"/admin/models/{model_id}", force=str(force).lower())

    # ---- downloads ----

    def downloads_list(self) -> list[dict[str, Any]]:
        return self._get("/admin/downloads")

    def download_get(self, download_id: str) -> dict[str, Any]:
        return self._get(f"/admin/downloads/{download_id}")

    def download_cancel(self, download_id: str) -> dict[str, Any]:
        return self._delete(f"/admin/downloads/{download_id}")

    # ---- origins ----

    def origins_list(self) -> list[dict[str, Any]]:
        return self._get("/admin/origins")

    def origin_create(self, name: str, *,
                      priority: int | None = None,
                      allowed_models: list[str] | None = None,
                      is_admin: bool = False) -> dict[str, Any]:
        return self._post("/admin/origins", {
            "name": name, "priority": priority,
            "allowed_models": allowed_models, "is_admin": is_admin,
        })

    def origin_delete(self, origin_id: int) -> dict[str, Any]:
        return self._delete(f"/admin/origins/{origin_id}")

    def origin_rotate_key(self, origin_id: int) -> dict[str, Any]:
        return self._post(f"/admin/origins/{origin_id}/rotate-key")

    def origin_update(self, origin_id: int, *,
                      priority: int | None = None,
                      allowed_models: list[str] | None = None,
                      is_admin: bool | None = None) -> dict[str, Any]:
        """Patch one or more fields on an existing origin. Pass None to
        leave a field alone — the server treats omitted fields as
        no-ops."""
        body: dict[str, Any] = {}
        if priority is not None: body["priority"] = priority
        if allowed_models is not None: body["allowed_models"] = allowed_models
        if is_admin is not None: body["is_admin"] = is_admin
        return self._request("PATCH", f"/admin/origins/{origin_id}",
                             json_body=body).json()

    # ---- queue ----

    def queue_cancel_all(self) -> dict[str, Any]:
        """Cancel every queued + in-flight request in one call."""
        return self._post("/admin/queue/cancel-all")

    # ---- LLM profiles ----

    def profiles_list(self, model_id: str) -> dict[str, Any]:
        return self._get("/admin/profiles", model=model_id)

    def profile_create(self, model_id: str, name: str, *,
                       mmproj: str = "",
                       ctx_size: int | None = None,
                       vram_limit_gb: float | None = None,
                       ram_spill_policy: str = "default",
                       ram_spill_limit_gb: float | None = None,
                       kv_cache_type: str = "",
                       flash_attn: str = "",
                       thinking: str = "",
                       reasoning_budget: int | None = None,
                       parallel: int | None = None,
                       mtp: bool = False,
                       mtp_n_max: int | None = None,
                       args: dict[str, Any] | None = None,
                       make_default: bool = False) -> dict[str, Any]:
        return self._post("/admin/profiles", {
            "model_id": model_id, "name": name,
            "mmproj": mmproj, "ctx_size": ctx_size,
            "vram_limit_gb": vram_limit_gb,
            "ram_spill_policy": ram_spill_policy,
            "ram_spill_limit_gb": ram_spill_limit_gb,
            "kv_cache_type": kv_cache_type,
            "flash_attn": flash_attn,
            "thinking": thinking,
            "reasoning_budget": reasoning_budget,
            "parallel": parallel,
            "mtp": mtp, "mtp_n_max": mtp_n_max,
            "args": args or {},
            "make_default": make_default,
        })

    def profile_update(self, name: str, model_id: str, *,
                       mmproj: str | None = None,
                       ctx_size: int | None = None,
                       vram_limit_gb: float | None = None,
                       ram_spill_policy: str | None = None,
                       ram_spill_limit_gb: float | None = None,
                       kv_cache_type: str | None = None,
                       flash_attn: str | None = None,
                       thinking: str | None = None,
                       reasoning_budget: int | None = None,
                       parallel: int | None = None,
                       mtp: bool | None = None,
                       mtp_n_max: int | None = None,
                       args: dict[str, Any] | None = None,
                       new_name: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id}
        if mmproj is not None: body["mmproj"] = mmproj
        if ctx_size is not None: body["ctx_size"] = ctx_size
        if vram_limit_gb is not None: body["vram_limit_gb"] = vram_limit_gb
        if ram_spill_policy is not None: body["ram_spill_policy"] = ram_spill_policy
        if ram_spill_limit_gb is not None: body["ram_spill_limit_gb"] = ram_spill_limit_gb
        if kv_cache_type is not None: body["kv_cache_type"] = kv_cache_type
        if flash_attn is not None: body["flash_attn"] = flash_attn
        if thinking is not None: body["thinking"] = thinking
        if reasoning_budget is not None: body["reasoning_budget"] = reasoning_budget
        if parallel is not None: body["parallel"] = parallel
        if mtp is not None: body["mtp"] = mtp
        if mtp_n_max is not None: body["mtp_n_max"] = mtp_n_max
        if args is not None: body["args"] = args
        if new_name is not None: body["new_name"] = new_name
        return self._request("PATCH", f"/admin/profiles/{name}",
                             json_body=body).json()

    def profile_delete(self, name: str, model_id: str) -> dict[str, Any]:
        return self._delete(f"/admin/profiles/{name}", model_id=model_id)

    def profile_clone(self, name: str, model_id: str,
                      new_name: str) -> dict[str, Any]:
        return self._post(f"/admin/profiles/{name}/clone",
                          {"model_id": model_id, "new_name": new_name})

    def profile_set_model_default(self, model_id: str,
                                  profile_name: str = "") -> dict[str, Any]:
        return self._post("/admin/profiles/set-model-default",
                          {"model_id": model_id,
                           "profile_name": profile_name})

    # ---- models housekeeping ----

    def model_set_default(self, model_id: str) -> dict[str, Any]:
        return self._post("/admin/models/set-default", {"model_id": model_id})

    def model_add_existing(self, file_path: str) -> dict[str, Any]:
        return self._post("/admin/models/add-existing",
                          {"file_path": file_path})

    def models_set_dir(self, models_dir: str) -> dict[str, Any]:
        return self._post("/admin/models/set-dir", {"models_dir": models_dir})

    # ---- setup / config ----

    def setup_llama_binary(self, binary_path: str) -> dict[str, Any]:
        return self._post("/admin/setup/llama-binary",
                          {"binary_path": binary_path})

    def setup_hidream(self, *, python: str | None = None,
                      repo: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if python is not None: body["hidream_python"] = python
        if repo is not None: body["hidream_repo"] = repo
        return self._post("/admin/setup/hidream", body)

    def setup_z_image(self, python: str) -> dict[str, Any]:
        return self._post("/admin/setup/z-image", {"z_image_python": python})

    def setup_flux2(self, *, sd_cli: str | None = None,
                    device_index: int | None = None,
                    clear_device_index: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"clear_device_index": clear_device_index}
        if sd_cli is not None: body["flux2_sd_cli"] = sd_cli
        if device_index is not None: body["flux2_device_index"] = device_index
        return self._post("/admin/setup/flux2", body)

    def setup_coexistence(self, *,
                          unload_text_on_arrival: bool | None = None,
                          restart_text_after_image: bool | None = None,
                          allow_concurrent: bool | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if unload_text_on_arrival is not None:
            body["unload_text_on_arrival"] = unload_text_on_arrival
        if restart_text_after_image is not None:
            body["restart_text_after_image"] = restart_text_after_image
        if allow_concurrent is not None:
            body["allow_concurrent"] = allow_concurrent
        return self._post("/admin/setup/coexistence", body)

    def setup_default_args(self, engine: str,
                           args: dict[str, Any]) -> dict[str, Any]:
        return self._post("/admin/setup/default-args",
                          {"engine": engine, "args": args})

    def setup_autolaunch(self, enabled: bool) -> dict[str, Any]:
        return self._post("/admin/setup/autolaunch", {"enabled": enabled})

    def setup_autorestart(self, enabled: bool) -> dict[str, Any]:
        return self._post("/admin/setup/autorestart", {"enabled": enabled})

    def setup_install_llama_server(self, *, source: str = "llama.cpp",
                                   backend: str = "",
                                   version: str = "") -> dict[str, Any]:
        return self._post("/admin/setup/install-llama-server",
                          {"source": source, "backend": backend,
                           "version": version})

    def setup_engine_versions(self, variant: str) -> dict[str, Any]:
        return self._get("/admin/setup/engine-versions", variant=variant)

    def setup_check_updates(self, variant: str = "") -> dict[str, Any]:
        params = {"variant": variant} if variant else {}
        return self._get("/admin/setup/check-updates", **params)

    def setup_install_llama_server_status(self, variant: str) -> dict[str, Any]:
        return self._get("/admin/setup/install-llama-server/status",
                         variant=variant)

    def setup_switch_variant(self, variant: str) -> dict[str, Any]:
        return self._post("/admin/setup/switch-variant", {"variant": variant})

    # ---- auto-update-when-idle ----

    def setup_auto_update_list(self) -> dict[str, Any]:
        return self._get("/admin/setup/auto-update")

    def setup_auto_update(self, engine: str, enabled: bool) -> dict[str, Any]:
        return self._post("/admin/setup/auto-update",
                          {"engine": engine, "enabled": enabled})

    def setup_auto_update_settings(self, *,
                                   idle_seconds: int | None = None,
                                   check_interval_seconds: int | None = None,
                                   ) -> dict[str, Any]:
        return self._post("/admin/setup/auto-update/settings", {
            "idle_seconds": idle_seconds,
            "check_interval_seconds": check_interval_seconds,
        })

    # ---- diffusion ----
    #
    # Mirrors the /ui/diffusion-models page over JSON so the CLI can
    # list catalogs, kick installs, and manage profiles for image
    # engines. Server-side handlers live in api_admin.py.

    def diffusion_engines(self) -> dict[str, Any]:
        return self._get("/admin/diffusion/engines")

    def diffusion_install(self, engine: str, *,
                          patch_flash_attn: bool = False,
                          diffusers_version: str = "",
                          reset_diffusers: bool = False) -> dict[str, Any]:
        return self._post(f"/admin/diffusion/engines/{engine}/install",
                          {"patch_flash_attn": patch_flash_attn,
                           "diffusers_version": diffusers_version,
                           "reset_diffusers": reset_diffusers})

    def diffusion_versions(self, engine: str) -> dict[str, Any]:
        return self._get(f"/admin/diffusion/engines/{engine}/versions")

    def diffusion_cancel_install(self, engine: str) -> dict[str, Any]:
        return self._post(f"/admin/diffusion/engines/{engine}/cancel-install")

    def diffusion_models(self) -> dict[str, Any]:
        return self._get("/admin/diffusion/models")

    def diffusion_activate(self, model_id: str) -> dict[str, Any]:
        return self._post("/admin/diffusion/models/activate",
                          {"model_id": model_id})

    def diffusion_profiles(self, model_id: str) -> dict[str, Any]:
        return self._get("/admin/diffusion/profiles", model=model_id)

    def diffusion_profile_create(self, model_id: str, name: str,
                                  fields: dict[str, Any] | None = None,
                                  *,
                                  make_default: bool = False) -> dict[str, Any]:
        return self._post("/admin/diffusion/profiles", {
            "model_id": model_id, "name": name,
            "fields": fields or {}, "make_default": make_default,
        })

    def diffusion_profile_update(self, name: str, model_id: str,
                                  fields: dict[str, Any] | None = None,
                                  *, new_name: str | None = None) -> dict[str, Any]:
        return self._request("PATCH", f"/admin/diffusion/profiles/{name}",
                              json_body={
                                  "model_id": model_id,
                                  "fields": fields or {},
                                  "new_name": new_name,
                              }).json()

    def diffusion_profile_delete(self, name: str,
                                  model_id: str) -> dict[str, Any]:
        return self._delete(f"/admin/diffusion/profiles/{name}",
                             model_id=model_id)

    def diffusion_profile_clone(self, name: str, model_id: str,
                                 new_name: str) -> dict[str, Any]:
        return self._post(f"/admin/diffusion/profiles/{name}/clone",
                           {"model_id": model_id, "new_name": new_name})

    def diffusion_set_model_default_profile(self, model_id: str,
                                             profile_name: str = "") -> dict[str, Any]:
        return self._post("/admin/diffusion/profiles/set-model-default",
                           {"model_id": model_id,
                            "profile_name": profile_name})

    def diffusion_materialize_defaults(self, model_id: str,
                                        engine: str) -> dict[str, Any]:
        return self._post("/admin/diffusion/profiles/materialize-defaults",
                           {"model_id": model_id, "engine": engine})

    # ---- self-update ----

    def check_update(self) -> dict[str, Any]:
        return self._get("/admin/update/check")

    def self_update(self) -> dict[str, Any]:
        """Trigger ``pip install --upgrade llamanager`` on the daemon side
        and restart. Returns ``{ok, log, mode, ...}``.

        Editable installs return ``ok=False, mode='editable'`` *without*
        raising — the caller is expected to render the manual-update
        instructions from ``log``. Genuine failures (pip error, network)
        still raise via ``AdminClientError``.
        """
        url = f"{self.base_url}/admin/update"
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        try:
            if self.client is not None:
                r = self.client.request("POST", url, headers=headers)
            else:
                r = httpx.request("POST", url, headers=headers,
                                  timeout=self.timeout)
        except httpx.HTTPError as e:
            raise AdminClientError(
                f"could not reach llamanager at {self.base_url}: {e}"
            ) from e
        try:
            body = r.json()
        except (ValueError, json.JSONDecodeError):
            body = {"ok": False, "log": r.text,
                    "error": f"HTTP {r.status_code}",
                    "mode": "unknown"}
        # Editable-install refusal is a soft failure — pass it through
        # so the CLI can render the operator-facing instructions instead
        # of a generic "request failed".
        if r.status_code == 409 and body.get("mode") == "editable":
            return body
        if r.status_code >= 400:
            raise AdminClientError(
                f"POST /admin/update -> {r.status_code}: "
                f"{body.get('error') or body.get('detail') or r.text}"
            )
        return body

    # ---- ASR (speech-to-text) ----
    def asr_engines(self) -> dict[str, Any]:
        return self._get("/admin/asr/engines")

    def asr_install(self, *, engine: str = "asr",
                    torch_backend: str = "auto") -> dict[str, Any]:
        return self._post("/admin/asr/install",
                          {"engine": engine, "torch_backend": torch_backend})

    def asr_cancel_install(self, *, engine: str = "asr") -> dict[str, Any]:
        return self._post("/admin/asr/cancel-install", {"engine": engine})

    def asr_setup(self, python: str, *, engine: str = "asr") -> dict[str, Any]:
        return self._post("/admin/asr/setup",
                          {"python": python, "engine": engine})

    def asr_models(self) -> dict[str, Any]:
        return self._get("/admin/asr/models")

    def asr_models_dir(self, path: str = "") -> dict[str, Any]:
        return self._post("/admin/asr/models-dir", {"path": path})

    def asr_defaults(self, *, vram_budget_gb: float | None = None,
                     coexist: bool | None = None,
                     idle_timeout_s: int | None = None,
                     decode_interval_s: float | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if vram_budget_gb is not None:
            body["vram_budget_gb"] = vram_budget_gb
        if coexist is not None:
            body["coexist"] = coexist
        if idle_timeout_s is not None:
            body["idle_timeout_s"] = idle_timeout_s
        if decode_interval_s is not None:
            body["decode_interval_s"] = decode_interval_s
        return self._post("/admin/asr/defaults", body)

    def asr_profiles(self, model_id: str) -> dict[str, Any]:
        return self._get("/admin/asr/profiles", model=model_id)

    def asr_profile_create(self, model_id: str, name: str, *,
                           fields: dict[str, Any] | None = None,
                           make_default: bool = False) -> dict[str, Any]:
        return self._post("/admin/asr/profiles", {
            "model_id": model_id, "name": name,
            "fields": fields or {}, "make_default": make_default,
        })

    def asr_profile_delete(self, name: str, model_id: str) -> dict[str, Any]:
        return self._delete(f"/admin/asr/profiles/{name}", model_id=model_id)

    def asr_profile_set_default(self, model_id: str, *,
                                profile_name: str = "") -> dict[str, Any]:
        return self._post("/admin/asr/profiles/set-default", {
            "model_id": model_id, "profile_name": profile_name,
        })

    def asr_transcribe(self, file: str, model: str, *, language: str = "",
                       task: str = "transcribe", profile: str = "") -> dict[str, Any]:
        return self._post("/admin/asr/transcribe", {
            "file": file, "model": model, "language": language,
            "task": task, "profile": profile,
        })
