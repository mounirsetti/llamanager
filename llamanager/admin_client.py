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
