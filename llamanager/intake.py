"""The operator intake switch — "stop taking requests".

One boolean, ``cfg.accepting_requests`` (persisted under ``[queue]`` in
config.toml), decides whether llamanager accepts inference work at all.
When it is off, every request-generating endpoint refuses at the door with
``503`` + ``Retry-After`` instead of enqueueing, so clients fail fast and
retry later rather than piling up behind a queue that will never drain.

This is deliberately *not* ``QueueManager.pause()``. Pausing the queue keeps
accepting requests and holds them until they hit ``queue_timeout_s``; the
intake switch refuses them outright and stays off across daemon restarts —
"until the switch is back off" is meant literally.

What the switch does NOT block: the admin API (``/admin/*``) and the operator
UI (``/ui/*``) — closing the door must never lock the operator out of the
control that reopens it — and the model listings (``GET /v1/models``,
``GET /anthropic/v1/models``), which are pure discovery and touch no engine.
Everything that would put load on the machine is refused, including requests
from admin-scope keys and the built-in UI chat / image pages.

Flip it from the top bar, ``llamanager intake pause`` / ``resume``, or
``POST /admin/intake/{pause,resume}``.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

log = logging.getLogger(__name__)

#: Advertised in ``Retry-After`` on the refusal. A pause is an operator action
#: of unknown duration, so this is a "check back shortly" hint, not a promise.
RETRY_AFTER_S = 60

MESSAGE = ("llamanager is not accepting requests right now "
           "(intake paused by the operator)")


def is_accepting(app: Any) -> bool:
    """Whether the daemon is currently taking requests.

    Reads ``app.state.cfg`` rather than a cached copy: the UI and admin
    handlers swap in a freshly loaded Config after every write, so the live
    object on app.state is the only reliable view of the switch.
    """
    return bool(getattr(app.state.cfg, "accepting_requests", True))


def require_open(app: Any) -> None:
    """Raise 503 when intake is closed. Call at the top of every endpoint
    that would enqueue work or drive an engine."""
    if is_accepting(app):
        return
    raise HTTPException(
        status_code=503,
        detail=MESSAGE,
        headers={"Retry-After": str(RETRY_AFTER_S)},
    )


def set_accepting(app: Any, accepting: bool) -> dict[str, Any]:
    """Flip the switch: persist it, apply it live, and drop the backlog.

    Closing the door cancels everything still queued (in-flight work is left
    to finish — no client sees a mid-stream abort). Returns a summary dict
    suitable as an API/CLI response.
    """
    from .config import load_config, update_queue_settings

    cfg = app.state.cfg
    accepting = bool(accepting)
    was = bool(getattr(cfg, "accepting_requests", True))

    update_queue_settings(cfg.config_path, accepting_requests=accepting)
    # Re-read so the live Config matches the file exactly (the same
    # write-then-reload dance the UI and admin config writers use), keeping
    # the runtime-only bind/port stable.
    fresh = load_config(cfg.config_path)
    fresh.bind = cfg.bind
    fresh.port = cfg.port
    fresh.models_dir_override = getattr(cfg, "models_dir_override", None)
    fresh.vram_total_gb = getattr(cfg, "vram_total_gb", None)
    app.state.cfg = fresh

    dropped = 0
    if not accepting and was:
        qm = getattr(app.state, "queue", None)
        if qm is not None:
            dropped = qm.cancel_pending()

    app.state.db.log_event(
        "intake_changed", {"accepting": accepting, "dropped_queued": dropped},
    )
    log.info("intake %s%s", "resumed" if accepting else "paused",
             f" ({dropped} queued request(s) dropped)" if dropped else "")
    return {"ok": True, "accepting": accepting, "dropped_queued": dropped}
