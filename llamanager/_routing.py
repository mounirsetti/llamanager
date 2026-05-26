"""Per-request upstream URL resolution.

Centralises the lookup that turns "this request belongs to slot X"
into "talk to ``http://127.0.0.1:portX``". Used by the proxy in
``api_v1`` and the Anthropic facade so multi-slot routing doesn't
require duplicating the lookup logic in every consumer.

In single-slot mode the queue dispatcher never sets ``slot_id`` on
the request, so the helper falls back to the pool default (slot 0)
— byte-identical to the legacy ``sm.upstream_base`` access.
"""
from __future__ import annotations

from typing import Any


def upstream_base(sm: Any, qr: Any | None = None) -> str:
    """Return the base URL for the slot owning ``qr``.

    ``sm`` is the ``ServerPool`` (or, in legacy single-slot tests, a
    plain ``ServerManager``). When ``qr`` is None or has no ``slot_id``
    attribute, returns the pool default — i.e. slot 0.
    """
    slot_id = getattr(qr, "slot_id", None) if qr is not None else None
    if slot_id is not None and hasattr(sm, "upstream_for"):
        return sm.upstream_for(slot_id)
    return sm.upstream_base
