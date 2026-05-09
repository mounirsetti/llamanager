"""Audit log helpers, layered on top of the events table."""
from __future__ import annotations

import json
from typing import Any

from .db import DB


def list_events(db: DB, limit: int = 200) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT id, ts, kind, payload_json FROM events"
        " ORDER BY id DESC LIMIT ?", (limit,)
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except Exception:
            payload = {"raw": r["payload_json"]}
        out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"],
                    "payload": payload})
    return out
