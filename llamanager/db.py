"""SQLite state store. Single file, single writer, hand-rolled migrations.

Tables (per spec §4.7):
  origins, requests, downloads, events.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSIONS: list[str] = [
    # v1: initial schema.
    """
    CREATE TABLE origins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        key_hash TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 50,
        allowed_models_json TEXT NOT NULL DEFAULT '["default"]',
        is_admin INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE TABLE requests (
        id TEXT PRIMARY KEY,
        origin_id INTEGER,
        model TEXT,
        priority INTEGER,
        status TEXT NOT NULL,
        enqueued_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        error TEXT
    );
    CREATE TABLE downloads (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        files_json TEXT NOT NULL,
        status TEXT NOT NULL,
        bytes_done INTEGER NOT NULL DEFAULT 0,
        bytes_total INTEGER NOT NULL DEFAULT 0,
        started_at REAL NOT NULL,
        finished_at REAL,
        error TEXT
    );
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );
    CREATE INDEX idx_requests_status ON requests(status);
    CREATE INDEX idx_requests_origin ON requests(origin_id);
    CREATE INDEX idx_events_ts ON events(ts);
    """,
    # v2: deterministic per-key lookup so auth.verify is O(1) instead of
    # argon2-hashing every origin row on every request. Pre-existing rows
    # stay NULL until rotated; AuthManager falls back to a scan for those.
    """
    ALTER TABLE origins ADD COLUMN key_lookup TEXT;
    CREATE INDEX idx_origins_key_lookup ON origins(key_lookup);
    """,
]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA user_version")
    version = int(cur.fetchone()[0])
    target = len(SCHEMA_VERSIONS)
    if version >= target:
        return
    for i in range(version, target):
        conn.executescript(SCHEMA_VERSIONS[i])
    conn.execute(f"PRAGMA user_version={target}")


class DB:
    """Thin synchronous wrapper. SQLite is fast enough that wrapping every
    call in `asyncio.to_thread` would buy nothing for v1's load."""

    def __init__(self, path: Path):
        self.path = path
        self.conn = _connect(path)
        migrate(self.conn)

    # ---- generic helpers ----
    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    # ---- events / audit log (also used for free-form logging) ----
    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, kind, payload_json) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, default=str)),
        )

    # ---- request lifecycle ----
    def insert_request(self, *, request_id: str, origin_id: int | None,
                       model: str | None, priority: int) -> None:
        self.conn.execute(
            "INSERT INTO requests(id, origin_id, model, priority, status, enqueued_at)"
            " VALUES (?, ?, ?, ?, 'queued', ?)",
            (request_id, origin_id, model, priority, time.time()),
        )

    def update_request_status(self, request_id: str, status: str,
                              **fields: Any) -> None:
        cols = ["status=?"]
        vals: list[Any] = [status]
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(v)
        vals.append(request_id)
        self.conn.execute(
            f"UPDATE requests SET {', '.join(cols)} WHERE id=?", tuple(vals)
        )

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
