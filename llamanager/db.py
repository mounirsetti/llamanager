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
    # v3: per-diffusion-engine dependency installer jobs. One row per
    # install run; the installer task streams stdout into ``log`` and
    # bumps ``progress_pct`` for the UI. ``kind`` distinguishes
    # ``pip`` (create venv + pip install) from ``binary`` (future:
    # download a release archive).
    """
    CREATE TABLE engine_installs (
        id TEXT PRIMARY KEY,
        engine TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        progress_pct INTEGER NOT NULL DEFAULT 0,
        message TEXT NOT NULL DEFAULT '',
        log TEXT NOT NULL DEFAULT '',
        started_at REAL NOT NULL,
        finished_at REAL,
        error TEXT
    );
    CREATE INDEX idx_engine_installs_engine ON engine_installs(engine);
    """,
    # v4: opaque JSON blob for per-install options (e.g. patch_flash_attn
    # for the hidream AMD path, target_rocm_release, etc.). NULL on legacy
    # rows. Kept as TEXT rather than a wide column-per-flag so future
    # installers can add their own without a schema bump.
    """
    ALTER TABLE engine_installs ADD COLUMN options_json TEXT;
    """,
    # v5: persist the prompt sent and the response generated so the UI's
    # request-detail view can show what actually happened. Both are NULL on
    # legacy rows and on requests where capture isn't possible (e.g. image
    # jobs, or the Anthropic streaming path). Stored text is clipped to a
    # sane cap by the writer (see queue_mgr._clip_text) so a runaway context
    # can't bloat the row.
    """
    ALTER TABLE requests ADD COLUMN prompt_text TEXT;
    ALTER TABLE requests ADD COLUMN response_text TEXT;
    """,
    # v6: per-origin on/off switch. When 0, the origin authenticates fine but
    # may not submit work (inference/image requests are rejected with 403).
    # Existing origins default to enabled so the migration is non-disruptive.
    """
    ALTER TABLE origins ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1;
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

    def prune(self, max_age_days: int = 90) -> dict[str, int]:
        """Delete old requests, events, and completed downloads.

        Returns a dict with the number of rows deleted per table.
        """
        cutoff = time.time() - (max_age_days * 86400)
        counts: dict[str, int] = {}
        for table, col in [
            ("requests", "enqueued_at"),
            ("events", "ts"),
        ]:
            cur = self.conn.execute(
                f"DELETE FROM {table} WHERE {col} < ?", (cutoff,)
            )
            counts[table] = cur.rowcount
        # Only prune finished downloads (done/failed/cancelled)
        cur = self.conn.execute(
            "DELETE FROM downloads WHERE finished_at IS NOT NULL AND finished_at < ?",
            (cutoff,),
        )
        counts["downloads"] = cur.rowcount
        # Reclaim disk space
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return counts

    def prune_conversations(self, retention_days: int) -> int:
        """Clear stored prompt/response text past the retention window while
        keeping the request row (counts/timing). ``retention_days <= 0``
        wipes all captured text. Returns the number of rows cleared."""
        where = "prompt_text IS NOT NULL OR response_text IS NOT NULL"
        params: tuple = ()
        if retention_days > 0:
            cutoff = time.time() - (retention_days * 86400)
            where = f"enqueued_at < ? AND ({where})"
            params = (cutoff,)
        cur = self.conn.execute(
            f"UPDATE requests SET prompt_text=NULL, response_text=NULL WHERE {where}",
            params,
        )
        return cur.rowcount

    # Non-terminal request states. A row left in one of these by a previous
    # process is orphaned: the in-memory QueueManager that owned it is gone,
    # so it will never be dispatched, completed, or cancelled on its own.
    NONTERMINAL_REQUEST_STATES = ("queued", "swapping_model", "running")

    def reconcile_orphaned_requests(self, *, error: str) -> int:
        """Mark request rows stuck in a non-terminal state as failed.

        Called once at startup. The previous process may have crashed or been
        restarted mid-flight, leaving ``queued``/``swapping_model``/``running``
        rows that the fresh QueueManager has no record of — so the dashboard
        shows them as "running" forever and ``cancel`` (which looks up the
        in-memory request) can't touch them. Resolve them to ``failed`` so the
        UI is truthful and the rows stop masquerading as live work.

        Returns the number of rows reconciled.
        """
        placeholders = ",".join("?" for _ in self.NONTERMINAL_REQUEST_STATES)
        rows = self.query(
            f"SELECT id FROM requests WHERE status IN ({placeholders})",
            tuple(self.NONTERMINAL_REQUEST_STATES),
        )
        if not rows:
            return 0
        now = time.time()
        cur = self.conn.execute(
            f"UPDATE requests SET status='failed', error=?, finished_at=?"
            f" WHERE status IN ({placeholders})",
            (error, now, *self.NONTERMINAL_REQUEST_STATES),
        )
        self.log_event("requests_reconciled", {"count": cur.rowcount,
                                               "error": error})
        return cur.rowcount

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
