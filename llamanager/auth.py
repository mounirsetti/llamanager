"""Origin / API-key auth.

Per spec §4.5 and §12.3:
- Keys are hashed with argon2id at rest.
- Each row also carries a `key_lookup` = HMAC-SHA256(server_secret, key)
  so verify is O(1): we look up the single candidate row by lookup id
  and run argon2id on that row only. The HMAC secret never leaves the
  data dir; without it, the lookup column is useless to an attacker who
  steals a DB snapshot.
- Verified keys are cached in-memory by cleartext so we skip argon2 on
  the hot path entirely. Cache lives only inside this process; rotated
  or revoked keys disappear naturally on the next request.
- Argon2 verifications are run via asyncio.to_thread so a single auth
  attempt never blocks the event loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from .db import DB

log = logging.getLogger(__name__)

_HASHER = PasswordHasher()
_KEY_PREFIX = "lm_"  # so a stray key in a logfile is identifiable.
# Tokens are urlsafe(32) -> 43 chars + "lm_" = 46. Bound conservatively.
_KEY_MIN_LEN = 16
_KEY_MAX_LEN = 256

# How long a known-bad key is remembered (per process) so repeated probes
# don't trigger argon2 work for every origin row.
_NEG_CACHE_TTL_S = 60.0
_NEG_CACHE_MAX = 4096


@dataclass
class Origin:
    id: int
    name: str
    priority: int
    allowed_models: list[str]
    is_admin: bool
    created_at: float

    @classmethod
    def from_row(cls, row: Any) -> "Origin":
        return cls(
            id=row["id"],
            name=row["name"],
            priority=row["priority"],
            allowed_models=json.loads(row["allowed_models_json"]),
            is_admin=bool(row["is_admin"]),
            created_at=row["created_at"],
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "allowed_models": self.allowed_models,
            "is_admin": self.is_admin,
            "created_at": self.created_at,
        }


def generate_key() -> str:
    return _KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return _HASHER.hash(key)


def load_or_create_lookup_secret(data_dir: Path) -> bytes:
    """Read (or create) the server-only HMAC key used to derive `key_lookup`
    ids for origins. Stored 0600 next to the other secrets in `data_dir`.

    Don't rotate this casually: existing key_lookup ids become stale if you
    do, and AuthManager falls back to the legacy O(N) scan until every key
    is rotated.
    """
    p = data_dir / ".key-lookup-secret"
    if p.exists():
        try:
            data = p.read_bytes()
            if len(data) >= 16:
                return data
        except OSError:
            pass
    secret = secrets.token_bytes(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(secret)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return secret


class AuthManager:
    def __init__(self, db: DB, *, lookup_secret: bytes,
                 default_priority: int = 50):
        if not lookup_secret or len(lookup_secret) < 16:
            raise ValueError("lookup_secret must be at least 16 bytes")
        self.db = db
        self.default_priority = default_priority
        self._lookup_secret = lookup_secret
        # cache: api_key (cleartext) -> (origin_id, expires_at)
        self._cache: dict[str, tuple[int, float]] = {}
        self._cache_ttl = 300.0
        # Negative cache keyed on a sha256 of the cleartext (so we don't
        # store the key itself in memory longer than necessary). Any key
        # listed here gets a fast `None` for _NEG_CACHE_TTL_S, avoiding
        # the per-row argon2id verify storm an attacker could otherwise
        # induce by replaying the same garbage token.
        self._neg_cache: dict[str, float] = {}
        self._warn_legacy_rows()

    def _key_lookup(self, key: str) -> str:
        return hmac.new(self._lookup_secret, key.encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def _warn_legacy_rows(self) -> None:
        row = self.db.query_one(
            "SELECT COUNT(*) AS c FROM origins WHERE key_lookup IS NULL"
        )
        if row and row["c"] > 0:
            log.warning(
                "auth: %d origin row(s) predate the v2 key_lookup migration "
                "and still require an O(N) argon2 scan on every auth. "
                "Rotate those keys via /admin/origins/<id>/rotate-key to "
                "move them onto the fast path.", row["c"],
            )

    # ---- bootstrap ----
    def ensure_bootstrap(self) -> str | None:
        """Create the bootstrap admin origin if no origins exist.

        Returns the cleartext key (caller must surface it once)."""
        row = self.db.query_one("SELECT COUNT(*) AS c FROM origins")
        if row and row["c"] > 0:
            return None
        key = generate_key()
        self._insert(
            name="bootstrap",
            key=key,
            priority=100,
            allowed_models=["*"],
            is_admin=True,
        )
        self.db.log_event("bootstrap_created", {"name": "bootstrap"})
        return key

    # ---- CRUD ----
    def list_origins(self) -> list[Origin]:
        rows = self.db.query("SELECT * FROM origins ORDER BY id")
        return [Origin.from_row(r) for r in rows]

    def get_origin(self, origin_id: int) -> Origin | None:
        row = self.db.query_one("SELECT * FROM origins WHERE id=?", (origin_id,))
        return Origin.from_row(row) if row else None

    def get_origin_by_name(self, name: str) -> Origin | None:
        row = self.db.query_one("SELECT * FROM origins WHERE name=?", (name,))
        return Origin.from_row(row) if row else None

    def create_origin(self, *, name: str, priority: int | None = None,
                      allowed_models: list[str] | None = None,
                      is_admin: bool = False) -> tuple[Origin, str]:
        key = generate_key()
        origin_id = self._insert(
            name=name,
            key=key,
            priority=priority if priority is not None else self.default_priority,
            allowed_models=allowed_models or ["default"],
            is_admin=is_admin,
        )
        origin = self.get_origin(origin_id)
        assert origin
        self.db.log_event("origin_created", {"id": origin_id, "name": name,
                                             "is_admin": is_admin})
        return origin, key

    def _insert(self, *, name: str, key: str, priority: int,
                allowed_models: list[str], is_admin: bool) -> int:
        cur = self.db.execute(
            "INSERT INTO origins(name, key_hash, key_lookup, priority,"
            " allowed_models_json, is_admin, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, hash_key(key), self._key_lookup(key), priority,
             json.dumps(allowed_models), 1 if is_admin else 0, time.time()),
        )
        return int(cur.lastrowid)

    def update_origin(self, origin_id: int, *, priority: int | None = None,
                      allowed_models: list[str] | None = None,
                      is_admin: bool | None = None) -> Origin | None:
        sets: list[str] = []
        vals: list[Any] = []
        if priority is not None:
            sets.append("priority=?"); vals.append(priority)
        if allowed_models is not None:
            sets.append("allowed_models_json=?"); vals.append(json.dumps(allowed_models))
        if is_admin is not None:
            sets.append("is_admin=?"); vals.append(1 if is_admin else 0)
        if not sets:
            return self.get_origin(origin_id)
        vals.append(origin_id)
        self.db.execute(
            f"UPDATE origins SET {', '.join(sets)} WHERE id=?", tuple(vals)
        )
        self.db.log_event("origin_updated", {"id": origin_id})
        self._cache.clear()
        return self.get_origin(origin_id)

    def delete_origin(self, origin_id: int) -> bool:
        cur = self.db.execute("DELETE FROM origins WHERE id=?", (origin_id,))
        ok = cur.rowcount > 0
        if ok:
            self.db.log_event("origin_deleted", {"id": origin_id})
            self._cache.clear()
        return ok

    def rotate_key(self, origin_id: int) -> str | None:
        if not self.get_origin(origin_id):
            return None
        key = generate_key()
        self.db.execute(
            "UPDATE origins SET key_hash=?, key_lookup=? WHERE id=?",
            (hash_key(key), self._key_lookup(key), origin_id),
        )
        self.db.log_event("origin_key_rotated", {"id": origin_id})
        self._cache.clear()
        return key

    # ---- verification ----
    def _neg_check(self, key: str, now: float) -> bool:
        """Return True if `key` is in the negative cache (still fresh)."""
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        exp = self._neg_cache.get(h)
        if exp is None:
            return False
        if exp <= now:
            self._neg_cache.pop(h, None)
            return False
        return True

    def _neg_remember(self, key: str, now: float) -> None:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if len(self._neg_cache) >= _NEG_CACHE_MAX:
            # Cheap eviction: drop the oldest half. Not LRU, but bounded.
            for old in sorted(self._neg_cache, key=self._neg_cache.get)[
                : _NEG_CACHE_MAX // 2
            ]:
                self._neg_cache.pop(old, None)
        self._neg_cache[h] = now + _NEG_CACHE_TTL_S

    async def verify(self, key: str) -> Origin | None:
        if not isinstance(key, str) or not key:
            return None
        # Cheap rejections before we touch argon2id at all.
        if len(key) < _KEY_MIN_LEN or len(key) > _KEY_MAX_LEN:
            return None
        if not key.startswith(_KEY_PREFIX):
            return None
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached[1] > now:
            return self.get_origin(cached[0])
        if self._neg_check(key, now):
            return None

        # Fast path: at most one row matches the lookup id.
        lookup = self._key_lookup(key)
        candidates = self.db.query(
            "SELECT * FROM origins WHERE key_lookup=?", (lookup,)
        )
        # Slow fallback for rows that predate v2 (key_lookup IS NULL).
        # Once those keys are rotated, this scan becomes empty.
        candidates.extend(self.db.query(
            "SELECT * FROM origins WHERE key_lookup IS NULL"
        ))

        for r in candidates:
            try:
                # Argon2id is intentionally CPU-heavy; offload so a single
                # auth doesn't stall every other coroutine on the loop.
                await asyncio.to_thread(_HASHER.verify, r["key_hash"], key)
            except VerifyMismatchError:
                continue
            except Exception:
                log.exception("argon2 verify error on origin id=%s", r["id"])
                continue
            self._cache[key] = (r["id"], now + self._cache_ttl)
            return Origin.from_row(r)
        # No origin matched — remember this so repeated probes are cheap.
        self._neg_remember(key, now)
        return None
