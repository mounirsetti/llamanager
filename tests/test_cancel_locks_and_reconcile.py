"""Tests for cancellation robustness, the model-loading lock, startup
reconciliation of orphaned rows, and the per-origin enable switch.

Pure-asyncio like test_queue_concurrency (the project doesn't enable
pytest-asyncio mode): each test wraps its body in ``asyncio.run``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from llamanager.auth import AuthManager, Origin, load_or_create_lookup_secret
from llamanager.config import Config
from llamanager.db import DB
from llamanager.queue_mgr import QueueManager
from llamanager.server_manager import ServerManager


def _make_cfg(tmp_path: Path, **over) -> Config:
    data = tmp_path / "lm"
    data.mkdir()
    (data / "logs").mkdir()
    (data / "models").mkdir()
    return Config(data_dir=data, **over)


def _make_origin(*, enabled: bool = True, name: str = "t") -> Origin:
    return Origin(id=1, name=name, priority=50, allowed_models=["*"],
                  is_admin=False, created_at=0.0, enabled=enabled)


def _plant_text_model(cfg: Config, name: str = "tiny.gguf") -> None:
    (cfg.models_dir / name).write_bytes(b"")


class _FakeProc:
    returncode = None


# --------------------------------------------------------------------------
# Resurrection race: cancel during a model swap must not leave a zombie.
# --------------------------------------------------------------------------

def test_cancel_during_swap_does_not_resurrect_running(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        # Pretend an engine is up with a *different* model loaded, so the
        # dispatch path takes the swap branch.
        sm.proc = _FakeProc()  # makes is_running True
        sm.runtime.current_model = "other.gguf"
        sm.runtime.current_profile = ""

        swap_entered = asyncio.Event()
        release_swap = asyncio.Event()

        async def blocking_swap(spec):
            swap_entered.set()
            await release_swap.wait()
            sm.runtime.current_model = spec.model_id
            return 0

        sm.swap = blocking_swap  # type: ignore[assignment]

        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_text_model(cfg)
            qm.start()
            try:
                req = await qm.enqueue(origin=_make_origin(),
                                       model_required="tiny.gguf")
                # Wait until the dispatcher is parked inside the swap.
                await asyncio.wait_for(swap_entered.wait(), timeout=2.0)
                assert qm._in_flight_count["text"] == 1

                # Cancel arrives mid-swap; the real handler's finally would
                # then refund the slot — simulate that here.
                assert qm.cancel(req.request_id) is True
                qm.mark_in_flight_done(
                    req, error="cancelled", cancelled=True,
                    prompt_tokens=None, completion_tokens=None,
                )
                assert qm._in_flight_count["text"] == 0

                # Let the swap finish. The dispatcher must NOT resurrect the
                # cancelled request into _in_flight / "running".
                release_swap.set()
                await asyncio.sleep(0.1)

                assert req.request_id not in qm._in_flight, \
                    "cancelled request was resurrected into _in_flight"
                assert req.status == "cancelled"
                assert qm._in_flight_count["text"] == 0, \
                    "slot count drifted — would let a second task run"
            finally:
                await qm.stop()

        asyncio.run(go())
    finally:
        db.close()


def test_handler_unwind_during_swap_no_cancel_does_not_resurrect(tmp_path):
    """The 'running task that cannot be cancelled' bug: a non-streaming
    request whose client disconnects / times out *during* a model swap. The
    handler coroutine unwinds and runs mark_in_flight_done WITHOUT ever
    setting qr.cancel. The dispatcher must still not resurrect it as running.
    """
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        sm.proc = _FakeProc()
        sm.runtime.current_model = "other.gguf"
        sm.runtime.current_profile = ""
        swap_entered = asyncio.Event()
        release_swap = asyncio.Event()

        async def blocking_swap(spec):
            swap_entered.set()
            await release_swap.wait()
            sm.runtime.current_model = spec.model_id
            return 0

        sm.swap = blocking_swap  # type: ignore[assignment]
        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_text_model(cfg)
            qm.start()
            try:
                req = await qm.enqueue(origin=_make_origin(),
                                       model_required="tiny.gguf")
                await asyncio.wait_for(swap_entered.wait(), timeout=2.0)

                # Handler unwinds (client gone) — finalizes WITHOUT cancel.
                assert req.cancel.is_set() is False
                qm.mark_in_flight_done(
                    req, error="client disconnected", cancelled=False,
                    prompt_tokens=None, completion_tokens=None,
                )
                assert req.request_id not in qm._by_id  # handler cleaned up

                release_swap.set()
                await asyncio.sleep(0.1)

                assert req.request_id not in qm._in_flight, \
                    "finalized request was resurrected into _in_flight"
                assert qm._in_flight_count["text"] == 0
                # And it shows as failed, not a phantom 'running'.
                row = db.query_one("SELECT status FROM requests WHERE id=?",
                                   (req.request_id,))
                assert row["status"] == "failed"
            finally:
                await qm.stop()

        asyncio.run(go())
    finally:
        db.close()


def test_snapshot_pending_excludes_finalized_heap_entry(tmp_path):
    """A 'pending task marked as done' must never appear: a heap entry whose
    request was finalized (done/failed) without being dispatched is a stale
    tombstone, not pending work."""
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)

        async def go():
            # Enqueue but never start the dispatcher, so it stays in the heap.
            req = await qm.enqueue(origin=_make_origin(), model_required=None)
            assert qm.snapshot()["pending"], "should be pending before finalize"

            # Simulate an early-abort finalize (e.g. pre-dispatch error) that
            # marks it done/failed and drops it from _by_id but leaves the
            # heap tombstone behind.
            qm.mark_in_flight_done(req, error=None, cancelled=False,
                                   prompt_tokens=None, completion_tokens=None)
            assert req.status == "done"
            assert req.request_id not in qm._by_id

            snap = qm.snapshot()
            assert snap["pending"] == [], \
                "finalized heap tombstone leaked into pending"
            assert snap["depth"] == 0

            # _pop_next must also drain it rather than re-dispatch it.
            assert qm._pop_next() is None

        asyncio.run(go())
    finally:
        db.close()


# --------------------------------------------------------------------------
# Model-loading lock.
# --------------------------------------------------------------------------

def test_lock_model_loading_rejects_swap(tmp_path):
    cfg = _make_cfg(tmp_path, lock_model_loading=True)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        sm.proc = _FakeProc()
        sm.runtime.current_model = "other.gguf"
        sm.runtime.current_profile = ""
        swapped = {"called": False}

        async def should_not_swap(spec):
            swapped["called"] = True
            return 0

        sm.swap = should_not_swap  # type: ignore[assignment]
        qm = QueueManager(cfg, db, sm)

        async def go():
            _plant_text_model(cfg)
            qm.start()
            try:
                req = await qm.enqueue(origin=_make_origin(),
                                       model_required="tiny.gguf")
                await asyncio.wait_for(req.ready.wait(), timeout=2.0)
                assert req.status == "failed"
                assert req.error and "locked" in req.error
                assert swapped["called"] is False, "lock must prevent the swap"
            finally:
                await qm.stop()

        asyncio.run(go())
    finally:
        db.close()


def test_lock_model_loading_allows_already_loaded_model(tmp_path):
    from llamanager.server_manager import resolve_spec
    cfg = _make_cfg(tmp_path, lock_model_loading=True)
    db = DB(cfg.db_path)
    try:
        _plant_text_model(cfg)
        sm = ServerManager(cfg, db)
        sm.proc = _FakeProc()
        # Align the loaded model AND profile so no swap is needed.
        spec = resolve_spec(cfg, model="tiny.gguf")
        sm.runtime.current_model = spec.model_id
        sm.runtime.current_profile = spec.profile_name
        qm = QueueManager(cfg, db, sm)

        async def go():
            qm.start()
            try:
                req = await qm.enqueue(origin=_make_origin(),
                                       model_required="tiny.gguf")
                await asyncio.wait_for(req.ready.wait(), timeout=2.0)
                # No swap needed → request runs even with the lock on.
                assert req.status == "running"
                assert req.error is None
            finally:
                qm.mark_in_flight_done(req, error=None, cancelled=False,
                                       prompt_tokens=None, completion_tokens=None)
                await qm.stop()

        asyncio.run(go())
    finally:
        db.close()


# --------------------------------------------------------------------------
# Startup reconciliation + stale-row cancel.
# --------------------------------------------------------------------------

def test_reconcile_orphaned_requests_marks_failed(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        for rid, status in [("a", "queued"), ("b", "running"),
                            ("c", "swapping_model"), ("d", "done")]:
            db.insert_request(request_id=rid, origin_id=1, model="m", priority=50)
            db.update_request_status(rid, status)

        n = db.reconcile_orphaned_requests(error="interrupted by daemon restart")
        assert n == 3
        rows = {r["id"]: r["status"] for r in db.query("SELECT id, status FROM requests")}
        assert rows == {"a": "failed", "b": "failed", "c": "failed", "d": "done"}
        # The terminal row keeps its status; reconciled rows carry the reason.
        err = db.query_one("SELECT error FROM requests WHERE id='b'")["error"]
        assert "restart" in err
    finally:
        db.close()


def test_cancel_clears_stale_db_row_not_in_memory(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        sm = ServerManager(cfg, db)
        qm = QueueManager(cfg, db, sm)
        # A non-terminal row with no in-memory request (e.g. orphaned).
        db.insert_request(request_id="ghost", origin_id=1, model="m", priority=50)
        db.update_request_status("ghost", "running")

        assert "ghost" not in qm._by_id
        assert qm.cancel("ghost") is True
        assert db.query_one("SELECT status FROM requests WHERE id='ghost'")["status"] \
            == "cancelled"

        # An unknown id still reports failure (404 upstream).
        assert qm.cancel("does-not-exist") is False
        # An already-terminal row is not "cancellable" again.
        db.insert_request(request_id="finished", origin_id=1, model="m", priority=50)
        db.update_request_status("finished", "done")
        assert qm.cancel("finished") is False
    finally:
        db.close()


def _insert_download(db: DB, did: str, status: str) -> None:
    db.execute(
        "INSERT INTO downloads(id, source, files_json, status, started_at)"
        " VALUES(?,?,?,?,?)",
        (did, "hf://x/y", "[]", status, 0.0),
    )


def test_reconcile_orphaned_downloads_marks_failed(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        for did, status in [("a", "running"), ("b", "done"),
                            ("c", "failed"), ("d", "cancelled")]:
            _insert_download(db, did, status)

        n = db.reconcile_orphaned_downloads(error="interrupted by daemon restart")
        assert n == 1
        rows = {r["id"]: r["status"]
                for r in db.query("SELECT id, status FROM downloads")}
        # Only the mid-flight 'running' row is resolved; terminals untouched.
        assert rows == {"a": "failed", "b": "done",
                        "c": "failed", "d": "cancelled"}
        row = db.query_one("SELECT error, finished_at FROM downloads WHERE id='a'")
        assert "restart" in row["error"]
        assert row["finished_at"] is not None
    finally:
        db.close()


def test_cancel_pull_clears_orphaned_download_row(tmp_path):
    from llamanager.registry import Registry

    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        reg = Registry(cfg, db)
        # A 'running' row with no in-memory cancel Event (orphaned by a
        # previous process). Cancel must resolve it, not silently no-op.
        _insert_download(db, "ghost", "running")
        assert "ghost" not in reg._cancel_flags
        assert reg.cancel_pull("ghost") is True
        assert db.query_one(
            "SELECT status FROM downloads WHERE id='ghost'")["status"] == "failed"

        # Unknown id still reports failure (404 upstream).
        assert reg.cancel_pull("does-not-exist") is False
        # An already-terminal row is not "cancellable" again.
        _insert_download(db, "finished", "done")
        assert reg.cancel_pull("finished") is False
    finally:
        db.close()


# --------------------------------------------------------------------------
# Per-origin enable switch.
# --------------------------------------------------------------------------

def test_origin_enabled_roundtrip_and_default(tmp_path):
    cfg = _make_cfg(tmp_path)
    db = DB(cfg.db_path)
    try:
        secret = load_or_create_lookup_secret(cfg.data_dir)
        am = AuthManager(db, lookup_secret=secret)
        origin, key = am.create_origin(name="client", allowed_models=["*"])
        assert origin.enabled is True  # new origins enabled by default

        updated = am.set_enabled(origin.id, False)
        assert updated is not None and updated.enabled is False
        # Verify-by-key reflects the change (auth still succeeds).

        async def go():
            o = await am.verify(key)
            assert o is not None and o.enabled is False

        asyncio.run(go())

        again = am.set_enabled(origin.id, True)
        assert again is not None and again.enabled is True
        assert am.set_enabled(9999, True) is None  # unknown origin
    finally:
        db.close()


def test_disabled_origin_gets_403_at_inference_endpoint(app):
    """A disabled origin authenticates but is refused at the task path with a
    403 (not a misleading 401). The gate sits in _origin_from_request, before
    the request is ever enqueued."""
    from fastapi.testclient import TestClient
    am = app.state.auth
    origin, key = am.create_origin(name="blocked", allowed_models=["*"])
    am.set_enabled(origin.id, False)
    # No `with` → lifespan/dispatcher don't run, which is fine: the 403 is
    # raised before enqueue, so the request never needs the queue.
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]
