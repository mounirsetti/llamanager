"""A stopped slot holds no VRAM, and must not claim to.

`SlotView.size_gb` is derived from the slot's `spec`, and a spec outlives
a stop — it is the last spec the slot ran, which a restart reuses. Without
a state check a stopped slot kept reporting the size of the model it used
to hold, so `total_loaded_size_gb` never came back down after an unload:
the dashboard's VRAM-pressure proxy, and an admission input, read ~19 GB
occupied with nothing running and 1.9 GB actually in use on the card.

Found while checking what the MCP `unload_model` tool reports back, where
the consequence is sharper: an agent reading `server_status` would think
the GPU was full and decline to load a model.
"""
from __future__ import annotations

import pytest

from llamanager.server_pool import _STATES_HOLDING_VRAM


class _Runtime:
    def __init__(self, state):
        self.state = state
        self.current_model = "m.gguf" if state != "stopped" else None
        self.current_profile = None
        self.pid = 123 if state != "stopped" else None
        self.started_at = 1.0


class _Spec:
    def __init__(self, path):
        self.model_path = path


class _Slot:
    """Minimal stand-in for a ServerManager inside the pool."""

    def __init__(self, state, path):
        self.runtime = _Runtime(state)
        self.spec = _Spec(path)      # a spec survives a stop, on purpose
        self._port = 7201


def _pool_with(state, path):
    from llamanager.server_pool import ServerPool

    pool = ServerPool.__new__(ServerPool)
    pool._slots = {0: _Slot(state, path)}
    return pool


@pytest.fixture
def weights(tmp_path):
    """A file that *reports* 2 GB without occupying 2 GB.

    truncate() leaves a hole: st_size is what the code under test reads,
    while nothing is written. Writing the bytes for real costs 2 GB per
    use — and pytest's tmp_path lives under /tmp, which is a RAM-backed
    tmpfs here, so the honest-looking version filled memory instead of
    disk.
    """
    f = tmp_path / "model.gguf"
    with open(f, "wb") as fh:
        fh.truncate(2 * 1024 ** 3)
    assert f.stat().st_size == 2 * 1024 ** 3
    return f


@pytest.mark.parametrize("state", sorted(_STATES_HOLDING_VRAM))
def test_a_slot_with_a_live_process_reports_its_size(state, weights):
    pool = _pool_with(state, weights)
    assert pool.slots()[0].size_gb == pytest.approx(2.0, abs=0.01)
    assert pool.total_loaded_size_gb() == pytest.approx(2.0, abs=0.01)


@pytest.mark.parametrize("state", ["stopped", "crashed"])
def test_a_slot_with_no_process_reports_nothing(state, weights):
    """The spec is still there; the model is not."""
    pool = _pool_with(state, weights)
    assert pool.slots()[0].size_gb is None
    assert pool.total_loaded_size_gb() == 0.0


def test_the_total_comes_back_down_after_an_unload(weights):
    """The regression itself: load, then unload, and watch the number."""
    pool = _pool_with("running", weights)
    assert pool.total_loaded_size_gb() > 0

    pool._slots[0].runtime.state = "stopped"      # what stop_slot leaves behind
    pool._slots[0].runtime.pid = None
    assert pool._slots[0].spec is not None, "precondition: the spec survives"

    assert pool.total_loaded_size_gb() == 0.0, (
        "a stopped slot still counted toward the VRAM-pressure proxy")


def test_an_empty_slot_contributes_zero(weights):
    pool = _pool_with("running", weights)
    pool._slots[0].spec = None
    assert pool.slots()[0].size_gb is None
    assert pool.total_loaded_size_gb() == 0.0


def test_a_missing_weights_file_is_not_fatal(tmp_path):
    """The file can be deleted out from under a running slot."""
    pool = _pool_with("running", tmp_path / "gone.gguf")
    assert pool.slots()[0].size_gb is None
    assert pool.total_loaded_size_gb() == 0.0
