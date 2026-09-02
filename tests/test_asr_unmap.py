"""The ASR load path must not hand memory-mapped tensors to the GPU.

``from_pretrained`` mmaps the safetensors file, so every parameter is a
view onto page cache. Measured on this box (ROCm 7, R9700, torch 2.10):

    file-backed tensor -> GPU   ~1-2 s   *per tensor, regardless of size*
    the same bytes from heap    ~0 s

Whisper large-v3 has 587 tensors, so ``.to(device)`` took about twenty
minutes. The worker never answered ``/healthz`` inside its 180 s window
and the daemon reported "ASR worker failed to become healthy in time" —
which read like a crash, but it was copying the whole time.

``_unmap`` clones the model off the mapping first (~0.15 s for the whole
model), after which the move is instant.

torch lives in the engine venvs, not llamanager's, so these drive the
helper with a stub tensor. What is being pinned is the helper's contract
— every tensor comes back owning its own storage — plus the wiring, since
a helper nobody calls fixes nothing.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

MODULES = ["llamanager.engines._asr_worker", "llamanager.engines._asr_runner"]


class _Tensor:
    """Minimal stand-in: a tensor that may be a view onto shared storage."""

    def __init__(self, values, storage_id):
        self.values = list(values)
        self.storage_id = storage_id      # shared => mmap-backed view
        self.dtype = "float16"
        self.shape = (len(self.values),)

    def clone(self):
        # A real clone allocates fresh heap storage; model that with a new id.
        t = _Tensor(self.values, storage_id=object())
        t.dtype, t.shape = self.dtype, self.shape
        return t


class _Param:
    def __init__(self, tensor):
        self.data = tensor


class _FakeModel:
    """Stands in for a transformers model: named parameters and buffers."""

    def __init__(self, params, buffers=None):
        self._p = params
        self._b = buffers or {}

    def named_parameters(self):
        return list(self._p.items())

    def named_buffers(self):
        return list(self._b.items())


def _mmap_backed_model():
    """Every tensor a view onto one backing buffer — the mmap situation."""
    backing = object()
    params = {f"layer{i}.weight": _Param(_Tensor([i, i + 1], backing))
              for i in range(4)}
    buffers = {"pos_embed": _Param(_Tensor([9, 9], backing))}
    return _FakeModel(params, buffers), backing


@pytest.mark.parametrize("module_name", MODULES)
def test_every_tensor_ends_up_owning_its_storage(module_name):
    """This is the whole fix: nothing may still point into the mapping."""
    unmap = importlib.import_module(module_name)._unmap
    model, backing = _mmap_backed_model()

    unmap(model)

    seen = []
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        assert p.data.storage_id is not backing, (
            f"{name} still points into the mapped file — its GPU copy would "
            f"take the slow file-backed path")
        seen.append(p.data.storage_id)
    # ...and each clone is independent, not one shared replacement buffer.
    assert len(set(map(id, seen))) == len(seen)


@pytest.mark.parametrize("module_name", MODULES)
def test_values_dtype_and_shape_survive(module_name):
    """Cloning must not disturb the weights it copies."""
    unmap = importlib.import_module(module_name)._unmap
    model, _ = _mmap_backed_model()
    before = {n: (list(p.data.values), p.data.dtype, p.data.shape)
              for n, p in list(model.named_parameters())
              + list(model.named_buffers())}

    unmap(model)

    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        assert (list(p.data.values), p.data.dtype, p.data.shape) == before[name]


@pytest.mark.parametrize("module_name", MODULES)
def test_buffers_are_covered_not_just_parameters(module_name):
    """Whisper's positional embeddings are buffers; missing them would leave
    a file-backed tensor in the model and the slow path with it."""
    unmap = importlib.import_module(module_name)._unmap
    model, backing = _mmap_backed_model()

    unmap(model)

    buffers = dict(model.named_buffers())
    assert buffers, "fixture should carry a buffer"
    assert all(p.data.storage_id is not backing for p in buffers.values())


@pytest.mark.parametrize("module_name", MODULES)
def test_it_returns_the_model_so_it_chains_into_to(module_name):
    """Both call sites read `_unmap(from_pretrained(...)).to(device)`."""
    unmap = importlib.import_module(module_name)._unmap
    model, _ = _mmap_backed_model()
    assert unmap(model) is model


@pytest.mark.parametrize("module_name", MODULES)
def test_the_load_path_actually_calls_it(module_name):
    """A helper nobody calls fixes nothing — pin the wiring too."""
    src = inspect.getsource(importlib.import_module(module_name))
    assert "_unmap(WhisperForConditionalGeneration.from_pretrained" in src, (
        f"{module_name} loads the model without lifting it off the mmap")
