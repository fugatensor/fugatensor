"""Allocate contiguous leaf tensors without constructing Python element lists."""
from __future__ import annotations

import ctypes

from ._shape import normalize_size
from ._state import require_not_tracing
from .backends.host import HostStorage
from .dtypes import bool as bool_dtype, fp32
from .engine import current_engine
from .ir import FLOAT_DTYPES, TensorType
from .tensor import Tensor

_UNINITIALIZED = object()


def _create(size, fill_value, *, dtype, device, requires_grad):
    require_not_tracing()
    tensor_type = TensorType(normalize_size(size), dtype)
    if type(requires_grad) is not bool:
        raise TypeError("requires_grad must be bool")
    if requires_grad and tensor_type.dtype not in FLOAT_DTYPES:
        raise TypeError("requires_grad is only supported for floating tensors")
    scalar = (None if fill_value is _UNINITIALIZED else
              HostStorage.from_flat((fill_value,), shape=(), dtype=tensor_type.dtype))
    engine = current_engine(device)
    with engine._lock:
        engine._check_open()
        if scalar is None and engine._ordinal is not None:
            storage = engine.session.tensor(tensor_type.shape, tensor_type.dtype)
        else:
            storage = HostStorage(tensor_type.shape, tensor_type.dtype)
            if scalar is not None and storage.nbytes and any(scalar.to_buffer(copy=False)):
                # Seed one encoded element, then double the initialized prefix.
                # The destination owns a single allocation, without a full-size
                # temporary byte array or one Python object per element.
                copied = scalar.nbytes
                ctypes.memmove(storage.data_ptr, scalar.data_ptr, copied)
                while copied < storage.nbytes:
                    amount = min(copied, storage.nbytes - copied)
                    ctypes.memmove(storage.data_ptr + copied, storage.data_ptr, amount)
                    copied += amount
            if engine._ordinal is not None:
                storage = engine.session.tensor(storage.shape, storage.dtype).copy_from_buffer(
                    storage.to_buffer(copy=False))
        return Tensor(engine, storage, requires_grad=requires_grad)


def zeros(*size, dtype=fp32, device=None, requires_grad=False):
    """Return a leaf filled with zeros; sizes may be positional or one sequence."""
    return _create(size, False if dtype == bool_dtype else 0,
                   dtype=dtype, device=device, requires_grad=requires_grad)


def ones(*size, dtype=fp32, device=None, requires_grad=False):
    """Return a leaf filled with ones; sizes may be positional or one sequence."""
    return _create(size, True if dtype == bool_dtype else 1,
                   dtype=dtype, device=device, requires_grad=requires_grad)


def empty(*size, dtype=fp32, device=None, requires_grad=False):
    """Allocate a leaf tensor whose initial values are unspecified."""
    return _create(size, _UNINITIALIZED, dtype=dtype, device=device, requires_grad=requires_grad)


def full(size, fill_value, *, dtype=fp32, device=None, requires_grad=False):
    """Return a leaf filled with one scalar, encoded at the requested precision."""
    if not isinstance(size, (tuple, list)):
        raise TypeError("full size must be a tuple or list of dimensions")
    return _create((size,), fill_value, dtype=dtype, device=device, requires_grad=requires_grad)
