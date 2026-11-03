"""Allocate contiguous leaf tensors without constructing Python element lists."""
from __future__ import annotations

import ctypes
import math

from ._shape import normalize_size
from ._state import require_not_tracing
from .backends.host import HostStorage
from ._ask.runtime import _encode
from .dtypes import bool as bool_dtype, fp32
from .engine import current_engine
from .ir import TensorType
from .tensor import Tensor

_UNINITIALIZED = object()


def _create(size, fill_value, *, dtype, device):
    require_not_tracing()
    tensor_type = TensorType(normalize_size(size), dtype)
    engine = current_engine(device)
    with engine._lock:
        engine._check_open()
        if engine._ordinal is not None:
            encoded = None if fill_value is _UNINITIALIZED else _encode((fill_value,), tensor_type.dtype, 1)
            storage = engine.session.tensor(tensor_type.shape, tensor_type.dtype)
            if encoded is not None:
                storage.copy_from_buffer(encoded * math.prod(tensor_type.shape))
        else:
            scalar = (None if fill_value is _UNINITIALIZED else
                      HostStorage.from_flat((fill_value,), shape=(), dtype=tensor_type.dtype))
            storage = HostStorage(tensor_type.shape, tensor_type.dtype)
            if scalar is not None and storage.nbytes and any(scalar.to_buffer(copy=False)):
                # Fill one CPU allocation by doubling the initialized prefix.
                copied = scalar.nbytes
                ctypes.memmove(storage.data_ptr, scalar.data_ptr, copied)
                while copied < storage.nbytes:
                    amount = min(copied, storage.nbytes - copied)
                    ctypes.memmove(storage.data_ptr + copied, storage.data_ptr, amount)
                    copied += amount
        return Tensor(engine, storage)


def zeros(*size, dtype=fp32, device=None):
    """Return a leaf filled with zeros; sizes may be positional or one sequence."""
    return _create(size, False if dtype == bool_dtype else 0,
                   dtype=dtype, device=device)


def ones(*size, dtype=fp32, device=None):
    """Return a leaf filled with ones; sizes may be positional or one sequence."""
    return _create(size, True if dtype == bool_dtype else 1,
                   dtype=dtype, device=device)


def empty(*size, dtype=fp32, device=None):
    """Allocate a leaf tensor whose initial values are unspecified."""
    return _create(size, _UNINITIALIZED, dtype=dtype, device=device)


def full(size, fill_value, *, dtype=fp32, device=None):
    """Return a leaf filled with one scalar, encoded at the requested precision."""
    if not isinstance(size, (tuple, list)):
        raise TypeError("full size must be a tuple or list of dimensions")
    return _create((size,), fill_value, dtype=dtype, device=device)
