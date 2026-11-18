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


def full_like(x, fill_value, *, dtype=None, device=None):
    """Return a tensor with the same shape (and by default dtype/device) filled with fill_value."""
    from ._state import active_trace
    from .devices import normalize_device
    from .tracing import TraceTensor
    from . import ops as _ops
    target_dtype = x.dtype if dtype is None else dtype
    trace = active_trace.get()
    if isinstance(x, TraceTensor) or trace is not None:
        if device is not None and trace is not None and trace.engine.device != normalize_device(device):
            raise RuntimeError("traced tensor creation device must match the active function device")
        base = x if dtype is None else x.astype(target_dtype)
        return _ops._filled_like_prim(base, fill_value)
    if not isinstance(x, Tensor):
        raise TypeError("full_like expects a Tensor")
    target_device = x.device if device is None else device
    return _create(x.shape, fill_value, dtype=target_dtype, device=target_device)


def _trace_or_bind_nullary(prim_fn, args, *, device):
    from ._state import active_trace
    from .devices import normalize_device
    trace = active_trace.get()
    if trace is not None:
        if device is not None and trace.engine.device != normalize_device(device):
            raise RuntimeError("traced tensor creation device must match the active function device")
        return prim_fn(*args)
    return prim_fn._bind_on_engine(current_engine(device), *args)


def _coerce_dtype(dtype):
    from . import dtypes as _dtypes
    mapping = {'bool': _dtypes.bool, 'i32': _dtypes.int32, 'i64': _dtypes.int64,
               'f32': _dtypes.fp32, 'f16': _dtypes.fp16, 'bf16': _dtypes.bf16}
    if isinstance(dtype, _dtypes.DType):
        return dtype
    if dtype in mapping:
        return mapping[dtype]
    raise TypeError(f"unsupported dtype {dtype!r}")


def iota(shape, *, dimension=0, dtype=fp32, device=None):
    """Create a tensor filled with element coordinates along one axis."""
    from .ir.types import normalize_dtype
    from . import ops as _ops
    normalized_shape = normalize_size(shape)
    if not normalized_shape:
        raise ValueError("iota requires rank at least one")
    if type(dimension) is not int:
        raise TypeError("iota dimension must be an integer")
    axis = dimension + len(normalized_shape) if dimension < 0 else dimension
    if not 0 <= axis < len(normalized_shape):
        raise ValueError("iota dimension out of range")
    dtype_obj = _coerce_dtype(dtype)
    if normalize_dtype(dtype_obj) == normalize_dtype(bool_dtype):
        raise TypeError("iota requires a numeric dtype")
    return _trace_or_bind_nullary(_ops._iota_prim, (normalized_shape, axis, dtype_obj), device=device)


def arange(start, stop=None, step=1, *, dtype=None, device=None):
    """Return evenly spaced values within a half-open interval."""
    from .dtypes import int64
    from . import ops as _ops
    if stop is None:
        start, stop = 0, start
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"arange {name} must be a real int or float")
    if step == 0:
        raise ValueError("arange step cannot be zero")
    if dtype is None:
        dtype = int64 if all(isinstance(v, int) for v in (start, stop, step)) else fp32
    dtype_obj = _coerce_dtype(dtype)
    if (step > 0 and start >= stop) or (step < 0 and start <= stop):
        return _create((0,), 0, dtype=dtype_obj, device=device)
    if all(isinstance(v, int) for v in (start, stop, step)):
        count = (stop - start + step - 1) // step if step > 0 else (start - stop + (-step) - 1) // (-step)
    else:
        count = math.ceil((stop - start) / step)
        if count <= 0:
            return _create((0,), 0, dtype=dtype_obj, device=device)
    return _trace_or_bind_nullary(_ops._arange_prim, (count, start, step, dtype_obj), device=device)


def linspace(start, stop, steps, *, dtype=fp32, device=None):
    """Return ``steps`` evenly spaced samples over ``[start, stop]``."""
    from .ir.types import normalize_dtype
    from . import ops as _ops
    for name, value in (("start", start), ("stop", stop)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"linspace {name} must be a real number")
    if isinstance(steps, bool) or type(steps) is not int or steps < 0:
        raise ValueError("linspace steps must be a nonnegative integer")
    dtype_obj = _coerce_dtype(dtype)
    if normalize_dtype(dtype_obj) == normalize_dtype(bool_dtype):
        raise TypeError("linspace requires a numeric dtype")
    if steps == 0:
        return _create((0,), 0, dtype=dtype_obj, device=device)
    return _trace_or_bind_nullary(
        _ops._linspace_prim, (float(start), float(stop), steps, dtype_obj), device=device)


def eye(n, m=None, *, dtype=fp32, device=None):
    """Create a 2-D matrix with ones on the diagonal and zeros elsewhere."""
    from . import ops as _ops
    if m is None:
        m = n
    if isinstance(n, bool) or type(n) is not int or n < 0 or isinstance(m, bool) or type(m) is not int or m < 0:
        raise ValueError("eye dimensions must be nonnegative integers")
    dtype_obj = _coerce_dtype(dtype)
    return _trace_or_bind_nullary(_ops._eye_prim, (n, m, dtype_obj), device=device)
