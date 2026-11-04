"""Tensor handles and shape conversion; arithmetic is delegated to ASK JIT."""
from __future__ import annotations

import math
from .dtypes import DType
from .devices import device as Device, resolve_device
from .ir import ScalarType, TensorType

from ._state import require_not_tracing


def normalize_dtype(dtype: DType | str) -> str:
    """Normalize the public Fuga element type."""
    return ScalarType(dtype).dtype


def normalize_shape(shape) -> tuple[int, ...]:
    result = TensorType(shape, "f32").shape
    if any(dimension is None for dimension in result):
        raise ValueError("FugaTensor requires concrete tensor dimensions")
    return result


def normalize_data(data, shape=None) -> tuple[list[bool | int | float], tuple[int, ...]]:
    """Flatten rectangular Python data, optionally assigning an explicit shape."""
    def flatten(value):
        if isinstance(value, (list, tuple)):
            if not value:
                return [], (0,)
            parts = [flatten(item) for item in value]
            item_shape = parts[0][1]
            if any(part_shape != item_shape for _, part_shape in parts):
                raise ValueError("tensor data must be rectangular; ragged sequences are unsupported")
            return [item for values, _ in parts for item in values], (len(value), *item_shape)
        if type(value) not in (bool, int, float):
            raise TypeError("tensor data must contain Python bool, int or float values")
        return [value], ()

    values, inferred = flatten(data)
    selected = inferred if shape is None else normalize_shape(shape)
    if math.prod(selected) != len(values):
        raise ValueError("tensor shape does not match the number of data elements")
    return values, selected


def _nested(values, shape):
    if len(values) != math.prod(shape):
        raise RuntimeError("tensor storage returned an unexpected number of elements")
    iterator = iter(values)

    def build(dimensions):
        if not dimensions:
            return next(iterator)
        return [build(dimensions[1:]) for _ in range(dimensions[0])]

    return build(shape)


class Tensor:
    """A deferred expression or buffer, with explicit shape and element type.

    Arithmetic records expressions for the engine's JIT. Creating or printing
    a Tensor does not copy its values back from a device.
    """
    __slots__ = ("_engine", "_storage", "_shape", "_dtype", "_pending", "_origin", "__weakref__")

    def __init__(self, engine, storage):
        self._engine = engine
        self._storage = storage
        self._shape = tuple(storage.shape)
        self._dtype = DType(storage.dtype)
        self._pending = self._origin = None

    @classmethod
    def _from_pending(cls, engine, shape, dtype, pending):
        from ._lazy import Result
        result = cls.__new__(cls)
        result._engine, result._storage = engine, None
        result._shape, result._dtype = tuple(shape), DType(dtype)
        result._pending = pending
        # Provenance survives materialization; a detached alias has no origin.
        result._origin = pending if isinstance(pending, Result) else None
        return result

    def detach(self):
        """Return a numerical alias that stops differentiation through this value."""
        require_not_tracing()
        self.engine._check_tensor(self)
        if self._storage is None:
            from ._lazy import Alias
            return Tensor._from_pending(self.engine, self.shape, self.dtype, Alias(self))
        return Tensor(self.engine, self._storage)

    @property
    def engine(self):
        return self._engine

    @property
    def storage(self):
        require_not_tracing()
        self.engine._check_tensor(self)
        if self._storage is None:
            self.engine.evaluate(self)
        return self._storage

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return resolve_device(self._engine.device)

    def tolist(self):
        """Wait for computation as needed and return nested Python values."""
        return _nested(self._engine.to_flat(self), self.shape)

    def to_buffer(self, *, copy=True):
        """Return a read-only host byte buffer without converting its elements."""
        require_not_tracing()
        self.engine._check_tensor(self)
        if type(copy) is not bool:
            raise TypeError("copy must be bool")
        if self.device.type == 'cpu':
            return self.storage.to_buffer(copy=copy)
        if not copy:
            raise ValueError("CUDA to_buffer requires copy=True for device-to-host transfer")
        return memoryview(self.storage.to_host_bytes())

    def astype(self, dtype: DType | str) -> Tensor:
        return self._engine.elementwise("cast", (self,), dtype=normalize_dtype(dtype))

    def reshape(self, *shape) -> Tensor:
        """Return a deferred tensor with the same elements and a new shape."""
        from ._shape import infer_reshape
        from .ops import _reshape
        return _reshape(self, infer_reshape(shape, math.prod(self.shape)))

    def _binary(self, operation, other, *, reverse=False):
        if not isinstance(other, Tensor) and type(other) not in (bool, int, float):
            return NotImplemented
        arguments = (other, self) if reverse else (self, other)
        return self._engine.elementwise(operation, arguments)

    def __add__(self, other):
        return self._binary("add", other)

    def __radd__(self, other):
        return self._binary("add", other, reverse=True)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __rmul__(self, other):
        return self._binary("mul", other, reverse=True)

    def __neg__(self):
        from .ops import negative
        return negative(self)

    def __sub__(self, other):
        from .ops import subtract
        return subtract(self, other)

    def __rsub__(self, other):
        from .ops import subtract
        return subtract(other, self)

    def square(self):
        from .ops import square
        return square(self)

    def __bool__(self):
        if self.shape != ():
            raise TypeError('control-flow conditions require a scalar Tensor with shape=()')
        return bool(self.tolist())

    def __truediv__(self, other):
        from .ops import divide
        return divide(self, other)

    def __rtruediv__(self, other):
        from .ops import divide
        return divide(other, self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul
        return matmul(other, self)

    def __eq__(self, other):
        from .ops import compare
        return compare(self, other, 'eq')

    def __ne__(self, other):
        from .ops import compare
        return compare(self, other, 'ne')

    def __lt__(self, other):
        from .ops import compare
        return compare(self, other, 'lt')

    def __le__(self, other):
        from .ops import compare
        return compare(self, other, 'le')

    def __gt__(self, other):
        from .ops import compare
        return compare(self, other, 'gt')

    def __ge__(self, other):
        from .ops import compare
        return compare(self, other, 'ge')

    def sum(self, axis=None, *, keepdims=False, dtype=None):
        from .ops import sum
        return sum(self, axis, keepdims=keepdims, dtype=dtype)

    def amax(self, axis=None, *, keepdims=False):
        from .ops import amax
        return amax(self, axis, keepdims=keepdims)

    def mean(self, axis=None, *, keepdims=False, dtype=None):
        from .ops import mean
        return mean(self, axis, keepdims=keepdims, dtype=dtype)

    def transpose(self, permutation=None):
        from .ops import transpose
        return transpose(self, permutation)

    @property
    def T(self):
        return self.transpose()

    def broadcast_to(self, shape):
        from .ops import broadcast_to
        return broadcast_to(self, shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, device={self.device!r})"
