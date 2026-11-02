"""Tensor handles and shape conversion; arithmetic is delegated to ASK JIT."""
from __future__ import annotations

import math
from .dtypes import DType
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
    __slots__ = ("_engine", "_storage", "_shape", "_dtype", "_pending", "_requires_grad", "_grad",
                 "_history", "_history_index", "__weakref__")

    def __init__(self, engine, storage, *, requires_grad=False):
        if type(requires_grad) is not bool:
            raise TypeError("requires_grad must be bool")
        if requires_grad and storage.dtype not in ("f16", "bf16", "f32"):
            raise TypeError("requires_grad is only supported for floating tensors")
        self._engine = engine
        self._storage = storage
        self._shape = tuple(storage.shape)
        self._dtype = DType(storage.dtype)
        self._pending = None
        self._requires_grad = requires_grad
        self._grad = None
        self._history = None
        self._history_index = None

    @classmethod
    def _from_pending(cls, engine, shape, dtype, pending):
        result = cls.__new__(cls)
        result._engine, result._storage = engine, None
        result._shape, result._dtype = tuple(shape), DType(dtype)
        result._pending = pending
        result._requires_grad = False
        result._grad = result._history = result._history_index = None
        return result

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad(self):
        return self._grad

    @property
    def is_leaf(self):
        return self._history is None

    def backward(self, gradient=None, *, retain_graph=None, create_graph=False):
        from .autograd import backward
        return backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph)

    def zero_grad(self, *, set_to_none=True):
        from .autograd import zero_grad
        zero_grad(self, set_to_none=set_to_none)

    def detach(self):
        """Return a handle sharing this buffer, without gradient history."""
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
    def device(self) -> str:
        return self._engine.device

    def tolist(self):
        """Wait for computation as needed and return nested Python values."""
        return _nested(self._engine.to_flat(self), self.shape)

    def to_buffer(self, *, copy=True):
        """Return a read-only host byte buffer without converting its elements."""
        require_not_tracing()
        self.engine._check_tensor(self)
        if type(copy) is not bool:
            raise TypeError("copy must be bool")
        if self.device == 'cpu':
            return self.storage.to_buffer(copy=copy)
        if not copy:
            raise ValueError("CUDA to_buffer requires copy=True for device-to-host transfer")
        return memoryview(self.storage.to_host_bytes())

    def astype(self, dtype: DType | str) -> Tensor:
        return self._engine.elementwise("cast", (self,), dtype=normalize_dtype(dtype))

    def _binary(self, operation, other, *, reverse=False):
        if isinstance(other, Tensor):
            if other.engine is not self.engine:
                raise ValueError("tensor operands must belong to the same Engine")
            if other.shape != self.shape:
                raise ValueError("tensor operands require equal shapes; broadcasting is not supported")
            if other.dtype != self.dtype:
                raise TypeError("tensor operands require equal dtypes; use astype explicitly")
        elif type(other) not in (bool, int, float):
            return NotImplemented
        if self.dtype == "bool":
            raise TypeError("bool arithmetic is not defined; use astype to select a numeric type")
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

    def __eq__(self, other):
        raise TypeError("tensor comparisons are not implemented; compare downloaded values explicitly")

    def __ne__(self, other):
        return self.__eq__(other)

    def __repr__(self):
        return f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, device={self.device!r})"
