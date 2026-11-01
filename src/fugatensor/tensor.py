"""Tensor handles and shape conversion; arithmetic is delegated to ASK JIT."""
from __future__ import annotations

import math
from .ir import ScalarType, TensorType

from ._state import require_not_tracing


def normalize_dtype(dtype: str) -> str:
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
    """A buffer owned by an Engine, with explicit shape and element type.

    Tensor arithmetic dispatches to the engine's JIT entry. Creating or printing
    a Tensor does not copy its values back from a device.
    """
    __slots__ = ("_engine", "_storage")

    def __init__(self, engine, storage):
        self._engine = engine
        self._storage = storage

    @property
    def engine(self):
        return self._engine

    @property
    def storage(self):
        require_not_tracing()
        return self._storage

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._storage.shape)

    @property
    def dtype(self) -> str:
        return self._storage.dtype

    @property
    def device(self) -> str:
        return self._engine.device

    def tolist(self):
        """Wait for computation as needed and return nested Python values."""
        return _nested(self._engine.to_flat(self), self.shape)

    def astype(self, dtype: str) -> Tensor:
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

    def __bool__(self):
        raise TypeError("Tensor truth values are unsupported; use static metadata for control flow")

    def __eq__(self, other):
        raise TypeError("tensor comparisons are not implemented; compare downloaded values explicitly")

    def __ne__(self, other):
        return self.__eq__(other)

    def __repr__(self):
        return f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, device={self.device!r})"
