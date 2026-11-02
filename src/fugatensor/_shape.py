"""Concrete size arguments shared by tensor creation and reshape."""
from __future__ import annotations

import math
import operator


def _dimensions(args, *, infer):
    if not args:
        raise TypeError("provide dimensions or an empty tuple for a scalar")
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = args[0]
    result = []
    for value in args:
        if isinstance(value, bool):
            raise TypeError("tensor dimensions must be integers, not bool")
        try:
            dimension = operator.index(value)
        except TypeError as error:
            raise TypeError("tensor dimensions must be integers") from error
        if not (-1 if infer else 0) <= dimension < 2**63:
            raise ValueError("tensor dimensions must be nonnegative signed i64 integers"
                             + (" or one inferred -1" if infer else ""))
        result.append(dimension)
    return tuple(result)


def normalize_size(args) -> tuple[int, ...]:
    """Normalize collected ``*size`` arguments; a single tuple/list is accepted."""
    return _dimensions(args, infer=False)


def infer_reshape(shape, numel) -> tuple[int, ...]:
    """Normalize collected ``*shape`` arguments and infer at most one ``-1``."""
    shape = _dimensions(shape, infer=True)
    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("reshape element count must be a nonnegative integer")
    inferred = shape.count(-1)
    if inferred > 1:
        raise ValueError("reshape permits at most one inferred -1 dimension")
    known = math.prod(dimension for dimension in shape if dimension != -1)
    if inferred:
        if known == 0:
            raise ValueError("reshape cannot infer -1 when another dimension is zero")
        if numel % known:
            raise ValueError("reshape dimensions do not match the number of elements")
        dimension = numel // known
        if dimension >= 2**63:
            raise ValueError("inferred reshape dimension overflows signed i64")
        shape = tuple(dimension if value == -1 else value for value in shape)
    elif known != numel:
        raise ValueError("reshape dimensions do not match the number of elements")
    return shape
