"""Warp-boundary inputs and an IEEE maximum reference independent of ASK."""
from itertools import combinations, product

from tests.support.covering import pairwise
import math

from tests.support.tensor_primitives import dtypes, indices, offset, transpose_reference


AXES = tuple(axes for count in range(1, 4) for axes in combinations(range(3), count))
COMPOSITIONS = ('plain', 'negative_relu')
LAYOUTS = ((0, 1, 2), (2, 0, 1))
WIDTHS = (0, 1, 31, 32, 33, 63, 64, 65)


def reduction_matrix(*, exhaustive=False):
    shapes = tuple((7, 2, width) for width in WIDTHS) + ((0, 2, 33), (7, 0, 33))
    return pairwise((maximum_dtypes(), shapes, LAYOUTS), exhaustive=exhaustive)


def maximum_dtypes():
    return tuple(name for name in dtypes() if name != 'bool')


def maximum_values(shape, dtype):
    """Keep exceptional-value rows separate while varying every reduced index."""
    result = []
    floating = dtype in ('fp32', 'fp16', 'bf16')
    bits = 64 if dtype == 'int64' else 32
    for group, row, column in indices(shape):
        position = row * shape[2] + column
        count = shape[1] * shape[2]
        if floating:
            patterns = (
                position % 7 - 3,
                math.nan if position == 0 else -2,
                math.nan if position == count - 1 else 2,
                -0.0 if position % 2 else 0.0,
                -0.0,
                math.inf if position == count - 1 else 1,
                -math.inf,
            )
        else:
            high = 1 << (bits - 2)
            patterns = (
                position % 7 - 3,
                -(1 << (bits - 1)) + position,
                (1 << (bits - 1)) - 1 - position,
                high + position,
                -high + position,
                (1 << (bits // 2)) - 1 + position,
                -position - 1,
            )
        result.append(float(patterns[group]) if floating else patterns[group])
    return result


def storage_layout(data, shape, permutation):
    return (transpose_reference(data, shape, permutation),
            tuple(shape[axis] for axis in permutation),
            tuple(permutation.index(axis) for axis in range(len(shape))))


def ieee_maximum(values):
    if any(isinstance(value, float) and math.isnan(value) for value in values):
        return math.nan
    result = max(values)
    if isinstance(result, float) and result == 0:
        return 0.0 if any(value == 0 and math.copysign(1, value) > 0 for value in values) else -0.0
    return result


def maximum_reference(data, shape, axes, composition='plain'):
    remaining = tuple(axis for axis in range(len(shape)) if axis not in axes)
    target = tuple(shape[axis] for axis in remaining)
    results = []
    for output in indices(target):
        entries = []
        for reduced in indices(tuple(shape[axis] for axis in axes)):
            coordinates = dict(zip(remaining, output)) | dict(zip(axes, reduced))
            value = data[offset(tuple(coordinates[axis] for axis in range(len(shape))), shape)]
            zero = 0.0 if isinstance(value, float) else 0
            entries.append(zero if composition == 'negative_relu' and value < 0 else value)
        value = ieee_maximum(entries)
        results.append(-value if composition == 'negative_relu' else value)
    return target, results
