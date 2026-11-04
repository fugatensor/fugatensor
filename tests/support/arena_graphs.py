"""Deferred graph identity cases and independent host references."""
from itertools import product

from .tensor_primitives import transpose_reference, values


def identity_matrix():
    return product(('fp32', 'int32'), ((), (2, 3)),
                   ('contiguous', 'transposed'), (False, True),
                   (False, True), ('none', 'left', 'both'))


def bindings(shape, dtype, layout, shared, phase):
    left = values(shape, dtype, phase)
    right = left if shared else values(shape, dtype, phase + 2)
    if layout == 'transposed':
        permutation = tuple(reversed(range(len(shape))))
        left = transpose_reference(left, shape, permutation)
        right = transpose_reference(right, shape, permutation)
    return tuple(2 * x - 3 * y for x, y in zip(left, right)), \
        tuple(2 * x + 3 * y for x, y in zip(left, right))
