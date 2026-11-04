"""Generated representations of equivalent static primitive parameters."""
from itertools import product


def reduction_parameters():
    for dtype, shape, maximum, keepdims in product(
            ('fp32', 'int32'), ((2, 3), (1, 2, 1)), (False, True), (False, True)):
        for axes in ((len(shape) - 1,), tuple(range(len(shape)))):
            yield dtype, shape, maximum, keepdims, axes


def axis_forms(axes, rank):
    yield 'tuple', axes
    yield 'list', list(axes)
    yield 'generator', (axis for axis in axes)
    yield 'negative tuple', tuple(axis - rank for axis in axes)
    yield 'reversed tuple', tuple(reversed(axes))
    if len(axes) == 1:
        yield 'integer', axes[0]
        yield 'negative integer', axes[0] - rank
    if axes == tuple(range(rank)):
        yield 'all dimensions', None


def broadcast_parameters():
    return product(('fp32', 'int32'), ((3,), (1, 2)))


def shape_forms(shape):
    yield 'tuple', shape
    yield 'list', list(shape)
    yield 'generator', (size for size in shape)


def invalid_axes(rank):
    for axis in (rank, -rank - 1):
        yield axis, ValueError, 'integer dimensions in range'
    for axes in ((0, 0), (0, -rank)):
        yield axes, ValueError, 'unique'
    for item in (True, 1.5, '0', object(), (0,)):
        yield [item], ValueError, 'integer dimensions in range'
    for axis in (True, 1.5):
        yield axis, TypeError, 'iterable'
