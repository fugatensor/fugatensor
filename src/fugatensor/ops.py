"""Builtins are visible Fuga expressions, including during derivative tracing."""
from ._jit import primitive
from .dtypes import bool as bool_dtype
from .tracing import TraceTensor
from .tensor import Tensor


@primitive
def add(x, y):
    return x + y


@primitive
def multiply(x, y):
    return x * y


@primitive
def negative(x):
    return x * -1


@primitive
def subtract(x, y):
    return x + y * -1


@primitive
def square(x):
    return x * x


@primitive
def cast(x, dtype):
    return x.astype(dtype)


@primitive
def _reshape(x, shape):
    return x.reshape(shape)


def reshape(input, shape):
    """Reshape a tensor, inferring at most one dimension specified as -1."""
    if not isinstance(input, (Tensor, TraceTensor)):
        raise TypeError("reshape requires a Tensor")
    if not isinstance(shape, (tuple, list)):
        raise TypeError("reshape shape must be a tuple or list of dimensions")
    return input.reshape(shape)


def _filled_like(x, value):
    if not isinstance(x, TraceTensor):
        raise TypeError('like operators expect a Tensor')
    scalar = x.context.graph.constant(bool(value) if x.dtype is bool_dtype else value,
                                      dtype=x.dtype)
    return x.context.wrap(x.context.graph.splat(scalar, x.shape))


@primitive
def zeros_like(x):
    return _filled_like(x, 0)


@primitive
def ones_like(x):
    return _filled_like(x, 1)


def _pair(x, y):
    if isinstance(x, TraceTensor):
        return x.context, x.value, x._operand(y)
    if isinstance(y, TraceTensor):
        return y.context, y._operand(x), y.value
    raise TypeError('operation requires a Tensor operand')


@primitive
def divide(x, y):
    context, left, right = _pair(x, y)
    return context.wrap(context.graph.div(left, right))


@primitive
def compare(x, y, predicate='eq'):
    context, left, right = _pair(x, y)
    return context.wrap(context.graph.compare(left, right, predicate))


@primitive
def where(condition, x, y):
    context, left, right = _pair(x, y)
    if not isinstance(condition, TraceTensor) or condition.context is not context:
        raise TypeError('where condition must be a bool Tensor in the same graph')
    return context.wrap(context.graph.select(condition.value, left, right))


@primitive
def exp(x):
    return x.context.wrap(x.context.graph.exp(x.value))


@primitive
def log(x):
    return x.context.wrap(x.context.graph.log(x.value))


@primitive
def sqrt(x):
    return x.context.wrap(x.context.graph.sqrt(x.value))


@primitive
def _broadcast(x, shape, dimensions):
    if dimensions is None:
        dimensions = tuple(range(len(shape) - len(x.shape), len(shape)))
    return x.context.wrap(x.context.graph.broadcast_in_dim(x.value, shape, dimensions))


def broadcast_in_dim(x, shape, dimensions):
    return _broadcast(x, tuple(shape), tuple(dimensions))


def broadcast_to(x, shape):
    return _broadcast(x, tuple(shape), None)


@primitive
def _transpose(x, permutation):
    return x.context.wrap(x.context.graph.transpose(x.value, permutation))


def transpose(x, permutation=None):
    return _transpose(x, tuple(reversed(range(len(x.shape)))) if permutation is None else tuple(permutation))


def _axes(axis, rank):
    axes = tuple(range(rank)) if axis is None else (axis,) if type(axis) is int else tuple(axis)
    if any(type(value) is not int or not -rank <= value < rank for value in axes):
        raise ValueError('reduction axes must be integer dimensions in range')
    axes = tuple(value % rank for value in axes)
    if len(set(axes)) != len(axes):
        raise ValueError('reduction axes must be unique')
    return tuple(sorted(axes))


def _axis_parameter(axis):
    if axis is None or type(axis) is int:
        return axis
    axes = tuple(axis)
    if any(type(value) is not int for value in axes):
        raise ValueError('reduction axes must be integer dimensions in range')
    return axes


@primitive
def _reduce(x, opcode, axis, keepdims, dtype):
    from .ir import ScalarType
    axes = _axes(axis, len(x.shape))
    if type(keepdims) is not bool:
        raise TypeError('keepdims must be bool')
    if dtype is not None:
        x = x.astype(dtype)
    graph = x.context.graph
    result = getattr(graph, opcode)(x.value, axes)
    shape = tuple(1 if axis in axes else size for axis, size in enumerate(x.shape))
    if isinstance(result.type, ScalarType):
        result = graph.splat(result, shape if keepdims else ())
    elif keepdims:
        result = graph.reshape(result, shape)
    return x.context.wrap(result)


def sum(x, axis=None, *, keepdims=False, dtype=None):
    return _reduce(x, 'reduce_sum', _axis_parameter(axis), keepdims, dtype)


def amax(x, axis=None, *, keepdims=False):
    return _reduce(x, 'reduce_max', _axis_parameter(axis), keepdims, None)


def mean(x, axis=None, *, keepdims=False, dtype=None):
    import math
    axes = _axes(axis, len(x.shape))
    count = math.prod(x.shape[axis] for axis in axes)
    reduced = sum(x, axes, keepdims=keepdims, dtype=dtype)
    if count == 0:
        return reduced / count
    try:
        return reduced / count
    except ValueError:
        return reduced * (1.0 / count)


@primitive
def matmul(a, b, *, transpose_a=False, transpose_b=False):
    if not isinstance(a, TraceTensor) or not isinstance(b, TraceTensor) or a.context is not b.context:
        raise TypeError('matmul requires two Tensors in the same graph')
    return a.context.wrap(a.context.graph.matmul(a.value, b.value,
                         transpose_a=transpose_a, transpose_b=transpose_b))
