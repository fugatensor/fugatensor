"""Builtins are visible Fuga expressions, including during derivative tracing."""
from ._jit import jit
from .dtypes import bool as bool_dtype
from .tracing import TraceTensor
from .tensor import Tensor


@jit
def add(x, y):
    return x + y


@jit
def multiply(x, y):
    return x * y


@jit
def negative(x):
    return x * -1


@jit
def subtract(x, y):
    return x + y * -1


@jit
def square(x):
    return x * x


@jit
def cast(x, dtype):
    return x.astype(dtype)


@jit
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


@jit
def zeros_like(x):
    return _filled_like(x, 0)


@jit
def ones_like(x):
    return _filled_like(x, 1)
