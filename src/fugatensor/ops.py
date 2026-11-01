"""Builtins are visible Fuga expressions, including during backward tracing."""
from ._jit import jit
from .tracing import TraceTensor


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


def _filled_like(x, value):
    if not isinstance(x, TraceTensor):
        raise TypeError('like operators expect a Tensor')
    scalar = x.context.graph.constant(value, dtype=x.dtype)
    return x.context.wrap(x.context.graph.splat(scalar, x.shape))


@jit
def zeros_like(x):
    return _filled_like(x, 0)


@jit
def ones_like(x):
    return _filled_like(x, 1)
