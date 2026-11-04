"""Custom functions whose temporary graph ownership can be observed with weakrefs."""
import fugatensor as ft
from .tensor_primitives import transpose_reference


class ForwardScale(ft.fun):
    def __init__(self, factor):
        self.factor = factor

    def forward(self, value):
        return value * self.factor


class ExplicitScale(ForwardScale):
    def vjp(self, primals, outputs, cotangents):
        return (cotangents[0] * self.factor,)


DEFINITIONS = (ForwardScale, ExplicitScale)


def concurrent_output(value, branch):
    shared = value + 1
    if branch == 0:
        return shared * 2
    if branch == 1:
        return shared * shared
    if branch == 2:
        return shared.T
    return shared.reshape(tuple(reversed(value.shape)))


def concurrent_references(data, shape):
    shifted = [value + 1 for value in data]
    reversed_shape = tuple(reversed(shape))
    permutation = tuple(reversed(range(len(shape))))
    return ((shape, [value * 2 for value in shifted]),
            (shape, [value * value for value in shifted]),
            (reversed_shape, transpose_reference(shifted, shape, permutation)),
            (reversed_shape, shifted))
