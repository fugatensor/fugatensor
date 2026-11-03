"""Generate public-operation compositions and independent polynomial derivatives.

Inputs are small dyadic values, so the default depth-1/2 matrix is exact in all
three floating formats. Precision edge cases use the backend rounding oracle;
this matrix isolates graph composition and differentiation from rounding error.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import cycle, islice, product
import math

import fugatensor as ft

from .selection import select

DTYPES = {name: getattr(ft, name) for name in ('fp32', 'fp16', 'bf16', 'int32', 'int64', 'bool')}
FLOATS = ('fp32', 'fp16', 'bf16')
NUMERIC = (*FLOATS, 'int32', 'int64')
SHAPES = {'scalar': (), 'vector': (4,), 'matrix': (2, 2),
          'singleton': (1, 4, 1), 'empty': (0, 2)}


class Polynomial:
    """A bivariate exact polynomial; independent of Fuga IR and autograd rules."""

    def __init__(self, terms):
        self.terms = {powers: Fraction(value) for powers, value in terms.items() if value}

    @staticmethod
    def constant(value):
        return Polynomial({(0, 0): value})

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            other = self.constant(other)
        terms = dict(self.terms)
        for powers, value in other.terms.items():
            terms[powers] = terms.get(powers, 0) + value
        return Polynomial(terms)

    def __neg__(self):
        return Polynomial({powers: -value for powers, value in self.terms.items()})

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            other = self.constant(other)
        terms = {}
        for (a, b), left in self.terms.items():
            for (c, d), right in other.terms.items():
                powers = (a + c, b + d)
                terms[powers] = terms.get(powers, 0) + left * right
        return Polynomial(terms)

    def derivative(self, axis):
        terms = {}
        for powers, value in self.terms.items():
            degree = powers[axis]
            if degree:
                result = list(powers)
                result[axis] -= 1
                terms[tuple(result)] = value * degree
        return Polynomial(terms)

    def evaluate(self, x, y):
        return sum((coefficient * Fraction(x)**a * Fraction(y)**b
                    for (a, b), coefficient in self.terms.items()), Fraction(0))


@dataclass(frozen=True)
class Operation:
    name: str
    apply: object
    reference: object
    dtypes: tuple = NUMERIC


# One registration extends single-operation, all ordered compositions, dtype,
# forward and derivative coverage. No expected combination is hand-written.
OPERATIONS = (
    Operation('add', ft.add, lambda x, y: x + y),
    Operation('subtract', ft.subtract, lambda x, y: x - y),
    Operation('multiply', ft.multiply, lambda x, y: x * y),
    Operation('negative', lambda x, y: ft.negative(x), lambda x, y: -x),
    Operation('square', lambda x, y: ft.square(x), lambda x, y: x * x),
    Operation('reshape', lambda x, y: ft.reshape(x, (math.prod(x.shape),)).reshape(x.shape),
              lambda x, y: x, tuple(DTYPES)),
    Operation('cast', lambda x, y: ft.cast(ft.cast(x, ft.fp32), x.dtype),
              lambda x, y: x, tuple(DTYPES)),
    Operation('zeros_like', lambda x, y: ft.zeros_like(x),
              lambda x, y: Polynomial.constant(0), tuple(DTYPES)),
    Operation('ones_like', lambda x, y: ft.ones_like(x),
              lambda x, y: Polynomial.constant(1), tuple(DTYPES)),
)


@dataclass(frozen=True)
class Case:
    operations: tuple

    @property
    def name(self):
        return ' -> '.join(operation.name for operation in self.operations)

    def apply(self, x, y):
        for operation in self.operations:
            x = operation.apply(x, y)
        return x

    def polynomial(self):
        x, y = Polynomial({(1, 0): 1}), Polynomial({(0, 1): 1})
        for operation in self.operations:
            x = operation.reference(x, y)
        return x


def plan_matrix():
    names = select('FUGATENSOR_TEST_OPS', [op.name for op in OPERATIONS])
    depths = select('FUGATENSOR_TEST_DEPTHS', ('1', '2', '3'), default=('1', '2'))
    dtypes = select('FUGATENSOR_TEST_DTYPES', DTYPES)
    shapes = select('FUGATENSOR_TEST_SHAPES', SHAPES)
    orders = tuple(int(value) for value in select('FUGATENSOR_TEST_ORDERS', ('0', '1', '2', '3'),
                                                default=('0', '1', '2')))
    selected = False
    for dtype in dtypes:
        operations = [op for op in OPERATIONS if op.name in names and dtype in op.dtypes]
        cases = tuple(Case(items) for depth in depths for items in product(operations, repeat=int(depth)))
        if cases:
            selected = True
            for shape in shapes:
                yield dtype, SHAPES[shape], cases, orders if dtype in FLOATS else (0,)
    if not selected:
        raise ValueError('selected operations and dtypes have no supported combinations')


def input_values(dtype, shape):
    if dtype == 'bool':
        values = (True, False, False, True), (False, True, False, True)
    elif dtype in FLOATS:
        values = (-1, -0.5, 0.5, 1), (0.5, -1, 1, -0.5)
    else:
        values = (-2, -1, 1, 2), (1, 2, -2, -1)
    return tuple(tuple(islice(cycle(items), math.prod(shape))) for items in values)


def nested_values(values, shape):
    values = iter(values)
    def build(dimensions):
        if not dimensions:
            return next(values)
        return [build(dimensions[1:]) for _ in range(dimensions[0])]
    return build(shape)
