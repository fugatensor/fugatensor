"""Generated public Tensor graphs and independent flat numerical references."""
from itertools import combinations, product

import fugatensor as ft
from tests.support.tensor_primitives import dtypes, values


class SharedPorts(ft.fun):
    def forward(self, x, y):
        shared = x + y
        return shared * x, y * 3, shared


class RepeatedPorts(ft.fun):
    def forward(self, x):
        shared = x * x
        return shared, shared, x


def selection_matrix():
    subsets = tuple(ports for count in range(1, 4) for ports in combinations(range(3), count))
    types = tuple(name for name in dtypes() if name in ('fp32', 'int32'))
    return product(types, ((), (2, 3)), (False, True), (False, True), subsets)


def port_values(shape, dtype, shared_input, phase):
    left = values(shape, dtype, phase)
    right = left if shared_input else values(shape, dtype, phase + 2)
    shared = [x + y for x, y in zip(left, right)]
    return left, right, ([s * x for s, x in zip(shared, left)],
                         [y * 3 for y in right], shared)
