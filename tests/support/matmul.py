"""Matmul input geometry and independent, cancellation-aware error bounds."""
from itertools import product
import math

from tests.support.covering import pairwise
from tests.support.tensor_primitives import indices, offset

HOST_AXES = (('fp32', 'fp16'), ((), (2,)), (3, 4, 5, 8),
             (15, 16, 17, 31, 32, 33), (3, 17), (False, True), (False, True), (False, True))


def host_cases(*, exhaustive=False):
    return pairwise(HOST_AXES, exhaustive=exhaustive)


def case_id(case):
    dtype, batch, m, n, k, ta, tb, shared = case
    return f'{dtype}-batch{batch}-m{m}-n{n}-k{k}-t{int(ta)}{int(tb)}-shared{int(shared)}'


def reference_and_scale(left, right, ashape, bshape, ta=False, tb=False):
    m, k = ashape[-2:][::-1] if ta else ashape[-2:]
    _, n = bshape[-2:][::-1] if tb else bshape[-2:]
    result, scale = [], []
    for batch in indices(ashape[:-2]):
        for row, column in product(range(m), range(n)):
            terms = [left[offset((*batch, inner, row) if ta else (*batch, row, inner), ashape)] *
                     right[offset((*batch, column, inner) if tb else (*batch, inner, column), bshape)]
                     for inner in range(k)]
            result.append(math.fsum(terms) if all(map(math.isfinite, terms)) else sum(terms))
            scale.append(sum(abs(value) for value in terms))
    return (*ashape[:-2], m, n), result, scale


def assert_matmul_values(actual, expected, scale, *, k, dtype, tf32=False):
    """Bound input rounding, f32 reduction and final output rounding separately."""
    assert len(actual) == len(expected) == len(scale)
    unit = 2**-24
    gamma = (2 * k + 1) * unit / (1 - (2 * k + 1) * unit)
    input_unit = 2**-11 if tf32 else 0
    output_unit = {'fp32': unit, 'fp16': 2**-11, 'bf16': 2**-8}[dtype]
    floor = {'fp32': 2**-149, 'fp16': 2**-24, 'bf16': 2**-133}[dtype]
    for index, (value, target, magnitude) in enumerate(zip(actual, expected, scale)):
        if math.isnan(target):
            assert math.isnan(value), (index, value, target)
        elif math.isinf(target):
            assert value == target, (index, value, target)
        else:
            bound = ((2 * input_unit + input_unit**2 + gamma * (1 + input_unit)**2) * magnitude
                     + output_unit * abs(target) + floor)
            assert math.isfinite(value) and abs(value - target) <= bound, (index, value, target, bound)
