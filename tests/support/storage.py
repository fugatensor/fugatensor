"""Generate leaf creation cases independently of storage implementation."""
from itertools import product
import math

import fugatensor as ft

from .ops_matrix import nested_values

SHAPES = ((), (2,), (2, 3), (0,))
CREATORS = ('tensor', 'zeros', 'ones', 'full', 'empty', 'from_buffer')
CPU_DTYPES = (ft.bool, ft.int32, ft.int64, ft.fp32, ft.fp16)
SCALAR_BITS = {
    ft.bool: b'\x01', ft.int32: b'\x02\x00\x00\x00',
    ft.int64: b'\x02\x00\x00\x00\x00\x00\x00\x00',
    ft.fp32: b'\x00\x00\x00\x40', ft.fp16: b'\x00\x40', ft.bf16: b'\x00\x40',
}


def creation_matrix(dtypes):
    return product(dtypes, SHAPES, CREATORS)


def create(name, shape, dtype):
    value = True if dtype is ft.bool else 2
    if name == 'tensor':
        return ft.tensor(nested_values([value] * math.prod(shape), shape), dtype=dtype)
    if name == 'full':
        return ft.full(shape, value, dtype=dtype)
    if name == 'from_buffer':
        return ft.from_buffer(SCALAR_BITS[dtype] * math.prod(shape), shape=shape, dtype=dtype)
    return getattr(ft, name)(shape, dtype=dtype)


def expected(name, shape, dtype):
    value = 0 if name == 'zeros' else 1 if name == 'ones' or dtype is ft.bool else 2
    return [value] * math.prod(shape)
