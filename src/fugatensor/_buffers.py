"""Validate byte transport independently of CPU or CUDA tensor storage."""
import math
import sys

_FORMATS = {
    'bool': (1, ('?',)), 'i32': (4, ('i', 'l')), 'i64': (8, ('q', 'l')),
    'f32': (4, ('f',)), 'f16': (2, ('e',)), 'bf16': (2, ()),
}


def contiguous_bytes(data, *, shape, dtype):
    view = memoryview(data)
    if not view.c_contiguous:
        raise ValueError('tensor buffers must be C-contiguous')
    if any(dim is None for dim in shape):
        raise ValueError('tensor buffers require a concrete shape')
    width, formats = _FORMATS[dtype]
    nbytes = math.prod(shape) * width
    if view.nbytes != nbytes:
        raise ValueError('buffer byte count does not match the tensor shape and dtype')
    format_ = view.format
    prefix = format_[0] if format_ and format_[0] in '@=<>!' else '@'
    format_ = format_[1:] if format_ and format_[0] in '@=<>!' else format_
    raw = format_ in ('B', 'b', 'c') and view.itemsize == 1
    native_endian = prefix in ('@', '=') or prefix == ('<' if sys.byteorder == 'little' else '>')
    if not raw and (format_ not in formats or view.itemsize != width or not native_endian):
        raise TypeError('typed buffer format must match the tensor dtype and native byte order; use bytes for raw bits')
    return view.cast('B')
