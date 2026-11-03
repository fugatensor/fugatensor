"""Generate CUDA Array Interface storage descriptions independently of the runtime."""
from itertools import product
import math

TYPES = {'bool': ('|b1', 1), 'i32': ('<i4', 4), 'i64': ('<i8', 8),
         'f32': ('<f4', 4), 'f16': ('<f2', 2)}
SHAPES = ((), (4,), (2, 3), (1, 4, 1), (0, 2))
LAYOUTS = ('implicit', 'contiguous', 'strided')


def storage_matrix():
    for (dtype, (typestr, width)), shape, layout in product(TYPES.items(), SHAPES, LAYOUTS):
        strides = tuple(math.prod(shape[axis + 1:]) * width for axis in range(len(shape)))
        contiguous = True
        if layout == 'strided':
            axes = [axis for axis, size in enumerate(shape) if size > 1]
            if axes:
                axis = axes[-1]
                strides = tuple(stride * 2 if index == axis else stride
                                for index, stride in enumerate(strides))
                contiguous = False
        yield dtype, shape, layout, contiguous, {
            'version': 3, 'shape': shape, 'typestr': typestr,
            'data': (4096 if math.prod(shape) else 0, False),
            'strides': None if layout == 'implicit' else strides, 'stream': None,
        }
