"""Builtins are visible Fuga expressions, including during derivative tracing."""
import builtins as _bi
import math as _math

from ._jit import primitive, variadic_primitive
from ._state import active_trace
from .dtypes import bool as bool_dtype, fp32, int32, int64
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
    if _bi.any(type(value) is not int or not -rank <= value < rank for value in axes):
        raise ValueError('reduction axes must be integer dimensions in range')
    axes = tuple(value % rank for value in axes)
    if len(set(axes)) != len(axes):
        raise ValueError('reduction axes must be unique')
    return tuple(sorted(axes))


def _axis_parameter(axis):
    if axis is None or type(axis) is int:
        return axis
    axes = tuple(axis)
    if _bi.any(type(value) is not int for value in axes):
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


@primitive
def _filled_like_prim(x, value):
    return _filled_like(x, value)


@primitive
def _unary(x, opcode):
    if not isinstance(x, TraceTensor):
        raise TypeError(f'{opcode} expects a Tensor')
    return x.context.wrap(getattr(x.context.graph, opcode)(x.value))


@primitive
def _binary(x, y, opcode):
    context, left, right = _pair(x, y)
    return context.wrap(getattr(context.graph, opcode)(left, right))


def abs(x):
    return _unary(x, 'abs')


def floor(x):
    return _unary(x, 'floor')


def ceil(x):
    return _unary(x, 'ceil')


def round(x):
    return _unary(x, 'round')


def sin(x):
    return _unary(x, 'sin')


def cos(x):
    return _unary(x, 'cos')


def tanh(x):
    return _unary(x, 'tanh')


def erf(x):
    return _unary(x, 'erf')


def bitwise_not(x):
    return _unary(x, 'bitwise_not')


def pow(x, y):
    return _binary(x, y, 'pow')


def floordiv(x, y):
    return _binary(x, y, 'floordiv')


def mod(x, y):
    return _binary(x, y, 'mod')


def maximum(x, y):
    return _binary(x, y, 'maximum')


def minimum(x, y):
    return _binary(x, y, 'minimum')


def clamp(x, min=None, max=None):
    if min is not None:
        x = maximum(x, min)
    if max is not None:
        x = minimum(x, max)
    return x


clip = clamp


def bitwise_and(x, y):
    return _binary(x, y, 'bitwise_and')


def bitwise_or(x, y):
    return _binary(x, y, 'bitwise_or')


def bitwise_xor(x, y):
    return _binary(x, y, 'bitwise_xor')


def shift_left(x, y):
    return _binary(x, y, 'shift_left')


def shift_right(x, y):
    return _binary(x, y, 'shift_right')


def logical_not(x):
    if not hasattr(x, 'dtype'):
        raise TypeError('logical_not expects a Tensor')
    xb = x if x.dtype is bool_dtype else x.astype(bool_dtype)
    return _unary(xb, 'bitwise_not')


@primitive
def _logical_bin(x, y, opcode):
    if isinstance(x, TraceTensor):
        xb = x if x.dtype is bool_dtype else x.context.wrap(x.context.graph.cast(x.value, dtype='bool'))
        if isinstance(y, TraceTensor):
            if y.context is not xb.context:
                raise ValueError('traced operands must belong to the same graph')
            if y.shape != xb.shape:
                raise ValueError('tensor operands require equal shapes; use broadcast_to explicitly')
            yb = y if y.dtype is bool_dtype else y.context.wrap(y.context.graph.cast(y.value, dtype='bool'))
            return xb.context.wrap(getattr(xb.context.graph, opcode)(xb.value, yb.value))
        if type(y) not in (bool, int, float):
            raise TypeError('logical operations accept tensors or Python scalars')
        return xb.context.wrap(getattr(xb.context.graph, opcode)(xb.value, xb._operand(_bi.bool(y))))
    if isinstance(y, TraceTensor):
        if type(x) not in (bool, int, float):
            raise TypeError('logical operations accept tensors or Python scalars')
        yb = y if y.dtype is bool_dtype else y.context.wrap(y.context.graph.cast(y.value, dtype='bool'))
        return yb.context.wrap(getattr(yb.context.graph, opcode)(yb._operand(_bi.bool(x)), yb.value))
    raise TypeError('operation requires a Tensor operand')


def logical_and(x, y):
    return _logical_bin(x, y, 'bitwise_and')


def logical_or(x, y):
    return _logical_bin(x, y, 'bitwise_or')


def logical_xor(x, y):
    return _logical_bin(x, y, 'bitwise_xor')


@primitive
def _iota_prim(shape, dimension, dtype):
    context = active_trace.get()
    return context.wrap(context.graph.iota(shape, dimension, dtype=str(dtype)))


@primitive
def _arange_prim(count, start, step, dtype):
    context = active_trace.get()
    dtype_name = str(dtype)
    idx = context.graph.iota((count,), 0, dtype=dtype_name)
    out = idx
    if step != 1:
        scale = context.graph.splat(context.graph.constant(step, dtype=dtype_name), (count,))
        out = context.graph.mul(out, scale)
    if start != 0:
        offset = context.graph.splat(context.graph.constant(start, dtype=dtype_name), (count,))
        out = context.graph.add(out, offset)
    return context.wrap(out)


@primitive
def _linspace_prim(start, stop, steps, dtype):
    context = active_trace.get()
    dtype_name = str(dtype)
    if steps == 1:
        return context.wrap(context.graph.splat(context.graph.constant(start, dtype=dtype_name), (1,)))
    idx = context.graph.iota((steps,), 0, dtype=dtype_name)
    step_val = (stop - start) / (steps - 1)
    scale = context.graph.splat(context.graph.constant(step_val, dtype=dtype_name), (steps,))
    offset = context.graph.splat(context.graph.constant(start, dtype=dtype_name), (steps,))
    vals = context.graph.add(context.graph.mul(idx, scale), offset)
    last_idx = context.graph.splat(context.graph.constant(steps - 1, dtype='i64'), (steps,))
    i_grid = context.graph.iota((steps,), 0, dtype='i64')
    is_last = context.graph.compare(i_grid, last_idx, 'eq')
    end_splat = context.graph.splat(context.graph.constant(stop, dtype=dtype_name), (steps,))
    return context.wrap(context.graph.select(is_last, end_splat, vals))


@primitive
def _eye_prim(n, m, dtype):
    context = active_trace.get()
    dtype_name = str(dtype)
    rows = context.graph.iota((n, m), 0, dtype='i64')
    cols = context.graph.iota((n, m), 1, dtype='i64')
    diag = context.graph.compare(rows, cols, 'eq')
    return context.wrap(context.graph.cast(diag, dtype=dtype_name))


def amin(x, axis=None, *, keepdims=False):
    return _reduce(x, 'reduce_min', _axis_parameter(axis), keepdims, None)


def prod(x, axis=None, *, keepdims=False, dtype=None):
    return _reduce(x, 'reduce_prod', _axis_parameter(axis), keepdims, dtype)


def any(x, axis=None, *, keepdims=False):
    if not hasattr(x, 'dtype'):
        raise TypeError('any expects a Tensor')
    xb = x if x.dtype is bool_dtype else x.astype(bool_dtype)
    return _reduce(xb, 'reduce_any', _axis_parameter(axis), keepdims, None)


def all(x, axis=None, *, keepdims=False):
    if not hasattr(x, 'dtype'):
        raise TypeError('all expects a Tensor')
    xb = x if x.dtype is bool_dtype else x.astype(bool_dtype)
    return _reduce(xb, 'reduce_all', _axis_parameter(axis), keepdims, None)


@primitive
def _argreduce_op(x, opcode, axis, keepdims):
    from .ir import ScalarType
    if not isinstance(x, TraceTensor):
        raise TypeError(f'{opcode} expects a Tensor')
    rank = len(x.shape)
    if rank == 0:
        raise ValueError(f'{opcode} requires at least 1 dimension')
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError(f'{opcode} axis out of range')
    result = getattr(x.context.graph, opcode)(x.value, ax)
    kept = tuple(1 if i == ax else s for i, s in enumerate(x.shape))
    if isinstance(result.type, ScalarType):
        result = x.context.graph.splat(result, kept if keepdims else ())
    elif keepdims:
        result = x.context.graph.reshape(result, kept)
    return x.context.wrap(result)


def _argreduce(x, opcode, axis, keepdims):
    if not hasattr(x, 'shape'):
        raise TypeError(f'{opcode} expects a Tensor')
    if axis is None:
        if _math.prod(x.shape) == 0:
            raise ValueError(f'{opcode} requires a nonempty tensor')
        flat = x.reshape((-1,))
        res = _argreduce_op(flat, opcode, 0, False)
        if keepdims and x.shape:
            return res.reshape((1,) * len(x.shape))
        return res
    if type(axis) is not int:
        raise TypeError(f'{opcode} axis must be an int or None')
    rank = len(x.shape)
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError(f'{opcode} axis out of range')
    if x.shape[ax] == 0:
        raise ValueError(f'{opcode} requires a nonempty reduction domain')
    return _argreduce_op(x, opcode, ax, keepdims)


def argmax(x, axis=None, *, keepdims=False):
    return _argreduce(x, 'argmax', axis, keepdims)


def argmin(x, axis=None, *, keepdims=False):
    return _argreduce(x, 'argmin', axis, keepdims)


@primitive
def _slice_prim(x, starts, stops, steps):
    if not isinstance(x, TraceTensor):
        raise TypeError('slice expects a Tensor')
    return x.context.wrap(x.context.graph.slice(x.value, starts, stops, steps))


def slice(x, starts, stops, steps=None):
    """Extract a strided slice along every axis of ``x``."""
    if not hasattr(x, 'shape'):
        raise TypeError('slice expects a Tensor')
    rank = len(x.shape)
    starts_t = tuple(starts)
    stops_t = tuple(stops)
    steps_t = (1,) * rank if steps is None else tuple(steps)
    if len(starts_t) != rank or len(stops_t) != rank or len(steps_t) != rank:
        raise ValueError(f'slice expects sequences of length {rank}')
    norm_starts, norm_stops, norm_steps = [], [], []
    for dim, st, sp, step in zip(x.shape, starts_t, stops_t, steps_t):
        if type(step) is not int or step == 0:
            raise ValueError('slice step must be a nonzero integer')
        if st is None and sp is None:
            st, sp, step = _bi.slice(None, None, step).indices(dim)
        elif (step > 0 and (st is None or sp is None or st < 0 or sp < 0)) or (step < 0 and (st is None or sp is None)):
            st, sp, step = _bi.slice(st, sp, step).indices(dim)
        if (step > 0 and st >= sp) or (step < 0 and st <= sp):
            norm_starts.append(0)
            norm_stops.append(0)
            norm_steps.append(1)
        else:
            norm_starts.append(st)
            norm_stops.append(sp)
            norm_steps.append(step)
    return _slice_prim(x, tuple(norm_starts), tuple(norm_stops), tuple(norm_steps))


@primitive
def _pad_prim(x, constant_value, pad_low, pad_high, interior):
    if not isinstance(x, TraceTensor):
        raise TypeError('pad expects a Tensor')
    if isinstance(constant_value, TraceTensor):
        if constant_value.context is not x.context:
            raise ValueError('pad constant_value must belong to the same graph')
        if constant_value.shape != () or constant_value.dtype != x.dtype:
            raise TypeError('pad constant_value must be a 0-D scalar of matching dtype')
        pv = constant_value.value
    else:
        if type(constant_value) not in (bool, int, float):
            raise TypeError('pad constant_value must be a scalar')
        val = _bi.bool(constant_value) if x.dtype == 'bool' else constant_value
        pv = x.context.graph.constant(val, dtype=x.dtype)
    return x.context.wrap(x.context.graph.pad(x.value, pv, pad_low, pad_high, interior))


def pad(x, pad_width, constant_value=0, *, interior=None):
    """Pad ``x`` with ``constant_value`` along edge borders and optional interior gaps."""
    if not hasattr(x, 'shape'):
        raise TypeError('pad expects a Tensor')
    rank = len(x.shape)
    if isinstance(pad_width, int):
        if pad_width < 0:
            raise ValueError('pad_width must be nonnegative')
        low = high = (pad_width,) * rank
    else:
        pairs = tuple(pad_width)
        if len(pairs) != rank:
            raise ValueError(f'pad_width must have length {rank}')
        low, high = [], []
        for pair in pairs:
            if isinstance(pair, int):
                low.append(pair)
                high.append(pair)
            else:
                l, h = pair
                low.append(l)
                high.append(h)
        low, high = tuple(low), tuple(high)
    if _bi.any(v < 0 for v in (*low, *high)):
        raise ValueError('pad_width entries must be nonnegative')
    if interior is None:
        inn = (0,) * rank
    elif isinstance(interior, int):
        inn = (interior,) * rank
    else:
        inn = tuple(interior)
    if len(inn) != rank or _bi.any(v < 0 for v in inn):
        raise ValueError(f'interior must be length {rank} of nonnegative integers')
    return _pad_prim(x, constant_value, low, high, inn)


@variadic_primitive
def _concatenate_prim(axis, *operands):
    if not operands or _bi.any(not isinstance(v, TraceTensor) for v in operands):
        raise TypeError('concatenate requires one or more Tensors')
    ctx = operands[0].context
    if _bi.any(v.context is not ctx for v in operands):
        raise ValueError('concatenate tensors must belong to the same graph')
    return ctx.wrap(ctx.graph.concatenate([v.value for v in operands], axis))


def concatenate(tensors, axis=0):
    """Join a non-empty sequence of tensors along ``axis``."""
    seq = tuple(tensors)
    if not seq:
        raise ValueError('concatenate requires at least one tensor')
    if not hasattr(seq[0], 'shape'):
        raise TypeError('concatenate expects Tensor items')
    rank = len(seq[0].shape)
    if rank == 0:
        raise ValueError('zero-dimensional tensors cannot be concatenated')
    if type(axis) is not int:
        raise TypeError('concatenate axis must be an integer')
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError('concatenate axis out of range')
    if len(seq) == 1:
        return seq[0].reshape(seq[0].shape)
    return _concatenate_prim(ax, *seq)


cat = concatenate


def stack(tensors, axis=0):
    """Join tensors along a new axis."""
    seq = tuple(tensors)
    if not seq:
        raise ValueError('stack requires at least one tensor')
    if not hasattr(seq[0], 'shape'):
        raise TypeError('stack expects Tensor items')
    rank = len(seq[0].shape)
    if type(axis) is not int:
        raise TypeError('stack axis must be an integer')
    ax = axis + rank + 1 if axis < 0 else axis
    if not 0 <= ax <= rank:
        raise ValueError('stack axis out of range')
    expanded = tuple(unsqueeze(t, ax) for t in seq)
    return concatenate(expanded, axis=ax)


def split(tensor, split_size_or_sections, axis=0):
    """Split ``tensor`` along ``axis`` into chunks of given size or explicit section lengths."""
    if not hasattr(tensor, 'shape'):
        raise TypeError('split expects a Tensor')
    rank = len(tensor.shape)
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError('split axis out of range')
    dim = tensor.shape[ax]
    if isinstance(split_size_or_sections, int):
        step = split_size_or_sections
        if step <= 0:
            raise ValueError('split size must be positive')
        sizes = [step] * (dim // step)
        if dim % step:
            sizes.append(dim % step)
        if not sizes and dim == 0:
            sizes = [0]
    else:
        sizes = list(split_size_or_sections)
        if _bi.any(type(s) is not int or s < 0 for s in sizes) or _bi.sum(sizes) != dim:
            raise ValueError(f'split sections must be nonnegative integers summing to {dim}')
    out = []
    offset = 0
    base_starts = [0] * rank
    base_stops = list(tensor.shape)
    steps = (1,) * rank
    for length in sizes:
        starts = list(base_starts)
        stops = list(base_stops)
        starts[ax] = offset
        stops[ax] = offset + length
        out.append(_slice_prim(tensor, tuple(starts), tuple(stops), steps))
        offset += length
    return tuple(out)


def chunk(tensor, chunks, axis=0):
    """Split ``tensor`` into up to ``chunks`` approximately equal pieces along ``axis``."""
    if not hasattr(tensor, 'shape'):
        raise TypeError('chunk expects a Tensor')
    if isinstance(chunks, bool) or type(chunks) is not int or chunks <= 0:
        raise ValueError('chunks must be a positive integer')
    rank = len(tensor.shape)
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError('chunk axis out of range')
    dim = tensor.shape[ax]
    if dim == 0:
        return split(tensor, [0] * chunks, axis=ax)
    step = _math.ceil(dim / chunks)
    return split(tensor, step, axis=ax)


def squeeze(x, axis=None):
    """Remove size-one dimensions from ``x``."""
    if not hasattr(x, 'shape'):
        raise TypeError('squeeze expects a Tensor')
    shape = x.shape
    rank = len(shape)
    if axis is None:
        return x.reshape(tuple(s for s in shape if s != 1))
    axes_raw = (axis,) if isinstance(axis, int) else tuple(axis)
    seen = set()
    for a in axes_raw:
        if type(a) is not int:
            raise TypeError('squeeze axis must be int or tuple of ints')
        norm = a + rank if a < 0 else a
        if not 0 <= norm < rank:
            raise ValueError('squeeze axis out of range')
        if norm in seen:
            raise ValueError('duplicate axis in squeeze')
        if shape[norm] != 1:
            raise ValueError(f'cannot squeeze axis {a} with extent {shape[norm]} != 1')
        seen.add(norm)
    return x.reshape(tuple(s for i, s in enumerate(shape) if i not in seen))


def unsqueeze(x, axis):
    """Insert a size-one dimension at ``axis``."""
    if not hasattr(x, 'shape'):
        raise TypeError('unsqueeze expects a Tensor')
    rank = len(x.shape)
    if type(axis) is not int:
        raise TypeError('unsqueeze axis must be an int')
    ax = axis + rank + 1 if axis < 0 else axis
    if not 0 <= ax <= rank:
        raise ValueError('unsqueeze axis out of range')
    new_shape = list(x.shape)
    new_shape.insert(ax, 1)
    return x.reshape(tuple(new_shape))


def flatten(x, start_dim=0, end_dim=-1):
    """Flatten contiguous dimensions [start_dim, end_dim] into a single dimension."""
    if not hasattr(x, 'shape'):
        raise TypeError('flatten expects a Tensor')
    shape = x.shape
    rank = len(shape)
    if rank == 0:
        if start_dim in (0, -1) and end_dim in (0, -1):
            return x.reshape(1)
        raise IndexError('flatten dimension out of range for scalar')
    sd = start_dim + rank if start_dim < 0 else start_dim
    ed = end_dim + rank if end_dim < 0 else end_dim
    if not (0 <= sd < rank and 0 <= ed < rank and sd <= ed):
        raise ValueError('invalid flatten start_dim/end_dim')
    new_shape = shape[:sd] + (_math.prod(shape[sd:ed + 1]),) + shape[ed + 1:]
    return x.reshape(new_shape)


@primitive
def _gather_prim(params, indices, axis):
    if not isinstance(params, TraceTensor) or not isinstance(indices, TraceTensor):
        raise TypeError('gather requires Tensor inputs')
    if params.context is not indices.context:
        raise ValueError('gather operands must belong to the same graph')
    return params.context.wrap(params.context.graph.gather(params.value, indices.value, axis))


def gather(params, indices, axis=0):
    """Gather elements or slices of ``params`` along ``axis`` using integer ``indices``."""
    if not hasattr(params, 'shape'):
        raise TypeError('gather params must be a Tensor')
    rank = len(params.shape)
    if rank == 0:
        raise ValueError('gather params must have rank at least one')
    if type(axis) is not int:
        raise TypeError('gather axis must be an int')
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError('gather axis out of range')
    if not hasattr(indices, 'dtype') or indices.dtype not in (int32, int64):
        raise TypeError('gather indices must be an int32 or int64 Tensor')
    return _gather_prim(params, indices, ax)


take = gather


@primitive
def _scatter_add_prim(base, indices, updates, axis):
    if not isinstance(base, TraceTensor) or not isinstance(indices, TraceTensor) or not isinstance(updates, TraceTensor):
        raise TypeError('scatter_add requires Tensor inputs')
    if indices.context is not base.context or updates.context is not base.context:
        raise ValueError('scatter_add operands must belong to the same graph')
    return base.context.wrap(base.context.graph.scatter_add(base.value, indices.value, updates.value, axis))


def scatter_add(base, indices, updates, axis=0):
    """Return a new tensor adding ``updates`` into ``base`` at ``indices`` along ``axis``."""
    if not hasattr(base, 'shape'):
        raise TypeError('scatter_add base must be a Tensor')
    rank = len(base.shape)
    if rank == 0:
        raise ValueError('scatter_add base must have rank at least one')
    if type(axis) is not int:
        raise TypeError('scatter_add axis must be an int')
    ax = axis + rank if axis < 0 else axis
    if not 0 <= ax < rank:
        raise ValueError('scatter_add axis out of range')
    return _scatter_add_prim(base, indices, updates, ax)


@primitive
def _tri_prim(x, diagonal, upper):
    if not isinstance(x, TraceTensor):
        raise TypeError('tril/triu expects a Tensor')
    rank = len(x.shape)
    if rank < 2:
        raise ValueError('tril/triu requires rank at least two')
    ctx = x.context
    row = ctx.graph.iota(x.shape, rank - 2, dtype='i64')
    col = ctx.graph.iota(x.shape, rank - 1, dtype='i64')
    rhs = col if diagonal == 0 else ctx.graph.add(
        col, ctx.graph.splat(ctx.graph.constant(-diagonal, dtype='i64'), x.shape))
    mask = ctx.graph.compare(rhs, row, 'ge' if upper else 'le')
    zero = ctx.graph.splat(ctx.graph.constant(False if x.dtype == 'bool' else 0, dtype=x.dtype), x.shape)
    return ctx.wrap(ctx.graph.select(mask, x.value, zero))


def tril(x, diagonal=0):
    """Zero elements above the ``k``-th diagonal."""
    if type(diagonal) is not int:
        raise TypeError('diagonal must be an int')
    return _tri_prim(x, diagonal, False)


def triu(x, diagonal=0):
    """Zero elements below the ``k``-th diagonal."""
    if type(diagonal) is not int:
        raise TypeError('diagonal must be an int')
    return _tri_prim(x, diagonal, True)


@primitive
def _one_hot_prim(indices, num_classes, dtype):
    if not isinstance(indices, TraceTensor):
        raise TypeError('one_hot indices must be a Tensor')
    if indices.dtype not in ('i32', 'i64'):
        raise TypeError('one_hot indices must have int32 or int64 dtype')
    ctx = indices.context
    out_shape = (*indices.shape, num_classes)
    dims = tuple(range(len(indices.shape)))
    expanded = ctx.graph.broadcast_in_dim(indices.value, out_shape, dims)
    cls_grid = ctx.graph.iota(out_shape, len(indices.shape), dtype=indices.dtype)
    mask = ctx.graph.compare(expanded, cls_grid, 'eq')
    target_dtype = str(dtype)
    return ctx.wrap(mask if target_dtype == 'bool' else ctx.graph.cast(mask, dtype=target_dtype))


def one_hot(indices, num_classes, *, dtype=fp32):
    """Encode integer indices as trailing one-hot vectors of width ``num_classes``."""
    from .creation import _coerce_dtype
    if isinstance(num_classes, bool) or type(num_classes) is not int or num_classes <= 0:
        raise ValueError('one_hot num_classes must be a positive integer')
    return _one_hot_prim(indices, num_classes, _coerce_dtype(dtype))


def getitem(x, key):
    """Index or slice a Tensor or TraceTensor using static slices, ints, None, Ellipsis, and at most one integer Tensor."""
    raw = key if isinstance(key, tuple) else (key,)
    n_ellipsis = _bi.sum(1 for item in raw if item is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    shape = x.shape
    rank = len(shape)
    non_none = _bi.sum(1 for item in raw if item is not None and item is not Ellipsis)
    if non_none > rank:
        raise IndexError(f'too many indices for tensor of dimension {rank}')
    expanded = []
    for item in raw:
        if item is Ellipsis:
            expanded.extend([_bi.slice(None)] * (rank - non_none))
        else:
            expanded.append(item)
    if non_none < rank and n_ellipsis == 0:
        expanded.extend([_bi.slice(None)] * (rank - non_none))

    starts, stops, steps = [], [], []
    squeeze_axes = []
    final_shape = []
    tensor_axis = None
    tensor_idx = None
    in_axis = 0
    post_slice_axis = 0

    for item in expanded:
        if item is None:
            final_shape.append(1)
            continue
        dim = shape[in_axis]
        if isinstance(item, int) and not isinstance(item, bool):
            idx = item + dim if item < 0 else item
            if not 0 <= idx < dim:
                raise IndexError(f'index {item} is out of bounds for axis {in_axis} with size {dim}')
            starts.append(idx)
            stops.append(idx + 1)
            steps.append(1)
            squeeze_axes.append(post_slice_axis)
            in_axis += 1
            post_slice_axis += 1
        elif isinstance(item, _bi.slice):
            if item.step == 0:
                raise ValueError('slice step cannot be zero')
            st, sp, step = item.indices(dim)
            ext = len(range(st, sp, step))
            if ext == 0:
                starts.append(0)
                stops.append(0)
                steps.append(1)
            else:
                starts.append(st)
                stops.append(sp)
                steps.append(step)
            final_shape.append(ext)
            in_axis += 1
            post_slice_axis += 1
        elif hasattr(item, 'dtype') and hasattr(item, 'shape'):
            if tensor_idx is not None:
                raise TypeError('indexing with multiple Tensor indices is not supported')
            if item.dtype not in (int32, int64):
                raise TypeError('Tensor indices must have int32 or int64 dtype')
            starts.append(0)
            stops.append(dim)
            steps.append(1)
            tensor_axis = post_slice_axis
            tensor_idx = item
            final_shape.extend(item.shape)
            in_axis += 1
            post_slice_axis += 1
        else:
            raise TypeError(f'unsupported index type {type(item).__name__}')

    need_slice = _bi.any(st != 0 or sp != dim or step != 1
                         for st, sp, step, dim in zip(starts, stops, steps, shape))
    result = _slice_prim(x, tuple(starts), tuple(stops), tuple(steps)) if (need_slice and rank > 0) else x
    if squeeze_axes:
        if tensor_axis is not None:
            tensor_axis -= _bi.sum(1 for a in squeeze_axes if a < tensor_axis)
        kept_shape = tuple(s for i, s in enumerate(result.shape) if i not in squeeze_axes)
        result = result.reshape(kept_shape)
    if tensor_idx is not None:
        result = gather(result, tensor_idx, axis=tensor_axis)
    if tuple(final_shape) != result.shape:
        result = result.reshape(tuple(final_shape))
    return result
