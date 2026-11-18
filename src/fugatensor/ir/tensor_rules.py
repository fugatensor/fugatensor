"""Typing and composable derivatives for tensor maps, reductions and contractions."""
import math

from .types import DTYPES, FLOAT_DTYPES, INT_DTYPES, ScalarType, TensorType, cast_type

_NUMERIC_DTYPES = FLOAT_DTYPES | INT_DTYPES
_BITWISE_DTYPES = INT_DTYPES | frozenset({'bool'})


def _shape(type_):
    return type_.shape if isinstance(type_, TensorType) else ()


def _same_type(inputs, attrs, count, allowed_dtypes):
    if len(inputs) != count or attrs or any(value != inputs[0] for value in inputs):
        raise TypeError('operation requires matching operands and no attributes')
    if inputs[0].dtype not in allowed_dtypes:
        raise TypeError('unsupported dtype for operation')
    return (inputs[0],)


def _float(inputs, attrs, count=1):
    return _same_type(inputs, attrs, count, FLOAT_DTYPES)


def _compare(inputs, attrs):
    if len(inputs) != 2 or inputs[0] != inputs[1]:
        raise TypeError('compare requires matching operand types')
    if set(attrs) != {'predicate'} or attrs['predicate'] not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
        raise ValueError('compare predicate must be eq/ne/lt/le/gt/ge')
    return (cast_type(inputs[0], 'bool'),)


def _select(inputs, attrs):
    if len(inputs) != 3 or attrs or inputs[1] != inputs[2] or inputs[0] != cast_type(inputs[1], 'bool'):
        raise TypeError('select requires a bool condition and matching value shapes and dtypes')
    return (inputs[1],)


def _broadcast(inputs, attrs):
    if len(inputs) != 1 or set(attrs) != {'shape', 'dimensions'}:
        raise TypeError('broadcast_in_dim requires shape and dimensions')
    output = TensorType(attrs['shape'], inputs[0].dtype)
    source, dims = _shape(inputs[0]), tuple(attrs['dimensions'])
    if None in source or None in output.shape:
        raise ValueError('broadcast requires concrete shapes')
    if (len(dims) != len(source) or any(type(axis) is not int or not 0 <= axis < len(output.shape) for axis in dims)
            or tuple(sorted(set(dims))) != dims):
        raise ValueError('broadcast dimensions must be an increasing input-to-output axis mapping')
    if any(size != 1 and size != output.shape[axis] for size, axis in zip(source, dims)):
        raise ValueError('broadcast mapped dimensions must match or have size one')
    return (output,)


def _transpose(inputs, attrs):
    if len(inputs) != 1 or set(attrs) != {'permutation'}:
        raise TypeError('transpose requires one operand and a permutation')
    permutation, shape = tuple(attrs['permutation']), _shape(inputs[0])
    if any(type(axis) is not int for axis in permutation) or sorted(permutation) != list(range(len(shape))):
        raise ValueError('transpose permutation must contain each axis exactly once')
    return (TensorType(tuple(shape[axis] for axis in permutation), inputs[0].dtype)
            if isinstance(inputs[0], TensorType) else inputs[0],)


def _reduce(inputs, attrs, maximum=False, *, nonempty=None):
    if nonempty is None:
        nonempty = maximum
    if len(inputs) != 1 or set(attrs) != {'axes'} or inputs[0].dtype == 'bool':
        raise TypeError('reduction requires one numeric operand and axes')
    shape, axes = _shape(inputs[0]), tuple(attrs['axes'])
    if None in shape:
        raise ValueError('reduction requires concrete shapes')
    if (any(type(axis) is not int or not 0 <= axis < len(shape) for axis in axes)
            or tuple(sorted(set(axes))) != axes):
        raise ValueError('reduction axes must be sorted, unique and in range')
    if nonempty and any(shape[axis] == 0 for axis in axes):
        raise ValueError('max/min reduction requires a nonempty reduction domain')
    remaining = tuple(size for axis, size in enumerate(shape) if axis not in axes)
    return (TensorType(remaining, inputs[0].dtype) if remaining else ScalarType(inputs[0].dtype),)


def _reduce_bool(inputs, attrs):
    if len(inputs) != 1 or set(attrs) != {'axes'} or inputs[0].dtype != 'bool':
        raise TypeError('boolean reduction requires one bool operand and axes')
    shape, axes = _shape(inputs[0]), tuple(attrs['axes'])
    if None in shape:
        raise ValueError('reduction requires concrete shapes')
    if (any(type(axis) is not int or not 0 <= axis < len(shape) for axis in axes)
            or tuple(sorted(set(axes))) != axes):
        raise ValueError('reduction axes must be sorted, unique and in range')
    remaining = tuple(size for axis, size in enumerate(shape) if axis not in axes)
    return (TensorType(remaining, 'bool') if remaining else ScalarType('bool'),)


def _argreduce(inputs, attrs):
    if len(inputs) != 1 or set(attrs) != {'axis'} or type(attrs['axis']) is not int:
        raise TypeError('argmax/argmin requires one operand and integer axis')
    shape = _shape(inputs[0])
    if not shape or None in shape:
        raise ValueError('argmax/argmin requires concrete tensor with rank at least one')
    axis = attrs['axis']
    if not 0 <= axis < len(shape):
        raise ValueError('argmax/argmin axis out of range')
    if shape[axis] == 0:
        raise ValueError('argmax/argmin requires a nonempty reduction domain')
    remaining = tuple(size for idx, size in enumerate(shape) if idx != axis)
    return (TensorType(remaining, 'i64') if remaining else ScalarType('i64'),)


def _iota(inputs, attrs):
    if inputs or set(attrs) != {'shape', 'dimension', 'dtype'}:
        raise TypeError('iota requires no operands and shape/dimension/dtype attributes')
    shape, dim, dtype = attrs['shape'], attrs['dimension'], attrs['dtype']
    if (not isinstance(shape, (tuple, list)) or not shape
            or any(type(d) is not int or d < 0 for d in shape)):
        raise ValueError('iota requires a nonempty tuple of nonnegative concrete extents')
    if type(dim) is not int or not 0 <= dim < len(shape):
        raise ValueError('iota dimension out of range')
    if dtype not in _NUMERIC_DTYPES:
        raise TypeError('iota requires a numeric dtype')
    return (TensorType(tuple(shape), dtype),)


def _slice(inputs, attrs):
    if len(inputs) != 1 or set(attrs) != {'starts', 'stops', 'steps'}:
        raise TypeError('slice requires one operand and starts/stops/steps attributes')
    shape = _shape(inputs[0])
    if None in shape:
        raise ValueError('slice requires concrete shapes')
    starts, stops, steps = attrs['starts'], attrs['stops'], attrs['steps']
    if (not isinstance(starts, (tuple, list)) or not isinstance(stops, (tuple, list))
            or not isinstance(steps, (tuple, list))
            or len(starts) != len(shape) or len(stops) != len(shape) or len(steps) != len(shape)):
        raise ValueError('slice starts, stops and steps must match operand rank')
    out_shape = []
    for dim, start, stop, step in zip(shape, starts, stops, steps):
        if type(start) is not int or type(stop) is not int or type(step) is not int or step == 0:
            raise ValueError('slice bounds must be integers and step nonzero')
        if step > 0:
            if not 0 <= start <= stop <= dim:
                raise ValueError('slice forward bounds out of range')
            out_shape.append(0 if stop == start else (stop - start + step - 1) // step)
        else:
            if start == stop:
                if not 0 <= start <= dim:
                    raise ValueError('slice empty bounds out of range')
                out_shape.append(0)
            else:
                if not (0 <= start < dim and -1 <= stop < start):
                    raise ValueError('slice reverse bounds out of range')
                out_shape.append((start - stop + (-step) - 1) // (-step))
    return (TensorType(tuple(out_shape), inputs[0].dtype)
            if isinstance(inputs[0], TensorType) else inputs[0],)


def _pad(inputs, attrs):
    if len(inputs) != 2 or set(attrs) != {'pad_low', 'pad_high', 'interior'}:
        raise TypeError('pad requires value, pad_value, and pad_low/pad_high/interior attributes')
    x, pv = inputs
    if isinstance(pv, TensorType):
        if pv.shape != () or pv.dtype != x.dtype:
            raise TypeError('pad requires a 0-D scalar pad_value with the same dtype as input')
    elif not isinstance(pv, ScalarType) or pv.dtype != x.dtype:
        raise TypeError('pad requires a 0-D scalar pad_value with the same dtype as input')
    shape = _shape(x)
    if None in shape:
        raise ValueError('pad requires concrete shapes')
    low, high, interior = attrs['pad_low'], attrs['pad_high'], attrs['interior']
    if (not isinstance(low, (tuple, list)) or not isinstance(high, (tuple, list))
            or not isinstance(interior, (tuple, list))
            or len(low) != len(shape) or len(high) != len(shape) or len(interior) != len(shape)):
        raise ValueError('pad attributes must match operand rank')
    out_shape = []
    for dim, l, h, inn in zip(shape, low, high, interior):
        if type(l) is not int or type(h) is not int or type(inn) is not int or l < 0 or h < 0 or inn < 0:
            raise ValueError('pad amounts and interior dilation must be nonnegative integers')
        inner = dim + (dim - 1) * inn if dim > 0 else 0
        out_shape.append(l + inner + h)
    return (TensorType(tuple(out_shape), x.dtype) if isinstance(x, TensorType) else x,)


def _concatenate(inputs, attrs):
    if not inputs or set(attrs) != {'axis'} or type(attrs['axis']) is not int:
        raise TypeError('concatenate requires at least one operand and integer axis')
    if any(not isinstance(v, TensorType) or None in v.shape for v in inputs):
        raise ValueError('concatenate requires concrete tensors with rank at least one')
    first = inputs[0]
    rank, dtype, axis = len(first.shape), first.dtype, attrs['axis']
    if rank < 1 or not 0 <= axis < rank:
        raise ValueError('concatenate axis out of range')
    total = 0
    for v in inputs:
        if v.dtype != dtype or len(v.shape) != rank:
            raise TypeError('concatenate operands must match rank and dtype')
        if any(idx != axis and s != first.shape[idx] for idx, s in enumerate(v.shape)):
            raise ValueError('concatenate non-axis dimensions must match')
        total += v.shape[axis]
    out_shape = (*first.shape[:axis], total, *first.shape[axis + 1:])
    return (TensorType(out_shape, dtype),)


def _gather(inputs, attrs):
    if len(inputs) != 2 or set(attrs) != {'axis'} or type(attrs['axis']) is not int:
        raise TypeError('gather requires (params, indices) and integer axis')
    params, indices = inputs
    if not isinstance(params, TensorType) or None in params.shape or len(params.shape) < 1:
        raise ValueError('gather params must be a concrete tensor of rank at least one')
    if indices.dtype not in INT_DTYPES:
        raise TypeError('gather indices must have i32 or i64 dtype')
    idx_shape = _shape(indices)
    if None in idx_shape:
        raise ValueError('gather indices must have concrete shape')
    axis = attrs['axis']
    if not 0 <= axis < len(params.shape):
        raise ValueError('gather axis out of range')
    out_shape = (*params.shape[:axis], *idx_shape, *params.shape[axis + 1:])
    return (TensorType(out_shape, params.dtype) if out_shape else ScalarType(params.dtype),)


def _scatter_add(inputs, attrs):
    if len(inputs) != 3 or set(attrs) != {'axis'} or type(attrs['axis']) is not int:
        raise TypeError('scatter_add requires (base, indices, updates) and integer axis')
    base, indices, updates = inputs
    if not isinstance(base, TensorType) or None in base.shape or len(base.shape) < 1:
        raise ValueError('scatter_add base must be a concrete tensor of rank at least one')
    if base.dtype not in _NUMERIC_DTYPES:
        raise TypeError('scatter_add base must have a numeric dtype')
    if indices.dtype not in INT_DTYPES or None in _shape(indices):
        raise TypeError('scatter_add indices must have concrete i32 or i64 type')
    axis = attrs['axis']
    if not 0 <= axis < len(base.shape):
        raise ValueError('scatter_add axis out of range')
    exp_shape = (*base.shape[:axis], *_shape(indices), *base.shape[axis + 1:])
    expected = TensorType(exp_shape, base.dtype) if exp_shape else ScalarType(base.dtype)
    if updates != expected:
        raise TypeError('scatter_add updates shape/dtype does not match base, indices and axis')
    return (base,)


def _matmul(inputs, attrs):
    if len(inputs) != 2 or any(not isinstance(value, TensorType) for value in inputs):
        raise TypeError('matmul requires two tensors')
    left, right = inputs
    if left.dtype != right.dtype or left.dtype not in FLOAT_DTYPES:
        raise TypeError('matmul requires matching f32/f16/bf16 dtypes')
    if set(attrs) != {'transpose_a', 'transpose_b'} or any(type(value) is not bool for value in attrs.values()):
        raise TypeError('matmul transpose flags must be bool')
    if len(left.shape) < 2 or len(right.shape) < 2 or None in (*left.shape, *right.shape):
        raise ValueError('matmul requires concrete tensors of rank at least two')
    if left.shape[:-2] != right.shape[:-2]:
        raise ValueError('matmul batch dimensions must match')
    m, k = left.shape[-2:][::-1] if attrs['transpose_a'] else left.shape[-2:]
    kb, n = right.shape[-2:][::-1] if attrs['transpose_b'] else right.shape[-2:]
    if k != kb:
        raise ValueError('matmul contraction dimensions must match')
    return (TensorType((*left.shape[:-2], m, n), left.dtype),)


def _filled(builder, type_, value):
    scalar = builder.constant(value, dtype=type_.dtype)
    return builder.splat(scalar, type_.shape) if isinstance(type_, TensorType) else scalar


def _restore(builder, value, type_):
    if value.type == type_:
        return value
    if isinstance(value.type, ScalarType):
        return builder.splat(value, type_.shape)
    return builder.reshape(value, type_.shape)


def _expand_reduction(builder, value, input_type, axes):
    if isinstance(input_type, ScalarType):
        return value
    dimensions = tuple(axis for axis in range(len(input_type.shape)) if axis not in axes)
    return builder.broadcast_in_dim(value, input_type.shape, dimensions)


def _neg(builder, value):
    return builder.mul(value, _filled(builder, value.type, -1))


def _sum_terms(builder, left, right):
    return right if left is None else left if right is None else builder.add(left, right)


def _div_vjp(b, op, p, g, active):
    quotient = b.div(p[0], p[1]) if active[1] else None
    return (b.div(g, p[1]) if active[0] else None,
            _neg(b, b.div(b.mul(g, quotient), p[1])) if active[1] else None)


def _div_jvp(b, op, p, t):
    right = None if t[1] is None else _neg(b, b.mul(b.div(p[0], p[1]), t[1]))
    total = _sum_terms(b, t[0], right)
    return (None if total is None else b.div(total, p[1]),)


def _unary_factor(b, opcode, x):
    if opcode == 'exp':
        return b.exp(x)
    if opcode == 'log':
        return b.div(_filled(b, x.type, 1), x)
    if opcode == 'sin':
        return b.cos(x)
    if opcode == 'cos':
        return _neg(b, b.sin(x))
    if opcode == 'tanh':
        t = b.tanh(x)
        return b.add(_filled(b, x.type, 1), _neg(b, b.mul(t, t)))
    if opcode == 'erf':
        scale = _filled(b, x.type, 2.0 / math.sqrt(math.pi))
        return b.mul(scale, b.exp(_neg(b, b.mul(x, x))))
    return b.div(_filled(b, x.type, 0.5), b.sqrt(x))


def _unary_vjp(b, op, p, g, active):
    return (b.mul(g, _unary_factor(b, op.opcode, p[0])) if active[0] else None,)


def _unary_jvp(b, op, p, t):
    return (None if t[0] is None else b.mul(t[0], _unary_factor(b, op.opcode, p[0])),)


def _abs_sign(b, x):
    zero = _filled(b, x.type, 0)
    one = _filled(b, x.type, 1)
    neg_one = _filled(b, x.type, -1)
    sign = b.select(b.compare(x, zero, 'gt'), one, b.select(b.compare(x, zero, 'lt'), neg_one, zero))
    return b.select(b.compare(x, x, 'ne'), x, sign)


def _abs_vjp(b, op, p, g, active):
    if not active[0] or p[0].type.dtype not in FLOAT_DTYPES:
        return (None,)
    return (b.mul(g, _abs_sign(b, p[0])),)


def _abs_jvp(b, op, p, t):
    if t[0] is None or p[0].type.dtype not in FLOAT_DTYPES:
        return (None,)
    return (b.mul(t[0], _abs_sign(b, p[0])),)


def _pow_vjp(b, op, p, g, active):
    if p[0].type.dtype not in FLOAT_DTYPES:
        return (None, None)
    x, y = p
    zero = _filled(b, x.type, 0)
    one = _filled(b, x.type, 1)
    dx = None
    if active[0]:
        y_is_zero = b.compare(y, zero, 'eq')
        safe_y_m1 = b.add(b.select(y_is_zero, one, y), _filled(b, y.type, -1))
        factor_x = b.select(y_is_zero, zero, b.mul(y, b.pow(x, safe_y_m1)))
        dx = b.mul(g, factor_x)
    dy = None
    if active[1]:
        x_pos = b.compare(x, zero, 'gt')
        safe_x = b.select(x_pos, x, one)
        factor_y = b.select(x_pos, b.mul(b.pow(x, y), b.log(safe_x)), zero)
        dy = b.mul(g, factor_y)
    return (dx, dy)


def _pow_jvp(b, op, p, t):
    if p[0].type.dtype not in FLOAT_DTYPES or (t[0] is None and t[1] is None):
        return (None,)
    x, y = p
    zero = _filled(b, x.type, 0)
    one = _filled(b, x.type, 1)
    tx_term = None
    if t[0] is not None:
        y_is_zero = b.compare(y, zero, 'eq')
        safe_y_m1 = b.add(b.select(y_is_zero, one, y), _filled(b, y.type, -1))
        tx_term = b.mul(t[0], b.select(y_is_zero, zero, b.mul(y, b.pow(x, safe_y_m1))))
    ty_term = None
    if t[1] is not None:
        x_pos = b.compare(x, zero, 'gt')
        safe_x = b.select(x_pos, x, one)
        ty_term = b.mul(t[1], b.select(x_pos, b.mul(b.pow(x, y), b.log(safe_x)), zero))
    return (_sum_terms(b, tx_term, ty_term),)


def _minmax_vjp(b, op, p, g, active, is_max):
    if p[0].type.dtype not in FLOAT_DTYPES:
        return (None, None)
    x, y = p
    zero = _filled(b, g.type, 0)
    half_g = b.mul(g, _filled(b, g.type, 0.5))
    either_nan = b.bitwise_or(b.compare(x, x, 'ne'), b.compare(y, y, 'ne'))
    nan_val = b.select(b.compare(x, x, 'ne'), x, y)
    eq = b.compare(x, y, 'eq')
    win_x = b.compare(x, y, 'gt' if is_max else 'lt')
    win_y = b.compare(y, x, 'gt' if is_max else 'lt')
    dx = b.select(either_nan, nan_val, b.select(win_x, g, b.select(eq, half_g, zero))) if active[0] else None
    dy = b.select(either_nan, nan_val, b.select(win_y, g, b.select(eq, half_g, zero))) if active[1] else None
    return (dx, dy)


def _minmax_jvp(b, op, p, t, is_max):
    if p[0].type.dtype not in FLOAT_DTYPES or (t[0] is None and t[1] is None):
        return (None,)
    x, y = p
    zero = _filled(b, x.type, 0)
    tx = zero if t[0] is None else t[0]
    ty = zero if t[1] is None else t[1]
    either_nan = b.bitwise_or(b.compare(x, x, 'ne'), b.compare(y, y, 'ne'))
    nan_val = b.select(b.compare(x, x, 'ne'), x, y)
    win_x = b.compare(x, y, 'gt' if is_max else 'lt')
    win_y = b.compare(y, x, 'gt' if is_max else 'lt')
    half_sum = b.mul(b.add(tx, ty), _filled(b, x.type, 0.5))
    return (b.select(either_nan, nan_val, b.select(win_x, tx, b.select(win_y, ty, half_sum))),)


def _mod_vjp(b, op, p, g, active):
    if p[0].type.dtype not in FLOAT_DTYPES:
        return (None, None)
    dx = g if active[0] else None
    dy = _neg(b, b.mul(g, b.floordiv(p[0], p[1]))) if active[1] else None
    return (dx, dy)


def _mod_jvp(b, op, p, t):
    if p[0].type.dtype not in FLOAT_DTYPES or (t[0] is None and t[1] is None):
        return (None,)
    right = None if t[1] is None else _neg(b, b.mul(b.floordiv(p[0], p[1]), t[1]))
    return (_sum_terms(b, t[0], right),)


def _select_vjp(b, op, p, g, active):
    zero = _filled(b, g.type, 0)
    return (None, b.select(p[0], g, zero) if active[1] else None,
            b.select(p[0], zero, g) if active[2] else None)


def _select_jvp(b, op, p, t):
    zero = _filled(b, p[1].type, 0)
    return (b.select(p[0], t[1] if t[1] is not None else zero,
                     t[2] if t[2] is not None else zero),)


def _broadcast_vjp(b, op, p, g, active):
    if not active[0]:
        return (None,)
    source, target = _shape(p[0].type), g.type.shape
    dims = tuple(op.attrs['dimensions'])
    expanded = {dim for size, dim in zip(source, dims) if size == 1 and target[dim] != 1}
    axes = tuple(axis for axis in range(len(target)) if axis not in dims or axis in expanded)
    reduced = b.reduce_sum(g, axes)
    return (_restore(b, reduced, p[0].type),)


def _map_jvp(b, op, p, t):
    return (None if t[0] is None else b.primitive(op.opcode, (t[0],), dict(op.attrs)),)


def _transpose_vjp(b, op, p, g, active):
    inverse = tuple(op.attrs['permutation'].index(axis) for axis in range(len(op.attrs['permutation'])))
    return (b.transpose(g, inverse) if active[0] else None,)


def _sum_vjp(b, op, p, g, active):
    return (_expand_reduction(b, g, p[0].type, op.attrs['axes']) if active[0] else None,)


def _extremum_weights(b, op, x, *, is_max):
    axes = op.attrs['axes']
    extremum = b.reduce_max(x, axes) if is_max else b.reduce_min(x, axes)
    extremum = _expand_reduction(b, extremum, x.type, axes)
    mask = b.compare(x, extremum, 'eq')
    counts = b.reduce_sum(b.cast(mask, dtype='f32'), axes)
    counts = _expand_reduction(b, counts, cast_type(x.type, 'f32'), axes)
    weight = b.cast(b.div(b.cast(mask, dtype='f32'), counts), dtype=x.type.dtype)
    # NaN extrema propagate through the value and its chosen derivative rule.
    return b.select(b.compare(extremum, extremum, 'ne'), _filled(b, x.type, float('nan')), weight)


def _max_vjp(b, op, p, g, active, *, is_max=True):
    return (b.mul(_extremum_weights(b, op, p[0], is_max=is_max),
                  _expand_reduction(b, g, p[0].type, op.attrs['axes']))
            if active[0] else None,)


def _max_jvp(b, op, p, t, *, is_max=True):
    return (None if t[0] is None else b.reduce_sum(
        b.mul(_extremum_weights(b, op, p[0], is_max=is_max), t[0]), op.attrs['axes']),)


def _prod_excl(b, x, axes):
    is_zero = b.compare(x, _filled(b, x.type, 0), 'eq')
    i32_full = TensorType(x.type.shape, 'i32')
    zero_count = _expand_reduction(b, b.reduce_sum(b.cast(is_zero, dtype='i32'), axes), i32_full, axes)
    one = _filled(b, x.type, 1)
    zero_f = _filled(b, x.type, 0)
    safe_x = b.select(is_zero, one, x)
    safe_prod = _expand_reduction(b, b.reduce_prod(safe_x, axes), x.type, axes)
    one_zero = b.select(b.compare(zero_count, _filled(b, i32_full, 1), 'eq'), safe_prod, zero_f)
    no_zero = b.select(b.compare(zero_count, _filled(b, i32_full, 0), 'eq'), b.div(safe_prod, safe_x), zero_f)
    return b.select(is_zero, one_zero, no_zero)


def _prod_vjp(b, op, p, g, active):
    if not active[0]:
        return (None,)
    axes = op.attrs['axes']
    if not axes:
        return (g,)
    return (b.mul(_prod_excl(b, p[0], axes), _expand_reduction(b, g, p[0].type, axes)),)


def _prod_jvp(b, op, p, t):
    if t[0] is None:
        return (None,)
    axes = op.attrs['axes']
    if not axes:
        return (t[0],)
    return (b.reduce_sum(b.mul(_prod_excl(b, p[0], axes), t[0]), axes),)


def _slice_vjp(b, op, p, g, active):
    if not active[0] or p[0].type.dtype not in FLOAT_DTYPES:
        return (None,)
    in_shape = _shape(p[0].type)
    if not in_shape:
        return (g,)
    out_shape = op.results[0].type.shape
    if 0 in out_shape:
        return (_filled(b, p[0].type, 0),)
    starts, steps = op.attrs['starts'], op.attrs['steps']
    if any(step < 0 for step in steps):
        flip_starts = tuple(ext - 1 if step < 0 else 0 for ext, step in zip(out_shape, steps))
        flip_stops = tuple(-1 if step < 0 else ext for ext, step in zip(out_shape, steps))
        flip_steps = tuple(-1 if step < 0 else 1 for step in steps)
        g = b.slice(g, flip_starts, flip_stops, flip_steps)
    pad_low, pad_high, interior = [], [], []
    for dim, ext, start, step in zip(in_shape, out_shape, starts, steps):
        abs_step = abs(step)
        first = start if step > 0 else start + (ext - 1) * step
        last = first + (ext - 1) * abs_step
        pad_low.append(first)
        interior.append(abs_step - 1)
        pad_high.append(dim - 1 - last)
    return (b.pad(g, b.constant(0, dtype=p[0].type.dtype), pad_low, pad_high, interior),)


def _slice_jvp(b, op, p, t):
    if t[0] is None or p[0].type.dtype not in FLOAT_DTYPES:
        return (None,)
    return (b.slice(t[0], op.attrs['starts'], op.attrs['stops'], op.attrs['steps']),)


def _pad_vjp(b, op, p, g, active):
    if p[0].type.dtype not in FLOAT_DTYPES:
        return (None, None)
    in_shape = _shape(p[0].type)
    if not in_shape:
        return (g if active[0] else None, _filled(b, p[1].type, 0) if active[1] else None)
    low, high, interior = op.attrs['pad_low'], op.attrs['pad_high'], op.attrs['interior']
    dx = None
    if active[0]:
        starts = tuple(l if dim > 0 else 0 for dim, l in zip(in_shape, low))
        stops = tuple(l + (dim - 1) * (inn + 1) + 1 if dim > 0 else 0
                      for dim, l, inn in zip(in_shape, low, interior))
        steps = tuple(inn + 1 for inn in interior)
        dx = b.slice(g, starts, stops, steps)
    dpad = None
    if active[1]:
        mask = b.pad(_filled(b, p[0].type, 0), _filled(b, p[1].type, 1), low, high, interior)
        axes = tuple(range(len(op.results[0].type.shape)))
        dpad = _restore(b, b.reduce_sum(b.mul(g, mask), axes), p[1].type)
    return (dx, dpad)


def _pad_jvp(b, op, p, t):
    if p[0].type.dtype not in FLOAT_DTYPES or (t[0] is None and t[1] is None):
        return (None,)
    dx = _filled(b, p[0].type, 0) if t[0] is None else t[0]
    dpad = _filled(b, p[1].type, 0) if t[1] is None else t[1]
    return (b.pad(dx, dpad, op.attrs['pad_low'], op.attrs['pad_high'], op.attrs['interior']),)


def _concat_vjp(b, op, p, g, active):
    if p[0].type.dtype not in FLOAT_DTYPES:
        return (None,) * len(p)
    axis = op.attrs['axis']
    out_shape = op.results[0].type.shape
    base_starts = [0] * len(out_shape)
    base_stops = list(out_shape)
    steps = (1,) * len(out_shape)
    grads = []
    offset = 0
    for operand, is_active in zip(p, active):
        extent = operand.type.shape[axis]
        if is_active:
            starts = list(base_starts)
            stops = list(base_stops)
            starts[axis] = offset
            stops[axis] = offset + extent
            grads.append(b.slice(g, tuple(starts), tuple(stops), steps))
        else:
            grads.append(None)
        offset += extent
    return tuple(grads)


def _concat_jvp(b, op, p, t):
    if p[0].type.dtype not in FLOAT_DTYPES or all(item is None for item in t):
        return (None,)
    parts = tuple(_filled(b, operand.type, 0) if tangent is None else tangent for operand, tangent in zip(p, t))
    return (b.concatenate(parts, op.attrs['axis']),)


def _gather_vjp(b, op, p, g, active):
    if not active[0] or p[0].type.dtype not in FLOAT_DTYPES:
        return (None, None)
    zeros = _filled(b, p[0].type, 0)
    return (b.scatter_add(zeros, p[1], g, op.attrs['axis']), None)


def _gather_jvp(b, op, p, t):
    if t[0] is None or p[0].type.dtype not in FLOAT_DTYPES:
        return (None,)
    return (b.gather(t[0], p[1], op.attrs['axis']),)


def _scatter_add_vjp(b, op, p, g, active):
    if p[0].type.dtype not in FLOAT_DTYPES:
        return (None, None, None)
    db = g if active[0] else None
    du = b.gather(g, p[1], op.attrs['axis']) if active[2] else None
    return (db, None, du)


def _scatter_add_jvp(b, op, p, t):
    if p[0].type.dtype not in FLOAT_DTYPES or (t[0] is None and t[2] is None):
        return (None,)
    if t[2] is None:
        return (t[0],)
    tb = _filled(b, p[0].type, 0) if t[0] is None else t[0]
    return (b.scatter_add(tb, p[1], t[2], op.attrs['axis']),)


def _matmul_vjp(b, op, p, g, active):
    ta, tb = op.attrs['transpose_a'], op.attrs['transpose_b']
    left = (b.matmul(p[1], g, transpose_a=tb, transpose_b=True) if ta
            else b.matmul(g, p[1], transpose_b=not tb)) if active[0] else None
    right = (b.matmul(g, p[0], transpose_a=True, transpose_b=ta) if tb
             else b.matmul(p[0], g, transpose_a=not ta)) if active[1] else None
    return left, right


def _matmul_jvp(b, op, p, t):
    left = None if t[0] is None else b.matmul(t[0], p[1], **op.attrs)
    right = None if t[1] is None else b.matmul(p[0], t[1], **op.attrs)
    return (_sum_terms(b, left, right),)


def _zero_vjp(b, op, p, g, active):
    return (None,) * len(p)


def _zero_jvp(*args):
    return (None,)


def rules(Rule):
    return {
        'div': Rule('div', lambda inputs, attrs: _float(inputs, attrs, 2), _div_vjp, _div_jvp, 'Floating division.'),
        'pow': Rule('pow', lambda inputs, attrs: _same_type(inputs, attrs, 2, _NUMERIC_DTYPES),
                    _pow_vjp, _pow_jvp, 'Elementwise power.', False),
        'maximum': Rule('maximum', lambda inputs, attrs: _same_type(inputs, attrs, 2, DTYPES),
                        lambda b, op, p, g, a: _minmax_vjp(b, op, p, g, a, True),
                        lambda b, op, p, t: _minmax_jvp(b, op, p, t, True), 'Elementwise maximum.', False),
        'minimum': Rule('minimum', lambda inputs, attrs: _same_type(inputs, attrs, 2, DTYPES),
                        lambda b, op, p, g, a: _minmax_vjp(b, op, p, g, a, False),
                        lambda b, op, p, t: _minmax_jvp(b, op, p, t, False), 'Elementwise minimum.', False),
        'floordiv': Rule('floordiv', lambda inputs, attrs: _same_type(inputs, attrs, 2, _NUMERIC_DTYPES),
                         _zero_vjp, _zero_jvp, 'Elementwise floor division.', False, ()),
        'mod': Rule('mod', lambda inputs, attrs: _same_type(inputs, attrs, 2, _NUMERIC_DTYPES),
                    _mod_vjp, _mod_jvp, 'Elementwise floor modulo.', False),
        **{name: Rule(name, _float, _unary_vjp, _unary_jvp, 'Floating elementary function.')
           for name in ('exp', 'log', 'sqrt', 'sin', 'cos', 'tanh', 'erf')},
        **{name: Rule(name, _float, _zero_vjp, _zero_jvp, 'Piecewise constant rounding.', False, ())
           for name in ('floor', 'ceil', 'round')},
        'abs': Rule('abs', lambda inputs, attrs: _same_type(inputs, attrs, 1, _NUMERIC_DTYPES),
                    _abs_vjp, _abs_jvp, 'Elementwise absolute value.', False),
        'bitwise_not': Rule('bitwise_not', lambda inputs, attrs: _same_type(inputs, attrs, 1, _BITWISE_DTYPES),
                            _zero_vjp, _zero_jvp, 'Elementwise bitwise/logical negation.', False, ()),
        **{name: Rule(name, lambda inputs, attrs: _same_type(inputs, attrs, 2, _BITWISE_DTYPES),
                      _zero_vjp, _zero_jvp, 'Elementwise bitwise/logical binary op.', False, ())
           for name in ('bitwise_and', 'bitwise_or', 'bitwise_xor')},
        **{name: Rule(name, lambda inputs, attrs: _same_type(inputs, attrs, 2, INT_DTYPES),
                      _zero_vjp, _zero_jvp, 'Elementwise bit shift.', False, ())
           for name in ('shift_left', 'shift_right')},
        'compare': Rule('compare', _compare, lambda *args: (None, None), lambda *args: (None,),
                        'Boolean comparison with no derivative edges.', False, ()),
        'select': Rule('select', _select, _select_vjp, _select_jvp,
                       'Choose values and route derivatives through value operands.', False, (1, 2)),
        'broadcast_in_dim': Rule('broadcast_in_dim', _broadcast, _broadcast_vjp, _map_jvp,
                                 'Broadcast tangents and sum replicated cotangents.'),
        'transpose': Rule('transpose', _transpose, _transpose_vjp, _map_jvp, 'Permute indices; invert for VJP.'),
        'reduce_sum': Rule('reduce_sum', _reduce, _sum_vjp, _map_jvp, 'Sum values; broadcast cotangents.'),
        'reduce_max': Rule('reduce_max', lambda inputs, attrs: _reduce(inputs, attrs, True),
                           lambda b, op, p, g, a: _max_vjp(b, op, p, g, a, is_max=True),
                           lambda b, op, p, t: _max_jvp(b, op, p, t, is_max=True),
                           'Average derivatives over tied maxima.'),
        'reduce_min': Rule('reduce_min', lambda inputs, attrs: _reduce(inputs, attrs, False, nonempty=True),
                           lambda b, op, p, g, a: _max_vjp(b, op, p, g, a, is_max=False),
                           lambda b, op, p, t: _max_jvp(b, op, p, t, is_max=False),
                           'Average derivatives over tied minima.'),
        'reduce_prod': Rule('reduce_prod', lambda inputs, attrs: _reduce(inputs, attrs, False),
                            _prod_vjp, _prod_jvp, 'Zero-safe product reduction derivative.'),
        'reduce_any': Rule('reduce_any', _reduce_bool, _zero_vjp, _zero_jvp,
                           'Boolean any reduction.', False, ()),
        'reduce_all': Rule('reduce_all', _reduce_bool, _zero_vjp, _zero_jvp,
                           'Boolean all reduction.', False, ()),
        'argmax': Rule('argmax', _argreduce, _zero_vjp, _zero_jvp,
                       'Index of maximum along axis.', False, ()),
        'argmin': Rule('argmin', _argreduce, _zero_vjp, _zero_jvp,
                       'Index of minimum along axis.', False, ()),
        'iota': Rule('iota', _iota, _zero_vjp, _zero_jvp,
                     'Coordinate generator along one axis.', False, ()),
        'slice': Rule('slice', _slice, _slice_vjp, _slice_jvp,
                      'Strided slice; VJP embeds cotangent into padded zero tensor.', False),
        'pad': Rule('pad', _pad, _pad_vjp, _pad_jvp,
                    'Edge and interior padding; VJP slices interior and sums pad borders.', False),
        'concatenate': Rule('concatenate', _concatenate, _concat_vjp, _concat_jvp,
                            'Concatenation along one axis; VJP slices cotangents.', False),
        'gather': Rule('gather', _gather, _gather_vjp, _gather_jvp,
                       'Axis indexing; VJP scatters cotangents additively.', False, (0,)),
        'scatter_add': Rule('scatter_add', _scatter_add, _scatter_add_vjp, _scatter_add_jvp,
                            'Functional scatter-add update; adjoint of gather.', False, (0, 2)),
        'matmul': Rule('matmul', _matmul, _matmul_vjp, _matmul_jvp, 'Batched contraction with f32 accumulation.'),
    }
