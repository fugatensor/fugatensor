"""Typing and composable derivatives for tensor maps, reductions and contractions."""
from .types import FLOAT_DTYPES, ScalarType, TensorType, cast_type


def _shape(type_):
    return type_.shape if isinstance(type_, TensorType) else ()


def _float(inputs, attrs, count=1):
    if len(inputs) != count or attrs or any(value != inputs[0] for value in inputs):
        raise TypeError('floating operation requires matching operands and no attributes')
    if inputs[0].dtype not in FLOAT_DTYPES:
        raise TypeError('operation requires a floating dtype')
    return (inputs[0],)


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


def _reduce(inputs, attrs, maximum=False):
    if len(inputs) != 1 or set(attrs) != {'axes'} or inputs[0].dtype == 'bool':
        raise TypeError('reduction requires one numeric operand and axes')
    shape, axes = _shape(inputs[0]), tuple(attrs['axes'])
    if None in shape:
        raise ValueError('reduction requires concrete shapes')
    if (any(type(axis) is not int or not 0 <= axis < len(shape) for axis in axes)
            or tuple(sorted(set(axes))) != axes):
        raise ValueError('reduction axes must be sorted, unique and in range')
    if maximum and any(shape[axis] == 0 for axis in axes):
        raise ValueError('max reduction requires a nonempty reduction domain')
    remaining = tuple(size for axis, size in enumerate(shape) if axis not in axes)
    return (TensorType(remaining, inputs[0].dtype) if remaining else ScalarType(inputs[0].dtype),)


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
    denominator = x if opcode == 'log' else b.mul(_filled(b, x.type, 2), b.sqrt(x))
    return b.div(_filled(b, x.type, 1), denominator)


def _unary_vjp(b, op, p, g, active):
    return (b.mul(g, _unary_factor(b, op.opcode, p[0])) if active[0] else None,)


def _unary_jvp(b, op, p, t):
    return (None if t[0] is None else b.mul(t[0], _unary_factor(b, op.opcode, p[0])),)


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


def _max_weights(b, op, x):
    axes = op.attrs['axes']
    maximum = b.reduce_max(x, axes)
    maximum = _expand_reduction(b, maximum, x.type, axes)
    mask = b.compare(x, maximum, 'eq')
    counts = b.reduce_sum(b.cast(mask, dtype='f32'), axes)
    counts = _expand_reduction(b, counts, cast_type(x.type, 'f32'), axes)
    weight = b.cast(b.div(b.cast(mask, dtype='f32'), counts), dtype=x.type.dtype)
    # NaN maxima propagate through the value and its chosen derivative rule.
    return b.select(b.compare(maximum, maximum, 'ne'), _filled(b, x.type, float('nan')), weight)


def _max_vjp(b, op, p, g, active):
    return (b.mul(_max_weights(b, op, p[0]), _expand_reduction(b, g, p[0].type, op.attrs['axes']))
            if active[0] else None,)


def _max_jvp(b, op, p, t):
    return (None if t[0] is None else b.reduce_sum(b.mul(_max_weights(b, op, p[0]), t[0]), op.attrs['axes']),)


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


def rules(Rule):
    return {
        'div': Rule('div', lambda inputs, attrs: _float(inputs, attrs, 2), _div_vjp, _div_jvp, 'Floating division.'),
        **{name: Rule(name, _float, _unary_vjp, _unary_jvp, 'Floating elementary function.')
           for name in ('exp', 'log', 'sqrt')},
        'compare': Rule('compare', _compare, lambda *args: (None, None), lambda *args: (None,),
                        'Boolean comparison with no derivative edges.', False, ()),
        'select': Rule('select', _select, _select_vjp, _select_jvp,
                       'Choose values and route derivatives through value operands.', False, (1, 2)),
        'broadcast_in_dim': Rule('broadcast_in_dim', _broadcast, _broadcast_vjp, _map_jvp,
                                 'Broadcast tangents and sum replicated cotangents.'),
        'transpose': Rule('transpose', _transpose, _transpose_vjp, _map_jvp, 'Permute indices; invert for VJP.'),
        'reduce_sum': Rule('reduce_sum', _reduce, _sum_vjp, _map_jvp, 'Sum values; broadcast cotangents.'),
        'reduce_max': Rule('reduce_max', lambda inputs, attrs: _reduce(inputs, attrs, True),
                           _max_vjp, _max_jvp, 'Average derivatives over tied maxima.'),
        'matmul': Rule('matmul', _matmul, _matmul_vjp, _matmul_jvp, 'Batched contraction with f32 accumulation.'),
    }
