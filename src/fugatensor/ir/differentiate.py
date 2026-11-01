"""A small reverse-mode transform over independent Fuga SSA operations.

This builds a graph; it does not execute tensor arithmetic or maintain a Python
tape. The caller supplies output cotangents explicitly, so vector outputs do
not acquire an implicit reduction or an invented scalar loss.
"""
from __future__ import annotations

from .core import Module
from .types import TensorType
from .transforms import specialize, _clone_operation


_FLOATS = frozenset(("f16", "bf16", "f32"))


class DifferentiationError(ValueError):
    """The requested derivative has no implemented Fuga IR rule."""


def _prune_flat(module):
    """Drop unused pure primal work after constructing the reverse graph."""
    result = Module()
    for source in module.functions:
        live = set(source.outputs)
        kept = set()
        for operation in reversed(source.operations):
            if operation.opcode == "return":
                continue
            if operation.effects or any(value in live for value in operation.results):
                kept.add(id(operation))
                live.update(operation.operands)
        target = result.graph(source.name,
                              inputs={name: value.type for name, value in source.inputs.items()},
                              results=[value.type for value in source.outputs])
        values = dict(zip(source.inputs.values(), target.inputs.values()))
        for operation in source.operations:
            if id(operation) not in kept:
                continue
            copied = _clone_operation(target, operation, tuple(values[value] for value in operation.operands))
            values.update(zip(operation.results, copied.results))
        target.return_(*(values[value] for value in source.outputs))
    result.verify()
    return result


def vjp(module, *, argnums=None, max_unroll=10000):
    """Build an explicit vector-Jacobian product for one floating graph.

    The new function receives original inputs followed by one cotangent seed
    per original result, and returns cotangents for ``argnums`` in that order.
    By default every floating input is selected. Original outputs and all
    selected inputs must be floating; integer/bool differentiation is rejected.

    Implemented rules are add, mul and floating casts. The cast rule casts the
    cotangent back to the input dtype: as in usual floating autodiff it treats
    the cast as a real-valued identity and does not differentiate quantization.
    A live external needs an explicit future VJP rule; it is never treated as
    identity. Differentiating scalar-to-tensor splat needs a reduction and is
    rejected until that operation has an execution path.
    """
    if not isinstance(module, Module):
        raise TypeError("vjp expects a Fuga IR Module")
    flat = specialize(module, max_unroll=max_unroll)
    if len(flat.functions) != 1:
        raise DifferentiationError("vjp currently requires exactly one Fuga function")
    source = flat.functions[0]
    source_inputs = tuple(source.inputs.values())
    if argnums is None:
        selected = tuple(index for index, value in enumerate(source_inputs) if value.type.dtype in _FLOATS)
    elif type(argnums) is int:
        selected = (argnums,)
    elif isinstance(argnums, (tuple, list)):
        selected = tuple(argnums)
    else:
        raise TypeError("argnums must be an input index or a tuple/list of input indices")
    if not selected:
        raise DifferentiationError("vjp requires at least one floating input to differentiate")
    if any(type(index) is not int or not 0 <= index < len(source_inputs) for index in selected):
        raise ValueError("argnums must contain valid nonnegative input indices")
    if len(set(selected)) != len(selected):
        raise ValueError("argnums must not repeat an input index")
    if any(source_inputs[index].type.dtype not in _FLOATS for index in selected):
        raise DifferentiationError("integer and bool inputs are not differentiable")
    if not source.outputs or any(value.type.dtype not in _FLOATS for value in source.outputs):
        raise DifferentiationError("vjp requires floating outputs with explicit floating cotangent seeds")
    for index in selected:
        type_ = source_inputs[index].type
        if isinstance(type_, TensorType) and any(dimension is None for dimension in type_.shape):
            raise DifferentiationError("vjp currently requires concrete differentiated tensor shapes")

    result = Module()
    input_types = {name: value.type for name, value in source.inputs.items()}
    seed_names = []
    for index, output in enumerate(source.outputs):
        name = f"seed{index}"
        while name in input_types:
            name = "_" + name
        seed_names.append(name)
        input_types[name] = output.type
    target = result.graph(source.name + "_vjp", inputs=input_types,
                          results=[source_inputs[index].type for index in selected])
    values = {value: target.inputs[name] for name, value in source.inputs.items()}
    active = {value: index in selected for index, value in enumerate(source_inputs)}
    operations = tuple(op for op in source.operations if op.opcode != "return")
    if any(operation.effects for operation in operations):
        raise DifferentiationError("vjp currently requires an effect-free graph")
    for operation in operations:
        copied = _clone_operation(target, operation, tuple(values[value] for value in operation.operands))
        values.update(zip(operation.results, copied.results))
        for output in operation.results:
            active[output] = any(active[value] for value in operation.operands)

    adjoints = {}

    def contribute(value, cotangent):
        if not active[value]:
            return
        if value in adjoints:
            cotangent = target.add(adjoints[value], cotangent)
        adjoints[value] = cotangent

    for value, name in zip(source.outputs, seed_names):
        contribute(value, target.inputs[name])
    for operation in reversed(operations):
        if not any(value in adjoints for value in operation.results):
            continue
        if operation.opcode == "constant":
            continue
        if operation.opcode == "external":
            raise DifferentiationError("fuga.external on a derivative path requires an explicit VJP rule")
        if len(operation.results) != 1:
            raise DifferentiationError(f"no reverse rule for multi-result fuga.{operation.opcode}")
        cotangent = adjoints[operation.results[0]]
        if operation.opcode == "add":
            for value in operation.operands:
                contribute(value, cotangent)
        elif operation.opcode == "mul":
            left, right = operation.operands
            if active[left]:
                contribute(left, target.mul(cotangent, values[right]))
            if active[right]:
                contribute(right, target.mul(cotangent, values[left]))
        elif operation.opcode == "cast":
            value = operation.operands[0]
            if value.type.dtype not in _FLOATS or operation.results[0].type.dtype not in _FLOATS:
                raise DifferentiationError("integer/bool casts on a derivative path have no floating VJP rule")
            contribute(value, target.cast(cotangent, dtype=value.type.dtype))
        elif operation.opcode == "splat":
            raise DifferentiationError("scalar-input splat VJP requires a sum reduction, which is not implemented")
        else:
            raise DifferentiationError(f"no reverse rule for fuga.{operation.opcode}")

    gradients = []
    for index in selected:
        value = source_inputs[index]
        gradient = adjoints.get(value)
        if gradient is None:
            gradient = target.constant(0.0, dtype=value.type.dtype)
            if isinstance(value.type, TensorType):
                gradient = target.splat(gradient, value.type.shape)
        gradients.append(gradient)
    target.return_(*gradients)
    result.verify()
    return _prune_flat(result)
