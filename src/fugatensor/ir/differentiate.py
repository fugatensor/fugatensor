"""A small reverse-mode transform over independent Fuga SSA operations.

This builds a graph; it does not execute tensor arithmetic or maintain a Python
tape. The caller supplies output cotangents explicitly, so vector outputs do
not acquire an implicit reduction or an invented scalar loss.
"""
from __future__ import annotations

from .core import Module
from .types import FLOAT_DTYPES, TensorType
from .transforms import _clone_operation, analyze_function, project_outputs, prune, specialize
from .errors import DifferentiationError
from .primitives import primitive_vjp


def _inline_derivative(target, module, bindings):
    source = module.functions[0]
    mapping = dict(zip(source.inputs.values(), bindings))
    for operation in source.operations:
        if operation.opcode == "return":
            return tuple(mapping[value] for value in operation.operands)
        copied = _clone_operation(target, operation, tuple(mapping[value] for value in operation.operands))
        mapping.update(zip(operation.results, copied.results))
    raise DifferentiationError("a residual derivative requires a return terminator")


def vjp(module, *, argnums=None, max_unroll=10000):
    """Build an explicit vector-Jacobian product for one floating graph.

    The new function receives original inputs followed by one cotangent seed
    per original result, and returns cotangents for ``argnums`` in that order.
    By default every floating input is selected. Original outputs and all
    selected inputs must be floating; integer/bool differentiation is rejected.

    Implemented rules are add, mul, reshape and floating casts. The cast rule casts the
    cotangent back to the input dtype: as in usual floating autodiff it treats
    the cast as a real-valued identity and does not differentiate quantization.
    A live external needs an explicit future VJP rule; it is never treated as
    identity. Differentiating scalar-to-tensor splat needs a reduction and is
    rejected until that operation has an execution path.
    """
    if not isinstance(module, Module):
        raise TypeError("vjp expects a Fuga IR Module")
    flat = specialize(module, max_unroll=max_unroll, expand_calls=False)
    if len(flat.functions) != 1:
        raise DifferentiationError("vjp currently requires exactly one Fuga function")
    source = flat.functions[0]
    source_inputs = tuple(source.inputs.values())
    if argnums is None:
        selected = tuple(index for index, value in enumerate(source_inputs) if value.type.dtype in FLOAT_DTYPES)
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
    if any(source_inputs[index].type.dtype not in FLOAT_DTYPES for index in selected):
        raise DifferentiationError("integer and bool inputs are not differentiable")
    if not source.outputs or any(value.type.dtype not in FLOAT_DTYPES for value in source.outputs):
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
    operations, numbered_values, plan = analyze_function(source, selected)
    active = dict(zip(numbered_values, plan['active_values']))
    if any(operation.effects for operation in operations):
        raise DifferentiationError("vjp currently requires an effect-free graph")
    for operation in operations:
        copied = _clone_operation(target, operation, tuple(values[value] for value in operation.operands))
        values.update(zip(operation.results, copied.results))

    adjoints = {}

    def contribute(value, cotangent):
        if not active[value]:
            return
        if value in adjoints:
            cotangent = target.add(adjoints[value], cotangent)
        adjoints[value] = cotangent

    for value, name in zip(source.outputs, seed_names):
        contribute(value, target.inputs[name])
    def zero(type_):
        value = target.constant(0.0, dtype=type_.dtype)
        return target.splat(value, type_.shape) if isinstance(type_, TensorType) else value

    for index in plan['reverse_operations']:
        operation = operations[index]
        if not any(value in adjoints for value in operation.results):
            continue
        if operation.opcode == "external":
            raise DifferentiationError(f"fuga.external {operation.id!r} on a derivative path requires an explicit VJP rule")
        if operation.opcode == 'call':
            spec = operation.attrs['spec']
            primals = tuple(values[value] for value in operation.operands)
            if any(value in adjoints for value in operation.results[:spec.output_count]):
                if not spec.has_backward:
                    raise DifferentiationError(f'{spec.name}.backward is required on this derivative path')
                cotangents = tuple(adjoints[value] if value in adjoints else zero(value.type)
                                   for value in operation.results[:spec.output_count])
                gradients = spec.build_backward(target, primals,
                    tuple(values[value] for value in operation.results), cotangents)
                inputs = tuple(value for value in operation.operands if value.type.dtype in FLOAT_DTYPES)
                if len(gradients) != len(inputs):
                    raise DifferentiationError(f'{spec.name}.backward returned an invalid gradient count')
                for value, gradient in zip(inputs, gradients):
                    if gradient.type != value.type:
                        raise DifferentiationError(f'{spec.name}.backward gradient shape and dtype must match {value.type}')
                    contribute(value, gradient)
            residual_indices = tuple(index for index in range(spec.output_count, len(operation.results))
                                     if operation.results[index] in adjoints)
            if residual_indices:
                selected_inputs = tuple(index for index, value in enumerate(operation.operands)
                                        if active[value] and value.type.dtype in FLOAT_DTYPES)
                if selected_inputs:
                    # Hidden values are the explicit computations saved by
                    # forward. Their derivatives follow that SSA graph, while
                    # any visible nested call still chooses its custom rule.
                    residuals = project_outputs(spec.forward, residual_indices,
                                                name_suffix="_residuals", max_unroll=max_unroll)
                    derivative = vjp(residuals, argnums=selected_inputs, max_unroll=max_unroll)
                    seeds = tuple(adjoints[operation.results[index]] for index in residual_indices)
                    gradients = _inline_derivative(target, derivative, (*primals, *seeds))
                    for index, gradient in zip(selected_inputs, gradients):
                        contribute(operation.operands[index], gradient)
            continue
        if len(operation.results) != 1:
            raise DifferentiationError(f"no reverse rule for multi-result fuga.{operation.opcode}")
        cotangent = adjoints[operation.results[0]]
        gradients = primitive_vjp(target, operation,
                                  tuple(values[value] for value in operation.operands), cotangent,
                                  tuple(active[value] for value in operation.operands))
        for value, gradient in zip(operation.operands, gradients):
            if gradient is not None:
                contribute(value, gradient)

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
    return prune(result)
