"""Forward-mode differentiation emits ordinary, composable Fuga primitives."""
from .core import Module
from .errors import DifferentiationError
from .transforms import _clone_operation, analyze_function, derivative_activity, prune, specialize
from .types import FLOAT_DTYPES, TensorType


def _constant_jvp(builder, operation, primals, tangents):
    return (None,)


def _splat_jvp(builder, operation, primals, tangents):
    return (None if tangents[0] is None else builder.splat(tangents[0], operation.results[0].type.shape),)


def _add_jvp(builder, operation, primals, tangents):
    left, right = tangents
    return (right if left is None else left if right is None else builder.add(left, right),)


def _mul_jvp(builder, operation, primals, tangents):
    left, right = tangents
    first = None if left is None else builder.mul(left, primals[1])
    second = None if right is None else builder.mul(primals[0], right)
    return (second if first is None else first if second is None else builder.add(first, second),)


def _cast_jvp(builder, operation, primals, tangents):
    return (None if tangents[0] is None else builder.cast(tangents[0], dtype=operation.results[0].type.dtype),)


def jvp(module, *, argnums=None, max_unroll=10000):
    """Return a graph receiving primals, then tangents for selected inputs.

    Results are all original outputs followed by their tangents in the same
    order. Unselected inputs have zero tangents. Floating casts use the usual
    real-valued identity derivative, ignoring storage rounding. Static regions
    are specialized first. Ordinary calls differentiate their forward graphs;
    calls with a custom VJP and external operations require an explicit JVP.
    """
    from .primitives import PRIMITIVE_RULES, validate_primitive
    if not isinstance(module, Module):
        raise TypeError("jvp expects a Fuga IR Module")
    flat = prune(specialize(module, max_unroll=max_unroll, expand_calls=False))
    if len(flat.functions) != 1:
        raise DifferentiationError("jvp currently requires exactly one Fuga function")
    source = flat.functions[0]
    inputs = tuple(source.inputs.values())
    selected = (tuple(index for index, value in enumerate(inputs) if value.type.dtype in FLOAT_DTYPES)
                if argnums is None else (argnums,) if type(argnums) is int else argnums)
    if not isinstance(selected, (tuple, list)):
        raise TypeError("argnums must be an input index or a tuple/list of indices")
    if not selected or any(type(index) is not int or not 0 <= index < len(inputs) for index in selected):
        raise ValueError("argnums must contain valid nonnegative input indices")
    if len(set(selected)) != len(selected):
        raise ValueError("argnums must not repeat an input index")
    if any(inputs[index].type.dtype not in FLOAT_DTYPES for index in selected):
        raise DifferentiationError("integer and bool inputs are not differentiable")
    if not source.outputs or any(value.type.dtype not in FLOAT_DTYPES for value in source.outputs):
        raise DifferentiationError("jvp requires floating outputs")
    if any(isinstance(value.type, TensorType) and None in value.type.shape
           for value in (*source.outputs, *(inputs[index] for index in selected))):
        raise DifferentiationError("jvp requires concrete differentiated tensor shapes")
    operations, numbered, plan = analyze_function(source, selected)
    if any(operation.effects for operation in operations):
        raise DifferentiationError("jvp currently requires an effect-free graph")
    active = derivative_activity(source, selected, live_only=True)
    input_types = {name: value.type for name, value in source.inputs.items()}
    tangent_names = []
    for index in selected:
        name = f"tangent{index}"
        while name in input_types:
            name = "_" + name
        input_types[name] = inputs[index].type
        tangent_names.append(name)
    result = Module()
    target = result.graph(source.name + "_jvp", inputs=input_types,
                          results=[value.type for value in source.outputs] * 2)
    primals = {value: target.inputs[name] for name, value in source.inputs.items()}
    tangents = dict.fromkeys(inputs)
    for index, name in zip(selected, tangent_names):
        tangents[inputs[index]] = target.inputs[name]
    for operation in operations:
        operands = tuple(primals[value] for value in operation.operands)
        copied = _clone_operation(target, operation, operands)
        primals.update(zip(operation.results, copied.results))
        tangent_results = (None,) * len(operation.results)
        if any(active[value] for value in operation.results):
            if operation.opcode == "call" and not operation.attrs["spec"].has_vjp:
                from .differentiate import _inline_derivative
                from .transforms import project_outputs
                spec = operation.attrs["spec"]
                spec.verify()
                selected_inputs = tuple(index for index, value in enumerate(operation.operands)
                                        if active[value] and value.type.dtype in FLOAT_DTYPES)
                selected_outputs = tuple(index for index, value in enumerate(operation.results)
                                         if active[value] and value.type.dtype in FLOAT_DTYPES)
                if selected_inputs and selected_outputs:
                    forward = project_outputs(spec.forward, selected_outputs, max_unroll=max_unroll)
                    derivative = jvp(forward, argnums=selected_inputs, max_unroll=max_unroll)
                    results = _inline_derivative(target, derivative,
                        (*operands, *(tangents[operation.operands[index]] for index in selected_inputs)))
                    mapped = dict(zip(selected_outputs, results[len(selected_outputs):]))
                    tangent_results = tuple(mapped.get(index) for index in range(len(operation.results)))
                tangents.update(zip(operation.results, tangent_results))
                continue
            if operation.opcode not in PRIMITIVE_RULES:
                raise DifferentiationError(f"fuga.{operation.opcode} {operation.id!r} requires an explicit JVP rule")
            rule = validate_primitive(operation)
            if not callable(rule.jvp):
                raise DifferentiationError(f"fuga.{operation.opcode} has no registered JVP rule")
            if rule.floating_values and any(value.type.dtype not in FLOAT_DTYPES for value in (*operation.operands, *operation.results)):
                raise DifferentiationError("integer/bool operations on a derivative path have no floating JVP rule")
            tangent_results = rule.jvp(target, operation, operands,
                                       tuple(tangents[value] for value in operation.operands))
            if len(tangent_results) != len(operation.results) or any(
                    tangent is not None and tangent.type != original.type
                    for tangent, original in zip(tangent_results, operation.results)):
                raise DifferentiationError(f"fuga.{operation.opcode} JVP returned invalid tangent types")
        tangents.update(zip(operation.results, tangent_results))

    def tangent(value):
        if tangents[value] is not None:
            return tangents[value]
        zero = target.constant(0, dtype=value.type.dtype)
        return target.splat(zero, value.type.shape) if isinstance(value.type, TensorType) else zero

    target.return_(*(primals[value] for value in source.outputs), *(tangent(value) for value in source.outputs))
    return prune(result.verify())
