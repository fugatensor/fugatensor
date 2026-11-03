"""Resolve static structured control flow before handing a graph to ASK.

The input module remains intact. In particular, a tensor-dependent predicate is
never evaluated using tracing inputs or replaced with a guessed branch.
"""
from __future__ import annotations

from .core import Module
from .types import FLOAT_DTYPES, ScalarType, constant_value


class StaticControlFlowError(ValueError):
    """The graph needs control-flow execution that ASK does not provide yet."""


def _clone_operation(target, operation, operands):
    """Retain an operation's identity, giving repeated expansions distinct IDs."""
    identifier, suffix = operation.id, 0
    while identifier in target.function._ids:
        suffix += 1
        identifier = f'{operation.id}.clone{suffix}'
    return target.operation(operation.opcode, operands,
                            tuple(value.type for value in operation.results),
                            attrs=operation.attrs, effects=operation.effects,
                            id=identifier, origin_id=operation.origin_id)


def _iterations(lower, upper, step):
    if step > 0:
        return max(0, (upper - lower + step - 1) // step)
    return max(0, (lower - upper - step - 1) // -step)


def _known_scalar(op, operands, constants):
    """Small exact integer analysis used only to decide static predicates.

    This is not a tensor evaluator or a floating constant-folding pass. Floating
    arithmetic stays in the graph and is executed by the numerical backend.
    """
    if op.opcode == "constant":
        return constant_value(op.attrs["value"], op.results[0].type.dtype)
    if len(op.results) != 1 or not isinstance(op.results[0].type, ScalarType):
        return None
    if not operands or any(value not in constants for value in operands):
        return None
    values = [constants[value] for value in operands]
    destination = op.results[0].type.dtype
    if op.opcode == "cast":
        if destination == "bool":
            # Literal floating values have already been rounded by the builder.
            return bool(values[0])
        if destination in ("i32", "i64") and type(values[0]) in (bool, int):
            value = int(values[0])
        else:
            return None
    elif op.opcode in ("add", "mul") and destination in ("i32", "i64"):
        value = values[0] + values[1] if op.opcode == "add" else values[0] * values[1]
    else:
        return None
    width = int(destination[1:])
    return (value + 2 ** (width - 1)) % (2 ** width) - 2 ** (width - 1)


def specialize(module, *, max_unroll=10000, expand_calls=True):
    """Return a new flat module for static ASK compilation.

    ``if`` requires a compile-time scalar bool. Counted ``for`` loops have
    static signed integer bounds and are unrolled with explicit carried values.
    The total number of expanded loop iterations is bounded by ``max_unroll``;
    exceeding it is an error, never silent truncation. Data-dependent ``if``
    and ``while`` remain valid Fuga IR but cannot enter this lowering path.
    Custom calls expand their forward graphs by default. AD uses
    ``expand_calls=False`` to retain each call's explicit VJP rule.
    """
    if not isinstance(module, Module):
        raise TypeError("specialize expects a Fuga IR Module")
    if type(max_unroll) is not int or max_unroll < 0:
        raise ValueError("max_unroll must be a nonnegative integer")
    if type(expand_calls) is not bool:
        raise TypeError("expand_calls must be bool")
    module.verify()
    result = Module()
    budget = [max_unroll]

    for function in module.functions:
        target = result.graph(function.name,
                              inputs={name: value.type for name, value in function.inputs.items()},
                              results=[value.type for value in function.outputs])
        constants = {}

        def clone_block(block, mapping):
            for operation in block.operations:
                operands = tuple(mapping[value] for value in operation.operands)
                if operation.opcode in ("return", "yield"):
                    return operands
                if operation.opcode == "if":
                    predicate = constants.get(operands[0])
                    if type(predicate) is not bool:
                        raise StaticControlFlowError(
                            "data-dependent fuga.if requires framework control-flow execution; "
                            "ASK accepts only a statically selected branch")
                    branch = operation.regions[0 if predicate else 1].blocks[0]
                    values = clone_block(branch, dict(zip(branch.arguments, operands[1:])))
                elif operation.opcode == "for":
                    lower, upper, step = (operation.attrs[key] for key in ("lower", "upper", "step"))
                    count = _iterations(lower, upper, step)
                    if count > budget[0]:
                        raise StaticControlFlowError(
                            f"static loop expansion exceeds max_unroll={max_unroll}; "
                            "the loop was not partially compiled")
                    budget[0] -= count
                    values = operands
                    body = operation.regions[0].blocks[0]
                    for index in range(lower, upper, step):
                        identifier = f'{operation.id}.iteration{index}.index'
                        while identifier in function._ids or identifier in target._ids:
                            identifier += '.generated'
                        induction = target.constant(index, dtype="i64", id=identifier)
                        constants[induction] = index
                        values = clone_block(body, dict(zip(body.arguments, (induction, *values))))
                elif operation.opcode == "while":
                    raise StaticControlFlowError(
                        "fuga.while requires framework control-flow execution; "
                        "ASK lowering currently accepts only static if/for control flow")
                elif operation.opcode == "call" and expand_calls:
                    callee = operation.attrs["spec"].forward.functions[0]
                    values = clone_block(callee.body.blocks[0], dict(zip(callee.inputs.values(), operands)))
                else:
                    if operation.regions:
                        raise StaticControlFlowError(
                            f"no static region lowering for fuga.{operation.opcode}")
                    copied = _clone_operation(target, operation, operands)
                    values = copied.results
                    known = _known_scalar(operation, operands, constants)
                    if known is not None and len(values) == 1:
                        constants[values[0]] = known
                mapping.update(zip(operation.results, values))
            raise ValueError("verified block has no terminator")

        mapping = dict(zip(function.inputs.values(), target.inputs.values()))
        target.return_(*clone_block(function.body.blocks[0], mapping))
    result.verify()
    return result


def analyze_function(function, selected=()):
    """Convert flat SSA edges once; the native C++ core owns liveness and AD reachability."""
    from fugatensor import _native
    values = list(function.inputs.values())
    indices = {value: index for index, value in enumerate(values)}
    operations = tuple(op for op in function.operations if op.opcode != "return")
    operands, counts, effects = [], [], []
    for operation in operations:
        operands.append([indices[value] for value in operation.operands])
        counts.append(len(operation.results))
        effects.append(bool(operation.effects))
        for value in operation.results:
            indices[value] = len(values)
            values.append(value)
    plan = _native.analyze(len(function.inputs), operands, counts,
                           [indices[value] for value in function.outputs], list(selected), effects)
    return operations, values, plan


def prune(module):
    """Drop dead effect-free operations from a flat Fuga module."""
    result = Module()
    for source in module.functions:
        operations, _, plan = analyze_function(source)
        kept = {id(operations[index]) for index in plan["live_operations"]}
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


def _derivative_graph(function, memo):
    """Describe per-result derivative edges for the native SSA analyzer.

    Each virtual operation defines one original value. Numerical operations
    keep their operand edges; stop_gradient and custom calls provide their
    semantic derivative edges without changing the numerical IR.
    """
    values = list(function.inputs.values())
    indices = {value: index for index, value in enumerate(values)}
    operands = []
    for operation in function.operations:
        if operation.opcode == "return":
            continue
        if operation.opcode == "stop_gradient":
            edges = ((),)
        elif operation.opcode == "call":
            spec = operation.attrs["spec"]
            spec.verify()
            if spec.has_vjp:
                selected = tuple(index for index, value in enumerate(operation.operands)
                                 if value.type.dtype in FLOAT_DTYPES)
                edges = tuple(selected if value.type.dtype in FLOAT_DTYPES else () for value in operation.results)
            else:
                if spec not in memo:
                    forward = specialize(spec.forward, expand_calls=False).functions[0]
                    memo[spec] = tuple(_derivative_input_indices(forward, (index,), memo)
                                       for index in range(len(forward.outputs)))
                edges = memo[spec]
        else:
            edges = (tuple(range(len(operation.operands))),) * len(operation.results)
        for result, edge in zip(operation.results, edges):
            operands.append([indices[operation.operands[index]] for index in edge])
            indices[result] = len(values)
            values.append(result)
    return values, operands, indices


def _derivative_analysis(function, selected, outputs, memo):
    from .. import _native
    values, operands, indices = _derivative_graph(function, memo)
    result_indices = [indices[function.outputs[index]] for index in outputs]
    plan = _native.analyze(len(function.inputs), operands, [1] * len(operands),
                           result_indices, list(selected), [False] * len(operands))
    return values, operands, result_indices, plan


def _derivative_input_indices(function, outputs, memo):
    _, operands, result_indices, plan = _derivative_analysis(function, (), outputs, memo)
    count = len(function.inputs)
    inputs = {index for index in result_indices if index < count}
    for operation in plan["live_operations"]:
        inputs.update(index for index in operands[operation] if index < count)
    return tuple(sorted(inputs))


def derivative_inputs(module, output_indices=None):
    """Use native reachability to find inputs on selected derivative paths."""
    indices = range(len(module.functions[0].outputs)) if output_indices is None else output_indices
    function = project_outputs(module, indices).functions[0]
    return _derivative_input_indices(function, range(len(function.outputs)), {})


def derivative_activity(function, selected, *, live_only=False):
    values, operands, outputs, plan = _derivative_analysis(function, selected, range(len(function.outputs)), {})
    if not live_only:
        return dict(zip(values, plan["active_values"]))
    live = set(outputs)
    for operation in plan["live_operations"]:
        live.add(len(function.inputs) + operation)
        live.update(operands[operation])
    return {value: bool(active and index in live)
            for index, (value, active) in enumerate(zip(values, plan["active_values"]))}


def project_outputs(module, indices, *, name_suffix="", max_unroll=10000):
    """Select a subset of result ports and prune unused operations without expanding custom calls."""
    source = specialize(module, max_unroll=max_unroll, expand_calls=False).functions[0]
    result = Module()
    target = result.graph(source.name + name_suffix,
                          inputs={name: value.type for name, value in source.inputs.items()},
                          results=[source.outputs[index].type for index in indices])
    mapping = dict(zip(source.inputs.values(), target.inputs.values()))
    for operation in source.operations:
        if operation.opcode == "return":
            break
        copied = _clone_operation(target, operation, tuple(mapping[value] for value in operation.operands))
        mapping.update(zip(operation.results, copied.results))
    target.return_(*(mapping[source.outputs[index]] for index in indices))
    return prune(result.verify())
