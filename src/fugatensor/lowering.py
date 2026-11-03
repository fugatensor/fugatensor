"""The explicit boundary from Fuga semantics to ASK's static graph API."""
from __future__ import annotations

from collections.abc import Mapping

from .ir import Module, ScalarType, TensorType, specialize


class LoweringError(ValueError):
    """A valid Fuga program has no supported ASK/device implementation."""


def _ask_type(type_):
    from fugatensor._ask import ScalarType as ASKScalar, TensorType as ASKTensor
    if isinstance(type_, TensorType):
        if None in type_.shape:
            raise LoweringError('ASK lowering requires concrete shapes; specialize symbolic Fuga dimensions first')
        return ASKTensor(type_.shape, type_.dtype)
    if isinstance(type_, ScalarType):
        return ASKScalar(type_.dtype)
    raise LoweringError(f'unsupported Fuga type: {type_}')


def _target_for(spec, target):
    implementation = dict(spec.target)
    if not implementation.get('backend'):
        raise LoweringError('external implementation must declare its supported backend')
    if target is not None:
        if target.get('backend') != implementation['backend']:
            raise LoweringError(f"external {spec.compiler!r} has no implementation for target {target.get('backend')!r}")
        for key, value in target.items():
            if key in implementation and implementation[key] != value:
                raise LoweringError(f'external implementation fixes {key}={implementation[key]!r}; requested {value!r}')
            implementation[key] = value
    if implementation['backend'] == 'cuda' and (type(implementation.get('arch')) is not int
                                               or implementation['arch'] < 10):
        raise LoweringError('external CUDA implementation requires an explicit target architecture at lowering')
    return implementation


def lower_to_ask(module, *, target=None, max_unroll=10000, source_map=None):
    """Verify and specialize Fuga control flow, then emit a real ASK Module.

    This operation never allocates device tensors or invokes a DSL compiler.
    Device selection binds external implementations here; ordinary operators
    and their types have exactly the same meaning on CPU and CUDA.
    """
    if not isinstance(module, Module):
        raise TypeError('lower_to_ask expects an independent Fuga IR Module')
    if source_map is not None and not isinstance(source_map, dict):
        raise TypeError('source_map must be a dictionary to receive the Fuga-to-ASK mapping')
    if target is not None and (not isinstance(target, Mapping)
                               or target.get('backend') not in ('host', 'cuda')):
        raise LoweringError('target must specify backend host or cuda')
    from fugatensor._ask import Module as ASKModule
    flat = specialize(module, max_unroll=max_unroll)
    output = ASKModule()
    external_count = 0
    mapping = {'schema_version': 1, 'values': []}
    multiple_functions = len(flat.functions) > 1

    def record(function, value, lowered):
        mapping['values'].append({'function': function.name,
                                  'fuga_value': value.name, 'fuga_value_id': value.value_id,
                                  'origin_value_id': value.origin_id, 'ask_value': lowered.name,
                                  'ask_value_id': lowered.value_id})
    for function in flat.functions:
        graph = output.graph(function.name,
                             inputs={name: _ask_type(value.type) for name, value in function.inputs.items()},
                             results=[_ask_type(value.type) for value in function.outputs])
        values = dict(zip(function.inputs.values(), graph.inputs.values()))
        for value, lowered in values.items():
            record(function, value, lowered)
        for op in function.operations:
            operands = [values[value] for value in op.operands]
            if op.opcode == 'return':
                graph.return_(*operands)
                continue
            if len(op.results) != 1 and op.opcode != 'external':
                raise LoweringError(f'ASK bridge requires one result for fuga.{op.opcode}')
            # Fuga IDs are local to a function; ASK IDs are module-wide.
            # Keep the existing single-graph spelling and qualify every ID in
            # multi-function modules to avoid collisions with user-supplied IDs.
            prefix = function.name + '::' if multiple_functions else ''
            kwargs = {'id': prefix + op.id}
            if len(op.results) == 1:
                kwargs['value_id'] = prefix + op.results[0].value_id
            if op.opcode == 'constant':
                result = graph.constant(op.attrs['value'], dtype=op.results[0].type.dtype, **kwargs)
            elif op.opcode == 'splat':
                result = graph.splat(operands[0], op.results[0].type.shape, **kwargs)
            elif op.opcode == 'cast':
                result = graph.cast(operands[0], dtype=op.results[0].type.dtype, **kwargs)
            elif op.opcode == 'reshape':
                result = graph.reshape(operands[0], op.results[0].type.shape, **kwargs)
            elif op.opcode == 'stop_gradient':
                result = operands[0]
            elif op.opcode in ('add', 'mul'):
                result = getattr(graph, op.opcode)(*operands, **kwargs)
            elif op.opcode == 'external':
                spec = op.attrs['spec']
                if spec.effects:
                    raise LoweringError('effectful external has no implemented ASK execution contract')
                symbol = f'{function.name}_external_{external_count}'
                external_count += 1
                semantic = output.semantics(symbol + '_semantics',
                    inputs=[_ask_type(value.type) for value in op.operands],
                    results=[_ask_type(value.type) for value in op.results],
                    shape_rule='fixed' if spec.shape_rule == 'explicit' else spec.shape_rule,
                    shape_operand=spec.shape_operand, effects='none')
                implementation = output.implementation(symbol + '_implementation', compiler=spec.compiler,
                    source=spec.source, entry=spec.entry, signature=dict(spec.signature),
                    source_bindings=dict(spec.source_bindings), constants=dict(spec.constants),
                    options=dict(spec.options), target=_target_for(spec, target), grid=spec.grid)
                result = graph.external(*operands, semantics=semantic, implementation=implementation,
                                        result_types=[_ask_type(value.type) for value in op.results], **kwargs)
            else:
                raise LoweringError(f'no ASK lowering for fuga.{op.opcode}')
            for value, lowered in zip(op.results, result if isinstance(result, tuple) else (result,)):
                values[value] = lowered
                record(function, value, lowered)
    if source_map is not None:
        source_map.clear()
        source_map.update(mapping)
    return output
