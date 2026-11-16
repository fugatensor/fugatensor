"""Native C++ cuDNN 9 SDPA plan cache, executable, and composite pipeline compiler."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _native
from ._ask.runtime import DeviceTensor
from .ir import Module, TensorType
from .ir.transforms import _clone_operation

CudnnExecutable = getattr(_native, 'CudnnExecutable', None)
CompositeExecutable = getattr(_native, 'CudnnCompositeExecutable', None)


def has_cudnn() -> bool:
    fn = getattr(_native, 'cudnn_available', None)
    return bool(fn()) if fn is not None else False


def _port_def(index: int, kind: str, ttype: TensorType) -> dict[str, Any]:
    return {
        'logical_port': f'{kind}:{index}',
        'shape': list(ttype.shape),
        'element_type': str(ttype.dtype),
        'layout': 'contiguous_row_major',
    }


def _make_cudnn_executable(engine, cudnn_state, spec, output_types: list[TensorType],
                           directory: Path, op_id: str):
    constants = spec.constants or {}
    return _native.CudnnExecutable(
        cudnn_state,
        engine.session,
        DeviceTensor,
        spec.entry,
        list(output_types),
        bool(constants.get('causal', 0)),
        constants.get('scale', None),
        int(constants.get('max_s_q', 0) or 0),
        int(constants.get('max_s_kv', 0) or 0),
        str(constants.get('layout', '')),
        directory,
        op_id,
    )


def partition_and_compile_cudnn(engine, flat_module, target: dict[str, Any],
                                directory: Path) -> Any:
    """Partition flat Fuga module into owned ASK stages and native C++ cuDNN stages."""
    cudnn_state = engine._get_cudnn_state()
    func = flat_module.functions[0]
    ops = [op for op in func.operations if op.opcode != 'return']

    stages_raw = []
    current_owned = []
    cudnn_count = 0
    owned_count = 0

    for op in ops:
        spec = op.attrs.get('spec', None) if op.opcode == 'external' else None
        if spec is not None and spec.compiler == 'cudnn':
            if current_owned:
                stages_raw.append(('owned', f'sub_owned_{owned_count}', current_owned))
                owned_count += 1
                current_owned = []
            stages_raw.append(('cudnn', f'cudnn_{spec.entry}_{cudnn_count}', [op]))
            cudnn_count += 1
        else:
            current_owned.append(op)
    if current_owned:
        stages_raw.append(('owned', f'sub_owned_{owned_count}', current_owned))
        owned_count += 1

    return_op = next(op for op in func.operations if op.opcode == 'return')

    if len(stages_raw) == 1 and stages_raw[0][0] == 'cudnn':
        op = stages_raw[0][2][0]
        return _make_cudnn_executable(
            engine, cudnn_state, op.attrs['spec'],
            [res.type for res in op.results], directory, stages_raw[0][1])

    # Assign flat integer slot indices for zero-overhead C++ composite dispatch.
    slot_of: dict[Any, int] = {}
    slot_shapes: list[list[int]] = []
    slot_dtypes: list[str] = []

    def ensure_slot(val) -> int:
        idx = slot_of.get(val)
        if idx is None:
            idx = len(slot_shapes)
            slot_of[val] = idx
            slot_shapes.append(list(getattr(val.type, 'shape', ())))
            slot_dtypes.append(str(val.type.dtype))
        return idx

    initial_inputs = list(func.inputs.values())
    initial_input_slots = [ensure_slot(val) for val in initial_inputs]
    for op in ops:
        for val in op.operands:
            ensure_slot(val)
        for val in op.results:
            ensure_slot(val)

    all_final_vals = set(return_op.operands)
    final_output_slots = [ensure_slot(val) for val in return_op.operands]

    stage_defs = []
    plan_steps = []
    report_steps = {}

    from .lowering import lower_to_ask

    for i, (kind, stage_id, stage_ops) in enumerate(stages_raw):
        stage_produced = {val for op in stage_ops for val in op.results}
        stage_inputs = []
        for op in stage_ops:
            for val in op.operands:
                if val not in stage_produced and val not in stage_inputs:
                    stage_inputs.append(val)

        future_uses = all_final_vals.copy()
        for _, _, later_ops in stages_raw[i + 1:]:
            for op in later_ops:
                future_uses.update(op.operands)

        stage_outputs = []
        for op in stage_ops:
            for val in op.results:
                if val in future_uses and val not in stage_outputs:
                    stage_outputs.append(val)

        in_slots = [slot_of[val] for val in stage_inputs]
        out_slots = [slot_of[val] for val in stage_outputs]

        if kind == 'owned':
            sub_mod = Module()
            sub_inputs = {f'in_{idx}': val.type for idx, val in enumerate(stage_inputs)}
            sub_graph = sub_mod.graph(stage_id, inputs=sub_inputs,
                                      results=[val.type for val in stage_outputs])
            val_map = {orig_val: sub_graph.inputs[f'in_{idx}']
                       for idx, orig_val in enumerate(stage_inputs)}
            for op in stage_ops:
                cloned = _clone_operation(sub_graph, op, tuple(val_map[o] for o in op.operands))
                val_map.update(zip(op.results, cloned.results))
            sub_graph.return_(*(val_map[out_val] for out_val in stage_outputs))
            sub_mod.verify()

            sub_dir = directory / stage_id
            sub_dir.mkdir(parents=True, exist_ok=True)
            ask_sub = lower_to_ask(sub_mod, target=target)
            sub_source = ask_sub.to_mlir()
            sub_kernel = engine.compiler.compile_cuda(
                sub_source, compilers=engine.compilers, llc=engine.llc, ptxas=engine.ptxas,
                llvm_link=engine.llvm_link, libdevice=engine.libdevice,
                cache_dir=engine.cache_dir / 'ask', output_dir=sub_dir, target=target)
            prepared = sub_kernel.prepare(engine.session, graph=True)
            stage_defs.append({
                'id': stage_id,
                'kind': 'owned',
                'input_slots': in_slots,
                'output_slots': out_slots,
                'program': prepared._native,
            })
            plan_steps.append({'id': stage_id, 'kind': 'owned'})
            report_steps[stage_id] = sub_kernel.report
        else:
            op = stage_ops[0]
            spec = op.attrs['spec']
            cudnn_exe = _make_cudnn_executable(
                engine, cudnn_state, spec,
                [res.type for res in op.results], directory / stage_id, stage_id)
            stage_defs.append({
                'id': stage_id,
                'kind': 'external',
                'input_slots': in_slots,
                'output_slots': out_slots,
                'cudnn': cudnn_exe,
            })
            plan_steps.append({'id': stage_id, 'kind': 'external'})
            report_steps[stage_id] = cudnn_exe.report

    return _native.CudnnCompositeExecutable(
        cudnn_state,
        engine.session,
        DeviceTensor,
        len(slot_shapes),
        initial_input_slots,
        final_output_slots,
        [_port_def(i, 'input', val.type) for i, val in enumerate(initial_inputs)],
        [_port_def(i, 'output', val.type) for i, val in enumerate(return_op.operands)],
        slot_shapes,
        slot_dtypes,
        stage_defs,
        directory,
        {'steps': plan_steps},
        {'phase': 'composite_plan', 'steps': report_steps},
    )
