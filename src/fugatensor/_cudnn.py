"""cuDNN SDPA plan cache, executable and composite pipeline support."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
from typing import Any

from .ir import Module, TensorType
from .ir.transforms import _clone_operation


def _contiguous_stride(shape: tuple[int, ...] | list[int]) -> list[int]:
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return stride


class CudnnPlan:
    def __init__(self, handle, entry: str, input_shapes: tuple[tuple[int, ...], ...],
                 input_dtypes: tuple[str, ...], constants: dict[str, Any]):
        import cudnn
        self.entry = entry
        self.graph = cudnn.pygraph(handle=handle)
        self.tensors = {}
        self.ws_size = 0

        causal = bool(constants.get('causal', 0))
        scale = constants.get('scale', None)

        if entry in ('sdpa_fwd', 'sdpa_stats'):
            q_shape, k_shape, v_shape = input_shapes[:3]
            q_dtype = input_dtypes[0]
            cdtype = cudnn.data_type.HALF if q_dtype == 'f16' else cudnn.data_type.BFLOAT16
            b, h, m, d = q_shape
            dv = v_shape[-1]
            scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(d)

            q_t = self.graph.tensor(name='q', data_type=cdtype, dim=list(q_shape), stride=_contiguous_stride(q_shape))
            k_t = self.graph.tensor(name='k', data_type=cdtype, dim=list(k_shape), stride=_contiguous_stride(k_shape))
            v_t = self.graph.tensor(name='v', data_type=cdtype, dim=list(v_shape), stride=_contiguous_stride(v_shape))

            if entry == 'sdpa_fwd':
                res = self.graph.sdpa(name='sdpa_fwd', q=q_t, k=k_t, v=v_t, is_inference=True,
                                      use_causal_mask=causal, attn_scale=scale_val)
                o_t = res[0]
                o_t.set_output(True).set_data_type(cdtype)
                self.tensors = {'q': q_t, 'k': k_t, 'v': v_t, 'o': o_t}
            else:
                res = self.graph.sdpa(name='sdpa_stats', q=q_t, k=k_t, v=v_t, generate_stats=True,
                                      use_causal_mask=causal, attn_scale=scale_val)
                stats_t = res[1]
                stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
                self.tensors = {'q': q_t, 'k': k_t, 'v': v_t, 'stats': stats_t}

        elif entry == 'sdpa_bwd':
            q_shape, k_shape, v_shape, o_shape, do_shape, stats_shape = input_shapes[:6]
            q_dtype = input_dtypes[0]
            cdtype = cudnn.data_type.HALF if q_dtype == 'f16' else cudnn.data_type.BFLOAT16
            d = q_shape[-1]
            scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(d)

            q_t = self.graph.tensor(name='q', data_type=cdtype, dim=list(q_shape), stride=_contiguous_stride(q_shape))
            k_t = self.graph.tensor(name='k', data_type=cdtype, dim=list(k_shape), stride=_contiguous_stride(k_shape))
            v_t = self.graph.tensor(name='v', data_type=cdtype, dim=list(v_shape), stride=_contiguous_stride(v_shape))
            o_t = self.graph.tensor(name='o', data_type=cdtype, dim=list(o_shape), stride=_contiguous_stride(o_shape))
            do_t = self.graph.tensor(name='do', data_type=cdtype, dim=list(do_shape), stride=_contiguous_stride(do_shape))
            stats_t = self.graph.tensor(name='stats', data_type=cudnn.data_type.FLOAT, dim=list(stats_shape),
                                        stride=_contiguous_stride(stats_shape))

            dq_t, dk_t, dv_t = self.graph.sdpa_backward(
                name='sdpa_bwd', q=q_t, k=k_t, v=v_t, o=o_t, dO=do_t, stats=stats_t,
                use_causal_mask=causal, attn_scale=scale_val)
            dq_t.set_output(True).set_data_type(cdtype)
            dk_t.set_output(True).set_data_type(cdtype)
            dv_t.set_output(True).set_data_type(cdtype)
            self.tensors = {'q': q_t, 'k': k_t, 'v': v_t, 'o': o_t, 'do': do_t, 'stats': stats_t,
                            'dq': dq_t, 'dk': dk_t, 'dv': dv_t}

        elif entry in ('sdpa_varlen_fwd', 'sdpa_varlen_stats'):
            q_shape, k_shape, v_shape = input_shapes[:3]
            cu_q_shape = input_shapes[3]
            b = cu_q_shape[0] - 1
            h = q_shape[1]
            d = q_shape[2]
            dv = v_shape[2]
            max_s_q = constants.get('max_s_q', 128)
            max_s_kv = constants.get('max_s_kv', 128)
            scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(d)
            cdtype = cudnn.data_type.HALF if input_dtypes[0] == 'f16' else cudnn.data_type.BFLOAT16

            rq = self.graph.tensor(name='rq', data_type=cudnn.data_type.INT64, dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1])
            rkv = self.graph.tensor(name='rkv', data_type=cudnn.data_type.INT64, dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1])
            slq = self.graph.tensor(name='slq', data_type=cudnn.data_type.INT32, dim=[b, 1, 1, 1], stride=[1, 1, 1, 1])
            slkv = self.graph.tensor(name='slkv', data_type=cudnn.data_type.INT32, dim=[b, 1, 1, 1], stride=[1, 1, 1, 1])

            thd_q = [max_s_q * h * d, d, h * d, 1]
            thd_k = [max_s_kv * h * d, d, h * d, 1]
            thd_v = [max_s_kv * h * dv, dv, h * dv, 1]
            thd_o = [max_s_q * h * dv, dv, h * dv, 1]

            q_t = self.graph.tensor(name='q', data_type=cdtype, dim=[b, h, max_s_q, d], stride=thd_q)
            q_t.set_ragged_offset(rq)
            k_t = self.graph.tensor(name='k', data_type=cdtype, dim=[b, h, max_s_kv, d], stride=thd_k)
            k_t.set_ragged_offset(rkv)
            v_t = self.graph.tensor(name='v', data_type=cdtype, dim=[b, h, max_s_kv, dv], stride=thd_v)
            v_t.set_ragged_offset(rkv)

            if entry == 'sdpa_varlen_fwd':
                res = self.graph.sdpa(name='sdpa_varlen_fwd', q=q_t, k=k_t, v=v_t,
                                      seq_len_q=slq, seq_len_kv=slkv, use_padding_mask=True,
                                      is_inference=True, use_causal_mask=causal, attn_scale=scale_val)
                o_t = res[0]
                o_t.set_output(True).set_data_type(cdtype).set_stride(thd_o).set_ragged_offset(rq)
                self.tensors = {'q': q_t, 'k': k_t, 'v': v_t, 'rq': rq, 'rkv': rkv,
                                'slq': slq, 'slkv': slkv, 'o': o_t}
            else:
                res = self.graph.sdpa(name='sdpa_varlen_stats', q=q_t, k=k_t, v=v_t,
                                      seq_len_q=slq, seq_len_kv=slkv, use_padding_mask=True,
                                      generate_stats=True, use_causal_mask=causal, attn_scale=scale_val)
                stats_t = res[1]
                stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
                self.tensors = {'q': q_t, 'k': k_t, 'v': v_t, 'rq': rq, 'rkv': rkv,
                                'slq': slq, 'slkv': slkv, 'stats': stats_t}

        elif entry == 'sdpa_varlen_bwd':
            q_shape, k_shape, v_shape, o_shape, do_shape, stats_shape, cu_q_shape, cu_kv_shape = input_shapes[:8]
            b = cu_q_shape[0] - 1
            h = q_shape[1]
            d = q_shape[2]
            dv = v_shape[2]
            max_s_q = constants.get('max_s_q', 128)
            max_s_kv = constants.get('max_s_kv', 128)
            scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(d)
            cdtype = cudnn.data_type.HALF if input_dtypes[0] == 'f16' else cudnn.data_type.BFLOAT16

            rq = self.graph.tensor(name='rq', data_type=cudnn.data_type.INT64, dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1])
            rkv = self.graph.tensor(name='rkv', data_type=cudnn.data_type.INT64, dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1])
            slq = self.graph.tensor(name='slq', data_type=cudnn.data_type.INT32, dim=[b, 1, 1, 1], stride=[1, 1, 1, 1])
            slkv = self.graph.tensor(name='slkv', data_type=cudnn.data_type.INT32, dim=[b, 1, 1, 1], stride=[1, 1, 1, 1])

            thd_q = [max_s_q * h * d, d, h * d, 1]
            thd_k = [max_s_kv * h * d, d, h * d, 1]
            thd_v = [max_s_kv * h * dv, dv, h * dv, 1]
            thd_o = [max_s_q * h * dv, dv, h * dv, 1]

            q_t = self.graph.tensor(name='q', data_type=cdtype, dim=[b, h, max_s_q, d], stride=thd_q)
            q_t.set_ragged_offset(rq)
            k_t = self.graph.tensor(name='k', data_type=cdtype, dim=[b, h, max_s_kv, d], stride=thd_k)
            k_t.set_ragged_offset(rkv)
            v_t = self.graph.tensor(name='v', data_type=cdtype, dim=[b, h, max_s_kv, dv], stride=thd_v)
            v_t.set_ragged_offset(rkv)
            o_t = self.graph.tensor(name='o', data_type=cdtype, dim=[b, h, max_s_q, dv], stride=thd_o)
            o_t.set_ragged_offset(rq)
            do_t = self.graph.tensor(name='do', data_type=cdtype, dim=[b, h, max_s_q, dv], stride=thd_o)
            do_t.set_ragged_offset(rq)
            stats_t = self.graph.tensor(name='stats', data_type=cudnn.data_type.FLOAT, dim=list(stats_shape),
                                        stride=_contiguous_stride(stats_shape))

            dq_t, dk_t, dv_t = self.graph.sdpa_backward(
                name='sdpa_varlen_bwd', q=q_t, k=k_t, v=v_t, o=o_t, dO=do_t, stats=stats_t,
                seq_len_q=slq, seq_len_kv=slkv, use_padding_mask=True,
                use_causal_mask=causal, attn_scale=scale_val)
            dq_t.set_output(True).set_data_type(cdtype).set_stride(thd_q).set_ragged_offset(rq)
            dk_t.set_output(True).set_data_type(cdtype).set_stride(thd_k).set_ragged_offset(rkv)
            dv_t.set_output(True).set_data_type(cdtype).set_stride(thd_v).set_ragged_offset(rkv)

            self.tensors = {'q': q_t, 'k': k_t, 'v': v_t, 'o': o_t, 'do': do_t, 'stats': stats_t,
                            'rq': rq, 'rkv': rkv, 'slq': slq, 'slkv': slkv,
                            'dq': dq_t, 'dk': dk_t, 'dv': dv_t}
        else:
            raise ValueError(f'unknown cuDNN entry: {entry}')

        self.graph.validate()
        self.graph.build_operation_graph()
        self.graph.create_execution_plans([cudnn.heur_mode.A])
        self.graph.check_support()
        self.graph.build_plans()
        self.ws_size = self.graph.get_workspace_size()


def get_cudnn_plan(handle, cache: dict, entry: str, input_shapes: tuple[tuple[int, ...], ...],
                   input_dtypes: tuple[str, ...], constants: dict[str, Any]) -> CudnnPlan:
    causal = int(constants.get('causal', 0))
    scale = constants.get('scale', None)
    max_s_q = constants.get('max_s_q', 0)
    max_s_kv = constants.get('max_s_kv', 0)
    key = (entry, input_shapes, input_dtypes, causal, scale, max_s_q, max_s_kv)
    plan = cache.get(key)
    if plan is None:
        plan = CudnnPlan(handle, entry, input_shapes, input_dtypes, constants)
        cache[key] = plan
    return plan


class CudnnExecutable:
    def __init__(self, engine, entry: str, output_types: list[TensorType],
                 constants: dict[str, Any], directory: Path, op_id: str):
        self.engine = engine
        self.entry = entry
        self.output_types = output_types
        self.constants = dict(constants)
        self.artifact_dir = directory
        self.id = op_id
        self.kind = 'external'
        self.execution_plan = {'steps': [{'id': op_id, 'kind': 'external'}]}
        self.report = {'compiler': 'cudnn', 'entry': entry}
        self.source = f'cudnn::{entry}'

    def run(self, inputs: list[Any]) -> list[Any]:
        session = self.engine.session
        handle = self.engine._get_cudnn_handle()
        input_shapes = tuple(tuple(item.shape) for item in inputs)
        input_dtypes = tuple(str(item.dtype) for item in inputs)

        # For varlen, compute max_s_q and max_s_kv dynamically if not fixed
        constants = dict(self.constants)
        if 'varlen' in self.entry:
            cu_q = inputs[3].to_host() if len(inputs) > 3 else None
            cu_kv = inputs[4].to_host() if len(inputs) > 4 else None
            if cu_q is not None:
                b = len(cu_q) - 1
                seq_lens_q = [cu_q[i+1] - cu_q[i] for i in range(b)]
                if not constants.get('max_s_q'):
                    constants['max_s_q'] = max(seq_lens_q) if seq_lens_q else 1
            if cu_kv is not None:
                b_kv = len(cu_kv) - 1
                seq_lens_kv = [cu_kv[i+1] - cu_kv[i] for i in range(b_kv)]
                if not constants.get('max_s_kv'):
                    constants['max_s_kv'] = max(seq_lens_kv) if seq_lens_kv else 1

        plan = get_cudnn_plan(handle, self.engine._cudnn_plan_cache, self.entry,
                              input_shapes, input_dtypes, constants)

        # Allocate or reuse workspace if needed
        ws_ptr = 0
        if plan.ws_size > 0:
            if getattr(plan, 'ws_tensor', None) is None:
                plan.ws_tensor = session.tensor((plan.ws_size,), 'bool')
            ws_ptr = plan.ws_tensor.data_ptr

        variant_pack = {}
        outputs = []

        if self.entry == 'sdpa_fwd':
            q, k, v = inputs[:3]
            out_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
            out = session.tensor(out_shape, q.dtype)
            variant_pack = {
                plan.tensors['q']: q.data_ptr,
                plan.tensors['k']: k.data_ptr,
                plan.tensors['v']: v.data_ptr,
                plan.tensors['o']: out.data_ptr,
            }
            outputs = [out]

        elif self.entry == 'sdpa_stats':
            q, k, v = inputs[:3]
            stats_shape = (q.shape[0], q.shape[1], q.shape[2], 1)
            stats = session.tensor(stats_shape, 'f32')
            variant_pack = {
                plan.tensors['q']: q.data_ptr,
                plan.tensors['k']: k.data_ptr,
                plan.tensors['v']: v.data_ptr,
                plan.tensors['stats']: stats.data_ptr,
            }
            outputs = [stats]

        elif self.entry == 'sdpa_bwd':
            q, k, v, o, do, stats = inputs[:6]
            dq = session.tensor(q.shape, q.dtype)
            dk = session.tensor(k.shape, k.dtype)
            dv = session.tensor(v.shape, v.dtype)
            variant_pack = {
                plan.tensors['q']: q.data_ptr,
                plan.tensors['k']: k.data_ptr,
                plan.tensors['v']: v.data_ptr,
                plan.tensors['o']: o.data_ptr,
                plan.tensors['do']: do.data_ptr,
                plan.tensors['stats']: stats.data_ptr,
                plan.tensors['dq']: dq.data_ptr,
                plan.tensors['dk']: dk.data_ptr,
                plan.tensors['dv']: dv.data_ptr,
            }
            outputs = [dq, dk, dv]

        elif self.entry == 'sdpa_varlen_fwd':
            q, k, v, cu_q, cu_kv = inputs[:5]
            cu_q_host = cu_q.to_host()
            cu_kv_host = cu_kv.to_host()
            b = len(cu_q_host) - 1
            h, d, dv = q.shape[1], q.shape[2], v.shape[2]
            ro_q = [val * (h * d) for val in cu_q_host]
            ro_kv = [val * (h * d) for val in cu_kv_host]
            sl_q = [cu_q_host[i+1] - cu_q_host[i] for i in range(b)]
            sl_kv = [cu_kv_host[i+1] - cu_kv_host[i] for i in range(b)]

            ro_q_dev = session.from_host(ro_q, shape=(b + 1, 1, 1, 1), dtype='i64')
            ro_kv_dev = session.from_host(ro_kv, shape=(b + 1, 1, 1, 1), dtype='i64')
            sl_q_dev = session.from_host(sl_q, shape=(b, 1, 1, 1), dtype='i32')
            sl_kv_dev = session.from_host(sl_kv, shape=(b, 1, 1, 1), dtype='i32')

            out_shape = (q.shape[0], h, dv)
            out = session.tensor(out_shape, q.dtype)
            variant_pack = {
                plan.tensors['q']: q.data_ptr,
                plan.tensors['k']: k.data_ptr,
                plan.tensors['v']: v.data_ptr,
                plan.tensors['rq']: ro_q_dev.data_ptr,
                plan.tensors['rkv']: ro_kv_dev.data_ptr,
                plan.tensors['slq']: sl_q_dev.data_ptr,
                plan.tensors['slkv']: sl_kv_dev.data_ptr,
                plan.tensors['o']: out.data_ptr,
            }
            outputs = [out]

        elif self.entry == 'sdpa_varlen_stats':
            q, k, v, cu_q, cu_kv = inputs[:5]
            cu_q_host = cu_q.to_host()
            cu_kv_host = cu_kv.to_host()
            b = len(cu_q_host) - 1
            h, d, dv = q.shape[1], q.shape[2], v.shape[2]
            ro_q = [val * (h * d) for val in cu_q_host]
            ro_kv = [val * (h * d) for val in cu_kv_host]
            sl_q = [cu_q_host[i+1] - cu_q_host[i] for i in range(b)]
            sl_kv = [cu_kv_host[i+1] - cu_kv_host[i] for i in range(b)]

            ro_q_dev = session.from_host(ro_q, shape=(b + 1, 1, 1, 1), dtype='i64')
            ro_kv_dev = session.from_host(ro_kv, shape=(b + 1, 1, 1, 1), dtype='i64')
            sl_q_dev = session.from_host(sl_q, shape=(b, 1, 1, 1), dtype='i32')
            sl_kv_dev = session.from_host(sl_kv, shape=(b, 1, 1, 1), dtype='i32')

            max_s_q = constants.get('max_s_q', 128)
            stats_shape = (b, h, max_s_q, 1)
            stats = session.tensor(stats_shape, 'f32')
            variant_pack = {
                plan.tensors['q']: q.data_ptr,
                plan.tensors['k']: k.data_ptr,
                plan.tensors['v']: v.data_ptr,
                plan.tensors['rq']: ro_q_dev.data_ptr,
                plan.tensors['rkv']: ro_kv_dev.data_ptr,
                plan.tensors['slq']: sl_q_dev.data_ptr,
                plan.tensors['slkv']: sl_kv_dev.data_ptr,
                plan.tensors['stats']: stats.data_ptr,
            }
            outputs = [stats]

        elif self.entry == 'sdpa_varlen_bwd':
            q, k, v, o, do, stats, cu_q, cu_kv = inputs[:8]
            cu_q_host = cu_q.to_host()
            cu_kv_host = cu_kv.to_host()
            b = len(cu_q_host) - 1
            h, d, dv = q.shape[1], q.shape[2], v.shape[2]
            ro_q = [val * (h * d) for val in cu_q_host]
            ro_kv = [val * (h * d) for val in cu_kv_host]
            sl_q = [cu_q_host[i+1] - cu_q_host[i] for i in range(b)]
            sl_kv = [cu_kv_host[i+1] - cu_kv_host[i] for i in range(b)]

            ro_q_dev = session.from_host(ro_q, shape=(b + 1, 1, 1, 1), dtype='i64')
            ro_kv_dev = session.from_host(ro_kv, shape=(b + 1, 1, 1, 1), dtype='i64')
            sl_q_dev = session.from_host(sl_q, shape=(b, 1, 1, 1), dtype='i32')
            sl_kv_dev = session.from_host(sl_kv, shape=(b, 1, 1, 1), dtype='i32')

            dq = session.tensor(q.shape, q.dtype)
            dk = session.tensor(k.shape, k.dtype)
            dv = session.tensor(v.shape, v.dtype)
            variant_pack = {
                plan.tensors['q']: q.data_ptr,
                plan.tensors['k']: k.data_ptr,
                plan.tensors['v']: v.data_ptr,
                plan.tensors['o']: o.data_ptr,
                plan.tensors['do']: do.data_ptr,
                plan.tensors['stats']: stats.data_ptr,
                plan.tensors['rq']: ro_q_dev.data_ptr,
                plan.tensors['rkv']: ro_kv_dev.data_ptr,
                plan.tensors['slq']: sl_q_dev.data_ptr,
                plan.tensors['slkv']: sl_kv_dev.data_ptr,
                plan.tensors['dq']: dq.data_ptr,
                plan.tensors['dk']: dk.data_ptr,
                plan.tensors['dv']: dv.data_ptr,
            }
            outputs = [dq, dk, dv]

        plan.graph.execute(variant_pack, ws_ptr)
        return outputs


class CompositeExecutable:
    def __init__(self, stages: list[dict[str, Any]], initial_inputs: list[str],
                 final_outputs: list[str], directory: Path):
        self.stages = stages
        self.initial_inputs = list(initial_inputs)
        self.final_outputs = list(final_outputs)
        self.artifact_dir = directory
        self.execution_plan = {
            'steps': [{'id': stage['id'], 'kind': stage['kind']} for stage in stages]
        }
        self.report = {
            'phase': 'composite_plan',
            'steps': {stage['id']: getattr(stage['executable'], 'report', {}) for stage in stages}
        }
        self.source = 'composite'

    def run(self, inputs: list[Any]) -> list[Any]:
        live = {}
        for name, tensor in zip(self.initial_inputs, inputs):
            live[name] = tensor

        for stage in self.stages:
            stage_inputs = [live[name] for name in stage['inputs']]
            stage_outputs = stage['executable'].run(stage_inputs)
            if not isinstance(stage_outputs, (list, tuple)):
                stage_outputs = [stage_outputs]
            for name, tensor in zip(stage['outputs'], stage_outputs):
                live[name] = tensor

        return [live[name] for name in self.final_outputs]


def partition_and_compile_cudnn(engine, flat_module, target: dict[str, Any],
                                directory: Path) -> Any:
    """Partition flat Fuga module into owned submodules and cuDNN executables."""
    func = flat_module.functions[0]
    ops = [op for op in func.operations if op.opcode != 'return']

    stages_raw = []
    current_owned = []
    cudnn_count = 0
    owned_count = 0

    for op in ops:
        if op.opcode == 'external' and op.attrs.get('spec', None) and op.attrs['spec'].compiler == 'cudnn':
            if current_owned:
                stages_raw.append(('owned', f'sub_owned_{owned_count}', current_owned))
                owned_count += 1
                current_owned = []
            stages_raw.append(('cudnn', f'cudnn_{op.attrs["spec"].entry}_{cudnn_count}', [op]))
            cudnn_count += 1
        else:
            current_owned.append(op)
    if current_owned:
        stages_raw.append(('owned', f'sub_owned_{owned_count}', current_owned))
        owned_count += 1

    return_op = next(op for op in func.operations if op.opcode == 'return')
    final_outputs = [val.name for val in return_op.operands]

    # If the whole function is ONLY 1 cuDNN operation and nothing else:
    if len(stages_raw) == 1 and stages_raw[0][0] == 'cudnn':
        op = stages_raw[0][2][0]
        spec = op.attrs['spec']
        return CudnnExecutable(engine, spec.entry, [res.type for res in op.results],
                               spec.constants, directory, stages_raw[0][1])

    compiled_stages = []
    all_final_vals = set(return_op.operands)

    for i, (kind, stage_id, stage_ops) in enumerate(stages_raw):
        stage_produced = {val for op in stage_ops for val in op.results}

        # Input values needed from earlier:
        stage_inputs = []
        for op in stage_ops:
            for val in op.operands:
                if val not in stage_produced and val not in stage_inputs:
                    stage_inputs.append(val)

        # Output values needed later:
        future_uses = all_final_vals.copy()
        for _, _, later_ops in stages_raw[i+1:]:
            for op in later_ops:
                future_uses.update(op.operands)

        stage_outputs = []
        for op in stage_ops:
            for val in op.results:
                if val in future_uses and val not in stage_outputs:
                    stage_outputs.append(val)

        if kind == 'owned':
            sub_mod = Module()
            sub_inputs = {f'in_{idx}': val.type for idx, val in enumerate(stage_inputs)}
            sub_graph = sub_mod.graph(stage_id, inputs=sub_inputs, results=[val.type for val in stage_outputs])
            val_map = {orig_val: sub_graph.inputs[f'in_{idx}'] for idx, orig_val in enumerate(stage_inputs)}
            for op in stage_ops:
                cloned = _clone_operation(sub_graph, op, tuple(val_map[operand] for operand in op.operands))
                val_map.update(zip(op.results, cloned.results))
            sub_graph.return_(*(val_map[out_val] for out_val in stage_outputs))
            sub_mod.verify()

            # Compile with normal ASK compiler
            sub_dir = directory / stage_id
            sub_dir.mkdir(parents=True, exist_ok=True)
            from .lowering import lower_to_ask
            from .engine import _CUDAExecutable
            ask_sub = lower_to_ask(sub_mod, target=target)
            sub_source = ask_sub.to_mlir()
            sub_kernel = engine.compiler.compile_cuda(
                sub_source, compilers=engine.compilers, llc=engine.llc, ptxas=engine.ptxas,
                llvm_link=engine.llvm_link, libdevice=engine.libdevice,
                cache_dir=engine.cache_dir / 'ask', output_dir=sub_dir, target=target)
            executable = _CUDAExecutable(sub_kernel, engine.session, sub_dir, sub_source)
            compiled_stages.append({
                'id': stage_id,
                'kind': 'owned',
                'executable': executable,
                'inputs': [val.name for val in stage_inputs],
                'outputs': [val.name for val in stage_outputs],
            })
        else:
            op = stage_ops[0]
            spec = op.attrs['spec']
            executable = CudnnExecutable(engine, spec.entry, [res.type for res in op.results],
                                         spec.constants, directory / stage_id, stage_id)
            compiled_stages.append({
                'id': stage_id,
                'kind': 'external',
                'executable': executable,
                'inputs': [val.name for val in stage_inputs],
                'outputs': [val.name for val in stage_outputs],
            })

    initial_input_names = [val.name for val in func.inputs.values()]
    return CompositeExecutable(compiled_stages, initial_input_names, final_outputs, directory)
