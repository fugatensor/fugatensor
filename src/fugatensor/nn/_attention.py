"""cuDNN 9 SDPA implementation for standard and variable-length sequences."""
from __future__ import annotations

from functools import lru_cache
import inspect
import math
from typing import Mapping

from .._jit import jit
from ..dtypes import fp16, bf16, fp32, int32
from ..function import fun
from ..ir import ExternalSpec, TensorType
from ..tensor import Tensor, normalize_dtype, normalize_shape
from ..triton import TensorSpec
from ..tracing import TraceTensor


def _dimensions(specs):
    q, k, v = specs[:3]
    if any(len(spec.shape) != 4 for spec in (q, k, v)):
        raise ValueError('attention inputs require [batch, head, sequence, feature]')
    b, h, m, d = q.shape
    bk, hk, n, dk = k.shape
    bv, hv, nv, dv = v.shape
    if (b, h, d) != (bk, hk, dk) or (b, h, n) != (bv, hv, nv):
        raise ValueError('attention batch, head, key and value dimensions disagree')
    if min(b, h, m, n, d, dv) <= 0:
        raise ValueError('attention dimensions must be positive')
    if d not in (32, 64, 128) or dv not in (32, 64, 128):
        raise ValueError('supported attention feature sizes are 32, 64 or 128')
    if q.dtype not in (fp16, bf16) or any(spec.dtype != q.dtype for spec in (k, v)):
        raise TypeError('attention requires matching f16 or bf16 inputs')
    return b, h, m, n, d, dv


def _varlen_dimensions(specs):
    q, k, v, cu_q, cu_kv = specs[:5]
    if any(len(spec.shape) not in (3, 4) for spec in (q, k, v)):
        raise ValueError('varlen attention inputs require [total_tokens, head, feature] or [batch, head, seq, feature]')
    if len(q.shape) == 3:
        t_q, h, d = q.shape
        t_k, hk, dk = k.shape
        t_v, hv, dv = v.shape
        if (h, d) != (hk, dk) or (h, t_k) != (hv, t_v):
            raise ValueError('varlen attention head and key/value dimensions disagree')
        if min(t_q, t_k, t_v, h, d, dv) <= 0:
            raise ValueError('attention dimensions must be positive')
    else:
        b, h, m, d = q.shape
        bk, hk, n, dk = k.shape
        bv, hv, nv, dv = v.shape
        if (b, h, d) != (bk, hk, dk) or (b, h, n) != (bv, hv, nv):
            raise ValueError('attention batch, head, key and value dimensions disagree')
        if min(b, h, m, n, d, dv) <= 0:
            raise ValueError('attention dimensions must be positive')
    if q.dtype not in (fp16, bf16) or any(spec.dtype != q.dtype for spec in (k, v)):
        raise TypeError('attention requires matching f16 or bf16 inputs')
    if cu_q.dtype != int32 or cu_kv.dtype != int32:
        raise TypeError('cu_seqlens must have int32 dtype')


class _CudnnKernel:
    def __init__(self, entry, arity, output_fn, dtype_fn, constants_fn):
        self.entry = entry
        self.arity = arity
        self.output_fn = output_fn
        self.dtype_fn = dtype_fn
        self.constants_fn = constants_fn

        def invoke(*values):
            return self._trace(*values)
        invoke.__signature__ = inspect.Signature([
            inspect.Parameter(f'value{i}', inspect.Parameter.POSITIONAL_ONLY)
            for i in range(arity)])
        self._record = jit(invoke, cache=False)

    def __call__(self, *values):
        if len(values) != self.arity:
            raise TypeError(f'{self.entry} expects {self.arity} tensors')
        if all(isinstance(v, Tensor) for v in values):
            return self._record(*values)
        return self._trace(*values)

    def _trace(self, *values):
        if any(not isinstance(v, TraceTensor) for v in values):
            raise TypeError('expected FugaTensor tensors from the current trace')
        context = values[0].context
        if any(v.context is not context for v in values):
            raise ValueError('kernel operands must belong to the same trace')
        specs = tuple(TensorSpec(tuple(v.shape), v.dtype, v.device) for v in values)
        constants = self.constants_fn(specs) if callable(self.constants_fn) else dict(self.constants_fn)
        result_shapes = self.output_fn(specs)
        result_dtypes = self.dtype_fn(specs)
        if not isinstance(result_shapes, (list, tuple)) or (result_shapes and isinstance(result_shapes[0], int)):
            result_shapes = [result_shapes]
            result_dtypes = [result_dtypes]
            single = True
        else:
            single = False
        result_types = [TensorType(tuple(s), d) for s, d in zip(result_shapes, result_dtypes)]

        sig = {f'in_{i}': f'*{v.dtype}' for i, v in enumerate(values)}
        sig.update({f'out_{i}': f'*{t.dtype}' for i, t in enumerate(result_types)})
        bindings = {f'in_{i}': f'input:{i}' for i in range(len(values))}
        bindings.update({f'out_{i}': f'output:{i}' for i in range(len(result_types))})

        spec = ExternalSpec(
            compiler='cudnn',
            source='cudnn_sdpa',
            entry=self.entry,
            signature=sig,
            source_bindings=bindings,
            constants=constants,
            target={'backend': 'cuda', 'arch': 80},
            grid=(1, 1, 1),
            shape_rule='explicit',
            effects=()
        )
        count = getattr(context, 'external_count', 0)
        symbol = f'fuga_cudnn_{self.entry}_{count}'
        context.external_count = count + 1
        result = context.graph.external(*(v.value for v in values), spec=spec,
                                        result_types=result_types, id=symbol)
        wrapped = tuple(context.wrap(v) for v in (result if isinstance(result, tuple) else (result,)))
        return wrapped[0] if single else wrapped


def _make_kernels(causal, scale):
    def constants(specs):
        b, h, m, n, d, dv = _dimensions(specs)
        s = scale if scale is not None else 1.0 / math.sqrt(d)
        return {'causal': int(causal), 'scale': s, 'b': b, 'h': h, 'm': m, 'n': n, 'd': d, 'dv': dv}

    def forward_shape(specs):
        b, h, m, _, _, dv = _dimensions(specs)
        return (b, h, m, dv)

    def stats_shape(specs):
        b, h, m, _, _, _ = _dimensions(specs)
        return (b, h, m, 1)

    def bwd_shapes(specs):
        q, k, v = specs[:3]
        return (tuple(q.shape), tuple(k.shape), tuple(v.shape))

    def bwd_dtypes(specs):
        q, k, v = specs[:3]
        return (q.dtype, k.dtype, v.dtype)

    def bwd_constants(specs):
        q, k, v = specs[:3]
        b, h, m, n, d, dv = _dimensions((q, k, v))
        s = scale if scale is not None else 1.0 / math.sqrt(d)
        return {'causal': int(causal), 'scale': s, 'b': b, 'h': h, 'm': m, 'n': n, 'd': d, 'dv': dv}

    fwd = _CudnnKernel('sdpa_fwd', 3, forward_shape, lambda s: s[0].dtype, constants)
    stats = _CudnnKernel('sdpa_stats', 3, stats_shape, lambda _: fp32, constants)
    bwd = _CudnnKernel('sdpa_bwd', 6, bwd_shapes, bwd_dtypes, bwd_constants)
    return fwd, stats, bwd


def _make_varlen_kernels(causal, scale, max_s_q, max_s_kv):
    def constants(specs):
        q, k, v = specs[:3]
        d = q.shape[-1]
        s = scale if scale is not None else 1.0 / math.sqrt(d)
        ms_q = max_s_q if max_s_q is not None else (q.shape[2] if len(q.shape) == 4 else 0)
        ms_kv = max_s_kv if max_s_kv is not None else (k.shape[2] if len(k.shape) == 4 else 0)
        return {'causal': int(causal), 'scale': s, 'max_s_q': ms_q, 'max_s_kv': ms_kv,
                'layout': 'thd' if len(q.shape) == 3 else 'padded'}

    def fwd_shape(specs):
        q, _, v = specs[:3]
        if len(q.shape) == 3:
            return (q.shape[0], q.shape[1], v.shape[2])
        return (q.shape[0], q.shape[1], q.shape[2], v.shape[3])

    def stats_shape(specs):
        q = specs[0]
        cu_q = specs[3]
        b = cu_q.shape[0] - 1 if len(cu_q.shape) == 1 else cu_q.shape[0]
        h = q.shape[1]
        ms_q = max_s_q if max_s_q is not None else (q.shape[2] if len(q.shape) == 4 else 0)
        return (b, h, ms_q, 1)

    def bwd_shapes(specs):
        q, k, v = specs[:3]
        return (tuple(q.shape), tuple(k.shape), tuple(v.shape))

    def bwd_dtypes(specs):
        q, k, v = specs[:3]
        return (q.dtype, k.dtype, v.dtype)

    fwd = _CudnnKernel('sdpa_varlen_fwd', 5, fwd_shape, lambda s: s[0].dtype, constants)
    stats = _CudnnKernel('sdpa_varlen_stats', 5, stats_shape, lambda _: fp32, constants)
    bwd = _CudnnKernel('sdpa_varlen_bwd', 8, bwd_shapes, bwd_dtypes, constants)
    return fwd, stats, bwd


class _FlashAttention(fun):
    """cuDNN 9 SDPA forward and backward."""
    def __init__(self, *, causal=False, scale=None):
        if type(causal) is not bool:
            raise TypeError('causal must be a bool')
        if scale is not None and not isinstance(scale, (int, float)):
            raise TypeError('scale must be a float or None')
        self.causal = causal
        self.scale = float(scale) if scale is not None else None
        (self._fuga_forward, self._fuga_stats, self._fuga_bwd) = _make_kernels(causal, self.scale)

    def __call__(self, *args, **kwargs):
        from .._state import active_trace
        if active_trace.get() is not None:
            return self._trace(*args, **kwargs)
        compiled = getattr(self, "_fuga_compiled", None)
        if compiled is None:
            def invoke(*args, **kwargs):
                return self._trace(*args, **kwargs)
            invoke.__signature__ = self.__signature__
            invoke.__name__ = type(self).__name__
            compiled = jit(invoke, cache=True)
            object.__setattr__(self, "_fuga_compiled", compiled)
        return compiled(*args, **kwargs)

    def forward(self, q, k, v):
        return self._fuga_forward(q, k, v)

    def vjp(self, primals, outputs, cotangents):
        q, k, v = primals
        output, = outputs
        do, = cotangents
        stats = self._fuga_stats(q, k, v)
        return self._fuga_bwd(q, k, v, output, do, stats)


class _FlashAttentionVarlen(fun):
    """cuDNN 9 variable-length SDPA forward and backward."""
    def __init__(self, *, causal=False, scale=None, max_seqlen_q=None, max_seqlen_kv=None):
        if type(causal) is not bool:
            raise TypeError('causal must be a bool')
        if scale is not None and not isinstance(scale, (int, float)):
            raise TypeError('scale must be a float or None')
        self.causal = causal
        self.scale = float(scale) if scale is not None else None
        self.max_seqlen_q = int(max_seqlen_q) if max_seqlen_q is not None else None
        self.max_seqlen_kv = int(max_seqlen_kv) if max_seqlen_kv is not None else None
        (self._fuga_forward, self._fuga_stats, self._fuga_bwd) = _make_varlen_kernels(
            causal, self.scale, self.max_seqlen_q, self.max_seqlen_kv)

    def __call__(self, *args, **kwargs):
        from .._state import active_trace
        if active_trace.get() is not None:
            return self._trace(*args, **kwargs)
        compiled = getattr(self, "_fuga_compiled", None)
        if compiled is None:
            def invoke(*args, **kwargs):
                return self._trace(*args, **kwargs)
            invoke.__signature__ = self.__signature__
            invoke.__name__ = type(self).__name__
            compiled = jit(invoke, cache=True)
            object.__setattr__(self, "_fuga_compiled", compiled)
        return compiled(*args, **kwargs)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_kv):
        return self._fuga_forward(q, k, v, cu_seqlens_q, cu_seqlens_kv)

    def vjp(self, primals, outputs, cotangents):
        q, k, v, cu_q, cu_kv = primals
        output, = outputs
        do, = cotangents
        stats = self._fuga_stats(q, k, v, cu_q, cu_kv)
        dq, dk, dv = self._fuga_bwd(q, k, v, output, do, stats, cu_q, cu_kv)
        return (dq, dk, dv)


@lru_cache(maxsize=16)
def attention(causal, scale=None):
    return _FlashAttention(causal=causal, scale=scale)


@lru_cache(maxsize=16)
def attention_varlen(causal, scale=None, max_seqlen_q=None, max_seqlen_kv=None):
    return _FlashAttentionVarlen(causal=causal, scale=scale,
                                 max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
