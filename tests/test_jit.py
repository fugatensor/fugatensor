"""Default lazy tensor graphs, mandatory ASK execution, caching, and Engine lifecycle."""
from __future__ import annotations

import pytest

import gc
from array import array
from itertools import product
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import weakref

from fugatensor._ask import CompilerError
from fugatensor._ask.cuda import CUDAError
import fugatensor as ft
from fugatensor.backends.host import HostStorage
from tests._helpers import requires_cuda


class TestHostStorage:
    def test_contiguous_buffers_preserve_bits_and_ownership(self):
        bit_patterns = {
            'bool': b'\x00\x01',
            'i32': b'\x00\x00\x00\x80\xff\xff\xff\x7f',
            'i64': b'\x00\x00\x00\x00\x00\x00\x00\x80' + b'\xff' * 7 + b'\x7f',
            'f32': b'\x01\x00\xc0\x7f\x00\x00\x00\x80',
            'f16': b'\x01\x7e\x00\x80',
        }
        for dtype, bits in bit_patterns.items():
            source = bytearray(bits)
            owned = HostStorage.from_buffer(source, shape=(2,), dtype=dtype)
            borrowed = HostStorage.from_buffer(source, shape=(2,), dtype=dtype, copy=False)
            assert (bytes(owned.to_buffer())) == (bits)
            view = borrowed.to_buffer(copy=False)
            assert (view.readonly)
            assert (bytes(view)) == (bits)
            source[0] ^= 1
            assert (bytes(owned.to_buffer())) == (bits)
            assert (bytes(view)) == (bytes(source))
            del source, borrowed
            gc.collect()
            assert (view[0]) == (bits[0] ^ 1)
        typed = array('i', [3, -4])
        assert (HostStorage.from_buffer(typed, shape=(1, 2), dtype='i32').to_flat()) == ([3, -4])
        assert (bytes(HostStorage.from_buffer(b'', shape=(0,), dtype='f32').to_buffer())) == (b'')
        with pytest.raises(ValueError, match='writable'):
            HostStorage.from_buffer(bytes(8), shape=(2,), dtype='i32', copy=False)
        with pytest.raises(ValueError, match='C-contiguous'):
            HostStorage.from_buffer(memoryview(bytearray(16))[::2], shape=(2,), dtype='i32')
        with pytest.raises(ValueError, match='byte count'):
            HostStorage.from_buffer(bytes(7), shape=(2,), dtype='i32')
        with pytest.raises(TypeError, match='format'):
            HostStorage.from_buffer(array('f', [1, 2]), shape=(2,), dtype='i32')
        with pytest.raises(ValueError, match='aligned'):
            HostStorage.from_buffer(memoryview(bytearray(9))[1:], shape=(2,), dtype='i32', copy=False)

    def test_storage_roundtrips_and_bounds(self):
        for dtype, values in (
            ("bool", [True, False]),
            ("i32", [-(2**31), 2**31 - 1]),
            ("i64", [-(2**63), 2**63 - 1]),
            ("f32", [1.5, -0.0]),
            ("f16", [1.5, -0.0]),
        ):
            s = HostStorage.from_flat(values, shape=(1, 2), dtype=dtype)
            assert (s.to_flat()) == (values)
            assert (s.data_ptr) > (0)
            if dtype in ("f32", "f16"):
                assert (math.copysign(1, s.to_flat()[1])) == (-1)
        assert (HostStorage.from_flat([], shape=(0,), dtype='f32').nbytes) == (0)
        assert (HostStorage.from_flat([1], shape=(), dtype='fp16').dtype) == ('f16')
        with pytest.raises(ValueError):
            HostStorage.from_flat([2**63], shape=(1,), dtype="i64")
        with pytest.raises(TypeError):
            HostStorage.from_flat([1], shape=(1,), dtype="bool")
        with pytest.raises(ValueError):
            HostStorage.from_flat([1e30], shape=(1,), dtype="f16")


@pytest.mark.usefixtures("engine_context")
class TestTensorCreation:
    def test_size_factories_create_buffers_without_compiling(self):
        before = self.engine.cache_info()['compilations']
        assert (ft.tensor(data=[[1, 2], [3, 4]]).shape) == ((2, 2))
        for dtype in (ft.bool, ft.int32, ft.int64, ft.fp32, ft.fp16):
            zero = ft.zeros(2, 3, dtype=dtype)
            one = ft.ones((2, 3), dtype=dtype)
            assert (zero.dtype) is (dtype)
            assert (zero.tolist()) == ([[0, 0, 0], [0, 0, 0]])
            assert (one.tolist()) == ([[1, 1, 1], [1, 1, 1]])
            assert (zero.engine) is (self.engine)
        assert (ft.full((2,), 3).dtype) is (ft.fp32)
        assert (ft.full((2, 3), -2 ** 63, dtype=ft.int64).tolist()) == ([[-2 ** 63] * 3] * 2)
        assert (ft.full((3,), True, dtype=ft.bool).tolist()) == ([True] * 3)
        negative_zero = ft.full((3,), -0.0, dtype=ft.fp16).tolist()
        assert (all((math.copysign(1, value) == -1 for value in negative_zero)))
        assert (ft.zeros(()).shape) == (())
        assert (ft.ones(()).tolist()) == (1.0)
        assert (ft.full((), 2).tolist()) == (2.0)
        assert (ft.zeros([0, 3]).tolist()) == ([])
        assert (ft.ones(2, device=ft.device('cpu')).device) == (ft.device('cpu'))
        allocated = ft.empty(2, 3)
        assert (allocated.shape) == ((2, 3))
        assert (allocated.to_buffer().nbytes) == (24)
        assert (self.engine.cache_info()['compilations']) == (before)

    def test_factory_gradients_and_rejected_sizes(self):
        x = ft.full((3,), 2)
        assert (ft.grad(x * x, x, seed=ft.ones(3)).tolist()) == ([4.0, 4.0, 4.0])
        for size in ((), (True,), (-1,), (1.5,), ((2,), 3), (2**63,), (2**61,)):
            with pytest.raises((TypeError, ValueError)):
                ft.zeros(*size)
        with pytest.raises(TypeError):
            ft.full((2,), 1, dtype=ft.bool)
        with pytest.raises(TypeError):
            ft.full(2, 1)
        with pytest.raises(TypeError):
            ft.reshape(ft.ones(2), 2)
        with pytest.raises(ValueError):
            ft.full((0,), 2**31, dtype=ft.int32)

    def test_shape_helpers_validate_inference_and_integer_dimensions(self):
        from fugatensor._shape import infer_reshape, normalize_size

        class Dimension:
            def __index__(self):
                return 3

        assert (normalize_size((2, Dimension()))) == ((2, 3))
        assert (infer_reshape((2, -1), 6)) == ((2, 3))
        assert (infer_reshape(([-1, 3],), 6)) == ((2, 3))
        assert (infer_reshape((-1, 3), 0)) == ((0, 3))
        assert (infer_reshape(((),), 1)) == (())
        assert (infer_reshape((0, 3), 0)) == ((0, 3))
        for shape, count in (((), 1), ((0, -1), 0), ((-1, -1), 6), ((2, -1), 5),
                             ((2, 3), 5), ((-2,), 2), ((True,), 1), ((-1,), 2**63)):
            with pytest.raises((TypeError, ValueError)):
                infer_reshape(shape, count)


class SharedJITExecutionMixin:
    def test_primitive_binding_matrix_preserves_keywords_defaults_and_cache_inputs(self):
        for dtype, shape, keywords in product((ft.fp32, ft.fp16), ((1, 2), (2, 3)), (False, True)):
            rows, columns = shape
            for scalar in (1, 3):
                x = ft.tensor([[scalar] * columns for _ in range(rows)], dtype=dtype)
                y = ft.tensor([[2] * columns for _ in range(rows)], dtype=dtype)
                added = ft.add(x=x, y=y) if keywords else ft.add(x, y)
                result = ft.sum(added, axis=1, keepdims=True)
                assert (result.tolist()) == ([[columns * (scalar + 2)]] * rows)
                assert (result.shape) == ((rows, 1))
                assert (result.dtype) is (dtype)
        x = ft.tensor([[1, 2], [3, 4]])
        assert (ft.matmul(x, x).tolist()) == ([[7, 10], [15, 22]])
        assert (ft.matmul(a=x, b=x, transpose_b=True).tolist()) == ([[5, 11], [11, 25]])
        for invoke in (lambda: ft.add(x, x, x=x), lambda: ft.add(x, missing=x),
                       lambda: ft.matmul(x, x, True)):
            with pytest.raises(TypeError):
                invoke()

    def test_primitive_type_contracts_reject_invalid_cold_and_hot_bindings(self):
        x = ft.tensor([1, 2])
        wrong_shape = ft.tensor([1, 2, 3])
        wrong_dtype = ft.tensor([1, 2], dtype=ft.int32)
        boolean = ft.tensor([True, False], dtype=ft.bool)
        wrong_bool_shape = ft.tensor([True], dtype=ft.bool)
        for operation, warm in product((lambda a, b: a + b, lambda a, b: a * b), (False, True)):
            if warm:
                ft.evaluate(operation(x, x))
            for left, right, error in ((x, wrong_shape, ValueError), (x, wrong_dtype, TypeError),
                                        (boolean, boolean, TypeError), (boolean, 1, TypeError),
                                        (1, boolean, TypeError), (boolean, wrong_bool_shape, ValueError)):
                with pytest.raises(error):
                    operation(left, right)

    def test_general_template_binding_retains_python_parameter_rules(self):
        from fugatensor._jit import jit

        @jit
        def operation(x, /, scale=2, *, offset=1):
            return x * scale + offset

        x = ft.tensor([1, 2])
        assert (operation(x).tolist()) == ([3, 5])
        assert (operation(x, scale=3, offset=4).tolist()) == ([7, 10])
        assert (operation(x, 3, offset=4).tolist()) == ([7, 10])
        for invoke in (lambda: operation(x=x), lambda: operation(x, 3, 4),
                       lambda: operation(x, 3, scale=4)):
            with pytest.raises(TypeError):
                invoke()

    def test_reshape_is_lazy_fuses_and_preserves_values(self):
        x = ft.tensor([1, 2, 3, 4, 5, 6])
        before = self.engine.stats
        with mock.patch.object(self.engine, "_compile", side_effect=AssertionError("eager reshape")):
            y = (x + 1).reshape(2, -1) * ft.ones(2, 3)
            assert (y.shape) == ((2, 3))
        assert (y.tolist()) == ([[2, 3, 4], [5, 6, 7]])
        assert (self.engine.stats['executions'] - before['executions']) == (1)
        assert ('fuga.reshape') in (self.engine.last_fuga_ir)
        assert (ft.reshape(y, [-1]).tolist()) == ([2, 3, 4, 5, 6, 7])
        for dtype, values in ((ft.bool, [True, False]), (ft.int64, [2**62 + 1, -(2**62)])):
            shaped = ft.tensor(values, dtype=dtype).reshape(1, 2)
            assert (shaped.dtype) is (dtype)
            assert (shaped.tolist()) == ([values])
        assert (ft.tensor(2).reshape(1, 1).tolist()) == ([[2]])
        assert (ft.tensor([2]).reshape(()).tolist()) == (2)
        assert (ft.empty(0, 3).reshape(-1).tolist()) == ([])
        for shape in ((), (4,), (-1, -1), (True, 6), (0, -1)):
            with pytest.raises((TypeError, ValueError)):
                x.reshape(*shape)
        with pytest.raises(TypeError):
            ft.tensor([1, 2], shape=(1, 2))

    def test_public_dtype_objects_survive_deferred_execution(self):
        x = ft.tensor([1, 2])
        assert (x.dtype) is (ft.fp32)
        assert ('dtype=ft.fp32') in (repr(x))
        low = ft.cast(x + 1, ft.fp16)
        assert (low.dtype) is (ft.fp16)
        assert (low.tolist()) == ([2.0, 3.0])
        assert (low.dtype) is (ft.fp16)
        assert (ft.tensor([1], dtype='fp16').dtype) is (ft.fp16)
        integer = ft.tensor([2], dtype=ft.int64)
        assert (integer.dtype) is (ft.int64)
        assert (integer.tolist()) == ([2])
        assert (ft.tensor([True], dtype=ft.bool).dtype) is (ft.bool)
        assert (ft.DType('f32')) is (ft.fp32)
        assert (json.dumps({'dtype': ft.fp32})) == ('{"dtype": "f32"}')

    def test_public_buffer_import_and_export(self):
        source = array('f', [1.5, -0.0])
        x = ft.from_buffer(source, shape=(2,), dtype=ft.fp32, engine=self.engine)
        assert (x.dtype) is (ft.fp32)
        source[0] = 9
        raw = (x * 2).to_buffer()
        assert (raw.readonly)
        assert (list(raw.cast('f'))) == ([3.0, -0.0])
        assert (math.copysign(1, raw.cast('f')[1])) == (-1)
        if self.device == 'cpu':
            borrowed = self.engine.from_buffer(source, shape=(2,), dtype='f32', copy=False)
            view = borrowed.to_buffer(copy=False)
            source[0] = 7
            assert (view.cast('f')[0]) == (7)
            assert (borrowed.tolist()) == ([7.0, -0.0])
        else:
            with pytest.raises(ValueError, match='copy=True'):
                self.engine.from_buffer(source, shape=(2,), dtype='f32', copy=False)
            with pytest.raises(ValueError, match='copy=True'):
                x.to_buffer(copy=False)

    def test_cache_hit_skips_projection_composition_and_ir_printing(self):
        from fugatensor import _lazy
        from fugatensor.ir import Module
        calls = []

        def expression(x, y):
            calls.append(True)
            shared = x + y
            return shared * shared + 3

        self.engine.clear_cache()
        x, y = ft.tensor([1, 2], dtype='i32'), ft.tensor([3, 4], dtype='i32')
        assert (expression(x, y).tolist()) == ([19, 39])
        changed = ft.tensor([5, 6], dtype='i32')
        hits = self.engine.stats['cache_hits']
        with mock.patch.object(_lazy, '_compose', side_effect=AssertionError('recomposed cached graph')), \
             mock.patch.object(_lazy, '_project', side_effect=AssertionError('reprojected cached template')), \
             mock.patch.object(Module, 'to_text', side_effect=AssertionError('printed cached graph')):
            assert (expression(changed, x).tolist()) == ([39, 67])
        assert (calls) == ([True, True])
        assert (self.engine.stats['cache_hits']) == (hits + 1)

        # Distinct and shared leaves require different physical binding tables.
        assert (expression(x, x).tolist()) == ([7, 19])
        assert (expression(y, y).tolist()) == ([39, 67])
        pending = x + y
        template = pending._pending.expression.module
        with pytest.raises(ValueError, match='immutable'):
            template.functions[0].name = 'changed'
        with pytest.raises(ValueError, match='immutable'):
            template.functions[0].block._operations = ()

        traces = []

        class Scale(ft.fun):
            def forward(self, value):
                traces.append(True)
                return value * 7

        scale = Scale()
        assert (scale(x).tolist()) == ([7, 14])
        with mock.patch.object(_lazy, '_compose', side_effect=AssertionError('recomposed cached custom graph')), \
             mock.patch.object(_lazy, '_project', side_effect=AssertionError('reprojected cached custom template')), \
             mock.patch.object(Module, 'to_text', side_effect=AssertionError('printed cached custom graph')):
            assert (scale(y).tolist()) == ([21, 28])
        assert (traces) == ([True, True])

    def test_lazy_fusion_cache_hits_and_artifacts(self):
        self.engine.clear_cache()
        calls = []

        def double(value):
            return value * 2

        def arithmetic(x, y):
            calls.append("called")
            return double(x + y)

        x = ft.tensor([[1, 2], [3, 4]], dtype="f32")
        y = ft.tensor([[10, 20], [30, 40]], dtype="f32")
        before = self.engine.stats
        with mock.patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before demand")), \
             mock.patch.object(self.engine, "_execute", side_effect=AssertionError("executed before demand")):
            pending = arithmetic(x, y)
            assert ((pending.shape, pending.dtype, pending.device)) == (((2, 2), ft.fp32, ft.get_default_device()))
            assert ('shape=(2, 2)') in (repr(pending))
        assert (pending.tolist()) == ([[22.0, 44.0], [66.0, 88.0]])
        assert (self.engine.stats['compilations'] - before['compilations']) == (1)
        assert (self.engine.stats['executions'] - before['executions']) == (1)
        executed = self.engine.stats["executions"]
        assert (ft.evaluate(pending)) is (pending)
        assert (pending.tolist()) == ([[22.0, 44.0], [66.0, 88.0]])
        assert (self.engine.stats['executions']) == (executed)

        changed = ft.tensor([[5, 6], [7, 8]], dtype="f32")
        hits = self.engine.stats["cache_hits"]
        assert (arithmetic(y=y, x=changed).tolist()) == ([[30.0, 52.0], [74.0, 96.0]])
        assert (calls) == (['called', 'called'])
        assert (self.engine.stats['compilations'] - before['compilations']) == (1)
        assert (self.engine.stats['cache_hits']) == (hits + 1)
        assert (self.engine.cache_info()['specializations']) == (1)
        assert (self.engine.last_report['phase']) == (self.phase)

        artifacts = Path(self.engine.last_artifact_dir)
        assert (list(artifacts.rglob(self.artifact_pattern)))
        assert ((artifacts / 'fuga.ir').read_text()) == (self.engine.last_fuga_ir)
        assert ((artifacts / 'frontend.mlir').read_text()) == (self.engine.last_ask_ir)
        mapping = json.loads((artifacts / "fuga-to-ask.json").read_text())
        assert (mapping['schema_version']) == (1)

    def test_joint_materialization_and_shared_values(self):
        self.engine.clear_cache()
        x = ft.tensor([1, 2, 3], dtype="i32")
        shared = x + 2
        left, right = shared * 3, shared + 7
        before = self.engine.stats
        outputs = ft.evaluate(left, right)
        assert isinstance(outputs, tuple)
        assert (outputs[0]) is (left)
        assert (outputs[1]) is (right)
        assert (self.engine.stats['compilations'] - before['compilations']) == (1)
        assert (self.engine.stats['executions'] - before['executions']) == (1)
        assert ((left.tolist(), right.tolist())) == (([9, 12, 15], [10, 11, 12]))
        assert (len({value.storage.data_ptr for value in (x, left, right)})) == (3)
        executed = self.engine.stats["executions"]
        duplicate = ft.evaluate(left, left)
        assert (duplicate[0]) is (duplicate[1])
        assert (self.engine.stats['executions']) == (executed)

    def test_unused_pending_inputs_do_not_execute_or_validate_dead_definitions(self):
        class Scale(ft.fun):
            factor = 2
            def forward(self, value):
                return value * self.factor

        class First(ft.fun):
            def forward(self, value, unused):
                return value + 3

        x = ft.tensor([1, 2], dtype="f32")
        scale = Scale()
        dead = scale(x)
        useful = First()(x, dead)
        constant = ft.ones_like(dead.detach())
        scale.factor = 7
        assert (useful.tolist()) == ([4.0, 5.0])
        assert (constant.tolist()) == ([1.0, 1.0])
        assert (dead._storage) is None
        with pytest.raises(RuntimeError, match='definition changed'):
            dead.tolist()

        opaque = ft.triton.from_source(
            "import triton\nimport triton.language as tl\n"
            "@triton.jit\ndef opaque(inp, out, N: tl.constexpr):\n pass\n",
            entry="opaque", input="inp", output="out", constants=lambda spec: {"N": spec.numel}, grid=(1,))
        pending = opaque(x)
        assert (ft.zeros_like(pending).tolist()) == ([0.0, 0.0])
        assert (pending._storage) is None
        assert ('ask.external') not in (self.engine.last_ask_ir)

    def test_requested_fun_outputs_merge_demands_and_exclude_dead_nested_definitions(self):
        class Pair(ft.fun):
            def forward(self, first, second):
                return first * 2, second * 3

        x, y = ft.tensor([1, 2]), ft.tensor([4, 5])
        left, right = Pair()(x, y)
        ft.evaluate(left + 1, right + 2, left)
        assert (left.tolist()) == ([2.0, 4.0])
        assert (right.tolist()) == ([12.0, 15.0])

        class Scale(ft.fun):
            factor = 2
            def forward(self, value):
                return value * self.factor

        scale = Scale()

        class NestedPair(ft.fun):
            def forward(self, value):
                return value + 1, scale(value)

        selected, discarded = NestedPair()(x)
        scale.factor = 9
        assert (selected.tolist()) == ([2.0, 3.0])
        assert (discarded._storage) is None
        with pytest.raises(RuntimeError, match='definition changed'):
            discarded.tolist()

    def test_compile_and_execution_failure_do_not_materialize_or_fall_back(self):
        self.engine.clear_cache()
        x = ft.tensor([1, -2, 3], dtype="i32")
        pending = (x + 2) * 3
        before = self.engine.stats["executions"]
        with mock.patch.object(self.engine, "_compile", side_effect=CompilerError("forced compiler failure")):
            with pytest.raises(CompilerError, match='forced compiler failure'):
                pending.tolist()
        assert (self.engine.stats['executions']) == (before)
        assert (pending.tolist()) == ([9, 0, 15])

        second = (x + 4) * 5
        with mock.patch.object(self.engine, "_execute", side_effect=RuntimeError("forced execution failure")):
            with pytest.raises(RuntimeError, match='forced execution failure'):
                second.tolist()
        assert (second.tolist()) == ([25, 10, 35])

    def test_all_precisions_shapes_casts_and_independent_outputs(self):
        def pair(x, y):
            compute = "i32" if x.dtype == "bool" else x.dtype
            lhs, rhs = x.astype(compute), y.astype(compute)
            return (lhs + rhs).astype(x.dtype), (lhs * rhs).astype(x.dtype)

        for dtype in ("bool", "i32", "i64", "f32", "f16", "bf16"):
            vals = [True, False, True, False] if dtype == "bool" else [1, -2, 3, 4]
            other = [True, True, False, False] if dtype == "bool" else [2, 3, 4, -5]
            if dtype == "bf16" and not self.supports_bf16:
                with pytest.raises((TypeError, CUDAError), match='CPU tensors support only|bf16.*support'):
                    ft.tensor(vals, dtype=dtype)
                continue
            s, m = ft.evaluate(*pair(ft.tensor(vals, dtype=dtype), ft.tensor(other, dtype=dtype)))
            conv = bool if dtype == "bool" else (lambda v: v)
            assert (s.tolist()) == ([conv(a + b) for a, b in zip(vals, other)])
            assert (m.tolist()) == ([conv(a * b) for a, b in zip(vals, other)])
            assert (s.device) == (ft.get_default_device())
            assert (s.storage.data_ptr) != (m.storage.data_ptr)

        # Rank-0, empty, partial-block, full-width i64, and ordinary Python aliases.
        def affine(x):
            return (x + 1) * 2

        for shape, vals, exp in (
            ((), [3], [8]),
            ((0,), [], []),
            ((2, 0), [], []),
            ((129,), list(range(129)), list(range(2, 260, 2))),
        ):
            res = affine(ft.tensor(vals, dtype=ft.int32).reshape(shape))
            assert ((res.shape, self.engine.to_flat(res))) == ((shape, exp))

        def aliases(x, y):
            return x, x + y, x

        big = 2**60 + 3
        ax = ft.tensor([big, -7], dtype="i64")
        o0, o1, o2 = aliases(ax, ax)
        assert (o1.tolist()) == ([2 * big, -14])
        assert (o0) is (ax)
        assert (o2) is (ax)
        assert (ax.storage.data_ptr) != (o1.storage.data_ptr)

        # Mixed-precision ULP preservation & saturating float->int/bool casts.
        def widened(x, offset):
            return (x.astype("f32") + offset + (-offset)).astype(x.dtype)

        for dtype, bits in (("f16", 10), ("bf16", 7)):
            if dtype == "bf16" and not self.supports_bf16:
                continue
            vals = [1 + 2**-bits, -1.25, 0.5]
            assert (widened(ft.tensor(vals, dtype=dtype), 2 ** bits).tolist()) == (vals)

        def convert(x):
            return x.astype("i32"), x.astype("i64"), x.astype("bool")

        n32, n64, b = convert(ft.tensor([float("nan"), float("inf"), float("-inf"), -0.0, -1.75, 2.5], dtype="f32"))
        assert (n32.tolist()) == ([0, 2 ** 31 - 1, -2 ** 31, 0, -1, 2])
        assert (n64.tolist()) == ([0, 2 ** 63 - 1, -2 ** 63, 0, -1, 2])
        assert (b.tolist()) == ([True, True, True, False, True, True])


@pytest.mark.usefixtures("engine_context")
class TestHostJIT(SharedJITExecutionMixin):
    device, phase, artifact_pattern = "cpu", "host_object", "*.so"

    def test_python_functions_captures_constants_and_engine_lifecycle(self):
        assert not (hasattr(ft, 'jit'))
        assert not (hasattr(ft, 'JITFunction'))
        captured = ft.tensor([3, 4], dtype="i32")
        x = ft.tensor([1, 2], dtype="i32")

        def ordinary(v):
            # Python executes normally; captured tensors join the same lazy graph.
            other = ft.tensor([7, 8], dtype="i32")
            assert (captured.tolist()) == ([3, 4])
            return v + captured + other

        assert (ordinary(x).tolist()) == ([11, 14])

        self.engine.clear_cache()
        fx = ft.tensor([1.0], dtype="f32")
        pos = (fx * 0.0).tolist()[0]
        neg = (fx * -0.0).tolist()[0]
        assert ((math.copysign(1.0, pos), math.copysign(1.0, neg))) == ((1.0, -1.0))
        assert (self.engine.cache_info()['specializations']) == (2)

        other = ft.Engine(cache_dir=self.cache_dir / "temp-engine")
        ox = other.tensor([5], dtype="i32")
        completed = ox + 1
        assert (completed.tolist()) == ([6])
        pending = ox * 3
        assert (other.cache_info()['specializations']) > (0)
        with pytest.raises(ValueError, match='same Engine|different.*Engine'):
            x + ox
        with pytest.raises(ValueError, match='Engine|engine'):
            ft.evaluate(x, pending)
        ref = weakref.ref(other)
        other.close()
        assert (other.cache_info()['specializations']) == (0)
        for value in (completed, pending):
            with pytest.raises(RuntimeError, match='closed'):
                value.tolist()
        with pytest.raises(RuntimeError, match='closed'):
            ox + 1
        del value, completed, pending, ox, other
        gc.collect()
        assert (ref()) is None

    def test_cuda_alias_and_external_shape_delegation(self):
        # cuda:0 normalization; ASK chooses a legal multi-kernel boundary.
        class _MetadataSession:
            capability = (8, 6)
            stats = {}
            def __init__(self, device):
                self.ordinal = device
            def from_host(self, values, *, shape, dtype):
                data = list(values)
                return SimpleNamespace(shape=shape, dtype=dtype, to_host=lambda: list(data))
            def close(self):
                pass

        id_kernel = ft.triton.from_source(
            "import triton\nimport triton.language as tl\n"
            "@triton.jit\ndef id_k(inp, out, N: tl.constexpr):\n pass\n",
            entry="id_k", input="inp", output="out",
            constants=lambda s: {"N": s.numel}, grid=(1,),
        )
        with mock.patch("fugatensor.engine.ExecutionSession", _MetadataSession):
            with ft.Engine("cuda:0", cache_dir=self.cache_dir / "meta-cuda") as ceng:
                cx = ft.tensor([1, 2], dtype="f32", device="cuda:0")
                assert ((cx.device, ceng.session.ordinal)) == ((ft.device('cuda:0'), 0))
                with pytest.raises(ValueError, match='device differs'):
                    ft.tensor([1], engine=ceng, device="cuda:1")

                def bad_ext(v):
                    return id_kernel(v).astype("f16")

                with mock.patch.object(ceng.compiler, "compile_cuda",
                                       side_effect=CompilerError("planner reached")) as cc:
                    with pytest.raises(CompilerError, match='planner reached'):
                        bad_ext(cx).tolist()
                    cc.assert_called_once()
                    assert (cc.call_args.kwargs['target']) == ({'backend': 'cuda', 'arch': 86})


@requires_cuda
@pytest.mark.usefixtures("engine_context")
class TestOwnedCUDAJIT(SharedJITExecutionMixin):
    def test_device_objects_factories_and_cross_device_errors(self):
        gpu = ft.ones(2, 2, device=ft.device('cuda', 0))
        cpu = ft.zeros(2, 2, device=ft.device('cpu'))
        assert (gpu.device) == (ft.device('cuda:0'))
        assert ((gpu.device.type, gpu.device.index)) == (('cuda', 0))
        assert (gpu.tolist()) == ([[1, 1], [1, 1]])
        assert (ft.full((2,), -0.0, device='cuda').to_buffer().tobytes()) == (b'\x00\x00\x00\x80' * 2)
        assert (ft.empty(0, 3, device='cuda').shape) == ((0, 3))
        for operation in (lambda: cpu + gpu, lambda: gpu * cpu,
                          lambda: ft.add(cpu, gpu), lambda: ft.multiply(gpu, cpu)):
            with pytest.raises(RuntimeError, match='same device'):
                operation()

    device, phase, artifact_pattern = "cuda", "cuda_llvm", "*.cubin"

    def test_cached_jit_replays_graph_through_native_bridge(self):
        from fugatensor._ask.executable import PreparedCUDAKernel, PreparedCUDAProgram
        from fugatensor._ask.runtime import ExecutionSession

        def pair(value):
            return value * 3 + 1, value * value

        self.engine.clear_cache()
        first_input = ft.tensor([1, 2, 3], dtype="i32")
        first = ft.evaluate(*pair(first_input))
        assert ([output.tolist() for output in first]) == ([[4, 7, 10], [1, 4, 9]])
        second_input = ft.tensor([5, 6, 7], dtype="i32")
        before = self.engine.stats
        assert (before['runtime']['graph_launches']) > (0)
        with mock.patch.object(PreparedCUDAKernel, "submit_many", side_effect=AssertionError("Python kernel submission")), \
             mock.patch.object(PreparedCUDAProgram, "submit_many", side_effect=AssertionError("Python plan submission")), \
             mock.patch.object(ExecutionSession, "submit", side_effect=AssertionError("Python launch dispatch")), \
             mock.patch.object(ExecutionSession, "tensor", side_effect=AssertionError("Python output allocation")), \
             mock.patch.object(self.engine, "_compile", side_effect=AssertionError("recompiled cached graph")):
            second = ft.evaluate(*pair(second_input))
            assert ([output.tolist() for output in second]) == ([[16, 19, 22], [25, 36, 49]])
            # Keep both result sets live: replay must not overwrite the prior outputs.
            assert ([output.tolist() for output in first]) == ([[4, 7, 10], [1, 4, 9]])
        after = self.engine.stats
        assert (after['cache_hits']) == (before['cache_hits'] + 1)
        assert (after['compilations']) == (before['compilations'])
        assert (after['runtime']['graph_launches']) == (before['runtime']['graph_launches'] + 1)
        assert (after['runtime']['graph_instantiations']) == (before['runtime']['graph_instantiations'])
        assert (len({value.storage.data_ptr for value in (*first, *second, first_input, second_input)})) == (6)

    def test_ask_ir_and_executable_caches_survive_engine_clear_cache(self):
        def linear(x):
            return x * 2 + 1

        assert (linear(ft.tensor([1, 2, 3])).tolist()) == ([3, 5, 7])
        assert (self.engine.last_report['ir_cache']['status']) == ('miss')
        assert (self.engine.last_report['executable_cache']['status']) == ('miss')
        self.engine.clear_cache()
        assert (linear(ft.tensor([4, 5, 6])).tolist()) == ([9, 11, 13])
        assert (self.engine.last_report['ir_cache']['status']) == ('hit')
        assert (self.engine.last_report['executable_cache']['status']) == ('hit')
