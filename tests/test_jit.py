"""Default lazy tensor graphs, mandatory ASK execution, caching, and Engine lifecycle."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

from ask import CompilerError
from ask.cuda import CUDAError
import fugatensor as ft
from fugatensor.backends.host import HostStorage
from tests._helpers import EngineTestCase, requires_cuda


class HostStorageTests(unittest.TestCase):
    def test_storage_roundtrips_and_bounds(self):
        for dtype, values in (
            ("bool", [True, False]),
            ("i32", [-(2**31), 2**31 - 1]),
            ("i64", [-(2**63), 2**63 - 1]),
            ("f32", [1.5, -0.0]),
            ("f16", [1.5, -0.0]),
            ("bf16", [1.5, -0.0]),
        ):
            with self.subTest(dtype=dtype):
                s = HostStorage.from_flat(values, shape=(1, 2), dtype=dtype)
                self.assertEqual(s.to_flat(), values)
                self.assertGreater(s.data_ptr, 0)
                if dtype in ("f32", "f16", "bf16"):
                    self.assertEqual(math.copysign(1, s.to_flat()[1]), -1)
        self.assertEqual(HostStorage.from_flat([], shape=(0,), dtype="f32").nbytes, 0)
        self.assertEqual(HostStorage.from_flat([1], shape=(), dtype="fp16").dtype, "f16")
        with self.assertRaises(ValueError):
            HostStorage.from_flat([2**63], shape=(1,), dtype="i64")
        with self.assertRaises(TypeError):
            HostStorage.from_flat([1], shape=(1,), dtype="bool")
        with self.assertRaises(ValueError):
            HostStorage.from_flat([1e30], shape=(1,), dtype="f16")


class SharedJITExecutionMixin:
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
            self.assertEqual((pending.shape, pending.dtype, pending.device), ((2, 2), "f32", self.device))
            self.assertIn("shape=(2, 2)", repr(pending))
        self.assertEqual(pending.tolist(), [[22.0, 44.0], [66.0, 88.0]])
        self.assertEqual(self.engine.stats["compilations"] - before["compilations"], 1)
        self.assertEqual(self.engine.stats["executions"] - before["executions"], 1)
        executed = self.engine.stats["executions"]
        self.assertIs(ft.evaluate(pending), pending)
        self.assertEqual(pending.tolist(), [[22.0, 44.0], [66.0, 88.0]])
        self.assertEqual(self.engine.stats["executions"], executed)

        changed = ft.tensor([[5, 6], [7, 8]], dtype="f32")
        hits = self.engine.stats["cache_hits"]
        self.assertEqual(arithmetic(y=y, x=changed).tolist(), [[30.0, 52.0], [74.0, 96.0]])
        self.assertEqual(calls, ["called", "called"])
        self.assertEqual(self.engine.stats["compilations"] - before["compilations"], 1)
        self.assertEqual(self.engine.stats["cache_hits"], hits + 1)
        self.assertEqual(self.engine.cache_info()["specializations"], 1)
        self.assertEqual(self.engine.last_report["phase"], self.phase)

        artifacts = Path(self.engine.last_artifact_dir)
        self.assertTrue(list(artifacts.rglob(self.artifact_pattern)))
        self.assertEqual((artifacts / "fuga.ir").read_text(), self.engine.last_fuga_ir)
        self.assertEqual((artifacts / "frontend.mlir").read_text(), self.engine.last_ask_ir)
        mapping = json.loads((artifacts / "fuga-to-ask.json").read_text())
        self.assertEqual(mapping["schema_version"], 1)

    def test_joint_materialization_and_shared_values(self):
        self.engine.clear_cache()
        x = ft.tensor([1, 2, 3], dtype="i32")
        shared = x + 2
        left, right = shared * 3, shared + 7
        before = self.engine.stats
        outputs = ft.evaluate(left, right)
        self.assertIsInstance(outputs, tuple)
        self.assertIs(outputs[0], left)
        self.assertIs(outputs[1], right)
        self.assertEqual(self.engine.stats["compilations"] - before["compilations"], 1)
        self.assertEqual(self.engine.stats["executions"] - before["executions"], 1)
        self.assertEqual((left.tolist(), right.tolist()), ([9, 12, 15], [10, 11, 12]))
        self.assertEqual(len({value.storage.data_ptr for value in (x, left, right)}), 3)
        executed = self.engine.stats["executions"]
        duplicate = ft.evaluate(left, left)
        self.assertIs(duplicate[0], duplicate[1])
        self.assertEqual(self.engine.stats["executions"], executed)

    def test_unused_pending_inputs_do_not_execute_or_validate_dead_definitions(self):
        class Scale(ft.fun):
            factor = 2
            def forward(self, ctx, value):
                return value * self.factor

        class First(ft.fun):
            def forward(self, ctx, value, unused):
                return value + 3

        x = ft.tensor([1, 2], dtype="f32")
        scale = Scale()
        dead = scale(x)
        useful = First()(x, dead)
        constant = ft.ones_like(dead.detach())
        scale.factor = 7
        self.assertEqual(useful.tolist(), [4.0, 5.0])
        self.assertEqual(constant.tolist(), [1.0, 1.0])
        self.assertIsNone(dead._storage)
        with self.assertRaisesRegex(RuntimeError, "definition changed"):
            dead.tolist()

        opaque = ft.triton.from_source(
            "import triton\nimport triton.language as tl\n"
            "@triton.jit\ndef opaque(inp, out, N: tl.constexpr):\n pass\n",
            entry="opaque", input="inp", output="out", constants=lambda spec: {"N": spec.numel}, grid=(1,))
        pending = opaque(x)
        self.assertEqual(ft.zeros_like(pending).tolist(), [0.0, 0.0])
        self.assertIsNone(pending._storage)
        self.assertNotIn("ask.external", self.engine.last_ask_ir)

    def test_requested_fun_outputs_merge_demands_and_exclude_dead_nested_definitions(self):
        class Pair(ft.fun):
            def forward(self, ctx, first, second):
                return first * 2, second * 3

        x, y = ft.tensor([1, 2]), ft.tensor([4, 5])
        left, right = Pair()(x, y)
        ft.evaluate(left + 1, right + 2, left)
        self.assertEqual(left.tolist(), [2.0, 4.0])
        self.assertEqual(right.tolist(), [12.0, 15.0])

        class Scale(ft.fun):
            factor = 2
            def forward(self, ctx, value):
                return value * self.factor

        scale = Scale()

        class NestedPair(ft.fun):
            def forward(self, ctx, value):
                return value + 1, scale(value)

        selected, discarded = NestedPair()(x)
        scale.factor = 9
        self.assertEqual(selected.tolist(), [2.0, 3.0])
        self.assertIsNone(discarded._storage)
        with self.assertRaisesRegex(RuntimeError, "definition changed"):
            discarded.tolist()

    def test_compile_and_execution_failure_do_not_materialize_or_fall_back(self):
        self.engine.clear_cache()
        x = ft.tensor([1, -2, 3], dtype="i32")
        pending = (x + 2) * 3
        before = self.engine.stats["executions"]
        with mock.patch.object(self.engine, "_compile", side_effect=CompilerError("forced compiler failure")):
            with self.assertRaisesRegex(CompilerError, "forced compiler failure"):
                pending.tolist()
        self.assertEqual(self.engine.stats["executions"], before)
        self.assertEqual(pending.tolist(), [9, 0, 15])

        second = (x + 4) * 5
        with mock.patch.object(self.engine, "_execute", side_effect=RuntimeError("forced execution failure")):
            with self.assertRaisesRegex(RuntimeError, "forced execution failure"):
                second.tolist()
        self.assertEqual(second.tolist(), [25, 10, 35])

    def test_all_precisions_shapes_casts_and_independent_outputs(self):
        def pair(x, y):
            compute = "i32" if x.dtype == "bool" else x.dtype
            lhs, rhs = x.astype(compute), y.astype(compute)
            return (lhs + rhs).astype(x.dtype), (lhs * rhs).astype(x.dtype)

        for dtype in ("bool", "i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                vals = [True, False, True, False] if dtype == "bool" else [1, -2, 3, 4]
                other = [True, True, False, False] if dtype == "bool" else [2, 3, 4, -5]
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        ft.evaluate(*pair(ft.tensor(vals, dtype=dtype), ft.tensor(other, dtype=dtype)))
                    continue
                s, m = ft.evaluate(*pair(ft.tensor(vals, dtype=dtype), ft.tensor(other, dtype=dtype)))
                conv = bool if dtype == "bool" else (lambda v: v)
                self.assertEqual(s.tolist(), [conv(a + b) for a, b in zip(vals, other)])
                self.assertEqual(m.tolist(), [conv(a * b) for a, b in zip(vals, other)])
                self.assertEqual(s.device, self.device)
                self.assertNotEqual(s.storage.data_ptr, m.storage.data_ptr)

        # Rank-0, empty, partial-block, full-width i64, and ordinary Python aliases.
        def affine(x):
            return (x + 1) * 2

        for shape, vals, exp in (
            ((), [3], [8]),
            ((0,), [], []),
            ((2, 0), [], []),
            ((129,), list(range(129)), list(range(2, 260, 2))),
        ):
            with self.subTest(shape=shape):
                res = affine(ft.tensor(vals, shape=shape, dtype="i32"))
                self.assertEqual((res.shape, self.engine.to_flat(res)), (shape, exp))

        def aliases(x, y):
            return x, x + y, x

        big = 2**60 + 3
        ax = ft.tensor([big, -7], dtype="i64")
        o0, o1, o2 = aliases(ax, ax)
        self.assertEqual(o1.tolist(), [2 * big, -14])
        self.assertIs(o0, ax)
        self.assertIs(o2, ax)
        self.assertNotEqual(ax.storage.data_ptr, o1.storage.data_ptr)

        # Mixed-precision ULP preservation & saturating float->int/bool casts.
        def widened(x, offset):
            return (x.astype("f32") + offset + (-offset)).astype(x.dtype)

        for dtype, bits in (("f16", 10), ("bf16", 7)):
            if dtype == "bf16" and not self.supports_bf16:
                continue
            vals = [1 + 2**-bits, -1.25, 0.5]
            self.assertEqual(widened(ft.tensor(vals, dtype=dtype), 2**bits).tolist(), vals)

        def convert(x):
            return x.astype("i32"), x.astype("i64"), x.astype("bool")

        n32, n64, b = convert(ft.tensor([float("nan"), float("inf"), float("-inf"), -0.0, -1.75, 2.5], dtype="f32"))
        self.assertEqual(n32.tolist(), [0, 2**31 - 1, -(2**31), 0, -1, 2])
        self.assertEqual(n64.tolist(), [0, 2**63 - 1, -(2**63), 0, -1, 2])
        self.assertEqual(b.tolist(), [True, True, True, False, True, True])


class HostJITTests(SharedJITExecutionMixin, EngineTestCase):
    device, phase, artifact_pattern = "cpu", "host_object", "*.so"

    def test_python_functions_captures_constants_and_engine_lifecycle(self):
        self.assertFalse(hasattr(ft, "jit"))
        self.assertFalse(hasattr(ft, "JITFunction"))
        captured = ft.tensor([3, 4], dtype="i32")
        x = ft.tensor([1, 2], dtype="i32")

        def ordinary(v):
            # Python executes normally; captured tensors join the same lazy graph.
            other = ft.tensor([7, 8], dtype="i32")
            self.assertEqual(captured.tolist(), [3, 4])
            return v + captured + other

        self.assertEqual(ordinary(x).tolist(), [11, 14])

        self.engine.clear_cache()
        fx = ft.tensor([1.0], dtype="f32")
        pos = (fx * 0.0).tolist()[0]
        neg = (fx * -0.0).tolist()[0]
        self.assertEqual((math.copysign(1.0, pos), math.copysign(1.0, neg)), (1.0, -1.0))
        self.assertEqual(self.engine.cache_info()["specializations"], 2)

        other = ft.Engine(cache_dir=self.cache_dir / "temp-engine")
        ox = other.tensor([5], dtype="i32")
        completed = ox + 1
        self.assertEqual(completed.tolist(), [6])
        pending = ox * 3
        self.assertGreater(other.cache_info()["specializations"], 0)
        with self.assertRaisesRegex(ValueError, "same Engine|different.*Engine"):
            x + ox
        with self.assertRaisesRegex(ValueError, "Engine|engine"):
            ft.evaluate(x, pending)
        ref = weakref.ref(other)
        other.close()
        self.assertEqual(other.cache_info()["specializations"], 0)
        for value in (completed, pending):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                value.tolist()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            ox + 1
        del value, completed, pending, ox, other
        gc.collect()
        self.assertIsNone(ref())

    def test_cuda_alias_and_external_storage_guard_before_codegen(self):
        # cuda:0 alias normalization & pre-compile Triton storage mismatch guard.
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
                self.assertEqual((cx.device, ceng.session.ordinal), ("cuda", 0))
                with self.assertRaisesRegex(ValueError, "device differs"):
                    ft.tensor([1], engine=ceng, device="cuda:1")

                def bad_ext(v):
                    return id_kernel(v).astype("f16")

                with mock.patch.object(ceng.compiler, "compile_cuda") as cc:
                    with self.assertRaisesRegex(NotImplementedError, "equal input/output shape and storage dtype"):
                        bad_ext(cx).tolist()
                    cc.assert_not_called()


@requires_cuda
class OwnedCUDAJITTests(SharedJITExecutionMixin, EngineTestCase):
    device, phase, artifact_pattern = "cuda", "cuda_llvm", "*.cubin"

    def test_ask_ir_and_executable_caches_survive_engine_clear_cache(self):
        def linear(x):
            return x * 2 + 1

        self.assertEqual(linear(ft.tensor([1, 2, 3])).tolist(), [3, 5, 7])
        self.assertEqual(self.engine.last_report["ir_cache"]["status"], "miss")
        self.assertEqual(self.engine.last_report["executable_cache"]["status"], "miss")
        self.engine.clear_cache()
        self.assertEqual(linear(ft.tensor([4, 5, 6])).tolist(), [9, 11, 13])
        self.assertEqual(self.engine.last_report["ir_cache"]["status"], "hit")
        self.assertEqual(self.engine.last_report["executable_cache"]["status"], "hit")


if __name__ == "__main__":
    unittest.main()
