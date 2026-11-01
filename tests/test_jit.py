"""HostStorage, CPU and native CUDA @ft.jit execution, specialization caching, and Engine lifecycle."""
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
    def test_cache_hits_artifacts_and_eager_operators(self):
        traces = 0

        @ft.jit
        def arithmetic(x, y):
            nonlocal traces
            traces += 1
            return (x + y) * 2

        x = ft.tensor([[1, 2], [3, 4]], dtype="f32")
        y = ft.tensor([[10, 20], [30, 40]], dtype="f32")
        self.assertEqual(arithmetic(x, y).tolist(), [[22.0, 44.0], [66.0, 88.0]])
        changed = ft.tensor([[5, 6], [7, 8]], dtype="f32")
        self.assertEqual(arithmetic(y=y, x=changed).tolist(), [[30.0, 52.0], [74.0, 96.0]])
        self.assertEqual(traces, 1)
        self.assertEqual(arithmetic.cache_info(), {"compilations": 1, "cache_hits": 1, "specializations": 1})
        self.assertEqual(arithmetic.last_report["phase"], self.phase)

        artifacts = Path(arithmetic.last_artifact_dir)
        self.assertTrue(list(artifacts.rglob(self.artifact_pattern)))
        self.assertEqual((artifacts / "fuga.ir").read_text(), arithmetic.last_fuga_ir)
        self.assertEqual((artifacts / "frontend.mlir").read_text(), arithmetic.last_ask_ir)
        mapping = json.loads((artifacts / "fuga-to-ask.json").read_text())
        self.assertEqual(mapping["schema_version"], 1)

        # Eager operators compile through JIT as well.
        e = ft.tensor([1, 2, 3], dtype="i32")
        self.assertEqual(((e + 2) * 3).tolist(), [9, 12, 15])

    def test_all_precisions_shapes_casts_and_independent_outputs(self):
        @ft.jit
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
                        pair(ft.tensor(vals, dtype=dtype), ft.tensor(other, dtype=dtype))
                    continue
                s, m = pair(ft.tensor(vals, dtype=dtype), ft.tensor(other, dtype=dtype))
                conv = bool if dtype == "bool" else (lambda v: v)
                self.assertEqual(s.tolist(), [conv(a + b) for a, b in zip(vals, other)])
                self.assertEqual(m.tolist(), [conv(a * b) for a, b in zip(vals, other)])
                self.assertEqual(s.device, self.device)
                self.assertNotEqual(s.storage.data_ptr, m.storage.data_ptr)

        # Rank-0, empty, partial-block, full-width i64, and distinct identity buffers.
        @ft.jit
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

        @ft.jit
        def aliases(x, y):
            return x, x + y, x

        big = 2**60 + 3
        ax = ft.tensor([big, -7], dtype="i64")
        o0, o1, o2 = aliases(ax, ax)
        self.assertEqual(o1.tolist(), [2 * big, -14])
        self.assertEqual(len({t.storage.data_ptr for t in (ax, o0, o1, o2)}), 4)

        # Mixed-precision ULP preservation & saturating float->int/bool casts.
        @ft.jit
        def widened(x, offset):
            return (x.astype("f32") + offset + (-offset)).astype(x.dtype)

        for dtype, bits in (("f16", 10), ("bf16", 7)):
            if dtype == "bf16" and not self.supports_bf16:
                continue
            vals = [1 + 2**-bits, -1.25, 0.5]
            self.assertEqual(widened(ft.tensor(vals, dtype=dtype), 2**bits).tolist(), vals)

        @ft.jit
        def convert(x):
            return x.astype("i32"), x.astype("i64"), x.astype("bool")

        n32, n64, b = convert(ft.tensor([float("nan"), float("inf"), float("-inf"), -0.0, -1.75, 2.5], dtype="f32"))
        self.assertEqual(n32.tolist(), [0, 2**31 - 1, -(2**31), 0, -1, 2])
        self.assertEqual(n64.tolist(), [0, 2**63 - 1, -(2**63), 0, -1, 2])
        self.assertEqual(b.tolist(), [True, True, True, False, True, True])


class HostJITTests(SharedJITExecutionMixin, EngineTestCase):
    device, phase, artifact_pattern = "cpu", "host_object", "*.so"

    def test_trace_guards_static_specialization_and_engine_eviction(self):
        captured = ft.tensor([3, 4], dtype="i32")
        x = ft.tensor([1, 2], dtype="i32")

        for action in (
            lambda: captured.tolist(),
            lambda: ft.tensor([7, 8], dtype="i32"),
            lambda: captured + 2,
        ):
            @ft.jit
            def invalid(v):
                action()
                return v + 1

            with self.assertRaisesRegex(TypeError, "while tracing"):
                invalid(x)

        # Signed-zero static specialization.
        @ft.jit
        def scale(v, factor):
            return v * factor

        fx = ft.tensor([1.0], dtype="f32")
        pos = scale(fx, 0.0).tolist()[0]
        neg = scale(fx, -0.0).tolist()[0]
        self.assertEqual((math.copysign(1.0, pos), math.copysign(1.0, neg)), (1.0, -1.0))
        self.assertEqual(scale.cache_info()["specializations"], 2)

        # Closing or GC-ing an Engine evicts only its specialization without leaking it.
        @ft.jit
        def inc(v):
            return v + 1

        self.assertEqual(inc(x).tolist(), [2, 3])
        other = ft.Engine(cache_dir=self.cache_dir / "temp-engine")
        ox = other.tensor([5], dtype="i32")
        self.assertEqual(inc(ox).tolist(), [6])
        self.assertEqual(inc.cache_info()["specializations"], 2)
        ref = weakref.ref(other)
        other.close()
        self.assertEqual(inc.cache_info()["specializations"], 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            inc(ox)
        del ox, other
        gc.collect()
        self.assertIsNone(ref())

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

                @ft.jit
                def bad_ext(v):
                    return id_kernel(v).astype("f16")

                with mock.patch.object(ceng.compiler, "compile_cuda") as cc:
                    with self.assertRaisesRegex(NotImplementedError, "equal input/output shape and storage dtype"):
                        bad_ext(cx)
                    cc.assert_not_called()


@requires_cuda
class OwnedCUDAJITTests(SharedJITExecutionMixin, EngineTestCase):
    device, phase, artifact_pattern = "cuda", "cuda_llvm", "*.cubin"

    def test_ask_ir_and_executable_caches_survive_jit_clear_cache(self):
        @ft.jit
        def linear(x):
            return x * 2 + 1

        self.assertEqual(linear(ft.tensor([1, 2, 3])).tolist(), [3, 5, 7])
        self.assertEqual(linear.last_report["ir_cache"]["status"], "miss")
        self.assertEqual(linear.last_report["executable_cache"]["status"], "miss")
        linear.clear_cache()
        self.assertEqual(linear(ft.tensor([4, 5, 6])).tolist(), [9, 11, 13])
        self.assertEqual(linear.last_report["ir_cache"]["status"], "hit")
        self.assertEqual(linear.last_report["executable_cache"]["status"], "hit")


if __name__ == "__main__":
    unittest.main()
