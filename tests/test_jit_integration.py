"""Execute FugaTensor against a real ASK compiler; never emulate arithmetic.

Set ASK_OPT for CPU tests. CUDA tests additionally require
FUGATENSOR_TEST_CUDA=1 plus the four FUGATENSOR_{TRITON_PYTHON,TRITON_WORKER,
LLC,PTXAS} paths. A requested but broken CUDA setup fails instead of skipping.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import fugatensor as ft
from ask import CompilerEnvironment, ExecutionSession
from ask.cuda import CUDAError


class HostJITTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ.get("ASK_OPT")
        if not cls.driver:
            raise unittest.SkipTest("set ASK_OPT to run real ASK host JIT integration tests")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fugatensor-host-test-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cpu", ask_opt=self.driver, cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)

    def test_changed_values_reuse_compiled_code_without_retracing(self):
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
        self.assertEqual(arithmetic(changed, y).tolist(), [[30.0, 52.0], [74.0, 96.0]])
        self.assertEqual(traces, 1)
        info = arithmetic.cache_info()
        self.assertEqual(info["compilations"], 1)
        self.assertEqual(info["cache_hits"], 1)
        self.assertEqual(info["specializations"], 1)
        self.assertIn('"ask.add"', arithmetic.last_ir)
        self.assertIn('"ask.mul"', arithmetic.last_ir)
        self.assertTrue(list(Path(arithmetic.last_artifact_dir).glob("*.so")))
        self.assertEqual(arithmetic.last_report["phase"], "host_object")

    def test_shape_and_dtype_are_separate_specializations(self):
        @ft.jit
        def arithmetic(x):
            return (x + 1) * 2

        for dtype in ("i32", "i64", "f32", "f16"):
            for values, expected in (([1, 2], [4, 6]), ([[1, 2]], [[4, 6]])):
                with self.subTest(dtype=dtype, values=values):
                    x = ft.tensor(values, dtype=dtype)
                    output = arithmetic(x)
                    self.assertEqual(output.tolist(), expected)
                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(output.dtype, dtype)
        self.assertEqual(arithmetic.cache_info()["compilations"], 8)
        self.assertEqual(arithmetic.cache_info()["specializations"], 8)

    def test_multiple_outputs_and_full_width_i64(self):
        @ft.jit
        def pair(x, y):
            return x + y, x * y

        left = 2**60 + 3
        x = ft.tensor([left, -left], dtype="i64")
        y = ft.tensor([2, 3], dtype="i64")
        added, multiplied = pair(x, y)
        self.assertEqual(added.tolist(), [left + 2, -left + 3])
        self.assertEqual(multiplied.tolist(), [left * 2, -left * 3])

    def test_bool_roundtrip_and_low_precision_with_explicit_f32(self):
        @ft.jit
        def bool_calculation(x):
            return (x.astype("i32") + 1).astype("bool")

        self.assertEqual(bool_calculation(ft.tensor([False, True], dtype="bool")).tolist(), [True, True])

        @ft.jit
        def widened(x):
            return (x.astype("f32") + 1024 + (-1024)).astype("f16")

        value = 1 + 2**-10
        self.assertEqual(widened(ft.tensor([value], dtype="f16")).tolist(), [value])
        self.assertIn('"ask.cast"', widened.last_ir)

    def test_tensor_operators_also_use_jit(self):
        x = ft.tensor([1, 2, 3], dtype="i32")
        self.assertEqual(((x + 2) * 3).tolist(), [9, 12, 15])
        scalar = ft.tensor(3, dtype="i32")
        self.assertEqual((scalar + 2).tolist(), 5)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1",
                     "set FUGATENSOR_TEST_CUDA=1 to run real GPU integration tests")
class CUDAJITTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = ("ASK_OPT", "FUGATENSOR_TRITON_PYTHON", "FUGATENSOR_TRITON_WORKER",
                    "FUGATENSOR_LLC", "FUGATENSOR_PTXAS")
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise RuntimeError("requested CUDA tests are missing: " + ", ".join(missing))
        with ExecutionSession(device=0) as probe:
            cls.supports_bf16 = probe.capability[0] >= 8
        cls.directory = tempfile.TemporaryDirectory(prefix="fugatensor-cuda-test-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.engine = ft.Engine(device="cuda", ask_opt=os.environ["ASK_OPT"],
                              compilers={"triton": CompilerEnvironment(
                                  python=os.environ["FUGATENSOR_TRITON_PYTHON"],
                                  worker=os.environ["FUGATENSOR_TRITON_WORKER"])},
                              llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"],
                              cache_dir=cls.directory.name)
        cls.engine.__enter__()
        cls.addClassCleanup(cls.engine.__exit__, None, None, None)
        fixtures = Path(__file__).parent / "fixtures"
        cls.scale = ft.triton.kernel(
            fixtures / "user_scale.py", entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )
        cls.identity = ft.triton.kernel(
            fixtures / "user_identity.py", entry="identity_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )

    def test_same_triton_graph_across_numeric_precisions_and_hot_values(self):
        traces = 0
        scale = self.scale

        @ft.jit
        def mixed(x):
            nonlocal traces
            traces += 1
            return scale(x + 1) + 3

        for dtype in ("i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex(CUDAError, "bf16"):
                        mixed(ft.tensor([[1, 2], [3, 4]], dtype=dtype))
                    continue
                x = ft.tensor([[1, 2], [3, 4]], dtype=dtype)
                output = mixed(x)
                self.assertEqual(output.tolist(), [[7, 9], [11, 13]])
                self.assertEqual(output.dtype, dtype)
                changed = ft.tensor([[5, 6], [7, 8]], dtype=dtype)
                self.assertEqual(mixed(changed).tolist(), [[15, 17], [19, 21]])
                self.assertIn('"ask.external"', mixed.last_ir)
                self.assertEqual(mixed.last_report["phase"], "fused_llvm")
                self.assertEqual(mixed.last_report["fusion"]["compiler"], "triton")
                self.assertTrue(list(Path(mixed.last_artifact_dir).rglob("*.cubin")))
        count = 5 if self.supports_bf16 else 4
        self.assertEqual(traces, count)
        self.assertEqual(mixed.cache_info()["compilations"], count)
        self.assertEqual(mixed.cache_info()["cache_hits"], count)

    def test_bool_external_and_mixed_precision_casts(self):
        identity = self.identity

        @ft.jit
        def boolean(x):
            return identity(x)

        self.assertEqual(boolean(ft.tensor([False, True, False], dtype="bool")).tolist(),
                         [False, True, False])

        @ft.jit
        def widened(x):
            offset = 1024 if x.dtype == "f16" else 128
            value = (x.astype("f32") + offset + (-offset)).astype(x.dtype)
            return identity(value)

        for dtype, value in (("f16", 1 + 2**-10), ("bf16", 1 + 2**-7)):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex(CUDAError, "bf16"):
                        widened(ft.tensor([value], dtype=dtype))
                    continue
                actual = widened(ft.tensor([value], dtype=dtype))
                self.assertEqual(actual.tolist(), [value])
                self.assertEqual(actual.dtype, dtype)
                self.assertIn('"ask.cast"', widened.last_ir)

    def test_pure_ask_cuda_has_no_implicit_cpu_fallback(self):
        @ft.jit
        def pure(x):
            return x + 1

        with self.assertRaisesRegex((ValueError, RuntimeError, NotImplementedError), "external|Triton|triton"):
            pure(ft.tensor([1, 2], dtype="f32"))


if __name__ == "__main__":
    unittest.main()
