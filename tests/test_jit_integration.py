"""Execute FugaTensor against a real ASK compiler; never emulate arithmetic.

Set ASK_OPT for CPU tests. FUGATENSOR_TEST_CUDA=1 additionally requires
FUGATENSOR_LLC and FUGATENSOR_PTXAS; ASK none is sufficient. External kernel
tests opt in separately with FUGATENSOR_TEST_TRITON=1 and the two Triton paths.
A requested but broken setup fails instead of skipping.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import fugatensor as ft
from ask import Compiler, CompilerEnvironment, CompilerError, ExecutionSession
from ask.cuda import CUDAError


class BasicJITGraphs:
    """Run the same graph definitions against either ASK execution backend."""

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
        self.assertEqual(arithmetic(y=y, x=changed).tolist(), [[30.0, 52.0], [74.0, 96.0]])
        self.assertEqual(traces, 1)
        info = arithmetic.cache_info()
        self.assertEqual(info["compilations"], 1)
        self.assertEqual(info["cache_hits"], 2)
        self.assertEqual(info["specializations"], 1)
        self.assertIn('"ask.add"', arithmetic.last_ir)
        self.assertIn('"ask.mul"', arithmetic.last_ir)
        self.assertTrue(list(Path(arithmetic.last_artifact_dir).rglob(self.artifact_pattern)))
        self.assertEqual(arithmetic.last_report["phase"], self.phase)

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

    def test_same_two_input_two_output_graph_across_all_types(self):
        @ft.jit
        def pair(x, y):
            compute = "i32" if x.dtype == "bool" else x.dtype
            left, right = x.astype(compute), y.astype(compute)
            return (left + right).astype(x.dtype), (left * right).astype(x.dtype)

        for dtype in ("bool", "i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                values = [True, False, True, False] if dtype == "bool" else [1, -2, 3, 4]
                other = [True, True, False, False] if dtype == "bool" else [2, 3, 4, -5]
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        pair(ft.tensor(values, dtype=dtype), ft.tensor(other, dtype=dtype))
                    continue
                x, y = ft.tensor(values, dtype=dtype), ft.tensor(other, dtype=dtype)
                summed, multiplied = pair(x, y)
                cast = bool if dtype == "bool" else lambda value: value
                self.assertEqual(summed.tolist(), [cast(a + b) for a, b in zip(values, other)])
                self.assertEqual(multiplied.tolist(), [cast(a * b) for a, b in zip(values, other)])
                self.assertEqual(summed.dtype, dtype)
                self.assertEqual(multiplied.dtype, dtype)
                self.assertEqual(summed.device, self.device)
                self.assertNotEqual(summed.storage.data_ptr, multiplied.storage.data_ptr)
                self.assertNotIn('"ask.external"', pair.last_ir)

    def test_outputs_have_independent_shapes_and_dtypes(self):
        @ft.jit
        def independent(x, y):
            return x + 1, y.astype("f16") * 2, x.astype("bool")

        x = ft.tensor([0, 2, -3], dtype="i64")
        y = ft.tensor(1.5, dtype="f32")
        incremented, scalar, boolean = independent(x, y)
        self.assertEqual(incremented.tolist(), [1, 3, -2])
        self.assertEqual((incremented.shape, incremented.dtype), ((3,), "i64"))
        self.assertEqual(scalar.tolist(), 3.0)
        self.assertEqual((scalar.shape, scalar.dtype), ((), "f16"))
        self.assertEqual(boolean.tolist(), [False, True, True])
        self.assertEqual(boolean.dtype, "bool")

    def test_repeated_inputs_and_identity_outputs_use_independent_storage(self):
        @ft.jit
        def pair(x, y):
            return x, x + y, x

        x = ft.tensor([2**60 + 3, -7], dtype="i64")
        original, doubled, repeated = pair(x, x)
        self.assertEqual(original.tolist(), [2**60 + 3, -7])
        self.assertEqual(doubled.tolist(), [2 * (2**60 + 3), -14])
        self.assertEqual(repeated.tolist(), original.tolist())
        self.assertEqual(len({value.storage.data_ptr for value in (x, original, doubled, repeated)}), 4)

    def test_scalar_empty_and_partial_block_shapes(self):
        @ft.jit
        def arithmetic(x):
            return (x + 1) * 2

        for shape, values, expected in (((), [3], [8]), ((0,), [], []), ((2, 0), [], []),
                                        ((129,), list(range(129)), list(range(2, 260, 2)))):
            with self.subTest(shape=shape):
                x = ft.tensor(values, shape=shape, dtype="i32")
                result = arithmetic(x)
                self.assertEqual(result.shape, shape)
                self.assertEqual(self.engine.to_flat(result), expected)

    def test_same_mixed_precision_graph_preserves_low_precision_bits(self):
        @ft.jit
        def widened(x, offset):
            return (x.astype("f32") + offset + (-offset)).astype(x.dtype)

        for dtype, precision in (("f16", 10), ("bf16", 7)):
            with self.subTest(dtype=dtype):
                values = [1 + 2**-precision, -1.25, 0.5]
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        widened(ft.tensor(values, dtype=dtype), 2**precision)
                    continue
                result = widened(ft.tensor(values, dtype=dtype), 2**precision)
                self.assertEqual(result.tolist(), values)
                self.assertEqual(result.dtype, dtype)
                self.assertIn('"ask.cast"', widened.last_ir)

    def test_float_casts_saturate_and_bool_keeps_nonzero_semantics(self):
        @ft.jit
        def convert(x):
            return x.astype("i32"), x.astype("i64"), x.astype("bool")

        values = [float("nan"), float("inf"), float("-inf"), -0.0, -1.75, 2.5]
        narrow, wide, boolean = convert(ft.tensor(values, dtype="f32"))
        self.assertEqual(narrow.tolist(), [0, 2**31 - 1, -(2**31), 0, -1, 2])
        self.assertEqual(wide.tolist(), [0, 2**63 - 1, -(2**63), 0, -1, 2])
        self.assertEqual(boolean.tolist(), [True, True, True, False, True, True])


class HostJITTests(BasicJITGraphs, unittest.TestCase):
    device, phase, artifact_pattern = "cpu", "host_object", "*.so"

    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ.get("ASK_OPT")
        if not cls.driver:
            raise unittest.SkipTest("set ASK_OPT to run real ASK host JIT integration tests")
        cls.supports_bf16 = Compiler(cls.driver).capabilities()["host_bf16_hardware_supported"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fugatensor-host-test-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cpu", ask_opt=self.driver, cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1",
                     "set FUGATENSOR_TEST_CUDA=1 to run ASK's own GPU lowering")
class OwnedCUDAJITTests(BasicJITGraphs, unittest.TestCase):
    device, phase, artifact_pattern = "cuda", "cuda_llvm", "*.cubin"

    @classmethod
    def setUpClass(cls):
        required = ("ASK_OPT", "FUGATENSOR_LLC", "FUGATENSOR_PTXAS")
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise RuntimeError("requested CUDA tests are missing: " + ", ".join(missing))
        cls.driver = os.environ["ASK_OPT"]
        capabilities = Compiler(cls.driver).capabilities()
        if os.environ.get("FUGATENSOR_TEST_NONE") == "1":
            if capabilities["third_party_compilers"]:
                raise AssertionError("none verification must use a build without third-party compilers")
        with ExecutionSession(device=0) as probe:
            cls.supports_bf16 = probe.capability[0] >= 8

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fugatensor-owned-cuda-test-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cuda", ask_opt=self.driver, compilers={},
                               llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"],
                               cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)

    def test_ir_and_executable_caches_survive_retracing(self):
        @ft.jit
        def arithmetic(x):
            return x * 2 + 1

        x = ft.tensor([1, 2, 3])
        self.assertEqual(arithmetic(x).tolist(), [3, 5, 7])
        self.assertEqual(arithmetic.last_report["ir_cache"]["status"], "miss")
        self.assertEqual(arithmetic.last_report["executable_cache"]["status"], "miss")
        arithmetic.clear_cache()
        self.assertEqual(arithmetic(ft.tensor([4, 5, 6])).tolist(), [9, 11, 13])
        self.assertEqual(arithmetic.last_report["ir_cache"]["status"], "hit")
        self.assertEqual(arithmetic.last_report["executable_cache"]["status"], "hit")


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_TRITON") == "1",
                     "set FUGATENSOR_TEST_TRITON=1 to run external GPU fusion tests")
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

    def test_pure_ask_cuda_works_with_a_triton_enabled_build(self):
        @ft.jit
        def pure(x):
            return x + 1

        result = pure(ft.tensor([1, 2], dtype="f32"))
        self.assertEqual(result.tolist(), [2, 3])
        self.assertEqual(result.device, "cuda")
        self.assertNotIn('"ask.external"', pure.last_ir)


if __name__ == "__main__":
    unittest.main()
