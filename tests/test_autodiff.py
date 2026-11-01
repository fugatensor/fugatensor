"""The same Tensor.backward graphs execute on CPU and CUDA through ASK."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ask import Compiler, CompilerEnvironment, CompilerError, ExecutionSession
from ask.cuda import CUDAError
import fugatensor as ft


@ft.jit
def scalar_polynomial(x):
    return ft.square(x) - 3 * x + 2


class AutodiffGraphs:
    def test_scalar_backward_across_float_storage_types_and_cache_hits(self):
        scalar_polynomial.clear_cache()
        for dtype in ("f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        value = scalar_polynomial(ft.tensor(3, dtype=dtype, requires_grad=True))
                        value.backward()
                    continue
                for value, expected in ((3, 3), (4, 5)):
                    x = ft.tensor(value, dtype=dtype, requires_grad=True)
                    self.assertIsNone(x.grad)
                    output = scalar_polynomial(x)
                    output.backward()
                    self.assertEqual(x.grad.tolist(), expected)
                    self.assertEqual((x.grad.shape, x.grad.dtype, x.grad.device), ((), dtype, self.device))
                    self.assertFalse(x.grad.requires_grad)
                    self.assertIsNone(output.grad)
                self.assertEqual(scalar_polynomial.last_report["phase"], self.phase)
        supported = 3 if self.supports_bf16 else 2
        self.assertEqual(scalar_polynomial.cache_info()["compilations"], supported)
        self.assertEqual(scalar_polynomial.cache_info()["cache_hits"], supported)

    def test_gradient_matches_finite_difference(self):
        @ft.jit
        def cubic(x):
            return x * x * x - 2 * x

        step = 1 / 64
        for value in (-1.5, 0.5, 2.0):
            with self.subTest(value=value):
                x = ft.tensor(value, requires_grad=True)
                cubic(x).backward()
                high = cubic(ft.tensor(value + step)).tolist()
                low = cubic(ft.tensor(value - step)).tolist()
                self.assertAlmostEqual(x.grad.tolist(), (high - low) / (2 * step), delta=0.002)

    def test_multiinput_multioutput_backward_accumulates_explicit_seeds(self):
        @ft.jit
        def product_and_double(x, y):
            return x * y, x + x

        x = ft.tensor([-2, 0.5, 3], requires_grad=True)
        y = ft.tensor([4, -2, 5], requires_grad=True)
        product, doubled = product_and_double(x, y)
        product.backward(ft.tensor([1, 2, 0.5]), retain_graph=True)
        doubled.backward(ft.tensor([3, -1, 2]))
        self.assertEqual(x.grad.tolist(), [10, -6, 6.5])
        self.assertEqual(y.grad.tolist(), [-2, 1, 1.5])
        self.assertIsNone(product.grad)
        self.assertIsNone(doubled.grad)

    def test_duplicate_outputs_and_repeated_input_accumulate_all_paths(self):
        @ft.jit
        def duplicated(x):
            value = ft.square(x)
            return value, value

        x = ft.tensor([1, -2, 3], requires_grad=True)
        first, second = duplicated(x)
        first.backward(ft.tensor([1, 2, 3]), retain_graph=True)
        second.backward(ft.tensor([2, 1, -1]))
        self.assertEqual(x.grad.tolist(), [6, -12, 12])

        @ft.jit
        def multiply(x, y):
            return x * y

        shared = ft.tensor(3, requires_grad=True)
        multiply(shared, shared).backward()
        self.assertEqual(shared.grad.tolist(), 6)

    def test_mixed_precision_backward_casts_to_leaf_type(self):
        @ft.jit
        def widened_square(x):
            wide = ft.cast(x, "f32")
            return ft.square(wide).astype(x.dtype)

        for dtype in ("f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        output = widened_square(ft.tensor([-2, 0.5, 3], dtype=dtype, requires_grad=True))
                        output.backward(ft.tensor([1, 2, 0.5], dtype=dtype))
                    continue
                x = ft.tensor([-2, 0.5, 3], dtype=dtype, requires_grad=True)
                widened_square(x).backward(ft.tensor([1, 2, 0.5], dtype=dtype))
                self.assertEqual(x.grad.tolist(), [-4, 2, 3])
                self.assertEqual(x.grad.dtype, dtype)

    def test_builtins_are_traceable_and_like_constants_ignore_data(self):
        @ft.jit
        def builtins(x, y):
            return ft.add(ft.multiply(x, y), ft.negative(ft.subtract(x, y)))

        self.assertEqual(builtins(ft.tensor([2, -3]), ft.tensor([4, 5])).tolist(), [10, -7])
        self.assertIn("fuga.add", builtins.last_fuga_ir)
        self.assertIn("fuga.mul", builtins.last_fuga_ir)

        @ft.jit
        def constants(x):
            return ft.zeros_like(x), ft.ones_like(x)

        zero, one = constants(ft.tensor([float("nan"), float("inf"), -0.0]))
        self.assertEqual(zero.tolist(), [0, 0, 0])
        self.assertEqual(one.tolist(), [1, 1, 1])

    def test_tape_connects_jit_calls_and_invalidated_nested_definitions(self):
        factor = 2

        @ft.jit
        def inner(x):
            return x * factor

        @ft.jit
        def outer(x):
            return inner(x) * x

        self.assertEqual(inner(ft.tensor(2)).tolist(), 4)
        x = ft.tensor(2, requires_grad=True)
        loss = outer(x) + x
        loss.backward()
        self.assertEqual((loss.tolist(), x.grad.tolist()), (10, 9))
        factor = 3
        inner.clear_cache()
        changed = ft.tensor(2, requires_grad=True)
        changed_loss = outer(changed) + changed
        changed_loss.backward()
        self.assertEqual((changed_loss.tolist(), changed.grad.tolist()), (14, 13))
        self.assertEqual(outer.cache_info()["compilations"], 2)
        self.assertIs(ft.jit(inner), inner)

    def test_fun_forward_and_backward_are_visible_to_outer_computation(self):
        class Square(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @ft.jit
            def backward(self, ctx, dy):
                (x,) = ctx.saved_tensors
                return (2 * x * dy,)

        square = Square()

        @ft.jit
        def loss(x):
            return square(x + 1) + x

        for value, expected in ((2, 7), (3, 9)):
            x = ft.tensor(value, requires_grad=True)
            loss(x).backward()
            self.assertEqual(x.grad.tolist(), expected)
        self.assertEqual(loss.cache_info()["compilations"], 1)

    def test_fun_multioutput_seeds_and_nonfloating_input_slots(self):
        class ProductPair(ft.fun):
            @ft.jit
            def forward(self, ctx, x, y, mask):
                ctx.save_for_backward(x, y)
                return x * y, x + x + mask.astype(x.dtype)

            @ft.jit
            def backward(self, ctx, dy):
                x, y = ctx.saved_tensors
                first, second = dy
                return first * y + second * 2, first * x

        x, y = ft.tensor([2, 3], requires_grad=True), ft.tensor([4, 5], requires_grad=True)
        first, second = ProductPair()(x, y, ft.tensor([True, False], dtype="bool"))
        first.backward(ft.tensor([1, 2]), retain_graph=True)
        second.backward(ft.tensor([3, 4]))
        self.assertEqual(x.grad.tolist(), [10, 18])
        self.assertEqual(y.grad.tolist(), [2, 6])

    def test_leaf_accumulation_zero_grad_detach_and_no_grad(self):
        x = ft.tensor(3, requires_grad=True)
        ft.square(x).backward()
        ft.square(x).backward()
        self.assertEqual(x.grad.tolist(), 12)
        x.zero_grad(set_to_none=False)
        self.assertEqual(x.grad.tolist(), 0)
        x.zero_grad()
        self.assertIsNone(x.grad)
        detached = x.detach()
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.tolist(), 3)
        with ft.no_grad():
            result = ft.square(x)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad)
        ft.square(x).backward()
        self.assertEqual(x.grad.tolist(), 6)


class HostAutodiffTests(AutodiffGraphs, unittest.TestCase):
    device, phase = "cpu", "host_object"

    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ.get("ASK_OPT")
        if not cls.driver:
            raise unittest.SkipTest("set ASK_OPT to execute real host autodiff tests")
        cls.supports_bf16 = Compiler(cls.driver).capabilities()["host_bf16_hardware_supported"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-backward-host-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device=self.device, ask_opt=self.driver, cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1", "enable real CUDA backward explicitly")
class CUDAAutodiffTests(AutodiffGraphs, unittest.TestCase):
    device, phase = "cuda", "cuda_llvm"

    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ["ASK_OPT"]
        with ExecutionSession() as probe:
            cls.supports_bf16 = probe.capability[0] >= 8

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-backward-cuda-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device=self.device, ask_opt=self.driver, cache_dir=self.directory.name,
                               compilers={}, llc=os.environ["FUGATENSOR_LLC"],
                               ptxas=os.environ["FUGATENSOR_PTXAS"])
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


class AutodiffContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-backward-contract-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cpu", ask_opt=os.environ.get("ASK_OPT"), cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)

    def test_fun_forward_and_backward_require_jit(self):
        with self.assertRaisesRegex(TypeError, "forward.*jit|@ft.jit|@jit"):
            class MissingForwardJIT(ft.fun):
                def forward(self, ctx, x):
                    return x
        with self.assertRaisesRegex(TypeError, "backward.*jit|@ft.jit|@jit"):
            class MissingBackwardJIT(ft.fun):
                @ft.jit
                def forward(self, ctx, x):
                    return x

                def backward(self, ctx, dy):
                    return (dy,)
        self.assertFalse(hasattr(ft, "grad"))
        self.assertFalse(hasattr(ft, "vjp"))

    def test_inference_does_not_invoke_backward_and_missing_rule_is_preflighted(self):
        class InferenceOnly(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return x * 2

        x = ft.tensor(3, requires_grad=True)
        output = InferenceOnly()(x)
        self.assertEqual(output.tolist(), 6)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("reverse compiled before validation")), \
             patch.object(self.engine, "_execute", side_effect=AssertionError("reverse executed before validation")):
            with self.assertRaisesRegex(ValueError, "backward|reverse|VJP"):
                output.backward()
        self.assertIsNone(x.grad)

        class ForbiddenBackward(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return x + 1

            @ft.jit
            def backward(self, ctx, dy):
                raise AssertionError("inference invoked backward")
        self.assertEqual(ForbiddenBackward()(ft.tensor(3)).tolist(), 4)

    def test_backward_seed_contracts_fail_before_reverse_execution(self):
        x = ft.tensor([1, 2], requires_grad=True)
        output = ft.square(x)
        with patch.object(self.engine, "_execute", side_effect=AssertionError("reverse executed before validation")):
            with self.assertRaisesRegex((TypeError, ValueError), "scalar|gradient|seed|shape"):
                output.backward()
            for seed in (ft.tensor(1), ft.tensor([1, 1], dtype="f16")):
                with self.subTest(seed_shape=seed.shape, seed_dtype=seed.dtype):
                    with self.assertRaisesRegex((TypeError, ValueError), "gradient|seed|shape|dtype|type"):
                        output.backward(seed)
        self.assertIsNone(x.grad)
        with self.assertRaisesRegex((TypeError, ValueError), "float|requires_grad|differentiable"):
            ft.tensor(2, dtype="i32", requires_grad=True)

    def test_custom_backward_validates_tuple_arity_dtype_and_shape(self):
        for mode in ("not_tuple", "arity", "dtype"):
            with self.subTest(mode=mode):
                class Invalid(ft.fun):
                    @ft.jit
                    def forward(self, ctx, x):
                        return x

                    @ft.jit
                    def backward(self, ctx, dy):
                        if mode == "not_tuple":
                            return dy
                        if mode == "arity":
                            return dy, dy
                        return (dy.astype("f16"),)
                x = ft.tensor(2, requires_grad=True)
                output = Invalid()(x)
                with patch.object(self.engine, "_execute", side_effect=AssertionError("reverse executed before validation")):
                    with self.assertRaisesRegex((TypeError, ValueError), "backward|gradient|tuple|dtype|type"):
                        output.backward()
                self.assertIsNone(x.grad)

        class WrongShape(ft.fun):
            @ft.jit
            def forward(self, ctx, x, y):
                ctx.save_for_backward(y)
                return x

            @ft.jit
            def backward(self, ctx, dy):
                (y,) = ctx.saved_tensors
                return y, ft.zeros_like(y)
        x, y = ft.tensor([1, 2], requires_grad=True), ft.tensor(3, requires_grad=True)
        output = WrongShape()(x, y)
        with patch.object(self.engine, "_execute", side_effect=AssertionError("reverse executed before validation")):
            with self.assertRaisesRegex((TypeError, ValueError), "shape|gradient|type"):
                output.backward(ft.tensor([1, 1]))

    def test_default_releases_graph_and_retain_graph_allows_reuse(self):
        x = ft.tensor(3, requires_grad=True)
        output = ft.square(x)
        output.backward(retain_graph=True)
        output.backward()
        self.assertEqual(x.grad.tolist(), 12)
        with self.assertRaisesRegex((RuntimeError, ValueError), "freed|retain_graph|backward|released"):
            output.backward()

    def test_detached_and_no_grad_outputs_cannot_start_backward(self):
        x = ft.tensor(2, requires_grad=True)
        with ft.no_grad():
            no_history = ft.square(x)
        for value in (x.detach(), no_history):
            with self.subTest(value=value.tolist()):
                with self.assertRaisesRegex((RuntimeError, ValueError), "grad|history|differentia"):
                    value.backward()
        self.assertIsNone(x.grad)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_TRITON") == "1", "enable real Triton backward explicitly")
class TritonAutodiffTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).parent / "fixtures/user_scale.py"
        self.scale = ft.triton.kernel(self.source, entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,))
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-backward-triton-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cuda", ask_opt=os.environ["ASK_OPT"], cache_dir=self.directory.name,
            compilers={"triton": CompilerEnvironment(python=os.environ["FUGATENSOR_TRITON_PYTHON"],
                worker=os.environ["FUGATENSOR_TRITON_WORKER"])},
            llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"])
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)

    def test_user_kernel_forward_and_backward_compile_through_ask(self):
        scale = self.scale
        class UserScale(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return scale(x)

            @ft.jit
            def backward(self, ctx, dy):
                return (scale(dy),)
        op = UserScale()

        @ft.jit
        def mixed(x):
            return op(x + 1) + 3

        x = ft.tensor([1, 2, 3], requires_grad=True)
        output = mixed(x)
        self.assertEqual(output.tolist(), [7, 9, 11])
        compiled = []
        original = self.engine._compile
        def record(context):
            executable = original(context)
            compiled.append(executable)
            return executable
        with patch.object(self.engine, "_compile", side_effect=record):
            output.backward(ft.tensor([1, 2, -1]))
        self.assertEqual(x.grad.tolist(), [2, 4, -2])
        reports = [mixed.last_report, *(entry.report for entry in compiled)]
        fused = [report for report in reports if report.get("phase") == "fused_llvm"]
        self.assertGreaterEqual(len(fused), 2)
        for report in fused:
            self.assertEqual(report["fusion"]["compiler"], "triton")
            self.assertEqual(report["external_requests"][0]["source"], self.source.read_text())

    def test_missing_external_rule_fails_before_reverse_execution(self):
        scale = self.scale
        @ft.jit
        def raw(x):
            return scale(x)
        x = ft.tensor(3, requires_grad=True)
        output = raw(x)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("reverse compiled before validation")), \
             patch.object(self.engine, "_execute", side_effect=AssertionError("reverse executed before validation")):
            with self.assertRaisesRegex(ValueError, "external.*(VJP|reverse|backward)|derivative.*external"):
                output.backward()
        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()
