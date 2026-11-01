"""Higher-order backward composes visible Fuga primitives on both devices."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ask import Compiler, CompilerEnvironment, CompilerError, ExecutionSession
from ask.cuda import CUDAError
import fugatensor as ft


class HigherOrderGraphs:
    def test_quartic_first_second_and_third_derivatives_across_precisions(self):
        @ft.jit
        def quartic(x):
            squared = x * x
            return squared * squared

        for dtype in ("f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        quartic(ft.tensor(2, dtype=dtype, requires_grad=True)).backward(create_graph=True)
                    continue
                x = ft.tensor(2, dtype=dtype, requires_grad=True)
                quartic(x).backward(create_graph=True)
                first = x.grad
                self.assertEqual(first.tolist(), 32)
                self.assertTrue(first.requires_grad)
                x.zero_grad()
                first.backward(create_graph=True)
                second = x.grad
                self.assertEqual(second.tolist(), 48)
                self.assertTrue(second.requires_grad)
                x.zero_grad()
                second.backward()
                self.assertEqual(x.grad.tolist(), 48)
                self.assertEqual(x.grad.dtype, dtype)
                self.assertFalse(x.grad.requires_grad)

    def test_separate_jit_executions_keep_higher_order_connections(self):
        @ft.jit
        def square(x):
            return x * x

        x = ft.tensor(2, requires_grad=True)
        square(square(x)).backward(create_graph=True)
        first = x.grad
        self.assertEqual(first.tolist(), 32)
        x.zero_grad()
        first.backward()
        self.assertEqual(x.grad.tolist(), 48)

    def test_runtime_selected_branch_and_loop_support_higher_order(self):
        @ft.jit
        def selected(x, flag, count):
            if flag:
                while count:
                    x = x * x
                    count = count - 1
                return x
            return x * x

        for flag, expected_first, expected_second in ((True, 32, 48), (False, 4, 2)):
            with self.subTest(flag=flag):
                x = ft.tensor(2, requires_grad=True)
                output = selected(x, ft.tensor(flag, dtype="bool"), ft.tensor(2, dtype="i32"))
                output.backward(create_graph=True)
                first = x.grad
                self.assertEqual(first.tolist(), expected_first)
                x.zero_grad()
                first.backward()
                self.assertEqual(x.grad.tolist(), expected_second)
                if flag:
                    self.assertGreater(len(selected.last_segments), 1)
                else:
                    self.assertEqual(len(selected.last_segments), 1)

    def test_pure_fun_saved_values_stay_connected(self):
        class Square(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                ctx.save_for_backward(x)
                return ft.square(x)

            @ft.jit
            def backward(self, ctx, dy):
                (x,) = ctx.saved_tensors
                return (2 * x * dy,)

        x = ft.tensor(3, requires_grad=True)
        Square()(x).backward(create_graph=True)
        first = x.grad
        self.assertEqual(first.tolist(), 6)
        self.assertTrue(first.requires_grad)
        x.zero_grad()
        first.backward()
        self.assertEqual(x.grad.tolist(), 2)

    def test_explicit_gradient_tensor_participates_in_derivative_graph(self):
        x = ft.tensor([2, 3], requires_grad=True)
        seed = ft.tensor([5, 7], requires_grad=True)
        ft.square(x).backward(seed, create_graph=True)
        first = x.grad
        self.assertEqual(first.tolist(), [20, 42])
        self.assertIsNone(seed.grad)
        x.zero_grad()
        first.backward(ft.tensor([1, 2]))
        self.assertEqual(x.grad.tolist(), [10, 28])
        self.assertEqual(seed.grad.tolist(), [4, 12])

    def test_accumulation_with_create_graph_retains_both_derivative_paths(self):
        @ft.jit
        def cubic(x):
            return x * x * x

        x = ft.tensor(2, requires_grad=True)
        ft.square(x).backward(create_graph=True)
        cubic(x).backward(create_graph=True)
        combined = x.grad
        self.assertEqual(combined.tolist(), 16)
        x.zero_grad()
        combined.backward()
        self.assertEqual(x.grad.tolist(), 14)

    def test_create_graph_explicitly_records_inside_outer_no_grad(self):
        x = ft.tensor(3, requires_grad=True)
        output = ft.square(x)
        with ft.no_grad():
            output.backward(create_graph=True)
            first = x.grad
            self.assertTrue(first.requires_grad)
            self.assertFalse(ft.square(x).requires_grad)
        x.zero_grad()
        first.backward()
        self.assertEqual(x.grad.tolist(), 2)


class HostHigherOrderTests(HigherOrderGraphs, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ.get("ASK_OPT")
        if not cls.driver:
            raise unittest.SkipTest("set ASK_OPT to execute real host higher-order tests")
        cls.supports_bf16 = Compiler(cls.driver).capabilities()["host_bf16_hardware_supported"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-higher-host-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cpu", ask_opt=self.driver, cache_dir=self.directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1", "enable real CUDA higher-order tests explicitly")
class CUDAHigherOrderTests(HigherOrderGraphs, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = os.environ["ASK_OPT"]
        with ExecutionSession() as probe:
            cls.supports_bf16 = probe.capability[0] >= 8

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-higher-cuda-")
        self.addCleanup(self.directory.cleanup)
        self.engine = ft.Engine(device="cuda", ask_opt=self.driver, compilers={},
            cache_dir=self.directory.name, llc=os.environ["FUGATENSOR_LLC"],
            ptxas=os.environ["FUGATENSOR_PTXAS"])
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_TRITON") == "1", "enable real opaque-external higher-order guards")
class OpaqueHigherOrderTests(unittest.TestCase):
    def test_opaque_forward_or_backward_rejects_before_any_reverse_work(self):
        scale = ft.triton.kernel(Path(__file__).parent / "fixtures/user_scale.py",
            entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,))

        class ForwardExternal(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return scale(x)

            @ft.jit
            def backward(self, ctx, dy):
                return (dy * 2,)

        class BackwardExternal(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return x * 2

            @ft.jit
            def backward(self, ctx, dy):
                return (scale(dy),)

        with tempfile.TemporaryDirectory(prefix="fuga-higher-external-") as directory, \
             ft.Engine(device="cuda", ask_opt=os.environ["ASK_OPT"], cache_dir=directory,
                compilers={"triton": CompilerEnvironment(python=os.environ["FUGATENSOR_TRITON_PYTHON"],
                    worker=os.environ["FUGATENSOR_TRITON_WORKER"])},
                llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"]) as engine:
            for type_ in (ForwardExternal, BackwardExternal):
                with self.subTest(type=type_.__name__):
                    x = ft.tensor(3, requires_grad=True)
                    output = type_()(x)
                    with patch.object(engine, "_compile", side_effect=AssertionError("reverse compiled before rejection")), \
                         patch.object(engine, "_execute", side_effect=AssertionError("reverse executed before rejection")):
                        with self.assertRaisesRegex((ValueError, NotImplementedError), "external|opaque|higher.order"):
                            output.backward(create_graph=True)
                    self.assertIsNone(x.grad)
                    # Failed higher-order preflight must leave ordinary backward usable.
                    output.backward()
                    self.assertEqual(x.grad.tolist(), 2)


if __name__ == "__main__":
    unittest.main()
