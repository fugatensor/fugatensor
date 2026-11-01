"""Scalar Python control flow executes and reconnects real ASK graph segments."""
import os
import tempfile
import unittest
from unittest.mock import patch

from ask import Compiler, CompilerError
import fugatensor as ft


class ControlFlowGraphs:
    """Identical numerical and cache checks for CPU and CUDA execution."""

    def test_scalar_predicate_selects_actual_path_and_reuses_segments(self):
        calls = []

        @ft.jit
        def choose(x, flag):
            calls.append("entered")
            prefix = x * x + 1
            if flag:
                return prefix + 2
            return prefix * 3

        for flag, expected in ((True, 7), (False, 15)):
            actual = choose(ft.tensor(2), ft.tensor(flag, dtype="bool"))
            self.assertEqual(actual.tolist(), expected)
        compilations = self.engine.stats["compilations"]
        for flag, expected in ((False, 30), (True, 12)):
            actual = choose(ft.tensor(3), ft.tensor(flag, dtype="bool"))
            self.assertEqual(actual.tolist(), expected)
        self.assertEqual(self.engine.stats["compilations"], compilations)
        # A break continues the current Python frame instead of replaying it.
        self.assertEqual(calls, ["entered"] * 4)

    def test_prefix_is_executed_before_python_enters_selected_branch(self):
        observations = []

        @ft.jit
        def choose(x, flag):
            prefix = x + 1
            if flag:
                observations.append(self.engine.stats["executions"])
                return prefix * 2
            observations.append(self.engine.stats["executions"])
            return prefix + 3

        before = self.engine.stats["executions"]
        self.assertEqual(choose(ft.tensor(2), ft.tensor(True, dtype="bool")).tolist(), 6)
        self.assertGreater(observations[0], before)

    def test_branch_downloads_only_one_scalar_predicate(self):
        @ft.jit
        def choose(x, flag):
            retained = x * x
            if flag:
                return retained + 1
            return retained * 2

        x, flag = ft.tensor([2, 3, 4]), ft.tensor(True, dtype="bool")
        read_shapes = []
        original = self.engine.to_flat

        def download(value):
            read_shapes.append(value.shape)
            return original(value)

        with patch.object(self.engine, "to_flat", side_effect=download):
            result = choose(x, flag)
        self.assertEqual(read_shapes, [()])
        self.assertEqual(result.tolist(), [5, 10, 17])

    def test_while_handles_zero_iterations_and_reuses_body_at_different_trip_counts(self):
        calls = []

        @ft.jit
        def repeated(x, counter):
            calls.append("entered")
            value = x + 1
            while counter:
                value = value + 2
                counter = counter - 1
            return value

        self.assertEqual(repeated(ft.tensor(2), ft.tensor(0, dtype="i32")).tolist(), 3)
        self.assertEqual(repeated(ft.tensor(2), ft.tensor(3, dtype="i32")).tolist(), 9)
        compilations = self.engine.stats["compilations"]
        self.assertEqual(repeated(ft.tensor(5), ft.tensor(7, dtype="i32")).tolist(), 20)
        self.assertEqual(repeated(ft.tensor(5), ft.tensor(1, dtype="i32")).tolist(), 8)
        self.assertEqual(self.engine.stats["compilations"], compilations)
        self.assertEqual(calls, ["entered"] * 4)

    def test_nested_jit_and_keyword_static_arguments_survive_breaks(self):
        @ft.jit
        def inner(x, *, flag, scale=2):
            if flag:
                return x * scale
            return x + scale

        @ft.jit
        def outer(x, *, flag, bias=3):
            retained = x * x
            alias = retained
            selected = inner(x + 1, flag=flag)
            return selected + retained + alias + bias

        self.assertEqual(outer(ft.tensor(2), flag=ft.tensor(True, dtype="bool")).tolist(), 17)
        self.assertEqual(outer(ft.tensor(2), flag=ft.tensor(False, dtype="bool")).tolist(), 16)

    def test_identical_forward_ir_keeps_distinct_custom_backward_closures(self):
        def make_rule(scale):
            class Rule(ft.fun):
                @ft.jit
                def forward(self, ctx, x):
                    return x * 2

                @ft.jit
                def backward(self, ctx, dy):
                    return (dy * scale,)

            return Rule()

        left, right = make_rule(2), make_rule(3)

        @ft.jit
        def choose(x, flag):
            return left(x) if flag else right(x)

        for flag, expected in ((True, 2), (False, 3), (False, 3), (True, 2)):
            with self.subTest(flag=flag, expected=expected):
                x = ft.tensor(4, requires_grad=True)
                result = choose(x, ft.tensor(flag, dtype="bool"))
                self.assertEqual(result.tolist(), 8)
                result.backward()
                self.assertEqual(x.grad.tolist(), expected)
        self.assertEqual(choose.cache_info()["compilations"], 2)
        self.assertEqual(choose.cache_info()["cache_hits"], 2)

    def test_scalar_numeric_truth_and_non_scalar_rejection(self):
        @ft.jit
        def choose(x, condition):
            if condition:
                return x + 1
            return x + 2

        for dtype, value, expected in (("i64", 0, 5), ("i64", -2, 4),
                                       ("f32", 0.0, 5), ("f32", -0.5, 4)):
            with self.subTest(dtype=dtype, value=value):
                self.assertEqual(choose(ft.tensor(3), ft.tensor(value, dtype=dtype)).tolist(), expected)
        for condition in (ft.tensor([True], dtype="bool"), ft.tensor([True, False], dtype="bool")):
            with self.subTest(shape=condition.shape), patch.object(
                    self.engine, "_execute", side_effect=AssertionError("executed invalid condition")):
                with self.assertRaisesRegex((TypeError, ValueError), "scalar|shape|truth"):
                    choose(ft.tensor(3), condition)

    def test_eager_scalar_truth_uses_value_and_rejects_non_scalar(self):
        self.assertTrue(bool(ft.tensor(True, dtype="bool")))
        self.assertFalse(bool(ft.tensor(0, dtype="i32")))
        self.assertTrue(bool(ft.tensor(-0.5)))
        with self.assertRaisesRegex((TypeError, ValueError), "scalar|shape|truth"):
            bool(ft.tensor([1, 2]))

    def test_backward_reconnects_values_across_selected_branch_segments(self):
        @ft.jit
        def loss(x, flag):
            retained = x * x
            if flag:
                return retained * x
            return retained + x

        for flag, expected_value, expected_gradient in ((True, 8, 12), (False, 6, 5)):
            with self.subTest(flag=flag):
                x = ft.tensor(2, requires_grad=True)
                result = loss(x, ft.tensor(flag, dtype="bool"))
                self.assertEqual(result.tolist(), expected_value)
                result.backward()
                self.assertEqual(x.grad.tolist(), expected_gradient)

    def test_backward_reconnects_loop_carried_values(self):
        @ft.jit
        def repeated(x, counter):
            value = x
            while counter:
                value = value * 2
                counter = counter - 1
            return value

        for iterations in (0, 3):
            with self.subTest(iterations=iterations):
                x = ft.tensor(3, requires_grad=True)
                result = repeated(x, ft.tensor(iterations, dtype="i32"))
                self.assertEqual(result.tolist(), 3 * 2**iterations)
                result.backward()
                self.assertEqual(x.grad.tolist(), 2**iterations)

    def test_plain_static_jit_still_avoids_python_on_cache_hit(self):
        calls = []

        @ft.jit
        def static_graph(x, enabled=True):
            calls.append(enabled)
            return x + 1 if enabled else x * 2

        self.assertEqual(static_graph(ft.tensor(2)).tolist(), 3)
        self.assertEqual(static_graph(ft.tensor(3)).tolist(), 4)
        self.assertEqual(calls, [True])
        self.assertEqual(static_graph.cache_info()["compilations"], 1)
        self.assertEqual(static_graph.cache_info()["cache_hits"], 1)

    def test_fun_internal_graph_break_is_rejected_before_execution(self):
        class Conditional(ft.fun):
            @ft.jit
            def forward(self, ctx, x, flag):
                value = x + 1
                if flag:
                    return value * 2
                return value

        with patch.object(self.engine, "_execute", side_effect=AssertionError("executed inside fun trace")):
            with self.assertRaisesRegex((TypeError, ValueError, NotImplementedError),
                                        "fun|graph.break|control.flow|runtime"):
                Conditional()(ft.tensor(2), ft.tensor(True, dtype="bool"))


class HostControlFlowTests(ControlFlowGraphs, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler = Compiler(os.environ.get("ASK_OPT"))
            capabilities = cls.compiler.capabilities()
        except CompilerError as error:
            raise unittest.SkipTest(f"ASK native compiler is required: {error}")
        if not capabilities.get("host_codegen"):
            raise unittest.SkipTest("ASK host code generation is disabled")

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fuga-control-host-")
        self.addCleanup(directory.cleanup)
        self.engine = ft.Engine(device="cpu", ask_opt=self.compiler.executable,
                                cache_dir=directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1", "enable real CUDA control-flow tests explicitly")
class CUDAControlFlowTests(ControlFlowGraphs, unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fuga-control-cuda-")
        self.addCleanup(directory.cleanup)
        self.engine = ft.Engine(device="cuda", ask_opt=os.environ["ASK_OPT"], cache_dir=directory.name,
                                compilers={}, llc=os.environ["FUGATENSOR_LLC"],
                                ptxas=os.environ["FUGATENSOR_PTXAS"])
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)


if __name__ == "__main__":
    unittest.main()
