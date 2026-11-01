"""Nested custom rules stay visible when their backward graph is transformed."""
import os
import tempfile
import unittest
from unittest.mock import patch

import fugatensor as ft
from ask import Compiler, CompilerError


class NestedRuleCases:
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.engine = ft.Engine(device=self.device, cache_dir=self.cache.name)
        self.engine.__enter__()

    def tearDown(self):
        self.engine.__exit__(None, None, None)
        self.cache.cleanup()

    def test_fun_called_by_backward_keeps_its_custom_derivative(self):
        class Identity(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return x

            @ft.jit
            def backward(self, ctx, dy):
                return (dy * 5,)

        inner = Identity()

        class Square(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @ft.jit
            def backward(self, ctx, dy):
                (x,) = ctx.saved_tensors
                return (inner(dy * x * 2),)

        x = ft.tensor(3.0, requires_grad=True)
        Square()(x).backward(create_graph=True)
        first = x.grad
        self.assertEqual(first.tolist(), 6.0)
        x.zero_grad()
        first.backward()
        # The inner rule contributes a factor of five. Expanding its forward
        # identity before the next AD pass would incorrectly return two.
        self.assertEqual(x.grad.tolist(), 10.0)

    def test_saved_intermediate_uses_nested_rule_through_three_orders(self):
        class Inner(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @ft.jit
            def backward(self, ctx, dy):
                (x,) = ctx.saved_tensors
                return (dy * x * 7,)

        inner = Inner()

        class Outer(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                ctx.save_for_backward(inner(x))
                return x * x * x

            @ft.jit
            def backward(self, ctx, dy):
                (saved,) = ctx.saved_tensors
                return (dy * saved,)

        x = ft.tensor(3.0, requires_grad=True)
        value = Outer()(x)
        for order, expected in enumerate((9.0, 21.0, 7.0)):
            value.backward(create_graph=order < 2)
            value = x.grad
            self.assertEqual(value.tolist(), expected)
            x.zero_grad()

    def test_visible_and_saved_result_cotangents_both_contribute(self):
        class Square(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @ft.jit
            def backward(self, ctx, dy):
                (x,) = ctx.saved_tensors
                return (dy * x * 3,)

        square = Square()

        @ft.jit
        def combined(x):
            return square(x) * x

        x = ft.tensor(2.0, requires_grad=True)
        value = combined(x)
        for order, expected in enumerate((16.0, 18.0, 9.0)):
            value.backward(create_graph=order < 2)
            value = x.grad
            self.assertEqual(value.tolist(), expected)
            x.zero_grad()

    def test_cached_backward_revalidates_nested_fun_configuration(self):
        class Inner(ft.fun):
            def __init__(self):
                self.factor = 2

            @ft.jit
            def forward(self, ctx, x):
                return x * self.factor

            @ft.jit
            def backward(self, ctx, dy):
                return (dy * self.factor,)

        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                inner = Inner()

                class Outer(ft.fun):
                    @ft.jit
                    def forward(self, ctx, x):
                        return x * x

                    @ft.jit
                    def backward(self, ctx, dy):
                        return (inner(dy),)

                x = ft.tensor(3.0, requires_grad=True)
                output = Outer()(x)
                output.backward(retain_graph=True, create_graph=create_graph)
                self.assertEqual(x.grad.tolist(), 2.0)
                inner.factor = 3
                before = self.engine.stats.copy()
                with patch.object(self.engine, "_execute", side_effect=AssertionError("stale rule executed")):
                    with self.assertRaisesRegex(ValueError, "definition changed"):
                        output.backward(retain_graph=True, create_graph=create_graph)
                self.assertEqual(self.engine.stats, before)
                self.assertEqual(x.grad.tolist(), 2.0)

    def test_cached_backward_revalidates_nested_jit_definition(self):
        configuration = {"factor": 2}

        @ft.jit
        def helper(x):
            return x * configuration["factor"]

        class Outer(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return x * x

            @ft.jit
            def backward(self, ctx, dy):
                return (helper(dy),)

        x = ft.tensor(3.0, requires_grad=True)
        output = Outer()(x)
        output.backward(retain_graph=True)
        self.assertEqual(x.grad.tolist(), 2.0)
        configuration["factor"] = 3
        helper.clear_cache()
        with patch.object(self.engine, "_execute", side_effect=AssertionError("stale helper executed")):
            with self.assertRaisesRegex(ValueError, "definition changed"):
                output.backward()
        self.assertEqual(x.grad.tolist(), 2.0)


class NestedRulesCPU(NestedRuleCases, unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        try:
            capabilities = Compiler(os.environ.get("ASK_OPT")).capabilities()
        except CompilerError as error:
            raise unittest.SkipTest(f"ASK native compiler is required: {error}")
        if not capabilities.get("host_codegen"):
            raise unittest.SkipTest("ASK host code generation is disabled")


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1", "enable real CUDA higher-order tests explicitly")
class NestedRulesCUDA(NestedRuleCases, unittest.TestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
