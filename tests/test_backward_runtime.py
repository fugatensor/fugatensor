"""Execute backward histories with shared nodes and cached custom rules."""
import tempfile
import unittest
from unittest.mock import patch

import fugatensor as ft


class ForwardOnly(ft.fun):
    @ft.jit
    def forward(self, ctx, x):
        return x + 3


class RuntimeBackwardTests(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.engine = ft.Engine(device="cpu", cache_dir=self.cache.name)
        self.engine.__enter__()

    def tearDown(self):
        self.engine.__exit__(None, None, None)
        self.cache.cleanup()

    def test_shared_history_accumulates_and_reuses_backward_executables(self):
        x = ft.tensor(3.0, requires_grad=True)
        middle = x * x
        output = middle * x
        output.backward(retain_graph=True)
        self.assertEqual(x.grad.tolist(), 27.0)
        self.assertIsNone(middle.grad)
        self.assertFalse(x.grad.requires_grad)
        compilations = self.engine.stats["compilations"]
        output.backward()
        self.assertEqual(x.grad.tolist(), 54.0)
        self.assertEqual(self.engine.stats["compilations"], compilations)
        with self.assertRaisesRegex(RuntimeError, "freed.*retain_graph"):
            output.backward()
        x.zero_grad(set_to_none=False)
        self.assertEqual(x.grad.tolist(), 0.0)
        x.zero_grad()
        self.assertIsNone(x.grad)

    def test_unused_result_does_not_require_its_custom_backward(self):
        @ft.jit
        def pair(x):
            return ForwardOnly()(x), x * x

        x = ft.tensor(3.0, requires_grad=True)
        unsupported, square = pair(x)
        square.backward(retain_graph=True)
        self.assertEqual(x.grad.tolist(), 6.0)
        before = self.engine.stats.copy()
        with patch.object(self.engine, "_compile", side_effect=AssertionError("backward compiled before rule validation")):
            with self.assertRaisesRegex(ValueError, "backward.*required"):
                unsupported.backward()
        self.assertEqual(x.grad.tolist(), 6.0)
        self.assertEqual(self.engine.stats, before)

    def test_unused_input_does_not_visit_an_unrelated_history(self):
        @ft.jit
        def ignore_second(x, unused):
            return x * x

        x = ft.tensor(3.0, requires_grad=True)
        unrelated = ForwardOnly()(x)
        output = ignore_second(x, unrelated)
        output.backward()
        self.assertEqual(x.grad.tolist(), 6.0)
        self.assertIsNone(unrelated.grad)

    def test_duplicate_arguments_and_multiple_output_ports_accumulate_by_identity(self):
        @ft.jit
        def pair(x, y):
            return x * y, x + y

        x = ft.tensor(3.0, requires_grad=True)
        square, doubled = pair(x, x)
        (square + doubled).backward()
        self.assertEqual(x.grad.tolist(), 8.0)

    def test_nondifferentiable_result_ports_do_not_enter_seed_abi(self):
        @ft.jit
        def pair(x):
            return x.astype("i32"), x * x

        x = ft.tensor([2, 3], requires_grad=True)
        integer, square = pair(x)
        self.assertFalse(integer.requires_grad)
        square.backward(ft.tensor([0.5, 2]))
        self.assertEqual(x.grad.tolist(), [2.0, 12.0])

    def test_cached_custom_backward_rejects_changed_static_configuration(self):
        class Scale(ft.fun):
            def __init__(self):
                self.factor = 2

            @ft.jit
            def forward(self, ctx, x):
                return x * self.factor

            @ft.jit
            def backward(self, ctx, dy):
                return (dy * self.factor,)

        scale = Scale()
        x = ft.tensor(3.0, requires_grad=True)
        output = scale(x)
        output.backward(retain_graph=True)
        self.assertEqual(x.grad.tolist(), 2.0)
        scale.factor = 3
        before = self.engine.stats.copy()
        with self.assertRaisesRegex(ValueError, "definition changed"):
            output.backward()
        self.assertEqual(self.engine.stats, before)
        self.assertEqual(x.grad.tolist(), 2.0)
        scale(x).backward()
        self.assertEqual(x.grad.tolist(), 5.0)

    def test_freeing_shared_history_is_an_error_on_later_branch_backward(self):
        x = ft.tensor(2.0, requires_grad=True)
        shared = x * x
        first, second = shared * 2, shared * 3
        first.backward()
        self.assertEqual(x.grad.tolist(), 8.0)
        before = self.engine.stats.copy()
        with self.assertRaisesRegex(RuntimeError, "freed"):
            second.backward()
        self.assertEqual(self.engine.stats, before)
        self.assertEqual(x.grad.tolist(), 8.0)

    def test_no_grad_detach_and_constant_outputs_do_not_acquire_history(self):
        x = ft.tensor(3.0, requires_grad=True)
        with ft.no_grad():
            with ft.no_grad():
                disabled = x * x
            disabled_again = x + 1
        self.assertFalse(disabled.requires_grad)
        self.assertFalse(disabled_again.requires_grad)
        detached = x.detach()
        self.assertIs(detached.storage, x.storage)
        self.assertFalse(detached.requires_grad)
        self.assertFalse(ft.ones_like(x).requires_grad)
        (x * x).backward()
        self.assertEqual(x.grad.tolist(), 6.0)

    def test_no_grad_inside_jit_is_rejected_before_execution(self):
        @ft.jit
        def invalid(x):
            with ft.no_grad():
                value = x * x
            return value + x

        x = ft.tensor(3.0, requires_grad=True)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("invalid graph compiled")):
            with self.assertRaisesRegex(TypeError, "while tracing"):
                invalid(x)

    def test_leaf_backward_and_seed_validation_do_not_need_a_forward_graph(self):
        x = ft.tensor(3.0, requires_grad=True)
        x.backward()
        x.backward(ft.tensor(2.0))
        self.assertEqual(x.grad.tolist(), 3.0)
        self.assertTrue(x.is_leaf)
        vector = ft.tensor([1, 2], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "explicit gradient"):
            vector.backward()
        with self.assertRaisesRegex(TypeError, "shape and dtype"):
            vector.backward(ft.tensor([1, 1], dtype="f16"))
        with self.assertRaisesRegex(TypeError, "create_graph must be bool"):
            x.backward(create_graph=1)


if __name__ == "__main__":
    unittest.main()
