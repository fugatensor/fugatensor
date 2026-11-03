"""Functional reverse-mode AD, explicit stopped paths, and deferred execution."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import fugatensor as ft
from tests._helpers import EngineTestCase, requires_cuda


class ForwardOnly(ft.fun):
    def forward(self, x):
        return x + 3


class SharedAutodiffMixin:
    def test_forward_and_two_derivative_orders_share_one_execution(self):
        x = ft.tensor(2.0)
        before = dict(self.engine.stats)
        output = x * x * x + x
        first = ft.grad(output, x)
        second = ft.grad(first, x)
        self.assertEqual(self.engine.stats["compilations"], before["compilations"])
        self.assertEqual(self.engine.stats["executions"], before["executions"])
        ft.evaluate(output, first, second)
        self.assertEqual((output.tolist(), first.tolist(), second.tolist()), (10.0, 13.0, 12.0))
        self.assertEqual(self.engine.stats["executions"], before["executions"] + 1)

    def test_stopped_paths_inside_fun_forward_and_vjp_compose(self):
        class StoppedSquare(ft.fun):
            def forward(self, x):
                return ft.stop_gradient(x * x) + x

        class StoppedCoefficient(ft.fun):
            def forward(self, x):
                return x.detach() * x

        class StoppedRule(ft.fun):
            def forward(self, x):
                return x * x

            def vjp(self, primals, outputs, cotangents):
                return (ft.stop_gradient(primals[0]) * cotangents[0],)

        x = ft.tensor(3.0)
        for operation, value, first_value in ((StoppedSquare(), 12.0, 1.0),
                                               (StoppedCoefficient(), 9.0, 3.0),
                                               (StoppedRule(), 9.0, 3.0)):
            with self.subTest(operation=type(operation).__name__):
                output = operation(x)
                first = ft.grad(output, x)
                second = ft.grad(first, x)
                ft.evaluate(output, first, second)
                self.assertEqual((output.tolist(), first.tolist(), second.tolist()),
                                 (value, first_value, 0.0))

    def test_intermediate_targets_and_repeated_ports_keep_all_derivative_paths(self):
        x = ft.tensor(2.0)
        intermediate = x * x
        output = intermediate * intermediate
        dx, dt = ft.grad(output, (x, intermediate))
        self.assertEqual((dx.tolist(), dt.tolist()), (32.0, 8.0))
        dx, dt = ft.grad((output, intermediate), (x, intermediate),
                         seed=(ft.tensor(2.0), ft.tensor(3.0)))
        self.assertEqual((dx.tolist(), dt.tolist()), (76.0, 19.0))
        repeated = ft.grad(output, (x, x))
        self.assertEqual(tuple(value.tolist() for value in repeated), (32.0, 32.0))
        self.assertEqual(ft.grad((output, output), x).tolist(), 64.0)

    def test_reshape_gradient_and_higher_order_preserve_input_shape(self):
        x = ft.tensor([1, 2, 3, 4])
        shaped = x.reshape(2, 2)
        first = ft.grad(shaped * shaped, x, seed=ft.ones(2, 2))
        self.assertEqual(first.shape, (4,))
        self.assertEqual(first.tolist(), [2, 4, 6, 8])
        second = ft.grad(first, x, seed=ft.ones(4))
        self.assertEqual(second.tolist(), [2, 2, 2, 2])
        self.assertEqual(second.shape, x.shape)

    def test_dtype_objects_are_static_fun_arguments_and_configuration(self):
        class Convert(ft.fun):
            output_dtype = ft.fp16

            def forward(self, x, dtype=ft.fp32):
                assert x.dtype is ft.fp32
                value = x if dtype is ft.fp32 else x + 1
                result = value.astype(self.output_dtype)
                assert result.dtype is self.output_dtype
                return result

            def vjp(self, primals, outputs, cotangents):
                (x,), (dy,) = primals, cotangents
                return (dy.astype(x.dtype),)

        operation = Convert()
        x = ft.tensor(2.0)
        output = operation(x)
        self.assertIs(output.dtype, ft.fp16)
        self.assertEqual(output.tolist(), 2.0)
        derivative = ft.grad(output, x)
        self.assertIs(derivative.dtype, ft.fp32)
        self.assertEqual(derivative.tolist(), 1.0)
        self.assertEqual(operation(x, "f32").tolist(), 3.0)
        self.assertEqual(operation(x, ft.fp32).tolist(), 2.0)
        operation.output_dtype = ft.fp32
        widened = operation(x, ft.fp32)
        self.assertIs(widened.dtype, ft.fp32)
        self.assertEqual(widened.tolist(), 2.0)

    def test_finite_difference_multiport_and_mixed_precision(self):
        def cubic(x):
            return x * x * x - 2 * x

        step = 1 / 64
        x = ft.tensor(-1.5)
        derivative = ft.grad(cubic(x), x)
        fd = (cubic(ft.tensor(-1.5 + step)).tolist() - cubic(ft.tensor(-1.5 - step)).tolist()) / (2 * step)
        self.assertAlmostEqual(derivative.tolist(), fd, delta=0.002)

        def pair(a, b):
            return a * b, a + a

        a, b = ft.tensor([-2, 0.5, 3]), ft.tensor([4, -2, 5])
        da, db = ft.grad(pair(a, b), (a, b), seed=(ft.tensor([1, 2, 0.5]), ft.tensor([3, -1, 2])))
        self.assertEqual((da.tolist(), db.tolist()), ([10, -6, 6.5], [-2, 1, 1.5]))
        shared = ft.tensor(3.0)
        s0, s1 = pair(shared, shared)
        self.assertEqual(ft.grad(s0 + s1, shared).tolist(), 8.0)

        wx = ft.tensor([-2, 0.5, 3], dtype="f16")
        widened = ft.square(ft.cast(wx, "f32")).astype(wx.dtype)
        dw = ft.grad(widened, wx, seed=ft.tensor([1, 2, 0.5], dtype="f16"))
        self.assertEqual((dw.tolist(), dw.dtype), ([-4, 2, 3], "f16"))

    def test_custom_fun_multioutput_gradients_and_repeated_transform(self):
        def builtins(x, y):
            return ft.add(ft.multiply(x, y), ft.negative(ft.subtract(x, y))), ft.zeros_like(x), ft.ones_like(x)

        res, z, o = builtins(ft.tensor([2, -3]), ft.tensor([4, 5]))
        self.assertEqual((res.tolist(), z.tolist(), o.tolist()), ([10, -7], [0, 0], [1, 1]))

        class ProductPair(ft.fun):
            def forward(self, x, y, mask):
                return x * y, x + x + mask.astype(x.dtype)

            def vjp(self, primals, outputs, cotangents):
                x, y, _ = primals
                d0, d1 = cotangents
                return d0 * y + d1 * 2, d0 * x

        x, y = ft.tensor([2, 3]), ft.tensor([4, 5])
        outputs = ProductPair()(x, y, ft.tensor([True, False], dtype="bool"))
        dx, dy = ft.grad(outputs, [x, y], seed=(ft.tensor([1, 2]), ft.tensor([3, 4])))
        self.assertEqual((dx.tolist(), dy.tolist()), ([10, 18], [2, 6]))

        scalar = ft.tensor(3.0)
        out = ft.square(scalar)
        before = dict(self.engine.stats)
        first, again = ft.grad(out, scalar), ft.grad(out, scalar)
        self.assertEqual(self.engine.stats["executions"], before["executions"])
        self.assertEqual(self.engine.stats["compilations"], before["compilations"])
        ft.evaluate(out, first, again)
        self.assertEqual(self.engine.stats["executions"], before["executions"] + 1)
        self.assertEqual((out.tolist(), first.tolist(), again.tolist()), (9.0, 6.0, 6.0))
        self.assertEqual(ft.grad(out, scalar).tolist(), 6.0)
        self.assertEqual(ft.grad(first, scalar).tolist(), 2.0)

    def test_evaluation_preserves_source_and_stop_gradient_cuts_only_selected_path(self):
        x = ft.tensor(3.0)
        out = x * x
        self.assertEqual(out.tolist(), 9.0)
        self.assertEqual(ft.grad(out, x).tolist(), 6.0)
        for stop in (ft.stop_gradient, lambda tensor: tensor.detach()):
            with self.subTest(stop=stop):
                pending = x * x + 1
                before = self.engine.stats["executions"]
                detached = stop(pending)
                self.assertEqual(self.engine.stats["executions"], before)
                self.assertEqual(ft.grad(detached, x).tolist(), 0.0)
                mixed = detached * x
                derivative = ft.grad(mixed, x)
                self.assertEqual(derivative.tolist(), 10.0)
                self.assertEqual(ft.grad(derivative, x).tolist(), 0.0)
                self.assertEqual(pending.tolist(), detached.tolist())
                self.assertEqual(pending.storage.data_ptr, detached.storage.data_ptr)
        other = ft.tensor(7.0)
        self.assertEqual(ft.grad(out, other).tolist(), 0.0)

    def test_custom_definition_change_rejects_pending_forward_and_retraces_new_call(self):
        class Scale(ft.fun):
            def __init__(self):
                self.factor = 2

            def forward(self, value):
                return value * self.factor

            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * self.factor,)

        scale = Scale()
        x = ft.tensor(3.0)
        pending = scale(x)
        scale.factor = 5
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled stale definition")):
            with self.assertRaisesRegex((ValueError, RuntimeError), "definition changed"):
                pending.tolist()
        fresh = scale(x)
        self.assertEqual(fresh.tolist(), 15.0)
        self.assertEqual(ft.grad(fresh, x).tolist(), 5.0)

    def test_failed_gradient_execution_preserves_other_results_and_can_retry(self):
        x = ft.tensor(3.0)
        existing = ft.grad(x * x, x)
        self.assertEqual(existing.tolist(), 6.0)
        out = x * x + 1
        derivative = ft.grad(out, x)
        with patch.object(self.engine, "_execute", side_effect=RuntimeError("gradient execution failed")):
            with self.assertRaisesRegex(RuntimeError, "gradient execution failed"):
                derivative.tolist()
        self.assertEqual(existing.tolist(), 6.0)
        self.assertEqual(derivative.tolist(), 6.0)


class HostAutodiffTests(SharedAutodiffMixin, EngineTestCase):
    device = "cpu"

    def test_unused_nested_fun_definition_does_not_invalidate_selected_gradients(self):
        class Inner(ft.fun):
            factor = 2

            def forward(self, x):
                return x * self.factor

            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * self.factor,)

        inner = Inner()

        class Pair(ft.fun):
            def forward(self, x):
                return x * x, inner(x)

        x = ft.tensor(3.0)
        selected, other = Pair()(x)
        first = ft.grad(selected, x)
        active = ft.grad(other, x)
        class KeepFirst(ft.fun):
            def forward(self, a, b):
                return a

        class IgnoreOperand(ft.fun):
            def forward(self, value):
                return KeepFirst()(value * value, inner(value))

        ignored = IgnoreOperand()(x)
        inner.factor = 4
        self.assertEqual(first.tolist(), 6.0)
        self.assertEqual(ft.grad(selected, x).tolist(), 6.0)
        self.assertEqual(ft.grad(first, x).tolist(), 2.0)
        self.assertEqual(ft.grad(ignored, x).tolist(), 6.0)
        with self.assertRaisesRegex((ValueError, RuntimeError), "definition changed"):
            ft.grad(other, x)
        with self.assertRaisesRegex((ValueError, RuntimeError), "definition changed"):
            active.tolist()

    def test_stopped_opaque_fun_needs_no_external_derivative_or_execution(self):
        spec = ft.ir.ExternalSpec(
            compiler="opaque", source="opaque source", entry="kernel",
            signature={"x": "ptr", "y": "ptr"},
            source_bindings={"x": "input:0", "y": "output:0"},
        )

        class StoppedOpaque(ft.fun):
            def forward(self, x):
                value = x.context.wrap(x.context.graph.external(x.value, spec=spec))
                return ft.stop_gradient(value)

        x = ft.tensor(3.0)
        output = StoppedOpaque()(x)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled opaque forward")):
            derivative = ft.grad(output, x)
        self.assertEqual(derivative.tolist(), 0.0)
        self.assertNotIn("fuga.external", self.engine.last_fuga_ir or "")
        self.assertIsNone(output._storage)

    def test_automatic_fun_rules_seed_guards_and_deep_chain(self):
        self.assertTrue(callable(ft.grad))
        x = ft.tensor(3.0)
        self.assertEqual(ft.grad(ForwardOnly()(x), x).tolist(), 1.0)
        self.assertEqual(ft.grad((x * x, x * x), x).tolist(), 12.0)
        self.assertIsInstance(ft.grad(x * x, [x]), tuple)

        for bad_dtype, value in (("bool", True), ("i32", 2), ("i64", 2)):
            tensor = ft.tensor(value, dtype=bad_dtype)
            with self.assertRaisesRegex((TypeError, ValueError), "float|dtype"):
                ft.grad(tensor, tensor)

        vx = ft.tensor([2, 3])
        vy = vx * vx
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
            with self.assertRaisesRegex(ValueError, "explicit|seed|scalar"):
                ft.grad(vy, vx)
            for bad_seed in (ft.tensor(1), ft.tensor([1, 1], dtype="f16"), [1, 1]):
                with self.assertRaises((TypeError, ValueError)):
                    ft.grad(vy, vx, seed=bad_seed)
            for bad_seed in (ft.ones(2), (ft.ones(2),), (ft.ones(2), ft.tensor(1))):
                with self.assertRaises((TypeError, ValueError)):
                    ft.grad((vy, vy), vx, seed=bad_seed)
            with self.assertRaises((TypeError, ValueError)):
                ft.grad(vy, [])

        class Scale(ft.fun):
            def __init__(self):
                self.factor = 2
            def forward(self, t):
                return t * self.factor
            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * self.factor,)

        scale = Scale()
        sout = scale(x)
        self.assertEqual(ft.grad(sout, x).tolist(), 2.0)
        scale.factor = 5
        with self.assertRaisesRegex(ValueError, "definition changed"):
            ft.grad(sout, x)

        chain_x = ft.tensor(0.0)
        cur = chain_x
        for _ in range(1100):
            cur = cur + 1
        self.assertEqual(ft.grad(cur, chain_x).tolist(), 1.0)


@requires_cuda
class CUDAAutodiffTests(SharedAutodiffMixin, EngineTestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
