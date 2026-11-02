"""Higher-order derivatives (create_graph=True) across primitives, control flow, nested ft.fun, and Triton guards."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from fugatensor._ask import CompilerError
from fugatensor._ask.cuda import CUDAError
import fugatensor as ft
from tests._helpers import TRITON_KERNELS, EngineTestCase, requires_cuda, requires_triton


class SharedHigherOrderMixin:
    def test_quartic_three_orders_control_flow_and_seed_participation(self):
        def quartic(x):
            sq = x * x
            return sq * sq

        for dtype in ("f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        quartic(ft.tensor(2, dtype=dtype, requires_grad=True)).backward(create_graph=True)
                    continue
                x = ft.tensor(2, dtype=dtype, requires_grad=True)
                val = quartic(x)
                actual = []
                for order in range(3):
                    val.backward(create_graph=order < 2)
                    val = x.grad
                    actual.append(val.tolist())
                    if order < 2:
                        self.assertTrue(val.requires_grad)
                        x.zero_grad()
                self.assertEqual(actual, [32, 48, 48])
                self.assertFalse(val.requires_grad)

        # Runtime branch + loop + explicit differentiable seed + outer no_grad override.
        def selected(x, flag, count):
            if flag:
                while count:
                    x = x * x
                    count = count - 1
                return x
            return x * x

        for flag, exp1, exp2 in ((True, 32, 48), (False, 4, 2)):
            x = ft.tensor(2.0, requires_grad=True)
            with ft.no_grad():
                pass
            out = selected(x, ft.tensor(flag, dtype="bool"), ft.tensor(2, dtype="i32"))
            with ft.no_grad():
                out.backward(create_graph=True)
                d1 = x.grad
                self.assertTrue(d1.requires_grad)
            x.zero_grad()
            d1.backward()
            self.assertEqual((d1.tolist(), x.grad.tolist()), (exp1, exp2))

        vx = ft.tensor([2.0, 3.0], requires_grad=True)
        seed = ft.tensor([5.0, 7.0], requires_grad=True)
        ft.square(vx).backward(seed, create_graph=True)
        first = vx.grad
        vx.zero_grad()
        first.backward(ft.tensor([1.0, 2.0]))
        self.assertEqual((vx.grad.tolist(), seed.grad.tolist()), ([10.0, 28.0], [4.0, 12.0]))

    def test_nested_fun_rules_saved_intermediates_and_mutation_revalidation(self):
        class Inner(ft.fun):
            def forward(self, ctx, x):
                ctx.save_for_backward(x)
                return x * x

            def backward(self, ctx, dy):
                (x,) = ctx.saved_tensors
                return (dy * x * 7,)

        inner = Inner()

        class Outer(ft.fun):
            def forward(self, ctx, x):
                ctx.save_for_backward(inner(x))
                return x * x * x

            def backward(self, ctx, dy):
                (saved,) = ctx.saved_tensors
                return (dy * saved,)

        x = ft.tensor(3.0, requires_grad=True)
        val = Outer()(x)
        for order, expected in enumerate((9.0, 21.0, 7.0)):
            val.backward(create_graph=order < 2)
            val = x.grad
            self.assertEqual(val.tolist(), expected)
            x.zero_grad()

        # Mutating a nested custom rule invalidates cached higher-order plans.
        class MutableInner(ft.fun):
            def __init__(self):
                self.factor = 2
            def forward(self, ctx, t):
                return t * self.factor
            def backward(self, ctx, dy):
                return (dy * self.factor,)

        mi = MutableInner()

        class Wrapper(ft.fun):
            def forward(self, ctx, t):
                return t * t
            def backward(self, ctx, dy):
                return (mi(dy),)

        wx = ft.tensor(3.0, requires_grad=True)
        wout = Wrapper()(wx)
        wout.backward(retain_graph=True, create_graph=True)
        self.assertEqual(wx.grad.tolist(), 2.0)
        mi.factor = 5
        with patch.object(self.engine, "_execute", side_effect=AssertionError("stale rule executed")):
            with self.assertRaisesRegex(ValueError, "definition changed"):
                wout.backward(retain_graph=True, create_graph=True)


class HostHigherOrderTests(SharedHigherOrderMixin, EngineTestCase):
    device = "cpu"


@requires_cuda
class CUDAHigherOrderTests(SharedHigherOrderMixin, EngineTestCase):
    device = "cuda"


@requires_triton
class OpaqueExternalHigherOrderTests(EngineTestCase):
    device = "cuda"
    enable_triton = True

    def test_opaque_triton_in_forward_or_backward_is_rejected_before_reverse_work(self):
        scale = ft.triton.kernel(
            TRITON_KERNELS,
            entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,),
        )

        class ForwardExternal(ft.fun):
            def forward(self, ctx, x):
                return scale(x)
            def backward(self, ctx, dy):
                return (dy * 2,)

        class BackwardExternal(ft.fun):
            def forward(self, ctx, x):
                return x * 2
            def backward(self, ctx, dy):
                return (scale(dy),)

        for cls in (ForwardExternal, BackwardExternal):
            with self.subTest(cls=cls.__name__):
                x = ft.tensor(3.0, requires_grad=True)
                out = cls()(x)
                with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before rejection")), \
                     patch.object(self.engine, "_execute", side_effect=AssertionError("executed before rejection")):
                    with self.assertRaisesRegex((ValueError, NotImplementedError), "external|opaque|higher.order"):
                        out.backward(create_graph=True)
                self.assertIsNone(x.grad)
                # First-order backward remains usable after higher-order preflight rejection.
                out.backward()
                self.assertEqual(x.grad.tolist(), 2.0)


if __name__ == "__main__":
    unittest.main()
