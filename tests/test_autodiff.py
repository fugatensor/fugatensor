"""Runtime reverse-mode autodiff (.backward(), .grad, no_grad, ft.fun, preflight guards) on CPU and CUDA."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ask import CompilerError
from ask.cuda import CUDAError
import fugatensor as ft
from tests._helpers import EngineTestCase, requires_cuda


class ForwardOnly(ft.fun):
    @ft.jit
    def forward(self, ctx, x):
        return x + 3


class SharedAutodiffMixin:
    def test_precisions_finite_difference_multiport_and_mixed_precision(self):
        @ft.jit
        def poly(x):
            return ft.square(x) - 3 * x + 2

        for dtype in ("f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), "bf16.*hardware|bf16.*support"):
                        poly(ft.tensor(3, dtype=dtype, requires_grad=True)).backward()
                    continue
                x = ft.tensor(4, dtype=dtype, requires_grad=True)
                out = poly(x)
                out.backward()
                self.assertEqual((x.grad.tolist(), x.grad.dtype, x.grad.device), (5, dtype, self.device))
                self.assertFalse(x.grad.requires_grad)

        # Finite-difference check on cubic.
        @ft.jit
        def cubic(x):
            return x * x * x - 2 * x

        step = 1 / 64
        x = ft.tensor(-1.5, requires_grad=True)
        cubic(x).backward()
        fd = (cubic(ft.tensor(-1.5 + step)).tolist() - cubic(ft.tensor(-1.5 - step)).tolist()) / (2 * step)
        self.assertAlmostEqual(x.grad.tolist(), fd, delta=0.002)

        # Multi-input, multi-output, duplicate ports, and mixed-precision backward.
        @ft.jit
        def pair(a, b):
            return a * b, a + a

        a = ft.tensor([-2, 0.5, 3], requires_grad=True)
        b = ft.tensor([4, -2, 5], requires_grad=True)
        p, d = pair(a, b)
        p.backward(ft.tensor([1, 2, 0.5]), retain_graph=True)
        d.backward(ft.tensor([3, -1, 2]))
        self.assertEqual((a.grad.tolist(), b.grad.tolist()), ([10, -6, 6.5], [-2, 1, 1.5]))

        shared = ft.tensor(3.0, requires_grad=True)
        s0, s1 = pair(shared, shared)
        (s0 + s1).backward()
        self.assertEqual(shared.grad.tolist(), 8.0)

        @ft.jit
        def widened_sq(v):
            return ft.square(ft.cast(v, "f32")).astype(v.dtype)

        wx = ft.tensor([-2, 0.5, 3], dtype="f16", requires_grad=True)
        widened_sq(wx).backward(ft.tensor([1, 2, 0.5], dtype="f16"))
        self.assertEqual((wx.grad.tolist(), wx.grad.dtype), ([-4, 2, 3], "f16"))

    def test_builtins_custom_fun_no_grad_zero_grad_and_retain_graph(self):
        @ft.jit
        def builtins(x, y):
            return ft.add(ft.multiply(x, y), ft.negative(ft.subtract(x, y))), ft.zeros_like(x), ft.ones_like(x)

        res, z, o = builtins(ft.tensor([2, -3]), ft.tensor([4, 5]))
        self.assertEqual((res.tolist(), z.tolist(), o.tolist()), ([10, -7], [0, 0], [1, 1]))

        class ProductPair(ft.fun):
            @ft.jit
            def forward(self, ctx, x, y, mask):
                ctx.save_for_backward(x, y)
                return x * y, x + x + mask.astype(x.dtype)

            @ft.jit
            def backward(self, ctx, dy):
                x, y = ctx.saved_tensors
                d0, d1 = dy
                return d0 * y + d1 * 2, d0 * x

        x, y = ft.tensor([2, 3], requires_grad=True), ft.tensor([4, 5], requires_grad=True)
        f0, f1 = ProductPair()(x, y, ft.tensor([True, False], dtype="bool"))
        f0.backward(ft.tensor([1, 2]), retain_graph=True)
        f1.backward(ft.tensor([3, 4]))
        self.assertEqual((x.grad.tolist(), y.grad.tolist()), ([10, 18], [2, 6]))

        # retain_graph, freed graph, zero_grad, detach, and no_grad.
        lx = ft.tensor(3.0, requires_grad=True)
        out = ft.square(lx)
        out.backward(retain_graph=True)
        out.backward()
        self.assertEqual(lx.grad.tolist(), 12.0)
        with self.assertRaisesRegex(RuntimeError, "freed"):
            out.backward()
        lx.zero_grad(set_to_none=False)
        self.assertEqual(lx.grad.tolist(), 0.0)
        lx.zero_grad()
        self.assertIsNone(lx.grad)
        with ft.no_grad():
            ng = ft.square(lx)
        self.assertFalse(ng.requires_grad)
        self.assertFalse(lx.detach().requires_grad)


class HostAutodiffTests(SharedAutodiffMixin, EngineTestCase):
    device = "cpu"

    def test_unused_port_pruning_preflight_guards_mutation_and_deep_chain(self):
        self.assertFalse(hasattr(ft, "grad"))
        self.assertFalse(hasattr(ft, "vjp"))

        # Unused sibling output or unused input does not visit ForwardOnly's missing backward rule.
        @ft.jit
        def with_forward_only(x):
            return ForwardOnly()(x), x * x

        @ft.jit
        def ignore_second(x, unused):
            return x * x

        x = ft.tensor(3.0, requires_grad=True)
        bad_out, sq_out = with_forward_only(x)
        sq_out.backward(retain_graph=True)
        self.assertEqual(x.grad.tolist(), 6.0)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
            with self.assertRaisesRegex(ValueError, "backward.*required"):
                bad_out.backward()
        self.assertEqual(x.grad.tolist(), 6.0)

        x2 = ft.tensor(3.0, requires_grad=True)
        ignore_second(x2, ForwardOnly()(x2)).backward()
        self.assertEqual(x2.grad.tolist(), 6.0)

        # Preflight seed / flag / dtype / no_grad-inside-jit guards.
        for bad_dtype, val in (("bool", True), ("i32", 2), ("i64", 2)):
            with self.assertRaisesRegex((TypeError, ValueError), "float|requires_grad"):
                ft.tensor(val, dtype=bad_dtype, requires_grad=True)

        vx = ft.tensor([2, 3], requires_grad=True)
        vy = vx * vx
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
            with self.assertRaisesRegex(ValueError, "explicit gradient"):
                vy.backward()
            for bad_seed in (ft.tensor(1), ft.tensor([1, 1], dtype="f16"), [1, 1]):
                with self.assertRaises((TypeError, ValueError)):
                    vy.backward(bad_seed)
            with self.assertRaises(TypeError):
                vy.backward(ft.tensor([1, 1]), create_graph=1)
        self.assertIsNone(vx.grad)

        @ft.jit
        def bad_no_grad(t):
            with ft.no_grad():
                return t * t

        with self.assertRaisesRegex(TypeError, "while tracing"):
            bad_no_grad(ft.tensor(2.0, requires_grad=True))

        # Mutated custom function state invalidates cached backward plan atomically.
        class Scale(ft.fun):
            def __init__(self):
                self.factor = 2
            @ft.jit
            def forward(self, ctx, t):
                return t * self.factor
            @ft.jit
            def backward(self, ctx, dy):
                return (dy * self.factor,)

        scale = Scale()
        sx = ft.tensor(3.0, requires_grad=True)
        sout = scale(sx)
        sout.backward(retain_graph=True)
        self.assertEqual(sx.grad.tolist(), 2.0)
        scale.factor = 5
        with self.assertRaisesRegex(ValueError, "definition changed"):
            sout.backward()
        self.assertEqual(sx.grad.tolist(), 2.0)

        # Iterative topological backward handles >1000 chained steps without RecursionError.
        @ft.jit
        def step(t):
            return t + 1

        chain_x = ft.tensor(0.0, requires_grad=True)
        cur = chain_x
        for _ in range(1100):
            cur = step(cur)
        cur.backward()
        self.assertEqual(chain_x.grad.tolist(), 1.0)


@requires_cuda
class CUDAAutodiffTests(SharedAutodiffMixin, EngineTestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
