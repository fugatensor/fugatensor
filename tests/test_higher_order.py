"""Composable functional gradients across primitives, control flow and custom rules."""
from __future__ import annotations

import pytest

from unittest.mock import patch

from fugatensor._ask.cuda import CUDAError
import fugatensor as ft
from tests._helpers import TRITON_KERNELS, requires_cuda, requires_triton


class SharedHigherOrderMixin:
    def test_quartic_three_orders_control_flow_and_seed_participation(self):
        def quartic(x):
            sq = x * x
            return sq * sq

        for dtype in ("f32", "f16", "bf16"):
            if dtype == "bf16" and not self.supports_bf16:
                with pytest.raises((TypeError, CUDAError), match='CPU tensors support only|bf16.*support'):
                    ft.tensor(2, dtype=dtype)
                continue
            x = ft.tensor(2, dtype=dtype)
            val = quartic(x)
            actual = []
            for _ in range(3):
                val = ft.grad(val, x)
                actual.append(val.tolist())
            assert (actual) == ([32, 48, 48])

        def selected(x, flag, count):
            if flag:
                while count:
                    x = x * x
                    count = count - 1
                return x
            return x * x

        for flag, exp1, exp2 in ((True, 32, 48), (False, 4, 2)):
            x = ft.tensor(2.0)
            out = selected(x, ft.tensor(flag, dtype="bool"), ft.tensor(2, dtype="i32"))
            d1 = ft.grad(out, x)
            assert ((d1.tolist(), ft.grad(d1, x).tolist())) == ((exp1, exp2))

        vx = ft.tensor([2.0, 3.0])
        seed = ft.tensor([5.0, 7.0])
        first = ft.grad(ft.square(vx), vx, seed=seed)
        dx, ds = ft.grad(first, (vx, seed), seed=ft.tensor([1.0, 2.0]))
        assert ((dx.tolist(), ds.tolist())) == (([10.0, 28.0], [4.0, 12.0]))

    def test_nested_fun_rules_recomputed_intermediates_and_mutation_revalidation(self):
        class Inner(ft.fun):
            def forward(self, x):
                return x * x
            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * primals[0] * 7,)

        inner = Inner()

        class Outer(ft.fun):
            def forward(self, x):
                return x * x * x
            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * inner(primals[0]),)

        x = ft.tensor(3.0)
        val = Outer()(x)
        for expected in (9.0, 21.0, 7.0):
            val = ft.grad(val, x)
            assert (val.tolist()) == (expected)

        class MutableInner(ft.fun):
            def __init__(self):
                self.factor = 2
            def forward(self, t):
                return t * self.factor
            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * self.factor,)

        mi = MutableInner()

        class Wrapper(ft.fun):
            def forward(self, t):
                return t * t
            def vjp(self, primals, outputs, cotangents):
                return (mi(cotangents[0]),)

        wx = ft.tensor(3.0)
        wout = Wrapper()(wx)
        gradient = ft.grad(wout, wx)
        assert (gradient.tolist()) == (2.0)
        mi.factor = 5
        with patch.object(self.engine, "_execute", side_effect=AssertionError("stale rule executed")):
            with pytest.raises(ValueError, match='definition changed'):
                ft.grad(wout, wx)


@pytest.mark.usefixtures("engine_context")
class TestHostHigherOrder(SharedHigherOrderMixin):
    device = "cpu"


@requires_cuda
@pytest.mark.usefixtures("engine_context")
class TestCUDAHigherOrder(SharedHigherOrderMixin):
    device = "cuda"


@requires_triton
@pytest.mark.usefixtures("engine_context")
class TestOpaqueExternalHigherOrder:
    device = "cuda"
    enable_triton = True

    def test_custom_vjp_composes_and_active_opaque_derivative_is_rejected(self):
        scale = ft.triton.kernel(
            TRITON_KERNELS, entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,),
        )

        class ForwardExternal(ft.fun):
            def forward(self, x):
                return scale(x)
            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * 2,)

        class DerivativeExternal(ft.fun):
            def forward(self, x):
                return x * 2
            def vjp(self, primals, outputs, cotangents):
                return (scale(cotangents[0] * primals[0]),)

        x = ft.tensor(3.0)
        first = ft.grad(ForwardExternal()(x), x)
        assert (first.tolist()) == (2.0)
        assert (ft.grad(first, x).tolist()) == (0.0)
        opaque_first = ft.grad(DerivativeExternal()(x), x)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before rejection")), \
             patch.object(self.engine, "_execute", side_effect=AssertionError("executed before rejection")):
            with pytest.raises((ValueError, NotImplementedError), match='external|opaque|VJP'):
                ft.grad(opaque_first, x)
        assert (opaque_first.tolist()) == (6.0)
