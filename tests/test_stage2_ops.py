"""Execution, autodiff, boundary, and neural-network functional tests for Stage 2 operations on CPU and CUDA."""
from __future__ import annotations

import math
import pytest

import fugatensor as ft
import fugatensor.nn.functional as F
from tests._helpers import requires_cuda
from tests.support.stage2 import (
    assert_allclose,
    py_floordiv_int,
    py_mod_int,
    py_shl_int,
    py_shr_int,
    ref_gelu,
    ref_layer_norm,
    ref_log_softmax,
    ref_rms_norm,
    ref_softmax,
    rint_even,
)


def _central_diff(fn, values: list[float], eps: float = 1e-3) -> list[float]:
    grads = []
    for i in range(len(values)):
        plus = list(values)
        minus = list(values)
        plus[i] += eps
        minus[i] -= eps
        grads.append((fn(plus) - fn(minus)) / (2.0 * eps))
    return grads


class Stage2OpsSuite:
    device = 'cpu'
    float_dtypes = (ft.fp32, ft.fp16)

    def _tol(self, dtype, strict=1e-5):
        if dtype is ft.fp32:
            return strict, strict
        if dtype is ft.fp16:
            return 2e-2, 2e-2
        return 3e-2, 3e-2

    def test_unary_and_binary_float_math(self):
        xs = [-1.7, -0.5, 0.25, 0.9, 1.5, 2.25]
        ys = [1.5, -2.0, 0.5, 2.0, -1.25, 3.0]
        pos_xs = [0.25, 0.75, 1.5, 2.5, 3.0, 0.5]
        for dtype in self.float_dtypes:
            atol, rtol = self._tol(dtype, 2e-5)
            tx = ft.tensor(xs, dtype=dtype, device=self.device)
            ty = ft.tensor(ys, dtype=dtype, device=self.device)
            tpx = ft.tensor(pos_xs, dtype=dtype, device=self.device)

            assert_allclose(ft.sin(tx).tolist(), [math.sin(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(ft.cos(tx).tolist(), [math.cos(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(ft.tanh(tx).tolist(), [math.tanh(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(ft.erf(tx).tolist(), [math.erf(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(tx.abs().tolist(), [abs(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(ft.floor(tx).tolist(), [math.floor(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(ft.ceil(tx).tolist(), [math.ceil(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose(ft.round(tx).tolist(), [rint_even(v) for v in xs], atol=atol, rtol=rtol)
            assert_allclose((tpx ** ty).tolist(), [a ** b for a, b in zip(pos_xs, ys)], atol=atol, rtol=rtol)
            assert_allclose((2.0 ** tpx).tolist(), [2.0 ** a for a in pos_xs], atol=atol, rtol=rtol)
            assert_allclose(ft.maximum(tx, ty).tolist(), [max(a, b) for a, b in zip(xs, ys)], atol=atol, rtol=rtol)
            assert_allclose(ft.minimum(tx, ty).tolist(), [min(a, b) for a, b in zip(xs, ys)], atol=atol, rtol=rtol)
            assert_allclose((tx // ty).tolist(), [a // b for a, b in zip(xs, ys)], atol=atol, rtol=rtol)
            assert_allclose((tx % ty).tolist(), [a % b for a, b in zip(xs, ys)], atol=atol, rtol=rtol)
            assert_allclose(tx.clamp(-0.6, 1.4).tolist(), [min(max(v, -0.6), 1.4) for v in xs], atol=atol, rtol=rtol)

        # Banker's rounding ties and NaN propagation in maximum/minimum
        ties = ft.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=ft.fp32, device=self.device)
        assert ft.round(ties).tolist() == [-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]
        nan_t = ft.tensor([1.0, float('nan'), 3.0], dtype=ft.fp32, device=self.device)
        other = ft.tensor([2.0, 5.0, float('nan')], dtype=ft.fp32, device=self.device)
        assert_allclose(ft.maximum(nan_t, other).tolist(), [2.0, float('nan'), float('nan')])
        assert_allclose(ft.minimum(nan_t, other).tolist(), [1.0, float('nan'), float('nan')])

    def test_integer_div_mod_pow_bitwise_shifts_and_logic(self):
        for dtype, bits in ((ft.int32, 32), (ft.int64, 64)):
            min_i = -(1 << (bits - 1))
            a_vals = [-7, 7, -7, 7, min_i, 5, -8, 12]
            b_vals = [2, -2, -2, 0, -1, 3, -1, 64]
            a = ft.tensor(a_vals, dtype=dtype, device=self.device)
            b = ft.tensor(b_vals, dtype=dtype, device=self.device)

            assert (a // b).tolist() == [py_floordiv_int(x, y, bits) for x, y in zip(a_vals, b_vals)]
            assert (a % b).tolist() == [py_mod_int(x, y, bits) for x, y in zip(a_vals, b_vals)]
            assert abs(a).tolist() == [min_i if v == min_i else abs(v) for v in a_vals]

            sh_l = ft.tensor([1, -3, 7, 15, 1, 1], dtype=dtype, device=self.device)
            sh_r = ft.tensor([0, 2, -1, bits, bits - 1, 3], dtype=dtype, device=self.device)
            assert (sh_l << sh_r).tolist() == [py_shl_int(x, y, bits) for x, y in zip([1, -3, 7, 15, 1, 1], [0, 2, -1, bits, bits - 1, 3])]
            assert (sh_l >> sh_r).tolist() == [py_shr_int(x, y, bits) for x, y in zip([1, -3, 7, 15, 1, 1], [0, 2, -1, bits, bits - 1, 3])]

            base = ft.tensor([2, -3, 1, -1, 0, 5], dtype=dtype, device=self.device)
            exp = ft.tensor([5, 3, -4, -3, 0, -1], dtype=dtype, device=self.device)
            assert (base ** exp).tolist() == [32, -27, 1, -1, 1, 0]

            bx = ft.tensor([0b1010, -5, 12], dtype=dtype, device=self.device)
            by = ft.tensor([0b1100, 3, 7], dtype=dtype, device=self.device)
            assert (bx & by).tolist() == [0b1010 & 0b1100, -5 & 3, 12 & 7]
            assert (bx | by).tolist() == [0b1010 | 0b1100, -5 | 3, 12 | 7]
            assert (bx ^ by).tolist() == [0b1010 ^ 0b1100, -5 ^ 3, 12 ^ 7]
            assert (~bx).tolist() == [~0b1010, ~(-5), ~12]

        # Boolean & general logical ops
        m1 = ft.tensor([True, True, False, False], dtype=ft.bool, device=self.device)
        m2 = ft.tensor([True, False, True, False], dtype=ft.bool, device=self.device)
        assert (m1 & m2).tolist() == [True, False, False, False]
        assert (m1 | m2).tolist() == [True, True, True, False]
        assert (m1 ^ m2).tolist() == [False, True, True, False]
        assert (~m1).tolist() == [False, False, True, True]
        num = ft.tensor([0.0, 2.5, -1.0, 0.0], dtype=ft.fp32, device=self.device)
        assert ft.logical_and(num, m2).tolist() == [False, False, True, False]
        assert ft.logical_or(num, m2).tolist() == [True, True, True, False]
        assert ft.logical_xor(num, m2).tolist() == [True, True, False, False]
        assert ft.logical_not(num).tolist() == [True, False, False, True]

    def test_reductions_and_warp_boundaries(self):
        # Test small and large (>32 and >256 elements) reductions to cover CUDA warp & multi-warp unroll
        for cols in (1, 10, 32, 65, 300):
            data = [[float(((r * 17 + c * 7) % 19) - 9) * 0.25 for c in range(cols)] for r in range(3)]
            t = ft.tensor(data, dtype=ft.fp32, device=self.device)
            assert_allclose(t.amin(axis=1).tolist(), [min(row) for row in data])
            assert_allclose(t.amax(axis=1).tolist(), [max(row) for row in data])
            assert t.argmax(axis=1).tolist() == [row.index(max(row)) for row in data]
            assert t.argmin(axis=1).tolist() == [row.index(min(row)) for row in data]

            flags_any = [[c == cols - 1 if r == 0 else False for c in range(cols)] for r in range(2)]
            flags_all = [[True for _ in range(cols)], [c != 0 for c in range(cols)]]
            assert ft.tensor(flags_any, dtype=ft.bool, device=self.device).any(axis=1).tolist() == [True, False]
            assert ft.tensor(flags_all, dtype=ft.bool, device=self.device).all(axis=1).tolist() == [True, False]

        # Product reduction & empty reductions
        p = ft.tensor([[1.5, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=ft.fp32, device=self.device)
        assert_allclose(p.prod(axis=1).tolist(), [-9.0, -2.0])
        assert p.prod().item() == pytest.approx(18.0)

        empty23 = ft.empty(2, 0, dtype=ft.fp32, device=self.device)
        assert empty23.prod(axis=1).tolist() == [1.0, 1.0]
        empty_b = ft.empty(2, 0, dtype=ft.bool, device=self.device)
        assert empty_b.any(axis=1).tolist() == [False, False]
        assert empty_b.all(axis=1).tolist() == [True, True]

        # Argmax/argmin tie-breaking (first occurrence) and NaN propagation + global axis=None + keepdims
        tie = ft.tensor([[2.0, 5.0, 5.0, 1.0], [3.0, 1.0, 1.0, 3.0]], dtype=ft.fp32, device=self.device)
        assert tie.argmax(axis=1, keepdims=True).tolist() == [[1], [0]]
        assert tie.argmin(axis=1).tolist() == [3, 1]
        assert tie.argmax().item() == 1
        assert tie.argmin(keepdims=True).tolist() == [[3]]

        nan_row = ft.tensor([1.0, float('nan'), 9.0, float('nan')], dtype=ft.fp32, device=self.device)
        assert nan_row.argmax(axis=0).item() == 1
        assert nan_row.argmin(axis=0).item() == 1

    def test_zero_input_creation_factories(self):
        assert ft.arange(6, dtype=ft.int32, device=self.device).tolist() == [0, 1, 2, 3, 4, 5]
        assert ft.arange(10, 2, -2, dtype=ft.int64, device=self.device).tolist() == [10, 8, 6, 4]
        assert ft.arange(5, 5, device=self.device).tolist() == []
        assert_allclose(ft.linspace(-1.0, 1.0, 5, device=self.device).tolist(), [-1.0, -0.5, 0.0, 0.5, 1.0])
        assert ft.linspace(2.0, 5.0, 1, device=self.device).tolist() == [2.0]
        assert ft.eye(2, 3, dtype=ft.int32, device=self.device).tolist() == [[1, 0, 0], [0, 1, 0]]
        assert ft.iota((2, 3), dimension=1, dtype=ft.int32, device=self.device).tolist() == [[0, 1, 2], [0, 1, 2]]

        base = ft.zeros(2, 3, dtype=ft.fp32, device=self.device)
        fl = ft.full_like(base, 7.5)
        assert fl.shape == (2, 3) and fl.device == base.device and fl.tolist() == [[7.5] * 3] * 2

        class Synth(ft.fun):
            def forward(self, x):
                grid = ft.arange(x.shape[1], dtype=x.dtype, device=x.device)
                return x + grid.reshape(1, -1).broadcast_to(x.shape) + ft.full_like(x, 1.0)

        out = Synth()(ft.zeros(2, 4, dtype=ft.fp32, device=self.device))
        assert out.tolist() == [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]

    def test_indexing_slicing_pad_concat_split_tril_one_hot(self):
        data = [[r * 5 + c for c in range(5)] for r in range(4)]
        x = ft.tensor([[float(v) for v in row] for row in data], dtype=ft.fp32, device=self.device)

        assert x.ndim == 2 and x.numel == 20
        assert x[::-1, 1:5:2].tolist() == [[16.0, 18.0], [11.0, 13.0], [6.0, 8.0], [1.0, 3.0]]
        assert x[-1, None, ::2].tolist() == [[15.0, 17.0, 19.0]]
        assert x[..., 0].tolist() == [0.0, 5.0, 10.0, 15.0]
        assert x[2:1].shape == (0, 5)
        assert x[1:3:-1].shape == (0, 5)

        tidx = ft.tensor([3, 0, 3, -1], dtype=ft.int64, device=self.device)
        assert x[tidx, 1:4].tolist() == [[16.0, 17.0, 18.0], [1.0, 2.0, 3.0], [16.0, 17.0, 18.0], [0.0, 0.0, 0.0]]

        # Pad with interior dilation and empty edge
        small = ft.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ft.fp32, device=self.device)
        padded = ft.pad(small, ((1, 0), (0, 1)), constant_value=-1.0, interior=(1, 1))
        assert padded.shape == (4, 4)
        assert padded.tolist() == [
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, 2.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [3.0, -1.0, 4.0, -1.0],
        ]

        # Concatenation, stack, split, chunk, squeeze/unsqueeze/flatten
        a = ft.tensor([[1.0, 2.0]], dtype=ft.fp32, device=self.device)
        b = ft.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=ft.fp32, device=self.device)
        c0 = ft.empty(0, 2, dtype=ft.fp32, device=self.device)
        cat = ft.cat([a, c0, b], axis=0)
        assert cat.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        stk = ft.stack([a.squeeze(0), a.squeeze(0)], axis=1)
        assert stk.shape == (2, 2)
        p0, p1, p2 = cat.split([1, 0, 2], axis=0)
        assert p0.tolist() == [[1.0, 2.0]] and p1.shape == (0, 2) and p2.tolist() == [[3.0, 4.0], [5.0, 6.0]]
        ch0, ch1 = cat.chunk(2, axis=0)
        assert ch0.shape == (2, 2) and ch1.shape == (1, 2)
        assert a.unsqueeze(2).squeeze((0, 2)).shape == (2,)
        assert ft.ones(1, 1, device=self.device).squeeze().shape == ()
        assert x[0, 0].item() == 0.0
        assert ft.clamp(small, min=1.5, max=3.5).tolist() == [[1.5, 2.0], [3.0, 3.5]]
        assert small.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]

        # tril, triu, one_hot
        m3 = ft.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=ft.int32, device=self.device)
        assert ft.tril(m3, diagonal=-1).tolist() == [[0, 0, 0], [4, 0, 0], [7, 8, 0]]
        assert ft.triu(m3, diagonal=1).tolist() == [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
        oh = ft.one_hot(ft.tensor([2, 0, -1], dtype=ft.int32, device=self.device), 4, dtype=ft.fp32)
        assert oh.tolist() == [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    def test_gather_and_scatter_add_duplicates_oob_and_reuse(self):
        for dtype in self.float_dtypes:
            base = ft.zeros(4, 2, dtype=dtype, device=self.device)
            idx = ft.tensor([1, 3, 1, -1, 4, 1], dtype=ft.int32, device=self.device)
            upd = ft.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 10.0], [20.0, 20.0], [0.5, -1.0]],
                            dtype=dtype, device=self.device)
            out = ft.scatter_add(base, idx, upd, axis=0)
            atol, rtol = self._tol(dtype, 1e-5)
            assert_allclose(out.tolist(), [[0.0, 0.0], [6.5, 7.0], [0.0, 0.0], [3.0, 4.0]], atol=atol, rtol=rtol)

        # Repeated cached run via ft.fun (exercises CUDA Graph child-graph parameter updates on GPU)
        class Updater(ft.fun):
            def forward(self, base, idx, upd):
                return ft.scatter_add(base, idx, upd, axis=1).abs()

        fn = Updater()
        for scale in (1.0, 2.0, -3.0):
            b = ft.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ft.fp32, device=self.device)
            i = ft.tensor([2, 0, 2], dtype=ft.int64, device=self.device)
            u = ft.tensor([[scale, scale, scale], [-scale, scale, 0.0]], dtype=ft.fp32, device=self.device)
            res = fn(b, i, u)
            assert_allclose(res.tolist(), [
                [abs(1.0 + scale), 2.0, abs(3.0 + 2.0 * scale)],
                [abs(4.0 + scale), 5.0, abs(6.0 - scale)],
            ])

        # Empty updates still preserves base via preCopy / host copy
        b_keep = ft.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ft.fp32, device=self.device)
        e_idx = ft.empty(0, dtype=ft.int32, device=self.device)
        e_upd = ft.empty(0, 2, dtype=ft.fp32, device=self.device)
        assert ft.scatter_add(b_keep, e_idx, e_upd, axis=0).tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_autograd_stage2_primitives_and_zero_prod_second_order(self):
        # Finite-difference checks across unary, pow, maximum/minimum, mod, slice, pad, cat, gather, scatter
        x_vals = [0.6, -1.2, 1.4, -0.4, 2.1]
        y_vals = [1.8, 1.5, -0.9, 2.0, 1.1]

        def py_fn(xv):
            s = 0.0
            for xi, yi in zip(xv, y_vals):
                u = math.sin(xi) + math.cos(xi * 0.5) - math.tanh(xi) + math.erf(xi) + abs(xi)
                v = ((xi + 2.5) ** yi) + max(u, 0.5) - min(u, -0.2) + (u % 1.3)
                s += v
            return s

        x = ft.tensor(x_vals, dtype=ft.fp32, device=self.device)
        y = ft.tensor(y_vals, dtype=ft.fp32, device=self.device)
        u = ft.sin(x) + ft.cos(x * 0.5) - ft.tanh(x) + ft.erf(x) + abs(x)
        loss = (((x + 2.5) ** y) + ft.maximum(u, 0.5) - ft.minimum(u, -0.2) + (u % 1.3)).sum()
        gx = ft.grad(loss, x)
        assert_allclose(gx.tolist(), _central_diff(py_fn, x_vals, eps=1e-3), atol=1e-2, rtol=1e-2)

        # Zero-safe reduce_prod first and second derivative
        pz = ft.tensor([0.0, 2.0, 3.0], dtype=ft.fp32, device=self.device)
        prod_loss = pz.prod()
        gpz = ft.grad(prod_loss, pz)
        assert_allclose(gpz.tolist(), [6.0, 0.0, 0.0])
        # Second-order mixed derivative: d^2(prod)/dx0 dx1 = x2 = 3.0
        g2 = ft.grad(gpz[0], pz)
        assert_allclose(g2.tolist(), [0.0, 3.0, 2.0])

        # Slice (reverse step), pad (with pad_value grad!), cat, gather, scatter_add adjoint gradients
        w = ft.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=ft.fp32, device=self.device)
        pv = ft.tensor(0.5, dtype=ft.fp32, device=self.device)
        idx = ft.tensor([1, 0, 1], dtype=ft.int32, device=self.device)
        sl = w[:, ::-2]  # cols 3, 1 -> (2, 2)
        pd = ft.pad(sl, ((1, 0), (0, 1)), constant_value=pv, interior=(1, 0))  # shape (4, 3)
        gth = ft.gather(pd, idx, axis=0)  # (3, 3)
        base_small = ft.ones(2, 2, dtype=ft.fp32, device=self.device)
        sc2 = ft.scatter_add(base_small, idx, gth[:, :2], axis=0)
        weights = ft.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ft.fp32, device=self.device)
        loss_geom = (sc2 * weights).sum()
        gw, gpv = ft.grad(loss_geom, (w, pv))
        # Row 0 receives scatter at idx=1 once with weight row 0: [1,2]? Check exact numerical AD consistency via finite diff:
        def geom_scalar(w_flat, pv_val):
            # sl[r, c] = w[r, [3, 1][c]]
            # pd shape (4, 3): pd[0,:]=pv; pd[1,0..1]=sl[0], pd[1,2]=pv; pd[2,:]=pv; pd[3,0..1]=sl[1], pd[3,2]=pv
            pd_rows = [
                [pv_val, pv_val, pv_val],
                [w_flat[3], w_flat[1], pv_val],
                [pv_val, pv_val, pv_val],
                [w_flat[7], w_flat[5], pv_val],
            ]
            res = [[1.0, 1.0], [1.0, 1.0]]
            for ii in [1, 0, 1]:
                src = pd_rows[ii]
                res[ii][0] += src[0]
                res[ii][1] += src[1]
            return res[0][0] * 1 + res[0][1] * 2 + res[1][0] * 3 + res[1][1] * 4

        w_flat0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        exp_gw = _central_diff(lambda wf: geom_scalar(wf, 0.5), w_flat0, eps=1e-3)
        exp_gpv = _central_diff(lambda pv_: geom_scalar(w_flat0, pv_[0]), [0.5], eps=1e-3)[0]
        assert_allclose(gw.tolist(), [exp_gw[:4], exp_gw[4:]], atol=1e-3, rtol=1e-3)
        assert_allclose(gpv.item(), exp_gpv, atol=1e-3, rtol=1e-3)

    def test_nn_functional_forward_and_backward(self):
        x = ft.tensor([[-2.0, -0.5, 0.0, 1.5], [1.0, -3.0, 2.0, 0.5]], dtype=ft.fp32, device=self.device)

        assert_allclose(F.relu(x).tolist(), [[0.0, 0.0, 0.0, 1.5], [1.0, 0.0, 2.0, 0.5]])
        assert_allclose(F.leaky_relu(x, 0.1).tolist(), [[-0.2, -0.05, 0.0, 1.5], [1.0, -0.3, 2.0, 0.5]])
        assert_allclose(F.sigmoid(x).tolist(), [[1.0 / (1.0 + math.exp(-v)) for v in r] for r in x.tolist()])
        assert_allclose(F.silu(x).tolist(), [[v / (1.0 + math.exp(-v)) for v in r] for r in x.tolist()])
        assert_allclose(F.gelu(x).tolist(), [[ref_gelu(v, 'none') for v in r] for r in x.tolist()], atol=1e-5, rtol=1e-5)
        assert_allclose(F.gelu(x, approximate='tanh').tolist(),
                        [[ref_gelu(v, 'tanh') for v in r] for r in x.tolist()], atol=1e-5, rtol=1e-5)

        # Overflow safety for sigmoid backward at large magnitude
        big = ft.tensor([-1000.0, 1000.0, 0.0], dtype=ft.fp32, device=self.device)
        assert_allclose(F.sigmoid(big).tolist(), [0.0, 1.0, 0.5])
        assert_allclose(ft.grad(F.sigmoid(big).sum(), big).tolist(), [0.0, 0.0, 0.25])

        assert_allclose(F.softmax(x, axis=1).tolist(), [ref_softmax(r) for r in x.tolist()])
        assert_allclose(F.log_softmax(x, axis=1).tolist(), [ref_log_softmax(r) for r in x.tolist()])

        w = ft.tensor([1.0, 2.0, 0.5, 1.5], dtype=ft.fp32, device=self.device)
        b = ft.tensor([0.1, -0.2, 0.3, 0.0], dtype=ft.fp32, device=self.device)
        ln = F.layer_norm(x, weight=w, bias=b)
        rn = F.rms_norm(x, weight=w)
        assert_allclose(ln.tolist(), [ref_layer_norm(r, 1e-5, w.tolist(), b.tolist()) for r in x.tolist()],
                        atol=1e-4, rtol=1e-4)
        assert_allclose(rn.tolist(), [ref_rms_norm(r, 1e-6, w.tolist()) for r in x.tolist()],
                        atol=1e-4, rtol=1e-4)

        # Embedding + cross_entropy forward and gradient
        emb_table = ft.tensor([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6], [-0.2, 0.8, 0.1]],
                              dtype=ft.fp32, device=self.device)
        proj = ft.tensor([[1.0, -1.0, 0.5, 0.2], [0.2, 0.7, -0.4, 1.1], [-0.3, 0.5, 0.9, -0.2]],
                         dtype=ft.fp32, device=self.device)
        tok = ft.tensor([0, 2], dtype=ft.int32, device=self.device)
        tgt = ft.tensor([3, 1], dtype=ft.int64, device=self.device)
        logits = F.embedding(tok, emb_table) @ proj
        ce = F.cross_entropy(logits, tgt)
        g_emb = ft.grad(ce, emb_table)
        # Expected reference CE on logits
        lg_rows = logits.tolist()
        exp_ce = 0.5 * (-ref_log_softmax(lg_rows[0])[3] - ref_log_softmax(lg_rows[1])[1])
        assert_allclose(ce.item(), exp_ce, atol=1e-5, rtol=1e-5)
        assert len(g_emb.tolist()) == 3 and g_emb.tolist()[1] == [0.0, 0.0, 0.0]


class TestStage2OpsCPU(Stage2OpsSuite):
    device = 'cpu'
    float_dtypes = (ft.fp32, ft.fp16)


@requires_cuda
class TestStage2OpsCUDA(Stage2OpsSuite):
    device = 'cuda'
    float_dtypes = (ft.fp32, ft.fp16, ft.bf16)

    def test_cross_device_and_cross_engine_errors(self):
        cpu_x = ft.tensor([1.0, 2.0], dtype=ft.fp32, device='cpu')
        cuda_y = ft.tensor([3.0, 4.0], dtype=ft.fp32, device='cuda')
        with pytest.raises(RuntimeError):
            _ = ft.maximum(cpu_x, cuda_y)
        with ft.Engine('cpu') as e1, ft.Engine('cpu') as e2:
            t1 = ft.tensor([1.0], device=ft.device('cpu'), engine=e1)
            t2 = ft.tensor([2.0], device=ft.device('cpu'), engine=e2)
            with pytest.raises(ValueError):
                _ = ft.maximum(t1, t2)
