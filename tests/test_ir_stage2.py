"""Fuga IR verification, lowering, AD closure, and static specialization tests for Stage 2 primitives."""
from __future__ import annotations

import math
import pytest

import fugatensor as ft
from fugatensor import ir
from fugatensor.ir.transforms import specialize


class TestStage2FugaIR:
    def test_stage2_primitives_typing_and_lowering_to_ask(self):
        module = ir.Module()
        f = module.graph(
            'stage2',
            inputs={
                'x': ir.TensorType((4, 5), 'f32'),
                'y': ir.TensorType((4, 5), 'f32'),
                'idx': ir.TensorType((3,), 'i64'),
                'flag': ir.TensorType((4, 5), 'bool'),
            },
        )
        s0 = f.sin(f.inputs['x'])
        s1 = f.cos(s0)
        s2 = f.tanh(s1)
        s3 = f.erf(s2)
        s4 = f.abs(s3)
        s5 = f.floor(s4)
        s6 = f.ceil(s5)
        s7 = f.round(s6)
        p = f.pow(s7, f.inputs['y'])
        mx = f.maximum(p, f.inputs['x'])
        mn = f.minimum(mx, f.inputs['y'])
        fd = f.floordiv(mn, f.inputs['y'])
        md = f.mod(fd, f.inputs['y'])
        grid = f.iota((4, 5), 1, dtype='f32')
        comb = f.add(md, grid)
        sl = f.slice(comb, (3, 1), (-1, 4), (-2, 1))  # shape (2, 3)
        pd = f.pad(sl, f.constant(0.5, dtype='f32'), (1, 0), (0, 1), (1, 0))  # shape (4, 4)
        gth = f.gather(pd, f.inputs['idx'], 0)  # shape (3, 4)
        scat = f.scatter_add(pd, f.inputs['idx'], gth, 0)  # shape (4, 4)
        rmin = f.reduce_min(scat, (0,))  # shape (4,)
        rprod = f.reduce_prod(f.concatenate((pd, gth), 0), (0,))  # shape (4,)
        amax = f.argmax(pd, 1)  # shape (4,)
        amin = f.argmin(pd, 0)  # shape (4,)
        rany = f.reduce_any(f.inputs['flag'], (1,))
        rall = f.reduce_all(f.inputs['flag'], (1,))
        f.return_(rmin, rprod, amax, amin, rany, rall)
        module.verify()

        ask_text = ft.lower_to_ask(module).to_mlir()
        for opcode in ('ask.sin', 'ask.cos', 'ask.tanh', 'ask.erf', 'ask.abs',
                       'ask.floor', 'ask.ceil', 'ask.round', 'ask.pow', 'ask.maximum',
                       'ask.minimum', 'ask.floordiv', 'ask.mod', 'ask.iota', 'ask.slice',
                       'ask.pad', 'ask.gather', 'ask.scatter_add', 'ask.reduce_min',
                       'ask.reduce_prod', 'ask.argmax', 'ask.argmin', 'ask.reduce_any',
                       'ask.reduce_all', 'ask.concatenate'):
            assert opcode in ask_text, f"missing lowered ASK opcode {opcode}"

    def test_stage2_ir_validation_rejections(self):
        def check_bad(build_fn):
            m = ir.Module()
            with pytest.raises((TypeError, ValueError)):
                build_fn(m)
                m.verify()

        check_bad(lambda m: m.graph('bad', inputs={'x': ir.TensorType((4,), 'i32')}).sin(
            next(iter(m.functions[-1].inputs.values()))))
        check_bad(lambda m: m.graph('bad', inputs={'x': ir.TensorType((4,), 'f32')}).bitwise_not(
            next(iter(m.functions[-1].inputs.values()))))
        check_bad(lambda m: m.graph('bad', inputs={'x': ir.TensorType((4,), 'bool')}).shift_left(
            next(iter(m.functions[-1].inputs.values())),
            next(iter(m.functions[-1].inputs.values()))))
        check_bad(lambda m: m.graph('bad', inputs={'x': ir.TensorType((0, 4), 'f32')}).argmax(
            next(iter(m.functions[-1].inputs.values())), 0))
        check_bad(lambda m: m.graph('bad', inputs={'x': ir.TensorType((0, 4), 'f32')}).reduce_min(
            next(iter(m.functions[-1].inputs.values())), (0,)))
        check_bad(lambda m: m.graph('bad', inputs={'x': ir.TensorType((4,), 'f32')}).slice(
            next(iter(m.functions[-1].inputs.values())), (0,), (5,), (1,)))
        check_bad(lambda m: m.graph('bad', inputs={'b': ir.TensorType((4,), 'bool'),
                                                   'i': ir.TensorType((2,), 'i32'),
                                                   'u': ir.TensorType((2,), 'bool')}).scatter_add(
            *m.functions[-1].inputs.values(), 0))
        check_bad(lambda m: m.graph('bad', inputs={}).iota((4,), 0, dtype='bool'))

    def test_vjp_jvp_and_higher_order_ad_closure(self):
        # Unary, binary, reduce_min, reduce_prod, slice, pad, concatenate, gather, scatter_add
        m = ir.Module()
        f = m.graph('f', inputs={
            'x': ir.TensorType((5, 4), 'f32'),
            'idx': ir.TensorType((3,), 'i64'),
            'pv': ir.ScalarType('f32'),
        })
        x, idx, pv = f.inputs['x'], f.inputs['idx'], f.inputs['pv']
        s = f.sin(x)
        sl = f.slice(s, (4, 1), (-1, 4), (-2, 2))  # (3, 2)
        pd = f.pad(sl, pv, (1, 0), (0, 1), (1, 1))  # (6, 4)
        g = f.gather(pd, idx, 0)  # (3, 4)
        sc = f.scatter_add(pd, idx, g, 0)  # (6, 4)
        pr = f.reduce_prod(f.add(sc, f.splat(f.constant(1.2, dtype='f32'), (6, 4))), (1,))
        mn = f.reduce_min(pr, (0,))
        f.return_(mn)
        m.verify()

        rev1 = ir.vjp(m, argnums=(0, 2))
        rev1.verify()
        assert len(rev1.functions[0].outputs) == 2  # dx, dpv

        # Second order VJP(VJP), JVP(VJP), and VJP(JVP) must all be valid closed IR lowerable to ASK
        rev2 = ir.vjp(rev1, argnums=(0, 2))
        rev2.verify()
        fwd1 = ir.jvp(m, argnums=(0, 2))
        fwd1.verify()
        fwd_over_rev = ir.jvp(rev1, argnums=(0, 2))
        fwd_over_rev.verify()
        rev_over_fwd = ir.vjp(fwd1, argnums=(0, 2))
        rev_over_fwd.verify()
        for graph_mod in (rev1, rev2, fwd1, fwd_over_rev, rev_over_fwd):
            ft.lower_to_ask(graph_mod)

    def test_known_scalar_control_flow_simplification(self):
        m = ir.Module()
        f = m.graph('cf', inputs={'x': ir.TensorType((4,), 'f32')})
        x = f.inputs['x']
        a = f.constant(6, dtype='i32')
        b = f.constant(3, dtype='i32')
        shifted = f.shift_left(f.bitwise_xor(a, b), f.constant(1, dtype='i32'))  # (6 ^ 3) << 1 == 10
        cond = f.compare(shifted, f.constant(10, dtype='i32'), 'eq')
        out = f.if_(cond, (x,), lambda tb, y: (tb.sin(y),), lambda eb, y: (eb.cos(y),))
        f.return_(out)
        flat = specialize(m)
        flat.verify()
        assert 'sin' in flat.to_text() and 'cos' not in flat.to_text()
