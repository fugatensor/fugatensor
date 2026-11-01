"""Scalar Python control flow (graph-break if/while) across CPU and CUDA with segment caching and AD."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import fugatensor as ft
from tests._helpers import EngineTestCase, requires_cuda


class ControlFlowGraphsMixin:
    def test_scalar_branch_executes_prefix_downloads_one_scalar_and_caches_segments(self):
        calls, observed_execs = [], []

        @ft.jit
        def choose(x, flag):
            calls.append("entered")
            prefix = x * x + 1
            if flag:
                observed_execs.append(self.engine.stats["executions"])
                return prefix + 2
            observed_execs.append(self.engine.stats["executions"])
            return prefix * 3

        before_execs = self.engine.stats["executions"]
        read_shapes = []
        orig_to_flat = self.engine.to_flat

        def spy_download(val):
            read_shapes.append(val.shape)
            return orig_to_flat(val)

        with patch.object(self.engine, "to_flat", side_effect=spy_download):
            out_true = choose(ft.tensor([2, 3]), ft.tensor(True, dtype="bool"))
        self.assertEqual(read_shapes, [()])
        self.assertGreater(observed_execs[0], before_execs)
        self.assertEqual(out_true.tolist(), [7.0, 12.0])
        self.assertEqual(choose(ft.tensor([2, 3]), ft.tensor(False, dtype="bool")).tolist(), [15.0, 30.0])

        comps = self.engine.stats["compilations"]
        self.assertEqual(choose(ft.tensor([3, 4]), ft.tensor(False, dtype="bool")).tolist(), [30.0, 51.0])
        self.assertEqual(choose(ft.tensor([3, 4]), ft.tensor(True, dtype="bool")).tolist(), [12.0, 19.0])
        self.assertEqual(self.engine.stats["compilations"], comps)
        self.assertEqual(calls, ["entered"] * 4)

    def test_while_loop_nested_jit_and_distinct_fun_closures(self):
        @ft.jit
        def repeated(x, counter):
            val = x + 1
            while counter:
                val = val + 2
                counter = counter - 1
            return val

        self.assertEqual(repeated(ft.tensor(2), ft.tensor(0, dtype="i32")).tolist(), 3)
        self.assertEqual(repeated(ft.tensor(2), ft.tensor(3, dtype="i32")).tolist(), 9)
        comps = self.engine.stats["compilations"]
        self.assertEqual(repeated(ft.tensor(5), ft.tensor(6, dtype="i32")).tolist(), 18)
        self.assertEqual(self.engine.stats["compilations"], comps)

        # Nested JIT across breaks + distinct ft.fun backward closures sharing identical forward IR.
        @ft.jit
        def inner(x, *, flag, scale=2):
            return x * scale if flag else x + scale

        @ft.jit
        def outer(x, *, flag, bias=3):
            sq = x * x
            return inner(x + 1, flag=flag) + sq + bias

        self.assertEqual(outer(ft.tensor(2), flag=ft.tensor(True, dtype="bool")).tolist(), 13)
        self.assertEqual(outer(ft.tensor(2), flag=ft.tensor(False, dtype="bool")).tolist(), 12)

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
        def pick_rule(x, flag):
            return left(x) if flag else right(x)

        for flag, exp_grad in ((True, 2), (False, 3), (True, 2)):
            x = ft.tensor(4, requires_grad=True)
            pick_rule(x, ft.tensor(flag, dtype="bool")).backward()
            self.assertEqual(x.grad.tolist(), exp_grad)

    def test_backward_across_branches_loops_and_rejection_guards(self):
        @ft.jit
        def branch_and_loop(x, flag, count):
            retained = x * x
            val = retained * x if flag else retained + x
            while count:
                val = val * 2
                count = count - 1
            return val

        for flag, trips, exp_val, exp_grad in ((True, 0, 8, 12), (False, 2, 24, 20)):
            x = ft.tensor(2, requires_grad=True)
            res = branch_and_loop(x, ft.tensor(flag, dtype="bool"), ft.tensor(trips, dtype="i32"))
            self.assertEqual(res.tolist(), exp_val)
            res.backward()
            self.assertEqual(x.grad.tolist(), exp_grad)

        # Scalar numeric truthiness & non-scalar rejection + fun internal graph-break rejection.
        self.assertTrue(bool(ft.tensor(-0.5)))
        self.assertFalse(bool(ft.tensor(0, dtype="i64")))
        with self.assertRaisesRegex((TypeError, ValueError), "scalar|shape|truth"):
            bool(ft.tensor([1, 2]))

        @ft.jit
        def bad_cond(x, cond):
            return x + 1 if cond else x + 2

        with patch.object(self.engine, "_execute", side_effect=AssertionError("executed invalid condition")):
            with self.assertRaisesRegex((TypeError, ValueError), "scalar|shape|truth"):
                bad_cond(ft.tensor(3), ft.tensor([True, False], dtype="bool"))

        class BrokenFun(ft.fun):
            @ft.jit
            def forward(self, ctx, x, flag):
                return x * 2 if flag else x

        with patch.object(self.engine, "_execute", side_effect=AssertionError("executed inside fun trace")):
            with self.assertRaisesRegex((TypeError, ValueError, NotImplementedError), "fun|graph.break|control.flow|runtime"):
                BrokenFun()(ft.tensor(2), ft.tensor(True, dtype="bool"))


class HostControlFlowTests(ControlFlowGraphsMixin, EngineTestCase):
    device = "cpu"


@requires_cuda
class CUDAControlFlowTests(ControlFlowGraphsMixin, EngineTestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
