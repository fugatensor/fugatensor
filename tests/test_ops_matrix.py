"""Public operations × generated compositions × dtypes × derivative orders."""
from __future__ import annotations

import unittest

import fugatensor as ft
from fugatensor._ask import CompilerError
from fugatensor._ask.cuda import CUDAError
from tests._helpers import EngineTestCase, requires_cuda
from tests.support.ops_matrix import DTYPES, FLOATS, input_values, plan_matrix


def _sum(values):
    values = iter(values)
    result = next(values, None)
    for value in values:
        result = result + value
    return result


class SharedOperationMatrix:
    def test_generated_operation_compositions(self):
        for name, cases, orders in plan_matrix():
            dtype = DTYPES[name]
            with self.subTest(dtype=name):
                if dtype is ft.bf16 and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), 'bf16.*hardware|bf16.*support'):
                        ft.square(ft.tensor([1], dtype=dtype)).tolist()
                    continue
                # Four ports per case (two inputs and two gradients), plus
                # derivative seeds, remain below the CUDA 64-buffer ABI limit.
                for offset in range(0, len(cases), 12):
                    self._run_batch(name, cases[offset:offset + 12], orders)

    def _run_batch(self, name, cases, orders):
        dtype = DTYPES[name]
        xv, yv = input_values(name)
        differentiable = name in FLOATS and max(orders) > 0
        leaves = [(ft.tensor(xv, dtype=dtype, requires_grad=differentiable),
                   ft.tensor(yv, dtype=dtype, requires_grad=differentiable)) for _ in cases]
        values = [case.apply(x, y) for case, (x, y) in zip(cases, leaves)]
        ft.evaluate(*values)
        polynomials = [case.polynomial() for case in cases]
        # Unused inputs have no gradient; connected constant derivatives may
        # have an explicit zero tensor. Both must agree with the same oracle.
        pending = [((), values, polynomials)]
        while pending:
            axes, current, references = pending.pop()
            order = len(axes)
            if order in orders:
                for case, value, reference in zip(cases, current, references):
                    with self.subTest(dtype=name, operations=case.name, derivative=axes):
                        expected = [float(reference.evaluate(x, y)) for x, y in zip(xv, yv)]
                        if value is None:
                            self.assertEqual(expected, [0] * len(xv))
                        else:
                            self.assertIs(value.dtype, dtype)
                            self.assertEqual(value.shape, (len(xv),))
                            self.assertEqual(value.device, leaves[0][0].device)
                            if differentiable and order == max(orders):
                                self.assertFalse(value.requires_grad)
                            actual = value.tolist()
                            self.assertEqual(len(actual), len(expected))
                            self.assertEqual(actual, expected)
            if order >= max(orders):
                continue
            for pair in leaves:
                for leaf in pair:
                    leaf.zero_grad()
            total = _sum(value for value in current if value is not None and value.requires_grad)
            if total is not None:
                total.backward(ft.ones(len(xv), dtype=dtype), retain_graph=True,
                               create_graph=order + 1 < max(orders))
            for axis in (0, 1):
                pending.append(((*axes, axis), [pair[axis].grad for pair in leaves],
                                [reference.derivative(axis) for reference in references]))


class HostOperationMatrixTests(SharedOperationMatrix, EngineTestCase):
    device = 'cpu'


@requires_cuda
class CUDAOperationMatrixTests(SharedOperationMatrix, EngineTestCase):
    device = 'cuda'


if __name__ == '__main__':
    unittest.main()
