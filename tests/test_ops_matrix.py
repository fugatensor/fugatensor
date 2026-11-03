"""Public operations × compositions × dtypes × shapes × derivative orders."""
from __future__ import annotations

import unittest

import fugatensor as ft
from fugatensor._ask import CompilerError
from fugatensor._ask.cuda import CUDAError
from tests._helpers import EngineTestCase, requires_cuda
from tests.support.ops_matrix import DTYPES, input_values, nested_values, plan_matrix


def _sum(values):
    values = iter(values)
    result = next(values, None)
    for value in values:
        result = result + value
    return result


class SharedOperationMatrix:
    def test_generated_operation_compositions(self):
        for name, shape, cases, orders in plan_matrix():
            dtype = DTYPES[name]
            with self.subTest(dtype=name, shape=shape):
                if dtype is ft.bf16 and not self.supports_bf16:
                    with self.assertRaisesRegex((CompilerError, CUDAError), 'bf16.*hardware|bf16.*support'):
                        ft.square(ft.tensor([1], dtype=dtype)).tolist()
                    continue
                # Four ports per case (two inputs and two gradients), plus
                # derivative seeds, remain below the CUDA 64-buffer ABI limit.
                for offset in range(0, len(cases), 12):
                    self._run_batch(name, shape, cases[offset:offset + 12], orders)

    def _run_batch(self, name, shape, cases, orders):
        dtype = DTYPES[name]
        xv, yv = input_values(name, shape)
        leaves = [(ft.tensor(xv, dtype=dtype).reshape(shape),
                   ft.tensor(yv, dtype=dtype).reshape(shape)) for _ in cases]
        values = [case.apply(x, y) for case, (x, y) in zip(cases, leaves)]
        ft.evaluate(*values)
        polynomials = [case.polynomial() for case in cases]
        # Disconnected paths and constant derivatives return explicit zero tensors.
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
                            self.assertEqual(value.shape, shape)
                            self.assertEqual(value.device, leaves[0][0].device)
                            self.assertEqual(value.tolist(), nested_values(expected, shape))
            if order >= max(orders):
                continue
            total = _sum(current)
            gradients = ft.grad(total, tuple(leaf for pair in leaves for leaf in pair),
                                seed=ft.ones(shape, dtype=dtype))
            ft.evaluate(*gradients)
            for axis in (0, 1):
                pending.append(((*axes, axis), list(gradients[axis::2]),
                                [reference.derivative(axis) for reference in references]))



class HostOperationMatrixTests(SharedOperationMatrix, EngineTestCase):
    device = 'cpu'


@requires_cuda
class CUDAOperationMatrixTests(SharedOperationMatrix, EngineTestCase):
    device = 'cuda'


if __name__ == '__main__':
    unittest.main()
