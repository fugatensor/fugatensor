"""Public operations × compositions × dtypes × shapes × derivative orders."""
from __future__ import annotations

import pytest


import fugatensor as ft
from fugatensor._ask.cuda import CUDAError
from tests._helpers import requires_cuda
from tests.support.ops_matrix import DTYPES, input_values, nested_values, plan_matrix


def _sum(values):
    values = iter(values)
    result = next(values, None)
    for value in values:
        result = result + value
    return result


def pytest_generate_tests(metafunc):
    if 'operation_batch' not in metafunc.fixturenames:
        return
    batches = []
    ids = []
    for name, shape, cases, orders in plan_matrix(exhaustive=metafunc.config.getoption('--exhaustive')):
        if getattr(metafunc.cls, 'device', 'cpu') == 'cpu' and name == 'bf16':
            continue
        # Batching preserves the existing 64-buffer ABI bound and amortizes compilation.
        for offset in range(0, len(cases), 12):
            batches.append((name, shape, cases[offset:offset + 12], orders))
            ids.append(f'{name}-shape{shape}-batch{offset // 12}')
    metafunc.parametrize('operation_batch', batches, ids=ids)


class SharedOperationMatrix:
    def test_generated_operation_compositions(self, operation_batch):
        name, shape, cases, orders = operation_batch
        self._run_batch(name, shape, cases, orders)

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
                    expected = [float(reference.evaluate(x, y)) for x, y in zip(xv, yv)]
                    if value is None:
                        assert (expected) == ([0] * len(xv))
                    else:
                        assert (value.dtype) is (dtype)
                        assert (value.shape) == (shape)
                        assert (value.device) == (leaves[0][0].device)
                        assert (value.tolist()) == (nested_values(expected, shape))
            if order >= max(orders):
                continue
            total = _sum(current)
            gradients = ft.grad(total, tuple(leaf for pair in leaves for leaf in pair),
                                seed=ft.ones(shape, dtype=dtype))
            ft.evaluate(*gradients)
            for axis in (0, 1):
                pending.append(((*axes, axis), list(gradients[axis::2]),
                                [reference.derivative(axis) for reference in references]))



@pytest.mark.shared_engine
@pytest.mark.usefixtures("engine_context")
class TestHostOperationMatrix(SharedOperationMatrix):
    device = 'cpu'


@requires_cuda
@pytest.mark.shared_engine
@pytest.mark.usefixtures("engine_context")
class TestCUDAOperationMatrix(SharedOperationMatrix):
    device = 'cuda'
