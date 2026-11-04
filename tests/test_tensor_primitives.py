"""Generated numerical/AD/fusion matrices for the orthogonal tensor basis."""

import pytest
from itertools import product
import math
import struct

import fugatensor as ft
from fugatensor.ir import Module, ScalarType, TensorType
from tests._helpers import requires_cuda
from tests.support.tensor_primitives import (DTYPES, dtypes, execute_ir, flatten, geometry_matrix,
    host_mixed_workspace_matrix, host_workspace_matrix, materialized_chain_reference,
    matmul_matrix, matmul_reference, reduction_reference, trace,
    transpose_reference, values)


def pytest_generate_tests(metafunc):
    if 'tensor_matmul_case' not in metafunc.fixturenames:
        return
    types = tuple(name for name in dtypes(True)
                  if name != 'bf16' or getattr(metafunc.cls, 'device', 'cpu') != 'cpu')
    cases = matmul_matrix(types=types, exhaustive=metafunc.config.getoption('--exhaustive'))
    metafunc.parametrize('tensor_matmul_case', tuple(cases),
                         ids=lambda row: f'{row[0]}-{row[1]}x{row[2]}-t{int(row[3])}{int(row[4])}')


class TestTensorPrimitiveContract:
    def test_type_rules_and_derivative_closure(self):
        functions = (
            lambda x: ft.sum(ft.sqrt(x * x + 1) / ft.exp(ft.log(x + 3))),
            lambda x: ft.sum(ft.where(x > 0, x * x, x / 2)),
            lambda x: ft.sum(ft.transpose(ft.broadcast_in_dim(x, (2, 2, 3), (1, 2)), (2, 0, 1))),
            lambda x: ft.amax(x, axis=1).sum(),
        )
        for function in functions:
            module = trace(function, [((2, 3), 'f32')])
            for transformed in (ft.ir.jvp(module), ft.ir.vjp(module),
                                ft.ir.jvp(ft.ir.vjp(module), argnums=0), ft.ir.vjp(ft.ir.vjp(module), argnums=0)):
                transformed.verify()
                assert ('fuga.external') not in (transformed.to_text())
                assert ('ask.') in (ft.lower_to_ask(transformed).to_mlir())
        module = Module()
        graph = module.graph('scalar', inputs={'x': ScalarType('f32')})
        graph.return_(graph.splat(graph.inputs['x'], (2, 3)))
        derivative = ft.ir.vjp(module)
        assert (derivative.functions[0].outputs[0].type) == (ScalarType('f32'))
        assert ('fuga.reduce_sum') in (derivative.to_text())

    def test_generated_invalid_contracts(self):
        invalid = (
            lambda x: ft.broadcast_in_dim(x, (2, 3), (1, 0)),
            lambda x: ft.broadcast_in_dim(x, (2, 4), (0, 1)),
            lambda x: ft.transpose(x, (0, 0)),
            lambda x: ft.sum(x, axis=(1, 1)),
            lambda x: ft.sum(x, axis=2),
            lambda x: ft.sum(x, keepdims=1),
            lambda x: ft.where(x, x, x),
            lambda x: ft.compare(x, x, 'invalid'),
            lambda x: ft.matmul(x, x),
            lambda x: ft.matmul(x, x.T, transpose_a=1),
        )
        for function in invalid:
            with pytest.raises((TypeError, ValueError)):
                trace(function, [((2, 3), 'f32')])
        for dtype, function in product(('bool', 'i32', 'i64'), (ft.exp, ft.log, ft.sqrt, lambda x: x / 2)):
            with pytest.raises((TypeError, ValueError)):
                trace(function, [((2,), dtype)])
        for axes in (0, (0, 1)):
            with pytest.raises(ValueError, match='nonempty'):
                trace(lambda x: ft.amax(x, axes), [((0, 2), 'f32')])
        for specs in ((((2, 3), 'f32'), ((3, 2), 'f16')),
                      (((2, 2, 3), 'f32'), ((3, 2), 'f32')),
                      (((3,), 'f32'), ((3,), 'f32'))):
            with pytest.raises((TypeError, ValueError)):
                trace(ft.matmul, specs)


class SharedTensorPrimitiveMatrix:
    def supported(self, name):
        return name != 'bf16' or self.supports_bf16

    def assert_values(self, tensor, shape, expected, tolerance=0):
        assert (tensor.shape) == (tuple(shape))
        actual = flatten(tensor.tolist())
        assert (len(actual)) == (len(expected))
        self.assert_numbers(actual, expected, tolerance)

    def assert_numbers(self, actual, expected, tolerance=0):
        assert (len(actual)) == (len(expected))
        for value, target in zip(actual, expected):
            if math.isnan(target):
                assert (math.isnan(value))
            elif math.isinf(target):
                assert (value) == (target)
            else:
                assert (abs(value - target)) <= (tolerance * max(1, abs(target)))

    def test_geometry_reduction_comparison_matrix(self):
        for name, shape, permutations, axes_sets in geometry_matrix():
            if not self.supported(name):
                continue
            data = values(shape, name)
            x = ft.tensor(data, dtype=DTYPES[name]).reshape(shape)
            pending = [(ft.transpose(x, permutation), tuple(shape[axis] for axis in permutation),
                        transpose_reference(data, shape, permutation)) for permutation in permutations]
            broadcast = ft.broadcast_in_dim(x, (2, *shape), tuple(range(1, len(shape) + 1)))
            pending.append((broadcast, (2, *shape), data * 2))
            for predicate, reference in (('eq', lambda value: value == 0), ('ne', lambda value: value != 0),
                ('lt', lambda value: value < 0), ('le', lambda value: value <= 0),
                ('gt', lambda value: value > 0), ('ge', lambda value: value >= 0)):
                zero = False if name == 'bool' else 0
                pending.append((ft.compare(x, zero, predicate), shape, [reference(value) for value in data]))
            if name != 'bool':
                pending.append((ft.where(x > 0, x, -x), shape, [abs(value) for value in data]))
                for axes in axes_sets:
                    for maximum in (False, True):
                        if maximum and any(shape[axis] == 0 for axis in axes):
                            continue
                        target, expected = reduction_reference(data, shape, axes, maximum)
                        operation = ft.amax if maximum else ft.sum
                        pending.append((operation(x, axes), target, expected))
            ft.evaluate(*(item[0] for item in pending))
            for result, target, expected in pending:
                self.assert_values(result, target, expected)

    def test_elementary_values_gradients_and_fusion(self):
        cases = (
            ('exp', ft.exp, math.exp, math.exp, math.exp),
            ('log', ft.log, math.log, lambda x: 1 / x, lambda x: -1 / (x * x)),
            ('sqrt', ft.sqrt, math.sqrt, lambda x: 0.5 / math.sqrt(x), lambda x: -0.25 / x**1.5),
            ('div', lambda x: x / (x + 1), lambda x: x / (x + 1),
             lambda x: 1 / (x + 1)**2, lambda x: -2 / (x + 1)**3),
        )
        for name, (operation, function, reference, first, second) in product(dtypes(True), cases):
            if not self.supported(name):
                continue
            data = [0.5, 1, 2]
            x = ft.tensor(data, dtype=DTYPES[name])
            y = function(x)
            dx = ft.grad(y, x, seed=ft.ones_like(y))
            ddx = ft.grad(dx, x, seed=ft.ones_like(dx))
            before = self.engine.session.stats if self.device != 'cpu' else None
            ft.evaluate(y, dx, ddx)
            tolerance = {'fp32': 2e-5, 'fp16': 0.003, 'bf16': 0.015}[name]
            for result, oracle in ((y, reference), (dx, first), (ddx, second)):
                self.assert_values(result, (3,), [oracle(value) for value in data], tolerance)
            if self.device != 'cpu':
                assert ([step['kind'] for step in self.engine.last_execution_plan['steps']]) == (['owned'])
                lowering = self.engine.last_report['cuda_lowering']
                assert ((lowering['kernel_count'], lowering['output_loop_count'], lowering['temporary_buffer_count'])) == ((1, 1, 0))
                after = self.engine.session.stats
                assert (after['launches'] - before['launches']) == (1)
                assert (after['program_submissions'] - before['program_submissions']) == (1)
            module = trace(function, [((3,), DTYPES[name])])
            forward = execute_ir(ft.ir.jvp(module), [data, [1] * 3], self.engine)
            reverse_forward = execute_ir(ft.ir.jvp(ft.ir.vjp(module), argnums=0),
                                         [data, [1] * 3, [1] * 3], self.engine)
            for actual, oracle in ((forward[0], reference), (forward[1], first),
                                   (reverse_forward[0], first), (reverse_forward[1], second)):
                self.assert_numbers(actual, [oracle(value) for value in data], tolerance)
        x = ft.tensor([[0.25, 0.5, 1]], dtype=ft.fp32)
        mapped = ft.transpose(ft.broadcast_in_dim(x, (2, 1, 3), (1, 2)), (2, 0, 1))
        y = ft.where(mapped > 0, ft.log(ft.exp(mapped)) / ft.sqrt(mapped + 1), mapped)
        self.assert_values(y, (3, 2, 1), [value / math.sqrt(value + 1) for value in (0.25, 0.25, 0.5, 0.5, 1, 1)], 2e-6)
        if self.device != 'cpu':
            assert ([step['kind'] for step in self.engine.last_execution_plan['steps']]) == (['owned'])

    def test_broadcast_reduction_max_and_jvp(self):
        for name in dtypes(True):
            if not self.supported(name):
                continue
            x = ft.tensor([[1, 2, 2], [3, 3, 0]], dtype=DTYPES[name])
            maximum = x.amax(axis=1)
            gradient = ft.grad(maximum, x, seed=ft.ones_like(maximum))
            self.assert_values(gradient, (2, 3), [0, 0.5, 0.5, 0.5, 0.5, 0])
            second = ft.grad(gradient.sum(), x)
            self.assert_values(second, (2, 3), [0] * 6)
            total = ft.broadcast_in_dim(x, (4, 2, 3), (1, 2)).transpose((2, 0, 1)).sum()
            self.assert_values(ft.grad(total, x), (2, 3), [4] * 6)
            module = trace(lambda value: ft.sum(ft.where(value > 0, value * value, value / 2)), [((3,), name)])
            result = execute_ir(ft.ir.jvp(module), [[-2, 1, 2], [1, 1, 1]], self.engine)
            assert (result) == ([[4], [6.5]])

    def test_reduction_rounding_and_exceptional_values(self):
        large = {'fp32': 2**24, 'fp16': 2**11, 'bf16': 2**8}
        for name in dtypes(True):
            if not self.supported(name):
                continue
            x = ft.tensor([large[name], 1, -large[name]], dtype=DTYPES[name])
            self.assert_values(x.sum(), (), [0])
            if name != 'fp32':
                self.assert_values(x.sum(dtype=ft.fp32), (), [1])
            x = ft.tensor([[1, float('nan'), 3], [2, 2, 0]], dtype=DTYPES[name])
            maximum = x.amax(axis=1)
            self.assert_values(maximum, (2,), [float('nan'), 2])
            self.assert_values(ft.grad(maximum, x, seed=ft.ones_like(maximum)), (2, 3),
                               [float('nan')] * 3 + [0.5, 0.5, 0])
            x = ft.tensor([0, -1, float('inf')], dtype=DTYPES[name])
            self.assert_values(ft.log(x), (3,), [-float('inf'), float('nan'), float('inf')])
            self.assert_values(ft.sqrt(x), (3,), [0, float('nan'), float('inf')])
        for name, width in (('int32', 32), ('int64', 64)):
            if name in dtypes():
                x = ft.tensor([2**(width - 1) - 1, 1], dtype=DTYPES[name])
                self.assert_values(x.sum(), (), [-2**(width - 1)])
        if 'fp32' in dtypes():
            direct = ft.tensor([7]) / ft.tensor([3])
            self.assert_values(direct, (1,), [struct.unpack('<f', struct.pack('<f', 7 / 3))[0]])
        x = ft.tensor([[1, 2, 3], [2, 3, 4]])
        shifted = x - ft.broadcast_to(x.amax(axis=1, keepdims=True), x.shape)
        numerator = ft.exp(shifted)
        probability = numerator / ft.broadcast_to(numerator.sum(axis=1, keepdims=True), x.shape)
        expected = [math.exp(value) / sum(math.exp(v) for v in (1, 2, 3)) for value in (1, 2, 3)] * 2
        self.assert_values(probability, (2, 3), expected, 2e-6)
        if self.device != 'cpu':
            plan = self.engine.last_execution_plan
            assert (all((step['kind'] == 'owned' for step in plan['steps'])))
            assert (len(plan['steps'])) <= (3)

    def test_matmul_dtype_shape_transpose_matrix(self, tensor_matmul_case):
        name, ashape, bshape, ta, tb = tensor_matmul_case
        av, bv = values(ashape, name), values(bshape, name, 1)
        a = ft.tensor(av, dtype=DTYPES[name]).reshape(ashape)
        b = ft.tensor(bv, dtype=DTYPES[name]).reshape(bshape)
        expected_shape, expected = matmul_reference(av, bv, ashape, bshape, ta, tb)
        result = ft.matmul(a, b, transpose_a=ta, transpose_b=tb)
        self.assert_values(result, expected_shape, expected)
        if self.device != 'cpu':
            kinds = [step['kind'] for step in self.engine.last_execution_plan['steps']]
            assert (kinds.count('library')) == (1)
            assert (self.engine.last_execution_plan['steps'][kinds.index('library')]['library']) == ('cublaslt')

    def test_matmul_derivative_transpose_and_composition_matrix(self):
        for name, batch, ta, tb in product(dtypes(True), ((), (2,)), (False, True), (False, True)):
            if not self.supported(name):
                continue
            ashape = (*batch, 3, 2) if ta else (*batch, 2, 3)
            bshape = (*batch, 2, 3) if tb else (*batch, 3, 2)
            av, bv = values(ashape, name), values(bshape, name, 1)
            a, b = (ft.tensor(data, dtype=DTYPES[name]).reshape(shape) for data, shape in ((av, ashape), (bv, bshape)))
            y = ft.matmul(a, b, transpose_a=ta, transpose_b=tb)
            ga, gb = ft.grad(y.sum(), (a, b))
            ones = [1] * math.prod(y.shape)
            _, expected_a = (matmul_reference(bv, ones, bshape, y.shape, tb, True) if ta else
                             matmul_reference(ones, bv, y.shape, bshape, False, not tb))
            _, expected_b = (matmul_reference(ones, av, y.shape, ashape, True, ta) if tb else
                             matmul_reference(av, ones, ashape, y.shape, not ta, False))
            ft.evaluate(ga, gb)
            self.assert_values(ga, ashape, expected_a)
            self.assert_values(gb, bshape, expected_b)
            self.assert_values(ft.grad(ga.sum(), b), bshape, [2] * math.prod(bshape))
        a, b = ft.tensor([[1, 2], [3, 4]]), ft.tensor([[2, 0], [0, 2]])
        result = ((a + 1) * 2 @ b + 3) * 2
        before = self.engine.session.stats if self.device != 'cpu' else None
        self.assert_values(result, (2, 2), [22, 30, 38, 46])
        if self.device != 'cpu':
            assert ([step['kind'] for step in self.engine.last_execution_plan['steps']]) == (['owned', 'library', 'owned'])
            for report in self.engine.last_report['steps'].values():
                if 'cuda_lowering' in report:
                    assert ((report['cuda_lowering']['output_loop_count'], report['cuda_lowering']['temporary_buffer_count'])) == ((1, 0))
            after = self.engine.session.stats
            assert (after['launches'] - before['launches']) == (3)
            assert (after['program_submissions'] - before['program_submissions']) == (1)
        module = trace(lambda left, right: left @ right, [((2, 2), 'f32')] * 2)
        expected = [[2, 4, 6, 8], [5, 5, 9, 9]]
        for graph in ((False, True) if self.device != 'cpu' else (False,)):
            assert (execute_ir(ft.ir.jvp(module), [[1, 2, 3, 4], [2, 0, 0, 2], [1] * 4, [1] * 4], self.engine, graph=graph)) == (expected)


@pytest.mark.shared_engine
@pytest.mark.usefixtures("engine_context")
class TestHostTensorPrimitiveMatrix(SharedTensorPrimitiveMatrix):
    device = 'cpu'

    def test_materialized_mixed_dtype_matrix(self, request):
        shape = (5, 3)
        for first, following, capture_scalar in host_mixed_workspace_matrix(exhaustive=request.config.getoption("--exhaustive")):
            data = values(shape, first)
            if first in ('fp16', 'fp32'):
                data = [value + 0.5 for value in data]
            storage_types = (first, following, following)
            expected, expected_scalar = materialized_chain_reference(data, shape, storage_types)
            current = ft.tensor(data, dtype=DTYPES[first]).reshape(shape)
            captured = None
            for stage, dtype in enumerate(storage_types):
                numeric = ft.int32 if dtype == 'bool' else DTYPES[dtype]
                stored = current.astype(numeric).sum(axis=1).astype(DTYPES[dtype])
                if stage == 0 and capture_scalar:
                    captured = stored.astype(numeric).sum()
                current = ft.broadcast_in_dim(stored, shape, (0,))
            mask = current == current
            outputs = (captured, current, mask) if capture_scalar else (current, mask)
            ft.evaluate(*outputs)
            self.assert_values(current, shape, expected)
            self.assert_values(mask, shape, [True] * math.prod(shape))
            if capture_scalar:
                self.assert_values(captured, (), [expected_scalar])

    def test_materialized_workspace_matrix(self, request):
        for name, rows, columns, depth, transposed, retain in host_workspace_matrix(exhaustive=request.config.getoption("--exhaustive")):
            shape = (rows, columns)
            data = values(shape, name)
            storage_shape = shape[::-1] if transposed else shape
            stored = transpose_reference(data, shape, (1, 0)) if transposed else data
            base = ft.tensor(stored, dtype=DTYPES[name]).reshape(storage_shape)
            x = base.T if transposed else base
            current, reference = x, data
            early, early_reference = None, None
            for stage in range(depth):
                reduced = current.sum(axis=1)
                sums = [sum(reference[row * columns:(row + 1) * columns]) for row in range(rows)]
                if stage == 0:
                    early, early_reference = reduced, sums
                current = ft.broadcast_in_dim(reduced, shape, (0,)) + x
                reference = [sums[row] + data[row * columns + column]
                             for row in range(rows) for column in range(columns)]
            outputs = (current, early) if retain else (current,)
            ft.evaluate(*outputs)
            self.assert_values(current, shape, reference)
            if retain:
                self.assert_values(early, (rows,), early_reference)

    def test_bf16_computation_is_rejected(self):
        from fugatensor._ask.compiler import CompilerError
        capabilities = self.engine.compiler.capabilities()
        assert ('bf16') not in (capabilities['host_supported_element_types'])
        assert ('host_bf16_hardware_supported') not in (capabilities)
        x = ft.tensor([[1, 2], [3, 4]], dtype=ft.fp32).astype(ft.bf16)
        for operation in (ft.exp, lambda value: value.sum(), lambda value: value @ value):
            with pytest.raises(CompilerError, match='Host backend supports only bool/i32/i64/f32/f16'):
                operation(x).tolist()


@requires_cuda
@pytest.mark.shared_engine
@pytest.mark.usefixtures("engine_context")
class TestCUDATensorPrimitiveMatrix(SharedTensorPrimitiveMatrix):
    device = 'cuda'
