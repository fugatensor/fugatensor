"""Generated numerical/AD/fusion matrices for the orthogonal tensor basis."""
from itertools import product
import math
import struct
import unittest

import fugatensor as ft
from fugatensor.ir import Module, ScalarType, TensorType
from tests._helpers import EngineTestCase, requires_cuda
from tests.support.tensor_primitives import (DTYPES, dtypes, execute_ir, flatten, geometry_matrix,
    matmul_matrix, matmul_reference, reduction_reference, trace, transpose_reference, values)


class TensorPrimitiveContractTests(unittest.TestCase):
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
                self.assertNotIn('fuga.external', transformed.to_text())
                self.assertIn('ask.', ft.lower_to_ask(transformed).to_mlir())
        module = Module()
        graph = module.graph('scalar', inputs={'x': ScalarType('f32')})
        graph.return_(graph.splat(graph.inputs['x'], (2, 3)))
        derivative = ft.ir.vjp(module)
        self.assertEqual(derivative.functions[0].outputs[0].type, ScalarType('f32'))
        self.assertIn('fuga.reduce_sum', derivative.to_text())

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
            with self.subTest(function=function), self.assertRaises((TypeError, ValueError)):
                trace(function, [((2, 3), 'f32')])
        for dtype, function in product(('bool', 'i32', 'i64'), (ft.exp, ft.log, ft.sqrt, lambda x: x / 2)):
            with self.subTest(dtype=dtype, function=function), self.assertRaises((TypeError, ValueError)):
                trace(function, [((2,), dtype)])
        for axes in (0, (0, 1)):
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                trace(lambda x: ft.amax(x, axes), [((0, 2), 'f32')])
        for specs in ((((2, 3), 'f32'), ((3, 2), 'f16')),
                      (((2, 2, 3), 'f32'), ((3, 2), 'f32')),
                      (((3,), 'f32'), ((3,), 'f32'))):
            with self.subTest(specs=specs), self.assertRaises((TypeError, ValueError)):
                trace(ft.matmul, specs)


class SharedTensorPrimitiveMatrix:
    def supported(self, name):
        return name != 'bf16' or self.supports_bf16

    def assert_values(self, tensor, shape, expected, tolerance=0):
        self.assertEqual(tensor.shape, tuple(shape))
        actual = flatten(tensor.tolist())
        self.assertEqual(len(actual), len(expected))
        self.assert_numbers(actual, expected, tolerance)

    def assert_numbers(self, actual, expected, tolerance=0):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            if math.isnan(target):
                self.assertTrue(math.isnan(value))
            elif math.isinf(target):
                self.assertEqual(value, target)
            else:
                self.assertLessEqual(abs(value - target), tolerance * max(1, abs(target)))

    def test_geometry_reduction_comparison_matrix(self):
        for name, shape, permutations, axes_sets in geometry_matrix():
            if not self.supported(name):
                continue
            with self.subTest(dtype=name, shape=shape):
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
            with self.subTest(dtype=name, operation=operation):
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
                    self.assertEqual([step['kind'] for step in self.engine.last_execution_plan['steps']], ['owned'])
                    lowering = self.engine.last_report['cuda_lowering']
                    self.assertEqual((lowering['kernel_count'], lowering['output_loop_count'],
                                      lowering['temporary_buffer_count']), (1, 1, 0))
                    after = self.engine.session.stats
                    self.assertEqual(after['launches'] - before['launches'], 1)
                    self.assertEqual(after['program_submissions'] - before['program_submissions'], 1)
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
            self.assertEqual([step['kind'] for step in self.engine.last_execution_plan['steps']], ['owned'])

    def test_broadcast_reduction_max_and_jvp(self):
        for name in dtypes(True):
            if not self.supported(name):
                continue
            with self.subTest(dtype=name):
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
                self.assertEqual(result, [[4], [6.5]])

    def test_reduction_rounding_and_exceptional_values(self):
        large = {'fp32': 2**24, 'fp16': 2**11, 'bf16': 2**8}
        for name in dtypes(True):
            if not self.supported(name):
                continue
            with self.subTest(dtype=name):
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
            self.assertTrue(all(step['kind'] == 'owned' for step in plan['steps']))
            self.assertLessEqual(len(plan['steps']), 3)

    def test_matmul_dtype_shape_transpose_matrix(self):
        for name, ashape, bshape, ta, tb in matmul_matrix():
            if not self.supported(name):
                continue
            with self.subTest(dtype=name, a=ashape, b=bshape, ta=ta, tb=tb):
                av, bv = values(ashape, name), values(bshape, name, 1)
                a = ft.tensor(av, dtype=DTYPES[name]).reshape(ashape)
                b = ft.tensor(bv, dtype=DTYPES[name]).reshape(bshape)
                expected_shape, expected = matmul_reference(av, bv, ashape, bshape, ta, tb)
                result = ft.matmul(a, b, transpose_a=ta, transpose_b=tb)
                self.assert_values(result, expected_shape, expected)
                if self.device != 'cpu':
                    kinds = [step['kind'] for step in self.engine.last_execution_plan['steps']]
                    self.assertEqual(kinds.count('library'), 1)
                    self.assertEqual(self.engine.last_execution_plan['steps'][kinds.index('library')]['library'], 'cublaslt')

    def test_matmul_derivative_transpose_and_composition_matrix(self):
        for name, batch, ta, tb in product(dtypes(True), ((), (2,)), (False, True), (False, True)):
            if not self.supported(name):
                continue
            with self.subTest(dtype=name, batch=batch, ta=ta, tb=tb):
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
            self.assertEqual([step['kind'] for step in self.engine.last_execution_plan['steps']], ['owned', 'library', 'owned'])
            for report in self.engine.last_report['steps'].values():
                if 'cuda_lowering' in report:
                    self.assertEqual((report['cuda_lowering']['output_loop_count'],
                                      report['cuda_lowering']['temporary_buffer_count']), (1, 0))
            after = self.engine.session.stats
            self.assertEqual(after['launches'] - before['launches'], 3)
            self.assertEqual(after['program_submissions'] - before['program_submissions'], 1)
        module = trace(lambda left, right: left @ right, [((2, 2), 'f32')] * 2)
        expected = [[2, 4, 6, 8], [5, 5, 9, 9]]
        for graph in ((False, True) if self.device != 'cpu' else (False,)):
            self.assertEqual(execute_ir(ft.ir.jvp(module), [[1, 2, 3, 4], [2, 0, 0, 2], [1] * 4, [1] * 4], self.engine, graph=graph), expected)


class HostTensorPrimitiveMatrix(SharedTensorPrimitiveMatrix, EngineTestCase):
    device = 'cpu'

    def test_bf16_computation_is_rejected(self):
        from fugatensor._ask.compiler import CompilerError
        capabilities = self.engine.compiler.capabilities()
        self.assertNotIn('bf16', capabilities['host_supported_element_types'])
        self.assertNotIn('host_bf16_hardware_supported', capabilities)
        x = ft.tensor([[1, 2], [3, 4]], dtype=ft.bf16)
        for operation in (ft.exp, lambda value: value.sum(), lambda value: value @ value):
            with self.subTest(operation=operation), self.assertRaisesRegex(CompilerError, 'bf16 is not supported by the Host backend'):
                operation(x).tolist()


@requires_cuda
class CUDATensorPrimitiveMatrix(SharedTensorPrimitiveMatrix, EngineTestCase):
    device = 'cuda'
