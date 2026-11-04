"""Public lazy-graph scheduling, output demand, and cache binding contracts."""
from itertools import product
import sys
import unittest

import fugatensor as ft
from tests._helpers import EngineTestCase
from tests.support.lazy_graphs import RepeatedPorts, SharedPorts, port_values, selection_matrix
from tests.support.tensor_primitives import DTYPES, flatten


class LazyScheduleTests(EngineTestCase):
    def test_output_subset_alias_input_binding_and_cache_matrix(self):
        operation = SharedPorts()
        for name, shape, shared_input, aliases, selected in selection_matrix():
            with self.subTest(dtype=name, shape=shape, shared_input=shared_input,
                              aliases=aliases, outputs=selected):
                retained = []
                for phase in (0, 1):
                    left, right, expected = port_values(shape, name, shared_input, phase)
                    x = self.engine.tensor(left, shape=shape, dtype=DTYPES[name])
                    y = x if shared_input else self.engine.tensor(right, shape=shape, dtype=DTYPES[name])
                    ports = operation(x, y)
                    roots = tuple(ports[index].detach() if aliases else ports[index] for index in selected)
                    before = self.engine.stats
                    result = ft.evaluate(*roots, roots[0])
                    after = self.engine.stats
                    self.assertEqual(after['executions'], before['executions'] + 1)
                    if phase:
                        self.assertEqual(after['compilations'], before['compilations'])
                        self.assertEqual(after['cache_hits'], before['cache_hits'] + 1)
                    self.assertIs(result[-1], roots[0])
                    for root, index in zip(roots, selected):
                        self.assertEqual(root.shape, shape)
                        self.assertEqual(root.dtype, DTYPES[name])
                        self.assertEqual(flatten(root.tolist()), expected[index])
                        self.assertIs(root.storage, ports[index].storage)
                        retained.append((root, expected[index]))
                    for index, port in enumerate(ports):
                        if index not in selected:
                            self.assertIsNone(port._storage)
                for root, expected in retained:
                    self.assertEqual(flatten(root.tolist()), expected)

    def test_output_order_changes_bindings(self):
        operation = SharedPorts()
        for order in ((0, 2, 1), (2, 0, 1), (1, 2, 0)):
            with self.subTest(order=order):
                left, right, expected = port_values((3,), 'fp32', False, 1)
                outputs = operation(ft.tensor(left), ft.tensor(right))
                roots = tuple(outputs[index] for index in order)
                ft.evaluate(*roots)
                self.assertEqual(tuple(flatten(root.tolist()) for root in roots),
                                 tuple(expected[index] for index in order))

    def test_repeated_fun_result_and_input_return(self):
        operation = RepeatedPorts()
        for aliases in (False, True):
            with self.subTest(aliases=aliases):
                x = ft.tensor([2, 3])
                outputs = operation(x)
                roots = tuple(value.detach() if aliases else value for value in outputs)
                before = self.engine.stats['executions']
                ft.evaluate(*roots)
                self.assertEqual(self.engine.stats['executions'], before + 1)
                self.assertEqual(tuple(root.tolist() for root in roots), ([4, 9], [4, 9], [2, 3]))
                self.assertEqual(x.tolist(), [2, 3])

    def test_constant_carrier_with_pending_alias_and_changed_shape(self):
        for shape, aliases, (factory, fill) in product(
                ((2, 3), (6,)), (False, True), ((ft.zeros_like, 0), (ft.ones_like, 1))):
            with self.subTest(shape=shape, aliases=aliases, factory=factory.__name__):
                x = self.engine.tensor(list(range(6)), shape=(2, 3))
                pending = x + 7
                source = pending.detach() if aliases else pending
                constant = factory(source.reshape(shape))
                before = self.engine.stats['executions']
                ft.evaluate(constant)
                self.assertEqual(self.engine.stats['executions'], before + 1)
                self.assertEqual(constant.shape, shape)
                self.assertEqual(flatten(constant.tolist()), [fill] * 6)
                self.assertIsNone(pending._storage)

    def test_materialized_intermediate_keeps_derivative_provenance(self):
        x = ft.tensor([2, 3])
        intermediate = x * x
        ft.evaluate(intermediate)
        stored = intermediate.storage
        result = intermediate + x
        derivative = ft.grad(result, x, seed=ft.ones_like(result))
        before = self.engine.stats['executions']
        ft.evaluate(result, derivative)
        self.assertEqual(self.engine.stats['executions'], before + 1)
        self.assertEqual(result.tolist(), [6, 12])
        self.assertEqual(derivative.tolist(), [5, 7])
        self.assertIs(intermediate.storage, stored)

    def test_deep_alias_and_expression_chains(self):
        depth = sys.getrecursionlimit() + 33
        for aliases in (False, True):
            with self.subTest(aliases=aliases, depth=depth):
                x = ft.tensor(2, dtype=ft.int32)
                value = x + 1
                for _ in range(depth):
                    value = value.detach() if aliases else value + 1
                before = self.engine.stats['executions']
                ft.evaluate(value)
                self.assertEqual(self.engine.stats['executions'], before + 1)
                self.assertEqual(value.tolist(), 3 if aliases else 3 + depth)


if __name__ == '__main__':
    unittest.main()
