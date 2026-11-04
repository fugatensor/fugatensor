"""Temporary custom definitions follow pending graph ownership, independently of Engine lifetime."""
import gc
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import threading
import time
import unittest
from unittest.mock import patch
import weakref

import fugatensor as ft
from fugatensor.ir import Module, TensorType
from tests._helpers import EngineTestCase
from tests.support.arena_ownership import DEFINITIONS, concurrent_output, concurrent_references
from tests.support.tensor_primitives import flatten, values


class ArenaOwnershipTests(EngineTestCase):
    def test_concurrent_projection_publishes_one_slice(self):
        from fugatensor import _lazy
        module = Module()
        graph = module.graph('pair', inputs={'x': TensorType((2,), 'f32')})
        value = graph.inputs['x']
        graph.return_(graph.add(value, value), graph.mul(value, value))
        template = _lazy._template(module)
        ready = threading.Barrier(2)
        original_project = _lazy._project
        def concurrent_project(*args):
            ready.wait(timeout=10)
            return original_project(*args)
        with patch.object(_lazy, '_project', side_effect=concurrent_project):
            with ThreadPoolExecutor(max_workers=2) as workers:
                selected = tuple(workers.map(template.project, ((0,), (0,))))
        self.assertIs(selected[0], selected[1])
        self.assertIs(template.project((0,)), selected[0])
        self.assertEqual(len(selected[0].module.functions[0].outputs), 1)

    def test_concurrent_cold_primitives_share_one_engine(self):
        from fugatensor import _lazy
        shape = (2, 3)
        data = values(shape)
        with ft.Engine(cache_dir=self.cache_dir / 'concurrent-cold') as engine:
            x = engine.tensor(data, shape=shape)
            start = threading.Barrier(4)
            original_project = _lazy._project
            def yield_project(*args, **kwargs):
                # Yield while cold metadata is being prepared, outside execution.
                time.sleep(0.001)
                return original_project(*args, **kwargs)
            def construct(branch):
                start.wait()
                return concurrent_output(x, branch)
            with patch.object(_lazy, '_project', side_effect=yield_project):
                with ThreadPoolExecutor(max_workers=4) as workers:
                    outputs = tuple(workers.map(construct, range(4)))
            self.assertEqual(engine.stats['executions'], 0)
            ft.evaluate(*outputs)
            self.assertEqual(engine.stats['executions'], 1)
            for output, (expected_shape, expected) in zip(outputs, concurrent_references(data, shape)):
                self.assertEqual(output.shape, expected_shape)
                self.assertEqual(output.dtype, ft.fp32)
                self.assertEqual(flatten(output.tolist()), expected)

    def test_discarded_custom_graphs_release_definitions_with_engine_alive(self):
        x = ft.tensor(2)
        before = self.engine.stats
        for definition, retain in product(DEFINITIONS, (False, True)):
            with self.subTest(definition=definition.__name__, retain=retain):
                kept = None
                if retain:
                    operation = definition(2)
                    kept = operation(x)
                    kept_definition = weakref.ref(operation)
                    del operation
                stable_counts = None
                for epoch in range(4):
                    references = []
                    for index in range(16):
                        operation = definition(1 + (epoch + index) % 3)
                        output = operation(x)
                        references.extend((weakref.ref(operation), weakref.ref(output)))
                        del operation, output
                    gc.collect()
                    self.assertTrue(all(reference() is None for reference in references))
                    if retain:
                        self.assertIsNotNone(kept_definition())
                    stats = self.engine._arena.stats
                    counts = tuple(stats[name] for name in (
                        'live_templates', 'live_slices', 'live_dependency_slots', 'live_exact_keys'))
                    if stable_counts is None:
                        stable_counts = counts
                    else:
                        self.assertEqual(counts, stable_counts)
                del kept
                gc.collect()
                if retain:
                    self.assertIsNone(kept_definition())
                self.assertFalse(self.engine.closed)
        self.assertEqual(self.engine.stats['executions'], before['executions'])
        self.assertEqual(self.engine.stats['compilations'], before['compilations'])


if __name__ == '__main__':
    unittest.main()
