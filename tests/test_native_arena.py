"""Native graph identity, storage boundaries, and handle lifetime contracts."""

import pytest
from array import array
import gc
from itertools import product
import weakref

import fugatensor as ft
from fugatensor import _native
from tests.support.arena_graphs import bindings, identity_matrix
from tests.support.tensor_primitives import DTYPES, flatten, values


@pytest.mark.usefixtures("engine_context")
class TestNativeArena:
    def test_graph_identity_and_materialization_matrix(self):
        for dtype, shape, layout, shared, reverse, cut in identity_matrix():
            retained = []
            for phase in (0, 1):
                x = self.engine.tensor(values(shape, dtype, phase), shape=shape,
                                       dtype=DTYPES[dtype])
                y = x if shared else self.engine.tensor(
                    values(shape, dtype, phase + 2), shape=shape, dtype=DTYPES[dtype])
                if layout == 'transposed':
                    x, y = x.T, y.T
                left, right = x * 2, y * 3
                if cut == 'left':
                    ft.evaluate(left)
                elif cut == 'both':
                    ft.evaluate(left, right)
                outputs = (left - right, left + right)
                expected = bindings(shape, dtype, layout, shared, phase)
                if reverse:
                    outputs, expected = outputs[::-1], expected[::-1]
                before = self.engine.stats
                ft.evaluate(*outputs)
                after = self.engine.stats
                assert (after['executions']) == (before['executions'] + 1)
                if phase:
                    assert (after['compilations']) == (before['compilations'])
                    assert (after['cache_hits']) == (before['cache_hits'] + 1)
                for output, reference in zip(outputs, expected):
                    assert (output.dtype) == (DTYPES[dtype])
                    assert (output.shape) == (shape[::-1] if layout == 'transposed' else shape)
                    assert (tuple(flatten(output.tolist()))) == (reference)
                    retained.append((output, reference))
            for output, reference in retained:
                assert (tuple(flatten(output.tolist()))) == (reference)

    def test_materializing_an_existing_parent_changes_the_numerical_frontier(self):
        for (dtype, code), detached in product(
                ((ft.fp32, 'f'), (ft.int32, 'i')), (False, True)):
            for phase in range(2):
                initial = [2 + phase, 3 + phase]
                source = array(code, initial)
                x = ft.from_buffer(source, shape=(2,), dtype=dtype, copy=False)
                intermediate = x * x
                parent = intermediate.detach() if detached else intermediate
                result = parent + x if phase == 0 else None
                ft.evaluate(intermediate)
                if result is None:
                    result = parent + x
                source[0], source[1] = 5 + phase, 7 + phase
                before = self.engine.stats
                assert (result.tolist()) == ([a * a + b for a, b in zip(initial, source)])
                if phase:
                    assert (self.engine.stats['compilations']) == (before['compilations'])
                    assert (self.engine.stats['cache_hits']) == (before['cache_hits'] + 1)

    def test_native_edges_preserve_gradients_after_wrappers_are_released(self):
        x = ft.tensor([2, 3])
        intermediate = x * x
        reference = weakref.ref(intermediate)
        output = intermediate + intermediate
        del intermediate
        gc.collect()
        assert (reference()) is None
        derivative = ft.grad(output, x, seed=ft.ones_like(output))
        ft.evaluate(output, derivative)
        assert (output.tolist()) == ([8, 18])
        assert (derivative.tolist()) == ([8, 12])

    def test_equal_fingerprints_require_exact_graph_identity(self):
        self.engine.clear_cache()
        x = ft.tensor([7, 8], dtype=ft.int32)
        first, second = x + (-1), x + (-2)
        first_key = _native.collect_pending((first,))[-1]
        second_key = _native.collect_pending((second,))[-1]
        assert (hash(first_key)) == (hash(second_key))
        assert (first_key) != (second_key)
        before = self.engine.stats['compilations']
        assert (first.tolist()) == ([6, 7])
        assert (second.tolist()) == ([5, 6])
        assert (self.engine.stats['compilations']) == (before + 2)

    def test_closed_engine_and_pending_handles_are_collectible(self):
        engine = ft.Engine(cache_dir=self.cache_dir / 'closed-arena')
        x = engine.tensor(2)
        output = (x + 1) * 3
        references = tuple(weakref.ref(value) for value in (engine, x, output))
        engine.close()
        with pytest.raises(RuntimeError, match='closed'):
            output.tolist()
        del output, x, engine
        gc.collect()
        assert (all((reference() is None for reference in references)))

    def test_node_slots_reuse_while_older_outputs_keep_their_history(self):
        x = ft.tensor([2, 3])
        saved = x * x + x
        ft.evaluate(saved)
        warm = (x + 1) * 2
        del warm
        gc.collect()
        baseline = self.engine._arena.stats
        capacities = []
        for _ in range(4):
            for _ in range(128):
                pending = x
                for _ in range(32):
                    pending = pending + 1
                del pending
            current = self.engine._arena.stats
            assert (current['live_nodes']) == (baseline['live_nodes'])
            assert (current['live_edges']) == (baseline['live_edges'])
            capacities.append((current['value_capacity'], current['expression_capacity'],
                               current['edge_slots']))
        assert (all((capacity == capacities[0] for capacity in capacities)))
        derivative = ft.grad(saved, x, seed=ft.ones_like(saved))
        assert (saved.tolist()) == ([6, 12])
        assert (derivative.tolist()) == ([5, 7])
