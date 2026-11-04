"""Native SSA planning contracts, independent of devices and ASK execution."""

import pytest
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import EXTENSION_SUFFIXES
import sys

from fugatensor import _native


class TestNativeGraph:
    def analyze(self, inputs=0, operands=(), results=(), outputs=(), selected=(), effects=None):
        if effects is None:
            effects = [False] * len(operands)
        return _native.analyze(inputs, operands, results, outputs, selected, effects)

    def test_loaded_module_is_required_native_extension(self):
        assert (any((_native.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES)))
        assert (_native.graph_api_version) == (1)

    def test_inactive_coefficient_is_live_without_its_reverse_rule(self):
        # Inputs x, y; coefficient=f(y); output=x*coefficient; dead=external(x).
        plan = self.analyze(2, [[1], [0, 2], [0]], [1, 1, 1], [3], [0])
        assert (plan) == ({'active_values': [True, False, False, True, True], 'reverse_operations': [1], 'live_operations': [0, 1], 'value_count': 5, 'operation_count': 3})
        assert (all((type(flag) is bool for flag in plan['active_values'])))

    def test_reconverging_paths_visit_shared_operation_once(self):
        # a=x*x; b=a+x; c=a*x; output=b+c. The two contributions to a meet.
        plan = self.analyze(1, [[0, 0], [1, 0], [1, 0], [2, 3]], [1] * 4, [4], [0])
        assert (plan['reverse_operations']) == ([3, 2, 1, 0])
        assert (plan['live_operations']) == ([0, 1, 2, 3])
        assert (plan['active_values']) == ([True] * 5)

    def test_multiresult_call_is_one_opaque_reverse_boundary(self):
        # Both outputs of custom(x,y) conservatively depend on both operands.
        plan = self.analyze(2, [[0, 1], [3, 1]], [2, 1], [2, 4, 2], [0])
        assert (plan['active_values']) == ([True, False, True, True, True])
        assert (plan['reverse_operations']) == ([1, 0])
        assert (plan['live_operations']) == ([0, 1])

    def test_effects_keep_their_dependencies_without_creating_reverse_paths(self):
        plan = self.analyze(1, [[], [1], [2], [0], [0]], [1, 1, 0, 1, 0], [], [0],
                            [False, False, True, False, True])
        assert (plan['reverse_operations']) == ([])
        assert (plan['live_operations']) == ([0, 1, 2, 4])
        assert (plan['active_values']) == ([True, False, False, True])

    def test_effectful_results_need_normal_output_path_for_reverse(self):
        plan = self.analyze(1, [[0], [0]], [1, 1], [2], [0], [True, False])
        assert (plan['reverse_operations']) == ([1])
        assert (plan['live_operations']) == ([0, 1])

    def test_constants_and_unselected_outputs_need_no_reverse_rules(self):
        plan = self.analyze(2, [[], [1, 2]], [1, 1], [3], [0])
        assert (plan['active_values']) == ([True, False, False, False])
        assert (plan['reverse_operations']) == ([])
        assert (plan['live_operations']) == ([0, 1])

    def test_empty_selected_inputs_produces_ordinary_dce(self):
        plan = self.analyze(1, [[0], [1], [0]], [1, 1, 1], [2])
        assert (plan['active_values']) == ([False] * 4)
        assert (plan['reverse_operations']) == ([])
        assert (plan['live_operations']) == ([0, 1])

    def test_identity_and_empty_graphs_have_no_operation_schedule(self):
        identity = self.analyze(2, outputs=[0, 0, 1], selected=[0])
        assert (identity['active_values']) == ([True, False])
        assert (identity['reverse_operations']) == ([])
        assert (identity['live_operations']) == ([])
        empty = self.analyze()
        assert (empty['active_values']) == ([])
        assert (empty['value_count']) == (0)
        assert (empty['operation_count']) == (0)

    def test_keyword_arguments_and_tuple_input(self):
        plan = _native.analyze(input_count=1, operands=((0,),), result_counts=(1,),
                               outputs=(1,), selected_inputs=(0,), effects=(False,))
        assert (plan['reverse_operations']) == ([0])

    def test_invalid_ssa_references_are_rejected(self):
        cases = [
            ("self reference", dict(inputs=1, operands=[[1]], results=[1])),
            ("forward reference", dict(inputs=1, operands=[[2], [0]], results=[1, 1])),
            ("no defined values", dict(operands=[[0]], results=[1])),
            ("missing output", dict(inputs=1, outputs=[1])),
            ("selected result", dict(inputs=1, operands=[[0]], results=[1], selected=[1])),
            ("duplicate selected", dict(inputs=1, selected=[0, 0])),
        ]
        for description, kwargs in cases:
            with pytest.raises(ValueError):
                self.analyze(**kwargs)

    def test_invalid_counts_and_effect_shapes_are_rejected(self):
        for kwargs in (
            dict(operands=[[]], results=[]),
            dict(operands=[], results=[1]),
            dict(operands=[[]], results=[1], effects=[]),
            dict(operands=[], results=[], effects=[False]),
            dict(inputs=sys.maxsize + 1),
            dict(inputs=sys.maxsize, operands=[[]], results=[1]),
        ):
            with pytest.raises(ValueError):
                self.analyze(**kwargs)

    def test_invalid_python_types_do_not_coerce_silently(self):
        cases = [
            dict(inputs=True), dict(inputs=-1), dict(inputs=1.0), dict(inputs=1 << 100),
            dict(operands=[[True]], results=[1]),
            dict(operands=[[-1]], results=[1]),
            dict(operands=[[]], results=[False]),
            dict(operands=[[]], results=[-1]),
            dict(outputs=[False]), dict(selected=[0.0]),
            dict(operands=[[]], results=[1], effects=[1]),
            dict(operands=[[]], results=[1], effects=[None]),
            dict(operands=[None], results=[1]),
            dict(outputs="0"), dict(selected=iter(())),
        ]
        for kwargs in cases:
            with pytest.raises(ValueError):
                self.analyze(**kwargs)

    def test_parallel_requests_do_not_share_analysis_state(self):
        operations = 4000
        operands = [[index] for index in range(operations)]

        def plan(selected):
            return self.analyze(1, operands, [1] * operations, [operations], selected)

        with ThreadPoolExecutor(max_workers=4) as executor:
            plans = list(executor.map(plan, ([0], [], [0], [], [0], [])))
        for index, result in enumerate(plans):
            assert (result['live_operations']) == (list(range(operations)))
            assert (result['reverse_operations']) == (list(reversed(range(operations))) if index % 2 == 0 else [])
