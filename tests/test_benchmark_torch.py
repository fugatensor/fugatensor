"""Lightweight contracts for recorded benchmarks; no tensor backend required."""
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
import unittest

from tools.benchmark_torch import check_output, comparisons, percentile
from tools.benchmark_workloads import Case, cases, expected


@dataclass
class RecordedTensor:
    shape: tuple
    dtype: str
    values: list

    def tolist(self):
        return self.values


class BenchmarkMatrixTests(unittest.TestCase):
    def test_only_relevant_axes_expand_each_workload(self):
        sizes = ('small', 'large')
        dtypes = ('fp32', 'fp16')
        layouts = ('row_major', 'transpose')
        depths, branches = (2, 4), (2, 3)
        generated = list(cases(sizes=sizes, dtypes=dtypes, layouts=layouts,
                               depths=depths, branches=branches))
        self.assertEqual(len(generated), len(set(generated)))
        base = len(sizes) * len(dtypes) * len(layouts)
        self.assertEqual(Counter(case.workload for case in generated), {
            'elementwise': base, 'matmul': base,
            'reduction_chain': base * len(depths), 'fork_join': base * len(branches),
        })
        for workload in ('elementwise', 'reduction_chain', 'fork_join', 'matmul'):
            selected = [case for case in generated if case.workload == workload]
            self.assertEqual({(case.size, case.dtype, case.layout) for case in selected},
                             set(product(sizes, dtypes, layouts)))
            self.assertEqual({case.depth for case in selected},
                             set(depths) if workload == 'reduction_chain' else {0})
            self.assertEqual({case.branches for case in selected},
                             set(branches) if workload == 'fork_join' else {1})

    def test_independent_references_on_small_exact_inputs(self):
        references = {
            'elementwise': ((2, 3), [-.5, .75, -.125, -.125, -.5, 1.375]),
            'reduction_chain': ((2, 3), [3, 8, 6, 9, 7, 12]),
            'fork_join': ((2, 3), [0, 3, 6, 7, 3, 6]),
            'matmul': ((2, 2), [.75, .5, .375, .25]),
        }
        for workload, layout in product(references, ('row_major', 'transpose')):
            shape, values = references[workload]
            case = Case(workload, 'small', 'fp32', layout, *shape,
                        depth=3 if workload == 'reduction_chain' else 0,
                        branches=2 if workload == 'fork_join' else 1)
            with self.subTest(workload=workload, layout=layout):
                self.assertEqual(expected(case), (shape, values))


class BenchmarkResultTests(unittest.TestCase):
    def test_wrong_values_and_output_contracts_are_rejected(self):
        case = Case('elementwise', 'small', 'fp32', 'row_major', 2, 3)
        values = [[-.5, .75, -.125], [-.125, -.5, 1.375]]
        self.assertEqual(check_output(RecordedTensor((2, 3), 'fp32', values), case, 'fp32'), 0)
        for value in (4, float('nan'), float('inf')):
            wrong = [[value, *values[0][1:]], values[1]]
            with self.subTest(value=value), self.assertRaisesRegex(AssertionError, 'numerical mismatch'):
                check_output(RecordedTensor((2, 3), 'fp32', wrong), case, 'fp32')
        for shape, dtype in (((3, 2), 'fp32'), ((2, 3), 'fp16')):
            with self.subTest(shape=shape, dtype=dtype), self.assertRaisesRegex(AssertionError, 'output contract'):
                check_output(RecordedTensor(shape, dtype, values), case, 'fp32')
        with self.assertRaisesRegex(AssertionError, 'element count'):
            check_output(RecordedTensor((2, 3), 'fp32', values[:1]), case, 'fp32')

    def test_ratio_direction_round_aggregation_and_failure_exclusion(self):
        identity = {'device': 'cpu', **asdict(Case('elementwise', 'small', 'fp32', 'row_major', 2, 3))}
        rows = []
        for backend, times in (('fuga', (100, 10, 20)), ('torch_eager', (8, 2, 4)),
                               ('torch_compile', (15, 5, 10))):
            for round_, value in enumerate(times):
                rows.append(identity | {'round': round_, 'backend': backend,
                                        'status': 'ok', 'complete_p50_us': value})
        rows.append(identity | {'round': 3, 'backend': 'torch_eager', 'status': 'error', 'error': 'failed'})
        rows.append(identity | {'round': 3, 'backend': 'fuga', 'status': 'ok', 'complete_p50_us': 1000})
        # A Torch-only shape must not be paired with a different Fuga shape.
        rows.append(identity | {'rows': 9, 'round': 0, 'backend': 'torch_eager',
                                'status': 'ok', 'complete_p50_us': 1})
        rows.extend(identity | {'device': 'cuda', 'round': 0, 'backend': backend,
                                'status': 'ok', 'complete_p50_us': value}
                    for backend, value in (('fuga', 4), ('torch_eager', 8)))
        result = {(row['device'], row['torch_mode']): row for row in comparisons(rows)}
        self.assertEqual(len(result), 3)
        eager = result['cpu', 'torch_eager']
        self.assertEqual((eager['fuga_p50_us'], eager['torch_p50_us']), (20, 4))
        self.assertEqual((eager['fuga_over_torch'], eager['difference_pct']), (5, 400))
        self.assertEqual((eager['fuga_rounds'], eager['torch_rounds']), (3, 3))
        compiled = result['cpu', 'torch_compile']
        self.assertEqual((compiled['fuga_over_torch'], compiled['difference_pct']), (2, 100))
        gpu = result['cuda', 'torch_eager']
        self.assertEqual((gpu['fuga_over_torch'], gpu['difference_pct']), (.5, -50))

    def test_percentile_preserves_units_and_interpolates_small_samples(self):
        self.assertEqual(percentile([9, 1, 8, 2], .5), 5)
        self.assertAlmostEqual(percentile([9, 1, 8, 2], .9), 8.7)
        self.assertEqual(percentile([7], .9), 7)


if __name__ == '__main__':
    unittest.main()
