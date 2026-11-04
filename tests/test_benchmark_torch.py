"""Lightweight contracts for recorded benchmarks; no tensor backend required."""

import pytest
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
import math

from tools.benchmark_torch import check_output, comparisons, percentile
from tools.benchmark_workloads import Case, cases, expected, input_data


@dataclass
class RecordedTensor:
    shape: tuple
    dtype: str
    values: list

    def tolist(self):
        return self.values


class TestBenchmarkMatrix:
    def test_only_relevant_axes_expand_each_workload(self):
        sizes = ('small', 'large')
        dtypes = ('fp32', 'fp16')
        layouts = ('row_major', 'transpose')
        depths, branches = (2, 4), (2, 3)
        generated = list(cases(sizes=sizes, dtypes=dtypes, layouts=layouts,
                               depths=depths, branches=branches))
        assert (len(generated)) == (len(set(generated)))
        base = len(sizes) * len(dtypes) * len(layouts)
        assert (Counter((case.workload for case in generated))) == ({'elementwise': base, 'matmul': base, 'reduction_chain': base * len(depths), 'fork_join': base * len(branches)})
        for workload in ('elementwise', 'reduction_chain', 'fork_join', 'matmul'):
            selected = [case for case in generated if case.workload == workload]
            assert ({(case.size, case.dtype, case.layout) for case in selected}) == (set(product(sizes, dtypes, layouts)))
            assert ({case.depth for case in selected}) == (set(depths) if workload == 'reduction_chain' else {0})
            assert ({case.branches for case in selected}) == (set(branches) if workload == 'fork_join' else {1})

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
            assert (expected(case)) == ((shape, values))

    def test_explicit_matmul_sizes_do_not_expand_other_workloads(self):
        generated = tuple(cases(matmul_sizes=(3, 11, 17), matmul_data='general', layouts=('row_major', 'transpose')))
        matrices = [case for case in generated if case.workload == 'matmul']
        assert len(matrices) == 6
        assert {case.shape for case in matrices} == {(n, n) for n in (3, 11, 17)}
        assert {case.size for case in matrices} == {'n3', 'n11', 'n17'}
        assert all(case.matmul_data == 'general' for case in matrices)
        other = [case for case in generated if case.workload != 'matmul']
        assert all(case.matmul_data == 'dyadic' for case in other)
        assert {case.shape for case in other} == {(32, 64), (512, 256)}

    @pytest.mark.parametrize('dtype,layout', tuple(product(('fp32', 'fp16'), ('row_major', 'transpose'))))
    def test_general_matmul_matches_independent_dot_products(self, dtype, layout):
        case = Case('matmul', 'n5', dtype, layout, 3, 5, matmul_data='general')
        left, right = input_data(case)
        if layout == 'transpose':
            left, right = ([list(row) for row in zip(*matrix)] for matrix in (left, right))
        reference = [math.fsum(left[row][k] * right[k][column] for k in range(5))
                     for row in range(3) for column in range(5)]
        shape, result = expected(case)
        assert shape == (3, 5)
        assert result == reference
        assert any(value * 1024 != int(value * 1024) for row in left for value in row)


class TestBenchmarkResult:
    def test_wrong_values_and_output_contracts_are_rejected(self):
        case = Case('elementwise', 'small', 'fp32', 'row_major', 2, 3)
        values = [[-.5, .75, -.125], [-.125, -.5, 1.375]]
        assert (check_output(RecordedTensor((2, 3), 'fp32', values), case, 'fp32')) == (0)
        for value in (4, float('nan'), float('inf')):
            wrong = [[value, *values[0][1:]], values[1]]
            with pytest.raises(AssertionError, match='numerical mismatch'):
                check_output(RecordedTensor((2, 3), 'fp32', wrong), case, 'fp32')
        for shape, dtype in (((3, 2), 'fp32'), ((2, 3), 'fp16')):
            with pytest.raises(AssertionError, match='output contract'):
                check_output(RecordedTensor(shape, dtype, values), case, 'fp32')
        with pytest.raises(AssertionError, match='element count'):
            check_output(RecordedTensor((2, 3), 'fp32', values[:1]), case, 'fp32')

    def test_ratio_direction_round_aggregation_and_failure_exclusion(self):
        identity = {'device': 'cpu', 'matmul_precision': 'ieee', **asdict(Case('elementwise', 'small', 'fp32', 'row_major', 2, 3))}
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
        assert (len(result)) == (3)
        eager = result['cpu', 'torch_eager']
        assert ((eager['fuga_p50_us'], eager['torch_p50_us'])) == ((20, 4))
        assert ((eager['fuga_over_torch'], eager['difference_pct'])) == ((5, 400))
        assert ((eager['fuga_rounds'], eager['torch_rounds'])) == ((3, 3))
        compiled = result['cpu', 'torch_compile']
        assert ((compiled['fuga_over_torch'], compiled['difference_pct'])) == ((2, 100))
        gpu = result['cuda', 'torch_eager']
        assert ((gpu['fuga_over_torch'], gpu['difference_pct'])) == ((0.5, -50))

    def test_tf32_tolerance_applies_only_to_f32_matmul(self):
        case = Case('matmul', 'n3', 'fp32', 'row_major', 3, 3, matmul_data='general')
        shape, reference = expected(case)
        values = [reference[i:i + 3] for i in range(0, 9, 3)]
        values[0][0] += 1e-4
        output = RecordedTensor(shape, 'fp32', values)
        assert check_output(output, case, 'fp32', 'tf32') > 0
        with pytest.raises(AssertionError, match='numerical mismatch'):
            check_output(output, case, 'fp32', 'ieee')
        values[0][0] += 1
        with pytest.raises(AssertionError, match='numerical mismatch'):
            check_output(output, case, 'fp32', 'tf32')
        other = Case('elementwise', 'small', 'fp32', 'row_major', 1, 1)
        shape, reference = expected(other)
        with pytest.raises(AssertionError, match='numerical mismatch'):
            check_output(RecordedTensor(shape, 'fp32', [[reference[0] + 1e-4]]), other, 'fp32', 'tf32')

    def test_precision_and_input_distribution_are_comparison_dimensions(self):
        case = Case('matmul', 'n16', 'fp32', 'row_major', 16, 16)
        identity = dict(device='cuda', round=0, status='ok', complete_p50_us=1, **asdict(case))
        rows = [identity | {'backend': 'fuga', 'matmul_precision': 'tf32'},
                identity | {'backend': 'torch_eager', 'matmul_precision': 'ieee'},
                identity | {'backend': 'torch_eager', 'matmul_precision': 'tf32', 'matmul_data': 'general'}]
        assert comparisons(rows) == []
        rows.append(identity | {'backend': 'torch_eager', 'matmul_precision': 'tf32'})
        assert len(comparisons(rows)) == 1

    def test_percentile_preserves_units_and_interpolates_small_samples(self):
        assert (percentile([9, 1, 8, 2], 0.5)) == (5)
        assert (percentile([9, 1, 8, 2], 0.9)) == pytest.approx(8.7, rel=0, abs=0.5 * 10 ** -(7))
        assert (percentile([7], 0.9)) == (7)


class TestBenchmarkWarmup:
    @pytest.mark.parametrize('value', ('0', '0.25', '1e-3'))
    def test_duration_accepts_finite_nonnegative_values(self, value):
        from tools.benchmark_torch import nonnegative_seconds
        assert nonnegative_seconds(value) == float(value)

    @pytest.mark.parametrize('value', ('-1', 'nan', 'inf', '-inf', '1e309'))
    def test_duration_rejects_negative_or_nonfinite_values(self, value):
        import argparse
        from tools.benchmark_torch import nonnegative_seconds
        with pytest.raises(argparse.ArgumentTypeError, match='finite and nonnegative'):
            nonnegative_seconds(value)

    @pytest.mark.parametrize('seconds', (None, 0.0, 0.25), ids=('default', 'zero', 'duration'))
    def test_cli_passes_duration_to_worker_and_records_actual_iterations(self, monkeypatch, tmp_path, seconds):
        import csv
        import json
        from types import SimpleNamespace
        from tools import benchmark_torch as benchmark

        output = tmp_path / 'recording'
        args = ['benchmark_torch.py', '--devices', 'cpu', '--backends', 'fuga',
                '--workloads', 'elementwise', '--sizes', 'small', '--rounds', '1',
                '--samples', '3', '--warmup', '7', '--output', str(output)]
        if seconds is not None:
            args.extend(('--warmup-seconds', str(seconds)))
        requests = []

        def worker(command, **kwargs):
            requests.append(json.loads(kwargs['input']))
            response = {'result': {'complete_p50_us': 1, 'warmup_iterations': 11, 'verified': True},
                        'samples': [], 'environment': {}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr='')

        monkeypatch.setattr(benchmark.sys, 'argv', args)
        monkeypatch.setattr(benchmark, 'subprocess',
                            SimpleNamespace(run=worker, TimeoutExpired=benchmark.subprocess.TimeoutExpired))
        assert benchmark.main() == 0
        assert len(requests) == 1
        request = requests[0]
        assert request['warmup_seconds'] == (1.0 if seconds is None else seconds)
        assert request['warmup'] == 7 and request['samples'] == 3
        metadata = json.loads((output / 'metadata.json').read_text())
        assert metadata['settings']['warmup_seconds'] == request['warmup_seconds']
        with (output / 'results.csv').open() as stream:
            assert next(csv.DictReader(stream))['warmup_iterations'] == '11'

    @pytest.mark.parametrize('minimum,seconds', tuple(product((1, 5), (0.0, 0.5, 1.25))))
    def test_measure_satisfies_count_and_duration_and_counts_real_executions(self, monkeypatch, tmp_path,
                                                                          minimum, seconds):
        import json
        from types import SimpleNamespace
        from tools import benchmark_torch as benchmark

        class Clock:
            elapsed = 0.0

            def now(self):
                return self.elapsed

            def now_ns(self):
                return int(self.elapsed * 1e9)

        clock = Clock()
        artifact = tmp_path / 'artifact'
        artifact.mkdir()
        (artifact / 'abi.json').write_text(json.dumps(
            {'functions': [{'lowering': {'temporary_workspace_bytes': 0}}]}))
        state = SimpleNamespace(executions=0, synchronizations=0, closed=False)

        class Engine:
            compiler = SimpleNamespace(capabilities=lambda: {})
            last_artifact_dir = artifact

            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                state.closed = True

            def tensor(self, data, **kwargs):
                return data

            def synchronize(self):
                state.synchronizations += 1
                clock.elapsed += 0.125

            @property
            def stats(self):
                return {'executions': state.executions, 'compilations': 1}

        def evaluate(output):
            state.executions += 1
            clock.elapsed += 0.125

        case = Case('elementwise', 'small', 'fp32', 'row_major', 1, 1)
        shape, values = expected(case)
        module = SimpleNamespace(Engine=Engine, evaluate=evaluate, fp32='fp32', __file__=__file__)
        monkeypatch.setitem(benchmark.sys.modules, 'fugatensor', module)
        monkeypatch.setattr(benchmark, 'time', SimpleNamespace(perf_counter=clock.now, perf_counter_ns=clock.now_ns))
        monkeypatch.setattr(benchmark.os, 'environ', dict(benchmark.os.environ))
        monkeypatch.setattr(benchmark, 'package_version', lambda name: 'fixture')
        monkeypatch.setattr(benchmark, 'make_function',
                            lambda *args: lambda *inputs: RecordedTensor(shape, 'fp32', [values]))
        request = dict(case=asdict(case), device='cpu', backend='fuga', matmul_precision='ieee',
                       cache_parent=str(tmp_path), cpu_core=None, threads=1, samples=3,
                       warmup=minimum, warmup_seconds=seconds)
        response = benchmark.measure(request)
        # Each completed call advances the independent clock by one quarter second.
        iterations = max(minimum, math.ceil(seconds / 0.25))
        result = response['result']
        assert result['warmup_iterations'] == iterations
        assert result['executions'] == state.executions == 1 + iterations + request['samples']
        assert state.synchronizations == state.executions + 1
        assert state.closed and result['verified'] and result['compilations'] == 1
        assert [(sample['phase'], sample['iteration']) for sample in response['samples']] == [
            ('first', 0), *(('hot', i) for i in range(request['samples']))]
        assert all(sample['call_us'] == 125000 and sample['complete_us'] == 250000
                   for sample in response['samples'])
