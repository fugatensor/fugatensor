#!/usr/bin/env python3
"""Record synchronized Fuga/PyTorch latency with isolated process and disk caches."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_workloads import Case, DTYPES, LAYOUTS, SIZES, WORKLOADS, cases, expected, input_data, make_function


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def check_output(output, case, dtype, matmul_precision='ieee'):
    shape, reference = expected(case)
    if tuple(output.shape) != tuple(shape) or output.dtype != dtype:
        raise AssertionError(f"output contract: {output.shape}, {output.dtype}")
    values = output.tolist()
    def flatten(value):
        if isinstance(value, list):
            return [item for child in value for item in flatten(child)]
        return [value]
    actual = flatten(values)
    if len(actual) != len(reference):
        raise AssertionError("output element count differs")
    tolerance = 3e-3 if case.dtype == 'fp16' else 2e-5
    if case.workload == 'matmul' and case.dtype == 'fp32' and matmul_precision == 'tf32':
        tolerance = 2e-3
    for index, (value, target) in enumerate(zip(actual, reference)):
        if not math.isfinite(value) or not math.isclose(value, target, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f"numerical mismatch at {index}: {value} != {target}")
    return max((abs(a - b) for a, b in zip(actual, reference)), default=0)


def warm_up(invoke, synchronize, iterations, seconds):
    until = time.perf_counter() + seconds
    completed = 0
    while completed < iterations or time.perf_counter() < until:
        output = invoke()
        synchronize()
        del output
        completed += 1
    return completed


def measure(request):
    case = Case(**request['case'])
    device, backend = request['device'], request['backend']
    gpu = device == 'cuda'
    precision = request['matmul_precision'] if gpu else 'ieee'
    os.environ.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', None)
    if precision == 'ieee':
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    else:
        os.environ.pop('NVIDIA_TF32_OVERRIDE', None)
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix='worker-', dir=request['cache_parent']) as cache:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(cache) / 'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(Path(cache) / 'triton')
        if request['cpu_core'] is not None:
            os.sched_setaffinity(0, {request['cpu_core']})
        if backend == 'fuga':
            import fugatensor as module
            dtype = getattr(module, case.dtype)
            engine = module.Engine(device=device, cache_dir=Path(cache) / 'fuga', compilers={})
            host_matmul = engine.compiler.capabilities().get('host_matmul', {})
            scope = engine
            inputs = [engine.tensor(data, dtype=dtype) for data in input_data(case)]
            function = make_function(case, 'fuga', module)
            def invoke():
                output = function(*inputs)
                module.evaluate(output)
                return output
            synchronize = engine.synchronize
            version = package_version('fugatensor')
            device_name = engine.session._native.device_name if gpu else platform.processor()
        else:
            import torch as module
            module.set_num_threads(request['threads'])
            module.set_num_interop_threads(1)
            module.backends.fp32_precision = 'ieee'
            module.backends.cuda.matmul.fp32_precision = precision
            dtype = {'fp32': module.float32, 'fp16': module.float16}[case.dtype]
            scope = module.inference_mode()
            inputs = [module.tensor(data, dtype=dtype, device=device) for data in input_data(case)]
            function = make_function(case, 'torch', module)
            if backend == 'torch_compile':
                function = module.compile(function, fullgraph=True, dynamic=False,
                                          mode='reduce-overhead' if gpu else 'default')
            if backend == 'torch_compile' and gpu:
                def invoke():
                    module.compiler.cudagraph_mark_step_begin()
                    return function(*inputs)
            else:
                def invoke():
                    return function(*inputs)
            synchronize = module.cuda.synchronize if gpu else lambda: None
            version = module.__version__
            device_name = module.cuda.get_device_name() if gpu else platform.processor()
        synchronize()
        setup_us = (time.perf_counter_ns() - started) / 1000
        samples = []
        def timed(phase, iteration):
            before = time.perf_counter_ns()
            output = invoke()
            returned = time.perf_counter_ns()
            synchronize()
            after = time.perf_counter_ns()
            samples.append(dict(phase=phase, iteration=iteration,
                                call_us=(returned - before) / 1000,
                                complete_us=(after - before) / 1000))
            return output
        with scope:
            output = timed('first', 0)
            error = check_output(output, case, dtype, precision)
            del output
            warmup_iterations = warm_up(invoke, synchronize, request['warmup'], request['warmup_seconds'])
            for iteration in range(request['samples']):
                output = timed('hot', iteration)
                if iteration == request['samples'] - 1:
                    error = max(error, check_output(output, case, dtype, precision))
                del output
            memory = dict(device_reserved_bytes='', device_peak_reserved_bytes='',
                          device_allocated_bytes='', device_peak_allocated_bytes='',
                          host_workspace_bytes='')
            executions = compilations = ''
            if backend == 'fuga':
                stats = engine.stats
                executions, compilations = stats['executions'], stats['compilations']
                if executions != 1 + warmup_iterations + request['samples'] or compilations != 1:
                    raise AssertionError(f"fresh graph execution/cache contract failed: {stats}")
                if gpu:
                    memory.update(device_reserved_bytes=stats['runtime']['resident_device_bytes'],
                                  device_peak_reserved_bytes=stats['runtime']['peak_resident_device_bytes'])
                else:
                    abi = json.loads((engine.last_artifact_dir / 'abi.json').read_text())
                    memory['host_workspace_bytes'] = sum(
                        f['lowering']['temporary_workspace_bytes'] for f in abi['functions'])
            elif gpu:
                memory.update(device_reserved_bytes=module.cuda.memory_reserved(),
                              device_peak_reserved_bytes=module.cuda.max_memory_reserved(),
                              device_allocated_bytes=module.cuda.memory_allocated(),
                              device_peak_allocated_bytes=module.cuda.max_memory_allocated())
            hot = [sample for sample in samples if sample['phase'] == 'hot']
            row = dict(setup_us=setup_us, first_complete_us=samples[0]['complete_us'],
                       call_p50_us=statistics.median(s['call_us'] for s in hot),
                       complete_p50_us=statistics.median(s['complete_us'] for s in hot),
                       complete_p90_us=percentile([s['complete_us'] for s in hot], .9),
                       host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       executions=executions, compilations=compilations, warmup_iterations=warmup_iterations,
                       max_abs_error=error,
                       verified=True, **memory)
        environment = dict(python=sys.version, executable=sys.executable,
                           framework_version=version, framework_path=module.__file__,
                           device_name=device_name, threads=1 if backend == 'fuga' else module.get_num_threads(),
                           cpu_affinity=sorted(os.sched_getaffinity(0)),
                           compile_mode=('reduce-overhead' if gpu else 'default')
                           if backend == 'torch_compile' else '',
                           cuda_version=module.version.cuda if backend != 'fuga' else '')
        environment['matmul_precision'] = precision
        if backend == 'fuga':
            environment['host_matmul'] = host_matmul
            if gpu:
                environment['runtime'] = stats['runtime']
        else:
            environment['blas_library'] = (str(module.backends.cuda.preferred_blas_library()) if gpu
                                           else 'MKL' if module.backends.mkl.is_available() else 'build-config')
            environment['build_config'] = module.__config__.show()
        return dict(result=row, samples=samples, environment=environment)


def comparisons(rows):
    groups = {}
    dimensions = ('device', 'workload', 'size', 'dtype', 'layout', 'rows', 'columns', 'depth', 'branches',
                  'matmul_data', 'matmul_precision')
    for row in rows:
        if row['status'] == 'ok':
            key = tuple(row[name] for name in dimensions)
            groups.setdefault(key, {}).setdefault(row['backend'], {})[row['round']] = row['complete_p50_us']
    result = []
    for key, backends in groups.items():
        if 'fuga' not in backends:
            continue
        for backend in ('torch_eager', 'torch_compile'):
            if backend in backends:
                rounds = backends['fuga'].keys() & backends[backend].keys()
                if not rounds:
                    continue
                fuga = statistics.median(backends['fuga'][r] for r in rounds)
                torch = statistics.median(backends[backend][r] for r in rounds)
                result.append(dict(zip(dimensions, key), torch_mode=backend,
                    fuga_p50_us=fuga, torch_p50_us=torch, fuga_over_torch=fuga / torch,
                    difference_pct=(fuga / torch - 1) * 100,
                    fuga_rounds=len(rounds), torch_rounds=len(rounds)))
    return result


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def nonnegative_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu', 'cuda'])
    parser.add_argument('--backends', nargs='+', choices=('fuga', 'torch_eager', 'torch_compile'),
                        default=['fuga', 'torch_eager', 'torch_compile'])
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument('--sizes', nargs='+', choices=SIZES, default=list(SIZES))
    parser.add_argument('--dtypes', nargs='+', choices=DTYPES, default=['fp32'])
    parser.add_argument('--layouts', nargs='+', choices=LAYOUTS, default=['row_major'])
    parser.add_argument('--depths', nargs='+', type=int, choices=(3, 8), default=[8])
    parser.add_argument('--branches', nargs='+', type=int, choices=(2, 4), default=[4])
    parser.add_argument('--matmul-sizes', nargs='+', type=positive,
                        help='override square matmul sizes; other workloads retain --sizes')
    parser.add_argument('--matmul-data', choices=('dyadic', 'general'), default='dyadic')
    parser.add_argument('--matmul-precision', choices=('tf32', 'ieee'), default='tf32',
                        help='CUDA f32 matmul policy for both frameworks')
    parser.add_argument('--rounds', type=positive, default=3)
    parser.add_argument('--samples', type=positive, default=31)
    parser.add_argument('--warmup', type=positive, default=5)
    parser.add_argument('--warmup-seconds', type=nonnegative_seconds, default=1.0,
                        help='minimum warmup duration in addition to --warmup iterations')
    parser.add_argument('--threads', type=positive, default=1)
    parser.add_argument('--cpu-core', type=int)
    parser.add_argument('--fuga-python', default=sys.executable)
    parser.add_argument('--torch-python', default=sys.executable)
    parser.add_argument('--timeout', type=positive, default=600)
    parser.add_argument('--output', type=Path, default=Path('build/benchmarks') /
                        datetime.now().strftime('torch-%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    if 'fuga' in args.backends and 'cpu' in args.devices and args.threads != 1:
        parser.error('matched Fuga CPU comparisons require --threads 1')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'output directory is not empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'logs').mkdir()
    matrix = list(cases(args.workloads, args.sizes, args.dtypes, args.layouts, args.depths, args.branches,
                        matmul_sizes=args.matmul_sizes, matmul_data=args.matmul_data))
    rows, samples, environments = [], [], []
    settings = vars(args) | {'output': str(args.output.resolve())}
    total = len(matrix) * len(args.devices) * len(args.backends) * args.rounds
    for round_ in range(args.rounds):
        offset = round_ % len(args.backends)
        order = args.backends[offset:] + args.backends[:offset]
        for device in args.devices:
            for case in matrix:
                for backend in order:
                    precision = args.matmul_precision if device == 'cuda' else 'ieee'
                    identity = dict(round=round_, device=device, backend=backend,
                                    matmul_precision=precision, **asdict(case))
                    request = dict(case=asdict(case), device=device, backend=backend,
                                   samples=args.samples, warmup=args.warmup, warmup_seconds=args.warmup_seconds,
                                   threads=args.threads,
                                   cpu_core=args.cpu_core, matmul_precision=precision)
                    python = args.fuga_python if backend == 'fuga' else args.torch_python
                    env = os.environ | {'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
                                        'OPENBLAS_NUM_THREADS': str(args.threads), 'GOTO_NUM_THREADS': str(args.threads),
                                        'TORCHINDUCTOR_COMPILE_THREADS': '1'}
                    env.pop('PYTHONPATH', None)
                    number = len(rows) + 1
                    print(f'[{number}/{total}] {device} {case.workload} {case.size} {case.dtype} {case.layout} {backend}', flush=True)
                    try:
                        with tempfile.TemporaryDirectory(prefix='fuga-torch-benchmark-') as cache_parent:
                            request['cache_parent'] = cache_parent
                            process = subprocess.run([python, '-I', str(Path(__file__).resolve()), '--worker'],
                                input=json.dumps(request), text=True, capture_output=True, env=env, timeout=args.timeout)
                        (args.output / 'logs' / f'{number:04d}.stderr').write_text(process.stderr)
                        if process.returncode:
                            raise RuntimeError(process.stderr.strip() or f'worker exited {process.returncode}')
                        response = json.loads(process.stdout)
                        rows.append(identity | {'status': 'ok'} | response['result'])
                        samples.extend(identity | sample for sample in response['samples'])
                        environments.append(identity | response['environment'])
                    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
                        rows.append(identity | {'status': 'error', 'error': str(error)})
                        print(f'  ERROR: {error}', file=sys.stderr)
                    write_csv(args.output / 'results.csv', rows)
                    write_csv(args.output / 'samples.csv', samples)
                    write_csv(args.output / 'comparison.csv', comparisons(rows))
                    (args.output / 'metadata.json').write_text(json.dumps(dict(
                        recorded_at=datetime.now(timezone.utc).isoformat(), settings=settings,
                        platform=platform.platform(), environments=environments), indent=2) + '\n')
    print(f'Recorded: {args.output.resolve()}')
    return int(any(row['status'] != 'ok' for row in rows))


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        print(json.dumps(measure(json.load(sys.stdin))))
    else:
        raise SystemExit(main())
