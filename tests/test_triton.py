"""User Triton source snapshots, standalone cubin ABI, and ASK GPU scheduling."""
from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import os
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from fugatensor._ask.cuda import CUDAError
import fugatensor as ft
from fugatensor.lowering import lower_to_ask
from fugatensor.tracing import TraceContext
from tests._helpers import EXAMPLES_DIR, EngineTestCase, make_ask_compiler, requires_triton

AFFINE_SRC = """import triton
import triton.language as tl

@triton.jit
def affine(in_data, out_data, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(in_data + offsets, mask=offsets < N, other=0)
    tl.store(out_data + offsets, value * 2 + 1, mask=offsets < N)
"""

IDENTITY_SRC = """import triton
import triton.language as tl

@triton.jit
def identity_kernel(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(inp + idx, mask=idx < N, other=0)
    tl.store(out + idx, val, mask=idx < N)
"""

POLYNOMIAL_SRC = """import triton
import triton.language as tl

@triton.jit
def polynomial(data, result, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(data + idx, mask=idx < N, other=0)
    tl.store(result + idx, val * val + 1, mask=idx < N)
"""


EXPANDED_SRC = """import triton
import triton.language as tl
BIAS = tl.constexpr(1.5)
CONFIGS = [triton.Config({'BLOCK': 64}), triton.Config({'BLOCK': 128})]
@triton.jit
def helper(x):
    return x + BIAS
@triton.autotune(configs=CONFIGS, key=['N'])
@triton.heuristics({'EVEN': lambda args: args['X'].numel() % args['BLOCK'] == 0 and args['X'].stride(0) == 1})
@triton.jit
def expanded(X, Y, Z, N, scale, BLOCK:tl.constexpr, EVEN:tl.constexpr,
             MODE:tl.constexpr='add', PAIR:tl.constexpr=(2,3), TYPE:tl.constexpr=tl.float32,
             ENABLED:tl.constexpr=True, OPTIONAL:tl.constexpr=None):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(TYPE)
    if MODE == 'add' and ENABLED and OPTIONAL is None:
        x = helper(x) + PAIR[0]
    tl.store(Y + i, x * scale, i < N)
    tl.store(Z + i, x + PAIR[1], i < N)
"""


class TritonRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = make_ask_compiler()

    def context(self, shape=(2, 5), dtype="f32", device="cuda"):
        return TraceContext([(shape, dtype)], engine=SimpleNamespace(device=device, capability=(8, 6)))

    def registration(self, **changes):
        opts = dict(
            entry="affine", input="in_data", output="out_data",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,),
        )
        opts.update(changes)
        return ft.triton.from_source(AFFINE_SRC, **opts)

    def prepare_request(self, op, ctx=None):
        ctx = self.context() if ctx is None else ctx
        mod = ctx.finish(op(ctx.inputs[0]))
        target = {"backend": "cuda", "arch": 86} if ctx.engine.device != "cpu" else None
        return self.compiler.prepare(lower_to_ask(mod, target=target)).report["external_requests"][0]

    def test_verbatim_snapshot_without_importing_triton_and_frozen_metadata(self):
        crlf_src = AFFINE_SRC.replace("\n", "\r\n")
        orig_import = builtins.__import__

        def block_triton(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("registration must not import triton")
            return orig_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_kernel.py"
            path.write_bytes(crlf_src.encode("utf-8"))
            with mock.patch("builtins.__import__", side_effect=block_triton):
                op = ft.triton.kernel(path, entry="affine", input="in_data", output="out_data",
                                      constants={"N": 10, "BLOCK": 128}, grid=(1,))
            path.write_text(IDENTITY_SRC, encoding="utf-8")
            self.assertEqual(op.source, crlf_src)
            self.assertEqual(self.prepare_request(op)["source"], crlf_src)

        # Frozen TensorSpec + copied dicts + all 6 storage pointer signatures.
        def check_spec(spec):
            with self.assertRaises(FrozenInstanceError):
                spec.dtype = "i32"
            return {"N": spec.numel, "BLOCK": 128}

        consts = {"N": 10, "BLOCK": 128}
        options = {"num_warps": 8, "enable_fp_fusion": False}
        copied_op = self.registration(constants=consts, options=options)
        consts["N"] = 999
        options["num_warps"] = 4
        self.assertEqual(self.prepare_request(copied_op)["constants"]["N"], 10)
        self.assertEqual(self.prepare_request(copied_op)["fixed_options"]["num_warps"], 8)
        self.assertEqual(self.prepare_request(self.registration(constants=check_spec), self.context((3, 129), "f16"))["grid"], [4, 1, 1])

        for dtype, sig in (("bool", "*i1"), ("i32", "*i32"), ("i64", "*i64"), ("f32", "*fp32"), ("f16", "*fp16"), ("bf16", "*bf16")):
            req = self.prepare_request(self.registration(), self.context(dtype=dtype))
            self.assertEqual(req["signature"], {"in_data": sig, "out_data": sig})

    def test_pending_kernel_calls_snapshot_changed_callback_configuration(self):
        class CapturedGraph(RuntimeError):
            pass

        for wrapped in (False, True):
            with self.subTest(custom_fun=wrapped), tempfile.TemporaryDirectory() as directory:
                configuration = {"block": 128}
                op = self.registration(
                    constants=lambda spec: {"N": spec.numel, "BLOCK": configuration["block"]},
                    grid=lambda spec: ((spec.numel + configuration["block"] - 1) // configuration["block"],),
                )

                class Forward(ft.fun):
                    def forward(self, ctx, value):
                        return op(value)

                call = Forward() if wrapped else op
                with ft.Engine(cache_dir=directory) as engine:
                    x = ft.tensor(list(range(257)), dtype="i32")
                    first = call(x)
                    configuration["block"] = 256
                    second = call(x)
                    requests = []

                    def capture(context):
                        module = ft.ir.specialize(context.module)
                        request = self.compiler.prepare(lower_to_ask(
                            module, target={"backend": "cuda", "arch": 86}
                        )).report["external_requests"][0]
                        requests.append(request)
                        # Inspect the queued compiler contract without requiring a GPU.
                        raise CapturedGraph("captured pending graph")

                    with mock.patch.object(engine, "_compile", side_effect=capture):
                        for output in (first, second):
                            with self.assertRaises(CapturedGraph):
                                output.tolist()
                    self.assertEqual([request["constants"]["BLOCK"] for request in requests], [128, 256])
                    self.assertEqual([request["grid"] for request in requests], [[3, 1, 1], [2, 1, 1]])
                    self.assertEqual(engine.stats["executions"], 0)

    def test_environment_selects_bundled_worker_without_importing_triton(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("environment setup must not import the compiler")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            environment = ft.triton.environment(python=sys.executable)
            self.assertEqual(Path(environment.worker).name, "_triton_worker.py")
            self.assertTrue(Path(environment.worker).is_file())
            with mock.patch.dict(os.environ, {"FUGATENSOR_TRITON_PYTHON": sys.executable}, clear=True):
                self.assertEqual(ft.triton.environment_from_env(), environment)
            with mock.patch.dict(os.environ, {"FUGATENSOR_TRITON_WORKER": "/worker.py"}, clear=True):
                with self.assertRaisesRegex(ValueError, "FUGATENSOR_TRITON_PYTHON"):
                    ft.triton.environment_from_env()

    def test_multiple_inputs_and_explicit_output_metadata(self):
        source = ('import triton\nimport triton.language as tl\n'
                  '@triton.jit\ndef product(Q, K, O, M: tl.constexpr):\n pass\n')
        observed = []

        def constants(specs):
            observed.append(tuple(spec.shape for spec in specs))
            return {'M': specs[0].shape[0]}

        operation = ft.triton.from_source(source, entry='product', inputs=('Q', 'K'), output='O',
            constants=constants, grid=(1,), output_shape=lambda s: (s[0].shape[0], s[1].shape[1]),
            output_dtype='f32')
        context = TraceContext([((2, 3), 'f16'), ((3, 4), 'bf16')],
                               engine=SimpleNamespace(device='cuda'))
        result = operation(*context.inputs)
        self.assertEqual((result.shape, result.dtype), ((2, 4), 'f32'))
        prepared = self.compiler.prepare(lower_to_ask(context.finish(result),
                                                      target={'backend': 'cuda', 'arch': 86}))
        request = prepared.report['external_requests'][0]
        self.assertEqual(request['source_bindings'], {'Q': 'input:0', 'K': 'input:1', 'O': 'output:0'})
        self.assertEqual(request['signature'], {'Q': '*fp16', 'K': '*bf16', 'O': '*fp32'})
        self.assertEqual(observed, [((2, 3), (3, 4))])
        with self.assertRaisesRegex(TypeError, 'expects 2'):
            operation(context.inputs[0])
        for changes in ({'inputs': ('Q', 'Q')}, {'input': 'Q'}, {'inputs': 'Q'}):
            parameters = dict(inputs=('Q', 'K'), output='O', grid=(1,), constants={'M': 2})
            parameters.update(changes)
            with self.assertRaises((ValueError, TypeError)):
                ft.triton.from_source(source, entry='product', **parameters)

    def test_deferred_arch_binding_and_invalid_registration_guards(self):
        class NoProbeEngine:
            device = "cuda"
            @property
            def capability(self):
                raise AssertionError("tracing must not probe physical GPU")

        ctx = TraceContext([((2, 5), "f32")], engine=NoProbeEngine())
        mod = ctx.finish(self.registration()(ctx.inputs[0]))
        before = mod.to_text()
        for arch in (80, 86):
            req = self.compiler.prepare(lower_to_ask(mod, target={"backend": "cuda", "arch": arch})).report["external_requests"][0]
            self.assertEqual((req["target"]["arch"], mod.to_text()), (arch, before))

        with self.assertRaisesRegex(ValueError, "no implementation"):
            lower_to_ask(mod, target={"backend": "host"})

        for bad in (
            {"entry": "missing"},
            {"input": "missing"},
            {"output": "in_data"},
            {"constants": {"BLOCK": 128}},
            {"grid": ()},
            {"grid": (0,)},
            {"grid": (2**31,)},
        ):
            with self.assertRaises(ValueError):
                self.registration(**bad)
        with self.assertRaisesRegex(TypeError, "finite JSON"):
            self.registration(constants={"N": float("nan"), "BLOCK": 128})


    def test_scalar_multioutput_and_recursive_constants_survive_lowering(self):
        op = ft.triton.from_source(EXPANDED_SRC, entry='expanded', input='X',
            outputs={'Y': {}, 'Z': {'dtype': 'i32'}}, grid=(1,), config=1,
            scalars=lambda spec: {'N': ('i32', spec.numel), 'scale': ('fp32', 2.5)},
            constants={'MODE': 'add', 'PAIR': [2, 3], 'TYPE': {'__triton_dtype__': 'float32'},
                       'ENABLED': True, 'OPTIONAL': None})
        ctx = self.context()
        results = op(ctx.inputs[0])
        self.assertEqual(tuple(result.dtype for result in results), ('f32', 'i32'))
        module = ctx.finish(results)
        prepared = self.compiler.prepare(lower_to_ask(module, target={'backend': 'cuda', 'arch': 86}))
        req = prepared.report['external_requests'][0]
        self.assertEqual(req['source_bindings'], {'X': 'input:0', 'Y': 'output:0', 'Z': 'output:1', 'N': 10, 'scale': 2.5})
        self.assertEqual(req['signature']['scale'], 'fp32')
        self.assertEqual(req['constants']['TYPE'], {'__triton_dtype__': 'float32'})
        self.assertIs(req['constants']['ENABLED'], True)
        self.assertIsNone(req['constants']['OPTIONAL'])
        self.assertEqual(req['port_shapes']['output:1'], [2, 5])

        texts = []
        for value in ({'a': 1, 'b': 2}, {'a=1, b': 2}, True, 1, 1.0):
            op = ft.triton.from_source(EXPANDED_SRC, entry='expanded', input='X', outputs={'Y': {}, 'Z': {}},
                scalars={'N': ('i32', 10), 'scale': ('fp32', 2.5)}, constants={'PAIR': value}, grid=(1,), config=1)
            context = self.context()
            texts.append(context.finish(op(context.inputs[0])).to_text())
        self.assertEqual(len(set(texts)), len(texts))

        source = """import triton
import triton.language as tl
from triton.language import constexpr as C
@triton.jit
def defaults(X, Y, N, scale=2.5, BLOCK:C=128):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i, tl.load(X+i, i<N, other=0)*scale, i<N)
"""
        op = ft.triton.from_source(source, entry='defaults', input='X', output='Y', grid=(1,),
                                  scalars={'N': ('i32', 10)})
        request = self.prepare_request(op)
        self.assertNotIn('BLOCK', request['signature'])
        self.assertEqual((request['signature']['scale'], request['source_bindings']['scale']), ('fp32', 2.5))

    def test_public_dtypes_bind_metadata_scalars_and_constexpr(self):
        source = '''import triton
import triton.language as tl
@triton.jit
def types(X, Y, enabled, count, wide, half, bfloat, scale,
          TYPE: tl.constexpr, PACK: tl.constexpr):
    pass
'''
        def output_dtype(spec):
            self.assertIs(spec.dtype, ft.fp32)
            return ft.fp16

        operation = ft.triton.from_source(
            source, entry="types", input="X", output="Y", output_dtype=output_dtype,
            grid=(1,), scalars={"enabled": (ft.bool, True), "count": (ft.int32, 2),
                "wide": (ft.int64, 2**40), "half": (ft.fp16, 1.5),
                "bfloat": (ft.bf16, 2.0), "scale": (ft.fp32, 3.0)},
            constants={"TYPE": ft.fp16, "PACK": (ft.int32, {"value": ft.bf16})})
        context = self.context(dtype=ft.fp32)
        self.assertIs(context.inputs[0].dtype, ft.fp32)
        output = operation(context.inputs[0])
        self.assertIs(output.dtype, ft.fp16)
        module = context.finish(output)
        request = self.compiler.prepare(lower_to_ask(
            module, target={"backend": "cuda", "arch": 86})).report["external_requests"][0]
        self.assertEqual(request["signature"], {"X": "*fp32", "Y": "*fp16", "enabled": "i1",
            "count": "i32", "wide": "i64", "half": "fp16", "bfloat": "bf16", "scale": "fp32"})
        self.assertEqual(request["constants"], {"TYPE": {"__triton_dtype__": "float16"},
            "PACK": [{"__triton_dtype__": "int32"}, {"value": {"__triton_dtype__": "bfloat16"}}]})



class TritonWorkerContractTests(unittest.TestCase):
    def test_binary_export_uses_source_abi_without_reading_compiler_ir(self):
        from fugatensor import _triton_worker as worker

        @dataclass
        class Target:
            backend: str = "cuda"
            arch: int = 86
            warp_size: int = 32

        request = {"signature": {"lhs": "*fp32", "out": "*fp32", "rhs": "*fp32"},
                   "source_bindings": {"lhs": "input:0", "out": "output:0", "rhs": "input:1"},
                   "target": {"backend": "cuda", "arch": 86, "warp_size": 32}, "grid": [2, 1, 1]}
        metadata = SimpleNamespace(name="subtract", triton_version="3.7.1", target=Target(),
            global_scratch_size=0, profile_scratch_size=0, tmem_size=0, num_ctas=1,
            tensordesc_meta=[], launch_cooperative_grid=False, launch_pdl=False,
            instrumentation_mode="", shared=1024, warp_size=32, num_warps=4)
        compiled = SimpleNamespace(name="subtract", metadata=metadata, asm={"cubin": b"binary fixture"})
        with mock.patch.object(worker, "probe", return_value={"producer": {"name": "triton", "version": "3.7.1"}}):
            result = worker.artifact(request, compiled, ["out", "rhs", "lhs"])
            self.assertEqual(result["arguments"], ["output:0", "input:1", "input:0", "null", "null"])
            self.assertEqual(result["launch"], {"grid": [2, 1, 1], "block": [128, 1, 1], "shared_bytes": 1024})
            for name, value in (("global_scratch_size", 8), ("profile_scratch_size", 8),
                                ("tmem_size", 32), ("num_ctas", 2), ("launch_cooperative_grid", True),
                                ("launch_pdl", True), ("tensordesc_meta", [1]), ("instrumentation_mode", "test")):
                with self.subTest(resource=name), mock.patch.object(metadata, name, value):
                    with self.assertRaisesRegex(ValueError, "unsupported"):
                        worker.artifact(request, compiled, ["out", "rhs", "lhs"])


@requires_triton
class TritonGPUExecutionTests(EngineTestCase):
    device = "cuda"
    enable_triton = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.scale_path = EXAMPLES_DIR / "kernels.py"
        cls.scale = ft.triton.kernel(
            cls.scale_path, entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )
        cls.identity = ft.triton.from_source(
            IDENTITY_SRC, entry="identity_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )
        cls.polynomial = ft.triton.from_source(
            POLYNOMIAL_SRC, entry="polynomial", input="data", output="result",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )

    def test_worker_tracks_transitive_imports_without_stale_bytecode(self):
        from fugatensor import _triton_worker as worker
        import py_compile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / 'fuga_import_leaf.py'
            helper.write_text('VALUE = 2\n', encoding='utf-8')
            py_compile.compile(str(helper), doraise=True)
            original = helper.stat()
            (root / 'fuga_import_parent.py').write_text('from fuga_import_leaf import VALUE\n', encoding='utf-8')
            source = ('from fuga_import_parent import VALUE\n' + AFFINE_SRC).replace(
                'N: tl.constexpr, BLOCK: tl.constexpr', 'N: tl.constexpr, BLOCK: tl.constexpr, GAIN: tl.constexpr=VALUE').replace('value * 2 + 1', 'value * GAIN + 1')
            request = {'protocol_version': 2, 'source': source, 'entry': 'affine',
                       'signature': {'in_data': '*fp32', 'out_data': '*fp32'},
                       'source_bindings': {'in_data': 'input:0', 'out_data': 'output:0'},
                       'constants': {'N': 10, 'BLOCK': 128}, 'fixed_options': {}, 'grid': [1, 1, 1],
                       'target': {'backend': 'cuda', 'arch': 86, 'warp_size': 32},
                       'port_shapes': {'input:0': [10], 'output:0': [10]}}
            path, output = root / 'request.json', root / 'artifact.json'
            path.write_text(json.dumps(request), encoding='utf-8')
            environment = {**os.environ, 'PYTHONPATH': directory}
            def run(*args):
                result = subprocess.run([sys.executable, str(Path(worker.__file__).resolve()), *map(str, args)],
                    env=environment, capture_output=True, text=True, check=True)
                return json.loads(result.stdout) if result.stdout else None
            first = run('--probe', '--request', path)
            run('--request', path, '--output', output)
            old = json.loads(output.read_text())
            self.assertEqual(old['producer'], first['producer'])
            helper.write_text('VALUE = 4\n', encoding='utf-8')
            os.utime(helper, ns=(original.st_atime_ns, original.st_mtime_ns))
            self.assertEqual(helper.stat().st_size, original.st_size)
            second = run('--probe', '--request', path)
            run('--request', path, '--output', output)
            new = json.loads(output.read_text())
            self.assertNotEqual(first, second)
            self.assertEqual(new['producer'], second['producer'])
            self.assertIn('import:fuga_import_leaf', second['producer']['dependencies'])
            from fugatensor._ask.adapters import normalize_artifact
            from fugatensor._ask.cuda import install_external
            from fugatensor._ask.gpu import CUDAKernel
            session = self.engine.session
            x, y = session.from_host(list(range(10)), dtype='f32'), session.tensor((10,), 'f32')
            for version, artifact, gain in (('old', old, 2), ('new', new, 4)):
                normalized = normalize_artifact(artifact, request)
                executable = CUDAKernel(normalized, install_external(normalized, root / version), {}, {}).prepare(session)
                executable.submit(x, y).wait()
                self.assertEqual(y.to_host(), [gain * value + 1 for value in range(10)])

    def test_programmatic_jit_keeps_stored_source_and_default_scalar(self):
        import triton
        source = """import triton.language as tl
from triton.language import constexpr as C
def plain(X, Y, N, scale=2.5, BLOCK:C=128):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i, tl.load(X+i, i<N, other=0)*scale, i<N)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plain.py'
            path.write_text(source, encoding='utf-8')
            spec = importlib.util.spec_from_file_location('fuga_plain_jit', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            compiled = triton.jit(module.plain, noinline=True)
            path.unlink()
            operation = ft.triton.from_jit(compiled, input='X', output='Y', grid=(1,),
                                          scalars=lambda spec: {'N': ('i32', spec.numel)})
            self.assertIn('noinline=True', operation.source)
            self.assertEqual(operation(ft.tensor([1., 2., 3.])).tolist(), [2.5, 5., 7.5])

    def test_existing_jit_dependencies_scalars_and_multiple_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'existing.py'
            # A same-named unused global must not be captured as a dependency.
            path.write_text(EXPANDED_SRC + '\nx = object()\n', encoding='utf-8')
            spec = importlib.util.spec_from_file_location('fuga_existing_triton', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            scale = {'value': 2.5}
            operation = ft.triton.from_jit(module.expanded, input='X',
                outputs={'Y': {}, 'Z': {'dtype': ft.int32}}, grid=(1,), config=1,
                scalars=lambda spec: {'N': (ft.int32, spec.numel), 'scale': (ft.fp32, scale['value'])},
                constants={'TYPE': ft.fp32, 'PAIR': (2, 3)})
            self.assertNotIn('object()', operation.source)
            module.BIAS = None  # Registered source no longer reads the original module.
            path.unlink()
            x = ft.tensor([0., 1., 2., 3.])
            y, z = operation(x)
            self.assertIs(y.dtype, ft.fp32)
            self.assertIs(z.dtype, ft.int32)
            scale['value'] = 4.0
            other, _ = operation(x)
            before = self.engine.session.stats['launches']
            result = ft.evaluate(y + 1, z, other)
            self.assertEqual([value.tolist() for value in result],
                             [[9.75, 12.25, 14.75, 17.25], [6, 7, 8, 9], [14., 18., 22., 26.]])
            # Two external nodes + one owned add; unused second output remains
            # allocated for the second kernel's physical ABI.
            self.assertEqual(self.engine.session.stats['launches'] - before, 3)
            self.assertEqual(self.engine.last_report['execution_plan']['steps'][0]['kind'], 'external')

    def test_scheduled_forward_across_precisions_algorithms_and_custom_backward(self):
        scale, identity, polynomial = self.scale, self.identity, self.polynomial

        def mixed(x):
            return scale(x + 1) + 3

        for dtype in ("i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex(CUDAError, "bf16"):
                        mixed(ft.tensor([[1, 2], [3, 4]], dtype=dtype)).tolist()
                    continue
                before = self.engine.stats["executions"]
                launches = self.engine.session.stats["launches"]
                out = mixed(ft.tensor([[1, 2], [3, 4]], dtype=dtype))
                self.assertEqual(self.engine.stats["executions"], before)
                self.assertEqual((out.tolist(), out.dtype), ([[7, 9], [11, 13]], dtype))
                self.assertEqual(self.engine.stats["executions"], before + 1)
                self.assertEqual(self.engine.session.stats["launches"], launches + 3)
                self.assertEqual(mixed(ft.tensor([[5, 6], [7, 8]], dtype=dtype)).tolist(), [[15, 17], [19, 21]])
                self.assertEqual(self.engine.last_report["phase"], "compiled_plan")
                self.assertEqual([step["kind"] for step in self.engine.last_report["execution_plan"]["steps"]],
                                 ["owned", "external", "owned"])
                self.assertEqual(len(list((Path(self.engine.last_artifact_dir) / "package").rglob("kernel.cubin"))), 3)

        # Bool kernel, mixed-precision casts, and custom polynomial kernel.
        def bool_id(x):
            return identity(x)

        self.assertEqual(bool_id(ft.tensor([False, True, False], dtype="bool")).tolist(), [False, True, False])

        def poly_graph(x):
            return polynomial(x + 1) + 3

        self.assertEqual(poly_graph(ft.tensor([1, 2, 3])).tolist(), [8, 13, 20])

        # Raw Triton on derivative paths fails preflight; ft.fun supplies the explicit backward.
        rx = ft.tensor(3.0, requires_grad=True)
        rout = mixed(rx)
        with mock.patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
            with self.assertRaisesRegex(ValueError, "external.*(VJP|reverse|backward)|derivative.*external"):
                rout.backward()
        self.assertIsNone(rx.grad)

        class UserScale(ft.fun):
            def forward(self, ctx, x):
                return scale(x)
            def backward(self, ctx, dy):
                return (scale(dy),)

        op = UserScale()

        def diff_mixed(x):
            return op(x + 1) + 3

        dx = ft.tensor([1.0, 2.0, 3.0], requires_grad=True)
        dout = diff_mixed(dx)
        self.assertEqual(dout.tolist(), [7.0, 9.0, 11.0])
        dout.backward(ft.tensor([1.0, 2.0, -1.0]))
        self.assertEqual(dx.grad.tolist(), [2.0, 4.0, -2.0])

    def test_multiple_external_nodes_and_different_storage_outputs(self):
        x = ft.tensor([1, 2, 3], dtype='f32')
        left = self.scale(x + 1)
        right = self.scale(left + 3)
        outputs = ft.evaluate(right, right.astype('f16'), x)
        self.assertEqual(outputs[0].tolist(), [14, 18, 22])
        self.assertEqual(outputs[1].tolist(), [14, 18, 22])
        self.assertIs(outputs[2], x)
        plan = self.engine.last_report['execution_plan']
        self.assertGreaterEqual(len(plan['steps']), 2)
        self.assertEqual(sum(step['kind'] == 'external' for step in plan['steps']), 2)

    def test_pointer_order_and_external_only_execution_without_ask_codegen_tools(self):
        source = """import triton
import triton.language as tl
@triton.jit
def subtract(out, N: tl.constexpr, rhs, lhs, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(lhs + index, index < N, 0)
    b = tl.load(rhs + index, index < N, 0)
    tl.store(out + index, a - b, index < N)
"""
        subtract = ft.triton.from_source(source, entry="subtract", inputs=("lhs", "rhs"), output="out",
            constants=lambda specs: {"N": specs[0].numel, "BLOCK": 128},
            grid=lambda specs: ((specs[0].numel + 127) // 128,))
        # Omit ASK codegen tools: the external worker returns finished machine code.
        with ft.Engine(device="cuda", compilers=self.engine.compilers,
                       cache_dir=self.cache_dir / "external-only") as engine:
            engine.llc = engine.ptxas = None
            x = ft.tensor([10, 20, 30], dtype="i32")
            y = ft.tensor([1, 3, 5], dtype="i32")
            before = engine.session.stats["launches"]
            self.assertEqual(subtract(x, y).tolist(), [9, 17, 25])
            self.assertEqual(engine.session.stats["launches"], before + 1)
            self.assertEqual([step["kind"] for step in engine.last_report["execution_plan"]["steps"]], ["external"])

    def test_flash_attention_backward_uses_multiple_external_kernels(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('fuga_attention_example_bwd', EXAMPLES_DIR / 'flash_attention2.py')
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        for dtype in ('f16', 'bf16'):
            if dtype == 'bf16' and not self.supports_bf16:
                continue
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    receipt = example.validate(dtype=dtype, causal=causal, backward=True,
                                               m=9, n=11, d=32, dv=32)
                    self.assertEqual(receipt['steps'], ['owned', 'external', 'owned'])
                    self.assertGreaterEqual(
                        sum(step == 'external' for step in receipt['backward_plan_steps']), 4)
                    self.assertTrue(receipt['verified'])
                    self.assertLess(receipt['dq_error'], 0.02)
                    self.assertLess(receipt['dk_error'], 0.02)
                    self.assertLess(receipt['dv_error'], 0.02)
        # Backward also covers a different head/value width, not just D == DV.
        receipt = example.validate(dtype='f16', causal=False, backward=True,
                                   m=7, n=8, d=64, dv=32)
        self.assertTrue(receipt['verified'])
        self.assertEqual(receipt['backward_plan_steps'].count('external'), 5)

    def test_flash_attention_forward_and_owned_surroundings(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('fuga_attention_example', EXAMPLES_DIR / 'flash_attention2.py')
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        for dtype in ('f16', 'bf16'):
            if dtype == 'bf16' and not self.supports_bf16:
                continue
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    receipt = example.validate(dtype=dtype, causal=causal)
                    self.assertEqual(receipt['steps'], ['owned', 'external', 'owned'])
                    self.assertTrue(receipt['verified'])


if __name__ == "__main__":
    unittest.main()
