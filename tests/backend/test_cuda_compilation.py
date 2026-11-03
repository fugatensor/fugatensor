"""Native CUDA graph lowering, CUDA compilation contract, adapter routing, and prepared executable validation."""
from __future__ import annotations

import base64
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))
__import__('sys').path.insert(0, str(Path(__file__).resolve().parent))

from fugatensor._ask import CompilerEnvironment, CompilerError, Module, ScalarType, TensorType
from fugatensor._ask._abi import parameter_layout, scalar_bytes
from fugatensor._ask.cuda import (
    CUDAError,
    _artifact_digest,
    _digest,
    _kernel_contract,
    compile_cuda,
    install_external,
    execute,
)
from fugatensor._ask.gpu import CUDAKernel, compile_graph
from fugatensor._ask.runtime import DeviceTensor, ExecutionSession, _encode, _decode
from support.host_runner import run_ask_opt
from support.compilation import owned_partition
from support.pointwise import DTYPES, build_module
from test_runtime_and_cache import FakeNativeSession


def make_artifact(size: int = 10, *, block: int = 128, dtype: str = "i32") -> dict:
    return {
        "format": "llvm-ir",
        "entry": "scale2",
        "payload": "define void @fixture() { ret void }\n",
        "target": {"backend": "cuda", "arch": 80},
        "port_map": [
            {"argument": idx, "logical_port": binding, "element_type": dtype,
             "shape": [size], "layout": "contiguous_row_major"}
            for idx, binding in enumerate(("input:0", "output:0"))
        ],
        "invocation": {
            "kind": "cuda-kernel",
            "version": 2,
            "grid": [max(1, (size + block - 1) // block), 1, 1],
            "block": [block, 1, 1],
            "shared_bytes": 0,
            "arguments": [
                {"argument": idx, "binding": binding, "llvm_type": "ptr addrspace(1)"}
                for idx, binding in enumerate(("input:0", "output:0"))
            ],
        },
        "codegen": {"fp_contract": "fast", "ptx_version": 80},
        "required_element_types": [dtype],
    }


def make_external_artifact(format_="ptx", *, shape=(2, 5), payload=None, arch=80):
    if payload is None:
        payload = (".version 8.0\n.target sm_80\n.address_size 64\n" if format_ == "ptx"
                   else base64.b64encode(b"\x7fELF" + b"\0" * 60).decode())
    artifact = make_artifact()
    artifact.pop("codegen")
    artifact.update(protocol_version=2, format=format_, payload=payload,
                    producer={"name": "fixture", "version": "1"}, target={"backend": "cuda", "arch": arch})
    artifact["invocation"]["completion"] = "device-event"
    artifact["invocation"]["arguments"] = [
        {"argument": i, "binding": name, "kind": "pointer"}
        for i, name in enumerate(("input:0", "output:0"))]
    for port in artifact["port_map"]:
        port["shape"] = list(shape)
    return artifact


def mock_tensor(session, shape=(2, 5), dtype="i32", pointer=4096, nbytes=40, readonly=False):
    buffer = SimpleNamespace(data_ptr=pointer, nbytes=nbytes, readonly=readonly)
    return DeviceTensor(session, buffer, shape, dtype)


class SessionRecorder(ExecutionSession):
    def __init__(self):
        self.loads = []
        self._lock, self._pending = threading.RLock(), []
        self._native = FakeNativeSession()

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return object()

    @property
    def submissions(self):
        return [submission for program in self._native.programs for submission in program.submissions]


class CUDAGraphLoweringTests(unittest.TestCase):
    def lower(self, module: Module, *, arch: int = 86, optimize: bool = True):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            flags = ["-", "--emit-cuda-llvm", f"--cuda-arch={arch}", "--report-json", report]
            if optimize:
                flags.append("--optimize")
            result = run_ask_opt(*flags, source=module.to_mlir())
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(report.read_text(encoding="utf-8"))
        artifact = metadata["cuda_artifact"]
        self.assertEqual(artifact["invocation"]["version"], 2)
        self.assertEqual(run_ask_opt("-", "--check-llir", source=result.stdout).returncode, 0)
        return artifact, metadata

    def test_all_precisions_ports_empty_and_64bit_indexing(self):
        for dtype in DTYPES:
            artifact, metadata = self.lower(build_module(129, dtype=dtype, external=False))
            self.assertIn("define ptx_kernel void @pipeline", artifact["payload"])
            self.assertEqual(artifact["invocation"]["grid"], [2, 1, 1])
            self.assertIn(dtype, artifact["required_element_types"])
            self.assertEqual(metadata["cuda_lowering"]["temporary_buffer_count"], 0)

        multi = Module()
        vec, other = TensorType((129,), "f32"), TensorType((3,), "i64")
        g = multi.graph("calc", inputs={"x": vec, "s": ScalarType("f32"), "y": other},
                        results=[vec, TensorType((129,), "i64"), other, ScalarType("f32")])
        val = g.add(g.inputs["x"], g.splat(g.inputs["s"], (129,)))
        g.return_(val, g.cast(val, dtype="i64"), g.inputs["y"], g.inputs["s"])
        art, rep = self.lower(multi)
        self.assertEqual(rep["cuda_lowering"]["output_loop_count"], 3)
        self.assertEqual(rep["cuda_lowering"]["groups"][0]["outputs"], [0, 1])

        empty_art, empty_rep = self.lower(build_module(shape=(0, 3), dtype="f16", external=False))
        self.assertNotIn(" load ", empty_art["payload"])
        self.assertEqual(empty_rep["cuda_lowering"]["zero_element_outputs"], 1)

        large = Module()
        lt = TensorType((2**31 + 17,), "f32")
        lg = large.graph("large", inputs={"x": lt}, results=[lt])
        lg.return_(lg.inputs["x"])
        large_art, _ = self.lower(large)
        self.assertEqual(large_art["invocation"]["grid"], [65535, 1, 1])
        self.assertIn("icmp ult i64", large_art["payload"])

    def test_bf16_arch_and_flag_rejections(self):
        bf = Module()
        t = TensorType((3,), "f32")
        g = bf.graph("conv", inputs={"x": t}, results=[t])
        g.return_(g.cast(g.cast(g.inputs["x"], dtype="bf16"), dtype="f32"))
        rejected = run_ask_opt("-", "--emit-cuda-llvm", "--cuda-arch=75", source=bf.to_mlir())
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("bf16 compilation requires target sm80", rejected.stderr)

        src = build_module(3, external=False).to_mlir()
        for flags, needle in (
            (["--emit-cuda-llvm"], "explicit --cuda-arch"),
            (["--emit-cuda-llvm", "--cuda-arch=86", "--cuda-block-size=0"], "block size"),
            (["--emit-llvm", "--emit-cuda-llvm"], "select only one"),
        ):
            res = run_ask_opt("-", *flags, source=src)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn(needle, res.stderr)
        ext = run_ask_opt("-", "--emit-cuda-llvm", "--cuda-arch=86", source=build_module(3).to_mlir())
        self.assertNotEqual(ext.returncode, 0)
        self.assertIn("cannot compile ask.external", ext.stderr)


class CUDAContractAndRoutingTests(unittest.TestCase):
    def compile_fixture(self, directory: Path, value: dict):
        source = directory / "source.ll"
        source.write_text(value["payload"], encoding="utf-8")

        def fake_tool(command, **kwargs):
            Path(command[-1]).write_bytes(b"fixture binary\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("fugatensor._ask.cuda.subprocess.run", side_effect=fake_tool) as run_mock:
            result = compile_cuda(source, value, directory / "codegen", llc="/test/llc", ptxas="/test/ptxas")
        self.assertEqual(run_mock.call_count, 2)
        return result

    def test_codegen_commands_tamper_checks_and_host_buffer_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            val = make_artifact()
            res = self.compile_fixture(work, val)
            self.assertEqual(res["commands"][0][1:5],
                             ["-march=nvptx64", "-mcpu=sm_80", "-mattr=+ptx80", "--fp-contract=fast"])
            self.assertEqual(res["commands"][1][1], "--gpu-name=sm_80")
            Path(res["cubin"]).write_bytes(b"tampered")
            with patch("fugatensor._ask.runtime.ExecutionSession") as drv:
                with self.assertRaises(CUDAError):
                    execute(val, res, [0] * 10)
                drv.assert_not_called()

        for mutate in (
            lambda a: a["invocation"].update(grid=[0, 1, 1]),
            lambda a: a["invocation"].update(version=1),
            lambda a: a["target"].update(arch="90a"),
            lambda a: a["port_map"][0].update(layout="strided"),
            lambda a: a["codegen"].update(fp_contract="unknown"),
        ):
            bad = make_artifact()
            mutate(bad)
            with patch("fugatensor._ask.runtime.ExecutionSession") as drv:
                with self.assertRaises(CUDAError):
                    execute(bad, {}, [0] * 10)
                drv.assert_not_called()

        # Large integer rounds directly to f32 without double intermediate loss.
        buf = _encode([2**100 + 2**76 + 1], "f32", 1)
        self.assertEqual(_decode(buf, "f32")[0], float(2**100 + 2**77))
        for dtype, bits in (("f16", 10), ("bf16", 7)):
            half_ulp = 2**(-bits - 1)
            src = _encode([1.0 + half_ulp, 1.0 + 3 * half_ulp, -0.0], dtype, 3)
            self.assertEqual(len(src), 6)
            dec = _decode(src, dtype)
            self.assertEqual(dec[:2], [1.0, 1.0 + 2**(1 - bits)])
            self.assertEqual(math.copysign(1.0, dec[2]), -1.0)

    def test_conditional_libdevice_linking_and_dependency_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ('llc', 'ptxas', 'llvm-link', 'libdevice.bc')}
            for name, path in paths.items():
                path.write_text(name)
            artifact = make_artifact()
            artifact['payload'] += 'declare float @__nv_expf(float)\n'
            source = root / 'input.ll'
            source.write_text(artifact['payload'])
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                Path(command[command.index('-o') + 1]).write_bytes(b'compiled')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with patch('fugatensor._ask.cuda.subprocess.run', side_effect=run):
                results = []
                for index, changed in enumerate((None, None, 'libdevice.bc', 'llvm-link')):
                    if changed:
                        paths[changed].write_text(changed + ' updated')
                    results.append(compile_cuda(source, artifact, root / str(index),
                        llc=paths['llc'], ptxas=paths['ptxas'], llvm_link=paths['llvm-link'],
                        libdevice=paths['libdevice.bc'], cache_dir=root / 'cache'))
                self.assertEqual([result['compiler_invocations'] for result in results], [3, 0, 3, 3])
                self.assertEqual(Path(commands[0][0]).name, 'llvm-link')
                self.assertIn('--only-needed', commands[0])
                self.assertIn('kernel.linked.bc', ' '.join(commands[1]))
                plain = make_artifact()
                source.write_text(plain['payload'])
                result = compile_cuda(source, plain, root / 'plain', llc=paths['llc'], ptxas=paths['ptxas'],
                                      llvm_link=root / 'missing-link', libdevice=root / 'missing-libdevice')
                self.assertEqual(result['compiler_invocations'], 2)
            source.write_text(artifact['payload'])
            with self.assertRaisesRegex(CUDAError, 'requires libdevice'):
                compile_cuda(source, artifact, root / 'missing', llc=paths['llc'], ptxas=paths['ptxas'],
                             libdevice=root / 'missing-libdevice')

    def test_owned_graph_routing(self):
        owned_compiler = Mock(executable="ask-opt", _native=None)
        owned_compiler.capabilities.return_value = {
            "external_execution": True, "external_formats": ["cubin", "ptx"],
            "cuda_graph_codegen": True, "llvm_revision": "rev",
        }
        owned_compiler.partition.return_value = owned_partition("graph")
        stats = {"status": "hit", "counters": {"hits": 1}}
        with tempfile.TemporaryDirectory() as directory, \
             patch("fugatensor._ask.gpu._tool", return_value={"path": "/t", "sha256": "a" * 64}), \
             patch("fugatensor._ask.gpu.IRCache") as cache_cls, \
             patch("fugatensor._ask.gpu.compile_cuda", return_value={"executable_cache": stats}):
            cache_cls.return_value.get_or_create.return_value = (
                {"kernel.ll": b"LL", "kernel.artifact.json": b"{}", "report.json": b"{}"}, stats
            )
            kernel = compile_graph(
                owned_compiler, "graph", target={"backend": "cuda", "arch": 86},
                cache_dir=Path(directory) / "c", output_dir=Path(directory) / "o", llc="llc", ptxas="ptxas",
            )
            key = cache_cls.return_value.get_or_create.call_args.args[0]
            self.assertEqual(key["producer"], {"name": "ask"})
            self.assertEqual(kernel.cache["status"], "hit")

    def test_owned_cache_tracks_loaded_compiler_library_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libask_compiler.so"
            library.write_bytes(b"compiler-version-one")
            compiler = Mock(_native=SimpleNamespace(compiler_library_path=lambda: str(library)))
            compiler.capabilities.return_value = {"cuda_graph_codegen": True, "llvm_revision": "rev"}
            compiler.partition.return_value = owned_partition("module {}")
            compiler.emit_cuda_llvm.side_effect = lambda *args, **kwargs: SimpleNamespace(
                ir=make_artifact()["payload"], cuda_artifact=make_artifact(), diagnostics={})
            with patch("fugatensor._ask.gpu._tool", return_value={"path": "/tool", "sha256": "tool"}), \
                 patch("fugatensor._ask.gpu.compile_cuda", return_value={"executable_cache": {"status": "hit"}}):
                first = compile_graph(compiler, "module {}", cache_dir=root / "cache", output_dir=root / "first",
                                      llc="llc", ptxas="ptxas", target={"backend": "cuda", "arch": 80})
                library.write_bytes(b"compiler-version-two")
                second = compile_graph(compiler, "module {}", cache_dir=root / "cache", output_dir=root / "second",
                                       llc="llc", ptxas="ptxas", target={"backend": "cuda", "arch": 80})
                third = compile_graph(compiler, "module {}", cache_dir=root / "cache", output_dir=root / "third",
                                      llc="llc", ptxas="ptxas", target={"backend": "cuda", "arch": 80})
            self.assertEqual(compiler.emit_cuda_llvm.call_count, 2)
            self.assertEqual((first.cache["status"], second.cache["status"], third.cache["status"]),
                             ("miss", "miss", "hit"))
            self.assertEqual(first.artifact, make_artifact())
            self.assertNotIn("cuda_artifact", first.report)
            self.assertNotEqual(first.report["ir_cache_key_inputs"]["tools"]["ask-compiler"]["sha256"],
                                second.report["ir_cache_key_inputs"]["tools"]["ask-compiler"]["sha256"])


class PreparedExecutableContractTests(unittest.TestCase):
    def make_kernel_package(self, directory: Path):
        val = make_artifact()
        for port in val["port_map"]:
            port["shape"] = [2, 5]
        src, cubin = directory / "in.ll", directory / "k.cubin"
        src.write_text(val["payload"], encoding="utf-8")
        cubin.write_bytes(b"synthetic cubin")
        receipt = {
            "schema_version": 1, "entry": val["entry"], "arch": 80, "invocation": copy.deepcopy(val["invocation"]),
            "element_type": "i32", "fp_contract": "fast", "artifact_sha256": _artifact_digest(val),
            "llvm_sha256": _digest(src.read_bytes()), "llvm_ir": str(src),
            "cubin_sha256": _digest(cubin.read_bytes()), "cubin": str(cubin),
        }
        return CUDAKernel(val, receipt, {"shape_snapshot": {"shape_bindings": []}}, {}), SessionRecorder()

    def test_single_and_multi_port_hot_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            kernel, session = self.make_kernel_package(Path(directory))
            exe = kernel.prepare(session)
            src, dst = mock_tensor(session), mock_tensor(session, pointer=8192)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("hot path I/O")):
                exe.submit(src, dst)
                exe.submit(src, dst)
            self.assertEqual((len(session.loads), len(session.submissions)), (1, 2))
            for bad, msg in (
                (mock_tensor(session, shape=(10,)), "shape"),
                (mock_tensor(session, dtype="f32"), "dtype"),
            ):
                with self.assertRaisesRegex(CUDAError, msg):
                    exe.submit(bad, dst)
            with self.assertRaisesRegex(CUDAError, "DeviceTensor"):
                exe.submit(src, SimpleNamespace(shape=(2, 5), dtype="i32"))

            # Multi-port scrambled physical order vs stable logical order.
            m_kernel, m_session = self.make_kernel_package(Path(directory))
            specs = [("output:1", [], "i64"), ("input:1", [], "i64"),
                     ("output:0", [2, 5], "f32"), ("input:0", [2, 5], "i32")]
            m_art = m_kernel.artifact
            m_art["port_map"] = [
                {"argument": idx, "logical_port": bind, "shape": shp,
                 "element_type": dt, "layout": "contiguous_row_major"}
                for idx, (bind, shp, dt) in enumerate(specs)
            ]
            m_art["invocation"]["arguments"] = [
                {"argument": idx, "binding": bind, "llvm_type": "ptr addrspace(1)"}
                for idx, (bind, _, _) in enumerate(specs)
            ]
            m_kernel.codegen.update(element_type=None, invocation=copy.deepcopy(m_art["invocation"]),
                                    artifact_sha256=_artifact_digest(m_art))
            m_exe = m_kernel.prepare(m_session)
            ins = [mock_tensor(m_session), mock_tensor(m_session, shape=(), dtype="i64", pointer=8192, nbytes=8)]
            outs = [mock_tensor(m_session, dtype="f32", pointer=12288),
                    mock_tensor(m_session, shape=(), dtype="i64", pointer=16384, nbytes=8)]
            m_exe.submit_many(ins, outs)
            self.assertEqual(m_session._native.programs[0].steps[0]["arguments"], [3, 1, 2, 0])
            self.assertEqual(m_session.submissions[0], ([value._buffer for value in ins], [value._buffer for value in outs]))

            bad_sparse = copy.deepcopy(m_art)
            bad_sparse["port_map"][1]["logical_port"] = "input:3"
            bad_sparse["invocation"]["arguments"][1]["binding"] = "input:3"
            with self.assertRaises(CUDAError):
                _kernel_contract(bad_sparse)


class ExternalExecutableTests(unittest.TestCase):
    def test_scalar_encodings_and_mixed_prepared_arguments(self):
        samples = [("i1", True, "01"), ("i8", -128, "80"), ("i16", -32768, "0080"),
                   ("i32", -2147483648, "00000080"), ("i64", -(2**63), "0000000000000080"),
                   ("u8", 255, "ff"), ("u16", 65535, "ffff"), ("u32", 2**32 - 1, "ffffffff"),
                   ("u64", 2**64 - 1, "ffffffffffffffff"), ("fp16", 1.5, "003e"),
                   ("bf16", -2.5, "20c0"), ("fp32", 1.25, "0000a03f"),
                   ("fp64", -0.0, "0000000000000080")]
        for dtype, value, expected in samples:
            with self.subTest(dtype=dtype):
                self.assertEqual(scalar_bytes(dtype, value).hex(), expected)
        # Encoding a large integer into f32 must avoid an intermediate f64 rounding.
        self.assertEqual(scalar_bytes("fp32", 2**100 + 2**76 + 1).hex(), "01008071")
        for dtype, value in (("i1", 1), ("i32", True), ("u8", -1), ("i8", 128),
                             ("u64", 2**64), ("fp32", True), ("fp32", float("inf")),
                             ("fp16", 65536), ("fp64", 10**400), ("bad", 1)):
            with self.subTest(dtype=dtype, value=value), self.assertRaises(ValueError):
                scalar_bytes(dtype, value)
        self.assertEqual(parameter_layout(["pointer", "i1", "i16", "fp32", "pointer", "fp64"]),
                         [{"offset": 0, "size": 8}, {"offset": 8, "size": 1},
                          {"offset": 10, "size": 2}, {"offset": 12, "size": 4},
                          {"offset": 16, "size": 8}, {"offset": 24, "size": 8}])
        with tempfile.TemporaryDirectory() as directory:
            artifact = make_external_artifact()
            artifact["invocation"]["arguments"].insert(1,
                {"argument": 1, "kind": "scalar", "type": "fp32", "value": 1.25})
            artifact["invocation"]["arguments"][2]["argument"] = 2
            artifact["port_map"][1]["argument"] = 2
            session = SessionRecorder()
            executable = CUDAKernel(artifact, install_external(artifact, directory), {}, {}).prepare(session)
            src, dst = mock_tensor(session), mock_tensor(session, pointer=8192)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("hot path I/O")):
                executable.submit(src, dst)
            self.assertEqual(session.loads[0][1]["parameter_types"], ["pointer", "fp32", "pointer"])
            self.assertEqual(session._native.programs[0].steps[0]["arguments"], [0, b"\x00\x00\xa0\x3f", 1])
            self.assertEqual(session.submissions[0], ([src._buffer], [dst._buffer]))

    def test_direct_formats_prepare_and_verify_without_compiler_tools(self):
        for format_ in ("cubin", "ptx"):
            with self.subTest(format=format_), tempfile.TemporaryDirectory() as directory:
                artifact = make_external_artifact(format_)
                with patch("fugatensor._ask.cuda.subprocess.run", side_effect=AssertionError("unexpected tool")):
                    receipt = install_external(artifact, directory)
                    session = SessionRecorder()
                    executable = CUDAKernel(artifact, receipt, {}, {}).prepare(session)
                    executable.submit(mock_tensor(session), mock_tensor(session, pointer=8192))
                self.assertEqual(receipt["compiler_invocations"], 0)
                self.assertNotIn("llvm_ir", receipt)
                expected = (base64.b64decode(artifact["payload"]) if format_ == "cubin"
                            else artifact["payload"].encode())
                self.assertEqual(session.loads[0][0][0], expected)
                self.assertEqual(Path(receipt[format_]).read_bytes(), expected)
                with self.assertRaisesRegex(CUDAError, "already contains"):
                    install_external(artifact, directory)
                Path(receipt[format_]).write_bytes(b"tampered")
                with self.assertRaisesRegex(CUDAError, "digest"), patch("fugatensor._ask.runtime.ExecutionSession") as driver:
                    execute(artifact, receipt, [0] * 10)
                driver.assert_not_called()

    def test_external_contract_payload_and_receipt_rejections(self):
        mutations = [
            lambda a: a.update(protocol_version=True),
            lambda a: a.update(format="llvm-bitcode"),
            lambda a: a.update(payload=".version 8.0\0"),
            lambda a: a.update(format="cubin", payload="invalid base64"),
            lambda a: a.update(format="cubin", payload=base64.b64encode(b"not ELF").decode()),
            lambda a: a["invocation"]["arguments"][0].update(kind="scalar"),
            lambda a: a["invocation"]["arguments"][0].update(binding="scratch:0"),
            lambda a: a["invocation"].update(completion="host"),
            lambda a: a["port_map"][0].update(argument=1),
        ]
        for mutate in mutations:
            artifact = make_external_artifact()
            mutate(artifact)
            with self.subTest(artifact=artifact), self.assertRaises(CUDAError):
                _kernel_contract(artifact)
        with tempfile.TemporaryDirectory() as directory:
            artifact = make_external_artifact()
            receipt = install_external(artifact, directory)
            for field, value in (("format", "cubin"), ("schema_version", 1),
                                 ("payload_sha256", "0" * 64), ("llvm_ir", "/etc/passwd")):
                bad = dict(receipt, **{field: value})
                with self.subTest(field=field), self.assertRaises(CUDAError):
                    CUDAKernel(artifact, bad, {}, {}).prepare(SessionRecorder())


if __name__ == "__main__":
    unittest.main()
