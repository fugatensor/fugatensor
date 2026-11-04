"""ASK Python SDK builder, numerical codecs, client protocol, and native bindings."""
from __future__ import annotations

import pytest

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
import ctypes
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))

from fugatensor._ask import Compiler, CompilerError, Module, ScalarType, TensorType, __version__
from fugatensor._ask.compiler import _native_result, _result
from fugatensor._ask.ir import mlir_string
from fugatensor._ask.numerics import decode_float, encode_f32, encode_float, round_f32, round_float
from tests.backend.support.host_runner import ask_opt_path
from tests.backend.support.graphs import external_module

NATIVE_AVAILABLE = importlib.util.find_spec("fugatensor._native") is not None
if os.environ.get("ASK_NATIVE_REQUIRED") and not NATIVE_AVAILABLE:
    raise RuntimeError("fugatensor._native extension is required when ASK_NATIVE_REQUIRED=1")


def self_owned_module() -> Module:
    module = Module()
    tensor = TensorType((10,))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
    t = graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (10,)), id="P")
    t = graph.mark(t, mark_id="after_P")
    y = graph.mul(t, graph.splat(graph.constant(2), (10,)), id="Q")
    graph.return_(y)
    return module


def mixed_f16_module() -> Module:
    module = Module()
    tensor = TensorType((2, 5), "f16")
    graph = module.graph("mixed", inputs={"x": tensor}, results=[tensor])
    wide = graph.cast(graph.inputs["x"], dtype="f32")
    bias = graph.splat(graph.constant(0.1, dtype="f32"), (2, 5))
    result = graph.cast(graph.add(wide, bias), dtype="f16")
    graph.return_(graph.mark(result, mark_id="output"))
    return module


class TestBuilderAndCodec:
    def test_types_constants_and_explicit_casts(self):
        assert (str(TensorType((None, 4)))) == ('tensor<?x4xi32>')
        assert (str(TensorType(()))) == ('tensor<i32>')
        assert (str(ScalarType('fp16'))) == ('f16')
        assert (str(ScalarType('bool'))) == ('i1')
        for bad_shape in ((-1,), (True,), (2**63,), ("10",)):
            with pytest.raises(ValueError):
                TensorType(bad_shape)
        with pytest.raises(ValueError):
            ScalarType("f64")

        module = Module()
        graph = module.graph("mixed", inputs={"x": TensorType((2, 3), "bool")},
                             results=[TensorType((2, 3), "f32")])
        widened = graph.cast(graph.inputs["x"], dtype="i64", id="widen", value_id="int_value")
        out = graph.cast(widened, dtype="f32")
        graph.return_(out)
        mlir = module.to_mlir()
        assert (mlir.count('"ask.cast"')) == (2)
        assert ('ask.value_id = "int_value"') in (mlir)

        consts = Module().graph("consts", inputs={}, results=[])
        true_val = consts.constant(True, dtype="bool")
        for op in (consts.add, consts.mul):
            with pytest.raises(TypeError, match='bool add/mul'):
                op(true_val, true_val)
        for bad_bool in (0, 1, "true"):
            with pytest.raises(ValueError):
                consts.constant(bad_bool, dtype="bool")
        for bad_i64 in (True, 2**63, -(2**63) - 1):
            with pytest.raises(ValueError):
                consts.constant(bad_i64, dtype="i64")
        consts.constant(-(2**63), dtype="i64")
        consts.constant(2**63 - 1, dtype="i64")
        consts.return_()

    def test_float_constants_and_direct_rne_codecs(self):
        for dtype, precision, zero, one in (
            ("f32", 23, "80000000", "3F800000"),
            ("f16", 10, "8000", "3C00"),
            ("bf16", 7, "8000", "3F80"),
        ):
            module = Module()
            graph = module.graph("main", inputs={}, results=[ScalarType(dtype)])
            graph.return_(graph.add(graph.constant(-0.0, dtype=dtype),
                                    graph.constant(1.0 + 2.0**-(precision + 1), dtype=dtype)))
            source = module.to_mlir()
            assert (f'0x{zero} : {dtype}') in (source)
            assert (f'0x{one} : {dtype}') in (source)
            assert ('overflow') not in (source)

        for dtype, fraction_bits, minimum, maximum in (
            ("f16", 10, -24, 65504.0),
            ("bf16", 7, -133, float.fromhex("0x1.fep127")),
        ):
            assert (encode_float(-0.0, dtype)) == (b'\x00\x80')
            assert (encode_float(2.0 ** minimum, dtype)) == (b'\x01\x00')
            assert (encode_float(2.0 ** (minimum - 1), dtype)) == (b'\x00\x00')
            assert (round_float(maximum, dtype)) == (maximum)
            with pytest.raises(ValueError):
                round_float(maximum * 2, dtype)
            midpoint = 1.0 + 2**-(fraction_bits + 1)
            assert (round_float(midpoint, dtype)) == (1.0)
            assert (round_float(math.nextafter(midpoint, math.inf), dtype)) == (1.0 + 2 ** (-fraction_bits))
            for bits in (0x0000, 0x0001, 0x3C00, 0x3F80, 0x7BFF, 0x7C00, 0x7E00, 0x8000, 0xFC00, 0xFFFF):
                encoded = bits.to_bytes(2, "little")
                reference = (struct.unpack("<e", encoded)[0] if dtype == "f16"
                             else struct.unpack("<f", bytes(2) + encoded)[0])
                actual = decode_float(encoded, dtype)
                if math.isnan(reference):
                    assert (math.isnan(actual))
                else:
                    assert (actual) == (reference)
                    assert (encode_float(actual, dtype)) == (encoded)

        bf16_mid = 2**80 + 2**72
        assert (round_float(bf16_mid, 'bf16')) == (float(2 ** 80))
        assert (round_float(bf16_mid + 1, 'bf16')) == (float(2 ** 80 + 2 ** 73))

        f32_mid = 2**80 + 2**56
        assert (round_f32(f32_mid)) == (float(2 ** 80))
        assert (round_f32(f32_mid + 1)) == (float(2 ** 80 + 2 ** 57))
        assert (encode_f32(-0.0)) == (bytes.fromhex('00000080'))

    def test_graph_scoping_escaping_and_markers(self):
        assert (mlir_string('a"\\\n\x00中')) == ('"a\\22\\5C\\0A\\00\\E4\\B8\\AD"')
        for bad_sym in ("main) {", "main\n", "主函数", "main.with.dot"):
            with pytest.raises(ValueError):
                Module().graph(bad_sym, inputs={}, results=[])

        module = Module()
        graph = module.graph("main", inputs={"a": TensorType((None, 4)), "b": TensorType((10, None)),
                                             "ask_v0": ScalarType()}, results=[TensorType((10, 4))])
        merged = graph.add(graph.inputs["a"], graph.inputs["b"], id="sum")
        assert (merged.type) == (TensorType((10, 4)))
        assert (graph.constant(1).name) != (graph.inputs['ask_v0'].name)
        marked = graph.mark(merged, mark_id="trace_sum")
        assert (marked.value_id) == (merged.value_id)
        with pytest.raises(ValueError):
            graph.constant(2, id="sum")
        graph.return_(marked)
        with pytest.raises(ValueError):
            graph.constant(3)

        was_imported = "triton" in sys.modules
        external_mlir = external_module().to_mlir()
        assert ('compiler = "triton"') in (external_mlir)
        assert ('triton' in sys.modules) == (was_imported)


class TestClientProtocol:
    def test_cli_result_envelopes_reject_missing_or_malformed_contracts(self):
        valid = {"schema_version": 1, "phase": "prepared", "shape_snapshot": {}, "external_requests": []}
        for missing in ("schema_version", "phase", "shape_snapshot", "external_requests"):
            with pytest.raises(CompilerError):
                _result("prepare", "module {}", {key: value for key, value in valid.items() if key != missing})
        for changes in ({"schema_version": True}, {"schema_version": 2}, {"phase": "partitioned"},
                        {"shape_snapshot": []}, {"external_requests": {}}, {"external_requests": [1]}):
            with pytest.raises(CompilerError):
                _result("prepare", "module {}", {**valid, **changes})
        for ir in ("", " \n"):
            with pytest.raises(CompilerError, match='nonempty IR'):
                _result("prepare", ir, valid)

        plan = {
            "schema_version": 1, "kind": "ask.cuda.execution_plan", "inputs": ["x"], "outputs": ["y"],
            "buffers": [{"id": name, "shape": [10], "element_type": "i32"} for name in ("x", "y")],
            "steps": [{"id": "step0", "kind": "owned", "function": "main", "inputs": ["x"],
                       "outputs": ["y"], "depends_on": []}],
            "scheduling": {"policy": "sequential", "stream": "session"},
        }
        partition = {**valid, "phase": "partitioned", "execution_plan": plan,
                     "step_sources": {"step0": "module {}"}}
        assert (_result('partition', 'module {}', partition).execution_plan) == (plan)
        for sources in ({}, {"step0": ""}, {"step0": "module {}", "unknown": "module {}"}):
            with pytest.raises(CompilerError):
                _result("partition", "module {}", {**partition, "step_sources": sources})

        from tests.backend.test_cuda_compilation import make_artifact
        artifact = make_artifact()
        cuda = {**valid, "phase": "cuda_llvm", "cuda_artifact": artifact}
        assert (_result('codegen', artifact['payload'], cuda, target='cuda-llvm').cuda_artifact) == (artifact)
        with pytest.raises(CompilerError, match='payload differs'):
            _result("codegen", artifact["payload"] + "; changed\n", cuda, target="cuda-llvm")

    def test_native_result_contract_does_not_depend_on_diagnostic_fields(self):
        native = SimpleNamespace(ir="module {}", shape_snapshot_json='{"shape_bindings": []}',
                                 external_requests_json="[]", diagnostics_json="{}")
        empty = _native_result("prepare", native)
        native.diagnostics_json = '{"message": "new diagnostic format"}'
        changed = _native_result("prepare", native)
        assert (empty) == (replace(changed, diagnostics={}))
        assert (changed.diagnostics) == ({'message': 'new diagnostic format'})

    def test_cli_driver_protocol_and_error_handling(self):
        def fake_run(command, **kwargs):
            assert ('shell') not in (kwargs)
            Path(command[command.index("-o") + 1]).write_text("module {}\n", encoding="utf-8")
            Path(command[command.index("--report-json") + 1]).write_text(json.dumps({
                "schema_version": 1, "phase": "prepared", "shape_snapshot": {}, "external_requests": [],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("fugatensor._ask.compiler.subprocess.run", side_effect=fake_run):
            result = Compiler("/path with spaces/ask-opt").prepare("module {}")
        assert (result.diagnostics) == ({'schema_version': 1, 'phase': 'prepared'})
        assert (result.shape_snapshot) == ({})
        assert (result.external_requests) == ([])

        with patch("fugatensor._ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "bad shape")):
            with pytest.raises(CompilerError) as caught:
                Compiler("ask-opt").prepare("module {}")
        assert (caught.value.diagnostics) == ('bad shape')

        with patch("fugatensor._ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], -11, "", "crash")):
            with pytest.raises(CompilerError, match='signal 11'):
                Compiler("ask-opt").prepare("module {}")

        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(CompilerError, match='not found'):
                Compiler(Path(directory) / "missing-ask-opt").prepare(self_owned_module())


class TestNativeCompilerAndBinding:
    def test_cli_prepare_shape_refinement_and_casts(self):
        compiler = Compiler(ask_opt_path())
        owned = compiler.prepare(self_owned_module())
        assert ('"ask.mark"') not in (owned.ir)
        assert ('after_P') in (json.dumps(owned.diagnostics))

        ext = compiler.prepare(external_module())
        assert ('"ask.external"') in (ext.ir)
        assert (len(ext.external_requests)) == (1)

        marker_id = '引号"\\\n\x00'
        escaped = self_owned_module().to_mlir().replace('mark_id = "after_P"', 'mark_id = ' + mlir_string(marker_id))
        assert (compiler.prepare(escaped).diagnostics['markers'][0]['mark_id']) == (marker_id)

        dtypes = ("bool", "i32", "i64", "f32", "f16", "bf16")
        cast_mod = Module()
        for src in dtypes:
            g = cast_mod.graph("from_" + src, inputs={"x": TensorType((2, 5), src)},
                               results=[TensorType((2, 5), dst) for dst in dtypes])
            g.return_(*(g.cast(g.inputs["x"], dtype=dst) for dst in dtypes))
        prepared = compiler.prepare(cast_mod)
        assert (prepared.ir.count('"ask.cast"')) == (36)

    def test_native_codegen_without_companion_and_cli_parity(self):
        module = Module()
        integer, floating = TensorType((1,), "i64"), TensorType((1,), "f32")
        graph = module.graph("literals", inputs={"x": integer}, results=[integer, floating])
        graph.return_(graph.add(graph.inputs["x"], graph.splat(graph.constant(2**63 - 1, dtype="i64"), (1,))),
                      graph.splat(graph.constant(-0.0, dtype="f32"), (1,)))
        native = Compiler()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("fugatensor._ask.compiler.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
                cuda = native.emit_cuda_llvm(module, arch=86)
                assert (cuda.cuda_artifact['payload']) == (cuda.ir)
                assert ('9223372036854775807') in (cuda.ir)
                assert ('-0.000000e+00') in (cuda.ir)
                with pytest.raises(CompilerError):
                    native.emit_cuda_llvm("module { unknown }", arch=86)
                with pytest.raises(ValueError):
                    native.emit_cuda_llvm(module, arch=True)
                if native.capabilities()["host_codegen"]:
                    host = native.emit_host_llvm(module)
                    obj = native.emit_host_object(module, root / "native.o")
                    assert (obj.ir) == ('')
                    assert (obj.diagnostics['phase']) == ('host_object')
                    assert ((root / 'native.o').read_bytes()[:4]) == (b'\x7fELF')
            if native.capabilities()["host_codegen"]:
                from tests.backend.support.host_runner import link_object_to_cdll
                library = link_object_to_cdll(root / "native.o", root / "native.so")
                function = library.literals
                function.argtypes = [ctypes.c_void_p] * 3
                source, integer_output, float_output = ctypes.c_int64(1), ctypes.c_int64(), ctypes.c_uint32()
                function(ctypes.byref(source), ctypes.byref(integer_output), ctypes.byref(float_output))
                assert (integer_output.value) == (-2 ** 63)
                assert (float_output.value) == (2147483648)
                from fugatensor.backends.host import HostExecutable, HostStorage
                executable = HostExecutable(root / "native.so", replace(obj, diagnostics={}), module.to_mlir(), root)
                outputs = executable.run([HostStorage.from_flat([1], shape=(1,), dtype="i64")])
                assert (outputs[0].to_flat()) == ([-2 ** 63])
                assert (outputs[1].to_buffer().tobytes()) == (b'\x00\x00\x00\x80')
                assert (executable.report) == ({})
            if os.environ.get("ASK_OPT"):
                cli = Compiler(os.environ["ASK_OPT"])
                assert (cuda) == (cli.emit_cuda_llvm(module, arch=86))
                if native.capabilities()["host_codegen"]:
                    assert (host) == (cli.emit_host_llvm(module))
                    cli_object = cli.emit_host_object(module, root / "cli.o")
                    assert (cli_object.output_path) == (str(root / 'cli.o'))
                    assert (obj) == (replace(cli_object, output_path=obj.output_path))

    def test_in_process_bindings_and_cli_parity(self):
        with patch("fugatensor._ask.compiler.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
            native = Compiler()
            caps = native.capabilities()
            prepared = native.prepare(mixed_f16_module())
            optimized = native.optimize(mixed_f16_module())
            with pytest.raises(CompilerError) as caught:
                native.prepare("module { unknown }")
        assert ('python_bindings') not in (caps)
        assert (caps['ask_version']) == (__version__)
        assert (prepared.diagnostics['phase']) == ('prepared')
        assert (optimized.diagnostics['phase']) == ('optimized')
        assert ('<ask-source>') in (caught.value.diagnostics)

        source = mixed_f16_module().to_mlir()
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(native.prepare, [source] * 8))
        assert (all((result == outputs[0] for result in outputs)))

        if os.environ.get("ASK_OPT"):
            cli = Compiler(os.environ["ASK_OPT"])
            assert (native.capabilities()) == (cli.capabilities())
            assert (native.prepare(mixed_f16_module())) == (cli.prepare(mixed_f16_module()))

        from fugatensor._ask import _native
        with patch.object(_native, "capabilities", return_value=json.dumps(dict(caps, ask_version="0.0.0"))):
            with pytest.raises(CompilerError, match='matching SDK'):
                Compiler()
