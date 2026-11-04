"""Native static graph partitioning and execution-plan SSA contracts, without a GPU."""
from __future__ import annotations

import pytest

import json
import math
from pathlib import Path
import subprocess
import tempfile

from fugatensor._ask import Compiler, CompilerError, Module, TensorType
from fugatensor._ask.compiler import _result
from tests.backend.support.host_runner import ask_opt_path

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))


class TestPartition:
    def partition(self, source, **options):
        return Compiler().partition(source, **options)

    def assert_plan(self, result):
        plan = result.execution_plan
        assert ((plan['schema_version'], plan['kind'])) == ((1, 'ask.cuda.execution_plan'))
        assert (plan['scheduling']) == ({'policy': 'sequential', 'stream': 'session'})
        buffers = {buffer["id"]: buffer for buffer in plan["buffers"]}
        available, completed = set(plan["inputs"]), set()
        for step in plan["steps"]:
            assert (set(step['inputs']) <= available)
            assert (set(step['depends_on']) <= completed)
            assert (set(step['outputs']) <= buffers.keys())
            assert not (set(step['outputs']) & available)
            available.update(step["outputs"])
            completed.add(step["id"])
            source = result.step_sources[step["id"]]
            assert (f'''ask.kernel_id = "{step['id']}"''') in (source)
            assert (f'''ask.kernel_kind = "{step['kind']}"''') in (source)
            Compiler().prepare(source)
        assert (set(plan['outputs']) <= available)
        assert ('ask.execution_plan') in (result.ir)
        return plan

    def test_owned_graph_and_identity_outputs_preserve_aliases(self):
        module = Module()
        type_ = TensorType((3,), "i32")
        graph = module.graph("main", inputs={"x": type_}, results=[type_, type_, type_])
        result = graph.add(graph.inputs["x"], graph.splat(graph.constant(2), (3,)))
        graph.return_(graph.inputs["x"], result, result)
        plan = self.assert_plan(self.partition(module))
        assert ([step['kind'] for step in plan['steps']]) == (['owned'])
        assert (plan['outputs'][0]) == (plan['inputs'][0])
        assert (plan['outputs'][1]) == (plan['outputs'][2])
        identity = self.partition("module { func.func @id(%x: tensor<i32>) -> tensor<i32> { return %x : tensor<i32> } }")
        plan = self.assert_plan(identity)
        assert (plan['steps']) == ([])
        assert (plan['inputs']) == (plan['outputs'])

    def test_external_boundaries_preserve_owned_regions_and_provenance(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        result = self.partition(source)
        plan = self.assert_plan(result)
        assert ([step['kind'] for step in plan['steps']]) == (['owned', 'external', 'owned'])
        assert (plan['steps'][1]['external_id']) == ('F')
        assert (len(result.diagnostics['markers'])) == (2)
        assert ('ask.source_markers') in (result.ir)
        assert (result.external_requests[0]['protocol_version']) == (2)
        assert (result.external_requests[0]['port_shapes']) == ({'input:0': [10], 'output:0': [10]})

    def test_fanout_preserves_public_outputs(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        source = source.replace('-> tensor<10xi32> {\n    %one', '-> (tensor<10xi32>, tensor<10xi32>) {\n    %one')
        source = source.replace('return %y : tensor<10xi32>', 'return %t, %y : tensor<10xi32>, tensor<10xi32>')
        plan = self.assert_plan(self.partition(source))
        assert (len(plan['steps'])) >= (2)
        assert (len(plan['outputs'])) == (2)

    def test_multiport_shape_change_and_duplicate_external_inputs(self):
        source = '''module {
          "ask.semantics"() {sym_name = "ports", function_type = (tensor<4xf32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<1xi32>), shape_rule = "fixed", effects = "none"} : () -> ()
          "ask.implementation"() {sym_name = "impl", compiler = "fixture", source = "opaque source", entry = "ports", signature = {a = "*fp32", b = "*fp32", c = "*fp32", d = "*i32"}, source_bindings = {a = "input:0", b = "input:1", c = "output:0", d = "output:1"}, constants = {}, options = {}, target = {backend = "cuda", arch = 80 : i64}, grid = array<i64: 1, 1, 1>} : () -> ()
          func.func @main(%x: tensor<4xf32>) -> (tensor<2xf32>, tensor<1xi32>) {
            %a, %b = "ask.external"(%x, %x) {ask.id = "multi", semantics = @ports, implementation = @impl} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<1xi32>)
            return %a, %b : tensor<2xf32>, tensor<1xi32>
          }
        }'''
        result = self.partition(source)
        plan = self.assert_plan(result)
        assert (len(plan['steps'])) == (1)
        step = plan["steps"][0]
        assert (step['kind']) == ('external')
        assert (step['inputs']) == ([plan['inputs'][0]] * 2)
        assert (len(step['outputs'])) == (2)
        assert (result.external_requests[0]['port_shapes']) == ({'input:0': [4], 'input:1': [4], 'output:0': [2], 'output:1': [1]})

    def test_sdk_multioutput_scalar_abi_and_json_constants_roundtrip(self):
        incoming = TensorType((4,), "f32")
        outputs = (TensorType((2,), "f32"), TensorType((1,), "i32"))
        scalar_types = ("i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                        "fp16", "bf16", "fp32", "fp64")
        scalar_values = (True, -128, -32768, -(2**31), -(2**63), 255, 65535,
                         2**32 - 1, 2**64 - 1, 1.5, 2, -0.0, 0.1)
        signature = {"x": "*fp32", "a": "*fp32", "b": "*i32", **dict(zip(scalar_types, scalar_types))}
        bindings = {"x": "input:0", "a": "output:0", "b": "output:1", **dict(zip(scalar_types, scalar_values))}
        constants = {"dtype": {"__compiler_dtype__": "float32"}, "flags": [True, False, None],
                     "one": 1.0, "minus_one": -1.0, "integer": -1,
                     "nested": {"key with spaces": ["text", 0.1, -0.0, -(2**63), 2**64-1]}}
        options = {"limit": 0.25, "features": ["a", None, {"enabled": True}]}
        module = Module()
        semantics = module.semantics("sem", inputs=[incoming], results=outputs, shape_rule="fixed")
        implementation = module.implementation("impl", compiler="fixture", source="opaque", entry="kernel",
            signature=signature, source_bindings=bindings, constants=constants, options=options,
            target={"backend": "cuda", "arch": 80})
        graph = module.graph("main", inputs={"x": incoming}, results=outputs)
        results = graph.external(graph.inputs["x"], semantics=semantics, implementation=implementation,
                                 result_types=outputs, id="multi")
        assert isinstance(results, tuple)
        assert ([value.value_id for value in results]) == (['multi.result0', 'multi.result1'])
        graph.return_(*results)
        compiled = self.partition(module)
        plan = self.assert_plan(compiled)
        assert (len(plan['steps'][0]['outputs'])) == (2)
        request = compiled.external_requests[0]
        assert (request['source_bindings']) == (bindings)
        assert (request['constants']) == (constants)
        assert (request['fixed_options']) == (options)
        assert (type(request['constants']['one'])) is (float)
        assert (type(request['constants']['minus_one'])) is (float)
        assert (type(request['constants']['integer'])) is (int)
        assert (math.copysign(1, request['source_bindings']['fp32'])) == (-1)
        assert (math.copysign(1, request['constants']['nested']['key with spaces'][2])) == (-1)
        for compiler in (Compiler(), Compiler(executable=ask_opt_path())):
            for method in (compiler.prepare, compiler.partition):
                restored = method(module).external_requests[0]
                assert (restored) == (request)
                assert (type(restored['constants']['one'])) is (float)
                assert (type(restored['constants']['minus_one'])) is (float)
                assert (type(restored['constants']['integer'])) is (int)
                assert (math.copysign(1, restored['source_bindings']['fp32'])) == (-1)

        for parameter, replacement in (("i1", 1), ("i8", 128), ("u8", -1), ("fp32", True),
                                       ("i64", 2**63), ("u64", -1)):
            bad = Module()
            sem = bad.semantics("sem", inputs=[incoming], results=outputs, shape_rule="fixed")
            impl = bad.implementation("impl", compiler="fixture", source="opaque", entry="kernel",
                signature=signature, source_bindings={**bindings, parameter: replacement},
                target={"backend": "cuda", "arch": 80})
            body = bad.graph("main", inputs={"x": incoming}, results=outputs)
            body.return_(*body.external(body.inputs["x"], semantics=sem, implementation=impl))
            with pytest.raises(CompilerError, match='invalid scalar source binding'):
                self.partition(bad)

    def test_cli_and_binding_have_the_same_contract_and_reject_dynamic_inputs(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        native = self.partition(source, optimize=False)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            output = Path(directory) / "partition.mlir"
            result = subprocess.run([ask_opt_path(), "--partition", "--no-optimize",
                                     "--report-json", str(report), "-o", str(output)],
                                    input=source, capture_output=True, text=True)
            assert (result.returncode) == (0), result.stderr
            assert (_result('partition', output.read_text(), json.loads(report.read_text()))) == (native)
        with pytest.raises(CompilerError, match='concrete'):
            self.partition('module { func.func @main(%x: tensor<?xi32>) -> tensor<?xi32> { return %x : tensor<?xi32> } }')
