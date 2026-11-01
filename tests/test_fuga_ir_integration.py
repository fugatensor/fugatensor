"""The independent frontend IR and its transforms still execute through ASK."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ask import Compiler, CompilerEnvironment, ExecutionSession
import fugatensor as ft
from fugatensor.backends.host import HostBackend, HostStorage
from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, vjp
from fugatensor.lowering import lower_to_ask
from fugatensor.tracing import TraceContext


class IRBoundaryTests(unittest.TestCase):
    def test_function_local_ids_remain_distinct_at_the_ask_boundary(self):
        module = Module()
        tensor_type = TensorType((3,), 'f32')
        for name in ('left', 'right'):
            graph = module.graph(name, inputs={'x': tensor_type})
            graph.return_(graph.add(graph.inputs['x'], graph.inputs['x'], id='shared'))
        module.verify()
        source_map = {}
        translated = lower_to_ask(module, source_map=source_map)
        rows = [row for row in source_map['values'] if row['origin_value_id'] == 'shared.result0']
        self.assertEqual({row['function'] for row in rows}, {'left', 'right'})
        self.assertEqual(len({row['ask_value_id'] for row in rows}), 2)
        for row in rows:
            self.assertIn(row['ask_value_id'], translated.to_mlir())
        # The native verifier, not just the Python builder, accepts both graphs.
        Compiler(executable=os.environ.get('ASK_OPT')).prepare(translated)

    def test_unrolled_operations_keep_distinct_ask_ids_and_their_fuga_origin(self):
        module = Module()
        t = TensorType((3,), "f32")
        graph = module.graph("mapped", inputs={"x": t}, results=[t])
        value = graph.for_(0, 3, 1, [graph.inputs["x"]],
                           lambda body, index, carried: body.add(carried, carried, id="loop_add"))
        graph.return_(value)
        source_map = {}
        translated = lower_to_ask(module, source_map=source_map)
        self.assertEqual(source_map["schema_version"], 1)
        rows = [row for row in source_map["values"] if row["origin_value_id"] == "loop_add.result0"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["ask_value_id"] for row in rows}), 3)
        for row in rows:
            self.assertIn(row["ask_value_id"], translated.to_mlir())
        # An unsuccessful lowering must not leave a half-valid map behind.
        dynamic = Module()
        invalid = dynamic.graph("unknown_shape", inputs={"x": TensorType((None,), "f32")})
        invalid.return_(invalid.inputs["x"])
        sentinel = {"old": "keep"}
        with self.assertRaisesRegex(ValueError, "concrete"):
            lower_to_ask(dynamic, source_map=sentinel)
        self.assertEqual(sentinel, {"old": "keep"})

    def test_explicit_external_result_shape_maps_to_ask_fixed_semantics(self):
        module = Module()
        incoming, outgoing = TensorType((3,), "f32"), TensorType((4,), "f32")
        graph = module.graph("explicit_external", inputs={"x": incoming}, results=[outgoing])
        # This fixture only checks the bridge protocol. No external compiler or
        # kernel is invoked, and no execution capability is inferred from it.
        spec = ExternalSpec(compiler="metadata_only", source="opaque metadata-only source",
                            entry="expand", signature={"inp": "*fp32", "out": "*fp32"},
                            source_bindings={"inp": "input:0", "out": "output:0"},
                            target={"backend": "host"}, shape_rule="explicit", effects=())
        graph.return_(graph.external(graph.inputs["x"], spec=spec, result_types=[outgoing]))
        translated = lower_to_ask(module, target={"backend": "host"})
        self.assertIn('shape_rule = "fixed"', translated.to_mlir())
        report = Compiler(executable=os.environ.get("ASK_OPT")).prepare(translated).report
        self.assertEqual(report["external_requests"][0]["source"], spec.source)
        self.assertIn("tensor<4xf32>", translated.to_mlir())

    def test_same_expression_has_identical_cpu_and_cuda_fuga_ir(self):
        texts = []
        for device in ("cpu", "cuda"):
            context = TraceContext([((3,), "f32"), ((3,), "f32")],
                                   engine=SimpleNamespace(device=device))
            x, y = context.inputs
            module = context.finish((x + y) * 2)
            texts.append(module.to_text())
            self.assertFalse(hasattr(x.value.type, "device"))
            self.assertNotIn("cuda", texts[-1])
            self.assertNotIn("cpu", texts[-1])
            self.assertNotIn("ask.", texts[-1])
        self.assertEqual(texts[0], texts[1])

    def test_dynamic_branch_is_not_fixed_from_example_data(self):
        module = Module()
        t = TensorType((3,), "f32")
        graph = module.graph("dynamic", inputs={"predicate": ScalarType("bool"), "x": t}, results=[t])
        selected = graph.if_(graph.inputs["predicate"], [graph.inputs["x"]],
                             lambda branch, x: branch.mul(x, x),
                             lambda branch, x: branch.add(x, x))
        graph.return_(selected)
        module.verify()
        self.assertIn("fuga.if", module.to_text())
        with self.assertRaisesRegex((ValueError, NotImplementedError), "dynamic|data-dependent|static"):
            lower_to_ask(module)


class GraphPrograms:
    """One set of graph definitions executes on both devices."""

    def test_static_branch_and_loop_execute_after_specialization(self):
        module = Module()
        t = TensorType((3,), "f32")
        graph = module.graph("regions", inputs={"x": t}, results=[t])
        selected = graph.if_(graph.constant(True, dtype="bool"), [graph.inputs["x"]],
                             lambda branch, x: branch.mul(x, x),
                             lambda branch, x: branch.add(x, x))
        result = graph.for_(0, 3, 1, [selected], lambda body, index, carried: body.add(carried, carried))
        graph.return_(result)
        original = module.to_text()
        self.assertIn("fuga.if", original)
        self.assertIn("fuga.for", original)
        ask_module = lower_to_ask(module)
        self.assertEqual(module.to_text(), original)
        self.assertNotIn("fuga.", ask_module.to_mlir())
        self.assertEqual(self.execute(ask_module, [[-2.0, 0.5, 3.0]]), [[32.0, 2.0, 72.0]])

    def test_reverse_graph_uses_explicit_seeds_and_adds_all_contributions(self):
        module = Module()
        t = TensorType((3,), "f32")
        graph = module.graph("primal", inputs={"x": t, "y": t}, results=[t, t])
        x, y = graph.inputs["x"], graph.inputs["y"]
        graph.return_(graph.mul(x, y), graph.add(x, x))
        original = module.to_text()
        backward = vjp(module, argnums=(0, 1))
        backward.verify()
        self.assertEqual(module.to_text(), original)
        self.assertEqual(len(backward.functions[0].inputs), 4)
        actual = self.execute(lower_to_ask(backward),
                              [[-2.0, 0.5, 3.0], [4.0, -2.0, 5.0],
                               [1.0, 2.0, 0.5], [3.0, -1.0, 2.0]])
        self.assertEqual(actual, [[10.0, -6.0, 6.5], [-2.0, 1.0, 1.5]])

    def test_jit_keeps_both_ir_artifacts_and_reuses_them(self):
        options = {"ask_opt": os.environ.get("ASK_OPT"), "cache_dir": Path(self.directory.name) / "jit"}
        if self.device == "cuda":
            options.update(llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"])
        with ft.Engine(device=self.device, **options):
            @ft.jit
            def arithmetic(x, y):
                return (x + y) * 2

            x, y = ft.tensor([1, 2, 3]), ft.tensor([10, 20, 30])
            self.assertEqual(arithmetic(x, y).tolist(), [22.0, 44.0, 66.0])
            self.assertIn("fuga.add", arithmetic.last_fuga_ir)
            self.assertIn("fuga.mul", arithmetic.last_fuga_ir)
            self.assertIn('"ask.add"', arithmetic.last_ask_ir)
            self.assertEqual(arithmetic.last_ir, arithmetic.last_ask_ir)
            artifacts = Path(arithmetic.last_artifact_dir)
            self.assertEqual((artifacts / "fuga.ir").read_text(), arithmetic.last_fuga_ir)
            self.assertEqual((artifacts / "frontend.mlir").read_text(), arithmetic.last_ask_ir)
            mapping = json.loads((artifacts / "fuga-to-ask.json").read_text())
            self.assertEqual(mapping["schema_version"], 1)
            self.assertTrue(mapping["values"])
            for row in mapping["values"]:
                self.assertIn(row["ask_value_id"], arithmetic.last_ask_ir)
            source = arithmetic.last_fuga_ir
            self.assertEqual(arithmetic(ft.tensor([4, 5, 6]), y).tolist(), [28.0, 50.0, 72.0])
            self.assertEqual(arithmetic.last_fuga_ir, source)
            self.assertEqual(arithmetic.last_artifact_dir, artifacts)
            self.assertEqual(arithmetic.cache_info()["compilations"], 1)
            self.assertEqual(arithmetic.cache_info()["cache_hits"], 1)
            arithmetic.clear_cache()
            self.assertIsNone(arithmetic.last_fuga_ir)
            self.assertIsNone(arithmetic.last_ask_ir)


class HostIRExecutionTests(GraphPrograms, unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-ir-host-")
        self.addCleanup(self.directory.cleanup)
        self.compiler = Compiler(executable=os.environ.get("ASK_OPT"))

    def execute(self, module, values):
        executable = HostBackend(self.compiler, self.directory.name).compile(module)
        inputs = [HostStorage.from_flat(value, shape=port.shape, dtype=port.dtype)
                  for port, value in zip(executable.inputs, values)]
        return [output.to_flat() for output in executable.run(inputs)]


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1", "enable real CUDA IR execution explicitly")
class CUDAIRExecutionTests(GraphPrograms, unittest.TestCase):
    device = "cuda"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fuga-ir-cuda-")
        self.addCleanup(self.directory.cleanup)
        self.compiler = Compiler(executable=os.environ.get("ASK_OPT"))
        self.session = ExecutionSession()
        self.addCleanup(self.session.close)

    def execute(self, module, values):
        major, minor = self.session.capability
        directory = Path(self.directory.name)
        kernel = self.compiler.compile_cuda(module, target={"backend": "cuda", "arch": major * 10 + minor},
            llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"],
            cache_dir=directory / "cache", output_dir=directory / "output")
        executable = kernel.prepare(self.session)
        inputs = [self.session.from_host(value, shape=port["shape"], dtype=port["element_type"])
                  for port, value in zip(executable.inputs, values)]
        outputs = [self.session.tensor(port["shape"], port["element_type"]) for port in executable.outputs]
        executable.submit_many(inputs, outputs).wait()
        return [output.to_host() for output in outputs]


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_TRITON") == "1", "enable user Triton fusion explicitly")
class MixedIRExecutionTests(unittest.TestCase):
    def test_user_triton_remains_linked_with_surrounding_ask_gpu_computation(self):
        source = Path(__file__).parent / "fixtures" / "user_scale.py"
        scale = ft.triton.kernel(source, entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,))
        with tempfile.TemporaryDirectory(prefix="fuga-ir-external-") as directory, \
             ft.Engine(device="cuda", ask_opt=os.environ.get("ASK_OPT"), cache_dir=directory,
                compilers={"triton": CompilerEnvironment(python=os.environ["FUGATENSOR_TRITON_PYTHON"],
                    worker=os.environ["FUGATENSOR_TRITON_WORKER"])},
                llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"]):
            @ft.jit
            def mixed(x):
                return scale((x + 1) * 2) + 3

            self.assertEqual(mixed(ft.tensor([1, 2, 3])).tolist(), [11.0, 15.0, 19.0])
            for opcode in ("fuga.add", "fuga.mul", "fuga.external"):
                self.assertIn(opcode, mixed.last_fuga_ir)
            self.assertIn('"ask.external"', mixed.last_ask_ir)
            self.assertEqual(mixed.last_report["phase"], "fused_llvm")
            self.assertEqual(mixed.last_report["fusion"]["compiler"], "triton")
            self.assertEqual(mixed.last_report["external_requests"][0]["source"], source.read_text())
            self.assertTrue((Path(mixed.last_artifact_dir) / "fused.ll").is_file())
            self.assertTrue(list(Path(mixed.last_artifact_dir).rglob("*.cubin")))


if __name__ == "__main__":
    unittest.main()
