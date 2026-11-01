"""Forward/reverse transforms compose as Fuga graphs and execute through ASK."""
import os
import tempfile
import unittest

from ask import Compiler, CompilerError
from fugatensor.backends.host import HostBackend, HostStorage
from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, jvp, vjp
from fugatensor.ir.core import CallSpec
from fugatensor.ir.differentiate import DifferentiationError
from fugatensor.lowering import lower_to_ask


def cubic(dtype="f32"):
    module = Module()
    graph = module.graph("cubic", inputs={"x": TensorType((3,), dtype)})
    x = graph.inputs["x"]
    graph.return_(graph.mul(graph.mul(x, x), x))
    return module


class JVPContractTests(unittest.TestCase):
    def test_float_casts_scalar_splat_and_source_preservation(self):
        module = Module()
        graph = module.graph("broadcast", inputs={"x": ScalarType("f16")})
        graph.return_(graph.splat(graph.cast(graph.inputs["x"], dtype="f32"), (3,)))
        before = module.to_text()
        transformed = jvp(module)
        self.assertEqual(module.to_text(), before)
        self.assertEqual([value.type for value in transformed.functions[0].outputs], [TensorType((3,), "f32")] * 2)
        self.assertEqual(list(transformed.functions[0].inputs), ["x", "tangent0"])
        self.assertEqual(sum(op.opcode == "splat" for op in transformed.functions[0].operations), 2)

    def test_active_opaque_and_custom_calls_need_their_own_jvp(self):
        module = Module()
        graph = module.graph("opaque", inputs={"x": TensorType((3,), "f32"),
                                               "unused": TensorType((3,), "f32")})
        spec = ExternalSpec(compiler="opaque", source="original DSL", entry="kernel",
                            signature={"x": "ptr", "y": "ptr"},
                            source_bindings={"x": "input:0", "y": "output:0"}, effects=())
        graph.return_(graph.external(graph.inputs["x"], spec=spec))
        with self.assertRaisesRegex(DifferentiationError, "external.*explicit JVP"):
            jvp(module)
        inactive = jvp(module, argnums=1)
        self.assertIn("original DSL", inactive.to_text())
        tangent = inactive.functions[0].outputs[1]
        self.assertEqual(tangent.producer.opcode, "splat")
        self.assertEqual(tangent.producer.operands[0].producer.attrs["value"], 0)
        module = Module()
        graph = module.graph("custom", inputs={"x": TensorType((3,), "f32")})
        call = CallSpec(name="custom_cubic", forward=cubic(), output_count=1)
        graph.return_(graph.call(graph.inputs["x"], spec=call))
        with self.assertRaisesRegex(DifferentiationError, "call.*explicit JVP"):
            jvp(module)

    def test_integer_derivative_paths_and_invalid_selection_are_rejected(self):
        module = Module()
        graph = module.graph("integer_cast", inputs={"x": TensorType((3,), "f32")})
        graph.return_(graph.cast(graph.cast(graph.inputs["x"], dtype="i32"), dtype="f32"))
        with self.assertRaisesRegex(DifferentiationError, "integer/bool"):
            jvp(module)
        for selected in (True, -1, (), (0, 0), [1]):
            with self.subTest(argnums=selected), self.assertRaises((TypeError, ValueError)):
                jvp(cubic(), argnums=selected)


class JVPExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler = Compiler(os.environ.get("ASK_OPT"))
            if not cls.compiler.capabilities().get("host_codegen"):
                raise unittest.SkipTest("ASK host code generation is disabled")
        except CompilerError as error:
            raise unittest.SkipTest(f"ASK native compiler is required: {error}")

    def execute(self, module, arguments):
        with tempfile.TemporaryDirectory(prefix="fuga-jvp-") as directory:
            executable = HostBackend(self.compiler, directory).compile(lower_to_ask(module))
            tensors = [HostStorage.from_flat(values, shape=value.type.shape, dtype=value.type.dtype)
                       for value, values in zip(module.functions[0].inputs.values(), arguments)]
            return [value.to_flat() for value in executable.run(tensors)]

    def test_cubic_jvp_and_both_orders_of_forward_reverse_composition(self):
        module = cubic()
        forward = jvp(module)
        x, tangent = [2, -1, 0.5], [0.5, 2, -1]
        self.assertEqual(self.execute(forward, [x, tangent]), [[8, -1, 0.125], [6, 6, -0.75]])
        reverse_forward = vjp(forward, argnums=0)
        forward_reverse = jvp(vjp(module), argnums=0)
        reverse_result = self.execute(reverse_forward, [x, tangent, [0] * 3, [1] * 3])[0]
        forward_result = self.execute(forward_reverse, [x, [1] * 3, tangent])[1]
        self.assertEqual(reverse_result, [6, -12, -3])
        self.assertEqual(forward_result, reverse_result)


if __name__ == "__main__":
    unittest.main()
