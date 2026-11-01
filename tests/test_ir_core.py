"""Independent Fuga SSA ownership, type, effect and region contracts."""
import math
import unittest

from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType


class FugaIRCoreTests(unittest.TestCase):
    def graph(self, **inputs):
        module = Module()
        graph = module.graph("calculate", inputs=inputs)
        return module, graph

    def test_independent_types_and_structured_nodes(self):
        module, graph = self.graph(x=TensorType((2, 3), "fp16"))
        x = graph.inputs["x"]
        y = graph.add(x, x)
        graph.return_(y)
        self.assertEqual(y.type, TensorType((2, 3), "f16"))
        self.assertEqual(type(y).__module__, "fugatensor.ir.core")
        self.assertEqual(y.producer.opcode, "add")
        self.assertEqual(y.producer.operands, (x, x))
        self.assertIs(y.owner, graph.body.blocks[0])
        self.assertIs(y.producer.results[0], y)
        self.assertIn("fuga.add", module.to_text())
        self.assertNotIn("ask.", module.to_text())
        self.assertEqual(module.to_text(), module.to_text())
        with self.assertRaises(TypeError):
            y.producer.attrs["unsafe"] = True

    def test_input_names_cannot_collide_with_generated_ssa_or_region_arguments(self):
        module, graph = self.graph(v0_0=TensorType((3,), "f32"), r0_arg0=ScalarType("bool"))
        x, predicate = graph.inputs.values()
        first = graph.add(x, x)
        result = graph.if_(predicate, [first], lambda block, value: block.mul(value, value),
                           lambda block, value: block.add(value, value))
        graph.return_(result)
        module.verify()
        self.assertEqual(first.name, "%v0_0_")
        region_argument = result.producer.regions[0].blocks[0].arguments[0]
        self.assertEqual(region_argument.name, "%r0_arg0_")

    def test_constants_own_precision_and_preserve_signed_zero(self):
        module, graph = self.graph()
        low = graph.constant(1 + 2**-11, dtype="f16")
        high = graph.constant(1 + 3 * 2**-11, dtype="f16")
        negative = graph.constant(-0.0, dtype="bf16")
        exact_integer = graph.constant(2**100 + 2**76 + 1, dtype="f32")
        graph.return_(low, high, negative, exact_integer)
        module.verify()
        self.assertEqual(low.producer.attrs["value"], 1.0)
        self.assertEqual(high.producer.attrs["value"], 1 + 2**-9)
        self.assertEqual(math.copysign(1, negative.producer.attrs["value"]), -1)
        self.assertEqual(exact_integer.producer.attrs["value"], float(2**100 + 2**77))
        self.assertIn("-0x0.0p+0", module.to_text())

    def test_raw_unrounded_constant_is_rejected(self):
        module, graph = self.graph()
        value = graph.operation("constant", result_types=[ScalarType("f16")],
                                attrs={"value": 1 + 2**-11}).results[0]
        graph.return_(value)
        with self.assertRaisesRegex(ValueError, "already be rounded"):
            module.verify()

    def test_clone_origins_are_structural_and_do_not_change_numeric_attributes(self):
        module, graph = self.graph(x=TensorType((3,), "i32"))
        x = graph.inputs["x"]
        original = graph.add(x, x)
        cloned = graph.operation("add", [original, x], [x.type], attrs={"overflow": "wrap"},
                                 id="calculate.unrolled1", origin_id=original.producer.id).results[0]
        graph.return_(cloned)
        module.verify()
        self.assertEqual(original.origin_id, original.value_id)
        self.assertEqual(cloned.origin_id, original.value_id)
        self.assertNotEqual(cloned.value_id, original.value_id)
        self.assertEqual(x.origin_id, x.value_id)
        self.assertEqual(dict(cloned.producer.attrs), {"overflow": "wrap"})
        self.assertIn('origin="calculate.op0"', module.to_text())
        with self.assertRaises((AttributeError, TypeError)):
            cloned.producer.origin_id = "changed"

    def test_foreign_values_and_implicit_region_captures_are_rejected(self):
        _, graph = self.graph(x=TensorType((3,), "f32"))
        _, another = self.graph(x=TensorType((3,), "f32"))
        x = graph.inputs["x"]
        with self.assertRaisesRegex(ValueError, "preceding values"):
            graph.add(x, another.inputs["x"])
        with self.assertRaisesRegex(ValueError, "captures"):
            graph.for_(0, 2, 1, [x], lambda body, index, carried: body.add(carried, x))
        escaped = []

        def body(builder, index, carried):
            value = builder.add(carried, carried)
            escaped.append(value)
            return value

        output = graph.for_(0, 2, 1, [x], body)
        with self.assertRaisesRegex(ValueError, "preceding values"):
            graph.add(output, escaped[0])

    def test_verifier_rejects_use_before_definition_and_wrong_result_types(self):
        module, graph = self.graph(x=TensorType((3,), "f32"))
        x = graph.inputs["x"]
        first = graph.add(x, x)
        second = graph.mul(first, x)
        graph.return_(second)
        graph.block._operations[:2] = reversed(graph.block._operations[:2])
        with self.assertRaisesRegex(ValueError, "dominate"):
            module.verify()
        module, graph = self.graph(x=TensorType((3,), "f32"))
        x = graph.inputs["x"]
        invalid = graph.operation("cast", [x], [TensorType((4,), "i64")]).results[0]
        graph.return_(invalid)
        with self.assertRaisesRegex(TypeError, "preserve"):
            module.verify()

    def test_branch_and_loop_results_preserve_types(self):
        _, graph = self.graph(x=TensorType((3,), "f32"))
        x = graph.inputs["x"]
        predicate = graph.constant(True, dtype="bool")
        with self.assertRaisesRegex(TypeError, "same result types"):
            graph.if_(predicate, [x], lambda body, value: value,
                      lambda body, value: body.cast(value, dtype="f16"))
        with self.assertRaisesRegex(TypeError, "loop-carried"):
            graph.for_(0, 2, 1, [x], lambda body, index, value: body.cast(value, dtype="f16"))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            graph.for_(0, 2, 0, [x], lambda body, index, value: value)

    def test_dynamic_control_flow_remains_representable_in_fuga(self):
        module, graph = self.graph(condition=ScalarType("bool"), x=TensorType((3,), "f32"))
        condition, x = graph.inputs.values()
        selected = graph.if_(condition, [x], lambda body, value: value,
                             lambda body, value: body.mul(value, value))
        result = graph.while_([condition, selected], lambda body, predicate, value: predicate,
                              lambda body, predicate, value: (predicate, body.add(value, value)))
        graph.return_(*result)
        text = module.to_text()
        self.assertIn("fuga.if", text)
        self.assertIn("fuga.while", text)

    def test_external_snapshot_is_immutable_and_target_binding_is_optional(self):
        source = "# user source\r\ndef kernel(x, y):\r\n    pass\r\n"
        options = {"nested": {"values": [1, 2]}}
        spec = ExternalSpec("toy", source, "kernel", {"x": "*f32", "y": "*f32"},
                            {"x": "input:0", "y": "output:0"}, options=options,
                            target={"backend": "cuda"})
        options["nested"]["values"].append(3)
        self.assertEqual(spec.source, source)
        self.assertEqual(spec.options["nested"]["values"], (1, 2))
        self.assertEqual(dict(spec.target), {"backend": "cuda"})
        self.assertEqual(spec.effects, ("unknown",))
        with self.assertRaises(TypeError):
            spec.options["nested"]["values"] = ()
        module, graph = self.graph(x=TensorType((3,), "f32"))
        value = graph.external(graph.inputs["x"], spec=spec)
        graph.return_(value)
        module.verify()
        self.assertEqual(value.producer.effects, ("unknown",))

    def test_control_flow_carries_nested_effects_and_invalid_bindings_fail(self):
        spec = ExternalSpec("toy", "user code", "kernel", {"x": "ptr", "y": "ptr"},
                            {"x": "input:0", "y": "output:0"}, effects=("write",))
        module, graph = self.graph(x=TensorType((3,), "f32"))
        x = graph.inputs["x"]
        result = graph.for_(0, 2, 1, [x], lambda body, index, value: body.external(value, spec=spec))
        graph.return_(result)
        module.verify()
        self.assertEqual(result.producer.effects, ("write",))
        invalid = ExternalSpec("toy", "code", "kernel", {"x": "ptr", "y": "ptr"},
                               {"x": "input:0", "y": "output:1"}, effects=())
        module, graph = self.graph(x=TensorType((3,), "f32"))
        graph.return_(graph.external(graph.inputs["x"], spec=invalid))
        with self.assertRaisesRegex(ValueError, "every input and output"):
            module.verify()

    def test_completed_blocks_and_incomplete_functions_are_rejected(self):
        module, graph = self.graph(x=TensorType((3,), "f32"))
        with self.assertRaisesRegex(ValueError, "final return"):
            module.verify()
        graph.return_(graph.inputs["x"])
        with self.assertRaisesRegex(ValueError, "terminator"):
            graph.constant(1)

    def test_unknown_dimensions_are_types_without_device_placement(self):
        type_ = TensorType((None, 3), "bf16")
        self.assertEqual(str(type_), "tensor<?x3xbf16>")
        self.assertFalse(hasattr(type_, "device"))
        for shape in ((True,), (-1,), (2**63,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorType(shape, "f32")
        with self.assertRaises(ValueError):
            TensorType((2**62,), "f32")


if __name__ == "__main__":
    unittest.main()
