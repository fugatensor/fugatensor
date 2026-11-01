"""Primitive typing and executable VJP rules are the shared transform entry."""
from dataclasses import replace
from types import MappingProxyType
import unittest
from unittest.mock import patch

from fugatensor.ir import Module, ScalarType, TensorType, vjp
from fugatensor.ir import primitives


class PrimitiveRuleTests(unittest.TestCase):
    def test_abstract_evaluation_carries_shape_and_dtype_without_values(self):
        type_ = TensorType((None, 4), "f16")
        self.assertEqual(primitives.infer_result_types("add", (type_, type_)), (type_,))
        self.assertEqual(primitives.infer_result_types("cast", (type_,), dtype="f32"),
                         (TensorType((None, 4), "f32"),))
        self.assertEqual(primitives.infer_result_types("splat", (ScalarType("bf16"),), {"shape": (2, 3)}),
                         (TensorType((2, 3), "bf16"),))
        with self.assertRaisesRegex(TypeError, "equal operand"):
            primitives.infer_result_types("mul", (type_, TensorType((4,), "f16")))
        with self.assertRaisesRegex(ValueError, "numerical attributes"):
            primitives.infer_result_types("add", (ScalarType("i32"), ScalarType("i32")))
        self.assertEqual(primitives.infer_result_types("add", (ScalarType("i32"), ScalarType("i32")),
                                                       {"overflow": "wrap"}), (ScalarType("i32"),))

    def test_builder_verifier_and_reverse_transform_call_registered_rules(self):
        original = primitives.PRIMITIVE_RULES["mul"]
        typed, reversed_ = [], []

        def abstract_eval(types, attrs):
            typed.append(types)
            return original.abstract_eval(types, attrs)

        def reverse(builder, operation, primals, cotangent, active):
            reversed_.append(operation)
            return original.vjp(builder, operation, primals, cotangent, active)

        rules = dict(primitives.PRIMITIVE_RULES)
        rules["mul"] = replace(original, abstract_eval=abstract_eval, vjp=reverse)
        with patch.object(primitives, "PRIMITIVE_RULES", MappingProxyType(rules)):
            module = Module()
            graph = module.graph("square", inputs={"x": TensorType((3,), "f32")})
            x = graph.inputs["x"]
            graph.return_(graph.mul(x, x))
            self.assertTrue(typed)  # The builder consulted abstract evaluation.
            count = len(typed)
            module.verify()
            self.assertGreater(len(typed), count)  # The verifier uses the same rule.
            first = vjp(module)
            self.assertEqual(len(reversed_), 1)
            second = vjp(first, argnums=0)
            self.assertGreater(len(reversed_), 1)
            self.assertEqual(second.functions[0].outputs[0].type, x.type)

    def test_reverse_rule_results_are_checked_before_becoming_graph_outputs(self):
        original = primitives.PRIMITIVE_RULES["add"]

        def malformed(builder, operation, primals, cotangent, active):
            return (builder.cast(cotangent, dtype="f16"), cotangent)

        module = Module()
        graph = module.graph("sum", inputs={"x": TensorType((3,), "f32")})
        x = graph.inputs["x"]
        graph.return_(graph.add(x, x))
        rules = dict(primitives.PRIMITIVE_RULES)
        rules["add"] = replace(original, vjp=malformed)
        with patch.object(primitives, "PRIMITIVE_RULES", MappingProxyType(rules)):
            with self.assertRaisesRegex(TypeError, "preserve operand shape and dtype"):
                vjp(module)

    def test_textual_cast_cannot_override_its_result_type_with_an_attribute(self):
        module = Module()
        graph = module.graph("bad_cast", inputs={"x": TensorType((3,), "f32")})
        result = graph.operation("cast", (graph.inputs["x"],), (TensorType((3,), "f16"),),
                                 attrs={"dtype": "i64"}).results[0]
        graph.return_(result)
        with self.assertRaisesRegex(TypeError, "cannot have attributes"):
            module.verify()


if __name__ == "__main__":
    unittest.main()
