"""Static control and reverse-mode transformations over Fuga's own SSA graph."""
import unittest

from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, specialize, vjp
from fugatensor.ir.differentiate import DifferentiationError
from fugatensor.ir.transforms import StaticControlFlowError


def tensor_graph(name="calculate", dtype="f32", names=("x",)):
    module = Module()
    type_ = TensorType((3,), dtype)
    return module, module.graph(name, inputs={name: type_ for name in names})


def opaque_spec(*, effects=()):
    return ExternalSpec(compiler="example", source="opaque original DSL\r\n",
                        entry="kernel", signature={"x": "ptr", "y": "ptr"},
                        source_bindings={"x": "input:0", "y": "output:0"},
                        effects=effects)


class StaticControlTests(unittest.TestCase):
    def test_static_branch_selects_only_its_region_without_mutating_source(self):
        module, graph = tensor_graph()
        x = graph.inputs["x"]

        def true_branch(block, value):
            return block.add(value, value)

        def false_branch(block, value):
            return block.while_((value,),
                                lambda condition, _: condition.constant(False, dtype="bool"),
                                lambda body, carry: carry)

        value = graph.if_(graph.constant(True, dtype="bool"), (x,), true_branch, false_branch)
        graph.return_(value)
        before = module.to_text()
        flat = specialize(module)
        self.assertEqual(module.to_text(), before)
        self.assertIn("fuga.if", before)
        self.assertIn("fuga.while", before)
        self.assertNotIn("fuga.if", flat.to_text())
        self.assertNotIn("fuga.while", flat.to_text())
        self.assertIn("fuga.add", flat.to_text())
        self.assertEqual(flat.to_text(), specialize(module).to_text())

    def test_zero_iterations_return_initial_values_and_negative_steps_are_exact(self):
        module, graph = tensor_graph()
        initial = graph.inputs["x"]
        zero = graph.for_(0, 0, 1, (initial,), lambda block, index, value: block.mul(value, value))
        graph.return_(zero)
        flat = specialize(module).functions[0]
        self.assertIs(flat.outputs[0], flat.inputs["x"])
        self.assertFalse(any(op.opcode == "mul" for op in flat.operations))

        module, graph = tensor_graph()
        value = graph.for_(5, -1, -2, (graph.inputs["x"],),
                           lambda block, index, carry: block.add(carry, carry))
        graph.return_(value)
        flat = specialize(module).functions[0]
        self.assertEqual([op.attrs["value"] for op in flat.operations if op.opcode == "constant"],
                         [5, 3, 1])
        self.assertEqual(sum(op.opcode == "add" for op in flat.operations), 3)

    def test_nested_expansion_has_a_total_budget_and_never_truncates(self):
        module, graph = tensor_graph()

        def outer(block, index, carry):
            return block.for_(0, 2, 1, (carry,), lambda inner, j, value: inner.add(value, value))

        graph.return_(graph.for_(0, 2, 1, (graph.inputs["x"],), outer))
        before = module.to_text()
        with self.assertRaisesRegex(StaticControlFlowError, "max_unroll=5"):
            specialize(module, max_unroll=5)
        self.assertEqual(module.to_text(), before)
        flat = specialize(module, max_unroll=6).functions[0]
        self.assertEqual(sum(op.opcode == "add" for op in flat.operations), 4)
        for limit in (True, -1, 2.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                specialize(module, max_unroll=limit)

    def test_static_integer_predicate_uses_wrapping_and_literal_float_rounding(self):
        for dtype, constant in (("i32", 0), ("f16", 2**-100)):
            module, graph = tensor_graph()
            zero = graph.constant(constant, dtype=dtype)
            predicate = graph.cast(zero, dtype="bool")
            x = graph.inputs["x"]
            graph.return_(graph.if_(predicate, (x,), lambda b, v: b.mul(v, v), lambda b, v: v))
            flat = specialize(module).functions[0]
            self.assertIs(flat.outputs[0], flat.inputs["x"])

        # The low-level API cannot bypass typed constant canonicalization.
        # Reject the malformed literal before it can choose the wrong branch.
        module, graph = tensor_graph()
        zero = graph.operation("constant", result_types=(ScalarType("f16"),),
                               attrs={"value": 2**-100}).results[0]
        x = graph.inputs["x"]
        graph.return_(graph.if_(graph.cast(zero, dtype="bool"), (x,),
                                lambda b, v: b.mul(v, v), lambda b, v: v))
        with self.assertRaisesRegex(ValueError, "already be rounded"):
            specialize(module)

    def test_dynamic_branch_and_while_are_not_frozen_from_runtime_data(self):
        module = Module()
        graph = module.graph("branch", inputs={"predicate": ScalarType("bool"), "x": TensorType((3,), "f32")})
        graph.return_(graph.if_(graph.inputs["predicate"], (graph.inputs["x"],),
                                lambda b, x: b.mul(x, x), lambda b, x: x))
        module.verify()
        before = module.to_text()
        with self.assertRaisesRegex(StaticControlFlowError, "data-dependent fuga.if"):
            specialize(module)
        self.assertEqual(module.to_text(), before)

        module, graph = tensor_graph()
        graph.return_(graph.while_((graph.inputs["x"],),
                                   lambda b, x: b.constant(False, dtype="bool"), lambda b, x: x))
        module.verify()
        with self.assertRaisesRegex(StaticControlFlowError, "fuga.while"):
            specialize(module)


class ReverseModeTests(unittest.TestCase):
    def test_explicit_seeds_accumulate_for_duplicate_outputs(self):
        module, graph = tensor_graph(names=("x", "seed0"))
        graph.return_(graph.inputs["x"], graph.inputs["x"])
        before = module.to_text()
        derivative = vjp(module, argnums=(0,)).functions[0]
        self.assertEqual(module.to_text(), before)
        self.assertEqual(tuple(derivative.inputs), ("x", "seed0", "_seed0", "seed1"))
        gradient = derivative.outputs[0]
        self.assertEqual(gradient.producer.opcode, "add")
        self.assertEqual(gradient.producer.operands,
                         (derivative.inputs["_seed0"], derivative.inputs["seed1"]))

    def test_disconnected_input_receives_a_typed_zero(self):
        for dtype in ("f16", "bf16", "f32"):
            with self.subTest(dtype=dtype):
                module, graph = tensor_graph(dtype=dtype, names=("x", "y"))
                graph.return_(graph.inputs["x"])
                derivative = vjp(module, argnums=1).functions[0]
                gradient = derivative.outputs[0]
                self.assertEqual(gradient.type, TensorType((3,), dtype))
                self.assertEqual(gradient.producer.opcode, "splat")
                self.assertEqual(gradient.producer.operands[0].producer.attrs["value"], 0.0)

    def test_identity_and_cast_rules_keep_input_types(self):
        module, graph = tensor_graph(dtype="f16")
        graph.return_(graph.cast(graph.inputs["x"], dtype="f32"))
        derivative = vjp(module).functions[0]
        self.assertEqual(derivative.inputs["seed0"].type, TensorType((3,), "f32"))
        self.assertEqual(derivative.outputs[0].type, TensorType((3,), "f16"))
        self.assertEqual(derivative.outputs[0].producer.opcode, "cast")

        module, graph = tensor_graph()
        graph.return_(graph.inputs["x"])
        derivative = vjp(module).functions[0]
        self.assertIs(derivative.outputs[0], derivative.inputs["seed0"])

    def test_constants_and_static_regions_can_be_differentiated(self):
        module, graph = tensor_graph()

        def body(block, index, carry):
            two = block.splat(block.constant(2.0, dtype="f32"), (3,))
            return block.mul(carry, two)

        graph.return_(graph.for_(0, 3, 1, (graph.inputs["x"],), body))
        derivative = vjp(module)
        derivative.verify()
        self.assertIn("fuga.for", module.to_text())
        self.assertNotIn("fuga.for", derivative.to_text())
        self.assertNotIn("device", derivative.to_text())

    def test_live_external_requires_vjp_but_dead_external_is_removed(self):
        module, graph = tensor_graph()
        external = graph.external(graph.inputs["x"], spec=opaque_spec())
        graph.return_(external)
        with self.assertRaisesRegex(DifferentiationError, "external.*explicit VJP"):
            vjp(module)

        module, graph = tensor_graph()
        graph.external(graph.inputs["x"], spec=opaque_spec())
        graph.return_(graph.inputs["x"])
        derivative = vjp(module)
        self.assertNotIn("fuga.external", derivative.to_text())

    def test_independent_external_is_a_coefficient_not_an_identity_derivative(self):
        module, graph = tensor_graph(names=("x", "y"))
        spec = opaque_spec()
        coefficient = graph.external(graph.inputs["y"], spec=spec)
        graph.return_(graph.mul(graph.inputs["x"], coefficient))
        derivative = vjp(module, argnums=(0,)).functions[0]
        external = next(op for op in derivative.operations if op.opcode == "external")
        self.assertIs(external.attrs["spec"], spec)
        self.assertEqual(external.attrs["spec"].source, "opaque original DSL\r\n")
        with self.assertRaisesRegex(DifferentiationError, "external.*explicit VJP"):
            vjp(module, argnums=(1,))

    def test_effects_integer_paths_and_scalar_reductions_are_explicit_errors(self):
        module, graph = tensor_graph()
        graph.external(graph.inputs["x"], spec=opaque_spec(effects=("unknown",)))
        graph.return_(graph.inputs["x"])
        with self.assertRaisesRegex(DifferentiationError, "effect-free"):
            vjp(module)

        module, graph = tensor_graph()
        graph.return_(graph.cast(graph.cast(graph.inputs["x"], dtype="i32"), dtype="f32"))
        with self.assertRaisesRegex(DifferentiationError, "integer/bool casts"):
            vjp(module)

        module = Module()
        graph = module.graph("broadcast_scalar", inputs={"x": ScalarType("f32")})
        graph.return_(graph.splat(graph.inputs["x"], (3,)))
        with self.assertRaisesRegex(DifferentiationError, "sum reduction"):
            vjp(module)

    def test_argument_selection_rejects_ambiguous_or_nondifferentiable_requests(self):
        module = Module()
        graph = module.graph("mixed", inputs={"x": TensorType((3,), "f32"), "y": TensorType((3,), "i32")})
        graph.return_(graph.inputs["x"])
        derivative = vjp(module).functions[0]
        self.assertEqual(len(derivative.outputs), 1)
        with self.assertRaisesRegex(DifferentiationError, "not differentiable"):
            vjp(module, argnums=1)
        for selected in (True, -1, 2, (0, 0), (), [False]):
            with self.subTest(selected=selected), self.assertRaises((ValueError, TypeError)):
                vjp(module, argnums=selected)


if __name__ == "__main__":
    unittest.main()
