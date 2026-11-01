"""Check custom-function boundaries without compiling or evaluating tensors."""
import inspect
import unittest

from fugatensor._state import active_trace
from fugatensor.function import fun
from fugatensor.ir import Module, TensorType, specialize, vjp
from fugatensor.ir.core import CallSpec
from fugatensor.ir.differentiate import DifferentiationError
from fugatensor.jit import jit
from fugatensor.tracing import TraceContext


def trace(function, specs=(((3,), "f32"),), *static):
    context = TraceContext(specs)
    token = active_trace.set(context)
    try:
        context.finish(function(*context.inputs, *static))
    finally:
        active_trace.reset(token)
    return context


class CustomSquare(fun):
    @jit
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @jit
    def backward(self, ctx, dy):
        x, = ctx.saved_tensors
        return (dy * x * 2,)


class FunContractTests(unittest.TestCase):
    def test_both_computational_methods_require_jit(self):
        with self.assertRaisesRegex(TypeError, "forward.*@ft.jit"):
            class MissingForwardDecorator(fun):
                def forward(self, ctx, x):
                    return x

        with self.assertRaisesRegex(TypeError, "backward.*@ft.jit"):
            class MissingBackwardDecorator(fun):
                @jit
                def forward(self, ctx, x):
                    return x

                def backward(self, ctx, dy):
                    return dy

        with self.assertRaisesRegex(TypeError, "backward.*signature"):
            class InvalidBackwardSignature(fun):
                @jit
                def forward(self, ctx, x):
                    return x

                @jit
                def backward(self, ctx, first, second):
                    return first

    def test_signature_keeps_static_defaults_and_keyword_only_parameters(self):
        class Scale(fun):
            @jit
            def forward(self, ctx, x, factor=2, *, offset=1):
                return x * factor + offset

        scale = Scale()
        self.assertEqual(str(inspect.signature(scale)), "(x, factor=2, *, offset=1)")
        context = trace(lambda x: scale(x, 3, offset=4))
        values = [op.attrs["value"] for op in specialize(context.module).functions[0].operations
                  if op.opcode == "constant"]
        self.assertEqual(values, [3.0, 4.0])

    def test_saved_residuals_are_hidden_ssa_results_until_expansion(self):
        square = CustomSquare()
        context = trace(square)
        call, returned = context.graph.operations
        spec = call.attrs["spec"]
        self.assertEqual(call.opcode, "call")
        self.assertEqual((spec.output_count, spec.saved_count), (1, 1))
        self.assertEqual(len(call.results), 2)
        self.assertEqual(returned.operands, (call.results[0],))
        self.assertEqual(spec.forward.functions[0].outputs[1], spec.forward.functions[0].inputs["arg0"])
        self.assertIs(specialize(context.module, expand_calls=False).functions[0].operations[0].attrs["spec"], spec)
        flat = specialize(context.module)
        self.assertNotIn("fuga.call", flat.to_text())
        self.assertEqual([op.opcode for op in flat.functions[0].operations], ["mul", "return"])
        derivative = vjp(context.module)
        self.assertIn("fuga.call", derivative.to_text())
        specialize(derivative).verify()

    def test_custom_rule_is_selected_instead_of_differentiating_forward(self):
        class Surrogate(fun):
            @jit
            def forward(self, ctx, x):
                return x * x

            @jit
            def backward(self, ctx, dy):
                return (dy * 7,)

        derivative = vjp(trace(Surrogate()).module).functions[0]
        self.assertNotIn("call", [op.opcode for op in derivative.operations])
        self.assertEqual(derivative.outputs[0].producer.opcode, "mul")
        self.assertEqual(derivative.outputs[0].producer.operands[0], derivative.inputs["seed0"])
        self.assertEqual([op.attrs["value"] for op in derivative.operations if op.opcode == "constant"], [7.0])

    def test_nonfloating_and_static_arguments_do_not_take_gradient_slots(self):
        class MixedInputs(fun):
            @jit
            def forward(self, ctx, x, indices, mask, factor):
                return x * factor

            @jit
            def backward(self, ctx, dy):
                return (dy * 3,)

        context = trace(MixedInputs(), (((3,), "f32"), ((3,), "i64"), ((3,), "bool")), 3)
        derivative = vjp(context.module, argnums=0).functions[0]
        self.assertEqual(len(derivative.outputs), 1)
        self.assertEqual(derivative.outputs[0].type, TensorType((3,), "f32"))

    def test_multiple_outputs_receive_all_cotangents_in_order(self):
        class Pair(fun):
            @jit
            def forward(self, ctx, x):
                return x, x * 2

            @jit
            def backward(self, ctx, dy):
                first, second = dy
                return (first + second * 2,)

        derivative = vjp(trace(Pair()).module).functions[0]
        first, product = derivative.outputs[0].producer.operands
        self.assertIs(first, derivative.inputs["seed0"])
        self.assertIs(product.producer.operands[0], derivative.inputs["seed1"])

    def test_singleton_tuple_preserves_cotangent_structure(self):
        class Singleton(fun):
            @jit
            def forward(self, ctx, x):
                return (x,)

            @jit
            def backward(self, ctx, dy):
                seed, = dy
                return (seed,)

        context = trace(Singleton())
        self.assertTrue(context.output_is_tuple)
        derivative = vjp(context.module).functions[0]
        self.assertIs(derivative.outputs[0], derivative.inputs["seed0"])

    def test_custom_call_expands_static_control_with_the_same_unroll_budget(self):
        module = Module()
        forward = module.graph("counted", inputs={"x": TensorType((3,), "f32")})
        value = forward.for_(0, 3, 1, (forward.inputs["x"],),
                             lambda block, index, carry: block.add(carry, carry))
        forward.return_(value, forward.inputs["x"])
        spec = CallSpec("counted", module, 1)
        outer = Module()
        graph = outer.graph("call_loop", inputs={"x": TensorType((3,), "f32")})
        visible, saved = graph.call(graph.inputs["x"], spec=spec)
        graph.return_(graph.add(visible, saved))
        preserved = specialize(outer, max_unroll=0, expand_calls=False)
        self.assertEqual(preserved.functions[0].operations[0].opcode, "call")
        with self.assertRaisesRegex(ValueError, "max_unroll=2"):
            specialize(outer, max_unroll=2)
        flat = specialize(outer, max_unroll=3).functions[0]
        self.assertEqual(sum(operation.opcode == "add" for operation in flat.operations), 4)
        self.assertIs(flat.outputs[0].producer.operands[1], flat.inputs["x"])

    def test_missing_rule_is_deferred_until_a_live_derivative_path(self):
        class ForwardOnly(fun):
            @jit
            def forward(self, ctx, x):
                return x + 1

        inference = trace(ForwardOnly())
        specialize(inference.module).verify()
        with self.assertRaisesRegex(DifferentiationError, "backward.*required"):
            vjp(inference.module)
        inactive = trace(lambda x: (ForwardOnly()(x), x)[1])
        self.assertIs(vjp(inactive.module).functions[0].outputs[0].producer, None)

    def test_invalid_backward_counts_types_and_none_are_rejected(self):
        cases = (("count", lambda dy: (dy, dy), ValueError, "1 gradients"),
                 ("dtype", lambda dy: (dy.astype("f16"),), TypeError, "shape and dtype"),
                 ("bare", lambda dy: dy, TypeError, "tuple of gradients"),
                 ("none", lambda dy: None, TypeError, "None"))
        for name, rule, error, message in cases:
            with self.subTest(name=name):
                class Broken(fun):
                    @jit
                    def forward(self, ctx, x):
                        return x

                    @jit
                    def backward(self, ctx, dy):
                        return rule(dy)

                with self.assertRaisesRegex(error, message):
                    vjp(trace(Broken()).module)

    def test_saved_values_must_belong_to_the_forward_trace(self):
        class InvalidSave(fun):
            @jit
            def forward(self, ctx, x):
                ctx.save_for_backward(1)
                return x

        with self.assertRaisesRegex(TypeError, "this forward trace"):
            trace(InvalidSave())

        class EarlyRead(fun):
            @jit
            def forward(self, ctx, x):
                ctx.saved_tensors
                return x

        with self.assertRaisesRegex(RuntimeError, "only available during backward"):
            trace(EarlyRead())

    def test_state_changes_invalidate_definitions_and_saved_specs(self):
        class Scale(fun):
            def __init__(self):
                self.factor = 2

            @jit
            def forward(self, ctx, x):
                return x * self.factor

            @jit
            def backward(self, ctx, dy):
                return (dy * self.factor,)

        scale = Scale()
        before = scale._definition_key()
        context = trace(scale)
        self.assertEqual(context.dependencies[scale], before)
        scale.factor = 3
        self.assertNotEqual(scale._definition_key(), before)
        with self.assertRaisesRegex(ValueError, "definition changed"):
            vjp(context.module)
        updated = trace(scale)
        self.assertNotEqual(updated.module.to_text(), context.module.to_text())
        scale.tensor_state = TraceContext([((3,), "f32")]).inputs[0]
        with self.assertRaisesRegex(TypeError, "pass Tensor state"):
            scale._definition_key()

    def test_nested_jit_and_custom_calls_expand_without_python_object_reprs(self):
        @jit
        def double(x):
            return x + x

        square = CustomSquare()

        class Nested(fun):
            @jit
            def forward(self, ctx, x):
                return square(double(x))

        context = trace(Nested())
        self.assertIn(square, context.dependencies)
        self.assertIn(double, context.dependencies)
        source = context.module.to_text()
        self.assertNotIn("object at 0x", source)
        flat = specialize(context.module)
        self.assertEqual([op.opcode for op in flat.functions[0].operations], ["add", "mul", "return"])

    def test_call_cannot_silently_change_its_captured_forward_definition(self):
        module = Module()
        forward = module.graph("identity", inputs={"x": TensorType((3,), "f32")})
        forward.return_(forward.inputs["x"])
        spec = CallSpec("identity", module, 1)
        extra = module.graph("extra", inputs={"x": TensorType((3,), "f32")})
        extra.return_(extra.inputs["x"])
        with self.assertRaisesRegex(ValueError, "forward graph changed"):
            spec.verify()


if __name__ == "__main__":
    unittest.main()
