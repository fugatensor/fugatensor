"""Fuga IR object model, constant rounding, primitive typing, static specialization, and pruning."""
from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType
import unittest
from unittest.mock import patch

import fugatensor as ft
from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, primitives, prune, specialize, vjp
from fugatensor.ir.transforms import StaticControlFlowError


class FugaIRCoreTests(unittest.TestCase):
    def graph(self, **inputs):
        module = Module()
        return module, module.graph("calculate", inputs=inputs)

    def test_reshape_typing_lowering_and_composable_derivatives(self):
        module, graph = self.graph(x=TensorType((4,), ft.fp32))
        shaped = graph.reshape(graph.inputs['x'], (2, 2))
        graph.return_(shaped)
        module.verify()
        lowered = ft.lower_to_ask(module).to_mlir()
        self.assertIn('ask.reshape', lowered)
        self.assertIn('shape = array<i64: 2, 2>', lowered)
        forward = ft.ir.jvp(module).verify().functions[0]
        reverse = ft.ir.vjp(module).verify().functions[0]
        self.assertEqual([value.type.shape for value in forward.outputs], [(2, 2), (2, 2)])
        self.assertEqual([value.type.shape for value in reverse.outputs], [(4,)])
        for source, target in ((TensorType((None,), ft.fp32), (2,)),
                               (TensorType((4,), ft.fp32), (3,)),
                               (ScalarType(ft.fp32), (1,))):
            _, invalid = self.graph(x=source)
            with self.subTest(source=source), self.assertRaises((TypeError, ValueError)):
                invalid.reshape(invalid.inputs['x'], target)

    def test_public_dtype_objects_normalize_to_canonical_ir_types(self):
        tensor_type = TensorType((2,), ft.fp32)
        scalar_type = ScalarType(ft.int64)
        self.assertIs(type(tensor_type.dtype), str)
        self.assertIs(type(scalar_type.dtype), str)
        self.assertEqual((tensor_type.dtype, scalar_type.dtype), ("f32", "i64"))
        module, graph = self.graph(x=tensor_type)
        low = graph.cast(graph.inputs["x"], dtype=ft.fp16)
        constant = graph.constant(True, dtype=ft.bool)
        graph.return_(low, constant)
        module.verify()
        text = module.to_text()
        self.assertIn("tensor<2xf16>", text)
        self.assertNotIn("ft.", text)
        self.assertNotIn("DType", text)
        lowered = ft.lower_to_ask(module, target={"backend": "host"}).to_mlir()
        self.assertIn("tensor<2xf16>", lowered)
        self.assertNotIn("ft.", lowered)

    def test_types_nodes_immutability_and_ssa_naming(self):
        module, graph = self.graph(x=TensorType((2, 3), "fp16"))
        x = graph.inputs["x"]
        y = graph.add(x, x)
        graph.return_(y)
        self.assertEqual(y.type, TensorType((2, 3), "f16"))
        self.assertEqual(type(y).__module__, "fugatensor.ir.core")
        self.assertEqual(y.producer.opcode, "add")
        self.assertIn("fuga.add", module.to_text())
        self.assertNotIn("ask.", module.to_text())
        with self.assertRaises(TypeError):
            y.producer.attrs["unsafe"] = True

        # Input names cannot collide with generated SSA or region block arguments.
        m2, g2 = self.graph(v0_0=TensorType((3,), "f32"), r0_arg0=ScalarType("bool"))
        xv, pred = g2.inputs.values()
        first = g2.add(xv, xv)
        res = g2.if_(pred, [first], lambda b, v: b.mul(v, v), lambda b, v: b.add(v, v))
        g2.return_(res)
        m2.verify()
        self.assertEqual(first.name, "%v0_0_")
        self.assertEqual(res.producer.regions[0].blocks[0].arguments[0].name, "%r0_arg0_")

        # Unknown dims and overflow guards.
        dynamic_t = TensorType((None, 3), "bf16")
        self.assertEqual(str(dynamic_t), "tensor<?x3xbf16>")
        self.assertFalse(hasattr(dynamic_t, "device"))
        for bad_shape in ((True,), (-1,), (2**63,), (2**62,)):
            with self.subTest(shape=bad_shape), self.assertRaises(ValueError):
                TensorType(bad_shape, "f32")

    def test_constants_own_rne_precision_and_reject_unrounded_literals(self):
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

        bad_mod, bad_g = self.graph()
        raw = bad_g.operation("constant", result_types=[ScalarType("f16")], attrs={"value": 1 + 2**-11}).results[0]
        bad_g.return_(raw)
        with self.assertRaisesRegex(ValueError, "already be rounded"):
            bad_mod.verify()

    def test_origins_verifier_dominance_regions_and_external_specs(self):
        module, graph = self.graph(x=TensorType((3,), "i32"))
        x = graph.inputs["x"]
        orig = graph.add(x, x)
        cloned = graph.operation("add", [orig, x], [x.type], attrs={"overflow": "wrap"},
                                 id="calculate.unrolled1", origin_id=orig.producer.id).results[0]
        graph.return_(cloned)
        module.verify()
        self.assertEqual(cloned.origin_id, orig.value_id)
        self.assertIn('origin="calculate.op0"', module.to_text())

        # Foreign values, implicit region captures, and escaped block values are rejected.
        _, g1 = self.graph(x=TensorType((3,), "f32"))
        _, g2 = self.graph(x=TensorType((3,), "f32"))
        x1 = g1.inputs["x"]
        with self.assertRaisesRegex(ValueError, "preceding values"):
            g1.add(x1, g2.inputs["x"])
        with self.assertRaisesRegex(ValueError, "captures"):
            g1.for_(0, 2, 1, [x1], lambda body, idx, carried: body.add(carried, x1))
        escaped = []
        out = g1.for_(0, 2, 1, [x1], lambda b, idx, c: (escaped.append(b.add(c, c)) or escaped[-1]))
        with self.assertRaisesRegex(ValueError, "preceding values"):
            g1.add(out, escaped[0])

        # Dominance, terminator, and branch/loop type preservation.
        dom_mod, dom_g = self.graph(x=TensorType((3,), "f32"))
        dx = dom_g.inputs["x"]
        a = dom_g.add(dx, dx)
        b = dom_g.mul(a, dx)
        dom_g.return_(b)
        dom_g.block._operations[:2] = reversed(dom_g.block._operations[:2])
        with self.assertRaisesRegex(ValueError, "dominate"):
            dom_mod.verify()
        with self.assertRaisesRegex(ValueError, "terminator"):
            dom_g.constant(1)

        _, cf_g = self.graph(x=TensorType((3,), "f32"))
        cx = cf_g.inputs["x"]
        pred = cf_g.constant(True, dtype="bool")
        with self.assertRaisesRegex(TypeError, "same result types"):
            cf_g.if_(pred, [cx], lambda body, v: v, lambda body, v: body.cast(v, dtype="f16"))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            cf_g.for_(0, 2, 0, [cx], lambda body, idx, v: v)

        # ExternalSpec immutability & effect propagation.
        opts = {"nested": {"values": [1, 2]}}
        spec = ExternalSpec("toy", "def kernel(x, y): pass\r\n", "kernel",
                            {"x": "*f32", "y": "*f32"}, {"x": "input:0", "y": "output:0"},
                            options=opts, target={"backend": "cuda"}, effects=("write",))
        opts["nested"]["values"].append(3)
        self.assertEqual(spec.options["nested"]["values"], (1, 2))
        ext_mod, ext_g = self.graph(x=TensorType((3,), "f32"))
        loop_out = ext_g.for_(0, 2, 1, [ext_g.inputs["x"]], lambda body, idx, v: body.external(v, spec=spec))
        ext_g.return_(loop_out)
        ext_mod.verify()
        self.assertEqual(loop_out.producer.effects, ("write",))


class PrimitiveRuleAndSpecializationTests(unittest.TestCase):
    def test_primitive_abstract_evaluation_and_rule_dispatch(self):
        t = TensorType((None, 4), "f16")
        self.assertEqual(primitives.infer_result_types("add", (t, t)), (t,))
        self.assertEqual(primitives.infer_result_types("cast", (t,), dtype="f32"), (TensorType((None, 4), "f32"),))
        self.assertEqual(primitives.infer_result_types("splat", (ScalarType("bf16"),), {"shape": (2, 3)}),
                         (TensorType((2, 3), "bf16"),))
        with self.assertRaisesRegex(TypeError, "equal operand"):
            primitives.infer_result_types("mul", (t, TensorType((4,), "f16")))
        with self.assertRaisesRegex(ValueError, "numerical attributes"):
            primitives.infer_result_types("add", (ScalarType("i32"), ScalarType("i32")))

        # Textual cast cannot carry a conflicting dtype attribute.
        bad_cast = Module()
        bg = bad_cast.graph("bad", inputs={"x": TensorType((3,), "f32")})
        bg.return_(bg.operation("cast", (bg.inputs["x"],), (TensorType((3,), "f16"),), attrs={"dtype": "i64"}).results[0])
        with self.assertRaisesRegex(TypeError, "cannot have attributes"):
            bad_cast.verify()

        # Malformed reverse rule return types are caught.
        original = primitives.PRIMITIVE_RULES["add"]
        rules = dict(primitives.PRIMITIVE_RULES)
        rules["add"] = replace(original, vjp=lambda b, op, p, c, a: (b.cast(c, dtype="f16"), c))
        mod = Module()
        g = mod.graph("sum", inputs={"x": TensorType((3,), "f32")})
        g.return_(g.add(g.inputs["x"], g.inputs["x"]))
        with patch.object(primitives, "PRIMITIVE_RULES", MappingProxyType(rules)):
            with self.assertRaisesRegex(TypeError, "preserve operand shape and dtype"):
                vjp(mod)

    def test_static_if_for_max_unroll_prune_and_dynamic_rejection(self):
        t = TensorType((3,), "f32")
        mod = Module()
        g = mod.graph("branch", inputs={"x": t})
        val = g.if_(
            g.constant(True, dtype="bool"),
            (g.inputs["x"],),
            lambda b, v: b.add(v, v),
            lambda b, v: b.while_((v,), lambda c, _: c.constant(False, dtype="bool"), lambda body, carry: carry),
        )
        # Unused pure op should be dropped by prune().
        g.mul(g.inputs["x"], g.inputs["x"])
        g.return_(val)
        before = mod.to_text()
        flat = prune(specialize(mod))
        self.assertEqual(mod.to_text(), before)
        self.assertNotIn("fuga.if", flat.to_text())
        self.assertNotIn("fuga.mul", flat.to_text())
        self.assertIn("fuga.add", flat.to_text())

        # Negative step & nested unroll budget.
        neg_mod = Module()
        ng = neg_mod.graph("neg", inputs={"x": t})
        ng.return_(ng.for_(5, -1, -2, (ng.inputs["x"],), lambda b, idx, c: b.add(c, c)))
        neg_flat = specialize(neg_mod).functions[0]
        self.assertEqual([op.attrs["value"] for op in neg_flat.operations if op.opcode == "constant"], [5, 3, 1])

        nested = Module()
        ug = nested.graph("nested", inputs={"x": t})
        ug.return_(ug.for_(0, 2, 1, (ug.inputs["x"],),
                           lambda b, i, c: b.for_(0, 2, 1, (c,), lambda inner, j, v: inner.add(v, v))))
        with self.assertRaisesRegex(StaticControlFlowError, "max_unroll=5"):
            specialize(nested, max_unroll=5)
        self.assertEqual(sum(op.opcode == "add" for op in specialize(nested, max_unroll=6).functions[0].operations), 4)

        # Data-dependent fuga.if and fuga.while remain valid Fuga IR but fail static specialization.
        dyn = Module()
        dg = dyn.graph("dyn", inputs={"pred": ScalarType("bool"), "x": t})
        dg.return_(dg.if_(dg.inputs["pred"], (dg.inputs["x"],), lambda b, x: b.mul(x, x), lambda b, x: x))
        dyn.verify()
        with self.assertRaisesRegex(StaticControlFlowError, "data-dependent fuga.if"):
            specialize(dyn)

        wmod = Module()
        wg = wmod.graph("w", inputs={"x": t})
        wg.return_(wg.while_((wg.inputs["x"],), lambda b, x: b.constant(False, dtype="bool"), lambda b, x: x))
        wmod.verify()
        with self.assertRaisesRegex(StaticControlFlowError, "fuga.while"):
            specialize(wmod)


if __name__ == "__main__":
    unittest.main()
