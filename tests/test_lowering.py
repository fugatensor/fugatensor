"""Tensor data normalization, TraceContext isolation, Fuga->ASK lowering, and source-map contracts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, Value
from fugatensor.lowering import lower_to_ask
from fugatensor.tensor import Tensor, normalize_data, normalize_dtype
from fugatensor.tracing import TraceContext
from fugatensor._state import active_trace
from tests._helpers import make_ask_compiler


class ShapeAndTracingTests(unittest.TestCase):
    def test_shape_normalization_and_tolist_reconstruction(self):
        self.assertEqual(normalize_data([[1, 2], [3, 4]]), ([1, 2, 3, 4], (2, 2)))
        self.assertEqual(normalize_data([1, 2, 3, 4], (2, 2)), ([1, 2, 3, 4], (2, 2)))
        self.assertEqual(normalize_data(1.5), ([1.5], ()))
        self.assertEqual(normalize_data([[], []]), ([], (2, 0)))
        self.assertEqual(normalize_dtype("fp16"), "f16")

        for bad in ([[1], [2, 3]], [1, [2]], [[[]], [[], []]]):
            with self.subTest(data=bad), self.assertRaisesRegex(ValueError, "rectangular"):
                normalize_data(bad)
        with self.assertRaisesRegex(ValueError, "number of data elements"):
            normalize_data([1, 2], (3,))
        for bad_shape in ((None,), (-1,), (True,)):
            with self.subTest(shape=bad_shape), self.assertRaises(ValueError):
                normalize_data([], bad_shape)

        engine = SimpleNamespace(device="cpu", to_flat=lambda t: t.storage.values,
                                 _check_tensor=lambda t: None)
        for shape, vals, expected in (((2, 2), [1, 2, 3, 4], [[1, 2], [3, 4]]), ((2, 0), [], [[], []]), ((), [7], 7)):
            t = Tensor(engine, SimpleNamespace(shape=shape, dtype="i32", values=vals))
            self.assertEqual(t.tolist(), expected)

    def test_trace_context_isolation_and_device_neutral_fuga_ir(self):
        first = TraceContext([((2,), "f32"), ((3,), "f32"), ((2,), "i32")])
        x, wrong_shape, wrong_dtype = first.inputs
        other = TraceContext([((2,), "f32")]).inputs[0]
        for rhs, err, msg in (
            (other, ValueError, "same graph"),
            (wrong_shape, ValueError, "equal shapes"),
            (wrong_dtype, TypeError, "equal dtypes"),
        ):
            with self.assertRaisesRegex(err, msg):
                x * rhs
        eager_t = Tensor(SimpleNamespace(device="cpu"), SimpleNamespace(shape=(2,), dtype="f32"))
        with self.assertRaisesRegex(TypeError, "captures"):
            x + eager_t
        with self.assertRaisesRegex(TypeError, "scalar"):
            bool(x)
        with self.assertRaisesRegex(TypeError, "while tracing"):
            x.tolist()
        token = active_trace.set(first)
        try:
            predicate = x == 0
        finally:
            active_trace.reset(token)
        self.assertEqual(predicate.value.type, TensorType((2,), 'bool'))
        self.assertEqual(predicate.value.producer.opcode, 'compare')
        scalar = TraceContext([((), 'f32')])
        token = active_trace.set(scalar)
        try:
            with self.assertRaisesRegex(TypeError, 'data-dependent'):
                bool(scalar.inputs[0] == 0)
        finally:
            active_trace.reset(token)

        # Freezing after finish().
        ctx = TraceContext([((1,), "i32")])
        cx = ctx.inputs[0]
        ctx.finish(cx + 1)
        with self.assertRaises(ValueError):
            cx * 2

        # Identical Fuga IR across cpu and cuda tracing contexts.
        ir_texts = []
        for dev in ("cpu", "cuda"):
            tc = TraceContext([((3,), "f32"), ((3,), "f32")], engine=SimpleNamespace(device=dev))
            a, b = tc.inputs
            mod = tc.finish((a + b) * 2)
            ir_texts.append(mod.to_text())
            self.assertNotIn(dev, ir_texts[-1])
            self.assertNotIn("ask.", ir_texts[-1])
        self.assertEqual(ir_texts[0], ir_texts[1])


class LoweringBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = make_ask_compiler()

    def test_mixed_precision_and_bool_lowering_verify_in_ask(self):
        for dtype in ("i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                ctx = TraceContext([((2, 3), dtype), ((2, 3), dtype)])
                x, y = ctx.inputs
                out = ((x.astype("f32") + y.astype("f32")) * 2).astype(dtype)
                fuga_mod = ctx.finish((out, x))
                self.assertIsInstance(x.value, Value)
                prepared = self.compiler.prepare(lower_to_ask(fuga_mod))
                self.assertEqual(prepared.diagnostics["phase"], "prepared")
                self.assertIn(f"tensor<2x3x{dtype}>", prepared.ir)

        bctx = TraceContext([((3,), "bool")])
        bx = bctx.inputs[0]
        with self.assertRaisesRegex(TypeError, "bool"):
            _ = bx + True
        b_mlir = lower_to_ask(bctx.finish((bx.astype("i32") * 2).astype("bool"))).to_mlir()
        self.assertIn("tensor<3xi1>", b_mlir)

    def test_source_map_ids_unrolled_origins_and_atomic_failure(self):
        # Distinct function-local IDs stay distinct at ASK module scope.
        multi = Module()
        t = TensorType((3,), "f32")
        for fn_name in ("left", "right"):
            g = multi.graph(fn_name, inputs={"x": t})
            g.return_(g.add(g.inputs["x"], g.inputs["x"], id="shared"))
        smap = {}
        ask_multi = lower_to_ask(multi, source_map=smap)
        rows = [r for r in smap["values"] if r["origin_value_id"] == "shared.result0"]
        self.assertEqual(len({r["ask_value_id"] for r in rows}), 2)
        self.compiler.prepare(ask_multi)

        # Unrolled loops preserve origin_value_id and distinct ask_value_id.
        loop_mod = Module()
        lg = loop_mod.graph("mapped", inputs={"x": t}, results=[t])
        lg.return_(lg.for_(0, 3, 1, [lg.inputs["x"]], lambda b, i, c: b.add(c, c, id="loop_add")))
        loop_map = {}
        ask_loop = lower_to_ask(loop_mod, source_map=loop_map)
        self.assertEqual(loop_map["schema_version"], 1)
        lrows = [r for r in loop_map["values"] if r["origin_value_id"] == "loop_add.result0"]
        self.assertEqual(len({r["ask_value_id"] for r in lrows}), 3)
        for row in lrows:
            self.assertIn(row["ask_value_id"], ask_loop.to_mlir())

        # Failed lowering does not corrupt caller's source_map dict.
        dyn = Module()
        dg = dyn.graph("unknown_shape", inputs={"x": TensorType((None,), "f32")})
        dg.return_(dg.inputs["x"])
        sentinel = {"old": "keep"}
        with self.assertRaisesRegex(ValueError, "concrete"):
            lower_to_ask(dyn, source_map=sentinel)
        self.assertEqual(sentinel, {"old": "keep"})

    def test_explicit_external_shape_and_dynamic_control_flow_rejection(self):
        mod = Module()
        inc, out = TensorType((3,), "f32"), TensorType((4,), "f32")
        g = mod.graph("explicit_ext", inputs={"x": inc}, results=[out])
        spec = ExternalSpec(
            compiler="metadata_only", source="opaque source", entry="expand",
            signature={"inp": "*fp32", "out": "*fp32"},
            source_bindings={"inp": "input:0", "out": "output:0"},
            target={"backend": "host"}, shape_rule="explicit", effects=(),
        )
        g.return_(g.external(g.inputs["x"], spec=spec, result_types=[out]))
        translated = lower_to_ask(mod, target={"backend": "host"})
        self.assertIn('shape_rule = "fixed"', translated.to_mlir())
        prepared = self.compiler.prepare(translated)
        self.assertEqual(prepared.external_requests[0]["source"], "opaque source")

        dyn_cf = Module()
        cg = dyn_cf.graph("dyn", inputs={"p": ScalarType("bool"), "x": inc}, results=[inc])
        cg.return_(cg.if_(cg.inputs["p"], [cg.inputs["x"]], lambda b, x: b.mul(x, x), lambda b, x: b.add(x, x)))
        with self.assertRaisesRegex((ValueError, NotImplementedError), "dynamic|data-dependent|static"):
            lower_to_ask(dyn_cf)


if __name__ == "__main__":
    unittest.main()
