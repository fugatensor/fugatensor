"""IR-level JVP, VJP, forward/reverse composition, and custom CallSpec / ft.fun contracts."""
from __future__ import annotations

import inspect
import tempfile
import unittest

from fugatensor._state import active_trace
from fugatensor.backends.host import HostBackend, HostStorage
from fugatensor.function import fun
from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, jvp, specialize, vjp
from fugatensor.ir.core import CallSpec
from fugatensor.ir.differentiate import DifferentiationError
from fugatensor.jit import jit
from fugatensor.lowering import lower_to_ask
from fugatensor.tracing import TraceContext
from tests._helpers import make_ask_compiler


def tensor_graph(name="calculate", dtype="f32", names=("x",)):
    module = Module()
    type_ = TensorType((3,), dtype)
    return module, module.graph(name, inputs={n: type_ for n in names})


def cubic(dtype="f32"):
    module, graph = tensor_graph("cubic", dtype=dtype)
    x = graph.inputs["x"]
    graph.return_(graph.mul(graph.mul(x, x), x))
    return module


def opaque_spec(*, effects=()):
    return ExternalSpec(
        compiler="example",
        source="opaque original DSL\r\n",
        entry="kernel",
        signature={"x": "ptr", "y": "ptr"},
        source_bindings={"x": "input:0", "y": "output:0"},
        effects=effects,
    )


def trace_fn(function, specs=(((3,), "f32"),), *static):
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
        (x,) = ctx.saved_tensors
        return (dy * x * 2,)


class ReverseAndForwardIRTests(unittest.TestCase):
    def test_vjp_duplicate_outputs_disconnected_zero_casts_and_static_loops(self):
        module, graph = tensor_graph(names=("x", "seed0"))
        graph.return_(graph.inputs["x"], graph.inputs["x"])
        before = module.to_text()
        derivative = vjp(module, argnums=(0,)).functions[0]
        self.assertEqual(module.to_text(), before)
        self.assertEqual(tuple(derivative.inputs), ("x", "seed0", "_seed0", "seed1"))
        self.assertEqual(derivative.outputs[0].producer.opcode, "add")

        for dtype in ("f16", "bf16", "f32"):
            with self.subTest(dtype=dtype):
                m, g = tensor_graph(dtype=dtype, names=("x", "y"))
                g.return_(g.inputs["x"])
                grad = vjp(m, argnums=1).functions[0].outputs[0]
                self.assertEqual(grad.type, TensorType((3,), dtype))
                self.assertEqual(grad.producer.opcode, "splat")

        # Floating cast VJP casts cotangent back to operand dtype.
        cm, cg = tensor_graph(dtype="f16")
        cg.return_(cg.cast(cg.inputs["x"], dtype="f32"))
        c_vjp = vjp(cm).functions[0]
        self.assertEqual(c_vjp.inputs["seed0"].type, TensorType((3,), "f32"))
        self.assertEqual(c_vjp.outputs[0].type, TensorType((3,), "f16"))

        # Static counted loop specializes cleanly inside vjp.
        lm, lg = tensor_graph()
        lg.return_(lg.for_(0, 3, 1, (lg.inputs["x"],),
                           lambda b, idx, carry: b.mul(carry, b.splat(b.constant(2.0, dtype="f32"), (3,)))))
        l_vjp = vjp(lm)
        l_vjp.verify()
        self.assertNotIn("fuga.for", l_vjp.to_text())

    def test_vjp_and_jvp_external_effects_splat_and_argnum_guards(self):
        # Active external rejected; dead external pruned; inactive external kept as coefficient.
        m_live, g_live = tensor_graph()
        g_live.return_(g_live.external(g_live.inputs["x"], spec=opaque_spec()))
        with self.assertRaisesRegex(DifferentiationError, "external.*explicit VJP"):
            vjp(m_live)
        with self.assertRaisesRegex(DifferentiationError, "external.*explicit JVP"):
            jvp(m_live)

        m_dead, g_dead = tensor_graph()
        g_dead.external(g_dead.inputs["x"], spec=opaque_spec())
        g_dead.return_(g_dead.inputs["x"])
        self.assertNotIn("fuga.external", vjp(m_dead).to_text())

        m_coeff, g_coeff = tensor_graph(names=("x", "y"))
        spec = opaque_spec()
        coeff = g_coeff.external(g_coeff.inputs["y"], spec=spec)
        g_coeff.return_(g_coeff.mul(g_coeff.inputs["x"], coeff))
        d0 = vjp(m_coeff, argnums=(0,)).functions[0]
        self.assertIs(next(op for op in d0.operations if op.opcode == "external").attrs["spec"], spec)
        with self.assertRaisesRegex(DifferentiationError, "external.*explicit VJP"):
            vjp(m_coeff, argnums=(1,))
        inactive_jvp = jvp(m_coeff, argnums=0)
        self.assertIn("opaque original DSL", inactive_jvp.to_text())

        # Effects, integer casts, scalar splat VJP vs JVP, and invalid argnums.
        m_eff, g_eff = tensor_graph()
        g_eff.external(g_eff.inputs["x"], spec=opaque_spec(effects=("unknown",)))
        g_eff.return_(g_eff.inputs["x"])
        with self.assertRaisesRegex(DifferentiationError, "effect-free"):
            vjp(m_eff)

        m_int, g_int = tensor_graph()
        g_int.return_(g_int.cast(g_int.cast(g_int.inputs["x"], dtype="i32"), dtype="f32"))
        with self.assertRaisesRegex(DifferentiationError, "integer/bool"):
            vjp(m_int)
        with self.assertRaisesRegex(DifferentiationError, "integer/bool"):
            jvp(m_int)

        m_splat = Module()
        g_splat = m_splat.graph("bcast", inputs={"x": ScalarType("f16")})
        g_splat.return_(g_splat.splat(g_splat.cast(g_splat.inputs["x"], dtype="f32"), (3,)))
        with self.assertRaisesRegex(DifferentiationError, "sum reduction"):
            vjp(m_splat)
        splat_jvp = jvp(m_splat)
        self.assertEqual([v.type for v in splat_jvp.functions[0].outputs], [TensorType((3,), "f32")] * 2)

        for bad in (True, -1, 2, (0, 0), (), [False]):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                vjp(cubic(), argnums=bad)
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                jvp(cubic(), argnums=bad)

    def test_jvp_vjp_composition_executes_identically_in_both_orders(self):
        compiler = make_ask_compiler()
        if not compiler.capabilities().get("host_codegen"):
            raise unittest.SkipTest("ASK host code generation is disabled")

        def run_host(mod, args):
            with tempfile.TemporaryDirectory(prefix="fuga-ir-ad-") as tmp:
                exe = HostBackend(compiler, tmp).compile(lower_to_ask(mod))
                tensors = [HostStorage.from_flat(vals, shape=v.type.shape, dtype=v.type.dtype)
                           for v, vals in zip(mod.functions[0].inputs.values(), args)]
                return [out.to_flat() for out in exe.run(tensors)]

        mod = cubic()
        fwd = jvp(mod)
        x, dx = [2, -1, 0.5], [0.5, 2, -1]
        self.assertEqual(run_host(fwd, [x, dx]), [[8, -1, 0.125], [6, 6, -0.75]])
        rev_of_fwd = vjp(fwd, argnums=0)
        fwd_of_rev = jvp(vjp(mod), argnums=0)
        r1 = run_host(rev_of_fwd, [x, dx, [0, 0, 0], [1, 1, 1]])[0]
        r2 = run_host(fwd_of_rev, [x, [1, 1, 1], dx])[1]
        self.assertEqual(r1, [6, -12, -3])
        self.assertEqual(r2, r1)


class CustomFunctionIRContractTests(unittest.TestCase):
    def test_decorator_signatures_saved_residuals_and_custom_rule_selection(self):
        with self.assertRaisesRegex(TypeError, "forward.*@ft.jit"):
            class MissingForward(fun):
                def forward(self, ctx, x):
                    return x
        with self.assertRaisesRegex(TypeError, "backward.*@ft.jit"):
            class MissingBackward(fun):
                @jit
                def forward(self, ctx, x):
                    return x
                def backward(self, ctx, dy):
                    return (dy,)
        with self.assertRaisesRegex(TypeError, "backward.*signature"):
            class BadSig(fun):
                @jit
                def forward(self, ctx, x):
                    return x
                @jit
                def backward(self, ctx, a, b):
                    return (a,)

        class Scale(fun):
            @jit
            def forward(self, ctx, x, factor=2, *, offset=1):
                return x * factor + offset

        scale = Scale()
        self.assertEqual(str(inspect.signature(scale)), "(x, factor=2, *, offset=1)")

        # Hidden saved residuals stay on CallSpec until expansion.
        ctx = trace_fn(CustomSquare())
        call, ret = ctx.graph.operations
        spec = call.attrs["spec"]
        self.assertEqual((spec.output_count, spec.saved_count, len(call.results)), (1, 1, 2))
        self.assertEqual(ret.operands, (call.results[0],))
        self.assertNotIn("fuga.call", specialize(ctx.module).to_text())

        # Custom backward rule overrides differentiating forward.
        class Surrogate(fun):
            @jit
            def forward(self, ctx, x):
                return x * x
            @jit
            def backward(self, ctx, dy):
                return (dy * 7,)

        d_surr = vjp(trace_fn(Surrogate()).module).functions[0]
        self.assertEqual([op.attrs["value"] for op in d_surr.operations if op.opcode == "constant"], [7.0])

        # Custom call has no automatic JVP unless registered.
        with self.assertRaisesRegex(DifferentiationError, "call.*explicit JVP"):
            jvp(ctx.module)

    def test_mixed_inputs_tuples_missing_rules_and_definition_invalidation(self):
        class MixedPair(fun):
            @jit
            def forward(self, ctx, x, idx, mask, factor):
                return x, x * factor
            @jit
            def backward(self, ctx, dy):
                d0, d1 = dy
                return (d0 + d1 * 3,)

        ctx = trace_fn(MixedPair(), (((3,), "f32"), ((3,), "i64"), ((3,), "bool")), 3)
        d = vjp(ctx.module, argnums=0).functions[0]
        self.assertEqual(len(d.outputs), 1)
        self.assertEqual(d.outputs[0].type, TensorType((3,), "f32"))

        # Singleton tuple preserves tuple cotangent structure.
        class Singleton(fun):
            @jit
            def forward(self, ctx, x):
                return (x,)
            @jit
            def backward(self, ctx, dy):
                (s,) = dy
                return (s,)

        s_ctx = trace_fn(Singleton())
        self.assertTrue(s_ctx.output_is_tuple)
        s_vjp = vjp(s_ctx.module).functions[0]
        self.assertIs(s_vjp.outputs[0], s_vjp.inputs["seed0"])

        # Invalid backward returns, invalid save_for_backward, and mutated state invalidation.
        for rule, err, msg in (
            (lambda dy: (dy, dy), ValueError, "1 gradients"),
            (lambda dy: (dy.astype("f16"),), TypeError, "shape and dtype"),
            (lambda dy: dy, TypeError, "tuple of gradients"),
            (lambda dy: None, TypeError, "None"),
        ):
            class Broken(fun):
                @jit
                def forward(self, ctx, x):
                    return x
                @jit
                def backward(self, ctx, dy):
                    return rule(dy)
            with self.assertRaisesRegex(err, msg):
                vjp(trace_fn(Broken()).module)

        class MutableScale(fun):
            def __init__(self):
                self.factor = 2
            @jit
            def forward(self, ctx, x):
                return x * self.factor
            @jit
            def backward(self, ctx, dy):
                return (dy * self.factor,)

        ms = MutableScale()
        traced = trace_fn(ms)
        ms.factor = 3
        with self.assertRaisesRegex(ValueError, "definition changed"):
            vjp(traced.module)
        ms.bad_tensor = TraceContext([((3,), "f32")]).inputs[0]
        with self.assertRaisesRegex(TypeError, "pass Tensor state"):
            ms._definition_key()

        # CallSpec forward mutation detection.
        mod = Module()
        fg = mod.graph("id", inputs={"x": TensorType((3,), "f32")})
        fg.return_(fg.inputs["x"])
        cspec = CallSpec("id", mod, 1)
        eg = mod.graph("extra", inputs={"x": TensorType((3,), "f32")})
        eg.return_(eg.inputs["x"])
        with self.assertRaisesRegex(ValueError, "forward graph changed"):
            cspec.verify()


if __name__ == "__main__":
    unittest.main()
