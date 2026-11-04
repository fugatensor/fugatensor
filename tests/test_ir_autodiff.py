"""IR-level JVP, VJP, forward/reverse composition, and custom CallSpec / ft.fun contracts."""
from __future__ import annotations

import pytest

import inspect
import tempfile

import fugatensor as ft
from fugatensor._state import active_trace
from fugatensor.backends.host import HostBackend, HostStorage
from fugatensor.function import fun
from fugatensor.ir import ExternalSpec, Module, ScalarType, TensorType, jvp, specialize, vjp
from fugatensor.ir.core import CallSpec
from fugatensor.ir.differentiate import DifferentiationError
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
    def forward(self, x):
        return x * x

    def vjp(self, primals, outputs, cotangents):
        return (cotangents[0] * primals[0] * 2,)


class TestReverseAndForwardIR:
    def test_stop_gradient_blocks_opaque_rules_in_both_ad_directions(self):
        module, graph = tensor_graph()
        opaque = graph.external(graph.inputs["x"], spec=opaque_spec())
        graph.return_(graph.stop_gradient(opaque))
        derivative = vjp(module)
        derivative.verify()
        assert ('fuga.external') not in (derivative.to_text())
        zero = derivative.functions[0].outputs[0]
        assert (zero.producer.opcode) == ('splat')
        assert (zero.producer.operands[0].producer.attrs['value']) == (0.0)
        forward = jvp(module)
        forward.verify()
        assert (len(forward.functions[0].outputs)) == (2)
        tangent = forward.functions[0].outputs[1]
        assert (tangent.producer.opcode) == ('splat')
        assert (tangent.producer.operands[0].producer.attrs['value']) == (0.0)

        class StoppedOpaque(fun):
            def forward(self, x):
                value = x.context.wrap(x.context.graph.external(x.value, spec=opaque_spec()))
                return ft.stop_gradient(value)

        wrapped = trace_fn(StoppedOpaque()).module
        wrapped_derivative = vjp(wrapped)
        wrapped_derivative.verify()
        assert ('fuga.external') not in (wrapped_derivative.to_text())
        assert ('fuga.call') not in (wrapped_derivative.to_text())
        jvp(wrapped).verify()

        class Stop(fun):
            def forward(self, x):
                return ft.stop_gradient(x)

        def external_before_stop(x):
            value = x.context.wrap(x.context.graph.external(x.value, spec=opaque_spec()))
            return Stop()(value)

        outside = trace_fn(external_before_stop).module
        assert ('fuga.external') not in (vjp(outside).to_text())
        tangent = jvp(outside).functions[0].outputs[1]
        assert (tangent.producer.operands[0].producer.attrs['value']) == (0.0)

        class PairWithOpaque(fun):
            def forward(self, x):
                opaque = x.context.wrap(x.context.graph.external(x.value, spec=opaque_spec()))
                return x * x, opaque

        selected = trace_fn(lambda x: PairWithOpaque()(x)[0]).module
        vjp(selected).verify()
        jvp(selected).verify()

        class KeepFirst(fun):
            def forward(self, x, unused):
                return x

        def dead_operand(x):
            opaque = x.context.wrap(x.context.graph.external(x.value, spec=opaque_spec()))
            return KeepFirst()(x * x, opaque)

        ignored = trace_fn(dead_operand).module
        assert ('fuga.external') not in (vjp(ignored).to_text())
        jvp(ignored).verify()

    def test_vjp_duplicate_outputs_disconnected_zero_casts_and_static_loops(self):
        module, graph = tensor_graph(names=("x", "seed0"))
        graph.return_(graph.inputs["x"], graph.inputs["x"])
        before = module.to_text()
        derivative = vjp(module, argnums=(0,)).functions[0]
        assert (module.to_text()) == (before)
        assert (tuple(derivative.inputs)) == (('x', 'seed0', '_seed0', 'seed1'))
        assert (derivative.outputs[0].producer.opcode) == ('add')

        for dtype in ("f16", "bf16", "f32"):
            m, g = tensor_graph(dtype=dtype, names=("x", "y"))
            g.return_(g.inputs["x"])
            grad = vjp(m, argnums=1).functions[0].outputs[0]
            assert (grad.type) == (TensorType((3,), dtype))
            assert (grad.producer.opcode) == ('splat')

        # Floating cast VJP casts cotangent back to operand dtype.
        cm, cg = tensor_graph(dtype="f16")
        cg.return_(cg.cast(cg.inputs["x"], dtype="f32"))
        c_vjp = vjp(cm).functions[0]
        assert (c_vjp.inputs['seed0'].type) == (TensorType((3,), 'f32'))
        assert (c_vjp.outputs[0].type) == (TensorType((3,), 'f16'))

        # Static counted loop specializes cleanly inside vjp.
        lm, lg = tensor_graph()
        lg.return_(lg.for_(0, 3, 1, (lg.inputs["x"],),
                           lambda b, idx, carry: b.mul(carry, b.splat(b.constant(2.0, dtype="f32"), (3,)))))
        l_vjp = vjp(lm)
        l_vjp.verify()
        assert ('fuga.for') not in (l_vjp.to_text())

    def test_vjp_and_jvp_external_effects_splat_and_argnum_guards(self):
        # Active external rejected; dead external pruned; inactive external kept as coefficient.
        m_live, g_live = tensor_graph()
        g_live.return_(g_live.external(g_live.inputs["x"], spec=opaque_spec()))
        with pytest.raises(DifferentiationError, match='external.*explicit VJP'):
            vjp(m_live)
        with pytest.raises(DifferentiationError, match='external.*explicit JVP'):
            jvp(m_live)

        m_dead, g_dead = tensor_graph()
        g_dead.external(g_dead.inputs["x"], spec=opaque_spec())
        g_dead.return_(g_dead.inputs["x"])
        assert ('fuga.external') not in (vjp(m_dead).to_text())

        m_coeff, g_coeff = tensor_graph(names=("x", "y"))
        spec = opaque_spec()
        coeff = g_coeff.external(g_coeff.inputs["y"], spec=spec)
        g_coeff.return_(g_coeff.mul(g_coeff.inputs["x"], coeff))
        d0 = vjp(m_coeff, argnums=(0,)).functions[0]
        assert (next((op for op in d0.operations if op.opcode == 'external')).attrs['spec']) is (spec)
        with pytest.raises(DifferentiationError, match='external.*explicit VJP'):
            vjp(m_coeff, argnums=(1,))
        inactive_jvp = jvp(m_coeff, argnums=0)
        assert ('opaque original DSL') in (inactive_jvp.to_text())

        # Effects, integer casts, scalar splat VJP vs JVP, and invalid argnums.
        m_eff, g_eff = tensor_graph()
        g_eff.external(g_eff.inputs["x"], spec=opaque_spec(effects=("unknown",)))
        g_eff.return_(g_eff.inputs["x"])
        with pytest.raises(DifferentiationError, match='effect-free'):
            vjp(m_eff)

        m_int, g_int = tensor_graph()
        g_int.return_(g_int.cast(g_int.cast(g_int.inputs["x"], dtype="i32"), dtype="f32"))
        with pytest.raises(DifferentiationError, match='integer/bool'):
            vjp(m_int)
        with pytest.raises(DifferentiationError, match='integer/bool'):
            jvp(m_int)

        m_splat = Module()
        g_splat = m_splat.graph("bcast", inputs={"x": ScalarType("f16")})
        g_splat.return_(g_splat.splat(g_splat.cast(g_splat.inputs["x"], dtype="f32"), (3,)))
        splat_vjp = vjp(m_splat)
        assert (splat_vjp.functions[0].outputs[0].type) == (ScalarType('f16'))
        assert ('fuga.reduce_sum') in (splat_vjp.to_text())
        splat_jvp = jvp(m_splat)
        assert ([v.type for v in splat_jvp.functions[0].outputs]) == ([TensorType((3,), 'f32')] * 2)

        for bad in (True, -1, 2, (0, 0), (), [False]):
            with pytest.raises((TypeError, ValueError)):
                vjp(cubic(), argnums=bad)
            with pytest.raises((TypeError, ValueError)):
                jvp(cubic(), argnums=bad)

    def test_jvp_vjp_composition_executes_identically_in_both_orders(self):
        compiler = make_ask_compiler()
        if not compiler.capabilities().get("host_codegen"):
            raise RuntimeError("ASK host code generation is disabled")

        def run_host(mod, args):
            with tempfile.TemporaryDirectory(prefix="fuga-ir-ad-") as tmp:
                exe = HostBackend(compiler, tmp).compile(lower_to_ask(mod))
                tensors = [HostStorage.from_flat(vals, shape=v.type.shape, dtype=v.type.dtype)
                           for v, vals in zip(mod.functions[0].inputs.values(), args)]
                return [out.to_flat() for out in exe.run(tensors)]

        mod = cubic()
        fwd = jvp(mod)
        x, dx = [2, -1, 0.5], [0.5, 2, -1]
        assert (run_host(fwd, [x, dx])) == ([[8, -1, 0.125], [6, 6, -0.75]])
        rev_of_fwd = vjp(fwd, argnums=0)
        fwd_of_rev = jvp(vjp(mod), argnums=0)
        r1 = run_host(rev_of_fwd, [x, dx, [0, 0, 0], [1, 1, 1]])[0]
        r2 = run_host(fwd_of_rev, [x, [1, 1, 1], dx])[1]
        assert (r1) == ([6, -12, -3])
        assert (r2) == (r1)


class TestCustomFunctionIRContract:
    def test_opaque_forward_keeps_custom_rule_boundary_in_higher_order_ir(self):
        class OutputRule(fun):
            def forward(self, x):
                return x.context.wrap(x.context.graph.external(x.value, spec=opaque_spec()))

            def vjp(self, primals, outputs, cotangents):
                return (outputs[0] * cotangents[0],)

        traced = trace_fn(OutputRule()).module
        original = traced.functions[0].operations[0].attrs["spec"]
        derivative = vjp(traced)
        second = vjp(derivative, argnums=0)
        second.verify()
        calls = [op for op in second.functions[0].operations if op.opcode == "call"]
        assert (calls)
        assert (all((op.attrs['spec'] is original for op in calls)))
        assert not (any((op.opcode == 'external' for op in second.functions[0].operations)))

        class PrimalRule(fun):
            def forward(self, x):
                return x.context.wrap(x.context.graph.external(x.value, spec=opaque_spec()))

            def vjp(self, primals, outputs, cotangents):
                return (primals[0] * primals[0] * cotangents[0],)

        second = vjp(vjp(trace_fn(PrimalRule()).module), argnums=0)
        second.verify()
        assert not (any((op.opcode in ('call', 'external') for op in second.functions[0].operations)))

    def test_plain_method_signatures_visible_outputs_and_custom_rule_selection(self):
        with pytest.raises(TypeError, match='define forward'):
            class MissingForward(fun):
                pass
        with pytest.raises(TypeError, match='instance method'):
            class StaticForward(fun):
                @staticmethod
                def forward(x):
                    return x
        with pytest.raises(TypeError, match='vjp.*signature'):
            class BadSig(fun):
                def forward(self, x):
                    return x

                def vjp(self, a, b):
                    return (a,)

        class Scale(fun):
            def forward(self, x, factor=2, *, offset=1):
                return x * factor + offset

        scale = Scale()
        assert (str(inspect.signature(scale))) == ('(x, factor=2, *, offset=1)')

        # Ordinary methods create stable internal definitions, including when
        # inherited. Repeated tracing must not invalidate their captured rules.
        square = CustomSquare()
        definition = square._definition_key()
        traced = trace_fn(square)
        assert (square._definition_key()) == (definition)
        assert (trace_fn(square).module.to_text()) == (traced.module.to_text())

        class InheritedSquare(CustomSquare):
            pass
        assert (inspect.getattr_static(InheritedSquare, 'forward')) is (inspect.getattr_static(CustomSquare, 'forward'))
        square.clear_cache()
        with pytest.raises(ValueError, match='definition changed'):
            vjp(traced.module)

        # Calls expose only ordinary forward outputs; VJP receives explicit primals and results.
        ctx = trace_fn(CustomSquare())
        call, ret = ctx.graph.operations
        spec = call.attrs["spec"]
        assert ((spec.output_count, len(call.results))) == ((1, 1))
        assert (ret.operands) == ((call.results[0],))
        assert ('fuga.call') not in (specialize(ctx.module).to_text())

        # Custom VJP rule overrides differentiating forward.
        class Surrogate(fun):
            def forward(self, x):
                return x * x

            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * 7,)

        d_surr = vjp(trace_fn(Surrogate()).module).functions[0]
        assert ([op.attrs['value'] for op in d_surr.operations if op.opcode == 'constant']) == ([7.0])

        # Forward-only calls compose with both AD directions.
        class AutomaticSquare(fun):
            def forward(self, x):
                return x * x

        automatic = trace_fn(AutomaticSquare()).module
        assert (len(vjp(automatic).functions[0].outputs)) == (1)
        assert (len(vjp(vjp(automatic), argnums=0).functions[0].outputs)) == (1)
        assert (len(jvp(automatic).functions[0].outputs)) == (2)

        # An explicit VJP is a separate rule from forward-mode differentiation.
        with pytest.raises(DifferentiationError, match='call.*explicit JVP'):
            jvp(ctx.module)

        class Conditional(fun):
            def forward(self, x):
                return x if x else -x

        with pytest.raises(TypeError, match='isolated fun'):
            trace_fn(Conditional(), (((), "f32"),))

    def test_mixed_inputs_tuples_missing_rules_and_definition_invalidation(self):
        class MixedPair(fun):
            def forward(self, x, idx, mask, factor):
                return x, x * factor

            def vjp(self, primals, outputs, cotangents):
                assert len(primals) == 3
                d0, d1 = cotangents
                return (d0 + d1 * 3,)

        ctx = trace_fn(MixedPair(), (((3,), "f32"), ((3,), "i64"), ((3,), "bool")), 3)
        d = vjp(ctx.module, argnums=0).functions[0]
        assert (len(d.outputs)) == (1)
        assert (d.outputs[0].type) == (TensorType((3,), 'f32'))

        # Singleton tuple preserves tuple cotangent structure.
        class Singleton(fun):
            def forward(self, x):
                return (x,)

            def vjp(self, primals, outputs, cotangents):
                (s,) = cotangents
                return (s,)

        s_ctx = trace_fn(Singleton())
        assert (s_ctx.output_is_tuple)
        s_vjp = vjp(s_ctx.module).functions[0]
        assert (s_vjp.outputs[0]) is (s_vjp.inputs['seed0'])

        # VJP return arity/types and function state are checked before execution.
        for rule, err, msg in (
            (lambda dy: (dy, dy), ValueError, "1 gradients"),
            (lambda dy: (dy.astype("f16"),), TypeError, "shape and dtype"),
            (lambda dy: dy, TypeError, "tuple of gradients"),
            (lambda dy: None, TypeError, "None"),
        ):
            class Broken(fun):
                def forward(self, x):
                    return x

                def vjp(self, primals, outputs, cotangents):
                    return rule(cotangents[0])
            with pytest.raises(err, match=msg):
                vjp(trace_fn(Broken()).module)

        class MutableScale(fun):
            def __init__(self):
                self.factor = 2

            def forward(self, x):
                return x * self.factor

            def vjp(self, primals, outputs, cotangents):
                return (cotangents[0] * self.factor,)

        ms = MutableScale()
        traced = trace_fn(ms)
        ms.factor = 3
        with pytest.raises(ValueError, match='definition changed'):
            vjp(traced.module)
        ms.bad_tensor = TraceContext([((3,), "f32")]).inputs[0]
        with pytest.raises(TypeError, match='pass Tensor state'):
            ms._definition_key()

        # Replacing a plain method still invalidates an already-captured rule;
        # the replacement is usable by a fresh trace without a decorator.
        class ReplaceRule(fun):
            def forward(self, x):
                return x

            def vjp(self, primals, outputs, cotangents):
                return cotangents

        replacement = ReplaceRule()
        old = trace_fn(replacement)

        def new_vjp(self, primals, outputs, cotangents):
            return (cotangents[0] * 9,)

        ReplaceRule.vjp = new_vjp
        with pytest.raises(ValueError, match='definition changed'):
            vjp(old.module)
        new_derivative = vjp(trace_fn(replacement).module).functions[0]
        assert (9.0) in ([op.attrs['value'] for op in new_derivative.operations if op.opcode == 'constant'])

        # CallSpec forward mutation detection.
        mod = Module()
        fg = mod.graph("id", inputs={"x": TensorType((3,), "f32")})
        fg.return_(fg.inputs["x"])
        cspec = CallSpec("id", mod)
        eg = mod.graph("extra", inputs={"x": TensorType((3,), "f32")})
        eg.return_(eg.inputs["x"])
        with pytest.raises(ValueError, match='forward graph changed'):
            cspec.verify()
