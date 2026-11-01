"""User @triton.jit source snapshotting, deferred SM arch binding, and real ASK + Triton GPU fusion."""
from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from ask.cuda import CUDAError
import fugatensor as ft
from fugatensor.lowering import lower_to_ask
from fugatensor.tracing import TraceContext
from tests._helpers import EXAMPLES_DIR, EngineTestCase, make_ask_compiler, requires_triton

AFFINE_SRC = """import triton
import triton.language as tl

@triton.jit
def affine(in_data, out_data, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(in_data + offsets, mask=offsets < N, other=0)
    tl.store(out_data + offsets, value * 2 + 1, mask=offsets < N)
"""

IDENTITY_SRC = """import triton
import triton.language as tl

@triton.jit
def identity_kernel(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(inp + idx, mask=idx < N, other=0)
    tl.store(out + idx, val, mask=idx < N)
"""

POLYNOMIAL_SRC = """import triton
import triton.language as tl

@triton.jit
def polynomial(data, result, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(data + idx, mask=idx < N, other=0)
    tl.store(result + idx, val * val + 1, mask=idx < N)
"""


class TritonRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = make_ask_compiler()

    def context(self, shape=(2, 5), dtype="f32", device="cuda"):
        return TraceContext([(shape, dtype)], engine=SimpleNamespace(device=device, capability=(8, 6)))

    def registration(self, **changes):
        opts = dict(
            entry="affine", input="in_data", output="out_data",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128,),
        )
        opts.update(changes)
        return ft.triton.from_source(AFFINE_SRC, **opts)

    def prepare_request(self, op, ctx=None):
        ctx = self.context() if ctx is None else ctx
        mod = ctx.finish(op(ctx.inputs[0]))
        target = {"backend": "cuda", "arch": 86} if ctx.engine.device != "cpu" else None
        return self.compiler.prepare(lower_to_ask(mod, target=target)).report["external_requests"][0]

    def test_verbatim_snapshot_without_importing_triton_and_frozen_metadata(self):
        crlf_src = AFFINE_SRC.replace("\n", "\r\n")
        orig_import = builtins.__import__

        def block_triton(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("registration must not import triton")
            return orig_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_kernel.py"
            path.write_bytes(crlf_src.encode("utf-8"))
            with mock.patch("builtins.__import__", side_effect=block_triton):
                op = ft.triton.kernel(path, entry="affine", input="in_data", output="out_data",
                                      constants={"N": 10, "BLOCK": 128}, grid=(1,))
            path.write_text(IDENTITY_SRC, encoding="utf-8")
            self.assertEqual(op.source, crlf_src)
            self.assertEqual(self.prepare_request(op)["source"], crlf_src)

        # Frozen TensorSpec + copied dicts + all 6 storage pointer signatures.
        def check_spec(spec):
            with self.assertRaises(FrozenInstanceError):
                spec.dtype = "i32"
            return {"N": spec.numel, "BLOCK": 128}

        consts = {"N": 10, "BLOCK": 128}
        options = {"num_warps": 8, "enable_fp_fusion": False}
        copied_op = self.registration(constants=consts, options=options)
        consts["N"] = 999
        options["num_warps"] = 4
        self.assertEqual(self.prepare_request(copied_op)["constants"]["N"], 10)
        self.assertEqual(self.prepare_request(copied_op)["fixed_options"]["num_warps"], 8)
        self.assertEqual(self.prepare_request(self.registration(constants=check_spec), self.context((3, 129), "f16"))["grid"], [4, 1, 1])

        for dtype, sig in (("bool", "*i1"), ("i32", "*i32"), ("i64", "*i64"), ("f32", "*fp32"), ("f16", "*fp16"), ("bf16", "*bf16")):
            req = self.prepare_request(self.registration(), self.context(dtype=dtype))
            self.assertEqual(req["signature"], {"in_data": sig, "out_data": sig})

    def test_deferred_arch_binding_and_invalid_registration_guards(self):
        class NoProbeEngine:
            device = "cuda"
            @property
            def capability(self):
                raise AssertionError("tracing must not probe physical GPU")

        ctx = TraceContext([((2, 5), "f32")], engine=NoProbeEngine())
        mod = ctx.finish(self.registration()(ctx.inputs[0]))
        before = mod.to_text()
        for arch in (80, 86):
            req = self.compiler.prepare(lower_to_ask(mod, target={"backend": "cuda", "arch": arch})).report["external_requests"][0]
            self.assertEqual((req["target"]["arch"], mod.to_text()), (arch, before))

        with self.assertRaisesRegex(ValueError, "no implementation"):
            lower_to_ask(mod, target={"backend": "host"})

        for bad in (
            {"entry": "missing"},
            {"input": "missing"},
            {"output": "in_data"},
            {"constants": {"BLOCK": 128}},
            {"grid": ()},
            {"grid": (0,)},
            {"grid": (2**31,)},
        ):
            with self.assertRaises(ValueError):
                self.registration(**bad)
        with self.assertRaisesRegex(TypeError, "integer constexpr"):
            self.registration(constants={"N": 10.0, "BLOCK": 128})


@requires_triton
class TritonGPUExecutionTests(EngineTestCase):
    device = "cuda"
    enable_triton = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.scale_path = EXAMPLES_DIR / "kernels.py"
        cls.scale = ft.triton.kernel(
            cls.scale_path, entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )
        cls.identity = ft.triton.from_source(
            IDENTITY_SRC, entry="identity_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )
        cls.polynomial = ft.triton.from_source(
            POLYNOMIAL_SRC, entry="polynomial", input="data", output="result",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1), options={"num_warps": 4},
        )

    def test_fused_forward_across_precisions_algorithms_and_custom_backward(self):
        scale, identity, polynomial = self.scale, self.identity, self.polynomial

        @ft.jit
        def mixed(x):
            return scale(x + 1) + 3

        for dtype in ("i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                if dtype == "bf16" and not self.supports_bf16:
                    with self.assertRaisesRegex(CUDAError, "bf16"):
                        mixed(ft.tensor([[1, 2], [3, 4]], dtype=dtype))
                    continue
                out = mixed(ft.tensor([[1, 2], [3, 4]], dtype=dtype))
                self.assertEqual((out.tolist(), out.dtype), ([[7, 9], [11, 13]], dtype))
                self.assertEqual(mixed(ft.tensor([[5, 6], [7, 8]], dtype=dtype)).tolist(), [[15, 17], [19, 21]])
                self.assertEqual(mixed.last_report["phase"], "fused_llvm")
                self.assertEqual(mixed.last_report["fusion"]["compiler"], "triton")
                self.assertEqual(mixed.last_report["external_requests"][0]["source"], self.scale_path.read_text())
                self.assertTrue((Path(mixed.last_artifact_dir) / "fused.ll").is_file())

        # Bool kernel, mixed-precision casts, and custom polynomial kernel.
        @ft.jit
        def bool_id(x):
            return identity(x)

        self.assertEqual(bool_id(ft.tensor([False, True, False], dtype="bool")).tolist(), [False, True, False])

        @ft.jit
        def poly_graph(x):
            return polynomial(x + 1) + 3

        self.assertEqual(poly_graph(ft.tensor([1, 2, 3])).tolist(), [8, 13, 20])

        # Raw Triton on derivative path fails preflight; ft.fun custom backward fuses both forward and backward!
        rx = ft.tensor(3.0, requires_grad=True)
        rout = mixed(rx)
        with mock.patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
            with self.assertRaisesRegex(ValueError, "external.*(VJP|reverse|backward)|derivative.*external"):
                rout.backward()
        self.assertIsNone(rx.grad)

        class UserScale(ft.fun):
            @ft.jit
            def forward(self, ctx, x):
                return scale(x)
            @ft.jit
            def backward(self, ctx, dy):
                return (scale(dy),)

        op = UserScale()

        @ft.jit
        def diff_mixed(x):
            return op(x + 1) + 3

        dx = ft.tensor([1.0, 2.0, 3.0], requires_grad=True)
        dout = diff_mixed(dx)
        self.assertEqual(dout.tolist(), [7.0, 9.0, 11.0])
        dout.backward(ft.tensor([1.0, 2.0, -1.0]))
        self.assertEqual(dx.grad.tolist(), [2.0, 4.0, -2.0])


if __name__ == "__main__":
    unittest.main()
