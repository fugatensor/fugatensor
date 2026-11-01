"""Real ASK host-object JIT checks; no Python arithmetic fallback."""
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ask import Compiler, CompilerError, Module, ScalarType, TensorType
from fugatensor.backends.host import HostBackend, HostStorage


class HostStorageTests(unittest.TestCase):
    def test_storage_roundtrips_and_boundaries(self):
        for dtype, values in (
            ("bool", [True, False]), ("i32", [-(2**31), 2**31 - 1]),
            ("i64", [-(2**63), 2**63 - 1]),
            ("f32", [1.5, -0.0]), ("f16", [1.5, -0.0]), ("bf16", [1.5, -0.0]),
        ):
            with self.subTest(dtype=dtype):
                storage = HostStorage.from_flat(values, shape=(1, 2), dtype=dtype)
                self.assertEqual(storage.to_flat(), values)
                self.assertGreater(storage.data_ptr, 0)
                if dtype.startswith("f") or dtype == "bf16":
                    self.assertEqual(math.copysign(1, storage.to_flat()[1]), -1)
        self.assertEqual(HostStorage.from_flat([], shape=(0,), dtype="f32").nbytes, 0)
        self.assertEqual(HostStorage.from_flat([1], shape=(), dtype="fp16").dtype, "f16")
        with self.assertRaises(ValueError):
            HostStorage.from_flat([2**63], shape=(1,), dtype="i64")
        with self.assertRaises(TypeError):
            HostStorage.from_flat([1], shape=(1,), dtype="bool")
        with self.assertRaises(ValueError):
            HostStorage.from_flat([1], shape=(2,), dtype="i32")
        with self.assertRaises(ValueError):
            HostStorage.from_flat([1e30], shape=(1,), dtype="f16")


class HostJITTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler = Compiler(os.environ.get("ASK_OPT"))
            cls.capabilities = cls.compiler.capabilities()
        except CompilerError as error:
            raise unittest.SkipTest(f"ASK native compiler is required: {error}")
        if not cls.capabilities.get("host_codegen"):
            raise unittest.SkipTest("ASK host code generation is disabled")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = HostBackend(self.compiler, self.directory.name)

    def binary_module(self, dtype, shape=(2, 2)):
        module = Module()
        type_ = TensorType(shape, dtype)
        graph = module.graph("different_entry_name", inputs={"x": type_, "y": type_}, results=[type_, type_])
        x, y = graph.inputs["x"], graph.inputs["y"]
        if dtype == "bool":
            x, y = graph.cast(x, dtype="i32"), graph.cast(y, dtype="i32")
        summed, product = graph.add(x, y), graph.mul(x, y)
        if dtype == "bool":
            summed, product = graph.cast(summed, dtype="bool"), graph.cast(product, dtype="bool")
        graph.return_(summed, product)
        return module

    def test_same_graph_executes_multiple_dtypes_and_outputs(self):
        for dtype in ("bool", "i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                module = self.binary_module(dtype)
                if dtype == "bf16" and not self.capabilities["host_bf16_hardware_supported"]:
                    with self.assertRaisesRegex(CompilerError, "bf16 requires hardware support"):
                        self.backend.compile(module)
                    continue
                values = [True, False, True, False] if dtype == "bool" else [1, -2, 3, 4]
                other = [True, True, False, False] if dtype == "bool" else [2, 3, 4, -5]
                compiled = self.backend.compile(module)
                inputs = [HostStorage.from_flat(v, shape=(2, 2), dtype=dtype) for v in (values, other)]
                actual = compiled.run(inputs)
                self.assertEqual(actual[0].to_flat(), [bool(a + b) if dtype == "bool" else a + b
                                                     for a, b in zip(values, other)])
                self.assertEqual(actual[1].to_flat(), [bool(a * b) if dtype == "bool" else a * b
                                                     for a, b in zip(values, other)])
                self.assertEqual(compiled.entry, "different_entry_name")
                self.assertTrue((compiled.artifact_dir / "graph.o").is_file())
                self.assertEqual(compiled.report["phase"], "host_object")
                self.assertNotEqual(actual[0].data_ptr, actual[1].data_ptr)
                with patch("subprocess.run", side_effect=AssertionError("warm call recompiled")):
                    self.assertEqual(compiled.run(inputs)[0].to_flat(), actual[0].to_flat())

    def test_input_signature_is_checked_before_call(self):
        compiled = self.backend.compile(self.binary_module("f32"))
        good = HostStorage.from_flat([1, 2, 3, 4], shape=(2, 2), dtype="f32")
        with self.assertRaisesRegex(ValueError, "input count"):
            compiled.run([good])
        for wrong in (HostStorage((4,), "f32"), HostStorage((2, 2), "i32")):
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                compiled.run([good, wrong])
        with self.assertRaises(TypeError):
            compiled.run([good, [1, 2, 3, 4]])

    def test_integer_precision_empty_and_rank_zero(self):
        for shape, values in (((1,), [2**60 + 3]), ((0,), []), ((), [7])):
            with self.subTest(shape=shape):
                compiled = self.backend.compile(self.binary_module("i64", shape))
                incoming = HostStorage.from_flat(values, shape=shape, dtype="i64")
                outputs = compiled.run([incoming, incoming])
                self.assertEqual(outputs[0].to_flat(), [value * 2 for value in values])

    def test_explicit_fp32_compute_preserves_fp16_ulp(self):
        module = Module()
        type_ = TensorType((1,), "f16")
        graph = module.graph("mixed", inputs={"x": type_}, results=[type_])
        x = graph.cast(graph.inputs["x"], dtype="f32")
        plus = graph.splat(graph.constant(1024, dtype="f32"), (1,))
        minus = graph.splat(graph.constant(-1024, dtype="f32"), (1,))
        y = graph.cast(graph.add(graph.add(x, plus), minus), dtype="f16")
        graph.return_(y)
        value = 1 + 2**-10
        compiled = self.backend.compile(module)
        incoming = HostStorage.from_flat([value], shape=(1,), dtype="f16")
        self.assertEqual(compiled.run([incoming])[0].to_flat(), [value])

    def test_scalar_boundary_is_rejected_without_calling_it(self):
        module = Module()
        graph = module.graph("scalar", inputs={"x": ScalarType("i32")}, results=[ScalarType("i32")])
        graph.return_(graph.inputs["x"])
        with self.assertRaisesRegex(CompilerError, "tensor input/output"):
            self.backend.compile(module)


if __name__ == "__main__":
    unittest.main()
