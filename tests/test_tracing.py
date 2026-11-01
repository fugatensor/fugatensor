"""Trace independent Fuga values, then explicitly lower a static graph to ASK."""
import unittest
from types import SimpleNamespace

from ask import Compiler
from fugatensor.ir import Module, Value
from fugatensor.lowering import lower_to_ask
from fugatensor.tensor import Tensor, normalize_data, normalize_dtype
from fugatensor.tracing import TraceContext


class ShapeTests(unittest.TestCase):
    def test_nested_and_explicit_shapes(self):
        self.assertEqual(normalize_data([[1, 2], [3, 4]]), ([1, 2, 3, 4], (2, 2)))
        self.assertEqual(normalize_data([1, 2, 3, 4], (2, 2)), ([1, 2, 3, 4], (2, 2)))
        self.assertEqual(normalize_data(1.5), ([1.5], ()))
        self.assertEqual(normalize_data([[], []]), ([], (2, 0)))
        self.assertEqual(normalize_dtype("fp16"), "f16")

    def test_bad_data_cannot_create_inconsistent_storage(self):
        for data in ([[1], [2, 3]], [1, [2]], [[[]], [[] , []]]):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "rectangular"):
                normalize_data(data)
        with self.assertRaisesRegex(ValueError, "number of data elements"):
            normalize_data([1, 2], (3,))
        for shape in ((None,), (-1,), (True,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                normalize_data([], shape)
        with self.assertRaises(TypeError):
            normalize_data([object()])

    def test_download_reconstructs_shape_without_losing_empty_axes(self):
        engine = SimpleNamespace(device="cpu", to_flat=lambda tensor: tensor.storage.values)
        for shape, values, expected in [((2, 2), [1, 2, 3, 4], [[1, 2], [3, 4]]),
                                        ((2, 0), [], [[], []]), ((), [7], 7)]:
            tensor = Tensor(engine, SimpleNamespace(shape=shape, dtype="i32", values=values))
            self.assertEqual(tensor.tolist(), expected)


class TraceTests(unittest.TestCase):
    def test_mixed_precision_multioutput_lowers_independent_fuga_ir(self):
        compiler = Compiler()
        for dtype in ("i32", "i64", "f32", "f16", "bf16"):
            with self.subTest(dtype=dtype):
                context = TraceContext([((2, 3), dtype), ((2, 3), dtype)])
                x, y = context.inputs
                result = ((x.astype("f32") + y.astype("f32")) * 2).astype(dtype)
                module = context.finish((result, x))
                self.assertIsInstance(module, Module)
                self.assertIsInstance(x.value, Value)
                module.verify()
                self.assertIn("fuga.add", module.to_text())
                self.assertIn("fuga.mul", module.to_text())
                self.assertNotIn('"ask.add"', module.to_text())
                prepared = compiler.prepare(lower_to_ask(module))
                self.assertEqual(prepared.report["phase"], "prepared")
                self.assertEqual(context.output_specs, (((2, 3), dtype), ((2, 3), dtype)))
                self.assertTrue(context.output_is_tuple)
                self.assertIn('"ask.add"', prepared.ir)
                self.assertIn('"ask.mul"', prepared.ir)
                self.assertIn(f"tensor<2x3x{dtype}>", prepared.ir)

    def test_bool_requires_explicit_cast_and_can_be_returned(self):
        context = TraceContext([((3,), "bool")])
        x = context.inputs[0]
        with self.assertRaisesRegex(TypeError, "bool"):
            x + True
        converted = (x.astype("i32") * 2).astype("bool")
        source = lower_to_ask(context.finish(converted)).to_mlir()
        self.assertIn('tensor<3xi1>', source)
        self.assertEqual(context.output_specs, (((3,), "bool"),))

    def test_vector_conditions_and_value_reads_are_rejected_during_trace(self):
        x = TraceContext([((2,), "f32")]).inputs[0]
        with self.assertRaisesRegex(TypeError, "scalar"):
            bool(x)
        with self.assertRaisesRegex(TypeError, "while tracing"):
            x.tolist()
        with self.assertRaisesRegex(TypeError, "data-dependent"):
            x == 0
        with self.assertRaisesRegex(TypeError, "data-dependent"):
            x != 0

    def test_cross_graph_shapes_dtypes_and_captures_are_rejected(self):
        first = TraceContext([((2,), "f32"), ((3,), "f32"), ((2,), "i32")])
        x, shape_mismatch, dtype_mismatch = first.inputs
        other_graph = TraceContext([((2,), "f32")]).inputs[0]
        for other, error_type, message in [(other_graph, ValueError, "same graph"),
                                           (shape_mismatch, ValueError, "equal shapes"),
                                           (dtype_mismatch, TypeError, "equal dtypes")]:
            with self.subTest(message=message), self.assertRaisesRegex(error_type, message):
                x * other
        tensor = Tensor(SimpleNamespace(device="cpu"), SimpleNamespace(shape=(2,), dtype="f32"))
        with self.assertRaisesRegex(TypeError, "captures"):
            x + tensor
        with self.assertRaisesRegex(TypeError, "captures"):
            tensor + x
        with self.assertRaisesRegex(ValueError, "current trace"):
            first.finish(other_graph)

    def test_real_tensors_reject_cross_engine_or_nonscalar_truth(self):
        first = Tensor(SimpleNamespace(device="cpu"), SimpleNamespace(shape=(2,), dtype="f32"))
        second = Tensor(SimpleNamespace(device="cpu"), SimpleNamespace(shape=(2,), dtype="f32"))
        with self.assertRaisesRegex(ValueError, "same Engine"):
            first + second
        with self.assertRaisesRegex(TypeError, "scalar"):
            bool(first)

    def test_graph_result_is_frozen_after_finish(self):
        context = TraceContext([((1,), "i32")])
        x = context.inputs[0]
        with self.assertRaises(TypeError):
            context.finish(())
        with self.assertRaises(TypeError):
            context.finish(3)
        context.finish(x + 1)
        with self.assertRaises(ValueError):
            context.finish(x)
        with self.assertRaises(ValueError):
            x * 2


if __name__ == "__main__":
    unittest.main()
