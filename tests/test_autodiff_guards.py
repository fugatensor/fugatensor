"""Runtime tape guards must fail before compiling an invalid backward graph."""
import os
import tempfile
import unittest
from unittest.mock import patch

from ask import Compiler, CompilerError
import fugatensor as ft


class AutodiffGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler = Compiler(os.environ.get("ASK_OPT"))
            capabilities = cls.compiler.capabilities()
        except CompilerError as error:
            raise unittest.SkipTest(f"ASK native compiler is required: {error}")
        if not capabilities.get("host_codegen"):
            raise unittest.SkipTest("ASK host code generation is disabled")

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fuga-autodiff-guards-")
        self.addCleanup(directory.cleanup)
        self.engine = ft.Engine(device="cpu", ask_opt=self.compiler.executable,
                                cache_dir=directory.name)
        self.engine.__enter__()
        self.addCleanup(self.engine.__exit__, None, None, None)

    def test_vector_backward_requires_an_explicit_seed(self):
        @ft.jit
        def square(x):
            return x * x

        x = ft.tensor([2, 3], requires_grad=True)
        result = square(x)
        with patch.object(self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
            with self.assertRaisesRegex((TypeError, ValueError), "scalar|gradient|seed"):
                result.backward()
        self.assertIsNone(x.grad)

    def test_backward_seed_shape_dtype_and_kind_are_checked_before_compilation(self):
        @ft.jit
        def square(x):
            return x * x

        x = ft.tensor([2, 3], requires_grad=True)
        result = square(x)
        for seed in (ft.tensor(1), ft.tensor([1, 1], dtype="f16"), [1, 1]):
            with self.subTest(seed=seed), patch.object(
                    self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
                with self.assertRaisesRegex((TypeError, ValueError), "gradient|seed|shape|dtype|Tensor"):
                    result.backward(seed)
        self.assertIsNone(x.grad)

    def test_integer_and_boolean_tensors_cannot_require_gradients(self):
        for dtype, data in (("bool", True), ("i32", 2), ("i64", 2)):
            with self.subTest(dtype=dtype), self.assertRaisesRegex((TypeError, ValueError), "float|grad"):
                ft.tensor(data, dtype=dtype, requires_grad=True)

    def test_builtin_backward_graph_can_be_differentiated_again(self):
        @ft.jit
        def square(x):
            return x * x

        x = ft.tensor(3, requires_grad=True)
        result = square(x)
        result.backward(create_graph=True)
        first_derivative = x.grad
        self.assertEqual(first_derivative.tolist(), 6)
        self.assertTrue(first_derivative.requires_grad)
        x.zero_grad()
        first_derivative.backward()
        self.assertEqual(x.grad.tolist(), 2)
        self.assertFalse(x.grad.requires_grad)

    def test_backward_graph_flags_do_not_accept_truthy_non_booleans(self):
        @ft.jit
        def square(x):
            return x * x

        x = ft.tensor(3, requires_grad=True)
        result = square(x)
        for options in ({"create_graph": 1}, {"create_graph": "yes"},
                        {"retain_graph": 1}, {"retain_graph": []}):
            with self.subTest(options=options), patch.object(
                    self.engine, "_compile", side_effect=AssertionError("compiled before validation")):
                with self.assertRaisesRegex(TypeError, "bool|retain_graph|create_graph"):
                    result.backward(**options)
        self.assertIsNone(x.grad)

    def test_compiled_forward_uses_current_requires_grad_and_updated_definition(self):
        factor = 2
        traces = []

        @ft.jit
        def scaled(x):
            traces.append(factor)
            return x * factor

        self.assertFalse(scaled(ft.tensor(4)).requires_grad)
        x = ft.tensor(4, requires_grad=True)
        result = scaled(x)
        self.assertTrue(result.requires_grad)
        result.backward()
        self.assertEqual(x.grad.tolist(), 2)
        self.assertEqual(traces, [2])
        factor = 5
        scaled.clear_cache()
        x.zero_grad()
        scaled(x).backward()
        self.assertEqual(x.grad.tolist(), 5)
        self.assertEqual(traces, [2, 5])

    def test_deep_cached_jit_chain_does_not_use_python_recursion_for_backward(self):
        @ft.jit
        def step(x):
            return x + 1

        x = ft.tensor(0, requires_grad=True)
        result = x
        for _ in range(1100):
            result = step(result)
        self.assertEqual(result.tolist(), 1100)
        self.assertEqual(step.cache_info()["compilations"], 1)
        self.assertEqual(step.cache_info()["cache_hits"], 1099)
        result.backward()
        self.assertEqual(x.grad.tolist(), 1)


if __name__ == "__main__":
    unittest.main()
