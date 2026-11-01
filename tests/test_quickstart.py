"""Readable public API examples backed by real ASK compilation.

Tool paths belong in setup. The test bodies show the tensor API used by a
program. Enable the GPU cases with FUGATENSOR_TEST_CUDA=1.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import fugatensor as ft
from ask import CompilerEnvironment


class CPUQuickstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="fugatensor-quickstart-")
        cls.addClassCleanup(cls.directory.cleanup)
        ft.configure(device="cpu", ask_opt=os.environ.get("ASK_OPT"), cache_dir=cls.directory.name)
        cls.addClassCleanup(ft.shutdown)

    def test_create_tensors_and_compute(self):
        x = ft.tensor([1, 2, 3])
        y = ft.tensor([10, 20, 30])
        result = (x + y) * 2

        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.dtype, "f32")
        self.assertEqual(result.tolist(), [22, 44, 66])

    def test_jit_reuses_code_for_new_values(self):
        @ft.jit
        def calculate(x, y):
            return (x + y) * 2

        x = ft.tensor([1, 2, 3])
        y = ft.tensor([10, 20, 30])
        self.assertEqual(calculate(x, y).tolist(), [22, 44, 66])

        new_x = ft.tensor([4, 5, 6])
        self.assertEqual(calculate(new_x, y).tolist(), [28, 50, 72])
        self.assertEqual(calculate.cache_info()["compilations"], 1)
        self.assertEqual(calculate.cache_info()["cache_hits"], 1)


@unittest.skipUnless(os.environ.get("FUGATENSOR_TEST_CUDA") == "1",
                     "set FUGATENSOR_TEST_CUDA=1 to run the real GPU quickstart")
class CUDAQuickstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = ("FUGATENSOR_TRITON_PYTHON", "FUGATENSOR_TRITON_WORKER", "FUGATENSOR_LLC", "FUGATENSOR_PTXAS")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise RuntimeError("requested CUDA quickstart is missing: " + ", ".join(missing))
        cls.directory = tempfile.TemporaryDirectory(prefix="fugatensor-cuda-quickstart-")
        cls.addClassCleanup(cls.directory.cleanup)
        ft.configure(device="cuda", ask_opt=os.environ.get("ASK_OPT"), cache_dir=cls.directory.name,
                     compilers={"triton": CompilerEnvironment(
                         python=os.environ["FUGATENSOR_TRITON_PYTHON"],
                         worker=os.environ["FUGATENSOR_TRITON_WORKER"])},
                     llc=os.environ["FUGATENSOR_LLC"], ptxas=os.environ["FUGATENSOR_PTXAS"])
        cls.addClassCleanup(ft.shutdown)

        # The source file below contains ordinary hand-written @triton.jit code.
        cls.source = Path(__file__).parent / "fixtures" / "user_scale.py"
        cls.scale = ft.triton.kernel(
            cls.source, entry="scale_kernel", input="inp", output="out",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
            options={"num_warps": 4},
        )

    def test_create_cuda_tensor_and_call_triton(self):
        x = ft.tensor([1, 2, 3], device="cuda")
        result = self.scale(x)

        self.assertEqual(result.device, "cuda")
        self.assertEqual(result.tolist(), [2, 4, 6])

    def test_jit_combines_ask_and_triton(self):
        scale = self.scale

        @ft.jit
        def calculate(x):
            return scale(x + 1) + 3

        x = ft.tensor([1, 2, 3], device="cuda")
        self.assertEqual(calculate(x).tolist(), [7, 9, 11])

        new_x = ft.tensor([4, 5, 6], device="cuda")
        self.assertEqual(calculate(new_x).tolist(), [13, 15, 17])
        self.assertEqual(calculate.cache_info()["compilations"], 1)
        self.assertEqual(calculate.cache_info()["cache_hits"], 1)
        self.assertEqual(calculate.last_report["fusion"]["compiler"], "triton")
        self.assertEqual(calculate.last_report["external_requests"][0]["source"], self.source.read_text())
        self.assertTrue(list(Path(calculate.last_artifact_dir).rglob("*.cubin")))

    def test_user_writes_a_different_triton_algorithm(self):
        source = Path(__file__).parent / "fixtures" / "user_polynomial.py"
        polynomial = ft.triton.kernel(
            source, entry="polynomial", input="data", output="result",
            constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
            grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
            options={"num_warps": 4},
        )

        @ft.jit
        def calculate(x):
            return polynomial(x + 1) + 3

        x = ft.tensor([1, 2, 3], device="cuda")
        self.assertEqual(calculate(x).tolist(), [8, 13, 20])
        self.assertEqual(calculate.last_report["external_requests"][0]["source"], source.read_text())
        self.assertEqual(calculate.last_report["external_requests"][0]["entry"], "polynomial")

    def test_configuring_cuda_does_not_change_default_cpu_device(self):
        x = ft.tensor([1, 2, 3])
        gpu_x = ft.tensor([1, 2, 3], device="cuda")

        self.assertEqual(x.device, "cpu")
        self.assertEqual(gpu_x.device, "cuda")
        self.assertEqual(gpu_x.tolist(), x.tolist())


if __name__ == "__main__":
    unittest.main()
