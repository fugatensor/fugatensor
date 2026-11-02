"""Shared class-scoped Engine fixtures and capability helpers for FugaTensor tests."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from fugatensor._ask import Compiler, CompilerError, ExecutionSession
import fugatensor as ft

ROOT = Path(__file__).resolve().parents[1]
TRITON_KERNELS = ROOT / "tests/fixtures/triton_kernels.py"

requires_cuda = unittest.skipUnless(
    os.environ.get("FUGATENSOR_TEST_CUDA") == "1",
    "set FUGATENSOR_TEST_CUDA=1 to run CUDA hardware tests",
)
requires_triton = unittest.skipUnless(
    os.environ.get("FUGATENSOR_TEST_TRITON") == "1",
    "set FUGATENSOR_TEST_TRITON=1 to run user Triton GPU scheduling tests",
)


def make_ask_compiler() -> Compiler:
    try:
        return Compiler(executable=os.environ.get("ASK_OPT"))
    except CompilerError as error:
        raise unittest.SkipTest(f"ASK compiler is not available: {error}") from error


class EngineTestCase(unittest.TestCase):
    """Manage one class-scoped Engine + cache directory so ASK caches persist across methods."""

    device = "cpu"
    enable_triton = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory(prefix=f"fuga-{cls.device}-")
        cls.addClassCleanup(cls._tmpdir.cleanup)
        cls.cache_dir = Path(cls._tmpdir.name)
        kwargs: dict[str, object] = {"cache_dir": cls.cache_dir}
        if not cls.enable_triton and cls.device != "cpu":
            kwargs["compilers"] = {}
        try:
            cls.engine = ft.Engine(device=cls.device, **kwargs)
        except CompilerError as error:
            if cls.device == "cpu":
                raise unittest.SkipTest(f"ASK host compiler is not available: {error}") from error
            raise
        cls.addClassCleanup(cls.engine.close)
        caps = cls.engine.compiler.capabilities()
        if cls.device == "cpu":
            if not caps.get("host_codegen"):
                raise unittest.SkipTest("ASK host code generation is disabled")
            cls.supports_bf16 = bool(caps.get("host_bf16_hardware_supported"))
        else:
            if not cls.engine.llc or not cls.engine.ptxas:
                raise RuntimeError(
                    "CUDA tests require llc and ptxas (via .venv/share/fugatensor/engine.json "
                    "or FUGATENSOR_LLC / FUGATENSOR_PTXAS)"
                )
            if cls.enable_triton and "triton" not in cls.engine.compilers:
                raise RuntimeError(
                    "Triton tests require a configured Triton worker (via engine.json or "
                    "FUGATENSOR_TRITON_PYTHON / FUGATENSOR_TRITON_WORKER)"
                )
            major, _ = cls.engine.session.capability
            cls.supports_bf16 = major >= 8

    def setUp(self):
        super().setUp()
        token = ft.engine._current.set(self.engine)
        self.addCleanup(ft.engine._current.reset, token)
