"""Engine configuration discovery, explicit isolation, and bootstrap CLI contracts."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fugatensor._config import runtime_defaults
from fugatensor.engine import Engine, configure, shutdown

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fuga_bootstrap", ROOT / "tools/bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class EngineConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "engine.json"
        self.path.write_text(json.dumps({
            "schema_version": 1,
            "engine_selection": "all",
            "llc": "/saved/llc",
            "ptxas": "/saved/ptxas",
            "compilers": {"triton": {"python": "/saved/python", "worker": "/saved/worker.py"}},
        }))
        environment = patch.dict(os.environ, {"FUGATENSOR_CONFIG": str(self.path)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(shutdown)

    def test_saved_paths_need_no_environment_exports_or_gpu_probe(self):
        with patch("fugatensor.engine.Compiler"), patch("fugatensor.engine.ExecutionSession") as session:
            with Engine(device="cuda") as engine:
                self.assertEqual((engine.llc, engine.ptxas), ("/saved/llc", "/saved/ptxas"))
                self.assertEqual(engine.compilers["triton"].worker, "/saved/worker.py")
                session.assert_not_called()
        self.assertEqual(runtime_defaults("cpu"), {})

    def test_environment_and_explicit_options_override_saved_values(self):
        with patch("fugatensor.engine.Compiler"), patch.dict(os.environ, {"FUGATENSOR_LLC": "/env/llc"}):
            with Engine(device="cuda", ptxas="/explicit/ptxas", compilers={}) as engine:
                self.assertEqual(engine.llc, "/env/llc")
                self.assertEqual(engine.ptxas, "/explicit/ptxas")
                self.assertEqual(engine.compilers, {})
            engine = configure(device="cuda", llc="/configured/llc", compilers={})
            self.assertEqual(engine.llc, "/configured/llc")
            self.assertEqual(engine.ptxas, "/saved/ptxas")

    def test_explicit_or_env_ask_opt_isolates_from_saved_workers(self):
        self.path.write_text("broken stale configuration")
        with patch("fugatensor.engine.Compiler") as compiler:
            with Engine(device="cuda", ask_opt="/other/ask-opt") as engine:
                self.assertIsNone(engine.llc)
                self.assertEqual(engine.compilers, {})
                compiler.assert_called_with(executable="/other/ask-opt")
            engine = configure(device="cuda", ask_opt="/other/ask-opt", ptxas="/other/ptxas")
            self.assertIsNone(engine.llc)
            self.assertEqual(engine.compilers, {})
        with patch("fugatensor.engine.Compiler") as compiler, patch.dict(os.environ, {"ASK_OPT": "/env/ask-opt"}):
            with Engine(device="cuda") as engine:
                self.assertIsNone(engine.llc)
                self.assertEqual(engine.compilers, {})
                compiler.assert_called_with(executable="/env/ask-opt")

    def test_invalid_configuration_reports_clear_error(self):
        for content in (
            "invalid json",
            json.dumps({"schema_version": True}),
            json.dumps({"schema_version": 1, "llc": []}),
            json.dumps({"schema_version": 1, "compilers": []}),
        ):
            self.path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                runtime_defaults("cuda")
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "does not name"):
            runtime_defaults("cuda")


class BootstrapTests(unittest.TestCase):
    def test_compiler_selection_and_flags(self):
        args = bootstrap.parser().parse_args([])
        self.assertEqual(args.compilers, "all")
        self.assertFalse(args.skip_ask)
        self.assertTrue(bootstrap.parser().parse_args(["--skip-ask"]).skip_ask)
        self.assertEqual(bootstrap.select_compilers("all", {"other": "other_dir", "triton": "triton"}),
                         ("other", "triton"))
        self.assertEqual(bootstrap.select_compilers("none", {"triton": "triton"}), ())
        self.assertEqual(bootstrap.select_compilers("triton", {"triton": "triton"}), ("triton",))
        for invalid in ("all;triton", "none;triton", "triton;triton", "missing", "triton;", "../triton"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bootstrap.select_compilers(invalid, {"triton": "triton"})

    def test_ask_install_command_passes_defines_and_only_nondefault_worker_pythons(self):
        venv_py = Path("/a venv/bin/python")
        cmd = bootstrap.ask_install_command(
            venv_py,
            Path("/source ask"),
            Path("/sdk llvm"),
            Path("/build area/all"),
            "triton;other",
            {"triton": Path("/worker env/bin/python"), "other": venv_py},
        )
        self.assertEqual(cmd[:4], ["/a venv/bin/python", "-m", "pip", "install"])
        self.assertIn("/source ask", cmd)
        self.assertIn("--no-build-isolation", cmd)
        self.assertIn("--no-deps", cmd)
        self.assertIn("-Ccmake.define.ASK_THIRD_PARTY_COMPILERS=triton;other", cmd)
        self.assertIn("-Ccmake.define.ASK_TRITON_PYTHON_EXECUTABLE=/worker env/bin/python", cmd)
        self.assertFalse(any("ASK_OTHER_PYTHON_EXECUTABLE" in arg for arg in cmd))
        self.assertIn("-Ccmake.define.LLVM_DIR=/sdk llvm/lib/cmake/llvm", cmd)
        self.assertIn("-Cbuild-dir=/build area/all/{wheel_tag}", cmd)

    @unittest.skipUnless(os.name == "posix", "executable symlinks need POSIX")
    def test_worker_python_symlink_and_build_tool_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "base-python"
            executable.write_bytes(b"fixture")
            executable.chmod(0o700)
            link = root / "venv-python"
            link.symlink_to(executable)
            self.assertEqual(bootstrap.compiler_overrides([f"triton={link}"], ("triton",)), {"triton": link})
            with self.assertRaisesRegex(ValueError, "twice"):
                bootstrap.compiler_overrides([f"triton={link}", f"triton={link}"], ("triton",))
            with self.assertRaises(ValueError):
                bootstrap.compiler_overrides([f"other={link}"], ("triton",))

            tools_dir = root / "tools"
            tools_dir.mkdir()
            self.assertEqual(bootstrap.missing_build_tools({"PATH": str(tools_dir)}), ["cmake>=3.28", "ninja"])
            cmake = tools_dir / "cmake"
            cmake.write_text("#!/bin/sh\necho 'cmake version 3.20.0'\n")
            cmake.chmod(0o700)
            self.assertEqual(bootstrap.missing_build_tools({"PATH": str(tools_dir)}), ["cmake>=3.28", "ninja"])
            cmake.write_text("#!/bin/sh\necho 'cmake version 3.28.2'\n")
            self.assertEqual(bootstrap.missing_build_tools({"PATH": str(tools_dir)}), ["ninja"])

    def test_engine_config_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "site/ask/share/ask/adapters/triton/exporter.py"
            worker.parent.mkdir(parents=True)
            worker.write_text("# installed fixture\n")
            config = bootstrap.engine_config(
                root / "sdk", root / "ptxas",
                {"triton": root / "worker-env/bin/python"}, root / "site/ask",
                {"triton": {"worker": "exporter.py"}}, "all",
            )
            self.assertNotIn("ask_opt", config)
            self.assertEqual(config["compilers"]["triton"]["worker"], str(worker))
            target = bootstrap.write_config(root / "venv", config)
            self.assertEqual(target, root / "venv/share/fugatensor/engine.json")
            self.assertEqual(json.loads(target.read_text()), config)
            worker.unlink()
            with self.assertRaisesRegex(ValueError, "worker is missing"):
                bootstrap.engine_config(root / "sdk", None, {"triton": root / "python"},
                                        root / "site/ask", {"triton": {"worker": "exporter.py"}}, "all")

        none_cfg = bootstrap.engine_config(Path("/sdk"), None, {}, Path("/site/ask"), {}, "none")
        self.assertEqual(none_cfg["compilers"], {})
        self.assertIsNone(none_cfg["ptxas"])
        self.assertEqual(none_cfg["llc"], "/sdk/bin/llc")


if __name__ == "__main__":
    unittest.main()
