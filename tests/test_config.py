"""Engine configuration discovery, explicit isolation, and bootstrap CLI contracts."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
import zipfile
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
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "fuga"
        self.root.mkdir()
        (self.root / "pyproject.toml").write_text(
            '[project]\nversion = "9.8.7"\ndependencies = ["ask-compiler==0.1.0"]\n'
            '[build-system]\nrequires = ["scikit-build-core==1.1.1", "nanobind==3.1.0"]\n')
        self.venv = self.directory / "venv"
        self.python = self.venv / "bin/python"
        self.package = self.venv / "lib/site-packages/ask"
        self.config = self.venv / "share/fugatensor/engine.json"
        self.installed = {
            "version": "0.1.0", "package": str(self.package),
            "capabilities": {"ask_version": "0.1.0", "host_codegen": True,
                             "cuda_graph_codegen": True,
                             "llvm_revision": "locked-revision"},
        }
        self.original_inspect = bootstrap.inspect_ask
        self.original_missing_tools = bootstrap.missing_build_tools
        for name, values in (
            ("ROOT", {"new": self.root}),
            ("ensure_venv", {"return_value": self.python}),
            ("installed_version", {"return_value": "0.1.0"}),
            ("missing_build_tools", {"return_value": []}),
            ("unsatisfied_requirements", {"return_value": []}),
            ("inspect_ask", {"return_value": self.installed}),
            ("run", {}),
        ):
            patcher = patch.object(bootstrap, name, **values)
            value = patcher.start()
            self.addCleanup(patcher.stop)
            setattr(self, "mock_" + name, value)
        patcher = patch.object(bootstrap.subprocess, "run", return_value=SimpleNamespace(returncode=0))
        self.mock_subprocess = patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *extra):
        return bootstrap.parser().parse_args(["--venv", str(self.venv), *extra])

    def execute(self, *extra):
        with redirect_stdout(io.StringIO()):
            return bootstrap.bootstrap(self.args(*extra))

    def wheel(self, name="ask-compiler", version="0.1.0"):
        path = self.directory / "ask_compiler-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("ask_compiler-0.1.0.dist-info/METADATA",
                             f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        return path

    def test_default_reuses_installed_ask_and_builds_only_fuga_without_sdk(self):
        # This fixture deliberately has no neighboring ASK checkout or SDK.
        with patch.object(bootstrap, "read_json", side_effect=AssertionError("unexpected source metadata read")):
            summary = self.execute()
        self.assertEqual(summary["ask_version"], "0.1.0")
        self.assertIsNone(summary["llc"])
        self.assertIsNone(summary["ptxas"])
        calls = [list(map(str, call.args[0])) for call in self.mock_run.call_args_list]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], [str(self.python), "-m", "pip", "install", "--no-deps",
                                   "--no-build-isolation", "--editable", str(self.root)])
        self.assertEqual(calls[1], [str(self.python), "-m", "pip", "check"])
        self.assertEqual(json.loads(self.config.read_text()),
                         {"schema_version": 1, "llc": None, "ptxas": None, "compilers": {}})
        self.mock_inspect_ask.assert_called_once_with(self.python, "0.1.0")

    def test_missing_or_wrong_ask_stops_before_building_fuga(self):
        for actual in (None, "0.0.1", "9.8.7"):
            with self.subTest(actual=actual):
                self.mock_installed_version.return_value = actual
                with self.assertRaisesRegex(ValueError, "ask-compiler==0.1.0.*Build ASK separately"):
                    self.execute()
        self.mock_run.assert_not_called()
        self.mock_inspect_ask.assert_not_called()
        self.assertFalse(self.config.exists())

    def test_old_combined_build_options_are_rejected(self):
        for values in (("--ask-root", "/other/ask"), ("--compilers", "none"), ("--skip-ask",)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "removed combined build"):
                self.execute(*values)
        self.mock_ensure_venv.assert_not_called()
        self.mock_run.assert_not_called()

    def test_local_wheel_is_installed_before_checking_and_building_fuga(self):
        wheel = self.wheel()
        self.mock_installed_version.return_value = "0.0.1"  # A provided wheel can replace an old ASK.
        self.execute("--ask-wheel", str(wheel))
        commands = [list(map(str, call.args[0])) for call in self.mock_run.call_args_list]
        self.assertEqual(commands[0], [str(self.python), "-m", "pip", "install", "--no-deps",
                                      "--force-reinstall", str(wheel)])
        self.assertIn("--editable", commands[1])
        self.mock_inspect_ask.assert_called_once_with(self.python, "0.1.0")

    def test_wheel_install_failure_stops_without_fuga_or_configuration(self):
        wheel = self.wheel()
        self.mock_run.side_effect = subprocess.CalledProcessError(1, ["pip", "install", str(wheel)])
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute("--ask-wheel", str(wheel))
        self.assertEqual(self.mock_run.call_count, 1)
        self.mock_inspect_ask.assert_not_called()
        self.assertFalse(self.config.exists())

    def test_ask_wheel_metadata_and_local_file_are_required(self):
        for name, version in (("other-package", "0.1.0"), ("ask-compiler", "0.2.0")):
            with self.subTest(name=name, version=version), self.assertRaisesRegex(ValueError, "ask-compiler==0.1.0"):
                self.execute("--ask-wheel", str(self.wheel(name, version)))
        for path in (self.directory / "missing.whl", self.root):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "existing local .whl"):
                self.execute("--ask-wheel", str(path))
        self.mock_run.assert_not_called()
        self.mock_ensure_venv.assert_not_called()

    def test_installed_python_native_and_metadata_versions_must_match(self):
        with patch.object(bootstrap, "python_json", return_value=self.installed):
            self.assertEqual(self.original_inspect(self.python, "0.1.0"), self.installed)
            self.installed["capabilities"]["ask_version"] = "0.0.1"
            with self.assertRaisesRegex(RuntimeError, "versions differ"):
                self.original_inspect(self.python, "0.1.0")

    def test_saved_gpu_paths_survive_rebuild_and_reset_is_explicit(self):
        original = {"schema_version": 1, "llc": "/saved/llc", "ptxas": "/saved/ptxas", "compilers": {}}
        bootstrap.write_config(self.venv, original)
        before = self.config.read_bytes()
        summary = self.execute()
        self.assertEqual(summary["llc"], "/saved/llc")
        self.assertEqual(self.config.read_bytes(), before)
        self.execute("--reset-config")
        self.assertIsNone(json.loads(self.config.read_text())["llc"])

    def test_missing_saved_worker_requires_explicit_configuration(self):
        bootstrap.write_config(self.venv, {"schema_version": 1, "compilers": {
            "triton": {"python": "/saved/python", "worker": "/saved/worker.py"}}})
        before = self.config.read_bytes()
        with self.assertRaisesRegex(ValueError, "worker is missing.*reset-config"):
            self.execute()
        self.assertEqual(self.config.read_bytes(), before)
        self.mock_run.assert_not_called()

    def test_bundled_worker_needs_no_ask_registry_and_does_not_install_compiler(self):
        worker = self.root / "src/fugatensor/_triton_worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# installed fixture\n")
        (worker.parent / "triton.dependencies.lock.json").write_text(json.dumps({
            "compiler": {"name": "triton", "version": "3.7.1"}}))
        with patch.object(bootstrap, "executable_path", return_value=self.python):
            self.mock_installed_version.side_effect = ["0.1.0", "3.7.1"]
            self.execute("--compiler-python", "triton=python")
            self.assertEqual(json.loads(self.config.read_text())["compilers"]["triton"]["worker"], str(worker))
            self.mock_run.reset_mock()
            for extra in ((), ("--compiler-python", "triton=python")):
                self.mock_installed_version.side_effect = ["0.1.0", None]
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "triton==3.7.1.*separately"):
                    self.execute(*extra)
            self.mock_run.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "executable fixtures need POSIX")
    def test_missing_and_old_build_tools_are_detected(self):
        tools = self.directory / "tools"
        tools.mkdir()
        self.assertEqual(self.original_missing_tools({"PATH": str(tools)}), ["cmake>=3.28", "ninja"])
        cmake = tools / "cmake"
        cmake.write_text("#!/bin/sh\n")
        cmake.chmod(0o700)
        self.mock_subprocess.return_value.stdout = "cmake version 3.20.0"
        self.assertEqual(self.original_missing_tools({"PATH": str(tools)}), ["cmake>=3.28", "ninja"])
        self.mock_subprocess.return_value.stdout = "cmake version 3.28.2"
        self.assertEqual(self.original_missing_tools({"PATH": str(tools)}), ["ninja"])

    @unittest.skipUnless(os.name == "posix", "executable symlinks need POSIX")
    def test_worker_python_symlink_is_not_resolved_into_base_environment(self):
        executable = self.directory / "base-python"
        executable.write_bytes(b"fixture")
        executable.chmod(0o700)
        link = self.directory / "venv-python"
        link.symlink_to(executable)
        self.assertEqual(bootstrap.compiler_overrides([f"triton={link}"], ("triton",)), {"triton": link})
        with self.assertRaisesRegex(ValueError, "twice"):
            bootstrap.compiler_overrides([f"triton={link}", f"triton={link}"], ("triton",))
        with self.assertRaises(ValueError):
            bootstrap.compiler_overrides([f"other={link}"], ("triton",))

    def test_engine_config_uses_fuga_worker_and_writes_atomically(self):
        worker = self.root / "src/fugatensor/_triton_worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# installed fixture\n")
        config = bootstrap.engine_config(
            self.directory / "sdk", self.directory / "ptxas", {"triton": self.python}, worker)
        self.assertNotIn("ask_opt", config)
        self.assertEqual(config["compilers"]["triton"]["worker"], str(worker))
        target = bootstrap.write_config(self.venv, config)
        self.assertEqual(json.loads(target.read_text()), config)
        self.assertEqual(list(target.parent.iterdir()), [target])
        worker.unlink()
        with self.assertRaisesRegex(ValueError, "worker is missing"):
            bootstrap.engine_config(None, None, {"triton": self.python}, worker)


if __name__ == "__main__":
    unittest.main()
