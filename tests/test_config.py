"""Engine configuration discovery, explicit isolation, and bootstrap CLI contracts."""
from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
import contextvars
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fugatensor as ft
from fugatensor._config import runtime_defaults
from fugatensor.devices import device as Device, normalize_device, resolve_device
from fugatensor.engine import Engine, configure, shutdown, current_engine

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

    def test_installed_triton_is_optional_and_yields_to_configured_workers(self):
        with patch("fugatensor.engine.Compiler"), patch("fugatensor.engine.ExecutionSession") as session, \
                patch("importlib.metadata.version", return_value="3.9.0") as version:
            with Engine(device="cuda") as engine:
                self.assertEqual(engine.compilers["triton"].python, "/saved/python")
            version.assert_not_called()
            self.path.write_text(json.dumps({"schema_version": 1, "compilers": {}}))
            with Engine(device="cuda") as engine:
                self.assertEqual(engine.compilers["triton"], ft.triton.environment(python=sys.executable))
            version.assert_called_once_with("triton")
            version.reset_mock()
            with Engine(device="cpu") as engine:
                self.assertEqual(engine.compilers, {})
            with Engine(device="cuda", compilers={}) as engine:
                self.assertEqual(engine.compilers, {})
            version.assert_not_called()
            with patch.dict(os.environ, {"FUGATENSOR_TRITON_PYTHON": "/env/python"}):
                with Engine(device="cuda") as engine:
                    self.assertEqual(engine.compilers["triton"].python, "/env/python")
            version.assert_not_called()
            version.side_effect = PackageNotFoundError("triton")
            with Engine(device="cuda") as engine:
                self.assertEqual(engine.compilers, {})
            session.assert_not_called()

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


class DeviceDescriptionTests(unittest.TestCase):
    def test_optional_index_identity_and_immutability(self):
        cpu, unspecified, indexed = Device("cpu"), Device("cuda"), Device("cuda", 0)
        self.assertEqual((cpu.type, cpu.index), ("cpu", None))
        self.assertEqual((unspecified.type, unspecified.index), ("cuda", None))
        self.assertEqual((indexed.type, indexed.index), ("cuda", 0))
        self.assertEqual(indexed, Device("cuda:0"))
        self.assertEqual(Device(indexed), indexed)
        self.assertNotEqual(unspecified, indexed)
        self.assertNotEqual(unspecified, "cuda")
        self.assertEqual({indexed: 1}[Device("cuda:0")], 1)
        self.assertEqual(str(indexed), "cuda:0")
        self.assertEqual(repr(cpu), "device(type='cpu')")
        self.assertEqual(repr(indexed), "device(type='cuda', index=0)")
        with self.assertRaises(AttributeError):
            indexed.index = 2
        self.assertEqual(resolve_device(unspecified), indexed)
        self.assertEqual(resolve_device(Device("cpu:0")), cpu)
        self.assertEqual(normalize_device(indexed), "cuda")
        self.assertEqual(normalize_device(Device("cuda", 2)), "cuda:2")

    def test_invalid_device_descriptions_are_rejected(self):
        for value in ("gpu", "cuda:-1", "cuda:", "cuda:one", "cpu:1", " cuda", "cuda:1:2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Device(value)
        for index in (True, 1.5, "0"):
            with self.subTest(index=index), self.assertRaises(TypeError):
                Device("cuda", index)
        with self.assertRaises(ValueError):
            Device("cuda", -1)
        with self.assertRaisesRegex(ValueError, "twice"):
            Device("cuda:1", 1)
        with self.assertRaises(TypeError):
            Device(Device("cuda"), 1)
        with self.assertRaises(TypeError):
            Device(0)


class DefaultDeviceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        environment = patch.dict(os.environ, {"FUGATENSOR_CACHE_DIR": str(self.directory / "cache")})
        environment.start()
        self.addCleanup(environment.stop)
        previous = ft.get_default_device()
        self.addCleanup(ft.set_default_device, previous)
        ft.set_default_device("cpu")
        ft.shutdown()
        self.addCleanup(ft.shutdown)

    def test_direct_tensor_calls_reuse_engine_and_support_autograd(self):
        x = ft.tensor(3.0, requires_grad=True)
        y = ft.tensor(2.0)
        self.assertEqual(x.dtype, "f32")
        self.assertEqual(ft.tensor([1, 2]).dtype, "f32")
        self.assertIs(x.engine, y.engine)
        with self.assertRaisesRegex(TypeError, "shape"):
            ft.tensor([1, 2], shape=(1, 2))
        result = x * x + y
        result.backward()
        self.assertEqual(result.tolist(), 11.0)
        self.assertEqual(x.grad.tolist(), 6.0)
        ft.set_default_device("cuda:0")
        self.assertEqual(ft.get_default_device(), Device("cuda:0"))
        self.assertFalse(x.engine.closed)
        self.assertEqual((x + 1).tolist(), 4.0)
        self.assertIs(ft.tensor(5, device=Device("cpu")).engine, x.engine)
        ft.shutdown(Device("cpu"))
        self.assertTrue(x.engine.closed)
        self.assertEqual(ft.get_default_device(), Device("cuda:0"))

    def test_selection_is_lazy_and_context_local(self):
        first, second = contextvars.Context(), contextvars.Context()
        with patch("fugatensor.engine.Engine") as engine:
            first.run(ft.set_default_device, Device("cuda", 2))
            second.run(ft.set_default_device, "cuda:0")
            self.assertEqual(first.run(ft.get_default_device), Device("cuda:2"))
            self.assertEqual(second.run(ft.get_default_device), Device("cuda:0"))
            self.assertEqual(ft.get_default_device(), Device("cpu"))
            with self.assertRaises(ValueError):
                first.run(ft.set_default_device, "gpu")
            self.assertEqual(first.run(ft.get_default_device), Device("cuda:2"))
            engine.assert_not_called()

    def test_explicit_engines_and_devices_override_selection(self):
        default = ft.tensor(1).engine
        ft.set_default_device("cuda:1")
        with Engine(Device("cpu")) as active:
            self.assertEqual(ft.get_default_device(), Device("cpu"))
            self.assertIs(ft.tensor(2).engine, active)
            self.assertIs(ft.tensor(3, engine=default).engine, default)
            with self.assertRaisesRegex(ValueError, "device differs"):
                ft.tensor(4, engine=active, device="cuda")
            with Engine(Device("cuda", 2), compilers={}) as cuda:
                self.assertEqual(ft.get_default_device(), Device("cuda:2"))
                self.assertIs(current_engine(), cuda)
                self.assertIs(current_engine(Device("cuda:2")), cuda)
                cpu_tensor = ft.tensor(5, device=Device("cpu"))
                self.assertIs(cpu_tensor.engine, default)
                with self.assertRaisesRegex(RuntimeError, "same device, got cuda:2 and cpu"):
                    cuda.evaluate(cpu_tensor)
                with self.assertRaisesRegex(ValueError, "different FugaTensor Engine"):
                    active.evaluate(cpu_tensor)
                self.assertIsNone(cuda._session)
            self.assertEqual(ft.get_default_device(), Device("cpu"))
        self.assertEqual(ft.get_default_device(), Device("cuda:1"))
        self.assertFalse(default.closed)
        configured = ft.configure(Device("cpu"))
        self.assertTrue(default.closed)
        self.assertFalse(configured.closed)
        self.assertEqual(ft.get_default_device(), Device("cuda:1"))

    def test_devices_are_static_fun_arguments_and_configuration(self):
        class Select(ft.fun):
            selected = Device("cuda")

            def forward(self, ctx, value, requested=Device("cuda")):
                return value + (1 if requested == self.selected else 2)

        operation = Select()
        value = ft.tensor(3.0)
        self.assertEqual(operation(value).tolist(), 4.0)
        self.assertEqual(operation(value, Device("cuda:0")).tolist(), 5.0)
        self.assertEqual(operation(value, "cuda").tolist(), 5.0)
        self.assertEqual(operation(value, Device("cuda")).tolist(), 4.0)
        operation.selected = Device("cuda:0")
        self.assertEqual(operation(value, Device("cuda:0")).tolist(), 4.0)
        operation.selected = (Device("cpu"), (2, 3))
        self.assertEqual(operation(value, (Device("cpu"), (2, 3))).tolist(), 4.0)
        self.assertEqual(operation(value, (Device("cpu"), (3, 2))).tolist(), 5.0)
        with self.assertRaises(TypeError):
            operation(value, [Device("cpu"), (2, 3)])

    def test_buffer_and_weight_factories_share_selected_engine(self):
        from array import array
        source = ft.from_buffer(array("i", [1, 2]), dtype="i32", shape=(2,))
        path = ft.save_weights(self.directory / "weights", {"x": source})
        restored = ft.load_weights(path)["x"]
        self.assertIs(restored.engine, source.engine)
        self.assertEqual(restored.tolist(), [1, 2])
        ft.set_default_device("cuda")
        with Engine("cpu") as active:
            imported = ft.from_buffer(source.to_buffer(), dtype="i32", shape=(2,))
            loaded = ft.load_weights(path)["x"]
            self.assertIs(imported.engine, active)
            self.assertIs(loaded.engine, active)
            self.assertEqual(loaded.tolist(), [1, 2])
        explicit = ft.load_weights(path, device="cpu")["x"]
        self.assertIs(explicit.engine, source.engine)

    def test_synchronize_selects_engine_without_materializing_graph(self):
        x = ft.tensor([1.0, 2.0])
        pending = x + 1
        before = x.engine.stats["executions"]
        with patch.object(x.engine, "synchronize", wraps=x.engine.synchronize) as synchronize:
            self.assertIsNone(ft.synchronize())
            synchronize.assert_called_once_with()
        self.assertEqual(x.engine.stats["executions"], before)
        with Engine("cuda", compilers={}) as active:
            with patch.object(active, "synchronize", wraps=active.synchronize) as synchronize:
                ft.synchronize()
                synchronize.assert_called_once_with()
            ft.synchronize("cpu")
            self.assertIsNone(active._session)
        self.assertEqual(pending.tolist(), [2.0, 3.0])

    def test_process_exit_closes_default_engine(self):
        marker = self.directory / "closed"
        script = '''
import atexit
from pathlib import Path
import sys
def check():
    Path(sys.argv[1]).write_text(str(x.engine.closed))
atexit.register(check)
import fugatensor as ft
x = ft.tensor([1, 2, 3])
assert (x + 1).tolist() == [2.0, 3.0, 4.0]
'''
        completed = subprocess.run([sys.executable, "-c", script, str(marker)],
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(marker.read_text(), "True")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "fuga"
        self.root.mkdir()
        (self.root / "pyproject.toml").write_text(
            '[project]\nversion = "9.8.7"\ndependencies = []\n'
            '[build-system]\nrequires = ["scikit-build-core==1.1.1", "nanobind==3.1.0"]\n')
        self.sdk = self.directory / "ask-sdk"
        for relative in ("lib/cmake/ASK/ASKConfig.cmake", "include/ask/Frontend/CompilerAPI.h",
                         "include/ask/Runtime/CUDA.h", "bin/ask-opt"):
            path = self.sdk / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.version_file = self.sdk / "lib/cmake/ASK/ASKConfigVersion.cmake"
        self.version_file.write_text('set(PACKAGE_VERSION "0.1.0")\n')
        (self.sdk / "bin/ask-opt").chmod(0o700)
        self.venv = self.directory / "venv"
        self.python = self.venv / "bin/python"
        self.config = self.venv / "share/fugatensor/engine.json"
        self.installed = {
            "version": "0.1.0", "prefix": str(self.sdk), "cmake_dir": str(self.sdk / "lib/cmake/ASK"),
            "capabilities": {"ask_version": "0.1.0", "host_codegen": True,
                             "cuda_graph_codegen": True,
                             "llvm_version": "23.1.4", "llvm_revision": ""},
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
        return bootstrap.parser().parse_args(["--venv", str(self.venv), "--ask-prefix", str(self.sdk), *extra])

    def execute(self, *extra):
        with redirect_stdout(io.StringIO()):
            return bootstrap.bootstrap(self.args(*extra))

    def test_build_uses_installed_sdk_without_ask_python_or_llvm_sdk(self):
        # Only an installed SDK is supplied; neither project's build reads an ASK checkout.
        with patch.object(bootstrap, "read_json", side_effect=AssertionError("unexpected source metadata read")), \
                patch.dict(os.environ, {"CMAKE_PREFIX_PATH": "/other-sdk"}):
            summary = self.execute()
        self.assertEqual(summary["ask_version"], "0.1.0")
        self.assertEqual(summary["ask_prefix"], str(self.sdk))
        self.assertIsNone(summary["llc"])
        self.assertIsNone(summary["ptxas"])
        calls = [list(map(str, call.args[0])) for call in self.mock_run.call_args_list]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], [str(self.python), "-m", "pip", "install",
                                   "--no-build-isolation",
                                   "--config-settings=cmake.define.ASK_DIR=" + self.installed["cmake_dir"],
                                   "--editable", f"{self.root}[pure]"])
        self.assertEqual(calls[1], [str(self.python), "-m", "pip", "check"])
        self.assertEqual(json.loads(self.config.read_text()),
                         {"schema_version": 1, "llc": None, "ptxas": None, "compilers": {}})
        self.mock_inspect_ask.assert_called_once_with(self.sdk)
        self.mock_installed_version.assert_not_called()
        environment = self.mock_run.call_args_list[0].kwargs["env"]
        self.assertEqual(environment["CMAKE_PREFIX_PATH"], str(self.sdk) + os.pathsep + "/other-sdk")

    def test_all_installs_optional_dependencies_through_pip(self):
        worker = self.root / "src/fugatensor/_triton_worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# installed fixture\n")
        self.mock_installed_version.side_effect = AssertionError("pip must install local Triton first")
        with patch.object(bootstrap, "executable_path", return_value=self.python):
            summary = self.execute("--variant", "all", "--compiler-python", f"triton={self.python}")
        command = list(map(str, self.mock_run.call_args_list[0].args[0]))
        self.assertNotIn("--no-deps", command)
        self.assertEqual(command[-2:], ["--editable", f"{self.root}[all]"])
        self.assertEqual(summary["variant"], "all")

    def test_missing_sdk_stops_before_creating_environment(self):
        self.mock_inspect_ask.side_effect = self.original_inspect
        for prefix in (self.directory / "missing", self.root):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "installed ASK SDK"):
                self.execute("--ask-prefix", str(prefix))
        self.mock_run.assert_not_called()
        self.mock_ensure_venv.assert_not_called()
        self.assertFalse(self.config.exists())

    def test_sdk_prefix_is_required_and_old_package_options_are_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bootstrap.parser().parse_args([])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--ask-wheel", "/removed.whl")
        self.mock_ensure_venv.assert_not_called()
        self.mock_run.assert_not_called()

    def test_fuga_build_failure_does_not_write_configuration(self):
        self.mock_run.side_effect = subprocess.CalledProcessError(1, ["pip", "install", "--editable"])
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        self.assertEqual(self.mock_run.call_count, 1)
        self.assertFalse(self.config.exists())

    def test_sdk_cli_and_cmake_metadata_versions_must_match(self):
        self.mock_subprocess.return_value.stdout = json.dumps(self.installed["capabilities"])
        self.assertEqual(self.original_inspect(self.sdk), self.installed)
        self.mock_subprocess.assert_called_once_with(
            [str(self.sdk / "bin/ask-opt"), "--capabilities"], capture_output=True, text=True, check=True)
        self.version_file.write_text('set(PACKAGE_VERSION "0.2.0")\n')
        with self.assertRaisesRegex(RuntimeError, "versions differ"):
            self.original_inspect(self.sdk)
        self.version_file.unlink()
        with self.assertRaisesRegex(ValueError, "version metadata"):
            self.original_inspect(self.sdk)

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

    def test_cuda_tool_version_matches_the_built_sdk_without_revision_headers(self):
        llvm = self.directory / "llvm"
        llc = llvm / "bin/llc"
        llc.parent.mkdir(parents=True)
        llc.write_text("#!/bin/sh\n")
        llc.chmod(0o700)
        self.mock_subprocess.return_value.stdout = "LLVM version 23.1.4\n  Optimized build.\n"
        self.execute("--llvm-root", str(llvm))
        self.assertEqual(json.loads(self.config.read_text())["llc"], str(llc))
        self.assertFalse((llvm / "include/llvm/Support/VCSRevision.h").exists())
        self.mock_run.reset_mock()
        for version in ("23.1.3", "24.1.0", "23.1.4git", "unknown"):
            self.mock_subprocess.return_value.stdout = f"LLVM version {version}\n"
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "match ASK LLVM 23.1.4"):
                self.execute("--llvm-root", str(llvm))
        self.mock_run.assert_not_called()

    def test_bundled_worker_needs_no_ask_registry_and_does_not_install_compiler(self):
        worker = self.root / "src/fugatensor/_triton_worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# installed fixture\n")
        (worker.parent / "triton.dependencies.lock.json").write_text(json.dumps({
            "compiler": {"name": "triton", "version": "3.9.0"}}))
        with patch.object(bootstrap, "executable_path", return_value=self.python):
            for version in ("3.9.0", "3.9.1"):
                self.mock_installed_version.return_value = version
                self.execute("--compiler-python", "triton=python")
            self.assertEqual(json.loads(self.config.read_text())["compilers"]["triton"]["worker"], str(worker))
            self.mock_run.reset_mock()
            for version in (None, "3.8.0", "3.10.0", "3.9.0rc1", "3.9.1+local"):
                for extra in ((), ("--compiler-python", "triton=python")):
                    self.mock_installed_version.return_value = version
                    with self.subTest(version=version, extra=extra), self.assertRaisesRegex(ValueError, "triton 3.9.x.*separately"):
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
