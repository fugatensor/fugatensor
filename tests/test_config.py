"""Engine configuration discovery and explicit isolation contracts."""
from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from importlib.metadata import PackageNotFoundError
import pytest

import fugatensor as ft
from fugatensor._config import runtime_defaults
from fugatensor.devices import device as Device, normalize_device, resolve_device
from fugatensor.engine import Engine, configure, shutdown, current_engine

class TestEngineConfig:
    @pytest.fixture(autouse=True)
    def configured(self, cleanup):
        temporary = tempfile.TemporaryDirectory()
        cleanup.callback(temporary.cleanup)
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
        cleanup.callback(environment.stop)
        cleanup.callback(shutdown)

    def test_saved_paths_need_no_environment_exports_or_gpu_probe(self):
        with patch("fugatensor.engine.Compiler"), patch("fugatensor.engine.ExecutionSession") as session:
            with Engine(device="cuda") as engine:
                assert ((engine.llc, engine.ptxas)) == (('/saved/llc', '/saved/ptxas'))
                assert (engine.compilers['triton'].worker) == ('/saved/worker.py')
                session.assert_not_called()
        assert (runtime_defaults('cpu')) == ({})

    def test_environment_and_explicit_options_override_saved_values(self):
        with patch("fugatensor.engine.Compiler"), patch.dict(os.environ, {"FUGATENSOR_LLC": "/env/llc"}):
            with Engine(device="cuda", ptxas="/explicit/ptxas", compilers={}) as engine:
                assert (engine.llc) == ('/env/llc')
                assert (engine.ptxas) == ('/explicit/ptxas')
                assert (engine.compilers) == ({})
            engine = configure(device="cuda", llc="/configured/llc", compilers={})
            assert (engine.llc) == ('/configured/llc')
            assert (engine.ptxas) == ('/saved/ptxas')

    def test_installed_triton_is_optional_and_yields_to_configured_workers(self):
        with patch("fugatensor.engine.Compiler"), patch("fugatensor.engine.ExecutionSession") as session, \
                patch("importlib.metadata.version", return_value="3.9.0") as version:
            with Engine(device="cuda") as engine:
                assert (engine.compilers['triton'].python) == ('/saved/python')
            version.assert_not_called()
            self.path.write_text(json.dumps({"schema_version": 1, "compilers": {}}))
            with Engine(device="cuda") as engine:
                assert (engine.compilers['triton']) == (ft.triton.environment(python=sys.executable))
            version.assert_called_once_with("triton")
            version.reset_mock()
            with Engine(device="cpu") as engine:
                assert (engine.compilers) == ({})
            with Engine(device="cuda", compilers={}) as engine:
                assert (engine.compilers) == ({})
            version.assert_not_called()
            with patch.dict(os.environ, {"FUGATENSOR_TRITON_PYTHON": "/env/python"}):
                with Engine(device="cuda") as engine:
                    assert (engine.compilers['triton'].python) == ('/env/python')
            version.assert_not_called()
            version.side_effect = PackageNotFoundError("triton")
            with Engine(device="cuda") as engine:
                assert (engine.compilers) == ({})
            session.assert_not_called()

    def test_explicit_or_env_ask_opt_isolates_from_saved_workers(self):
        self.path.write_text("broken stale configuration")
        with patch("fugatensor.engine.Compiler") as compiler:
            with Engine(device="cuda", ask_opt="/other/ask-opt") as engine:
                assert (engine.llc) is None
                assert (engine.compilers) == ({})
                compiler.assert_called_with(executable="/other/ask-opt")
            engine = configure(device="cuda", ask_opt="/other/ask-opt", ptxas="/other/ptxas")
            assert (engine.llc) is None
            assert (engine.compilers) == ({})
        with patch("fugatensor.engine.Compiler") as compiler, patch.dict(os.environ, {"ASK_OPT": "/env/ask-opt"}):
            with Engine(device="cuda") as engine:
                assert (engine.llc) is None
                assert (engine.compilers) == ({})
                compiler.assert_called_with(executable="/env/ask-opt")

    def test_invalid_configuration_reports_clear_error(self):
        for content in (
            "invalid json",
            json.dumps({"schema_version": True}),
            json.dumps({"schema_version": 1, "llc": []}),
            json.dumps({"schema_version": 1, "compilers": []}),
        ):
            self.path.write_text(content)
            with pytest.raises(ValueError):
                runtime_defaults("cuda")
        self.path.unlink()
        with pytest.raises(ValueError, match='does not name'):
            runtime_defaults("cuda")


class TestDeviceDescription:
    def test_optional_index_identity_and_immutability(self):
        cpu, unspecified, indexed = Device("cpu"), Device("cuda"), Device("cuda", 0)
        assert ((cpu.type, cpu.index)) == (('cpu', None))
        assert ((unspecified.type, unspecified.index)) == (('cuda', None))
        assert ((indexed.type, indexed.index)) == (('cuda', 0))
        assert (indexed) == (Device('cuda:0'))
        assert (Device(indexed)) == (indexed)
        assert (unspecified) != (indexed)
        assert (unspecified) != ('cuda')
        assert ({indexed: 1}[Device('cuda:0')]) == (1)
        assert (str(indexed)) == ('cuda:0')
        assert (repr(cpu)) == ("device(type='cpu')")
        assert (repr(indexed)) == ("device(type='cuda', index=0)")
        with pytest.raises(AttributeError):
            indexed.index = 2
        assert (resolve_device(unspecified)) == (indexed)
        assert (resolve_device(Device('cpu:0'))) == (cpu)
        assert (normalize_device(indexed)) == ('cuda')
        assert (normalize_device(Device('cuda', 2))) == ('cuda:2')

    def test_invalid_device_descriptions_are_rejected(self):
        for value in ("gpu", "cuda:-1", "cuda:", "cuda:one", "cpu:1", " cuda", "cuda:1:2"):
            with pytest.raises(ValueError):
                Device(value)
        for index in (True, 1.5, "0"):
            with pytest.raises(TypeError):
                Device("cuda", index)
        with pytest.raises(ValueError):
            Device("cuda", -1)
        with pytest.raises(ValueError, match='twice'):
            Device("cuda:1", 1)
        with pytest.raises(TypeError):
            Device(Device("cuda"), 1)
        with pytest.raises(TypeError):
            Device(0)


class TestDefaultDevice:
    @pytest.fixture(autouse=True)
    def configured(self, cleanup):
        temporary = tempfile.TemporaryDirectory()
        cleanup.callback(temporary.cleanup)
        self.directory = Path(temporary.name)
        environment = patch.dict(os.environ, {"FUGATENSOR_CACHE_DIR": str(self.directory / "cache")})
        environment.start()
        cleanup.callback(environment.stop)
        previous = ft.get_default_device()
        cleanup.callback(ft.set_default_device, previous)
        ft.set_default_device("cpu")
        ft.shutdown()
        cleanup.callback(ft.shutdown)

    def test_direct_tensor_calls_reuse_engine_and_support_autograd(self):
        x = ft.tensor(3.0)
        y = ft.tensor(2.0)
        assert (x.dtype) == ('f32')
        assert (ft.tensor([1, 2]).dtype) == ('f32')
        assert (x.engine) is (y.engine)
        with pytest.raises(TypeError, match='shape'):
            ft.tensor([1, 2], shape=(1, 2))
        result = x * x + y
        derivative = ft.grad(result, x)
        assert (result.tolist()) == (11.0)
        assert (derivative.tolist()) == (6.0)
        ft.set_default_device("cuda:0")
        assert (ft.get_default_device()) == (Device('cuda:0'))
        assert not (x.engine.closed)
        assert ((x + 1).tolist()) == (4.0)
        assert (ft.tensor(5, device=Device('cpu')).engine) is (x.engine)
        ft.shutdown(Device("cpu"))
        assert (x.engine.closed)
        assert (ft.get_default_device()) == (Device('cuda:0'))

    def test_selection_is_lazy_and_context_local(self):
        first, second = contextvars.Context(), contextvars.Context()
        with patch("fugatensor.engine.Engine") as engine:
            first.run(ft.set_default_device, Device("cuda", 2))
            second.run(ft.set_default_device, "cuda:0")
            assert (first.run(ft.get_default_device)) == (Device('cuda:2'))
            assert (second.run(ft.get_default_device)) == (Device('cuda:0'))
            assert (ft.get_default_device()) == (Device('cpu'))
            with pytest.raises(ValueError):
                first.run(ft.set_default_device, "gpu")
            assert (first.run(ft.get_default_device)) == (Device('cuda:2'))
            engine.assert_not_called()

    def test_explicit_engines_and_devices_override_selection(self):
        default = ft.tensor(1).engine
        ft.set_default_device("cuda:1")
        with Engine(Device("cpu")) as active:
            assert (ft.get_default_device()) == (Device('cpu'))
            assert (ft.tensor(2).engine) is (active)
            assert (ft.tensor(3, engine=default).engine) is (default)
            with pytest.raises(ValueError, match='device differs'):
                ft.tensor(4, engine=active, device="cuda")
            with Engine(Device("cuda", 2), compilers={}) as cuda:
                assert (ft.get_default_device()) == (Device('cuda:2'))
                assert (current_engine()) is (cuda)
                assert (current_engine(Device('cuda:2'))) is (cuda)
                cpu_tensor = ft.tensor(5, device=Device("cpu"))
                assert (cpu_tensor.engine) is (default)
                with pytest.raises(RuntimeError, match='same device, got cuda:2 and cpu'):
                    cuda.evaluate(cpu_tensor)
                with pytest.raises(ValueError, match='different FugaTensor Engine'):
                    active.evaluate(cpu_tensor)
                assert (cuda._session) is None
            assert (ft.get_default_device()) == (Device('cpu'))
        assert (ft.get_default_device()) == (Device('cuda:1'))
        assert not (default.closed)
        configured = ft.configure(Device("cpu"))
        assert (default.closed)
        assert not (configured.closed)
        assert (ft.get_default_device()) == (Device('cuda:1'))

    def test_devices_are_static_fun_arguments_and_configuration(self):
        class Select(ft.fun):
            selected = Device("cuda")

            def forward(self, value, requested=Device("cuda")):
                return value + (1 if requested == self.selected else 2)

        operation = Select()
        value = ft.tensor(3.0)
        assert (operation(value).tolist()) == (4.0)
        assert (operation(value, Device('cuda:0')).tolist()) == (5.0)
        assert (operation(value, 'cuda').tolist()) == (5.0)
        assert (operation(value, Device('cuda')).tolist()) == (4.0)
        operation.selected = Device("cuda:0")
        assert (operation(value, Device('cuda:0')).tolist()) == (4.0)
        operation.selected = (Device("cpu"), (2, 3))
        assert (operation(value, (Device('cpu'), (2, 3))).tolist()) == (4.0)
        assert (operation(value, (Device('cpu'), (3, 2))).tolist()) == (5.0)
        with pytest.raises(TypeError):
            operation(value, [Device("cpu"), (2, 3)])

    def test_buffer_and_weight_factories_share_selected_engine(self):
        from array import array
        source = ft.from_buffer(array("i", [1, 2]), dtype="i32", shape=(2,))
        path = ft.save_weights(self.directory / "weights", {"x": source})
        restored = ft.load_weights(path)["x"]
        assert (restored.engine) is (source.engine)
        assert (restored.tolist()) == ([1, 2])
        ft.set_default_device("cuda")
        with Engine("cpu") as active:
            imported = ft.from_buffer(source.to_buffer(), dtype="i32", shape=(2,))
            loaded = ft.load_weights(path)["x"]
            assert (imported.engine) is (active)
            assert (loaded.engine) is (active)
            assert (loaded.tolist()) == ([1, 2])
        explicit = ft.load_weights(path, device="cpu")["x"]
        assert (explicit.engine) is (source.engine)

    def test_synchronize_selects_engine_without_materializing_graph(self):
        x = ft.tensor([1.0, 2.0])
        pending = x + 1
        before = x.engine.stats["executions"]
        with patch.object(x.engine, "synchronize", wraps=x.engine.synchronize) as synchronize:
            assert (ft.synchronize()) is None
            synchronize.assert_called_once_with()
        assert (x.engine.stats['executions']) == (before)
        with Engine("cuda", compilers={}) as active:
            with patch.object(active, "synchronize", wraps=active.synchronize) as synchronize:
                ft.synchronize()
                synchronize.assert_called_once_with()
            ft.synchronize("cpu")
            assert (active._session) is None
        assert (pending.tolist()) == ([2.0, 3.0])

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
        assert (completed.returncode) == (0), completed.stderr
        assert (completed.stderr) == ('')
        assert (marker.read_text()) == ('True')

