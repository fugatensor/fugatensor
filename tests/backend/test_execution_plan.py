"""Plan contracts, portable packages and the single native submission boundary."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("mock_cuda_ports")

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from unittest.mock import patch

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))

from fugatensor._ask.cuda import CUDAError, install_external
from fugatensor._ask.gpu import CUDAKernel
from fugatensor._ask.plan import CUDAProgram, validate_plan
from fugatensor._ask.runtime import ExecutionSession
from tests.backend.support.compilation import matmul_plan
from tests.backend import test_cuda_compilation as compilation_fixtures
from tests.backend.support.prepared import mock_cuda_ports
from tests.backend.test_runtime_and_cache import FakeNativeSession


def sample_plan():
    return {"schema_version": 1, "kind": "ask.cuda.execution_plan",
            "inputs": ["x"], "outputs": ["z"],
            "buffers": [{"id": name, "shape": [2, 5], "element_type": "i32"} for name in ("x", "y", "z")],
            "steps": [{"id": "first", "kind": "owned", "function": "first", "inputs": ["x"],
                       "outputs": ["y"], "depends_on": []},
                      {"id": "second", "kind": "owned", "function": "second", "inputs": ["y"],
                       "outputs": ["z"], "depends_on": ["first"]}],
            "scheduling": {"policy": "sequential", "stream": "session"}}


def session_fixture():
    session = ExecutionSession.__new__(ExecutionSession)
    session._lock, session._pending = threading.RLock(), []
    session._native = FakeNativeSession()
    session._native.load = lambda *args: object()
    return session


class TestExecutionPlan:
    def make_program(self, directory, plan=None):
        kernel, _ = compilation_fixtures.TestPreparedExecutableContract().make_kernel_package(directory)
        return CUDAProgram(plan or sample_plan(), {"first": kernel, "second": kernel})

    def test_library_package_and_static_contract(self):
        from itertools import product
        for dtype, batch, ta, tb in product(('f32', 'f16', 'bf16'), ((), (2,)), (False, True), (False, True)):
            with tempfile.TemporaryDirectory() as directory:
                plan = matmul_plan(dtype, batch=batch, transpose_a=ta, transpose_b=tb)
                program = CUDAProgram(plan, {})
                program.save(Path(directory) / 'package')
                shutil.move(Path(directory) / 'package', Path(directory) / 'relocated')
                loaded = CUDAProgram.load(Path(directory) / 'relocated')
                with session_fixture() as session:
                    loaded.prepare(session, graph=True)
                    native = session._native.programs[0]
                    assert (native.steps) == ([{'inputs': [0, 1], 'outputs': [2], 'matmul': plan['steps'][0]['matmul']}])
                    assert (loaded.kernels) == ({})
        mutations = (
            lambda p: p['steps'][0].pop('matmul'),
            lambda p: p['steps'][0].update(kind='owned'),
            lambda p: p['steps'][0].update(library='unknown'),
            lambda p: p['steps'][0]['matmul'].update(m=7),
            lambda p: p['steps'][0]['matmul'].update(transpose_a=1),
            lambda p: p['steps'][0]['matmul'].update(batch_count=-1),
            lambda p: p['buffers'][1].update(element_type='f16'),
        )
        for mutate in mutations:
            plan = matmul_plan()
            mutate(plan)
            with pytest.raises(CUDAError):
                validate_plan(plan)

    def test_schema_ssa_dependency_and_abi_rejections(self):
        mutations = [
            lambda p: p.update(schema_version=True), lambda p: p.update(kind="future"),
            lambda p: p.update(unrecognized=1),
            lambda p: p["scheduling"].update(stream="default"),
            lambda p: p["buffers"][0].update(shape=[True]),
            lambda p: p["buffers"][0].update(shape=[2**63]),
            lambda p: p["buffers"][0].update(element_type="f64"),
            lambda p: p["buffers"].append(copy.deepcopy(p["buffers"][0])),
            lambda p: p["steps"][0].update(id="../escape"),
            lambda p: p["steps"][0].update(inputs=["y"]),
            lambda p: p["steps"][0].update(outputs=["x"]),
            lambda p: p["steps"][1].update(depends_on=[]),
            lambda p: p["steps"][1].update(depends_on=["first", "first"]),
            lambda p: p.update(outputs=["undefined"]),
        ]
        for mutate in mutations:
            plan = sample_plan()
            mutate(plan)
            with pytest.raises(CUDAError):
                validate_plan(plan)
        with tempfile.TemporaryDirectory() as directory:
            plan = sample_plan()
            plan["buffers"][1]["element_type"] = "f32"
            with pytest.raises(CUDAError, match='ABI'):
                self.make_program(Path(directory), plan)

    def test_package_relocation_digests_and_path_confinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = self.make_program(root)
            program.save(root / "package")
            shutil.move(root / "package", root / "relocated")
            loaded = CUDAProgram.load(root / "relocated")
            with session_fixture() as session:
                assert (loaded.prepare(session).inputs[0]['shape']) == ([2, 5])
            manifest = root / "relocated/program.json"
            receipt = json.loads(manifest.read_text())
            original = copy.deepcopy(receipt)
            receipt["steps"]["second"]["files"]["cubin"]["path"] = "../k.cubin"
            manifest.write_text(json.dumps(receipt))
            with pytest.raises(CUDAError, match='escapes'):
                CUDAProgram.load(root / "relocated")
            manifest.write_text(json.dumps(original))
            (root / "relocated/package/second/kernel.cubin").write_bytes(b"tampered")
            with pytest.raises(CUDAError, match='digest'):
                CUDAProgram.load(root / "relocated")


    def test_one_native_plan_submission_without_python_temporaries_or_hot_io(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session, graph=True)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            assert (len(session._native.programs)) == (1)
            plan = session._native.programs[0]
            assert (plan.graph)
            assert (plan.buffer_sizes) == ([40, 40, 40])
            assert ((plan.inputs, plan.outputs)) == (([0], [2]))
            assert ([step['arguments'] for step in plan.steps]) == ([[0, 1], [1, 2]])
            with patch.object(session, "tensor", side_effect=AssertionError("Python intermediate allocation")), \
                 patch.object(session, "submit", side_effect=AssertionError("per-kernel Python dispatch")), \
                 patch.object(Path, "read_bytes", side_effect=AssertionError("hot path I/O")):
                executable.submit(x, z)
            executable._ports.submit.assert_called_once_with(plan, session, [x], [z], executable._tensor_type)

    def test_native_failure_propagates_without_python_dispatch_fallback(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            with patch.object(executable._ports, "submit", side_effect=CUDAError("device launch failed")), \
                 patch.object(session, "submit", side_effect=AssertionError("Python fallback")):
                with pytest.raises(CUDAError, match='device launch failed'):
                    executable.submit(x, z)




    def test_external_formats_relocate_and_reject_tampered_packages(self):
        for format_ in ("cubin", "ptx"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact = compilation_fixtures.make_external_artifact(format_)
                codegen = install_external(artifact, root / "source")
                kernel = CUDAKernel(artifact, codegen, {}, {})
                plan = sample_plan()
                for step in plan["steps"]:
                    step.update(kind="external", external_id=step["id"])
                program = CUDAProgram(plan, {"first": kernel, "second": kernel})
                program.save(root / "package")
                shutil.move(root / "package", root / "relocated")
                shutil.rmtree(root / "source")
                loaded = CUDAProgram.load(root / "relocated")
                with session_fixture() as session:
                    executable = loaded.prepare(session)
                    x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
                    executable.submit(x, z).wait()
                    assert executable._ports.submit.call_count == 1
                manifest = root / "relocated/program.json"
                original = json.loads(manifest.read_text())
                bad = copy.deepcopy(original)
                bad["steps"]["second"]["files"][format_]["path"] = "../outside"
                manifest.write_text(json.dumps(bad))
                with pytest.raises(CUDAError, match='escapes'):
                    CUDAProgram.load(root / "relocated")
                manifest.write_text(json.dumps(original))
                path = root / "relocated/package/second" / ("kernel." + format_)
                path.write_bytes(path.read_bytes() + b"changed")
                with pytest.raises(CUDAError, match='digest'):
                    CUDAProgram.load(root / "relocated")
