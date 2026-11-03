"""Plan contracts, portable packages and the single native submission boundary."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))
__import__('sys').path.insert(0, str(Path(__file__).resolve().parent))

from fugatensor._ask.cuda import CUDAError, install_external
from fugatensor._ask.gpu import CUDAKernel
from fugatensor._ask.plan import CUDAProgram, validate_plan
from fugatensor._ask.runtime import ExecutionSession
from support.compilation import matmul_plan
import test_cuda_compilation as compilation_fixtures
from test_runtime_and_cache import ArrayOwner, FakeEvent, FakeNativeSession


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


class ExecutionPlanTests(unittest.TestCase):
    def make_program(self, directory, plan=None):
        kernel, _ = compilation_fixtures.PreparedExecutableContractTests().make_kernel_package(directory)
        return CUDAProgram(plan or sample_plan(), {"first": kernel, "second": kernel})

    def test_library_package_and_static_contract(self):
        from itertools import product
        for dtype, batch, ta, tb in product(('f32', 'f16', 'bf16'), ((), (2,)), (False, True), (False, True)):
            with self.subTest(dtype=dtype, batch=batch, ta=ta, tb=tb), tempfile.TemporaryDirectory() as directory:
                plan = matmul_plan(dtype, batch=batch, transpose_a=ta, transpose_b=tb)
                program = CUDAProgram(plan, {})
                program.save(Path(directory) / 'package')
                shutil.move(Path(directory) / 'package', Path(directory) / 'relocated')
                loaded = CUDAProgram.load(Path(directory) / 'relocated')
                with session_fixture() as session:
                    loaded.prepare(session, graph=True)
                    native = session._native.programs[0]
                    self.assertEqual(native.steps, [{'inputs': [0, 1], 'outputs': [2],
                                                    'matmul': plan['steps'][0]['matmul']}])
                    self.assertEqual(loaded.kernels, {})
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
            with self.subTest(plan=plan), self.assertRaises(CUDAError):
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
            with self.subTest(plan=plan), self.assertRaises(CUDAError):
                validate_plan(plan)
        with tempfile.TemporaryDirectory() as directory:
            plan = sample_plan()
            plan["buffers"][1]["element_type"] = "f32"
            with self.assertRaisesRegex(CUDAError, "ABI"):
                self.make_program(Path(directory), plan)

    def test_package_relocation_digests_and_path_confinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = self.make_program(root)
            program.save(root / "package")
            shutil.move(root / "package", root / "relocated")
            loaded = CUDAProgram.load(root / "relocated")
            with session_fixture() as session:
                self.assertEqual(loaded.prepare(session).inputs[0]["shape"], [2, 5])
            manifest = root / "relocated/program.json"
            receipt = json.loads(manifest.read_text())
            original = copy.deepcopy(receipt)
            receipt["steps"]["second"]["files"]["cubin"]["path"] = "../k.cubin"
            manifest.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(CUDAError, "escapes"):
                CUDAProgram.load(root / "relocated")
            manifest.write_text(json.dumps(original))
            (root / "relocated/package/second/kernel.cubin").write_bytes(b"tampered")
            with self.assertRaisesRegex(CUDAError, "digest"):
                CUDAProgram.load(root / "relocated")

    def test_all_ports_checked_before_any_launch(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session, session_fixture() as other:
            executable = self.make_program(Path(directory)).prepare(session)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            for invalid in (other.tensor((2, 5), "i32"), session.tensor((10,), "i32")):
                with self.assertRaises(CUDAError):
                    executable.submit(x, invalid)
                self.assertEqual(session._native.events, [])
            owner = ArrayOwner(shape=(2, 5), typestr="<i4")
            borrowed = session.from_cuda_array(owner)
            owner.description["data"] = (4096, True)
            with self.assertRaisesRegex(CUDAError, "changed"):
                executable.submit(borrowed, z)
            self.assertEqual(session._native.events, [])

    def test_one_native_plan_submission_without_python_temporaries_or_hot_io(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session, graph=True)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            self.assertEqual(len(session._native.programs), 1)
            plan = session._native.programs[0]
            self.assertTrue(plan.graph)
            self.assertEqual(plan.buffer_sizes, [40, 40, 40])
            self.assertEqual((plan.inputs, plan.outputs), ([0], [2]))
            self.assertEqual([step["arguments"] for step in plan.steps], [[0, 1], [1, 2]])
            with patch.object(session, "tensor", side_effect=AssertionError("Python intermediate allocation")), \
                 patch.object(session, "submit", side_effect=AssertionError("per-kernel Python dispatch")), \
                 patch.object(Path, "read_bytes", side_effect=AssertionError("hot path I/O")):
                executable.submit(x, z)
            self.assertEqual(plan.submissions, [([x._buffer], [z._buffer])])
            self.assertEqual(len(session._native.events), 1)

    def test_native_failure_propagates_without_python_dispatch_fallback(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            with patch.object(session._native.programs[0], "submit", side_effect=CUDAError("device launch failed")), \
                 patch.object(session, "submit", side_effect=AssertionError("Python fallback")):
                with self.assertRaisesRegex(CUDAError, "device launch failed"):
                    executable.submit(x, z)

    def test_program_event_keeps_external_python_owner_alive(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session)
            owner = ArrayOwner(shape=(2, 5), typestr="<i4")
            tracked = weakref.ref(owner)
            x, z = session.from_cuda_array(owner), session.tensor((2, 5), "i32")
            event = executable.submit(x, z)
            del x, owner, executable, event
            gc.collect()
            self.assertIsNotNone(tracked())
            session._native.events[0].complete = True
            session.reap()
            gc.collect()
            self.assertIsNone(tracked())

    def test_submission_retains_port_snapshot_when_caller_lists_change_during_launch(self):
        for method in ("run", "submit"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory, session_fixture() as session:
                executable = self.make_program(Path(directory)).prepare(session)
                owners = [ArrayOwner(typestr="<i4", data=(4096 + 4096 * index, False))
                          for index in range(2 if method == "submit" else 1)]
                tracked = [weakref.ref(owner) for owner in owners]
                inputs = [session.from_cuda_array(owners[0])]
                outputs = [session.from_cuda_array(owners[1])] if method == "submit" else []
                del owners
                completion = FakeEvent()

                def launch(*buffers):
                    inputs.clear()
                    outputs.clear()
                    gc.collect()
                    self.assertTrue(all(owner() is not None for owner in tracked))
                    return ([session._native.allocate(40)], completion) if method == "run" else completion

                with patch.object(session._native.programs[0], method, create=True, side_effect=launch):
                    if method == "run":
                        executable.run(inputs)
                    else:
                        executable.submit_many(inputs, outputs)
                gc.collect()
                self.assertTrue(all(owner() is not None for owner in tracked))
                completion.complete = True
                session.reap()
                gc.collect()
                self.assertTrue(all(owner() is None for owner in tracked))

    def test_output_aliases_and_zero_step_identity(self):
        plan = sample_plan()
        plan["outputs"] = ["z", "z", "x"]
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory), plan).prepare(session)
            x, z = [session.tensor((2, 5), "i32") for _ in range(2)]
            executable.submit_many([x], [z, z, x]).wait()
            self.assertEqual(session._native.programs[0].outputs, [2, 2, 0])
            self.assertEqual(session._native.programs[0].submissions, [([x._buffer], [z._buffer, z._buffer, x._buffer])])
            identity = dict(plan, outputs=["x", "x"], steps=[], buffers=[plan["buffers"][0]])
            complete = CUDAProgram(identity, {}).prepare(session).submit_many([x], [x, x])
            self.assertEqual(session._native.programs[-1].steps, [])
            self.assertIs(complete.wait(), complete)
            self.assertTrue(complete.query())

    def test_external_formats_relocate_and_reject_tampered_packages(self):
        for format_ in ("cubin", "ptx"):
            with self.subTest(format=format_), tempfile.TemporaryDirectory() as directory:
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
                    self.assertEqual(len(session._native.events), 1)
                manifest = root / "relocated/program.json"
                original = json.loads(manifest.read_text())
                bad = copy.deepcopy(original)
                bad["steps"]["second"]["files"][format_]["path"] = "../outside"
                manifest.write_text(json.dumps(bad))
                with self.assertRaisesRegex(CUDAError, "escapes"):
                    CUDAProgram.load(root / "relocated")
                manifest.write_text(json.dumps(original))
                path = root / "relocated/package/second" / ("kernel." + format_)
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(CUDAError, "digest"):
                    CUDAProgram.load(root / "relocated")


if __name__ == "__main__":
    unittest.main()
