"""CUDA Array Interface ownership, ExecutionSession lifecycle, IRCache, and executable cache tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))
__import__('sys').path.insert(0, str(Path(__file__).resolve().parent))

import fugatensor._ask as ask
from fugatensor._ask.cache import IRCache, fingerprint_file
from fugatensor._ask.cuda import CUDAError, compile_cuda
from fugatensor._ask.runtime import ExecutionSession, _cuda_array_description


def sample_cache_key() -> dict:
    return {
        "schema_version": 1, "pipeline_version": "gpu-llvm-v1",
        "graph": {"source": "P -> F -> Q"}, "dsl": "def F(t): return t * 2",
        "config": {"BLOCK": 128}, "llvm_revision": "llvm-commit",
        "producer": {"triton": "3.7.1"}, "adapter": {"revision": "adapter-commit"},
        "tools": {"ptxas": "tool-sha256"}, "target": {"arch": "sm_80"},
        "shape": [10], "numerics": {"overflow": "wrap"},
    }


def sample_artifact() -> dict:
    return {
        "format": "llvm-ir", "entry": "scale2", "payload": "define void @fixture() { ret void }\n",
        "target": {"backend": "cuda", "arch": 80},
        "port_map": [{"argument": i, "logical_port": b, "element_type": "i32",
                      "shape": [10], "layout": "contiguous_row_major"}
                     for i, b in enumerate(("input:0", "output:0"))],
        "invocation": {"kind": "cuda-kernel", "version": 2, "grid": [1, 1, 1], "block": [128, 1, 1],
                       "shared_bytes": 0,
                       "arguments": [{"argument": i, "binding": b, "llvm_type": "ptr addrspace(1)"}
                                     for i, b in enumerate(("input:0", "output:0"))]},
        "codegen": {"fp_contract": "fast", "ptx_version": 80},
        "required_element_types": ["i32"],
    }


def _worker_process(root: str, start, output) -> None:
    try:
        if not start.wait(10):
            raise RuntimeError("start timeout")
        def build():
            with (Path(root) / "builders.log").open("ab", buffering=0) as s:
                s.write(b"builder\n")
            time.sleep(0.05)
            return {"module.ll": b"define void @kernel() { ret void }"}
        artifacts, stats = IRCache(root).get_or_create(sample_cache_key(), build)
        output.put({"artifacts": artifacts, "stats": stats})
    except BaseException as exc:
        output.put({"error": repr(exc)})


def _crashing_builder(root: str, entered) -> None:
    def build():
        entered.set()
        os._exit(7)
    IRCache(root).get_or_create(sample_cache_key(), build)


class ArrayOwner:
    def __init__(self, **changes):
        self.description = {
            "version": 3, "shape": (2, 5), "typestr": "<f4",
            "data": (4096, False), "strides": None, "stream": 7,
        }
        self.description.update(changes)
        self.reads = 0

    @property
    def __cuda_array_interface__(self):
        self.reads += 1
        return copy.deepcopy(self.description)


class FakeEvent:
    def __init__(self):
        self.complete = False

    def query(self):
        return self.complete

    def wait(self):
        self.complete = True


class FakeBuffer:
    def __init__(self, pointer, nbytes, readonly=False):
        self.data_ptr, self.nbytes, self.readonly = pointer, nbytes, readonly


class FakeNativeProgram:
    """Record the Python/native boundary without reimplementing the scheduler."""
    def __init__(self, session, buffer_sizes, inputs, outputs, steps, graph):
        self.session = session
        self.buffer_sizes, self.inputs, self.outputs = buffer_sizes, inputs, outputs
        self.steps, self.graph = steps, graph
        self.submissions = []

    def submit(self, inputs, outputs):
        self.submissions.append((list(inputs), list(outputs)))
        event = FakeEvent()
        self.session.events.append(event)
        return event


class FakeNativeSession:
    def __init__(self):
        self.closed = False
        self.stream = 99
        self.device = 0
        self.capability = (8, 6)
        self.stats = {}
        self.log = []
        self.events = []
        self.programs = []
        self.next_pointer = 8192

    def check_types(self, types):
        self.log.append(("check_types", types))

    def allocate(self, nbytes):
        ptr = self.next_pointer
        self.next_pointer += max(nbytes, 64)
        return FakeBuffer(ptr, nbytes)

    def import_buffer(self, pointer, nbytes, readonly):
        self.log.append(("import", pointer, nbytes, readonly))
        return FakeBuffer(pointer, nbytes, readonly)

    def wait_for_external_stream(self, stream):
        self.log.append(("wait", stream))

    def submit(self, kernel, buffers, grid, block, shared_bytes):
        self.log.append(("submit", grid, block))
        ev = FakeEvent()
        self.events.append(ev)
        return ev

    def prepare_program(self, buffer_sizes, inputs, outputs, steps, graph=False):
        program = FakeNativeProgram(self, buffer_sizes, inputs, outputs, steps, graph)
        self.programs.append(program)
        return program

    def reap(self):
        pass

    def synchronize(self):
        for ev in self.events:
            ev.complete = True

    def close(self):
        self.synchronize()
        self.closed = True


class RuntimeOwnershipTests(unittest.TestCase):
    def prepare_identity(self, session):
        return ask.CUDAProgram({
            "schema_version": 1, "kind": "ask.cuda.execution_plan",
            "inputs": ["input"], "outputs": ["input", "input"],
            "buffers": [{"id": "input", "shape": [2, 5], "element_type": "f32"}],
            "steps": [], "scheduling": {"policy": "sequential", "stream": "session"},
        }, {}).prepare(session)

    def test_prepared_run_preserves_aliases_and_retains_external_owner_until_completion(self):
        native = FakeNativeSession()
        ext = SimpleNamespace(CUDASession=SimpleNamespace(create=lambda device: native))
        with patch.object(ask, "_native", ext, create=True), ExecutionSession() as session:
            prepared = self.prepare_identity(session)
            owner = ArrayOwner()
            weak_owner = weakref.ref(owner)
            borrowed = session.from_cuda_array(owner)
            owner.description["stream"] = 19
            completion = FakeEvent()
            with patch.object(native.programs[0], "run", create=True,
                              return_value=([borrowed._buffer, borrowed._buffer], completion)) as run:
                outputs = prepared.run([borrowed])
                run.assert_called_once_with([borrowed._buffer])
            self.assertIs(outputs[0], borrowed)
            self.assertIs(outputs[1], borrowed)
            self.assertIn(("wait", 19), native.log)
            del outputs, owner, borrowed
            gc.collect()
            self.assertIsNotNone(weak_owner())
            completion.complete = True
            session.reap()
            gc.collect()
            self.assertIsNone(weak_owner())

    def test_prepared_entrypoints_reject_invalid_ports_before_native_submission(self):
        native = FakeNativeSession()
        ext = SimpleNamespace(CUDASession=SimpleNamespace(create=lambda device: native))
        with patch.object(ask, "_native", ext, create=True), ExecutionSession() as session:
            prepared = self.prepare_identity(session)
            borrowed = session.from_cuda_array(ArrayOwner())
            wrong_shape = session.tensor((10,), "f32")
            wrong_dtype = session.tensor((2, 5), "i32")
            before = list(native.log)
            with patch.object(native.programs[0], "run", create=True) as run, \
                 patch.object(native.programs[0], "submit") as submit:
                for values in (None, [], [object()], [wrong_shape], [wrong_dtype]):
                    with self.subTest(inputs=values):
                        with self.assertRaises(CUDAError):
                            prepared.run(values)
                        with self.assertRaises(CUDAError):
                            prepared.submit_many(values, [borrowed, borrowed])
                for values in (None, [], [borrowed], [borrowed, object()],
                               [borrowed, wrong_shape], [borrowed, wrong_dtype]):
                    with self.subTest(outputs=values), self.assertRaises(CUDAError):
                        prepared.submit_many([borrowed], values)
                self.assertEqual(native.log, before, "invalid ports must be rejected before external stream waits")
                with ExecutionSession() as other:
                    foreign = other.tensor((2, 5), "f32")
                    for call in (lambda: prepared.run([foreign]),
                                 lambda: prepared.submit_many([foreign], [borrowed, borrowed])):
                        with self.assertRaisesRegex(CUDAError, "different execution session"):
                            call()
                run.assert_not_called()
                submit.assert_not_called()

    def test_cuda_array_interface_v2_v3_and_invalid_descriptors(self):
        for stream in (None, 1, 2, 7340032):
            self.assertEqual(_cuda_array_description(ArrayOwner(strides=(20, 4), stream=stream)),
                             ((2, 5), "f32", 4096, False, stream))
        v2 = ArrayOwner(version=2)
        del v2.description["stream"]
        with self.assertRaisesRegex(CUDAError, "version 2.*producer_stream"):
            _cuda_array_description(v2)
        self.assertEqual(_cuda_array_description(v2, producer_stream=11)[4], 11)

        for invalid in (
            {"version": 1}, {"typestr": "<u2"}, {"typestr": ">f4"},
            {"shape": (2, -1)}, {"data": (0, False)}, {"strides": (4, 8)},
            {"mask": object()}, {"stream": 0},
        ):
            with self.assertRaises((CUDAError, ValueError)):
                _cuda_array_description(ArrayOwner(**invalid))

    def test_session_stream_refresh_owner_retention_and_cross_session_guards(self):
        native = FakeNativeSession()
        ext = SimpleNamespace(CUDASession=SimpleNamespace(create=lambda device: native))
        with patch.object(ask, "_native", ext, create=True):
            with ExecutionSession() as session:
                empty = session.tensor((0, 5), "f32")
                self.assertEqual(empty.__cuda_array_interface__["data"], (0, False))
                self.assertEqual(session.from_cuda_array(empty).data_ptr, 0)

                owner = ArrayOwner()
                weak_owner = weakref.ref(owner)
                borrowed = session.from_cuda_array(owner)
                owner.description["stream"] = 19
                session.submit(object(), [borrowed], grid=(1, 1, 1), block=(128, 1, 1))
                self.assertIn(("wait", 19), native.log)

                owner.description["shape"] = (10,)
                with self.assertRaisesRegex(CUDAError, "changed after import"):
                    session.submit(object(), [borrowed], grid=(1, 1, 1), block=(128, 1, 1))
                owner.description["shape"] = (2, 5)

                del owner, borrowed
                gc.collect()
                self.assertIsNotNone(weak_owner())
                native.events[0].complete = True
                session.reap()
                gc.collect()
                self.assertIsNone(weak_owner())

                owned = session.tensor((2, 5))
                with ExecutionSession() as other:
                    with self.assertRaisesRegex(CUDAError, "different execution session"):
                        other.submit(object(), [owned], grid=(1, 1, 1), block=(128, 1, 1))


class IRCacheAndExecutableCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="ask-cache-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_tool_fingerprint_reuses_reads_and_detects_restored_mtime(self):
        tool = self.root / "compiler.so"
        tool.write_bytes(b"version-one")
        original_status = tool.stat()
        original_open = Path.open
        with patch.object(Path, "open", autospec=True, side_effect=original_open) as reads:
            first = fingerprint_file(tool)
            self.assertEqual(fingerprint_file(tool), first)
            self.assertEqual(reads.call_count, 1)
        tool.write_bytes(b"version-two")
        os.utime(tool, ns=(original_status.st_atime_ns, original_status.st_mtime_ns))
        with patch.object(Path, "open", autospec=True, side_effect=original_open) as reads:
            changed = fingerprint_file(tool)
            self.assertNotEqual(changed["sha256"], first["sha256"])
            self.assertEqual(changed["sha256"], hashlib.sha256(b"version-two").hexdigest())
            self.assertEqual(fingerprint_file(tool), changed)
            self.assertEqual(reads.call_count, 1)
        replacement = self.root / "replacement"
        replacement.write_bytes(b"version-new")
        os.utime(replacement, ns=(original_status.st_atime_ns, original_status.st_mtime_ns))
        os.replace(replacement, tool)
        self.assertNotEqual(fingerprint_file(tool)["sha256"], changed["sha256"])

    def test_tool_fingerprint_rejects_changes_while_reading(self):
        tool = self.root / "unstable"
        tool.write_bytes(b"before")
        original_fstat = os.fstat
        calls = 0
        def changed_status(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                tool.write_bytes(b"after!")
            return original_fstat(descriptor)
        with patch("fugatensor._ask.cache.os.fstat", side_effect=changed_status):
            with self.assertRaisesRegex(OSError, "changed while fingerprinting"):
                fingerprint_file(tool)
        self.assertEqual(fingerprint_file(tool)["sha256"], hashlib.sha256(b"after!").hexdigest())

    def test_ir_cache_persistence_canonical_keys_corruption_and_concurrency(self):
        cache_dir = self.root / "ir"
        cache = IRCache(cache_dir)
        key = sample_cache_key()
        calls = 0

        def builder():
            nonlocal calls
            calls += 1
            return {"llvm/module.ll": b"IR payload", "launch.json": b'{"grid":[1,1,1]}'}

        artifacts, miss = cache.get_or_create(key, builder)
        reordered = dict(reversed(list(key.items())), shape=(10,))
        loaded, hit = IRCache(cache_dir).get_or_create(reordered, builder)
        self.assertEqual((miss["status"], hit["status"], calls), ("miss", "hit", 1))
        self.assertEqual(loaded, artifacts)

        # Corrupt payload triggers rebuild and invalid counter.
        entry = cache_dir / "entries" / miss["key"]
        (entry / "artifacts/llvm/module.ll").write_bytes(b"corrupted")
        _, repaired = cache.get_or_create(key, builder)
        self.assertEqual((repaired["status"], calls), ("invalid", 2))

        # Tampered manifest path is rejected.
        manifest_path = entry / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"]["../escape.ll"] = manifest["artifacts"].pop("llvm/module.ll")
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest_path.write_bytes(raw)
        (entry / "manifest.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
        self.assertEqual(cache.get_or_create(key, builder)[1]["status"], "invalid")

        for unsafe in ("../escape.ll", "/abs.ll", "a\\b"):
            with self.assertRaises(ValueError):
                cache.get_or_create({**key, "shape": [99]}, lambda: {unsafe: b"x"})

        with self.assertRaisesRegex(RuntimeError, "recursively"):
            cache.get_or_create({**key, "shape": [77]},
                                lambda: cache.get_or_create({**key, "shape": [77]}, builder)[0])

        # Multi-process single-builder lock & crash recovery.
        proc_dir = self.root / "mp"
        ctx = multiprocessing.get_context("spawn")
        start, queue = ctx.Event(), ctx.Queue()
        procs = [ctx.Process(target=_worker_process, args=(str(proc_dir), start, queue)) for _ in range(3)]
        for p in procs:
            p.start()
        start.set()
        results = [queue.get(timeout=20) for _ in procs]
        for p in procs:
            p.join(timeout=10)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(sorted(r["stats"]["status"] for r in results), ["hit", "hit", "miss"])
        self.assertEqual((proc_dir / "builders.log").read_bytes(), b"builder\n")
        queue.close()

        crash_dir = self.root / "crash"
        entered = ctx.Event()
        dying = ctx.Process(target=_crashing_builder, args=(str(crash_dir), entered))
        dying.start()
        self.assertTrue(entered.wait(10))
        dying.join(timeout=10)
        self.assertEqual(IRCache(crash_dir).get_or_create(key, builder)[1]["status"], "miss")

    def test_executable_cache_hits_and_tool_fingerprint_invalidation(self):
        llc, ptxas = self.root / "llc", self.root / "ptxas"
        llc.write_bytes(b"llc v1")
        ptxas.write_bytes(b"ptxas v1")
        tool_calls = []

        def fake_tool(cmd, **kwargs):
            tool_calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"PTX" if cmd[0] == str(llc) else b"CUBIN")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def run_compile(name: str):
            val = sample_artifact()
            ll = self.root / f"{name}.ll"
            ll.write_text(val["payload"], encoding="utf-8")
            return compile_cuda(ll, val, self.root / name, llc=llc, ptxas=ptxas, cache_dir=self.root / "exec_cache")

        with patch("fugatensor._ask.cuda.subprocess.run", side_effect=fake_tool):
            cold = run_compile("cold")
            warm = run_compile("warm")
            self.assertEqual((cold["executable_cache"]["status"], warm["executable_cache"]["status"]), ("miss", "hit"))
            self.assertEqual(len(tool_calls), 2)
            ptxas.write_bytes(b"ptxas v2")
            updated = run_compile("updated")
            self.assertEqual(updated["executable_cache"]["status"], "miss")


if __name__ == "__main__":
    unittest.main()
