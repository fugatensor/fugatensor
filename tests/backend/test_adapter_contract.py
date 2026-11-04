"""Portable executable worker protocol, rejection paths and cache identity."""
from __future__ import annotations

import pytest

import base64
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from fugatensor._ask import Compiler, CompilerEnvironment, CompilerError
from fugatensor._ask.adapters import export_kernel, normalize_artifact, worker_command, worker_identity
from fugatensor._ask.cuda import CUDAError

ROOT = Path(__import__('os').environ.get('ASK_SOURCE_ROOT', Path(__file__).resolve().parents[3] / 'ask'))


def request():
    return {"protocol_version": 2, "source": "opaque source", "entry": "source_entry",
            "signature": {"a": "*fp32", "b": "*i32", "out": "*fp16"},
            "source_bindings": {"a": "input:0", "b": "input:1", "out": "output:0"},
            "constants": {}, "fixed_options": {},
            "target": {"backend": "cuda", "arch": 80, "warp_size": 32}, "grid": [2, 1, 1],
            "port_shapes": {"input:0": [8, 4], "input:1": [2], "output:0": [8]}}


def response():
    return {"protocol_version": 2, "format": "cubin", "payload": base64.b64encode(b'\x7fELF' + bytes(60)).decode(),
            "entry": "mangled_entry", "target": request()["target"],
            "producer": {"name": "fixture", "version": "1"},
            "launch": {"grid": [2, 1, 1], "block": [64, 1, 1], "shared_bytes": 1024},
            "arguments": ["input:1", "null", "input:0", "output:0"],
            "required_element_types": ["f32", "i32", "f16"]}


class TestAdapterContract:
    def test_binary_ports_are_derived_from_graph_and_physical_order(self):
        artifact = normalize_artifact(response(), request())
        assert ([p['logical_port'] for p in artifact['port_map']]) == (['input:1', 'input:0', 'output:0'])
        assert ([p['shape'] for p in artifact['port_map']]) == ([[2], [8, 4], [8]])
        assert ([p['storage_bytes'] for p in artifact['port_map']]) == ([4, 4, 2])
        assert (artifact['invocation']['shared_bytes']) == (1024)
        assert (artifact['invocation']['arguments'][1]['binding']) == ('null')

    def test_wrong_target_missing_ports_duplicate_bindings_and_launch_are_rejected(self):
        variants = []
        bad = response(); bad["target"]["arch"] = 86; variants.append(bad)
        bad = response(); bad["arguments"].pop(); variants.append(bad)
        bad = response(); bad["arguments"][1] = "input:0"; variants.append(bad)
        bad = response(); bad["launch"]["grid"] = [1, 1, 1]; variants.append(bad)
        bad = response(); bad["launch"]["block"] = [True, 1, 1]; variants.append(bad)
        bad = response(); bad["protocol_version"] = 1; variants.append(bad)
        bad = response(); bad["format"] = "llvm-ir"; variants.append(bad)
        bad = response(); bad["unexpected"] = {}; variants.append(bad)
        for artifact in variants:
            with pytest.raises((CompilerError, CUDAError)):
                normalize_artifact(artifact, request())

    def test_mixed_scalar_abi_preserves_values_and_only_maps_tensor_ports(self):
        req, reply = request(), response()
        values = {"n": ("u64", 2**64 - 1), "scale": ("fp32", -0.0),
                  "flag": ("i1", True), "other_flag": ("i1", True)}
        req["signature"].update({name: dtype for name, (dtype, _) in values.items()})
        req["source_bindings"].update({name: value for name, (_, value) in values.items()})
        reply["arguments"][1:1] = [{"type": dtype, "value": value} for dtype, value in values.values()]
        artifact = normalize_artifact(reply, req)
        assert ([p['argument'] for p in artifact['port_map']]) == ([0, 6, 7])
        assert (artifact['invocation']['arguments'][1]) == ({'argument': 1, 'kind': 'scalar', 'type': 'u64', 'value': 2 ** 64 - 1})
        assert (len(artifact['port_map'])) == (3)
        # Matching must retain signed zero, type, and repeated scalar count.
        for index, replacement in ((1, {"type": "i64", "value": -1}),
                                   (2, {"type": "fp32", "value": 0.0}),
                                   (3, {"type": "i1", "value": 1}),
                                   (4, "null")):
            bad = json.loads(json.dumps(reply))
            bad["arguments"][index] = replacement
            with pytest.raises((CompilerError, CUDAError)):
                normalize_artifact(bad, req)

    def test_explicit_worker_needs_no_registry_and_failures_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.py"
            worker.write_text('''import argparse, json
p=argparse.ArgumentParser()
p.add_argument('--probe', action='store_true'); p.add_argument('--request'); p.add_argument('--output')
a=p.parse_args()
if a.probe:
    print(json.dumps({'producer': {'name': 'fixture', 'version': '1'}, 'adapter_version': '1'}))
else:
    with open(a.request) as f: r=json.load(f)
    assert r['protocol_version']==2
    with open(a.output,'w') as f: json.dump(RESPONSE,f)
'''.replace('RESPONSE', repr(response())))
            env = CompilerEnvironment(python=sys.executable, worker=worker)
            command = worker_command(env)
            identity = worker_identity(command, timeout=10)
            assert (identity['probe']['producer']['name']) == ('fixture')
            artifact = export_kernel(command, request(), timeout=10)
            assert (artifact['entry']) == ('mangled_entry')
            worker.write_text("import sys\nprint('compiler crashed',file=sys.stderr)\nsys.exit(2)\n")
            with pytest.raises(CompilerError, match='compiler crashed'):
                export_kernel(command, request(), timeout=10)

    def test_generic_compiler_is_accepted_by_the_native_frontend(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text().replace('compiler = "triton"', 'compiler = "new_vendor"')
        result = Compiler().partition(source)
        assert (result.external_requests[0]['compiler']) == ('new_vendor')
        assert ([s['kind'] for s in result.execution_plan['steps']]) == (['owned', 'external', 'owned'])

    def test_dependencies_changing_during_export_are_not_cached(self):
        from fugatensor._ask.gpu import _compile_external_graph
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text('''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--probe', action='store_true'); p.add_argument('--request'); p.add_argument('--output')
a=p.parse_args()
revision=Path(__file__).with_suffix('.revision')
if a.probe:
    print(json.dumps({'producer': {'name': 'fixture', 'version': '1'}, 'adapter_version': '1',
                      'dependency': revision.read_text() if revision.exists() else 'original'}))
else:
    Path(a.output).write_text(json.dumps(RESPONSE))
    revision.write_text('changed')
'''.replace('RESPONSE', repr(response())))
            prepared = SimpleNamespace(external_requests=[{**request(), "compiler": "fixture"}])
            with pytest.raises(CompilerError, match='dependencies changed'):
                _compile_external_graph(SimpleNamespace(timeout=10), prepared,
                    {"fixture": CompilerEnvironment(python=sys.executable, worker=worker)},
                    cache_dir=root / "cache", output_dir=root / "output")
            assert not (any((root / 'cache' / 'external' / 'entries').glob('*')))

    def test_public_contract_schemas_are_valid_json_objects(self):
        for name in ("execution-plan-v1.schema.json", "external-compiler-request-v2.schema.json",
                     "external-kernel-artifact-v2.schema.json"):
            value = json.loads((ROOT / "schemas" / name).read_text())
            assert (value['$schema']) == ('https://json-schema.org/draft/2020-12/schema')
            assert (value['type']) == ('object')
            assert ('required') in (value)
