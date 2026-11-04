"""Portable PyTree weight export: roundtrip, dedup, integrity and rejection paths."""
from __future__ import annotations

import pytest

import json
from pathlib import Path
import tempfile

import fugatensor as ft
from fugatensor._ask.cuda import CUDAError
from fugatensor.weights import WeightsError
from tests._helpers import requires_cuda


@pytest.mark.usefixtures("engine_context")
class TestWeightExport:
    device = "cpu"

    def make_tree(self):
        return {
            "bool": ft.tensor([True, False, True], dtype="bool"),
            "i32": ft.tensor([-2, 0, 7], dtype="i32"),
            "i64": ft.tensor([-(2 ** 40), 2 ** 40], dtype="i64"),
            "f32": ft.tensor([1.5, -2.25], dtype=ft.fp32),
            "f16": ft.tensor([[1.5, -2.0], [3.25, 4.0]], dtype="f16"),
            "nested": [{"weight": ft.tensor([[1.0, 2.0]], dtype="f32")},
                       (ft.tensor(3.0, dtype="f32"),)],
            "empty": ft.empty(2, 0, 3),
        }

    def test_published_manifest_schema_describes_the_format(self):
        schema = json.loads((Path(__file__).resolve().parents[1] /
                             "schemas/weights-manifest-v1.schema.json").read_text(encoding="utf-8"))
        assert (schema['properties']['format']['const']) == ('fuga.weights')
        assert (schema['properties']['schema_version']['const']) == (1)
        assert ('tree') in (schema['properties'])
        assert ('array') in (schema['$defs'])

    def test_roundtrip_cpu_dtypes_nested_tree_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            tree = self.make_tree()
            ft.weights.save(target, tree, metadata={"step": 12, "arch": "demo"})
            manifest = ft.weights.read_manifest(target)
            assert ((manifest['format'], manifest['schema_version'])) == ((ft.weights.FORMAT, 1))
            assert (ft.weights.read_metadata(target)) == ({'arch': 'demo', 'step': 12})
            loaded = ft.weights.load(target, engine=self.engine)
            assert isinstance(loaded['nested'], list)
            assert isinstance(loaded['nested'][1], tuple)
            for name in ("bool", "i32", "i64", "f32", "f16"):
                assert (loaded[name].dtype) is (ft.DType(name))
                assert (loaded[name].dtype) == (tree[name].dtype)
                assert (loaded[name].shape) == (tree[name].shape)
                assert (loaded[name].tolist()) == (tree[name].tolist())
            assert ({record['dtype'] for record in manifest['arrays'].values()}) == ({'bool', 'i32', 'i64', 'f32', 'f16'})
            assert (loaded['nested'][0]['weight'].tolist()) == ([[1.0, 2.0]])
            assert (loaded['nested'][1][0].tolist()) == (3.0)
            assert (loaded['empty'].shape) == ((2, 0, 3))
            assert (loaded['empty'].tolist()) == ([[], []])

    def test_content_addressed_dedup_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            first = ft.tensor([1.0, 2.0], dtype="f32")
            tree = {"a": first, "b": first, "copy": ft.tensor([1.0, 2.0], dtype="f32")}
            ft.weights.save(target, tree)
            manifest = ft.weights.read_manifest(target)
            assert (len(manifest['arrays'])) == (1)
            assert (len(manifest['tree']['items'])) == (3)
            with pytest.raises(FileExistsError):
                ft.weights.save(target, tree)
            assert (ft.weights.load(target, engine=self.engine)['a'].tolist()) == ([1.0, 2.0])
            ft.weights.save(target, {"x": ft.tensor([3], dtype="i32")}, overwrite=True)
            assert (ft.weights.load(target, engine=self.engine)['x'].tolist()) == ([3])

    def test_rejects_non_tensor_leaves_bad_keys_cycles_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            for tree in ({"x": 1}, {1: ft.tensor([1])}, [ft.tensor([1]), object()]):
                with pytest.raises(TypeError):
                    ft.weights.save(target, tree)
            cycle = {}
            cycle["self"] = cycle
            with pytest.raises(WeightsError, match='cycle'):
                ft.weights.save(target, cycle)
            with pytest.raises(TypeError, match='JSON'):
                ft.weights.save(target, {"x": ft.tensor([1])}, metadata={"bad": object()})

    def test_rejects_integrity_escape_and_version_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            ft.weights.save(target, {"x": ft.tensor([1.0, 2.0], dtype="f32")})
            manifest_path = target / "manifest.json"
            original = json.loads(manifest_path.read_text(encoding="utf-8"))

            def write(value):
                manifest_path.write_text(json.dumps(value), encoding="utf-8")

            for field, value, message in (
                    ("schema_version", 2, "schema version"),
                    ("format", "other.weights", "format")):
                tampered = json.loads(json.dumps(original))
                tampered[field] = value
                write(tampered)
                with pytest.raises(WeightsError, match=message):
                    ft.weights.load(target, engine=self.engine)

            tampered = json.loads(json.dumps(original))
            identifier = next(iter(tampered["arrays"]))
            tampered["arrays"][identifier]["file"] = "../escape.bin"
            write(tampered)
            with pytest.raises(WeightsError, match='escapes'):
                ft.weights.load(target, engine=self.engine)

            write(original)
            blob = target / original["arrays"][identifier]["file"]
            blob.write_bytes(b"tampered")
            with pytest.raises(WeightsError, match='mismatch'):
                ft.weights.load(target, engine=self.engine)

    def test_optional_extensions_and_minimal_manifest_are_forward_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            ft.weights.save(target, {"x": ft.tensor([1.0, 2.0], dtype="f32")})
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("producer", None)
            manifest.pop("metadata", None)
            manifest["future_extension"] = {"ignored": True}
            manifest["tree"]["items"][0]["value"]["metadata"] = {"scale": [1.0, 1.0]}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = ft.weights.load(target, engine=self.engine)
            assert (loaded['x'].tolist()) == ([1.0, 2.0])
            assert (ft.weights.read_metadata(target)) == ({})

    def test_loaded_weights_are_differentiable_in_explicit_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            ft.weights.save(target, {"w": ft.tensor([1.0, 2.0], dtype="f32")})
            with ft.Engine("cpu") as other:
                loaded = ft.weights.load(target, engine=other)
                weight = loaded["w"]
                derivative = ft.grad(weight * weight, weight, seed=ft.ones_like(weight))
                assert (derivative.tolist()) == ([2.0, 4.0])
                assert (loaded['w'].engine) == (other)
            ft.weights.save(Path(directory) / "ints", {"w": ft.tensor([1], dtype="i32")})
            integers = ft.weights.load(Path(directory) / "ints", engine=self.engine)["w"]
            with pytest.raises((TypeError, ValueError), match='float|dtype'):
                ft.grad(integers, integers)


@requires_cuda
@pytest.mark.usefixtures("engine_context")
class TestCUDAWeightExport:
    device = "cuda"

    def test_gpu_tensors_roundtrip_through_host_blobs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights"
            dtypes = ("f16", "f32", "bf16") if self.supports_bf16 else ("f16", "f32")
            if not self.supports_bf16:
                with pytest.raises(CUDAError, match='bf16.*support'):
                    ft.tensor([1], dtype=ft.bf16)
            tree = {dtype: ft.tensor([1.5, -2.0, 3.25], dtype=dtype)
                    for dtype in dtypes}
            ft.weights.save(target, tree)
            loaded = ft.weights.load(target, engine=self.engine)
            for dtype, value in tree.items():
                assert (loaded[dtype].tolist()) == (value.tolist())
                assert (loaded[dtype].dtype) == (value.dtype)
                assert (loaded[dtype].device) == (value.device)
            if self.supports_bf16:
                with pytest.raises(TypeError, match='CPU tensors support only bool/i32/i64/f32/f16'):
                    ft.weights.load(target, device="cpu")
