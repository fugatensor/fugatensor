"""Portable PyTree weight export and import with a stable, versioned format.

The format stores a tree of arrays, not model classes or operator names. New
architectures, renamed parameters and newly added primitive operators change
only the tree keys and shapes; they do not require a format revision. Each
array is content-addressed and written as little-endian contiguous bytes.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile

from fugatensor._ask.numerics import decode_float, encode_float

from .tensor import Tensor

FORMAT = "fuga.weights"
SCHEMA_VERSION = 1
LAYOUT = "little-endian-contiguous-row-major"
_WIDTH = {"bool": 1, "i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2}
_KIND = {"dict", "list", "tuple", "array"}


class WeightsError(ValueError):
    """The weight package is malformed, incomplete or does not match its schema."""


def _require(condition, message):
    if not condition:
        raise WeightsError(message)


def _version():
    try:
        return importlib.metadata.version("fugatensor")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _json_object(value, field):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        # Round-trip through JSON so the caller cannot mutate the saved object
        # after validation and non-JSON values fail at the API boundary.
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must contain only JSON values") from error


def _count(shape):
    return math.prod(shape)


def _encode_values(values, dtype):
    count = len(values)
    if dtype == "bool":
        if any(type(value) is not bool for value in values):
            raise TypeError("bool weights require Python bool elements")
        return bytes(int(value) for value in values)
    if dtype in ("i32", "i64"):
        code = "i" if dtype == "i32" else "q"
        try:
            return struct.pack(f"<{count}{code}", *values)
        except (struct.error, TypeError) as error:
            raise TypeError(f"{dtype} weights require signed integer elements") from error
    if dtype not in ("f32", "f16", "bf16"):
        raise WeightsError(f"unsupported weight dtype: {dtype!r}")
    try:
        return b"".join(encode_float(value, dtype) for value in values)
    except (TypeError, ValueError) as error:
        raise WeightsError(f"cannot encode {dtype} weight values") from error


def _decode_values(data, dtype, count):
    width = _WIDTH[dtype]
    _require(len(data) == count * width,
             f"{dtype} array has {len(data)} bytes; expected {count * width}")
    if dtype == "bool":
        if any(byte not in (0, 1) for byte in data):
            raise WeightsError("bool weights must contain only 0 or 1 bytes")
        return [bool(byte) for byte in data]
    if dtype in ("i32", "i64"):
        code = "i" if dtype == "i32" else "q"
        return list(struct.unpack(f"<{count}{code}", data)) if count else []
    return [decode_float(data[index:index + width], dtype)
            for index in range(0, len(data), width)]


def _array_id(dtype, shape, data):
    header = json.dumps({"dtype": dtype, "shape": list(shape), "layout": LAYOUT},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + data).hexdigest()


def _tensor_bytes(tensor, path):
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{path} must be a Tensor or a container of Tensors")
    dtype, shape = tensor.dtype, tensor.shape
    _require(dtype in _WIDTH, f"{path} has unsupported dtype {dtype!r}")
    _require(all(type(dimension) is int and dimension >= 0 for dimension in shape),
             f"{path} has an invalid shape")
    count = _count(shape)
    _require(count <= (2**63 - 1) // _WIDTH[dtype], f"{path} byte count overflows int64")
    values = tensor.engine.to_flat(tensor)
    _require(isinstance(values, list) and len(values) == count,
             f"{path} storage returned an unexpected element count")
    return dtype, shape, _encode_values(values, dtype)


def _build_tree(value, arrays, staging, seen, path):
    if isinstance(value, Tensor):
        dtype, shape, data = _tensor_bytes(value, path)
        identifier = _array_id(dtype, shape, data)
        if identifier not in arrays:
            (staging / "arrays" / f"{identifier}.bin").write_bytes(data)
            arrays[identifier] = {
                "file": f"arrays/{identifier}.bin",
                "dtype": dtype,
                "shape": list(shape),
                "nbytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "layout": LAYOUT,
            }
        return {"kind": "array", "array": identifier}
    if isinstance(value, Mapping):
        if id(value) in seen:
            raise WeightsError(f"{path} contains a reference cycle")
        seen.add(id(value))
        items, keys = [], set()
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be nonempty strings")
            if key in keys:
                raise WeightsError(f"{path} contains duplicate key {key!r}")
            keys.add(key)
            items.append({"key": key, "value": _build_tree(child, arrays, staging, seen,
                                                              f"{path}[{key!r}]")})
        seen.remove(id(value))
        return {"kind": "dict", "items": items}
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            raise WeightsError(f"{path} contains a reference cycle")
        seen.add(id(value))
        items = [_build_tree(child, arrays, staging, seen, f"{path}[{index}]")
                 for index, child in enumerate(value)]
        seen.remove(id(value))
        return {"kind": "list" if isinstance(value, list) else "tuple", "items": items}
    raise TypeError(f"{path} must be a Tensor, Mapping, list or tuple")


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def save(directory, tree, *, metadata=None, overwrite=False):
    """Export a PyTree of Tensors to a new weight directory.

    ``directory`` is created atomically. Existing destinations are rejected
    unless ``overwrite=True``. The tree is materialized before writing, so the
    caller's Tensors may live on CPU or an active CUDA session.
    """
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    target = Path(directory).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"weight directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_node = _json_object(metadata, "metadata")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (staging / "arrays").mkdir()
        arrays = {}
        tree_node = _build_tree(tree, arrays, staging, set(), "$")
        manifest = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "producer": {"name": "fugatensor", "version": _version()},
            "metadata": metadata_node,
            "tree": tree_node,
            "arrays": {identifier: arrays[identifier] for identifier in sorted(arrays)},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        backup = None
        if target.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old.", dir=target.parent))
            backup.rmdir()
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup is not None and backup.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            try:
                _remove(backup)
            except OSError:
                # The new package is already published; a stale hidden backup
                # must not turn a successful export into an API failure.
                pass
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target




def _reject_constant(value):
    raise WeightsError(f"non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WeightsError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _manifest(directory):
    root = Path(directory).expanduser().resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"weight manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                                           parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WeightsError(f"cannot read weight manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise WeightsError("weight manifest must be a JSON object")
    if manifest.get("format") != FORMAT:
        raise WeightsError(f"unsupported weight format: {manifest.get('format')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION or type(manifest.get("schema_version")) is not int:
        raise WeightsError(f"unsupported weight schema version: {manifest.get('schema_version')!r}")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise WeightsError("weight manifest arrays must be an object")
    for identifier, record in arrays.items():
        if not isinstance(identifier, str) or not identifier:
            raise WeightsError("array IDs must be nonempty strings")
        if not isinstance(record, dict):
            raise WeightsError(f"array {identifier!r} must be an object")
        dtype, shape = record.get("dtype"), record.get("shape")
        if not isinstance(dtype, str) or dtype not in _WIDTH:
            raise WeightsError(f"array {identifier!r} has unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise WeightsError(f"array {identifier!r} has an invalid shape")
        width = _WIDTH[dtype]
        expected = _count(shape) * width
        if expected > 2**63 - 1:
            raise WeightsError(f"array {identifier!r} byte count overflows int64")
        if type(record.get("nbytes")) is not int or record["nbytes"] != expected:
            raise WeightsError(f"array {identifier!r} nbytes does not match its shape")
        if record.get("layout") != LAYOUT:
            raise WeightsError(f"array {identifier!r} uses an unsupported layout")
        if not isinstance(record.get("file"), str) or not record["file"]:
            raise WeightsError(f"array {identifier!r} has no file")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise WeightsError(f"array {identifier!r} has an invalid digest")
        try:
            int(record["sha256"], 16)
        except ValueError as error:
            raise WeightsError(f"array {identifier!r} has an invalid digest") from error
    _validate_tree(manifest.get("tree"), arrays, set(), "$")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise WeightsError("weight metadata must be an object")
    return root, manifest


def _validate_tree(node, arrays, seen, path):
    if not isinstance(node, dict):
        raise WeightsError(f"{path} must be a tree object")
    kind = node.get("kind")
    if not isinstance(kind, str) or kind not in _KIND:
        raise WeightsError(f"{path} has unsupported node kind {kind!r}")
    if kind == "array":
        identifier = node.get("array")
        if not isinstance(identifier, str) or identifier not in arrays:
            raise WeightsError(f"{path} references an unknown array")
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            raise WeightsError(f"{path} array metadata must be an object")
        return
    items = node.get("items")
    if not isinstance(items, list):
        raise WeightsError(f"{path} {kind} node requires an items array")
    if kind == "dict":
        keys = set()
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("key"), str) or not item["key"]:
                raise WeightsError(f"{path} dict item {index} has an invalid key")
            if item["key"] in keys:
                raise WeightsError(f"{path} contains duplicate key {item['key']!r}")
            keys.add(item["key"])
            _validate_tree(item.get("value"), arrays, seen, f"{path}[{item['key']!r}]")
    else:
        for index, item in enumerate(items):
            _validate_tree(item, arrays, seen, f"{path}[{index}]")
    if id(node) in seen:
        raise WeightsError(f"{path} contains a reference cycle")
    seen.add(id(node))


def _blob(root, identifier, record):
    relative = record["file"]
    if "\\" in relative:
        raise WeightsError("array file paths must use POSIX separators")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise WeightsError("array file path escapes the weight directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise WeightsError("array file path escapes the weight directory")
    try:
        data = resolved.read_bytes()
    except OSError as error:
        raise WeightsError(f"cannot read array {identifier!r}: {error}") from error
    if len(data) != record["nbytes"]:
        raise WeightsError(f"array {identifier!r} byte count mismatch")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise WeightsError(f"array {identifier!r} digest mismatch")
    if _array_id(record["dtype"], record["shape"], data) != identifier:
        raise WeightsError(f"array {identifier!r} content identity mismatch")
    return _decode_values(data, record["dtype"], _count(record["shape"]))


def _restore(node, root, arrays, make_tensor):
    kind = node["kind"]
    if kind == "array":
        identifier = node["array"]
        record = arrays[identifier]
        values = _blob(root, identifier, record)
        return make_tensor(values, dtype=record["dtype"], shape=tuple(record["shape"]))
    if kind == "dict":
        return {item["key"]: _restore(item["value"], root, arrays, make_tensor)
                for item in node["items"]}
    values = [_restore(item, root, arrays, make_tensor) for item in node["items"]]
    return values if kind == "list" else tuple(values)


def load(directory, *, engine=None, device=None):
    """Load a weight PyTree, creating Tensors on ``engine`` or the current Engine."""
    root, manifest = _manifest(directory)
    from .engine import _select_engine
    selected = _select_engine(engine, device)
    def restore(values, *, dtype, shape):
        return selected.tensor(values, dtype=dtype, shape=shape)
    return _restore(manifest["tree"], root, manifest["arrays"], restore)


def read_manifest(directory):
    """Return a validated manifest copy without reading array data."""
    _, manifest = _manifest(directory)
    return json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))


def read_metadata(directory):
    """Return the caller metadata stored in the manifest."""
    return read_manifest(directory).get("metadata", {})


export = save
