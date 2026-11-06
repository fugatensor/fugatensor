# Weight Format and Export

Back to Architecture Overview: [Weight Format and Export](README.md#arch-weights) · Related topics: [Automatic Differentiation and Custom Operators](autograd.md) · [Lowering Fuga IR to ASK](../ir/lowering.md)

Weights are represented as PyTrees: nested containers store parameter structures, and Tensor leaves store arrays. Tree keys are named by the caller, and devices are chosen by dtype at load time: CPU receives bool, i32, i64, f32, and f16, while weights containing bf16 use CUDA. Weights can be restored across sessions and processes.

API: [Weight Reading and Writing](../API/weights.md)

<a id="weight-goals"></a>

## 1. Design Boundaries

- **Parameter Tree**: Nested `dict`, `list`, `tuple`, and Tensor form the tree, preserving container types and ordering upon loading.
- **Dense Array Leaves**: Store numerical values for six dtypes: `bool`, `i32`, `i64`, `f32`, `f16`, `bf16`, restoring as new leaf Tensors.
- **Content Addressing**: Each array takes the SHA-256 of "dtype + shape + layout + data" as its ID, with files placed at `arrays/<id>.bin`; identical content is written only once.
- **Versioning**: `format = "fuga.weights"`, `schema_version = 1`. Versions advance with changes to container or storage semantics, while changes to parameter naming and network architectures continue using the same format.
- **Atomic Export**: Files are completely written to a peer temporary directory first, then renamed to the destination directory.
- **Array Format**: Little-endian, contiguous, row-major. Values are read from host memory upon saving and encoded using ASK's single-step RNE rule; round-tripping NaN payloads may normalize to the default NaN.

<a id="weight-layout"></a>

## 2. Directory Structure and Manifest

```text
checkpoint/
├── manifest.json
└── arrays/
    ├── 3f2a...c1.bin
    └── 8d09...7e.bin
```

`manifest.json` example:

```json
{
  "format": "fuga.weights",
  "schema_version": 1,
  "producer": {"name": "fugatensor", "version": "0.1.0"},
  "metadata": {"step": 1200, "run": "demo"},
  "tree": {
    "kind": "dict",
    "items": [
      {"key": "encoder", "value": {
        "kind": "dict",
        "items": [
          {"key": "weight", "value": {"kind": "array", "array": "3f2a...c1"}}
        ]
      }},
      {"key": "blocks", "value": {
        "kind": "list",
        "items": [{"kind": "array", "array": "8d09...7e"}]
      }}
    ]
  },
  "arrays": {
    "3f2a...c1": {
      "file": "arrays/3f2a...c1.bin",
      "dtype": "f16",
      "shape": [768, 768],
      "nbytes": 1179648,
      "sha256": "b92f...0a",
      "layout": "little-endian-contiguous-row-major"
    },
    "8d09...7e": {
      "file": "arrays/8d09...7e.bin",
      "dtype": "f32",
      "shape": [768],
      "nbytes": 3072,
      "sha256": "18cd...7f",
      "layout": "little-endian-contiguous-row-major"
    }
  }
}
```

- The `tree` contains only four node kinds: `dict`, `list`, `tuple`, `array`; `items` in `dict` preserve key ordering, and duplicate keys are disallowed at load time.
- `array` nodes reference IDs in `arrays` only. Array IDs are computed from a canonical JSON header plus a NUL byte plus raw data, preventing identical bytes with different dtype/shape from being erroneously merged.
- `metadata` is a caller-defined JSON object, and array nodes can also attach `metadata`; the loader checks only its type.
- `producer` records the generating tool and version for diagnostic purposes.
- The array's `nbytes` must equal `prod(shape) * element_width`; `sha256` is the digest of the data file, while the `array` ID additionally covers the dtype/shape layout header.

For field constraints, see [`schemas/weights-manifest-v1.schema.json`](../../../schemas/weights-manifest-v1.schema.json). dtype uses string labels, and new types can be extended by loaders.

<a id="weight-api"></a>

## 3. Python API

```python
import fugatensor as ft

weights = {
    "encoder": {"weight": ft.tensor([[1.0, 2.0]])},
    "blocks": [ft.tensor([1.5, 2.5], dtype=ft.fp16)],
    "steps": ft.tensor(1200, dtype=ft.int32),
}

ft.weights.save("checkpoint", weights, metadata={"run": "demo"})
# Equivalent entry point: ft.save_weights("checkpoint", weights, ...)

loaded = ft.weights.load("checkpoint")
print(loaded["blocks"][0].tolist())         # [1.5, 2.5]
print(ft.weights.read_metadata("checkpoint"))  # {'run': 'demo'}
```

- `save(directory, tree, *, metadata=None, overwrite=False)`: Raises an error by default if the target already exists; `overwrite=True` permits replacement. Returns target `Path`.
- `load(directory, *, engine=None, device=None)`: Follows the current default device, or selects resources via `device` or `engine` according to [Tensor Creation Rules](../API/tensor.md#tensor-create); when both are provided, they must match. bf16 weights select CUDA; floating-point weights can be explicitly differentiated via `ft.grad` in subsequent expressions.
- `read_manifest(directory)` returns a validated copy of the manifest without loading large arrays; `read_metadata(directory)` returns caller metadata only.
- `ft.save_weights` / `ft.load_weights` are corresponding top-level entry points.

Saving evaluates pending weights and reads data back to the host; loading returns standalone data inputs without restoring pre-save expression provenance.

<a id="weight-validation"></a>

## 4. Validation and Security

The loader verifies the manifest and tree structure first:

1. JSON only permits finite numbers; duplicate keys and unknown `format`/`schema_version` fail immediately.
2. The tree accepts only four node kinds; array references must exist, and `dict` keys cannot duplicate.
3. Array dtype, shape, `nbytes`, layout, and digest format are verified item by item.

It then restores leaves in tree order: validating POSIX relative paths within the directory, reading arrays and verifying length, `sha256`, and content ID, then creating Tensors on the chosen device. Path checking rejects absolute paths, `..`, backslashes, and symlink escapes.

The save side similarly rejects cyclic references, non-Tensor leaves, non-string or empty mapping keys, non-JSON metadata, and integer overflows.

<a id="weight-extensibility"></a>

## 5. Compatibility for Future Refactoring

- Changes to network layers and parameter naming only alter keys and structure in `tree`, continuing to use v1.
- New dtypes are extended by loaders; older loaders reject unknown types.
- Top-level and array node `metadata` can store additional information. The loader ignores unknown top-level extension fields, while trees and arrays are still validated against existing rules.
- Sparse, sharded, compressed, or quantized storage requires a new `layout` or version.

Weights are saved and loaded on a per-directory basis, writing the complete manifest and referenced array data on each save.

<a id="weight-tests"></a>

## 6. Verification

`tests/test_weights.py` covers:

- Round-tripping six dtypes, 0-D, empty tensors, nested dict/list/tuple, and metadata;
- Deduplication of identical content, rejection of duplicate exports, and explicit overwrite;
- Rejection of non-Tensor leaves, cyclic references, and invalid metadata;
- Rejection of tampered schema versions/format, path escapes, and digest or byte count mismatches;
- Loading with specified Engine, device, and floating-point weights, followed by differentiation;
- Round-tripping GPU Tensors through host byte format when `FUGATENSOR_TEST_CUDA=1`.

Public schemas, Python APIs, and documentation are maintained in the same commit; modifying array fields or tree node semantics requires incrementing `schema_version` while retaining the read path for older versions.
