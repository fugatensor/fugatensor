Back to API Index: [Categories](README.md#categories) · Related Modules: [Tensor Creation](tensor.md) · Architecture: [Weight Format & Export](../architecture/weights.md)

# Weight Persistence

FugaTensor provides a native weight container format (`.fuga`) that supports content-addressed serialization of PyTree model parameters composed of arbitrary nested structures (`dict` / `list` / `tuple`) and Tensors.

<a id="save-weights"></a>

## Saving Weights

### `ft.save_weights(path, pytree)` / `ft.weights.save(path, pytree)`
Saves a parameter dictionary or PyTree to a single-file container.

- **Features**:
  - Automatically deduplicates identical tensor data via SHA-256 content addressing;
  - Records shapes, data types, and alignment information;
  - Does not store any specific Python classes, device states, or operator graphs.

```python
import fugatensor as ft

params = {
    "layer1": {
        "weight": ft.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias": ft.zeros(2),
    },
    "tokens": [ft.ones(4), ft.zeros(4)],
}

ft.save_weights("model.fuga", params)
```

<a id="load-weights"></a>

## Loading Weights

### `ft.load_weights(path, *, device=None)` / `ft.weights.load(path, *, device=None)`
Restores previously saved PyTree structures and tensor values.

- **Parameters**:
  - `path`: File path to the weight container;
  - `device`: Target device for loading tensors; defaults to the current default device.

```python
loaded = ft.load_weights("model.fuga", device="cpu")
print("Weights:", loaded["layer1"]["weight"].tolist())  # [[1.0, 2.0], [3.0, 4.0]]
print("Bias:", loaded["layer1"]["bias"].tolist())        # [0.0, 0.0]
```

<a id="inspect-weights"></a>

## Inspecting Manifest Metadata

### `ft.weights.manifest(path)`
Quickly reads the container's metadata manifest without materializing large tensor data in memory: includes `schema_version`, PyTree skeleton structure, and `dtype`/`shape` records for each tensor blob.

```python
meta = ft.weights.manifest("model.fuga")
print("Schema version:", meta["schema_version"])  # 1
```
