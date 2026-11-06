Back to API Index: [Categories](README.md#categories) · Related Modules: [Tensor Creation](tensor.md) · Architecture: [Multi-Backend Execution](../architecture/backend.md)

# Device Management and Execution Scheduling

FugaTensor manages a global resident engine (Engine) per device by default. Tensors are constructed using lazy graphs; compilation and execution are triggered only upon explicit evaluation or when inspecting values.

<a id="devices"></a><a id="default-device"></a><a id="default-device"></a>

## Device Objects and Default Device

### `ft.device(name)`
Constructs a device descriptor object. Supported formats:
- `"cpu"`: CPU device;
- `"cuda"` or `"cuda:N"`: CUDA GPU device (defaults to device 0 if index is omitted).

### `ft.set_default_device(device)` / `ft.get_default_device()`
Gets or sets the default device for subsequent tensor creations within the current context.

```python
import fugatensor as ft

print("Initial default:", ft.get_default_device())  # cpu

# Set default device to CUDA for subsequent tensor creations
# ft.set_default_device("cuda")
# x = ft.ones(2, 2)
# print(x.device)  # cuda:0
```

<a id="evaluation"></a>

## Lazy Execution and Stream Synchronization

### `ft.evaluate(*tensors)`
Explicitly materializes the unevaluated graph nodes behind one or more tensors.

- **Returns**: Returns the tensor handle if a single tensor is passed, or a list of the original tensors if multiple are passed.
- **Mechanism**: Collects ancestor nodes of all evaluated tensors, fuses them into a single static graph, and executes them via the backend engine. The physical memory/VRAM allocation is populated directly in-place into the original tensor handles, preserving expression provenance.

```python
import fugatensor as ft

a = ft.tensor([1.0, 2.0])
b = ft.tensor([3.0, 4.0])
c = a + b  # Memory has not yet been allocated for the result at this point

# Explicitly trigger graph compilation and execution
ft.evaluate(c)
print(c.tolist())  # [4.0, 6.0]
```

### `ft.synchronize(device=None)`
Waits for all asynchronous compute tasks submitted to the target device stream to complete, without triggering materialization of unevaluated graphs. When `device=None`, synchronizes the current default device.

```python
# Ensure all in-flight kernels on the target device have finished execution
ft.synchronize("cpu")
```

<a id="engine-class"></a>

## Standalone Engine `ft.Engine`

An advanced resource management class for isolating VRAM, compilation caches, and execution lifecycles:

```python
import fugatensor as ft

# Create an isolated CPU engine
with ft.Engine(device="cpu") as eng:
    t = eng.tensor([10.0, 20.0])
    print(t.tolist())  # [10.0, 20.0]
    print(eng.cache_info())  # {'compilations': 1, 'cache_hits': 0, ...}
```

- **Methods**:
  - `eng.tensor(...)`: Creates a tensor within the current engine;
  - `eng.evaluate(*tensors)`: Materializes tensors using this engine's compilation cache and streams;
  - `eng.synchronize()`: Waits until the streams associated with this engine are idle;
  - `eng.clear_cache()`: Clears compiled JIT execution plans;
  - `eng.close()`: Closes the engine and releases resident sessions and memory pools.
