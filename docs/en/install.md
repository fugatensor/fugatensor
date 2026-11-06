# FugaTensor Installation

Back to Documentation Index: [Documentation Index](README.md) · Related topics: [ASK Bridge and Multi-Backend Execution](architecture/backend.md#bootstrap-and-config)

To install an existing wheel, refer directly to Section 4. When building from source, install the ASK SDK first, then build FugaTensor. Please replace `/absolute/path/...` with your actual paths.

<a id="requirements"></a>

## 1. Version and Environment Requirements

FugaTensor 0.1.0 `pure` installation requires Python 3.10+; `all` requires Python 3.11+ and a Triton-supported platform. Source builds use ASK SDK 0.1.0. The wheel must match the Python ABI, operating system, and architecture; the currently validated platform is Linux x86-64.

| Usage | Additional Requirements |
| --- | --- |
| Install wheel and run CPU graphs | C toolchain supporting `cc -shared`, used for JIT linking of object files. |
| Build FugaTensor from source | ASK SDK, C++17 compiler, Python development headers, CMake 3.28+, and build tools. For Python build dependencies, see `pyproject.toml`. |
| Build ASK SDK locally | Matching stable LLVM/MLIR SDK supported by ASK and C++ build tools; dependency retrieval and building are handled by the ASK project. |
| CUDA execution | NVIDIA driver and devices. Owned computations use `llc` matching the ASK build version and `ptxas` supporting the target PTX; third-party computations use corresponding workers. |

<a id="build-ask"></a>

## 2. Installing C++ SDK in the ASK Project

Build and install the SDK in the ASK source directory:

```bash
cd /absolute/path/to/ask
python3 tools/fetch_deps.py
cmake --preset core -DCMAKE_INSTALL_PREFIX=/absolute/path/to/ask-sdk
cmake --build --preset core
cmake --install build/core
/absolute/path/to/ask-sdk/bin/ask-opt --capabilities
```

FugaTensor links against `libask_compiler.so` and `libask_runtime.so` in the SDK, and bundles `libask_openblas.so.0` used for CPU fp32 matrix multiplication. For other installation options, see [ASK Installation](../../../ask/docs/en/install.md).

<a id="install-fuga-source"></a>
<a id="build-fugatensor"></a>

## 3. Building FugaTensor Wheel

Pass the installed SDK prefix to CMake:

```bash
CMAKE_PREFIX_PATH=/absolute/path/to/ask-sdk \
  python3 -m pip wheel --no-deps /absolute/path/to/fugatensor \
  --wheel-dir /absolute/path/to/fuga-wheel-output
```

pip installs `scikit-build-core` and `nanobind` declared in `pyproject.toml`, then builds the extension using SDK headers and shared libraries. The resulting wheel places ASK and OpenBLAS shared libraries in `fugatensor/_lib/`, while installing the OpenBLAS license.

For development, an editable install can be used instead:

```bash
CMAKE_PREFIX_PATH=/absolute/path/to/ask-sdk \
  python -m pip install -e '/absolute/path/to/fugatensor[pure]'
python -m pip check
```

Change `[pure]` to `[all]` to install Triton simultaneously.

<a id="install-wheels"></a>
<a id="installation-variants"></a>

## 4. Installing FugaTensor Wheel

`pure` and `all` are standard optional dependencies of the same distribution package:

| Installation Command | Dependencies and Usage |
| --- | --- |
| `python -m pip install 'fugatensor[pure]'` | Base dependencies, suitable for CPU and ASK owned CUDA computation; equivalent to installing `fugatensor` directly. |
| `python -m pip install 'fugatensor[all]'` | Automatically installs `triton>=3.9,<3.10` for user Triton operators. Python and platform must meet Triton requirements. |

Both options share the wheel, which contains C++ extensions, ASK shared libraries, and adapter source code. Environments already containing Triton can use that compiler directly. `all` fails installation if Python or the platform is incompatible.

When installing a locally built wheel, specify the corresponding extra after the full path:

```bash
python3 -m venv /absolute/path/to/fuga-env
source /absolute/path/to/fuga-env/bin/activate

# Substitute the full filename of the actual wheel
fuga_wheel='/absolute/path/to/fugatensor-0.1.0-<python>-<abi>-<platform>.whl'
python -m pip install "${fuga_wheel}[pure]"
# To use Triton, change [pure] in the line above to [all]
python -m pip check
```

`all` is resolved and installed by pip for Triton; for offline installation, prepare a compatible Triton wheel in advance. Linkers and CUDA tools are configured in subsequent sections.

<a id="verify-cpu"></a>

## 5. Verifying CPU Installation Outside Source Directory

In the virtual environment, switch to `/tmp` to check post-installation JIT computation:

```bash
cd /tmp
env -u PYTHONPATH -u PYTHONHOME -u ASK_OPT -u FUGATENSOR_CONFIG python -I - <<'PY'
from importlib.metadata import version
import fugatensor as ft

print("Version:", version("fugatensor"))  # 0.1.0
def arithmetic(x, y):
    return (x + y) * 2

x = ft.tensor([1, 2])
y = ft.tensor([10, 20])
result = arithmetic(x, y).tolist()
print(result)  # [22.0, 44.0]
print("FugaTensor:", ft.__file__)
PY
```

When using a different linker, call `ft.configure(cc="/absolute/path/to/cc")` prior to creating tensors. For standard usage, see [API](API/README.md).

<a id="configure-cuda"></a>

## 6. Configuring Owned CUDA Paths

Owned CUDA computations use `llc` to generate PTX and `ptxas` to generate cubins, executed by the NVIDIA driver. `llc --version` must match the `llvm_version` in ASK's `ask-opt --capabilities`.

CUDA `exp/log` uses the matching version of `llvm-link` and `nvvm/libdevice/libdevice.10.bc` from the CUDA Toolkit. Defaults discover `llvm-link` from `llc`'s directory and `libdevice` from `CUDA_HOME` or `/usr/local/cuda*`; they can also be set via `ft.configure(llvm_link=..., libdevice=...)` or `FUGATENSOR_LLVM_LINK` / `FUGATENSOR_LIBDEVICE`. Only cubin compilations containing these functions read and cache these two dependencies. CUDA matrix multiplication dynamically loads system cuBLASLt via the ASK Runtime.

```python
import fugatensor as ft

ft.configure(
    device="cuda",
    llc="/absolute/path/to/llvm/bin/llc",
    ptxas="/absolute/path/to/cuda/bin/ptxas",
)
x = ft.tensor([1, 2], device="cuda")
print((x * 2 + 1).tolist())  # [3.0, 5.0]
```

Tool paths can also be provided via `FUGATENSOR_LLC`, `FUGATENSOR_PTXAS`. The default device is set via `ft.set_default_device("cuda")`; `configure` sets tools and runtime options.

<a id="configure-triton"></a>

## 7. Optional Triton Compilation Environment

`all` defaults to running the in-tree Triton worker with the current Python environment. When graphs include ASK owned computations, configure `llc` / `ptxas` as described in the previous section.

Triton 3.9.x stable is supported, with 3.9.0 validated. For version baselines and ABIs, see [`triton.dependencies.lock.json`](../../src/fugatensor/triton.dependencies.lock.json); caching records actual compiler and tool information.

When using an isolated Python 3.11+ environment to compile Triton:

```bash
/path/to/triton-env/bin/python -m pip install 'triton>=3.9,<3.10'
```

```python
import fugatensor as ft

ft.configure(
    device="cuda",
    llc="/absolute/path/to/llvm/bin/llc",
    ptxas="/absolute/path/to/cuda/bin/ptxas",
    compilers={"triton": ft.triton.environment(python="/path/to/triton-env/bin/python")},
)
```

Hosts on Python 3.10 can use isolated workers via the configuration above; `from_jit` requires the host environment to satisfy Triton version requirements. You can also specify the interpreter via `FUGATENSOR_TRITON_PYTHON` and worker via `FUGATENSOR_TRITON_WORKER`. For configuration precedence, see [Configuration Rules](architecture/backend.md#bootstrap-and-config); for registration methods, see [README](../../README.md).

<a id="source-bootstrap"></a>

## 8. Persistent Configuration File

To persist paths for `llc`, `ptxas`, or an isolated Triton interpreter, write JSON configuration directly to `share/fugatensor/engine.json` in the Python virtual environment:

```json
{
  "schema_version": 1,
  "llc": "/path/to/llc",
  "ptxas": "/path/to/ptxas",
  "compilers": {
    "triton": {
      "python": "/path/to/triton-env/bin/python"
    }
  }
}
```

For configuration precedence and field descriptions, see [ASK Bridge and Multi-Backend Execution](architecture/backend.md#bootstrap-and-config).

Related notes: [ASK Installation](../../../ask/docs/en/install.md), [ASK Version Contract](../../../ask/docs/en/build.md#version-contract).
