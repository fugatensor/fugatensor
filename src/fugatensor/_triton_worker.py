"""Compile user Triton source to a standalone cubin for ASK's worker protocol.

This file runs in the selected compiler environment, independently of FugaTensor
and ASK imports. The supported Triton CUDA ABI appends two scratch pointers after
the runtime source arguments; unsupported launch resources are rejected below.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import dataclasses
import importlib.util
import hashlib
import json
import inspect
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

# Direct execution puts the package directory first, where triton.py is our
# registration API rather than the compiler distribution selected by Python.
if __name__ == "__main__":
    sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).parent.resolve()]

DEPENDENCIES_LOCK = Path(__file__).with_name("triton.dependencies.lock.json")
LOCK = json.loads(DEPENDENCIES_LOCK.read_text(encoding="utf-8"))
TRITON_VERSION = LOCK["compiler"]["version"]
TRITON_SERIES = TRITON_VERSION.rsplit(".", 1)[0]
ADAPTER_VERSION = LOCK["adapter_version"]
ELEMENT_TYPES = {"*i1": "bool", "*i32": "i32", "*i64": "i64",
                 "*fp32": "f32", "*fp16": "f16", "*bf16": "bf16"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_triton():
    require(type(LOCK["protocol_version"]) is int and LOCK["protocol_version"] == 2,
            "dependency lock must describe worker protocol_version=2")
    require(not any(os.environ.get(name) for name in
                    ("TRITON_KERNEL_OVERRIDE", "TRITON_OVERRIDE_DIR", "TRITON_OVERRIDE_ARCH",
                     "LLVM_PASS_PLUGIN_PATH", "PTXAS_OPTIONS")),
            "external IR/target overrides, LLVM plugins and free-form ptxas options are not supported")
    import triton
    require(re.fullmatch(re.escape(TRITON_SERIES) + r"\.\d+", triton.__version__) is not None,
            f"Triton CUDA ABI requires a stable triton {TRITON_SERIES}.x release; found {triton.__version__}")
    return triton


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def source_module(request):
    # Source is trusted Python, just as a normal Triton module is. The same
    # loader is used for probing and compilation so imports share one identity.
    with tempfile.TemporaryDirectory(prefix="fuga-triton-") as directory:
        path = Path(directory) / "kernel.py"
        path.write_text(request["source"], encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_fuga_triton_source", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(spec.name, None)


def probe(request=None):
    triton = load_triton()
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.runtime.cache import triton_key
    from triton.backends.nvidia.compiler import CUDAOptions
    root = Path(triton.__file__).parent
    dependencies = {"triton": hashlib.sha256(triton_key().encode()).hexdigest(),
                    "adapter_lock": file_digest(DEPENDENCIES_LOCK)}
    paths = {"ptxas": triton.knobs.nvidia.ptxas.path,
             "ptxas_blackwell": triton.knobs.nvidia.ptxas_blackwell.path,
             **dict(CUDAOptions().extern_libs)}
    paths.update({str(path.relative_to(root)): path for path in
                  [root / "__init__.py", root / "knobs.py", *sorted((root / "runtime").rglob("*.py"))]})
    if request is not None:
        with source_module(request):
            # Each worker is a fresh process. Hash the actually loaded import
            # closure, including dynamic imports and cross-package helpers.
            for name, module in tuple(sys.modules.items()):
                path = getattr(module, '__file__', None)
                if (path and Path(path).is_file() and Path(path).resolve() != Path(__file__).resolve()
                        and name != '_fuga_triton_source' and name != 'triton' and not name.startswith('triton.')):
                    paths['import:' + name] = path
    dependencies.update({name: file_digest(path) for name, path in paths.items()})
    return {"producer": {"name": "triton", "version": triton.__version__,
                         "python_version": platform.python_version(), "dependencies": dependencies,
                         "environment": get_cache_invalidating_env_vars()},
            "adapter_version": ADAPTER_VERSION}


def artifact(request, compiled, names, producer=None):
    """Translate public signature and launch metadata, without parsing compiler IR."""
    producer = producer if producer is not None else probe()["producer"]
    metadata = compiled.metadata
    require(metadata.name == compiled.name and metadata.triton_version == producer["version"],
            "compiled entry/version metadata differs from the selected compiler")
    require(dataclasses.asdict(metadata.target) == request["target"],
            "compiled target differs from request")
    require(isinstance(compiled.asm.get("cubin"), bytes) and compiled.asm["cubin"],
            "compiler did not return a cubin")
    for resource in ("global_scratch_size", "profile_scratch_size", "tmem_size"):
        require(getattr(metadata, resource, None) == 0,
                f"unsupported launch resource {resource}={getattr(metadata, resource, None)!r}")
    require(metadata.num_ctas == 1 and not metadata.tensordesc_meta
            and metadata.launch_cooperative_grid is False and metadata.launch_pdl is False
            and not metadata.instrumentation_mode,
            "unsupported clustered, tensor-descriptor, cooperative, PDL or instrumented launch")
    require(type(metadata.shared) is int and 0 <= metadata.shared < 2**31,
            "invalid dynamic shared memory size")
    require(metadata.warp_size == 32 and type(metadata.num_warps) is int
            and 0 < metadata.num_warps <= 32, "invalid CUDA thread-block dimensions")
    # Triton 3.9 NVIDIA driver.c: flattened non-constexpr data parameters,
    # followed by global_scratch and profile_scratch. No descriptor/tuple args
    # are accepted by this flat pointer/scalar interface, so no ABI expansion occurs.
    arguments = [request["source_bindings"][name] if request["signature"][name].startswith("*")
                 else {"type": request["signature"][name], "value": request["source_bindings"][name]}
                 for name in names] + ["null", "null"]
    return {"protocol_version": 2, "format": "cubin",
            "payload": base64.b64encode(compiled.asm["cubin"]).decode("ascii"),
            "entry": compiled.name, "target": request["target"],
            "producer": producer,
            "launch": {"grid": request["grid"],
                       "block": [metadata.num_warps * metadata.warp_size, 1, 1],
                       "shared_bytes": metadata.shared},
            "arguments": arguments,
            "required_element_types": sorted({ELEMENT_TYPES[value]
                                               for value in request["signature"].values() if value in ELEMENT_TYPES})}


@dataclasses.dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: object

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def stride(self, axis=None):
        strides = tuple(math.prod(self.shape[index + 1:]) for index in range(self.ndim))
        return strides if axis is None else strides[axis]


def decode_constant(value):
    import triton.language as tl
    if isinstance(value, dict):
        if set(value) == {"__triton_dtype__"}:
            dtype = getattr(tl, value["__triton_dtype__"], None)
            require(isinstance(dtype, tl.dtype), "unknown Triton constexpr dtype")
            return dtype
        return {name: decode_constant(item) for name, item in value.items()}
    if isinstance(value, list):
        return tuple(decode_constant(item) for item in value)
    return value


def compile_request(request):
    require(type(request.get("protocol_version")) is int and request["protocol_version"] == 2,
            "worker requires protocol_version=2")
    producer = probe(request)["producer"]
    signature = request["signature"]
    constants = {name: decode_constant(value) for name, value in request["constants"].items()}
    require(set(request["source_bindings"]) == set(signature), "source_bindings must cover the runtime signature")
    triton = load_triton()
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDAOptions
    from triton.runtime.autotuner import Autotuner, Heuristics
    options = dict(request["fixed_options"])
    config_index = options.pop('autotune_config', None)
    require(set(options) <= {field.name for field in dataclasses.fields(CUDAOptions)},
            "unknown Triton CUDA compiler option")
    require(not set(options).intersection({"arch", "backend_name", "ir_override", "extern_libs", "ptx_options"}),
            "target/source overrides, custom extern_libs and free-form ptxas options are not supported")
    with source_module(request) as module:
        function = getattr(module, request["entry"])
        heuristics = []
        while isinstance(function, (Autotuner, Heuristics)):
            if isinstance(function, Autotuner):
                require(type(config_index) is int and 0 <= config_index < len(function.configs),
                        "autotuned kernels require an explicit valid config index; ASK does not benchmark inputs")
                config = function.configs[config_index]
                require(config.pre_hook is None and config.ir_override is None
                        and not function.reset_to_zero and not function.restore_value
                        and not function.user_defined_pre_hook and not function.user_defined_post_hook,
                        "autotune hooks and input mutation require an explicit launch contract")
                selected = config.all_kwargs()
                constants = {**config.kwargs, **constants}
                options = {**{name: value for name, value in selected.items() if name not in config.kwargs}, **options}
            else:
                heuristics.append(function.values)
            function = function.fn
        require(isinstance(function, triton.JITFunction), "entry must be a Triton JIT function")
        defaults = {name: parameter.default for name, parameter in inspect.signature(function.fn).parameters.items()
                    if parameter.default is not inspect.Parameter.empty}
        constexprs = {function.arg_names[index] for index in function.constexprs}
        constants = {**{name: value for name, value in defaults.items() if name in constexprs}, **constants}
        import triton.language as tl
        metadata = {**request["source_bindings"], **constants, **options}
        for name, dtype in signature.items():
            if dtype.startswith('*'):
                element = {'i1': 'int1', 'i32': 'int32', 'i64': 'int64', 'fp16': 'float16',
                           'bf16': 'bfloat16', 'fp32': 'float32'}[dtype[1:]]
                metadata[name] = TensorMetadata(shape=tuple(request['port_shapes'][request['source_bindings'][name]]),
                                                 dtype=getattr(tl, element))
        for group in heuristics:
            for name, compute in group.items():
                constants[name] = metadata[name] = compute(metadata)
        names = [name for name in function.arg_names if name not in constexprs]
        require(constexprs == set(constants) and set(names) == set(signature),
                "source parameters disagree with runtime and constexpr bindings")
        ordered_signature = {name: signature[name] for name in names}
        triton.knobs.compilation.always_compile = True
        compiled = triton.compile(ASTSource(function, ordered_signature, constexprs=constants),
                                  target=GPUTarget(**request["target"]), options=options)
        require(compiled.src.signature == ordered_signature, "compiled source signature differs from request")
        return artifact(request, compiled, names, producer)


@contextlib.contextmanager
def compiler_output_to_stderr():
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    temporary = None
    try:
        with tempfile.TemporaryDirectory(prefix="fuga-python-cache-") as cache:
            previous_prefix, previous_write = sys.pycache_prefix, sys.dont_write_bytecode
            sys.pycache_prefix, sys.dont_write_bytecode = cache, True
            try:
                with compiler_output_to_stderr():
                    if args.probe:
                        result = probe(json.loads(args.request.read_text(encoding="utf-8")) if args.request else None)
                    else:
                        require(args.request is not None and args.output is not None,
                                "provide --request and --output, or --probe")
                        result = compile_request(json.loads(args.request.read_text(encoding="utf-8")))
            finally:
                sys.pycache_prefix, sys.dont_write_bytecode = previous_prefix, previous_write
        if args.probe:
            print(json.dumps(result, sort_keys=True))
        else:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=args.output.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(result, stream, allow_nan=False, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, args.output)
    except Exception as error:
        print(json.dumps({"error": {"code": type(error).__name__, "message": str(error)}}), file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
