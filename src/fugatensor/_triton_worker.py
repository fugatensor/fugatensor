"""Compile user Triton source to a standalone cubin for ASK's worker protocol.

This file runs in the selected compiler environment, independently of FugaTensor
and ASK imports. The pinned Triton CUDA ABI appends two scratch pointers after
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
import os
from pathlib import Path
import platform
import sys
import tempfile

# Direct execution puts the package directory first, where triton.py is our
# registration API rather than the compiler distribution selected by Python.
if __name__ == "__main__":
    sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).parent.resolve()]

DEPENDENCIES_LOCK = Path(__file__).with_name("triton.dependencies.lock.json")
LOCK = json.loads(DEPENDENCIES_LOCK.read_text(encoding="utf-8"))
TRITON_VERSION = LOCK["compiler"]["version"]
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
    require(triton.__version__ == TRITON_VERSION,
            f"Triton CUDA ABI requires triton=={TRITON_VERSION}; found {triton.__version__}")
    return triton


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe():
    triton = load_triton()
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
    dependencies.update({name: file_digest(path) for name, path in paths.items()})
    return {"producer": {"name": "triton", "version": triton.__version__,
                         "python_version": platform.python_version(), "dependencies": dependencies},
            "adapter_version": ADAPTER_VERSION}


def artifact(request, compiled, names):
    """Translate public signature and launch metadata, without parsing compiler IR."""
    metadata = compiled.metadata
    require(metadata.name == compiled.name and metadata.triton_version == TRITON_VERSION,
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
    # Triton 3.7.1 NVIDIA driver.c: flattened non-constexpr data parameters,
    # followed by global_scratch and profile_scratch. No descriptor/tuple args
    # are accepted by this pointer-only interface, so no ABI expansion occurs.
    arguments = [request["source_bindings"][name] for name in names] + ["null", "null"]
    return {"protocol_version": 2, "format": "cubin",
            "payload": base64.b64encode(compiled.asm["cubin"]).decode("ascii"),
            "entry": compiled.name, "target": request["target"],
            "producer": probe()["producer"],
            "launch": {"grid": request["grid"],
                       "block": [metadata.num_warps * metadata.warp_size, 1, 1],
                       "shared_bytes": metadata.shared},
            "arguments": arguments,
            "required_element_types": sorted({ELEMENT_TYPES[value]
                                               for value in request["signature"].values()})}


def compile_request(request):
    require(type(request.get("protocol_version")) is int and request["protocol_version"] == 2,
            "worker requires protocol_version=2")
    signature, constants = request["signature"], request["constants"]
    require(signature and all(value in ELEMENT_TYPES for value in signature.values()),
            "worker requires named bool/i32/i64/f32/f16/bf16 data pointers")
    require(set(request["source_bindings"]) == set(signature),
            "source_bindings must cover the data-pointer signature")
    triton = load_triton()
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDAOptions
    options = request["fixed_options"]
    require(set(options) <= {field.name for field in dataclasses.fields(CUDAOptions)},
            "unknown Triton CUDA compiler option")
    require(not set(options).intersection({"arch", "backend_name", "ir_override", "extern_libs", "ptx_options"}),
            "target/source overrides, custom extern_libs and free-form ptxas options are not supported")
    # A source snapshot has no implicit user-module dependencies. This is a
    # packaging contract, not a sandbox for executing untrusted Python.
    tree = ast.parse(request["source"])
    for node in tree.body:
        if isinstance(node, ast.Import):
            require(all(alias.name in {"triton", "triton.language", "math"} for alias in node.names),
                    "source must be self-contained; unsupported import")
        elif isinstance(node, ast.ImportFrom):
            require(node.level == 0 and node.module in {"triton", "triton.language", "math"},
                    "source must be self-contained; unsupported import")
        elif isinstance(node, ast.FunctionDef):
            require(len(node.decorator_list) == 1 and ast.unparse(node.decorator_list[0]) == "triton.jit"
                    and not node.args.defaults and not any(node.args.kw_defaults),
                    "source functions must use @triton.jit without default arguments")
        else:
            require(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str), "unsupported top-level source statement")
    # Keep a real source file alive: Triton's JITFunction uses inspect to read it.
    with tempfile.TemporaryDirectory(prefix="fuga-triton-") as directory:
        path = Path(directory) / "kernel.py"
        path.write_text(request["source"], encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_fuga_triton_source", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            function = getattr(module, request["entry"])
            require(isinstance(function, triton.JITFunction), "entry must be a @triton.jit function")
            constexprs = {function.arg_names[index] for index in function.constexprs}
            names = [name for name in function.arg_names if name not in constexprs]
            require(constexprs == set(constants) and set(names) == set(signature),
                    "source parameters disagree with data-pointer and constexpr bindings")
            ordered_signature = {name: signature[name] for name in names}
            # ASK owns the outer cache. A miss must not reuse Triton's cached
            # cubin after a same-version assembler/library replacement.
            triton.knobs.compilation.always_compile = True
            compiled = triton.compile(ASTSource(function, ordered_signature, constexprs=constants),
                                      target=GPUTarget(**request["target"]), options=options)
            require(compiled.src.signature == ordered_signature,
                    "compiled source signature differs from request")
            return artifact(request, compiled, names)
        finally:
            sys.modules.pop(spec.name, None)


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
        with compiler_output_to_stderr():
            if args.probe:
                result = probe()
            else:
                require(args.request is not None and args.output is not None,
                        "provide --request and --output, or --probe")
                result = compile_request(json.loads(args.request.read_text(encoding="utf-8")))
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
