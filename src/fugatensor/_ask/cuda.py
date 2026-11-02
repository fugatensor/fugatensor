"""CUDA artifact validation and packaging; execution uses the ASK C++ runtime.

Owned code generation consumes LLVM IR; external cubin/PTX is loaded unchanged.
"""
from __future__ import annotations

import base64
import binascii
import ctypes
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import tempfile

from .cache import IRCache, fingerprint_file
from ._abi import parameter_layout, parameter_types, scalar_bytes


class CUDAError(RuntimeError):
    pass


_STORAGE_TYPES = {"bool": ctypes.c_uint8, "i32": ctypes.c_int32, "i64": ctypes.c_int64, "f32": ctypes.c_float,
                  "f16": ctypes.c_uint16, "bf16": ctypes.c_uint16}


def _require(condition, message):
    if not condition:
        raise CUDAError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _artifact_digest(artifact):
    return _digest(json.dumps(artifact, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))


def _kernel_contract(artifact):
    """Validate the compiler-independent CUDA invocation schema (version 2).

    A producer adapter supplies concrete port shapes, argument bindings, launch
    dimensions and backend options. The runner does not reconstruct these from
    a producer's DSL constants, scheduling metadata or private proof records.
    """
    _require(isinstance(artifact, dict), "artifact must be an object")
    format_ = artifact.get("format")
    _require(format_ in ("llvm-ir", "cubin", "ptx"), "expected LLVM IR, cubin or PTX artifact")
    _require(isinstance(artifact.get("payload"), str) and bool(artifact["payload"]),
             "artifact must carry its code payload")
    if format_ != "llvm-ir":
        _require(type(artifact.get("protocol_version")) is int and artifact["protocol_version"] == 2,
                 "external executable protocol version 2 is required")
        _external_payload(artifact)
    entry = artifact.get("entry")
    _require(isinstance(entry, str) and entry.isascii() and entry.isidentifier(), "unsupported entry name")
    target = artifact.get("target")
    _require(isinstance(target, dict) and target.get("backend") == "cuda"
             and type(target.get("arch")) is int and 10 <= target["arch"] <= 999,
             "CUDA runtime requires an explicit ordinary integer sm architecture")
    invocation = artifact.get("invocation")
    _require(isinstance(invocation, dict) and invocation.get("kind") == "cuda-kernel"
             and invocation.get("version") == 2,
             "CUDA invocation version 2 is required; re-export legacy artifacts through their compiler adapter")
    for field in ("grid", "block"):
        dimensions = invocation.get(field)
        _require(isinstance(dimensions, list) and len(dimensions) == 3
                 and all(type(n) is int and 0 < n < 2**32 for n in dimensions),
                 f"CUDA {field} must contain three positive 32-bit dimensions")
    _require(type(invocation.get("shared_bytes")) is int and 0 <= invocation["shared_bytes"] < 2**32,
             "CUDA shared_bytes must be a nonnegative 32-bit integer")
    arguments = invocation.get("arguments")
    _require(isinstance(arguments, list) and 0 < len(arguments) <= 64,
             "CUDA ABI requires between 1 and 64 declared arguments")
    bindings = {}
    for index, argument in enumerate(arguments):
        _require(isinstance(argument, dict) and type(argument.get("argument")) is int
                 and argument["argument"] == index, "invalid CUDA argument index")
        if argument.get("kind") == "scalar":
            _require(format_ != "llvm-ir" and set(argument) == {"argument", "kind", "type", "value"},
                     "scalar argument requires type and value")
            try:
                scalar_bytes(argument["type"], argument["value"])
            except ValueError as error:
                raise CUDAError(str(error)) from error
            continue
        _require(argument.get("llvm_type") in ("ptr", "ptr addrspace(1)") if format_ == "llvm-ir"
                 else argument.get("kind") == "pointer", "invalid CUDA pointer argument mapping")
        binding = argument.get("binding")
        _require(isinstance(binding, str) and (binding == "null" or
                 re.fullmatch(r"(?:input|output):(?:0|[1-9][0-9]*)", binding)),
                 "CUDA bindings must name input:N/output:N buffers or explicit null pointers")
        if binding != "null":
            _require(binding not in bindings, "logical buffer is mapped to multiple kernel arguments")
            bindings[binding] = index
    ports = artifact.get("port_map")
    _require(isinstance(ports, list) and len(ports) == len(bindings),
             "each non-null pointer argument requires one typed logical port")
    logical_ports = {}
    for port in ports:
        _require(isinstance(port, dict), "logical ports must be objects")
        binding, dtype = port.get("logical_port"), port.get("element_type")
        _require(isinstance(binding, str) and binding in bindings and binding not in logical_ports
                 and type(port.get("argument")) is int and port["argument"] == bindings[binding],
                 "logical port and invocation argument mappings disagree")
        _require(isinstance(dtype, str) and dtype in _STORAGE_TYPES, "unsupported CUDA port element type")
        shape = port.get("shape")
        _require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape),
                 "CUDA ports require concrete nonnegative shapes")
        width = ctypes.sizeof(_STORAGE_TYPES[dtype])
        _require(math.prod(shape) <= (2**63 - 1) // width, "CUDA tensor byte count overflows")
        _require(port.get("layout") == "contiguous_row_major", "CUDA ports require contiguous row-major layout")
        _require("storage_bytes" not in port or
                 (type(port["storage_bytes"]) is int and port["storage_bytes"] == width),
                 "port storage byte width disagrees with its element type")
        logical_ports[binding] = port
    ordered = []
    for kind in ("input", "output"):
        group = [port for name, port in logical_ports.items() if name.startswith(kind + ":")]
        _require(all(f"{kind}:{index}" in logical_ports for index in range(len(group))),
                 f"CUDA {kind} port indices must be consecutive starting at zero")
        ordered.append(tuple(logical_ports[f"{kind}:{index}"] for index in range(len(group))))
    incoming, outgoing = ordered
    _require(bool(outgoing), "CUDA graph requires at least one output")
    if format_ == "llvm-ir":
        _fp_contract(artifact)
    else:
        _require(invocation.get("completion") == "device-event", "external completion must use a device event")
        _require(isinstance(artifact.get("producer"), dict), "external artifact requires producer metadata")
    _required_types(artifact)
    return entry, invocation, incoming, outgoing


def _contract(artifact):
    """Compatibility contract for the one-input, same-type host-buffer runner.

    Multi-port and mixed-storage kernels use PreparedCUDAKernel.submit_many.
    Code generation itself accepts the complete normalized pointer ABI.
    """
    entry, invocation, inputs, outputs = _kernel_contract(artifact)
    _require(len(inputs) == len(outputs) == 1,
             "host-buffer execute requires one input and one output; use prepare().submit_many")
    incoming, outgoing = inputs[0], outputs[0]
    _require(incoming["element_type"] == outgoing["element_type"] and incoming["shape"] == outgoing["shape"],
             "host-buffer execute requires matching input/output shapes and element types; use prepare().submit_many")
    return entry, invocation, math.prod(incoming["shape"]), incoming["element_type"]


def _required_types(artifact, dtype=None):
    required = artifact.get("required_element_types")
    _require(isinstance(required, list) and
             all(isinstance(value, str) and value in _STORAGE_TYPES for value in required),
             "normalized artifact requires explicit required_element_types")
    return {*required, *(port["element_type"] for port in artifact["port_map"]),
            *(() if dtype is None else (dtype,))}


def _storage_dtype(inputs, outputs):
    """Retain the old receipt field when every port uses one storage dtype."""
    types = {port["element_type"] for port in (*inputs, *outputs)}
    return next(iter(types)) if len(types) == 1 else None


def _fp_contract(artifact, dtype=None):
    """Read ASK-owned LLVM code generation policy."""
    options = artifact.get("codegen")
    _require(isinstance(options, dict) and options.get("fp_contract") in ("on", "off", "fast")
             and type(options.get("ptx_version")) is int and 10 <= options["ptx_version"] <= 999,
             "normalized CUDA codegen requires fp_contract=on/off/fast and an explicit PTX version")
    return options["fp_contract"]


_CODEGEN_CACHE_VERSION = "ask.cuda.executable.v2"


def _codegen_tool(path):
    try:
        return fingerprint_file(path)
    except OSError as exc:
        raise CUDAError(f"cannot fingerprint CUDA compiler tool {path}: {exc}") from exc


def compile_cuda(llvm_ir, artifact, output_directory, *, llc, ptxas, cache_dir=None):
    """Compile supplied LLVM bytes, optionally reusing ASK's verified disk artifacts.

    llc and ptxas are explicit executable paths. No target or launch policy is
    selected here. A cache hit skips both tools; it still validates the complete
    artifact and materializes PTX, cubin and a receipt in the requested directory.
    No producer-supplied binary is reused. Cache counters describe this request.
    """
    entry, invocation, inputs, outputs = _kernel_contract(artifact)
    _require(artifact["format"] == "llvm-ir", "compile_cuda requires owned LLVM IR; use install_external for cubin/PTX")
    dtype = _storage_dtype(inputs, outputs)
    arch = artifact["target"]["arch"]
    required = _required_types(artifact, dtype)
    _require("bf16" not in required or arch >= 80, "bf16 compilation requires target sm80 or newer")
    fp_contract = _fp_contract(artifact, dtype)
    llvm_ir, output_directory = Path(llvm_ir).resolve(), Path(output_directory).resolve()
    llc, ptxas = Path(llc).resolve(), Path(ptxas).resolve()
    source = llvm_ir.read_bytes()
    _require(source == artifact["payload"].encode("utf-8"),
             "LLVM file differs from artifact payload; bind the transformed IR explicitly")
    output_directory.mkdir(parents=True, exist_ok=True)
    ptx = output_directory / "kernel.ptx"
    cubin = output_directory / "kernel.cubin"
    receipt = output_directory / "codegen.json"
    _require(not any(path.exists() for path in (ptx, cubin, receipt)),
             "codegen destination already contains artifacts; use a fresh directory")
    ptx_version = artifact["codegen"]["ptx_version"]
    options = [["-march=nvptx64", f"-mcpu=sm_{arch}", f"-mattr=+ptx{ptx_version}", "--fp-contract=" + fp_contract],
               [f"--gpu-name=sm_{arch}", "-v"]]

    def commands_for(directory):
        return [[str(llc), *options[0], str(llvm_ir), "-o", str(directory / "kernel.ptx")],
                [str(ptxas), *options[1], str(directory / "kernel.ptx"), "-o", str(directory / "kernel.cubin")]]

    tools = {"llc": _codegen_tool(llc), "ptxas": _codegen_tool(ptxas)} if cache_dir is not None else None

    def build():
        # Failed tool invocations never leave an apparently complete local
        # result or publish a cache entry. Only finished bytes leave staging.
        with tempfile.TemporaryDirectory(prefix="ask-cuda-codegen-") as temporary:
            staging = Path(temporary)
            logs = []
            for command in commands_for(staging):
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise CUDAError(f"code generation failed: {exc}") from exc
                if result.returncode:
                    raise CUDAError(f"code generation failed ({result.returncode}): {' '.join(command)}\n"
                                    + result.stdout + result.stderr)
                logs.append(result.stdout + result.stderr)
            _require(llvm_ir.read_bytes() == source, "LLVM input changed during compilation")
            if tools is not None:
                _require(tools == {"llc": _codegen_tool(llc), "ptxas": _codegen_tool(ptxas)},
                         "CUDA compiler tool changed during compilation")
            return {"kernel.ptx": (staging / "kernel.ptx").read_bytes(),
                    "kernel.cubin": (staging / "kernel.cubin").read_bytes(),
                    "build.json": json.dumps({"compiler_logs": logs}, sort_keys=True).encode()}

    if cache_dir is None:
        built = build()
        stats = {"key": None, "status": "disabled", "reason": "not_configured",
                 "counters": {"requests": 0, "hits": 0, "misses": 0, "invalid": 0, "builds": 1}}
    else:
        # Reuse IRCache's process/thread locks, byte checksums, atomic publish
        # and corrupt-entry recovery. Its key vocabulary also describes this
        # later compilation stage: graph is LLVM, no DSL remains to compile.
        key = {
            "schema_version": 1, "pipeline_version": _CODEGEN_CACHE_VERSION,
            "graph": {"llvm_sha256": _digest(source)}, "dsl": None,
            "config": {"artifact_sha256": _artifact_digest(artifact), "options": options,
                       "environment_sha256": _digest(json.dumps(dict(os.environ), sort_keys=True).encode())},
            "llvm_revision": "binary-sha256:" + tools["llc"]["sha256"],
            "producer": artifact.get("producer", {}),
            "adapter": {"codegen_sha256": _digest(Path(__file__).read_bytes())},
            "tools": tools, "target": artifact["target"],
            "shape": artifact["port_map"],
            "numerics": {"fp_contract": fp_contract, "required_element_types": sorted(required)},
        }
        built, stats = IRCache(cache_dir).get_or_create(key, build)
    _require(llvm_ir.read_bytes() == source, "LLVM input changed during compilation")
    if tools is not None:
        _require(tools == {"llc": _codegen_tool(llc), "ptxas": _codegen_tool(ptxas)},
                 "CUDA compiler tool changed during cache access")
    logs = json.loads(built["build.json"])["compiler_logs"]
    result = {"schema_version": 1, "entry": entry, "arch": arch, "element_type": dtype,
              "fp_contract": fp_contract,
              "artifact_sha256": _artifact_digest(artifact),
              "llvm_ir": str(llvm_ir), "llvm_sha256": _digest(source),
              "ptx": str(ptx), "ptx_sha256": _digest(built["kernel.ptx"]),
              "cubin": str(cubin), "cubin_sha256": _digest(built["kernel.cubin"]),
              "commands": commands_for(output_directory), "compiler_logs": logs,
              "commands_role": "reproduction", "compiler_invocations": 0 if stats["status"] == "hit" else 2,
              "executable_cache": stats,
              "invocation": invocation, "original_producer_binary_used": False}
    created = []
    try:
        for path, payload in ((ptx, built["kernel.ptx"]), (cubin, built["kernel.cubin"]),
                              (receipt, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(payload)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return result


def _external_payload(artifact):
    """Decode executable bytes without translating or modifying producer code."""
    payload = artifact["payload"]
    if artifact["format"] == "cubin":
        try:
            binary = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as error:
            raise CUDAError("cubin payload must be valid base64") from error
        _require(len(binary) >= 64 and binary.startswith(b"\x7fELF"),
                 "cubin payload must contain a CUDA ELF image")
        return binary
    _require(artifact["format"] == "ptx" and "\0" not in payload,
             "PTX payload must be text without embedded NUL bytes")
    return payload.encode("utf-8")


def install_external(artifact, output_directory):
    """Package a producer cubin or PTX without invoking an ASK compiler tool.

    PTX remains PTX on disk. The CUDA driver performs JIT compilation when the
    session loads it; no llc or ptxas executable is needed for this path.
    """
    entry, invocation, inputs, outputs = _kernel_contract(artifact)
    _require(artifact["format"] in ("cubin", "ptx"), "external executable must be cubin or PTX")
    binary = _external_payload(artifact)
    format_ = artifact["format"]
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path, receipt = directory / ("kernel." + format_), directory / "codegen.json"
    _require(not path.exists() and not receipt.exists(), "codegen destination already contains artifacts; use a fresh directory")
    result = {"schema_version": 2, "format": format_, "entry": entry,
              "arch": artifact["target"]["arch"], "element_type": _storage_dtype(inputs, outputs),
              "artifact_sha256": _artifact_digest(artifact), "payload_sha256": _digest(binary),
              format_: str(path), format_ + "_sha256": _digest(binary),
              "invocation": invocation, "compiler_invocations": 0,
              "original_producer_binary_used": format_ == "cubin"}
    created = []
    try:
        for destination, contents in ((path, binary), (receipt,
                (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())):
            with destination.open("xb") as stream:
                created.append(destination)
                stream.write(contents)
    except BaseException:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    return result


def _validated_image(artifact, codegen):
    """Validate a package before any driver call, returning its exact code bytes."""
    entry, invocation, inputs, outputs = _kernel_contract(artifact)
    dtype = _storage_dtype(inputs, outputs)
    external = artifact["format"] != "llvm-ir"
    _require(isinstance(codegen, dict) and type(codegen.get("schema_version")) is int
             and codegen["schema_version"] == (2 if external else 1), "unsupported codegen receipt version")
    _require(codegen.get("entry") == entry and codegen.get("arch") == artifact["target"]["arch"]
             and codegen.get("invocation") == invocation and codegen.get("element_type") == dtype,
             "codegen and execution contracts disagree")
    _require(codegen.get("artifact_sha256") == _artifact_digest(artifact),
             "artifact digest differs from the compiled artifact")
    if external:
        format_ = artifact["format"]
        _require(codegen.get("format") == format_, "executable format differs from its artifact")
        _require(set(codegen) & {"llvm_ir", "cubin", "ptx"} == {format_}
                 and isinstance(codegen.get(format_), str) and bool(codegen[format_]),
                 "external receipt must reference exactly its executable format")
        binary = Path(codegen[format_]).read_bytes()
        _require(binary == _external_payload(artifact) and _digest(binary) == codegen.get("payload_sha256")
                 and _digest(binary) == codegen.get(format_ + "_sha256"), "executable payload digest mismatch")
    else:
        _require(codegen.get("fp_contract") == _fp_contract(artifact, dtype), "codegen and execution contracts disagree")
        source_digest = _digest(artifact["payload"].encode("utf-8"))
        _require(source_digest == codegen.get("llvm_sha256")
                 and _digest(Path(codegen["llvm_ir"]).read_bytes()) == source_digest,
                 "LLVM input digest differs from the compiled IR")
        binary = Path(codegen["cubin"]).read_bytes()
        _require(_digest(binary) == codegen.get("cubin_sha256"), "cubin digest mismatch")
    return binary


def execute(artifact, codegen, values, *, device_ordinal=0):
    """Host-value convenience entry using the same native runtime as JIT execution."""
    from .gpu import CUDAKernel
    from .runtime import ExecutionSession, _encode

    entry, invocation, size, dtype = _contract(artifact)
    binary = _validated_image(artifact, codegen)
    data = _encode(values, dtype, size)
    shape = next(port["shape"] for port in artifact["port_map"] if port["logical_port"] == "input:0")
    # Reject broken contracts, receipts and host values before opening a session.
    with ExecutionSession(device=device_ordinal) as session:
        prepared = CUDAKernel(artifact, codegen, {}, {}).prepare(session)
        source = session.tensor(shape, dtype).copy_from_buffer(data)
        destination = session.tensor(shape, dtype)
        prepared.submit(source, destination).wait()
        return {"output": destination.to_host(), "entry": entry, "element_type": dtype,
                "device": {"ordinal": device_ordinal, "name": session._native.device_name,
                           "compute_capability": list(session.capability)},
                "grid": invocation["grid"], "block": invocation["block"],
                "parameter_layout": parameter_layout(parameter_types(invocation)),
                "code_sha256": _digest(binary),
                **({"llvm_sha256": codegen["llvm_sha256"], "cubin_sha256": codegen["cubin_sha256"]}
                   if artifact["format"] == "llvm-ir" else {artifact["format"] + "_sha256": _digest(binary)}),
                "device_execution_performed": True,
                "original_producer_binary_used": artifact["format"] == "cubin"}
