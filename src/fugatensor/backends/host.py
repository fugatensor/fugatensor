"""Compile ASK graphs on first use and call their native host functions.

The Python code allocates and encodes tensors; arithmetic runs in the object
emitted by ASK. Each compilation has a fresh artifact directory. FugaTensor's
JIT owns specialization reuse, so this layer never reuses an unchecked .so.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from fugatensor._ask import Compiler, CompilerError, Module, TensorType
from fugatensor._ask.numerics import decode_float, encode_float

_STORAGE = {
    "bool": ctypes.c_uint8,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "f32": ctypes.c_uint32,
    "f16": ctypes.c_uint16,
    "bf16": ctypes.c_uint16,
}


@dataclass(frozen=True, init=False)
class HostStorage:
    """Owned contiguous memory with a concrete ASK shape and element type."""

    shape: tuple[int, ...]
    dtype: str
    _buffer: ctypes.Array

    def __init__(self, shape, dtype):
        tensor = TensorType(tuple(shape), dtype)
        if any(dim is None for dim in tensor.shape):
            raise ValueError("host storage needs a concrete shape")
        count = math.prod(tensor.shape)
        if count > (2**63 - 1) // ctypes.sizeof(_STORAGE[tensor.dtype]):
            raise ValueError("tensor byte count overflows the host ABI")
        object.__setattr__(self, "shape", tensor.shape)
        object.__setattr__(self, "dtype", tensor.dtype)
        # An empty tensor owns a valid pointer, but exposes zero logical bytes.
        object.__setattr__(self, "_buffer", (_STORAGE[tensor.dtype] * max(count, 1))())

    @classmethod
    def from_flat(cls, values, *, shape, dtype):
        result = cls(shape, dtype)
        values = list(values)
        if len(values) != result.size:
            raise ValueError("input length does not match the tensor shape")
        for index, value in enumerate(values):
            if result.dtype == "bool":
                if type(value) is not bool:
                    raise TypeError("bool tensors require Python bool values")
                result._buffer[index] = int(value)
            elif result.dtype in ("i32", "i64"):
                bits = 32 if result.dtype == "i32" else 64
                if type(value) is not int:
                    raise TypeError(f"{result.dtype} tensors require Python int values")
                if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
                    raise ValueError(f"input is outside the {result.dtype} range")
                result._buffer[index] = value
            else:
                # Store bits, not a Python-computed arithmetic result. ASK's
                # encoder rounds directly to the declared storage precision.
                result._buffer[index] = int.from_bytes(encode_float(value, result.dtype), "little")
        return result

    @classmethod
    def from_buffer(cls, data, *, shape, dtype, copy=True):
        """Import native contiguous elements, or raw bytes with an explicit dtype.

        The default owns a bulk copy. A borrowed buffer must be writable and
        aligned; the ctypes view retains its exporter for the storage lifetime.
        """
        if type(copy) is not bool:
            raise TypeError("copy must be bool")
        view = memoryview(data)
        if not view.c_contiguous:
            raise ValueError("tensor buffers must be C-contiguous")
        tensor = TensorType(tuple(shape), dtype)
        if any(dim is None for dim in tensor.shape):
            raise ValueError("host storage needs a concrete shape")
        width = ctypes.sizeof(_STORAGE[tensor.dtype])
        count = math.prod(tensor.shape)
        nbytes = count * width
        if nbytes > 2**63 - 1 or view.nbytes != nbytes:
            raise ValueError("buffer byte count does not match the tensor shape and dtype")
        format_ = view.format
        prefix = format_[0] if format_ and format_[0] in "@=<>!" else "@"
        format_ = format_[1:] if format_ and format_[0] in "@=<>!" else format_
        raw = format_ in ("B", "b", "c") and view.itemsize == 1
        formats = {"bool": ("?",), "i32": ("i", "l"), "i64": ("q", "l"),
                   "f32": ("f",), "f16": ("e",), "bf16": ()}
        native_endian = prefix in ("@", "=") or prefix == ("<" if sys.byteorder == "little" else ">")
        if not raw and (format_ not in formats[tensor.dtype] or view.itemsize != width or not native_endian):
            raise TypeError("typed buffer format must match the tensor dtype and native byte order; use bytes for raw bits")
        raw_view = view.cast("B")
        if copy:
            result = cls(tensor.shape, tensor.dtype)
            # Bulk copies preserve NaN payloads and low-precision bits. A
            # writable exporter exposes its pointer without a staging copy.
            if nbytes:
                source = ((ctypes.c_char * nbytes).from_buffer(raw_view) if not view.readonly
                          else data if type(data) is bytes else raw_view.tobytes())
                ctypes.memmove(result.data_ptr, source, nbytes)
            return result
        if view.readonly:
            raise ValueError("copy=False requires a writable buffer")
        if not nbytes:
            return cls(tensor.shape, tensor.dtype)
        borrowed = (_STORAGE[tensor.dtype] * count).from_buffer(raw_view)
        if ctypes.addressof(borrowed) % width:
            raise ValueError("borrowed buffer is not aligned for its tensor dtype")
        result = cls.__new__(cls)
        object.__setattr__(result, "shape", tensor.shape)
        object.__setattr__(result, "dtype", tensor.dtype)
        object.__setattr__(result, "_buffer", borrowed)
        return result

    def to_buffer(self, *, copy=True):
        """Return a read-only byte view, preserving the exact element bits."""
        if type(copy) is not bool:
            raise TypeError("copy must be bool")
        view = memoryview(self._buffer).cast("B")[:self.nbytes]
        return memoryview(view.tobytes()) if copy else view.toreadonly()

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * ctypes.sizeof(_STORAGE[self.dtype])

    @property
    def data_ptr(self):
        return ctypes.addressof(self._buffer)

    def to_flat(self):
        if self.dtype == "bool":
            return [bool(value) for value in self._buffer[:self.size]]
        if self.dtype in ("i32", "i64"):
            return list(self._buffer[:self.size])
        width = ctypes.sizeof(_STORAGE[self.dtype])
        return [decode_float(int(value).to_bytes(width, "little"), self.dtype)
                for value in self._buffer[:self.size]]


def _ports(function, field, start):
    ports = function.get(field)
    if not isinstance(ports, list):
        raise CompilerError(f"ASK host ABI is missing {field}")
    descriptions = []
    for index, port in enumerate(ports, start):
        if not isinstance(port, dict) or port.get("kind") != "tensor":
            raise CompilerError("FugaTensor host JIT requires tensor input/output ports")
        if type(port.get("argument")) is not int or port["argument"] != index:
            raise CompilerError("ASK host ABI has an unexpected argument order")
        if port.get("dtype") not in _STORAGE or not isinstance(port.get("shape"), list):
            raise CompilerError("ASK host ABI has an unsupported tensor description")
        tensor = TensorType(tuple(port["shape"]), port["dtype"])
        if any(dim is None for dim in tensor.shape):
            raise CompilerError("ASK host ABI has a dynamic tensor shape")
        width = ctypes.sizeof(_STORAGE[tensor.dtype])
        if port.get("element_bytes") != width or port.get("minimum_alignment") != width:
            raise CompilerError("ASK host ABI element width/alignment does not match the dtype")
        descriptions.append(tensor)
    return tuple(descriptions)


class HostExecutable:
    """A loaded ASK function whose ABI was checked before exposing run()."""

    execution_plan = None

    def __init__(self, library, compiled, source, artifact_dir):
        self.report = compiled.diagnostics
        self.source = source
        self.artifact_dir = Path(artifact_dir)
        abi = compiled.abi
        if compiled.target != "host-object" or not isinstance(abi, dict) or abi.get("schema_version") != 1:
            raise CompilerError("ASK did not produce the supported host object ABI")
        functions = abi.get("functions")
        if not isinstance(functions, list) or len(functions) != 1:
            raise CompilerError("FugaTensor host JIT requires exactly one graph function")
        function = functions[0]
        if not isinstance(function, dict) or function.get("calling_convention") != "c":
            raise CompilerError("ASK host function must use the C calling convention")
        if function.get("layout") != "contiguous_row_major":
            raise CompilerError("FugaTensor host JIT requires contiguous row-major tensors")
        if function.get("requires_disjoint_outputs") is not True or function.get("outputs_must_not_alias_inputs") is not True:
            raise CompilerError("ASK host ABI has an unsupported aliasing contract")
        self.inputs = _ports(function, "inputs", 0)
        self.outputs = _ports(function, "outputs", len(self.inputs))
        if not self.outputs:
            raise CompilerError("FugaTensor JIT requires at least one output tensor")
        name = function.get("name")
        if not isinstance(name, str) or not name or "\0" in name:
            raise CompilerError("ASK host ABI is missing its entry symbol")
        self.entry = name
        try:
            self._library = ctypes.CDLL(str(library))
            self._function = getattr(self._library, self.entry)
        except (OSError, AttributeError) as error:
            raise CompilerError(f"cannot load ASK host entry {name!r}: {error}") from error
        self._function.argtypes = [ctypes.c_void_p] * (len(self.inputs) + len(self.outputs))
        self._function.restype = None

    def run(self, inputs):
        inputs = tuple(inputs)
        if len(inputs) != len(self.inputs):
            raise ValueError("input count does not match the compiled ASK function")
        for storage, expected in zip(inputs, self.inputs):
            if not isinstance(storage, HostStorage):
                raise TypeError("host JIT inputs must be HostStorage objects")
            if storage.shape != expected.shape or storage.dtype != expected.dtype:
                raise ValueError("input shape/dtype does not match this JIT specialization")
        outputs = [HostStorage(port.shape, port.dtype) for port in self.outputs]
        self._function(*(storage.data_ptr for storage in (*inputs, *outputs)))
        return outputs


class HostBackend:
    def __init__(self, compiler: Compiler, cache_dir, cc="cc"):
        self.compiler = compiler
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cc = str(cc)

    def _run(self, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=self.compiler.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CompilerError(f"host JIT could not run {command[0]!r}: {error}") from error
        if result.returncode:
            raise CompilerError("host JIT compilation failed", diagnostics=result.stderr or result.stdout,
                                returncode=result.returncode)
        return result

    def compile(self, module: Module | str):
        source = module.to_mlir() if isinstance(module, Module) else module
        if not isinstance(source, str):
            raise TypeError("host JIT compilation expects an ASK Module or MLIR text")
        capabilities = self.compiler.capabilities()
        if not capabilities.get("host_codegen"):
            raise CompilerError("this ASK build has no host code generation")
        cc = shutil.which(self.cc)
        if cc is None:
            raise CompilerError(f"host shared-library linker not found: {self.cc}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="host-", dir=self.cache_dir))
        source_path, obj = directory / "graph.mlir", directory / "graph.o"
        report_path, library = directory / "report.json", directory / "graph.so"
        source_path.write_text(source, encoding="utf-8")
        compiled = self.compiler.emit_host_object(source, obj)
        report_path.write_text(json.dumps(compiled.diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (directory / "abi.json").write_text(json.dumps(compiled.abi, indent=2) + "\n", encoding="utf-8")
        self._run([cc, "-shared", str(obj), "-o", str(library), "-lm"])
        return HostExecutable(library, compiled, source, directory)
