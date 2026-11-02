"""Validate packages once; native programs own binding, scheduling and resources."""
from __future__ import annotations

import copy
import ctypes
import math

from ._abi import parameter_types, scalar_bytes
from .cuda import (_kernel_contract, _validated_image, _required_types,
                   _require, _storage_dtype, _STORAGE_TYPES)


def validate_kernel_package(kernel):
    artifact, codegen = copy.deepcopy(kernel.artifact), copy.deepcopy(kernel.codegen)
    entry, invocation, inputs, outputs = _kernel_contract(artifact)
    binary = _validated_image(artifact, codegen)
    return artifact, codegen, binary, entry, invocation, inputs, outputs


def _load_step(kernel, session, input_indices, output_indices):
    artifact, _, binary, entry, invocation, inputs, outputs = validate_kernel_package(kernel)
    dtype = _storage_dtype(inputs, outputs)
    native = session.load(binary, entry, arch=artifact["target"]["arch"],
                          required_types=sorted(_required_types(artifact, dtype)),
                          parameter_types=parameter_types(invocation))
    bindings = {port["logical_port"]: index for port, index in
                zip((*inputs, *outputs), (*input_indices, *output_indices))}
    arguments = [scalar_bytes(arg["type"], arg["value"]) if arg.get("kind") == "scalar"
                 else None if arg["binding"] == "null" else bindings[arg["binding"]]
                 for arg in invocation["arguments"]]
    return {"kernel": native, "arguments": arguments,
            "grid": invocation["grid"], "block": invocation["block"],
            "shared_bytes": invocation["shared_bytes"],
            "inputs": input_indices, "outputs": output_indices}


def _nbytes(port):
    return math.prod(port["shape"]) * ctypes.sizeof(_STORAGE_TYPES[port["element_type"]])


class _Prepared:
    @property
    def inputs(self):
        return copy.deepcopy(self._inputs)

    @property
    def outputs(self):
        return copy.deepcopy(self._outputs)

    @property
    def shape(self):
        _require(len(self._outputs) == 1, "multiple outputs have separate shapes; use outputs")
        return tuple(self._outputs[0]["shape"])

    @property
    def dtype(self):
        _require(len(self._outputs) == 1, "multiple outputs have separate dtypes; use outputs")
        return self._outputs[0]["element_type"]

    def _check_ports(self, inputs, outputs):
        from .runtime import DeviceTensor
        for kind, tensors, ports in (("inputs", inputs, self._inputs), ("outputs", outputs, self._outputs)):
            _require(isinstance(tensors, (tuple, list)) and len(tensors) == len(ports),
                     f"program expects {len(ports)} {kind}")
            for tensor, port in zip(tensors, ports):
                _require(isinstance(tensor, DeviceTensor), "program ports require DeviceTensor values")
                _require(tuple(tensor.shape) == tuple(port["shape"]) and tensor.dtype == port["element_type"],
                         "program port shape/dtype differs from its compiled type")
        # The native program checks sizes, session ownership, writable outputs
        # and aliases together before enqueueing any kernel.
        for tensor in (*inputs, *outputs):
            tensor._prepare_access(self._session)

    def submit(self, input, output):
        return self.submit_many([input], [output])

    def submit_many(self, inputs, outputs):
        session = self._session
        with session._lock:
            session._check_open()
            self._check_ports(inputs, outputs)
            session.reap()
            event = self._native.submit([value._buffer for value in inputs],
                                        [value._buffer for value in outputs])
            return session._retain(event, (*inputs, *outputs))


class PreparedCUDAKernel(_Prepared):
    """One kernel using the same native program machinery as a multi-step plan."""
    def __init__(self, kernel, session, *, graph=False):
        _require(type(graph) is bool, "graph must be bool")
        _, _, self._inputs, self._outputs = _kernel_contract(copy.deepcopy(kernel.artifact))
        self._session = session
        ninputs = len(self._inputs)
        indices = list(range(ninputs + len(self._outputs)))
        self._step = _load_step(kernel, session, indices[:ninputs], indices[ninputs:])
        self._kernel = self._step["kernel"]
        self._native = session._native.prepare_program(
            [_nbytes(port) for port in (*self._inputs, *self._outputs)],
            indices[:ninputs], indices[ninputs:], [self._step], graph)

    def _parameters(self, inputs, outputs):
        # Low-level diagnostic access to the frozen ABI; normal submission
        # binds these indices entirely in C++.
        self._check_ports(inputs, outputs)
        tensors = (*inputs, *outputs)
        return [tensors[arg] if type(arg) is int else arg for arg in self._step["arguments"]]


class PreparedCUDAProgram(_Prepared):
    """A checked native plan, optionally instantiated as a CUDA Graph."""
    def __init__(self, program, session, *, graph=False):
        from .plan import validate_plan, validate_kernels
        _require(type(graph) is bool, "graph must be bool")
        self._plan = validate_plan(program.plan)
        validate_kernels(self._plan, program.kernels)
        self._session = session
        buffers = {buffer["id"]: buffer for buffer in self._plan["buffers"]}
        indices = {name: index for index, name in enumerate(buffers)}
        for kind in ("inputs", "outputs"):
            setattr(self, "_" + kind, [dict(logical_port=f"{kind[:-1]}:{index}",
                shape=list(buffers[name]["shape"]), element_type=buffers[name]["element_type"],
                layout="contiguous_row_major") for index, name in enumerate(self._plan[kind])])
        steps = [_load_step(program.kernels[step["id"]], session,
                            [indices[name] for name in step["inputs"]],
                            [indices[name] for name in step["outputs"]]) for step in self._plan["steps"]]
        self._native = session._native.prepare_program(
            [_nbytes(buffer) for buffer in buffers.values()],
            [indices[name] for name in self._plan["inputs"]],
            [indices[name] for name in self._plan["outputs"]], steps, graph)

    @property
    def plan(self):
        return copy.deepcopy(self._plan)
