"""Trace tensor expressions into device-independent Fuga SSA."""
from __future__ import annotations

import math
from .ir import Module, TensorType, Value
from .dtypes import DType
from .devices import device as Device, resolve_device

from .tensor import normalize_dtype, normalize_shape
from ._state import active_trace


class TraceTensor:
    """A Fuga SSA value used only while an operation is being traced."""
    __slots__ = ("context", "value", "__weakref__")

    def __init__(self, context: TraceContext, value: Value):
        if not isinstance(value, Value) or value.owner is not context.graph.block:
            raise ValueError("traced value must belong to its TraceContext graph")
        if not isinstance(value.type, TensorType):
            raise TypeError("TraceTensor requires a Fuga tensor value")
        self.context = context
        self.value = value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.type.shape

    @property
    def dtype(self) -> DType:
        return DType(self.value.type.dtype)

    @property
    def device(self) -> Device:
        return resolve_device(self.context.engine.device if self.context.engine is not None else "cpu")

    def astype(self, dtype: DType | str) -> TraceTensor:
        value = self.context.graph.cast(self.value, dtype=normalize_dtype(dtype))
        return self.context.wrap(value)

    def reshape(self, *shape) -> TraceTensor:
        from ._shape import infer_reshape
        shape = infer_reshape(shape, math.prod(self.shape))
        return self.context.wrap(self.context.graph.reshape(self.value, shape))

    def detach(self) -> TraceTensor:
        return self.context.wrap(self.context.graph.stop_gradient(self.value))

    def _operand(self, other):
        if isinstance(other, TraceTensor):
            if other.context is not self.context:
                raise ValueError("traced operands must belong to the same graph")
            if other.shape != self.shape:
                raise ValueError("tensor operands require equal shapes; use broadcast_to explicitly")
            if other.dtype != self.dtype:
                raise TypeError("tensor operands require equal dtypes; use astype explicitly")
            return other.value
        if type(other) not in (bool, int, float):
            raise TypeError("traced expressions accept traced tensors or Python scalars; pass tensor captures as operation arguments")
        constant = self.context.graph.constant(other, dtype=self.dtype)
        return self.context.graph.splat(constant, self.shape)

    def _binary(self, operation, other, *, reverse=False):
        if self.dtype == "bool" and not isinstance(other, TraceTensor):
            raise TypeError("bool arithmetic is not defined; use astype to select a numeric type")
        other_value = self._operand(other)
        lhs, rhs = (other_value, self.value) if reverse else (self.value, other_value)
        value = getattr(self.context.graph, operation)(lhs, rhs)
        return self.context.wrap(value)

    def __add__(self, other):
        return self._binary("add", other)

    def __radd__(self, other):
        return self._binary("add", other, reverse=True)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __rmul__(self, other):
        return self._binary("mul", other, reverse=True)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def square(self):
        return self * self

    def __bool__(self):
        if self.shape != ():
            raise TypeError('control-flow conditions require a scalar Tensor with shape=()')
        raise TypeError('data-dependent control flow inside an isolated fun or VJP trace is unsupported')

    def tolist(self):
        raise TypeError("tensor values cannot be read while tracing an isolated operation")

    def __truediv__(self, other):
        from .ops import divide
        return divide(self, other)

    def __rtruediv__(self, other):
        from .ops import divide
        return divide(other, self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul
        return matmul(other, self)

    def __eq__(self, other):
        from .ops import compare
        return compare(self, other, 'eq')

    def __ne__(self, other):
        from .ops import compare
        return compare(self, other, 'ne')

    def __lt__(self, other):
        from .ops import compare
        return compare(self, other, 'lt')

    def __le__(self, other):
        from .ops import compare
        return compare(self, other, 'le')

    def __gt__(self, other):
        from .ops import compare
        return compare(self, other, 'gt')

    def __ge__(self, other):
        from .ops import compare
        return compare(self, other, 'ge')

    def sum(self, axis=None, *, keepdims=False, dtype=None):
        from .ops import sum
        return sum(self, axis, keepdims=keepdims, dtype=dtype)

    def amax(self, axis=None, *, keepdims=False):
        from .ops import amax
        return amax(self, axis, keepdims=keepdims)

    def mean(self, axis=None, *, keepdims=False, dtype=None):
        from .ops import mean
        return mean(self, axis, keepdims=keepdims, dtype=dtype)

    def transpose(self, permutation=None):
        from .ops import transpose
        return transpose(self, permutation)

    @property
    def T(self):
        return self.transpose()

    def broadcast_to(self, shape):
        from .ops import broadcast_to
        return broadcast_to(self, shape)

    def __repr__(self):
        return f"TraceTensor(shape={self.shape!r}, dtype={self.dtype!r}, value={self.value.name!r})"


class TraceContext:
    """Own one Fuga graph and infer its result types from the Python return."""
    def __init__(self, input_specs, *, engine=None, name="fuga_main"):
        self.engine = engine
        self.module = Module()
        self.input_specs = tuple((normalize_shape(shape), normalize_dtype(dtype))
                                 for shape, dtype in input_specs)
        self.graph = self.module.graph(
            name,
            inputs={f"arg{index}": TensorType(shape, dtype)
                    for index, (shape, dtype) in enumerate(self.input_specs)},
        )
        self.inputs = tuple(TraceTensor(self, value) for value in self.graph.inputs.values())
        self.output_specs = None
        self.output_is_tuple = None
        self._finished = False
        self.external_count = 0
        parent = active_trace.get()
        self.dependencies = parent.dependencies if parent is not None else {}

    @classmethod
    def from_graph(cls, graph, *, engine=None):
        """Attach tracing glue to an existing Fuga builder during transforms."""
        context = cls.__new__(cls)
        parent = active_trace.get()
        context.engine = engine if engine is not None else (parent.engine if parent is not None else None)
        context.module = graph.module
        context.graph = graph
        context.input_specs = tuple((value.type.shape, value.type.dtype)
                                    for value in graph.inputs.values() if isinstance(value.type, TensorType))
        context.inputs = tuple(TraceTensor(context, value) for value in graph.inputs.values()
                               if isinstance(value.type, TensorType))
        context.output_specs = context.output_is_tuple = None
        context._finished = False
        context.external_count = sum(op.opcode == 'external' for op in graph.operations)
        context.dependencies = parent.dependencies if parent is not None else {}
        return context

    def depend_on(self, definition):
        self.dependencies[definition] = definition._definition_key()

    def wrap(self, value: Value) -> TraceTensor:
        if self._finished:
            raise ValueError("cannot extend a completed trace")
        return TraceTensor(self, value)

    def finish(self, result) -> Module:
        if self._finished:
            raise ValueError("a trace can only be finished once")
        is_tuple = isinstance(result, tuple)
        outputs = result if is_tuple else (result,)
        if not outputs or any(not isinstance(value, TraceTensor) for value in outputs):
            raise TypeError("an operation trace must return a traced Tensor or a nonempty tuple of traced Tensors")
        if any(value.context is not self for value in outputs):
            raise ValueError("operation outputs must belong to the current trace")
        self.graph.return_(*(value.value for value in outputs))
        self.output_specs = tuple((value.shape, value.dtype) for value in outputs)
        self.output_is_tuple = is_tuple
        self._finished = True
        return self.module.verify()
