"""Trace tensor expressions into device-independent Fuga SSA."""
from __future__ import annotations
import weakref

from .ir import Module, TensorType, Value

from .tensor import normalize_dtype, normalize_shape
from ._state import active_trace


class TraceTensor:
    """A Fuga SSA value used only while a JIT specialization is being traced."""
    __slots__ = ("context", "value", "__weakref__")

    def __init__(self, context: TraceContext, value: Value):
        if not isinstance(value, Value) or value.owner is not context.graph.block:
            raise ValueError("traced value must belong to its TraceContext graph")
        if not isinstance(value.type, TensorType):
            raise TypeError("TraceTensor requires a Fuga tensor value")
        self.context = context
        self.value = value
        context._handles[id(self)] = self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.type.shape

    @property
    def dtype(self) -> str:
        return self.value.type.dtype

    @property
    def device(self) -> str:
        return self.context.engine.device if self.context.engine is not None else "cpu"

    def astype(self, dtype: str) -> TraceTensor:
        value = self.context.graph.cast(self.value, dtype=normalize_dtype(dtype))
        return self.context.wrap(value)

    def _operand(self, other):
        if isinstance(other, TraceTensor):
            if other.context is not self.context:
                raise ValueError("traced operands must belong to the same graph")
            if other.shape != self.shape:
                raise ValueError("tensor operands require equal shapes; broadcasting is not supported")
            if other.dtype != self.dtype:
                raise TypeError("tensor operands require equal dtypes; use astype explicitly")
            return other.value
        if type(other) not in (bool, int, float):
            raise TypeError("traced expressions accept traced tensors or Python scalars; pass tensor captures as JIT arguments")
        constant = self.context.graph.constant(other, dtype=self.dtype)
        return self.context.graph.splat(constant, self.shape)

    def _binary(self, operation, other, *, reverse=False):
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
        if self.context._runtime is None:
            raise TypeError('data-dependent control flow inside an isolated fun or backward trace is unsupported')
        return self.context._runtime.break_graph(self)

    def tolist(self):
        raise TypeError("tensor values cannot be read while tracing a jit function")

    def __eq__(self, other):
        raise TypeError("tensor comparisons and data-dependent control flow are unsupported inside jit")

    def __ne__(self, other):
        return self.__eq__(other)

    def __repr__(self):
        return f"TraceTensor(shape={self.shape!r}, dtype={self.dtype!r}, value={self.value.name!r})"


class TraceContext:
    """Own one Fuga graph and infer its result types from the Python return."""
    def __init__(self, input_specs, *, engine=None, name="fuga_main"):
        self.engine = engine
        self._handles = weakref.WeakValueDictionary()
        self._runtime = None
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
        context._handles = weakref.WeakValueDictionary()
        context._runtime = None
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
            raise TypeError("a jit function must return a traced Tensor or a nonempty tuple of traced Tensors")
        if any(value.context is not self for value in outputs):
            raise ValueError("jit outputs must belong to the current trace")
        self.graph.return_(*(value.value for value in outputs))
        self.output_specs = tuple((value.shape, value.dtype) for value in outputs)
        self.output_is_tuple = is_tuple
        self._finished = True
        return self.module.verify()
