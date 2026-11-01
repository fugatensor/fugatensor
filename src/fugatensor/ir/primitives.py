"""Executable typing and differentiation rules for the Fuga primitive basis.

Builders and verifiers use abstract evaluation. Reverse/forward transforms call
registered VJP/JVP callbacks that emit this same basis, so another transform can
consume the result without evaluating tensor values in Python.
"""
from dataclasses import dataclass
from types import MappingProxyType

from .errors import DifferentiationError
from .types import FLOAT_DTYPES, ScalarType, TensorType, as_type, cast_type, constant_value


class HigherOrderError(DifferentiationError):
    """A program has no executable composable rule for an operation."""


@dataclass(frozen=True)
class PrimitiveRule:
    """Typing and graph transforms for one primitive.

    abstract_eval(input_types, semantic_attrs) -> tuple[result types]. Semantic
    attrs include the declared dtype for constant/cast, whose textual IR stores
    that dtype in its result type.
    vjp(builder, op, primals, cotangent, active) -> tuple[operand cotangent|None].
    jvp(builder, op, primals, tangents) -> tuple[result tangent|None].
    None is a structural zero, not an unknown derivative.
    """
    opcode: str
    abstract_eval: object
    vjp: object
    jvp: object
    description: str
    floating_values: bool = True


def _constant_type(inputs, attrs):
    if inputs or set(attrs) != {"dtype", "value"}:
        raise TypeError("constant requires a dtype, scalar value, and no operands")
    result = ScalarType(attrs["dtype"])
    constant_value(attrs["value"], result.dtype)
    return (result,)


def _splat_type(inputs, attrs):
    if len(inputs) != 1 or not isinstance(inputs[0], ScalarType) or set(attrs) != {"shape"}:
        raise TypeError("splat requires one scalar and an explicit shape")
    return (TensorType(attrs["shape"], inputs[0].dtype),)


def _cast_type(inputs, attrs):
    if len(inputs) != 1 or set(attrs) != {"dtype"}:
        raise TypeError("cast requires one operand and a result dtype")
    return (cast_type(inputs[0], attrs["dtype"]),)


def _arithmetic_type(inputs, attrs):
    if len(inputs) != 2 or inputs[0] != inputs[1]:
        raise TypeError("arithmetic requires exactly equal operand and result types")
    result = inputs[0]
    if result.dtype == "bool":
        raise TypeError("bool arithmetic is undefined")
    expected = {"overflow": "wrap"} if result.dtype in ("i32", "i64") else {}
    if dict(attrs) != expected:
        raise ValueError("arithmetic numerical attributes disagree with the declared dtype")
    return (result,)


def _constant_vjp(builder, operation, primals, cotangent, active):
    return ()


def _splat_vjp(builder, operation, primals, cotangent, active):
    if active[0]:
        raise DifferentiationError("scalar-input splat VJP requires a sum reduction, which is not implemented")
    return (None,)


def _add_vjp(builder, operation, primals, cotangent, active):
    return tuple(cotangent if selected else None for selected in active)


def _mul_vjp(builder, operation, primals, cotangent, active):
    return (builder.mul(cotangent, primals[1]) if active[0] else None,
            builder.mul(cotangent, primals[0]) if active[1] else None)


def _cast_vjp(builder, operation, primals, cotangent, active):
    if not active[0]:
        return (None,)
    if operation.operands[0].type.dtype not in FLOAT_DTYPES or operation.results[0].type.dtype not in FLOAT_DTYPES:
        raise DifferentiationError("integer/bool casts on a derivative path have no floating VJP rule")
    return (builder.cast(cotangent, dtype=operation.operands[0].type.dtype),)


def _jvp_constant(*args):
    from .jvp import _constant_jvp
    return _constant_jvp(*args)


def _jvp_splat(*args):
    from .jvp import _splat_jvp
    return _splat_jvp(*args)


def _jvp_add(*args):
    from .jvp import _add_jvp
    return _add_jvp(*args)


def _jvp_mul(*args):
    from .jvp import _mul_jvp
    return _mul_jvp(*args)


def _jvp_cast(*args):
    from .jvp import _cast_jvp
    return _cast_jvp(*args)


PRIMITIVE_RULES = MappingProxyType({
    "constant": PrimitiveRule("constant", _constant_type, _constant_vjp, _jvp_constant,
                              "A literal has zero derivative.", False),
    "splat": PrimitiveRule("splat", _splat_type, _splat_vjp, _jvp_splat,
                           "Broadcast a scalar; transposing an active scalar input needs a sum reduction."),
    "add": PrimitiveRule("add", _arithmetic_type, _add_vjp, _jvp_add,
                         "Sum tangents; send a cotangent to each active operand."),
    "mul": PrimitiveRule("mul", _arithmetic_type, _mul_vjp, _jvp_mul,
                         "Multiply each derivative by the opposite primal operand."),
    "cast": PrimitiveRule("cast", _cast_type, _cast_vjp, _jvp_cast,
                          "Floating casts cast the derivative too; rounding is ignored."),
})


def infer_result_types(opcode, input_types=(), attrs=None, *, dtype=None):
    """Abstract evaluation needs types and static parameters, never values."""
    rule = PRIMITIVE_RULES.get(opcode)
    if rule is None:
        raise TypeError(f"no primitive abstract evaluation for fuga.{opcode}")
    parameters = dict(attrs or {})
    if dtype is not None:
        parameters["dtype"] = dtype
    return tuple(rule.abstract_eval(tuple(as_type(value) for value in input_types), parameters))


def validate_primitive(operation):
    rule = PRIMITIVE_RULES.get(operation.opcode)
    if rule is None or not callable(rule.abstract_eval):
        raise DifferentiationError(f"no primitive typing rule for fuga.{operation.opcode}")
    if operation.opcode == "cast" and operation.attrs:
        raise TypeError("cast stores its dtype in the result type and cannot have attributes")
    declared = tuple(value.type for value in operation.results)
    dtype = declared[0].dtype if len(declared) == 1 and operation.opcode in ("constant", "cast") else None
    inferred = infer_result_types(operation.opcode, (value.type for value in operation.operands),
                                  operation.attrs, dtype=dtype)
    if declared != inferred:
        raise TypeError(f"fuga.{operation.opcode} results must preserve the shape and dtype inferred by primitive abstract evaluation")
    return rule


def primitive_vjp(builder, operation, primals, cotangent, active):
    """Dispatch the executable rule and check its returned cotangent types."""
    rule = validate_primitive(operation)
    if not callable(rule.vjp):
        raise DifferentiationError(f"no reverse rule for fuga.{operation.opcode}")
    if len(operation.results) != 1 or cotangent.type != operation.results[0].type:
        raise TypeError(f"fuga.{operation.opcode} cotangent must match its single result type")
    if len(primals) != len(operation.operands) or len(active) != len(operation.operands):
        raise ValueError("primitive VJP operand binding count mismatch")
    if any(type(value) is not bool for value in active):
        raise TypeError("primitive VJP activity must contain bool values")
    if any(primal.type != operand.type for primal, operand in zip(primals, operation.operands)):
        raise TypeError("primitive VJP primal types must match the operation operands")
    gradients = tuple(rule.vjp(builder, operation, tuple(primals), cotangent, tuple(active)))
    if len(gradients) != len(operation.operands):
        raise DifferentiationError(f"fuga.{operation.opcode} VJP returned an invalid gradient count")
    for operand, gradient in zip(operation.operands, gradients):
        if gradient is not None and gradient.type != operand.type:
            raise TypeError(f"fuga.{operation.opcode} VJP must preserve operand shape and dtype")
    return gradients


def validate_composable(module, *, phase="forward"):
    """Check executable rules, including nested custom forward computations."""
    from .transforms import specialize
    flat = specialize(module, expand_calls=False)
    for function in flat.functions:
        for operation in function.operations:
            if operation.opcode == "return":
                continue
            if operation.opcode == "call":
                spec = operation.attrs["spec"]
                validate_composable(spec.forward, phase=f"{phase} in {spec.name}")
                continue
            if operation.opcode == "external":
                compiler = operation.attrs["spec"].compiler
                raise HigherOrderError(f"higher-order differentiation has no composable primitive rule for "
                                       f"external compiler {compiler!r} ({phase}, {operation.id})")
            rule = PRIMITIVE_RULES.get(operation.opcode)
            if rule is None or not callable(rule.abstract_eval) or not callable(rule.vjp):
                raise HigherOrderError(f"higher-order differentiation has no registered primitive rule for "
                                       f"fuga.{operation.opcode} ({phase}, {operation.id})")
            validate_primitive(operation)
            if rule.floating_values and any(value.type.dtype not in FLOAT_DTYPES
                                            for value in (*operation.operands, *operation.results)):
                raise HigherOrderError(f"higher-order differentiation requires floating operands/results for "
                                       f"fuga.{operation.opcode} ({phase}, {operation.id})")
    return module
