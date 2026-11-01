"""Fuga value types and constant encoding, independent of a compiler backend."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

DTYPES = ("bool", "i32", "i64", "f32", "f16", "bf16")
_WIDTH = {"bool": 1, "i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2}
_FLOAT = {"f32": (8, 23), "f16": (5, 10), "bf16": (8, 7)}


def normalize_dtype(dtype):
    dtype = "f16" if dtype == "fp16" else dtype
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise ValueError(f"unsupported Fuga element type: {dtype!r}")
    return dtype


@dataclass(frozen=True)
class ScalarType:
    dtype: str

    def __post_init__(self):
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))

    def __str__(self):
        return self.dtype


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int | None, ...]
    dtype: str

    def __post_init__(self):
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise ValueError("tensor shape must be a sequence") from error
        if any(dim is not None and (type(dim) is not int or not 0 <= dim < 2**63) for dim in shape):
            raise ValueError("Fuga shape dimensions must be nonnegative integers or None")
        dtype = normalize_dtype(self.dtype)
        if None not in shape and math.prod(shape) > (2**63 - 1) // _WIDTH[dtype]:
            raise ValueError("Fuga tensor byte count overflows signed 64-bit storage")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    def __str__(self):
        dimensions = "".join(("?" if dim is None else str(dim)) + "x" for dim in self.shape)
        return f"tensor<{dimensions}{self.dtype}>"


Type = ScalarType | TensorType


def as_type(value):
    if isinstance(value, (ScalarType, TensorType)):
        return value
    if isinstance(value, str):
        return ScalarType(value)
    raise TypeError("expected a Fuga ScalarType or TensorType")


def cast_type(type_, dtype):
    return TensorType(type_.shape, dtype) if isinstance(type_, TensorType) else ScalarType(dtype)


def _round_shift(integer, shift):
    if shift <= 0:
        return integer << -shift
    retained, remainder = divmod(integer, 1 << shift)
    half = 1 << (shift - 1)
    return retained + (remainder > half or (remainder == half and retained & 1))


def constant_value(value, dtype):
    """Store the value at its declared precision, with one RNE rounding.

    Constants own their numeric meaning before ASK lowering. Integer inputs are
    rounded directly, so large integers do not lose low bits through binary64.
    """
    dtype = normalize_dtype(dtype)
    if dtype == "bool":
        if type(value) is not bool:
            raise ValueError("bool constants require Python bool values")
        return value
    if dtype in ("i32", "i64"):
        width = _WIDTH[dtype] * 8
        if type(value) is not int or not -(2**(width - 1)) <= value < 2**(width - 1):
            raise ValueError(f"constant must be a signed {dtype} integer")
        return value
    exponent_bits, fraction_bits = _FLOAT[dtype]
    if type(value) is int:
        sign, significand, power = int(value < 0), abs(value), 0
    elif type(value) is float:
        bits = int.from_bytes(struct.pack("<d", value), "little")
        sign, exponent = bits >> 63, (bits >> 52) & 0x7ff
        significand = bits & ((1 << 52) - 1)
        if exponent == 0x7ff:
            return math.copysign(float("nan") if significand else float("inf"), -1 if sign else 1)
        if exponent:
            significand |= 1 << 52
            power = exponent - 1023 - 52
        else:
            power = -1074
    else:
        raise ValueError(f"{dtype} constants require Python int or float values")
    multiplier = -1.0 if sign else 1.0
    if not significand:
        return math.copysign(0.0, multiplier)
    bias = (1 << (exponent_bits - 1)) - 1
    exponent = significand.bit_length() - 1 + power
    if exponent > bias:
        raise ValueError(f"finite constant is outside the {dtype} range")
    scale = max(exponent, 1 - bias) - fraction_bits
    rounded = _round_shift(significand, scale - power)
    if rounded >= 1 << (fraction_bits + 1):
        rounded >>= 1
        scale += 1
        exponent += 1
    if exponent > bias:
        raise ValueError(f"finite constant is outside the {dtype} range")
    return multiplier * math.ldexp(rounded, scale)
