"""Pure-Python numerical reference helpers for Stage 2 tensor operations."""
from __future__ import annotations

import math


def flatten_nested(value):
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(flatten_nested(item))
        return out
    return [value]


def assert_allclose(actual, expected, *, atol=1e-5, rtol=1e-5):
    actual_flat = flatten_nested(actual)
    expected_flat = flatten_nested(expected)
    assert len(actual_flat) == len(expected_flat), f"length mismatch: {len(actual_flat)} != {len(expected_flat)}"
    for index, (a, e) in enumerate(zip(actual_flat, expected_flat)):
        if isinstance(e, bool):
            assert bool(a) is e, f"index {index}: {a!r} != {e!r}"
        elif isinstance(e, int):
            assert a == e, f"index {index}: {a!r} != {e!r}"
        elif math.isnan(e):
            assert math.isnan(a), f"index {index}: expected NaN, got {a!r}"
        elif math.isinf(e):
            assert math.isinf(a) and (a > 0) == (e > 0), f"index {index}: {a!r} != {e!r}"
        else:
            tol = atol + rtol * abs(e)
            assert abs(a - e) <= tol, f"index {index}: |{a} - {e}| = {abs(a - e)} > {tol}"


def rint_even(x: float) -> float:
    return float(round(x))


def py_floordiv_int(a: int, b: int, bits: int = 32) -> int:
    if b == 0:
        return 0
    min_val = -(1 << (bits - 1))
    if a == min_val and b == -1:
        return min_val
    return a // b


def py_mod_int(a: int, b: int, bits: int = 32) -> int:
    if b == 0:
        return 0
    min_val = -(1 << (bits - 1))
    if a == min_val and b == -1:
        return 0
    return a % b


def py_shl_int(a: int, b: int, bits: int = 32) -> int:
    if not (0 <= b < bits):
        return 0
    shifted = (a << b) & ((1 << bits) - 1)
    if shifted >= (1 << (bits - 1)):
        shifted -= 1 << bits
    return shifted


def py_shr_int(a: int, b: int, bits: int = 32) -> int:
    if b < 0:
        return 0
    if b >= bits:
        return -1 if a < 0 else 0
    return a >> b


def ref_softmax(row: list[float]) -> list[float]:
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex]


def ref_log_softmax(row: list[float]) -> list[float]:
    m = max(row)
    shifted = [v - m for v in row]
    lse = math.log(sum(math.exp(v) for v in shifted))
    return [v - lse for v in shifted]


def ref_gelu(x: float, approximate: str = 'none') -> float:
    if approximate == 'none':
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def ref_layer_norm(row: list[float], eps: float = 1e-5,
                   weight: list[float] | None = None, bias: list[float] | None = None) -> list[float]:
    mu = sum(row) / len(row)
    var = sum((v - mu) ** 2 for v in row) / len(row)
    inv = 1.0 / math.sqrt(var + eps)
    out = [(v - mu) * inv for v in row]
    if weight is not None:
        out = [o * w for o, w in zip(out, weight)]
    if bias is not None:
        out = [o + b for o, b in zip(out, bias)]
    return out


def ref_rms_norm(row: list[float], eps: float = 1e-6, weight: list[float] | None = None) -> list[float]:
    ms = sum(v * v for v in row) / len(row)
    inv = 1.0 / math.sqrt(ms + eps)
    out = [v * inv for v in row]
    if weight is not None:
        out = [o * w for o, w in zip(out, weight)]
    return out
