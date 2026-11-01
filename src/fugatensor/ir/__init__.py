"""Independent Fuga tensor IR, before lowering to any execution compiler."""
from .types import DTYPES, FLOAT_DTYPES, ScalarType, TensorType, Type
from .core import Block, BlockBuilder, ExternalSpec, Function, Module, Operation, Region, Value
from .jvp import jvp

__all__ = ["DTYPES", "FLOAT_DTYPES", "ScalarType", "TensorType", "Type", "Block", "BlockBuilder",
           "ExternalSpec", "Function", "Module", "Operation", "Region", "Value",
           "analyze_function", "project_outputs", "prune", "specialize", "vjp", "jvp"]


def __getattr__(name):
    if name in ("analyze_function", "project_outputs", "prune", "specialize"):
        from . import transforms
        return getattr(transforms, name)
    if name == "vjp":
        from .differentiate import vjp
        return vjp
    raise AttributeError(name)
