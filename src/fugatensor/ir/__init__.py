"""Independent Fuga tensor IR, before lowering to any execution compiler."""
from .types import DTYPES, ScalarType, TensorType, Type
from .core import Block, BlockBuilder, ExternalSpec, Function, Module, Operation, Region, Value
from .jvp import jvp

__all__ = ["DTYPES", "ScalarType", "TensorType", "Type", "Block", "BlockBuilder",
           "ExternalSpec", "Function", "Module", "Operation", "Region", "Value",
           "specialize", "vjp", "jvp"]


def __getattr__(name):
    if name == "specialize":
        from .transforms import specialize
        return specialize
    if name == "vjp":
        from .differentiate import vjp
        return vjp
    raise AttributeError(name)
