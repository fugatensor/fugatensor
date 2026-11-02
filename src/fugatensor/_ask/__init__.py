"""Private host-side lowering and compiler orchestration for FugaTensor.

ASK itself exposes C++ libraries. This module is an implementation detail,
not a second public tensor or compiler API.
"""

from .. import _native

from ._version import __version__
from .compiler import Compiler, CompilerError, PreparedModule
from .ir import Graph, Implementation, Module, Semantics, Value
from .types import ScalarType, TensorType
from .runtime import ExecutionSession, DeviceTensor, Event
from .adapters import CompilerEnvironment
from .plan import CUDAProgram
from .gpu import CUDAKernel

__all__ = [
    "__version__",
    "Compiler", "CompilerError", "PreparedModule", "Graph", "Implementation",
    "Module", "Semantics", "Value", "ScalarType", "TensorType",
    "ExecutionSession", "DeviceTensor", "Event",
    "CompilerEnvironment",
    "CUDAKernel", "CUDAProgram",
]
