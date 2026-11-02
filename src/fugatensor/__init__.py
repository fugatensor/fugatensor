"""Tensor API and device-independent Fuga IR, executed through ASK."""
from .dtypes import DType, bool, int32, int64, fp32, fp16, bf16
from .engine import (Engine, configure, shutdown, tensor, from_buffer, evaluate,
                     set_default_device, get_default_device, synchronize)
from .tensor import Tensor
from . import ir, triton, weights
from .lowering import lower_to_ask
from .function import fun
from .autograd import no_grad
from .ops import add, multiply, subtract, negative, square, cast, zeros_like, ones_like

save_weights = weights.save
load_weights = weights.load

__all__ = ['configure', 'shutdown', 'Engine', 'Tensor', 'tensor', 'from_buffer', 'evaluate',
           'DType', 'bool', 'int32', 'int64', 'fp32', 'fp16', 'bf16',
           'set_default_device', 'get_default_device', 'synchronize',
           'ir', 'lower_to_ask', 'triton', 'weights', 'fun', 'no_grad',
           'save_weights', 'load_weights',
           'add', 'multiply', 'subtract', 'negative', 'square', 'cast', 'zeros_like', 'ones_like']
