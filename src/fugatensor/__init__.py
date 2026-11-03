"""Tensor API and device-independent Fuga IR, executed through ASK."""
from .dtypes import DType, bool, int32, int64, fp32, fp16, bf16
from .devices import device
from .engine import (Engine, configure, shutdown, tensor, from_buffer, evaluate,
                     set_default_device, get_default_device, synchronize)
from .tensor import Tensor
from .creation import empty, zeros, ones, full
from . import ir, triton, weights
from .lowering import lower_to_ask
from .function import fun
from .autograd import grad, stop_gradient
from . import nn
from .ops import add, multiply, subtract, negative, square, cast, reshape, zeros_like, ones_like

save_weights = weights.save
load_weights = weights.load

__all__ = ['configure', 'shutdown', 'Engine', 'Tensor', 'tensor', 'from_buffer', 'evaluate',
           'DType', 'bool', 'int32', 'int64', 'fp32', 'fp16', 'bf16',
           'device', 'reshape',
           'empty', 'zeros', 'ones', 'full',
           'set_default_device', 'get_default_device', 'synchronize',
           'ir', 'lower_to_ask', 'triton', 'weights', 'fun', 'grad', 'stop_gradient', 'nn',
           'save_weights', 'load_weights',
           'add', 'multiply', 'subtract', 'negative', 'square', 'cast', 'zeros_like', 'ones_like']
