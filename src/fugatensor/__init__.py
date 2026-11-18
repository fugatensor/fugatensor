"""Tensor API and device-independent Fuga IR, executed through ASK."""
from .dtypes import DType, bool, int32, int64, fp32, fp16, bf16
from .devices import device
from .engine import (Engine, configure, shutdown, tensor, from_buffer, evaluate,
                     set_default_device, get_default_device, synchronize)
from .tensor import Tensor
from .creation import empty, zeros, ones, full, full_like, iota, arange, linspace, eye
from . import ir, triton, weights
from .lowering import lower_to_ask
from .function import fun
from .autograd import grad, stop_gradient
from . import nn
from .ops import (add, multiply, subtract, negative, square, abs, floor, ceil, round,
                  cast, reshape, zeros_like, ones_like, divide, floordiv, mod, pow,
                  maximum, minimum, clamp, clip, compare, where,
                  exp, log, sqrt, sin, cos, tanh, erf,
                  logical_and, logical_or, logical_xor, logical_not,
                  bitwise_and, bitwise_or, bitwise_xor, bitwise_not, shift_left, shift_right,
                  broadcast_in_dim, broadcast_to, transpose, squeeze, unsqueeze, flatten,
                  slice, pad, concatenate, cat, stack, split, chunk,
                  gather, take, scatter_add, tril, triu, one_hot,
                  sum, amax, amin, prod, mean, any, all, argmax, argmin, matmul)

save_weights = weights.save
load_weights = weights.load

__all__ = ['configure', 'shutdown', 'Engine', 'Tensor', 'tensor', 'from_buffer', 'evaluate',
           'DType', 'bool', 'int32', 'int64', 'fp32', 'fp16', 'bf16',
           'device', 'reshape',
           'empty', 'zeros', 'ones', 'full', 'full_like', 'iota', 'arange', 'linspace', 'eye',
           'set_default_device', 'get_default_device', 'synchronize',
           'ir', 'lower_to_ask', 'triton', 'weights', 'fun', 'grad', 'stop_gradient', 'nn',
           'save_weights', 'load_weights',
           'add', 'multiply', 'subtract', 'negative', 'square', 'abs', 'floor', 'ceil', 'round',
           'cast', 'zeros_like', 'ones_like',
           'divide', 'floordiv', 'mod', 'pow', 'maximum', 'minimum', 'clamp', 'clip',
           'compare', 'where', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tanh', 'erf',
           'logical_and', 'logical_or', 'logical_xor', 'logical_not',
           'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not', 'shift_left', 'shift_right',
           'broadcast_in_dim', 'broadcast_to', 'transpose', 'squeeze', 'unsqueeze', 'flatten',
           'slice', 'pad', 'concatenate', 'cat', 'stack', 'split', 'chunk',
           'gather', 'take', 'scatter_add', 'tril', 'triu', 'one_hot',
           'sum', 'amax', 'amin', 'prod', 'mean', 'any', 'all', 'argmax', 'argmin', 'matmul']
