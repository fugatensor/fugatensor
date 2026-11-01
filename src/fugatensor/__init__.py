"""Tensor API and device-independent Fuga IR, executed through ASK."""
from .engine import Engine, configure, shutdown, tensor, evaluate
from .tensor import Tensor
from . import ir, triton, weights
from .lowering import lower_to_ask
from .function import fun
from .autograd import no_grad
from .ops import add, multiply, subtract, negative, square, cast, zeros_like, ones_like

save_weights = weights.save
load_weights = weights.load

__all__ = ['configure', 'shutdown', 'Engine', 'Tensor', 'tensor', 'evaluate',
           'ir', 'lower_to_ask', 'triton', 'weights', 'fun', 'no_grad',
           'save_weights', 'load_weights',
           'add', 'multiply', 'subtract', 'negative', 'square', 'cast', 'zeros_like', 'ones_like']
