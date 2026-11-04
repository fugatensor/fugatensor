"""Shared locations, markers and explicit ASK compiler construction."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fugatensor._ask import Compiler

ROOT = Path(__file__).resolve().parents[1]
TRITON_KERNELS = ROOT / 'tests/fixtures/triton_kernels.py'
requires_cuda = pytest.mark.cuda
requires_triton = pytest.mark.triton


def make_ask_compiler() -> Compiler:
    return Compiler(executable=os.environ.get('ASK_OPT'))
