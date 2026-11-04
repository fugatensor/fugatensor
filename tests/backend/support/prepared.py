"""Mock the native port boundary for Python package and dispatch tests."""
from unittest.mock import Mock

import pytest

from fugatensor._ask.executable import _Prepared
from fugatensor._ask.runtime import DeviceTensor


@pytest.fixture
def mock_cuda_ports(monkeypatch):
    def prepare_ports(prepared):
        prepared._ports = Mock(spec=("run", "submit"))
        prepared._tensor_type = DeviceTensor

    monkeypatch.setattr(_Prepared, "_prepare_ports", prepare_ports)
