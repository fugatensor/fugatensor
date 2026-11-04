"""Install a built wheel with each extra in a fresh, isolated environment."""
from __future__ import annotations

import pytest
import re

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


PROBE = '''
import importlib.metadata as metadata
import json
import sys
import fugatensor as ft

try:
    triton_version = metadata.version("triton")
except metadata.PackageNotFoundError:
    triton_version = None
x = ft.tensor([1, 2, 3])
y = x * x + 1
gradient = ft.grad(y, x, seed=ft.ones_like(y))
result = {"values": y.tolist(), "gradients": gradient.tolist(),
          "triton_version": triton_version, "gpu_values": None}
with ft.Engine("cuda") as engine:
    worker = engine.compilers.get("triton")
    result["worker_python"] = worker.python if worker else None
    if sys.argv[1] == "all" and sys.argv[2] == "1":
        operation = ft.triton.from_source(
            "import triton\\nimport triton.language as tl\\n"
            "@triton.jit\\ndef twice(X, Y, N: tl.constexpr):\\n"
            "    i = tl.program_id(0) * 32 + tl.arange(0, 32)\\n"
            "    tl.store(Y + i, tl.load(X + i, i < N, 0) * 2, i < N)\\n",
            entry="twice", input="X", output="Y", grid=1,
            constants=lambda spec: {"N": spec.numel})
        result["gpu_values"] = operation(ft.tensor([1, 2, 3], device="cuda")).tolist()
result["host_imported_triton"] = any(name == "triton" or name.startswith("triton.") for name in sys.modules)
print(json.dumps(result))
'''


class TestWheelInstallation:
    @pytest.mark.parametrize('extra', ('pure', 'all'))
    def test_installation_matrix(self, request, extra):
        selected = request.config.getoption('--wheel')
        if not selected:
            pytest.fail('wheel installation tests require --wheel=/absolute/path.whl')
        wheel = Path(selected).resolve(strict=True)
        gpu = '1' if request.config.getoption('--triton') else '0'
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith('FUGATENSOR_')
                       and name not in ('PYTHONPATH', 'PYTHONHOME', 'ASK_OPT')}
        with tempfile.TemporaryDirectory(prefix='fuga-install-') as directory:
            root = Path(directory) / extra
            venv.EnvBuilder(with_pip=False).create(root)
            python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            pip = [sys.executable, '-m', 'pip', '--python', str(python)]
            for command in ([*pip, 'install', f'{wheel}[{extra}]'], [*pip, 'check']):
                installed = subprocess.run(command, env=environment, text=True, capture_output=True)
                assert (installed.returncode) == (0), installed.stdout + installed.stderr
            observed = subprocess.run([str(python), '-I', '-c', PROBE, extra, gpu],
                                      cwd=directory, env=environment, text=True, capture_output=True)
            assert (observed.returncode) == (0), observed.stdout + observed.stderr
            result = json.loads(observed.stdout)
            assert (result['values']) == ([2.0, 5.0, 10.0])
            assert (result['gradients']) == ([2.0, 4.0, 6.0])
            assert not (result['host_imported_triton'])
            if extra == 'pure':
                assert (result['triton_version']) is None
                assert (result['worker_python']) is None
            else:
                assert re.search('^3\\.9\\.\\d+$', result['triton_version'])
                assert (result['worker_python']) == (str(python))
                if gpu == '1':
                    assert (result['gpu_values']) == ([2.0, 4.0, 6.0])
