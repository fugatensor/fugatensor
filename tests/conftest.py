"""Pytest resource fixtures and explicit hardware/integration selection."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import pytest
import fugatensor as ft


def pytest_addoption(parser):
    parser.addoption('--cuda', action='store_true', help='Run physical CUDA tests; missing tools or devices fail.')
    parser.addoption('--triton', action='store_true', help='Run CUDA and Triton worker tests; missing dependencies fail.')
    parser.addoption('--wheel', help='Wheel path for explicitly selected installation tests.')
    parser.addoption('--exhaustive', action='store_true', help='Expand the generated matrices to their full combinations.')


def pytest_ignore_collect(collection_path, config):
    if collection_path.name not in ('backend', 'packaging'):
        return False
    selected = [Path(str(arg).split('::', 1)[0]).resolve() for arg in config.args]
    return not any(path == collection_path or collection_path in path.parents for path in selected)


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        triton = item.get_closest_marker('triton') is not None
        cuda = item.get_closest_marker('cuda') is not None
        enabled = (not triton or config.getoption('--triton')) and (
            not cuda or config.getoption('--cuda') or config.getoption('--triton'))
        (selected if enabled else deselected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def _open_engine(request, cache_dir):
    cls = request.cls
    device = getattr(cls, 'device', 'cpu')
    triton = getattr(cls, 'enable_triton', False)
    kwargs = {'cache_dir': cache_dir}
    if device != 'cpu' and not triton:
        kwargs['compilers'] = {}
    engine = ft.Engine(device=device, **kwargs)
    try:
        if device == 'cpu':
            if not engine.compiler.capabilities().get('host_codegen'):
                pytest.fail('ASK host code generation is required')
        else:
            if not engine.llc or not engine.ptxas:
                pytest.fail('CUDA tests require configured llc and ptxas')
            if triton and 'triton' not in engine.compilers:
                pytest.fail('Triton tests require a configured worker')
            engine.session.capability
    except BaseException:
        engine.close()
        raise
    return engine


@pytest.fixture(scope='class')
def shared_engine(request, tmp_path_factory):
    with _open_engine(request, tmp_path_factory.mktemp(request.cls.__name__)) as engine:
        yield engine


@pytest.fixture
def engine_context(request, tmp_path):
    # Numerical matrices share prepared templates; state tests start with a fresh Engine.
    with ExitStack() as stack:
        if request.node.get_closest_marker('shared_engine'):
            engine = request.getfixturevalue('shared_engine')
        else:
            engine = stack.enter_context(_open_engine(request, tmp_path / 'cache'))
        instance = request.instance
        instance.engine = engine
        instance.device = getattr(request.cls, "device", "cpu")
        instance.cache_dir = engine.cache_dir
        instance.supports_bf16 = engine.device != 'cpu' and engine.session.capability[0] >= 8
        token = ft.engine._current.set(engine)
        try:
            yield engine
        finally:
            ft.engine._current.reset(token)


@pytest.fixture
def cleanup():
    with ExitStack() as resources:
        yield resources
