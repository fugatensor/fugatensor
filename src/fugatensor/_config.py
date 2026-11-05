"""Load optional tool configuration from the installed environment, once per Engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def runtime_defaults(device):
    override = os.environ.get('FUGATENSOR_CONFIG')
    path = (Path(override).expanduser() if override else
            Path(sys.prefix) / 'share' / 'fugatensor' / 'engine.json')
    if not path.is_file():
        if override:
            raise ValueError(f'FUGATENSOR_CONFIG does not name a configuration file: {path}')
        return {}
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot read FugaTensor engine configuration: {path}') from error
    if not isinstance(config, dict) or type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('unsupported FugaTensor engine configuration schema')
    if device == 'cpu':
        return {}
    result = {}
    for name in ('llc', 'ptxas', 'llvm_link', 'libdevice'):
        value = config.get(name)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f'configured {name} must be a nonempty path')
            result[name] = value
    environments = config.get('compilers', {})
    if not isinstance(environments, dict):
        raise ValueError('configured compilers must be an object keyed by compiler ID')
    from fugatensor._ask import CompilerEnvironment
    compilers = {}
    for name, environment in environments.items():
        if not isinstance(environment, dict) or set(environment) - {'python', 'worker'}:
            raise ValueError(f'invalid configured environment for compiler {name!r}')
        env = dict(environment)
        worker_path = Path(env.get('worker', ''))
        if not worker_path.is_file():
            fallback = Path(__file__).resolve().parent / '_triton_worker.py'
            if fallback.is_file():
                env['worker'] = str(fallback)
        compilers[name] = CompilerEnvironment(**env)
    result['compilers'] = compilers
    return result
