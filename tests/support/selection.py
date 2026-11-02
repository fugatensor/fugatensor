"""Strict comma-separated selectors shared by parameterized test suites."""
from __future__ import annotations

import os


def select(name, choices, *, default=None):
    choices = tuple(choices)
    raw = os.environ.get(name)
    result = tuple(choices if default is None else default) if raw is None else tuple(raw.split(','))
    if not result or any(not value or value not in choices for value in result):
        raise ValueError(f'{name} must select comma-separated values from {", ".join(choices)}')
    return tuple(dict.fromkeys(result))
