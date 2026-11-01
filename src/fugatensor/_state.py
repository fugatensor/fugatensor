"""Per-context tracing state used to reject accidental reads of real tensors."""
from contextvars import ContextVar

active_trace = ContextVar('fugatensor_active_trace', default=None)


def require_not_tracing():
    if active_trace.get() is not None:
        raise TypeError('real tensor data and operations are unavailable while tracing; '
                        'pass tensors as explicit operation arguments')
