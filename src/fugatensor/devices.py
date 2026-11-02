"""Device descriptions and the runtime's canonical device names."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True, init=False, repr=False)
class device:
    """Describe a CPU or CUDA device without initializing its runtime.

    A missing CUDA index is unspecified in the description. FugaTensor resolves
    it to device 0 when selecting an engine; allocated tensors expose that index.
    """

    type: str
    index: int | None

    def __init__(self, type, index=None):
        if isinstance(type, device):
            if index is not None:
                raise TypeError("a device object cannot be combined with an index")
            kind, ordinal = type.type, type.index
        elif isinstance(type, str):
            match = re.fullmatch(r"(cpu|cuda)(?::([0-9]+))?", type)
            if match is None:
                raise ValueError("device must name 'cpu', 'cuda', or a nonnegative device index")
            kind, text_index = match.groups()
            if text_index is not None and index is not None:
                raise ValueError("device index is specified twice")
            if index is not None and index.__class__ is not int:
                raise TypeError("device index must be an integer")
            ordinal = int(text_index) if text_index is not None else index
            if ordinal is not None and ordinal < 0:
                raise ValueError("device index must be nonnegative")
        else:
            raise TypeError("device must be a string or a device object")
        if kind == "cpu" and ordinal not in (None, 0):
            raise ValueError("CPU device index must be 0 or omitted")
        object.__setattr__(self, "type", kind)
        object.__setattr__(self, "index", ordinal)

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self):
        index = "" if self.index is None else f", index={self.index}"
        return f"device(type={self.type!r}{index})"


def resolve_device(value):
    """Return an allocated-device description without probing hardware."""
    selected = device(value)
    return device("cpu") if selected.type == "cpu" else device("cuda", selected.index or 0)


def normalize_device(value):
    """Keep internal Engine and backend keys compatible with existing caches."""
    selected = resolve_device(value)
    return selected.type if selected.index in (None, 0) else str(selected)


__all__ = ["device"]
