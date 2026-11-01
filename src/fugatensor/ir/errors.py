"""Shared transform errors, independent of differentiation mode."""


class DifferentiationError(ValueError):
    """The requested derivative has no implemented Fuga IR rule."""
