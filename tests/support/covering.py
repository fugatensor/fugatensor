"""Deterministic two-factor coverage, with exhaustive expansion on request."""
from itertools import combinations, product


def pairwise(axes, *, exhaustive=False):
    rows = tuple(product(*axes))
    if exhaustive or not rows:
        return rows
    pairs = tuple(combinations(range(len(axes)), 2))
    features = tuple(frozenset((i, j, row[i], row[j]) for i, j in pairs) for row in rows)
    uncovered = set().union(*features)
    selected = []
    while uncovered:
        index = max(range(len(rows)), key=lambda i: len(features[i] & uncovered))
        selected.append(rows[index])
        uncovered.difference_update(features[index])
    return tuple(selected)
