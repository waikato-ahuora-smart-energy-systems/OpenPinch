"""Shared index-grid construction helpers for HEN synthesis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from operator import itemgetter
from typing import Any


def build_index_grid(
    build_value: Callable[..., Any],
    lims: Sequence[int],
) -> list:
    """Build a nested grid from ordered index limits."""

    index_limits = tuple(lims)
    if any(limit < 0 for limit in index_limits):
        raise ValueError("index grid limits must be non-negative")

    # The usual depths are written out as comprehensions, so each cell costs
    # one call to ``build_value``; a recursive helper per cell was millions of
    # extra calls per large design. Cells are built in row-major order.
    f = build_value
    match index_limits:
        case ():
            return f()
        case (a,):
            return [f(i) for i in range(a)]
        case (a, b):
            return [[f(i, j) for j in range(b)] for i in range(a)]
        case (a, b, c):
            return [
                [[f(i, j, k) for k in range(c)] for j in range(b)] for i in range(a)
            ]
        case (a, b, c, d):
            return [
                [
                    [[f(n, i, j, k) for k in range(d)] for j in range(c)]
                    for i in range(b)
                ]
                for n in range(a)
            ]

    innermost = index_limits[-1]

    def build_nested_grid(axis_index: int, indices: tuple[int, ...]) -> Any:
        if axis_index == len(index_limits) - 1:
            return [f(*indices, index) for index in range(innermost)]
        return [
            build_nested_grid(axis_index + 1, (*indices, index))
            for index in range(index_limits[axis_index])
        ]

    return build_nested_grid(0, ())


def ordered_mapping_keys(index_by_key: Mapping[str, int]) -> tuple[str, ...]:
    """Return mapping keys ordered by their stored index value."""

    return tuple(
        key
        for key, _index in sorted(
            index_by_key.items(),
            key=itemgetter(1),
        )
    )
