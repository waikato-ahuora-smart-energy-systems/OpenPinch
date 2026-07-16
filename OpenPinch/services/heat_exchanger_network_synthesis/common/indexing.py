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

    def build_nested_grid(axis_index: int, indices: tuple[int, ...]) -> Any:
        if axis_index == len(index_limits):
            return build_value(*indices)
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
