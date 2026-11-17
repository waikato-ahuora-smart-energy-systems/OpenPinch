"""Tests for shared HEN synthesis indexing helpers."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.indexing import (
    build_index_grid,
    ordered_mapping_keys,
)


def test_build_index_grid_builds_expected_nested_shapes():
    recovery_grid = build_index_grid(
        lambda i, j, k: f"H{i}-C{j}-S{k}",
        (2, 3, 2),
    )
    assert recovery_grid == [
        [
            ["H0-C0-S0", "H0-C0-S1"],
            ["H0-C1-S0", "H0-C1-S1"],
            ["H0-C2-S0", "H0-C2-S1"],
        ],
        [
            ["H1-C0-S0", "H1-C0-S1"],
            ["H1-C1-S0", "H1-C1-S1"],
            ["H1-C2-S0", "H1-C2-S1"],
        ],
    ]

    period_grid = build_index_grid(
        lambda n, i, j, k: f"P{n}-H{i}-C{j}-S{k}",
        (2, 2, 3, 2),
    )
    assert period_grid[0][1][2][1] == "P0-H1-C2-S1"
    assert period_grid[1][1][2][1] == "P1-H1-C2-S1"
    assert len(period_grid) == 2

    assert build_index_grid(lambda i, k: f"H{i}-S{k}", (2, 2)) == [
        ["H0-S0", "H0-S1"],
        ["H1-S0", "H1-S1"],
    ]
    assert build_index_grid(lambda i: f"H{i}", (2,)) == ["H0", "H1"]


def test_build_index_grid_rejects_negative_limits():
    with pytest.raises(ValueError, match="non-negative"):
        build_index_grid(lambda bad: bad, (-1,))


def test_ordered_mapping_keys_sorts_keys_by_stored_index():
    assert ordered_mapping_keys({"period_2": 2, "period_0": 0, "period_1": 1}) == (
        "period_0",
        "period_1",
        "period_2",
    )


def test_build_index_grid_calls_only_the_cell_builder_per_cell():
    # A recursive helper call per cell (and a tuple copy at every level) was
    # 19 million extra calls per pulp mill design.
    import sys

    order = []

    def cell(i, j, k):
        order.append((i, j, k))
        return i + j + k

    other_calls = []

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is not cell.__code__:
            other_calls.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        grid = build_index_grid(cell, (3, 4, 5))
    finally:
        sys.setprofile(None)

    assert grid[2][3][4] == 9
    # Cells are built in row-major order: callers create Pyomo variables in it.
    assert order == [(i, j, k) for i in range(3) for j in range(4) for k in range(5)]
    assert len(other_calls) <= 3 * 4


def test_build_index_grid_builds_deeper_and_scalar_grids():
    deep = build_index_grid(lambda *index: index, (2, 1, 3, 1, 2))

    assert deep[1][0][2][0][1] == (1, 0, 2, 0, 1)
    assert len(deep[1][0]) == 3
    assert build_index_grid(lambda: "scalar", ()) == "scalar"
    assert build_index_grid(lambda i, j: (i, j), (2, 0)) == [[], []]
