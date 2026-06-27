"""Tests for shared HEN synthesis indexing helpers."""

from __future__ import annotations

import pytest

from OpenPinch.services.heat_exchanger_network_synthesis.common.indexing import (
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
