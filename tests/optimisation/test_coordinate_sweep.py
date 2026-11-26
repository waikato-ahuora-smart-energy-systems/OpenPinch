"""A coordinate sweep finds a better basin a local search cannot reach.

A local search stops in the basin it starts in. Where the objective has a
small rise between that basin and a deeper one along one coordinate (a
heat pump level between two kinks of a composite curve, say), a sweep of
that coordinate across its bounds steps over the rise.
"""

from __future__ import annotations

import pytest

from OpenPinch.optimisation.candidates import sweep_coordinates


def two_basins(point) -> float:
    """A shallow basin at x = 0.2 and a deeper one at x = 0.7."""
    x, y = point
    return min((x - 0.2) ** 2 + 0.05, (x - 0.7) ** 2) + (y - 0.5) ** 2


def test_a_sweep_carries_a_point_over_a_barrier_along_one_coordinate():
    start = (0.2, 0.5)

    point, value = sweep_coordinates(two_basins, start, ((0.0, 1.0), (0.0, 1.0)))

    assert point == pytest.approx((0.7, 0.5), abs=1e-9)
    assert value == pytest.approx(0.0, abs=1e-12)


def kinked(point) -> float:
    """A kink at x = 0.2 and a lower one at x = 0.7037, off the sweep's grid."""
    (x,) = point
    return min(abs(x - 0.2) + 0.05, abs(x - 0.7037))


def test_a_sweep_refines_between_the_grid_values_beside_its_best():
    # The optimum of a heat pump level sits at a kink of a composite curve,
    # which a grid of a hundred values misses by up to half a step.
    point, _value = sweep_coordinates(kinked, (0.2,), ((0.0, 1.0),))

    assert point[0] == pytest.approx(0.7037, abs=1.5e-4)
