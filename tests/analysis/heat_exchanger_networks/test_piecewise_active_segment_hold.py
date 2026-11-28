"""A relaxed mapping that cycles between two segments is held at one.

After each solve without integers a mapping moves to the segment its
solved heat lies in. With the optimum at a kink, each segment's line,
extrapolated, puts the heat in the other segment, and the mapping would
go back and forth until the solve ran out of attempts. A mapping whose
heat returns to a segment it has left is held instead: its heat
coordinate is bounded to one segment, where the optimum is the kink.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis.heat_exchanger_networks.models._base import piecewise
from OpenPinch.analysis.heat_exchanger_networks.solver.piecewise import (
    PiecewiseThermalProfile,
)


class _Variable:
    """A model variable or, fixed, a parameter."""

    def __init__(self, value: float, lb=None, ub=None, fixed: bool = False):
        self.value = value
        self.lb = lb
        self.ub = ub
        self.fixed = fixed

    def is_variable_type(self) -> bool:
        return True

    def set_value(self, value: float, skip_validation: bool = False) -> None:
        self.value = value

    def setlb(self, value: float) -> None:
        self.lb = value

    def setub(self, value: float) -> None:
        self.ub = value


def _profile(contributions) -> PiecewiseThermalProfile:
    """A hot stream 380 -> 230 -> 100 degC: 2000 kW, then 800 kW."""

    return PiecewiseThermalProfile(
        identities=("s0", "s1"),
        temperatures_in=np.array([380.0, 230.0]),
        temperatures_out=np.array([230.0, 100.0]),
        duties=np.array([2000.0, 800.0]),
        heat_capacity_flowrates=np.array([2000.0 / 150.0, 800.0 / 130.0]),
        heat_transfer_coefficients=np.ones(2),
        temperature_contributions=np.asarray(contributions, dtype=float),
    )


def _mapping(coordinate, profile) -> dict:
    return {
        "heat_coordinate": coordinate,
        "profile": profile,
        "selectors": [_Variable(1.0, fixed=True), _Variable(0.0, fixed=True)],
        "active_segment": 0,
    }


def test_a_mapping_that_returns_to_a_segment_it_left_is_held():
    coordinate = _Variable(2499.0, lb=0.0, ub=2800.0)
    mapping = _mapping(coordinate, _profile([1.0, 1.0]))
    model = SimpleNamespace(_piecewise_active_mappings=[mapping])

    # On the first segment's line the heat lies in the second.
    moved = piecewise._update_piecewise_active_segments(model)
    # On the second's it lies back in the first: held where it is.
    coordinate.value = 1973.0
    held = piecewise._update_piecewise_active_segments(model)
    # Solved again, the heat is at the kink and nothing changes.
    coordinate.value = 2000.0
    settled = not piecewise._update_piecewise_active_segments(model)

    assert moved and held and settled
    assert mapping["active_segment"] == 1
    assert [selector.value for selector in mapping["selectors"]] == [0.0, 1.0]
    assert (coordinate.lb, coordinate.ub) == (2000.0, 2800.0)


def test_a_cycling_contribution_mapping_is_held_on_the_kinks_segment():
    # At a kink a contribution profile carries the stricter of the two
    # segments' contributions, here the first's 3 K: the mapping is held
    # there, not on the second it was in when it cycled.
    coordinate = _Variable(2499.0, lb=0.0, ub=2800.0)
    profile = _profile([3.0, 1.0])
    mapping = _mapping(coordinate, profile) | {
        "segment_index_at_heat": profile.contribution_index_at_heat
    }
    model = SimpleNamespace(_piecewise_active_mappings=[mapping])

    piecewise._update_piecewise_active_segments(model)
    coordinate.value = 1973.0
    held = piecewise._update_piecewise_active_segments(model)

    assert held
    assert mapping["active_segment"] == 0
    assert [selector.value for selector in mapping["selectors"]] == [1.0, 0.0]
    assert (coordinate.lb, coordinate.ub) == (0.0, 2000.0)
