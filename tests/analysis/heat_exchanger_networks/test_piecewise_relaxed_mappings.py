"""A relaxed piecewise mapping is one equality, whatever its segments.

A solver without integers follows one segment of a profile at a time:
each segment has a selector parameter, 1 on the active segment and 0 on
the others, which the solve loop moves. One equality per segment left
every inactive one in the model with nothing in it, and Ipopt counts
equalities against variables before it starts ("Problem has too few
degrees of freedom").
"""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_exchanger_networks.solver.piecewise import (
    PiecewiseThermalProfile,
    add_piecewise_cost_mapping,
    add_piecewise_temperature_contribution_mapping,
    add_piecewise_temperature_mapping,
)


class _Number(float):
    """A model variable or parameter, as far as a mapping reads one."""

    def __new__(cls, value: float, fixed: bool):
        number = super().__new__(cls, value)
        number.fixed = fixed
        return number

    def is_fixed(self) -> bool:
        return self.fixed


class _Model:
    """Records what a mapping adds."""

    def __init__(self) -> None:
        self.equations = []

    def Var(self, value=0.0, lb=None, ub=None, integer=False):
        return _Number(value, fixed=False)

    def Param(self, value=0.0):
        return _Number(value, fixed=True)

    def Equation(self, expression):
        self.equations.append(expression)
        return expression


def _profile() -> PiecewiseThermalProfile:
    """A hot stream 200 -> 150 -> 100 -> 50 degC, 50 kW a segment."""

    temperatures = np.array([200.0, 150.0, 100.0, 50.0])
    return PiecewiseThermalProfile(
        identities=("s0", "s1", "s2"),
        temperatures_in=temperatures[:-1],
        temperatures_out=temperatures[1:],
        duties=np.full(3, 50.0),
        heat_capacity_flowrates=np.ones(3),
        heat_transfer_coefficients=np.ones(3),
        prices=np.array([1.0, 2.0, 3.0]),
        temperature_contributions=np.array([1.0, 2.0, 3.0]),
    )


def test_a_relaxed_temperature_mapping_is_one_equality():
    model = _Model()

    mapping = add_piecewise_temperature_mapping(
        model,
        model.Var(value=75.0),
        model.Var(value=125.0),
        _profile(),
        integer_capable=False,
        initial_segment=1,
    )

    assert len(model.equations) == 1
    assert len(mapping["selectors"]) == 3


def test_a_relaxed_temperature_mapping_of_two_parameters_adds_nothing():
    # A stream's supply end: its temperature and its heat (zero) are both
    # parameters, so an equality between them would have no variable in it.
    model = _Model()

    mapping = add_piecewise_temperature_mapping(
        model,
        model.Param(value=0.0),
        model.Param(value=200.0),
        _profile(),
        integer_capable=False,
    )

    assert model.equations == []
    assert mapping is None


def test_a_relaxed_cost_mapping_is_one_equality():
    model = _Model()

    mapping = add_piecewise_cost_mapping(
        model,
        model.Var(value=75.0),
        model.Var(value=100.0),
        _profile(),
        integer_capable=False,
    )

    assert len(model.equations) == 1
    assert len(mapping["selectors"]) == 3


def test_a_contribution_mapping_is_one_equality():
    model = _Model()

    mapping = add_piecewise_temperature_contribution_mapping(
        model,
        model.Var(value=75.0),
        model.Var(value=2.0),
        _profile(),
        initial_segment=1,
    )

    assert len(model.equations) == 1
    assert len(mapping["selectors"]) == 3
