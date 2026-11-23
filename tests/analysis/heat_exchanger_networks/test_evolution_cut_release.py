"""A cut without integers is released once its boundary leaves the branch.

Network evolution's models have no integers, so a cut for a boundary
inside a match holds in that stage whatever the branch does next. If the
re-solved branch no longer spans the boundary, the cut no longer stands
for a crossing; where it still binds it holds the stream back for nothing
(or makes the stage infeasible), so it is released and the model solved
again.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    interior_approach,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import piecewise
from OpenPinch.analysis.heat_exchanger_networks.solver.piecewise import (
    PiecewiseThermalProfile,
)

KEY = (0, 0, 0, 0, "hot", 150.0)


class _Constraint:
    def __init__(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def activate(self) -> None:
        self.active = True


def _profile(temperatures, contributions):
    temperatures_in = np.asarray(temperatures[:-1], dtype=float)
    temperatures_out = np.asarray(temperatures[1:], dtype=float)
    count = len(temperatures_in)
    return PiecewiseThermalProfile(
        identities=tuple(f"s{index}" for index in range(count)),
        temperatures_in=temperatures_in,
        temperatures_out=temperatures_out,
        duties=np.abs(temperatures_out - temperatures_in),
        heat_capacity_flowrates=np.ones(count),
        heat_transfer_coefficients=np.ones(count),
        temperature_contributions=np.asarray(contributions, dtype=float),
    )


def _owner(constraint):
    """The hot stream 200 -> 150 -> 100 C now runs 200 -> 170 C in the
    stage, above its 150 C boundary, and the cold side leaves at 148 C: the
    cut from an earlier crossing (boundary less cold outlet at least 2 K)
    binds though the boundary is no longer inside the match."""

    return SimpleNamespace(
        integers=False,
        N_periods=1,
        S=1,
        I=1,
        J=1,
        z_allowed=[[[1]]],
        Q_r_by_period=[[[[30.0]]]],
        T_h_in_period=[[200.0]],
        T_h_out_period=[[100.0]],
        T_c_in_period=[[95.0]],
        T_c_out_period=[[195.0]],
        T_h_by_period=[[[200.0, 170.0]]],
        T_h_mid_by_period=[[[200.0, 170.0]]],
        T_c_by_period=[[[148.0, 118.0]]],
        T_c_mid_by_period=[[[148.0, 118.0]]],
        solver_arrays=None,
        _interior_cuts={KEY: (150.0, 2.0)},
        _interior_constraints={KEY: constraint},
    )


def test_a_binding_cut_whose_boundary_left_the_branch_is_released(monkeypatch):
    profiles = {
        "hot": _profile([200.0, 150.0, 100.0], [1.0, 1.0]),
        "cold": _profile([95.0, 195.0], [1.0]),
    }
    monkeypatch.setattr(
        piecewise,
        "profile_from_solver_arrays",
        lambda _arrays, *, side, parent_index, period_index: profiles[side],
    )
    constraint = _Constraint()

    again = interior_approach._add_interior_approach_cuts(
        _owner(constraint), [[[1000.0]]]
    )

    assert again
    assert not constraint.active


def test_a_released_cut_crossed_again_comes_back_for_good(monkeypatch):
    # Released, the branch may cross at the boundary again: the cut is put
    # back and kept from then on, so the solve does not cycle.
    profiles = {
        "hot": _profile([200.0, 150.0, 100.0], [1.0, 1.0]),
        "cold": _profile([95.0, 195.0], [1.0]),
    }
    monkeypatch.setattr(
        piecewise,
        "profile_from_solver_arrays",
        lambda _arrays, *, side, parent_index, period_index: profiles[side],
    )
    constraint = _Constraint()
    constraint.deactivate()
    owner = _owner(constraint)
    # Across the boundary again: hot 200 -> 100 C against cold 120 -> 195 C
    # puts the cold side at 157.5 C at the boundary, above it.
    owner.T_h_by_period = owner.T_h_mid_by_period = [[[200.0, 100.0]]]
    owner.T_c_by_period = owner.T_c_mid_by_period = [[[195.0, 120.0]]]

    again = interior_approach._add_interior_approach_cuts(owner, [[[1000.0]]])
    owner.T_h_by_period = owner.T_h_mid_by_period = [[[200.0, 170.0]]]
    owner.T_c_by_period = owner.T_c_mid_by_period = [[[148.0, 118.0]]]
    interior_approach._add_interior_approach_cuts(owner, [[[1000.0]]])

    assert again
    assert constraint.active


def test_a_cut_without_integers_is_recorded_for_release():
    added = []

    def equation(expression):
        added.append(_Constraint())
        return added[-1]

    owner = SimpleNamespace(
        integers=False,
        m=SimpleNamespace(Equation=equation),
        z=[[[1.0]]],
        T_c_mid_by_period=[[[148.0, 118.0]]],
        _interior_cuts={},
        _interior_constraints={},
    )

    interior_approach._cut(owner, [[[1000.0]]], 0, 0, 0, 0, "hot", 150.0, 2.0)

    assert owner._interior_constraints == {KEY: added[0]}
