"""The interior cuts read a non-isothermal match on its own branch.

In the non-isothermal stage-wise model a split stream's branches leave a
stage at their own temperatures, and the stage's mixed outlet lies
between them. The interior approach check of a match, and its cut, use
the match's branch outlets (``T_h_out_x``, ``T_c_out_y``), not the stage's.
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


def test_a_non_isothermal_match_is_checked_on_its_own_branch():
    owner = SimpleNamespace(
        non_isothermal_model=True,
        T_h_by_period=[[[200.0, 170.0]]],
        T_h_mid_by_period=[[[200.0, 170.0]]],
        T_h_out_x_by_period=[[[[165.0]]]],
        T_c_by_period=[[[150.0, 120.0]]],
        T_c_mid_by_period=[[[150.0, 120.0]]],
        T_c_out_y_by_period=[[[[155.0]]]],
    )

    ends = interior_approach._branch_ends(owner, 0, 0, 0, 0)

    assert ends == (200.0, 165.0, 120.0, 155.0)


def test_a_model_without_branch_outlets_is_checked_on_its_stages():
    # The pinch design side model takes non_isothermal_model from the same
    # settings but has stage temperatures only.
    owner = SimpleNamespace(
        non_isothermal_model=True,
        T_h_by_period=[[[200.0, 170.0]]],
        T_h_mid_by_period=[[[200.0, 170.0]]],
        T_c_by_period=[[[150.0, 120.0]]],
        T_c_mid_by_period=[[[150.0, 120.0]]],
    )

    ends = interior_approach._branch_ends(owner, 0, 0, 0, 0)

    assert ends == (200.0, 170.0, 120.0, 150.0)


def _profile(temperatures):
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
        temperature_contributions=np.ones(count),
    )


def test_a_branch_end_a_solver_tolerance_past_its_stream_end_is_checked(
    monkeypatch,
):
    # A solver holds a variable's bounds to a relative tolerance: Ipopt
    # relaxes each by 1e-8 of its size, 4e-6 K on a 140 C target. The match
    # below runs the whole of a hot stream 250 -> 220 -> 150 C against the
    # whole of a cold stream 95 -> 195 C, each branch outlet that far past
    # its stream's end. At the hot kink the cold side is at 165 C: no cut.
    profiles = {
        "hot": _profile([250.0, 220.0, 150.0]),
        "cold": _profile([95.0, 195.0]),
    }
    monkeypatch.setattr(
        piecewise,
        "profile_from_solver_arrays",
        lambda _arrays, *, side, parent_index, period_index: profiles[side],
    )
    owner = SimpleNamespace(
        integers=False,
        N_periods=1,
        S=1,
        I=1,
        J=1,
        z_allowed=[[[1]]],
        Q_r_by_period=[[[[100.0]]]],
        T_h_in_period=[[250.0]],
        T_h_out_period=[[150.0]],
        T_c_in_period=[[95.0]],
        T_c_out_period=[[195.0]],
        T_h_by_period=[[[250.0, 150.0]]],
        T_h_out_x_by_period=[[[[149.999996]]]],
        T_c_by_period=[[[195.0, 95.0]]],
        T_c_out_y_by_period=[[[[195.000004]]]],
        solver_arrays=None,
        _interior_cuts={},
        _interior_constraints={},
    )

    added = interior_approach._add_interior_approach_cuts(owner, [[[1000.0]]])

    assert not added
    assert owner._interior_cuts == {}
