"""The interior approach check holds each segment's own contribution.

A segmented stream's segments may carry different temperature
contributions. At a boundary inside a match the approach is the one of the
segments meeting there (the stricter of the two at the boundary itself),
not the parent stream's.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    interior_approach as equations,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import piecewise
from OpenPinch.analysis.heat_exchanger_networks.solver.piecewise import (
    PiecewiseThermalProfile,
)


def _profile(temperatures, cp, contributions):
    temperatures_in = np.asarray(temperatures[:-1], dtype=float)
    temperatures_out = np.asarray(temperatures[1:], dtype=float)
    count = len(temperatures_in)
    return PiecewiseThermalProfile(
        identities=tuple(f"s{index}" for index in range(count)),
        temperatures_in=temperatures_in,
        temperatures_out=temperatures_out,
        duties=np.abs(temperatures_out - temperatures_in) * cp,
        heat_capacity_flowrates=np.full(count, float(cp)),
        heat_transfer_coefficients=np.ones(count),
        temperature_contributions=np.asarray(contributions, dtype=float),
    )


def _owner():
    """One match in one stage: the hot stream 200 -> 150 -> 100 C, whose
    lower segment has a contribution of 10 K, against a cold stream
    95 -> 195 C of 1 K. The parent pair approach is 2 K; at the 150 C
    boundary the cold side is at 145 C, 5 K short of the 11 K held there."""

    def grid(value):
        return [[[value, value]]]

    return SimpleNamespace(
        N_periods=1,
        S=1,
        I=1,
        J=1,
        z_allowed=[[[1]]],
        z_i_active_period=[[1]],
        z_j_active_period=[[1]],
        Q_r_by_period=[[[[100.0]]]],
        T_h_in_period=[[200.0]],
        T_h_out_period=[[100.0]],
        T_c_in_period=[[95.0]],
        T_c_out_period=[[195.0]],
        T_h_by_period=[[[200.0, 100.0]]],
        T_h_mid_by_period=[[[200.0, 100.0]]],
        T_c_by_period=[[[195.0, 95.0]]],
        T_c_mid_by_period=[[[195.0, 95.0]]],
        dT_r=grid(2.0),
        dT_r_period=[[[2.0]]],
        solver_arrays=None,
    )


def test_a_boundary_short_of_its_segment_approach_is_cut_at_that_approach(
    monkeypatch,
):
    profiles = {
        "hot": _profile([200.0, 150.0, 100.0], 1.0, [1.0, 10.0]),
        "cold": _profile([95.0, 195.0], 1.0, [1.0]),
    }
    monkeypatch.setattr(
        piecewise,
        "profile_from_solver_arrays",
        lambda _arrays, *, side, parent_index, period_index: profiles[side],
    )
    cuts = []
    monkeypatch.setattr(
        equations,
        "_cut_pair",
        lambda _owner, _big_m, n, i, j, side, boundary, approach: (
            cuts.append((side, boundary, approach)) or True
        ),
    )

    added = equations._add_interior_approach_cuts(_owner(), [[[1000.0]]])

    assert added
    assert cuts == [("hot", 150.0, 11.0)]
