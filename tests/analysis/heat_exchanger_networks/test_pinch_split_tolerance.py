"""Streams that start or end at the pinch (issue 297).

The problem table rounds temperatures to ``tol``, so the pinch can sit up to
half of ``tol`` past the temperature of the stream that sets it. The pinch
decomposition must not keep that stream on the far side as a sliver with no
heat load: its heat balance then has no variables and Pyomo cannot build it.
"""

from __future__ import annotations

import numpy as np
import pytest

import OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition as pdm_decomposition
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
)
from OpenPinch.application.problem import PinchProblem

# How far the rounded pinch sat from C1's supply in the issue's case, in K.
SLIVER = 2.1e-7


@pytest.mark.parametrize(
    ("pinch_location", "T_h_in", "T_h_out", "T_c_in", "T_c_out"),
    [
        (
            "above",
            [405.0 + SLIVER, 500.0],
            [380.0, 450.0],
            [380.0, 390.0],
            [395.0 + SLIVER, 430.0],
        ),
        (
            "below",
            [500.0, 410.0],
            [405.0 - SLIVER, 380.0],
            [395.0 - SLIVER, 370.0],
            [430.0, 390.0],
        ),
    ],
)
def test_streams_within_tol_of_the_pinch_are_inactive_past_it(
    pinch_location, T_h_in, T_h_out, T_c_in, T_c_out
):
    z_i_active, z_j_active = pdm_decomposition._clip_stream_temperatures(
        T_h_in=np.array(T_h_in),
        T_h_out=np.array(T_h_out),
        T_c_in=np.array(T_c_in),
        T_c_out=np.array(T_c_out),
        hot_contributions=np.array([5.0, 5.0]),
        cold_contributions=np.array([5.0, 5.0]),
        shifted_pinch_temperature=400.0,
        pinch_location=pinch_location,
    )

    assert z_i_active == (0, 1)
    assert z_j_active == (0, 1)


def _couenne_available() -> bool:
    try:
        require_solver_binary("couenne", purpose="pinch split regression")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _stream(name, t_supply, t_target, heat_flow):
    return {
        "zone": "Z[P]",
        "name": name,
        "t_supply": {"value": t_supply, "unit": "degC"},
        "t_target": {"value": t_target, "unit": "degC"},
        "heat_flow": {"value": heat_flow, "unit": "kW"},
        "dt_cont": {"value": 5.0, "unit": "degC"},
    }


def _utility(name, kind, t_supply, t_target):
    return {
        "name": name,
        "type": kind,
        "t_supply": {"value": t_supply, "unit": "degC"},
        "t_target": {"value": t_target, "unit": "degC"},
        "dt_cont": {"value": 0, "unit": "degC"},
        "heat_flow": None,
    }


@pytest.mark.solver
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif("not _couenne_available()", reason="needs the couenne binary")
def test_pinch_design_solves_when_the_pinch_stream_has_a_many_decimal_supply():
    problem = PinchProblem(
        {
            "streams": [
                _stream("C1", 64.123456789, 90.0, 500.0),
                _stream("H1", 46.0, 45.0, 60.0),
            ],
            "utilities": [
                _utility("Steam", "Hot", 180, 179),
                _utility("Cooling water", "Cold", 20, 25),
            ],
            "options": {},
        },
        project_name="issue_297",
    )
    problem.target.direct_heat_integration()

    view = problem.design.pinch_design(approach_temperatures=[10.0])

    # H1 is too cold to heat C1, so steam supplies all of C1's 500 kW.
    assert view.total_hot_utility == pytest.approx(500.0, abs=1e-3)
