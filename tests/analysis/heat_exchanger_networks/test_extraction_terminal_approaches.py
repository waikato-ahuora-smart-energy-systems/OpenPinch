"""A multiperiod recovery match reports the approaches its streams pass at.

The stage-wise model's approach variables (``theta_1``, ``theta_2``) only
bound the terminal temperature differences. Where the area is sized in
another period they do not bind, and the solver leaves them below the real
differences: on the dairy case on ``design_basis="all"`` a match reported
(0.94, 6.30) K in turndown where its streams pass at (0.94, 8.19) K (review
finding 26). In one period they bind, so a single-period network keeps them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
    extract_heat_exchanger_network,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)


def _two_periods() -> PinchProblem:
    def stream(name, supply, target, duty):
        return {
            "zone": "Site",
            "name": name,
            "t_supply": supply,
            "t_target": target,
            "heat_flow": {"values": duty, "unit": "kW"},
        }

    return PinchProblem(
        {
            "streams": [
                stream("H1", 200.0, 100.0, [100.0, 100.0]),
                stream("C1", 50.0, 135.0, [85.0, 85.0]),
            ],
            "utilities": [
                {"name": "HU", "type": "Hot", "t_supply": 250.0, "t_target": 249.0},
                {"name": "CU", "type": "Cold", "t_supply": 20.0, "t_target": 25.0},
            ],
            "options": {
                "PROBLEM_PERIOD_IDS": ["a", "b"],
                "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
            },
        }
    )


def test_a_multiperiod_match_reports_its_terminal_temperature_differences():
    arrays = problem_to_solver_arrays(_two_periods(), 10.0)
    # One stage; in period b the approach variables sit well under the
    # differences the streams pass at (65 K at both ends).
    solved = SimpleNamespace(
        S=1,
        Q_r=[[[85.0]]],
        Q_r_by_period=[[[[85.0]]], [[[85.0]]]],
        z=[[[1.0]]],
        z_allowed=[[[1]]],
        Q_h=[0.0],
        Q_c=[0.0],
        z_hu=[0.0],
        z_cu=[0.0],
        T_h=[[473.15, 388.15]],
        T_c=[[408.15, 323.15]],
        T_h_by_period=[[[473.15, 388.15]], [[473.15, 388.15]]],
        T_c_by_period=[[[408.15, 323.15]], [[408.15, 323.15]]],
        theta_1_by_period=[[[[65.0]]], [[[10.0]]]],
        theta_2_by_period=[[[[65.0]]], [[[12.0]]]],
        area_r=[[[999.0]]],
    )

    network = extract_heat_exchanger_network(solved, arrays, run_id="periods")

    state = network.exchangers[0].state("b")
    assert state.approach_temperatures == pytest.approx((65.0, 65.0))
