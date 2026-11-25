"""Network evolution settles a stream whose optimum sits at its kink.

Without integers a segmented stream follows one segment's line at a
time, extrapolated, and the solve loop moves it to the segment its
solved heat lies in. Where the optimum is at a kink, each line puts it in
the other segment: on the case below, hot stream H1 (2000 kW down to
230 degC, then 800 kW) solved at 2499 kW on its first line and at
1973 kW on its second, turn about, until the solve ran out of attempts
("piecewise active segments did not stabilise"), and evolution from that
network returned nothing.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _stream(name: str, dt_cont: float, segments: list[tuple]) -> dict:
    return {
        "zone": "P",
        "name": name,
        "dt_cont": dt_cont,
        "segments": [
            {
                "t_supply": supply,
                "t_target": target,
                "heat_flow": duty,
                "htc": htc,
                "dt_cont": dt_cont,
            }
            for supply, target, duty, htc in segments
        ],
    }


def _two_kinks_problem() -> PinchProblem:
    """Two hot and two cold streams; H1 and C1 each change heat capacity
    flowrate once."""

    source = {
        "options": {
            "COSTING_ANNUAL_OP_TIME": 1000.0,
            "COSTING_DISCOUNT_RATE": 0.0,
            "COSTING_SERVICE_LIFE": 1.0,
            "COSTING_HX_AREA_COEFF": 150.0,
            "COSTING_HX_AREA_EXP": 1.0,
            "COSTING_HX_UNIT_COST": 5500.0,
            "HENS_DT_CONT_MULTIPLIERS": [1.0],
            "HENS_MAX_PARALLEL": 1,
        },
        "streams": [
            _stream(
                "H1", 5.0, [(380.0, 230.0, 2000.0, 1.0), (230.0, 100.0, 800.0, 0.5)]
            ),
            {
                "zone": "P",
                "name": "H2",
                "t_supply": 320.0,
                "t_target": 100.0,
                "heat_flow": 4400.0,
                "htc": 1.0,
                "dt_cont": 5.0,
            },
            _stream(
                "C1", 5.0, [(140.0, 250.0, 1100.0, 0.5), (250.0, 380.0, 2500.0, 1.0)]
            ),
            {
                "zone": "P",
                "name": "C2",
                "t_supply": 80.0,
                "t_target": 230.0,
                "heat_flow": 1950.0,
                "htc": 1.0,
                "dt_cont": 5.0,
            },
        ],
        "utilities": [
            {
                "name": "HPS",
                "type": "Hot",
                "t_supply": 410.0,
                "t_target": 409.9,
                "htc": 5.0,
                "price": 80.0,
                "dt_cont": 5.0,
            },
            {
                "name": "CW",
                "type": "Cold",
                "t_supply": 30.0,
                "t_target": 50.0,
                "htc": 1.0,
                "price": 15.0,
                "dt_cont": 5.0,
            },
        ],
    }
    return PinchProblem(source, project_name="two kinks")


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs highspy and ipopt",
)
def test_evolution_settles_a_stream_whose_optimum_is_at_its_kink():
    # The cheapest pinch design network is the seed whose root cycles; the
    # others' roots settle.
    problem = _two_kinks_problem()
    seed = problem.design.pinch_design().result.ranked_networks[0].network

    view = problem.design.network_evolution([seed])

    assert [outcome.network for outcome in view.result.ranked_networks]
