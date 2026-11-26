"""An overdesign margin keeps the networks that idle an exchanger in a period.

A waste heat stream H2 is hot in two periods and lukewarm in the third,
where it arrives 2 K above the cold stream C2 and below where C1 sits once
H1 has heated it. A design on every period (``design_basis="all"``) leaves
its H2 matches idle in that period. The margin is applied by rating each
network, and the rating held every match's approach in every period, idle
or not: those networks could not be rated and were dropped without a word.
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


pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.solver,
    pytest.mark.skipif(
        not (_available("highs") and _available("ipopt-pyomo")),
        reason="needs highspy and ipopt",
    ),
]


def _quantity(value, unit):
    key = "values" if isinstance(value, list) else "value"
    return {key: value, "unit": unit}


def _stream(name, supply, target, heat_flow):
    return {
        "zone": "P",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "htc": _quantity(1.0, "kW/m^2/degC"),
        "dt_cont": _quantity(5.0, "delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "htc": _quantity(1.0, "kW/m^2/degC"),
        "price": _quantity(price, "$/MWh"),
    }


def lukewarm_period_problem() -> PinchProblem:
    return PinchProblem(
        {
            "options": {
                "PROBLEM_PERIOD_IDS": ["a", "b", "c"],
                "PROBLEM_PERIOD_WEIGHTS": [0.4, 0.4, 0.2],
                "COSTING_HX_AREA_COEFF": 1000.0,
            },
            "streams": [
                _stream("H1", 180.0, 60.0, 1200.0),
                _stream("H2", [140.0, 140.0, 62.0], 40.0, [1000.0, 1000.0, 220.0]),
                _stream("C1", 30.0, 150.0, 1200.0),
                _stream("C2", 60.0, 120.0, 900.0),
            ],
            "utilities": [
                _utility("HU", "Hot", 250.0, 249.9, 40.0),
                _utility("CU", "Cold", 15.0, 20.0, 5.0),
            ],
        },
        project_name="lukewarm",
    )


def _design(**arguments):
    return lukewarm_period_problem().design.heat_exchanger_network(
        design_basis="all", width=0, depth=0, **arguments
    )


def test_an_overdesign_margin_drops_no_network_that_idles_in_a_period():
    designed = _design()
    with_margin = _design(overdesign_percentage=10.0)

    assert len(with_margin.result.ranked_networks) == len(
        designed.result.ranked_networks
    )
