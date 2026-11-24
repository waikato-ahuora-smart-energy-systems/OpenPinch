"""A heat pump that draws on ambient air brings the air as a stream.

When the process has too little heat below the pinch, the target's
evaporator takes the rest from air. Without that heat in the case the
evaporator stream could not be served, so the air comes too: a hot stream
at ambient temperature with the air's own contribution, marked as the
machine's ambient source.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import full_load_target


def _short_of_heat() -> PinchProblem:
    """1000 kW of heating from 40 to 90 degC against 200 kW of waste heat."""

    return PinchProblem(
        source={
            "streams": [
                {
                    "zone": "Plant",
                    "name": "C1",
                    "t_supply": 40.0,
                    "t_target": 90.0,
                    "heat_flow": 1000.0,
                },
                {
                    "zone": "Plant",
                    "name": "H1",
                    "t_supply": 60.0,
                    "t_target": 30.0,
                    "heat_flow": 200.0,
                },
            ],
            "utilities": [
                {
                    "name": "Steam",
                    "type": "Hot",
                    "t_supply": 150.0,
                    "t_target": 149.0,
                    "heat_flow": 0.0,
                },
                {
                    "name": "CW",
                    "type": "Cold",
                    "t_supply": 10.0,
                    "t_target": 15.0,
                    "heat_flow": 0.0,
                },
            ],
        },
        project_name="air",
    )


def test_the_air_an_evaporator_draws_on_is_added_as_its_ambient_source():
    problem = _short_of_heat()
    target = full_load_target(problem)
    # 374 kW at the first start of the search, 428 kW once it has run.
    assert float(target.hpr_ambient_hot) > 300.0

    case = problem.with_heat_pump_streams(target)

    air = [
        record
        for record in case.to_problem_json()["streams"]
        if (record["heat_pump"] or {}).get("role") == "ambient_source"
    ]
    assert [record["name"] for record in air] == ["AIR_H1"]
    assert air[0]["heat_flow"]["value"] == pytest.approx(float(target.hpr_ambient_hot))
    # HPR_DT_ENV_CONT, the air's own contribution, not HPR_DT_CONT.
    assert air[0]["dt_cont"] == {"value": 10.0, "unit": "delta_degC"}
