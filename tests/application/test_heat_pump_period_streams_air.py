"""A heat pump of several periods that draws on air brings the air too.

The short-of-heat case over two periods, heating 800 kW in one and
1000 kW in the other against 200 kW of waste heat: the shared design's
evaporator takes the rest from air, more in the second period. The air
is the same stream in each period (the ambient temperature is one
setting), at that period's air duty, so the case writes it with a duty
per period at its own contribution (``HPR_DT_ENV_CONT``).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import full_load_target
from tests.application.test_heat_pump_stream_case_air_source import _short_of_heat


def short_of_heat_in_two_periods() -> PinchProblem:
    source = _short_of_heat().to_problem_json()
    source["options"] = (source.get("options") or {}) | {
        "PROBLEM_PERIOD_IDS": ["low", "high"],
        "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
        "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True,
    }
    for stream in source["streams"]:
        if stream["name"] == "C1":
            stream["heat_flow"] = {"values": [800.0, 1000.0], "unit": "kW"}
    return PinchProblem(source=source, project_name="air periods")


def test_the_air_is_written_at_each_periods_duty():
    problem = short_of_heat_in_two_periods()
    target = full_load_target(problem)
    outputs = target.hpr_details.period_outputs
    drawn = [float(outputs[period]["Q_amb_hot"]) for period in ("low", "high")]
    assert drawn[1] > drawn[0] > 0.0

    case = problem.with_heat_pump_streams(target)

    air = [
        record
        for record in case.to_problem_json()["streams"]
        if (record["heat_pump"] or {}).get("role") == "ambient_source"
    ]
    assert [record["name"] for record in air] == ["AIR_H1"]
    assert air[0]["heat_flow"]["values"] == pytest.approx(drawn)
    assert air[0]["dt_cont"] == {"value": 10.0, "unit": "delta_degC"}
