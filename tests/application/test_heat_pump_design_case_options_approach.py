"""``HPR_DT_CONT`` given in the heat pump's options is the heat pump's approach.

``options`` are the target method's options, and ``HPR_DT_CONT`` is how a
target is told its approach there. The case must target at that value and
write its streams with it, not fall back to the problem's own setting.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.application.heat_pump_stream_cases import basic_source


def test_the_approach_in_the_heat_pumps_options_is_used():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    run = heat_pump_design_case(
        problem,
        1.0,
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
        options={"HPR_DT_CONT": 7.0},
    )

    assert run.target.provenance.effective_settings["HPR_DT_CONT"] == 7.0
    assert [
        record["dt_cont"]["value"]
        for record in run.case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    ] == [7.0, 7.0]
