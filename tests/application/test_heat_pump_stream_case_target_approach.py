"""Heat pump streams carry the approach their target was solved with.

A target can be solved with its own ``minimum_approach_temperature``,
which sets ``HPR_DT_CONT`` for that solve alone. The streams must carry
that value, not the problem's: at the problem's 1 K a condenser sized at
7 K would be designed 6 K closer to the process than the target allowed.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import basic_source, full_load_target


def test_streams_take_the_approach_the_target_was_solved_with():
    problem = PinchProblem(source=basic_source(), project_name="basic")
    target = full_load_target(problem, minimum_approach_temperature=7.0)

    case = problem.with_heat_pump_streams(target)

    assert [
        record["dt_cont"]
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    ] == [{"value": 7.0, "unit": "delta_degC"}] * 2
