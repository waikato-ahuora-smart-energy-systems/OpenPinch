"""A heat pump design case's record carries the machine's streams.

A network designed on the case names the heat pump's condenser and
evaporator streams, which the problem itself does not hold. The record
every such network takes holds those streams as the case's input records,
so the case can be built again from the problem and the network alone.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.application.heat_pump_stream_cases import basic_source


def test_the_record_holds_the_stream_records_the_case_added():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    run = heat_pump_design_case(
        problem,
        1.0,
        cycle="carnot",
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
        cost_per_kw=200.0,
    )

    added = [
        record
        for record in run.case.to_problem_json()["streams"]
        if record.get("heat_pump")
    ]
    assert [record["name"] for record in added] == ["HP_H1", "HP_C1"]
    assert list(run.record.streams) == added
