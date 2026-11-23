"""A heat pump design case on several periods is solved as a shared design.

A design serves every period with one network, so its heat pump has to be
one machine over every period too. The case asks the target for that
itself: a caller need not know the option.
"""

from __future__ import annotations

import json
from pathlib import Path

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_the_case_on_several_periods_holds_streams_for_every_period():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )

    run = heat_pump_design_case(
        problem,
        1.0,
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
    )

    assert run.target.hpr_shared_design
    marked = [
        record
        for record in run.case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    ]
    assert [len(record["heat_flow"]["values"]) for record in marked] == [3, 3]
