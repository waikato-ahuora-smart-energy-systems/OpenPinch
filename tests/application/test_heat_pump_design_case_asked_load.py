"""A heat pump's record names the load fraction it was asked for.

A target says what share of the available load it took, and the record
takes its size from that. A shared design over several periods gives no
such summary, so the record falls back on the fraction the case was asked
for: without it the sizes in one ranking could not be told apart.
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


def test_a_shared_designs_record_takes_the_fraction_it_was_asked_for():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )

    run = heat_pump_design_case(
        problem,
        1.0,
        load={"fraction": 0.5},
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
    )

    assert run.record.load_fraction == 0.5
