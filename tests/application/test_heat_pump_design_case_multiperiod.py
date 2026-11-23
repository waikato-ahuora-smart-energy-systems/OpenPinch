"""A heat pump design case is refused on a problem with several periods.

The same refusal as ``with_heat_pump_streams``: a target's streams hold
one period's values, and written into every period they would be the
wrong size in the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_a_design_case_is_refused_on_a_problem_with_several_periods():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )

    with pytest.raises(ValueError, match="3 periods"):
        heat_pump_design_case(
            problem,
            1.0,
            levels={"condensers": 1, "evaporators": 1},
            search={"restarts": 0, "evaluations": 1},
        )
