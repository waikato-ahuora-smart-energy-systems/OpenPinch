"""A problem a heat pump design cannot take is refused before any design.

The refusal does not depend on anything a design finds, so it comes
first: not after the design without a heat pump has run its solvers and
replaced the problem's stored design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_several_periods_are_refused_before_the_design_without_a_heat_pump():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )

    with pytest.raises(ValueError, match="3 periods"):
        problem.design.heat_exchanger_network(
            heat_pump={"search": {"restarts": 0, "evaluations": 1}},
            heat_pump_cost=200.0,
            width=0,
            depth=0,
            max_parallel=1,
        )

    assert problem.results is None or problem.results.design is None
