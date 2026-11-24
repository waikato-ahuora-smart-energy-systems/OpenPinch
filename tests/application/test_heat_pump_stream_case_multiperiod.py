"""A multiperiod problem is refused until its heat pump streams vary by period.

A heat pump target on a problem with several periods describes one
period's machine unless it was solved as a shared design. Its condenser
and evaporator written into every period would be the wrong size in the
others, so the builder refuses rather than build a case that looks
complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import full_load_target

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_a_problem_with_several_periods_is_refused():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )
    target = full_load_target(problem)

    with pytest.raises(ValueError, match="3 periods"):
        problem.with_heat_pump_streams(target)
