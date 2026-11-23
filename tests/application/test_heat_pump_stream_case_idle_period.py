"""A case is built when the heat pump is off in the period its target shows.

A shared design's own stream collections hold one period's streams. When
that period is one the machine is off in, they are empty, and the heat
pump's streams have to come from the periods it runs in: with no duty in
the idle period, not left out of the case.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _case_with_heat_pump_streams
from tests.application.heat_pump_stream_cases import full_load_target

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_streams_come_from_the_running_periods_when_the_shown_period_is_idle():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )
    solved = full_load_target(
        problem, options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True}
    )
    outputs = dict(solved.hpr_details.period_outputs)
    base = float(
        next(s for _key, s in outputs["base"]["hpr_hot_streams"].items()).heat_flow.value
    )
    # The same design with the machine off at turndown, the period shown.
    del outputs["turndown"]
    target = SimpleNamespace(
        provenance=solved.provenance,
        hpr_shared_design=True,
        hpr_hot_streams={},
        hpr_cold_streams={},
        hpr_details=SimpleNamespace(amb_streams={}, period_outputs=outputs),
    )

    case = _case_with_heat_pump_streams(problem, target, dt_cont=1.0)

    duties = {
        record["name"]: record["heat_flow"]["values"]
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    assert set(duties) == {"HP_H1", "HP_C1"}
    assert duties["HP_H1"][0] == 0.0
    assert duties["HP_H1"][1] == pytest.approx(base)
