"""A heat pump target becomes streams of a new case.

``problem.with_heat_pump_streams(target)`` returns the problem with each
condenser of the target as a hot stream and each evaporator as a cold
stream, marked with their machine and role and carrying ``HPR_DT_CONT``,
the approach the target assumed against the process. Targeting the new
case for direct heat integration then gives the heat pump target's
utilities, and synthesis designs the process and the heat pump together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import full_load_target

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)


@pytest.fixture(scope="module")
def solved():
    """The basic pinch case and its Carnot heat pump target at full load."""

    problem = PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )
    return problem, full_load_target(problem)


def _heat_pump_records(case) -> dict[str, dict]:
    return {
        record["name"]: record
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }


def test_the_new_case_holds_the_condenser_and_evaporator_as_marked_streams(solved):
    problem, target = solved

    case = problem.with_heat_pump_streams(target)

    records = _heat_pump_records(case)
    assert {name: r["heat_pump"] for name, r in records.items()} == {
        "HP_H1": {"machine": "HP", "role": "condenser"},
        "HP_C1": {"machine": "HP", "role": "evaporator"},
    }
    for record in records.values():
        assert record["zone"] == "basic/Plant"
        assert record["dt_cont"] == {"value": 1.0, "unit": "delta_degC"}
    condenser = next(stream for _key, stream in target.hpr_hot_streams.items())
    assert records["HP_H1"]["heat_flow"]["value"] == pytest.approx(
        float(condenser.heat_flow.value)
    )
    assert len(case.to_problem_json()["streams"]) == (
        len(problem.to_problem_json()["streams"]) + 2
    )
