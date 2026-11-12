"""The first pinch design run uses each stream's ΔT contribution as entered."""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.configuration import Configuration
from tests.support.paths import REPOSITORY_ROOT

BASIC_PINCH = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/basic_pinch.json"


def _basic_pinch(options: dict | None = None) -> PinchProblem:
    payload = json.loads(BASIC_PINCH.read_text(encoding="utf-8"))
    # Distinct contributions, so scaling would show.
    for index, stream in enumerate(payload["streams"]):
        stream["dt_cont"] = {"value": 2.0 + index, "unit": "degC"}
    payload["options"] = payload.get("options", {}) | (options or {})
    problem = PinchProblem(source=payload, project_name="basic")
    problem.target.all_heat_integration()
    return problem


def test_approach_temperatures_are_unset_by_default():
    assert Configuration().hens.approach_temperatures is None


@pytest.mark.parametrize(("dt_cont", "approach"), [(5.0, 10.0), (7.5, 15.0)])
def test_unset_approach_temperature_is_twice_the_reference_contribution(
    dt_cont: float, approach: float
):
    problem = _basic_pinch({"THERMAL_DT_CONT": dt_cont})

    assert workflow_settings_from_problem(problem).approach_temperatures == (approach,)


def test_set_approach_temperatures_are_kept():
    problem = _basic_pinch({"HENS_APPROACH_TEMPERATURES": [14.0, 20.0]})

    assert workflow_settings_from_problem(problem).approach_temperatures == (
        14.0,
        20.0,
    )


def test_first_pdm_run_uses_each_stream_contribution_as_entered():
    problem = _basic_pinch()
    first = workflow_settings_from_problem(problem).approach_temperatures[0]

    arrays = problem_to_solver_arrays(problem, first).arrays

    contributions = sorted(
        float(value)
        for key in ("T_h_cont_period", "T_c_cont_period")
        for value in arrays[key][0]
    )
    entered = sorted(
        2.0 + index
        for index in range(len(json.loads(BASIC_PINCH.read_text())["streams"]))
    )
    assert contributions == pytest.approx(entered)
