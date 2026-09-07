"""Public boundary regressions found in the repository-wide audit."""

from copy import deepcopy

import pytest

from OpenPinch import PinchProblem
from OpenPinch.contracts.input import TargetInput


@pytest.mark.parametrize("source_kind", ["mapping", "model"])
def test_loaded_inputs_do_not_alias_the_caller(source_kind):
    payload = PinchProblem("basic_pinch.json").to_problem_json()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["0"]
    source = TargetInput.model_validate(payload) if source_kind == "model" else payload
    problem = PinchProblem(source)
    problem.target.all_heat_integration()
    before = deepcopy(problem.to_problem_json())
    results = problem.results.model_dump_json()
    if source_kind == "model":
        source.streams[0].heat_flow = 999999.0
        source.options["PROBLEM_PERIOD_IDS"].append("external")
    else:
        source["streams"][0]["heat_flow"] = 999999.0
        source["options"]["PROBLEM_PERIOD_IDS"].append("external")
    assert problem.to_problem_json() == before
    assert problem.results.model_dump_json() == results


@pytest.mark.parametrize("source_kind", ["mapping", "model"])
def test_validation_result_is_detached_from_loaded_input(source_kind):
    payload = PinchProblem("basic_pinch.json").to_problem_json()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["0"]
    source = TargetInput.model_validate(payload) if source_kind == "model" else payload
    problem = PinchProblem(source)
    before = deepcopy(problem.to_problem_json())
    validated = problem.validate()
    validated.streams[0].heat_flow = 999999.0
    validated.options["PROBLEM_PERIOD_IDS"].append("external")
    assert problem.to_problem_json() == before
    assert problem.period_ids == {"0": 0}
