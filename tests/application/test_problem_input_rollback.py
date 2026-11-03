"""A failed source replacement leaves the problem as it was."""

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application._problem.input.loading import (
    prepare_in_memory_problem_source,
    replace_loaded_source,
)


def test_failed_replacement_keeps_retained_hpr_targets():
    problem = PinchProblem("OpenPinch/tutorials/sample_cases/process_mvr.json")
    retained = {"identity": object()}
    problem._retained_hpr_targets = retained
    source = prepare_in_memory_problem_source(
        problem.to_problem_json(), source_kind="target_input"
    )

    def failing_rebuild():
        problem._rebuild_problem_state()
        raise RuntimeError("component replay failed")

    with pytest.raises(RuntimeError, match="component replay failed"):
        replace_loaded_source(problem, source, rebuild=failing_rebuild)

    assert problem._retained_hpr_targets == retained
