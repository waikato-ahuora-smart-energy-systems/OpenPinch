"""Real public HPR/MVR workflows over the standard targeting corpus."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from OpenPinch import PinchProblem
from OpenPinch.contracts.hpr import HPRTargetingError
from tests.e2e.cases import standard_problem_paths
from tests.e2e.hpr_benchmark import (
    HPRBenchmarkAssignment,
    HPROutcomeKind,
    assert_atomic_outcome,
    assert_bounded_convergence,
    assert_strict_success_contract,
    assert_typed_failure_contract,
    build_assignments,
    build_convergence_witness,
    classify_hpr_outcome,
    is_sentinel_assignment,
    observe_hpr_search,
    prepare_hpr_baseline,
    profile_search_observations,
    snapshot_problem_state,
)

HPR_BENCHMARK_ASSIGNMENTS = build_assignments(standard_problem_paths())


@pytest.mark.parametrize(
    "assignment",
    HPR_BENCHMARK_ASSIGNMENTS,
    ids=lambda assignment: assignment.parameter_id,
)
def test_standard_problem_hpr_service_is_bounded_and_robust(
    assignment: HPRBenchmarkAssignment,
) -> None:
    problem = PinchProblem(
        json.loads(assignment.path.read_text()),
        project_name=assignment.path.stem.removeprefix("p_"),
    )
    prepare_hpr_baseline(problem, assignment.profile)
    before = snapshot_problem_state(problem)
    error = None

    with observe_hpr_search() as observations:
        try:
            target = getattr(problem.target, assignment.profile.service_name)(
                **assignment.profile.invocation_kwargs()
            )
        except HPRTargetingError as caught:
            target = None
            error = caught

    outcome = classify_hpr_outcome(target=target, error=error)
    after = snapshot_problem_state(problem)
    assert_atomic_outcome(before, after, outcome)
    assert len(observations) <= assignment.profile.maximum_search_observations, (
        f"{assignment.parameter_id} exceeded its public search observation bound"
    )
    selected_observations = profile_search_observations(
        observations, assignment.profile
    )
    assert len(selected_observations) <= assignment.profile.maximum_evaluations

    if outcome.kind is HPROutcomeKind.SOLVED:
        assert_strict_success_contract(outcome.target, problem)
    elif outcome.kind is HPROutcomeKind.TYPED_FAILURE:
        assert_typed_failure_contract(
            outcome.error,
            maximum_evaluations=assignment.profile.maximum_evaluations,
        )

    if is_sentinel_assignment(assignment):
        assert outcome.kind is HPROutcomeKind.SOLVED, (
            f"strict-success sentinel {assignment.parameter_id} returned "
            f"{outcome.kind.value}"
        )
        witness = build_convergence_witness(
            selected_observations,
            selected_objective=float(outcome.target.hpr_details.obj),
            maximum_evaluations=assignment.profile.maximum_evaluations,
        )
        assert_bounded_convergence(witness)


def test_direct_process_mvr_and_downstream_targeting_succeed_end_to_end() -> None:
    problem = PinchProblem("process_mvr.json", project_name="Process MVR E2E")
    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        n_stages=2,
        liquid_injection=False,
        mvr_stage_t_lift=10.0,
        compressor_efficiency=0.7,
        motor_efficiency=0.95,
    )

    assert problem.components.inventory[component.id] is component
    assert component.active is True
    assert component.original_streams
    source = component.original_streams[0]
    replacement_streams = component.replacement_streams
    assert len(replacement_streams) >= 2
    assert all(stream.is_active for stream in replacement_streams)
    assert all(
        math.isfinite(float(stream.supply_pressure))
        and float(stream.supply_pressure) > float(source.supply_pressure)
        for stream in replacement_streams
    )

    stages = [
        stage
        for period_stages in component.stage_results_by_period.values()
        for stage in period_stages
    ]
    assert len(stages) == 2
    assert [stage.stage_index for stage in stages] == [1, 2]
    assert all(
        math.isfinite(value) and value > 0.0
        for stage in stages
        for value in (stage.p_in, stage.p_out, stage.heat_flow, stage.work)
    )
    assert all(stage.p_out > stage.p_in for stage in stages)
    assert all(
        not {"engine", "model", "state"}.intersection(vars(stage)) for stage in stages
    )
    copied_stages = deepcopy(stages)
    assert copied_stages is not stages
    for copied, original in zip(copied_stages, stages, strict=True):
        assert copied is not original
        assert copied.work == original.work
        assert copied.heat_flow == original.heat_flow
        assert copied.th_curve is not original.th_curve
        assert copied.linearised_profile is not original.linearised_profile
        assert (copied.th_curve == original.th_curve).all()
        assert (copied.linearised_profile == original.linearised_profile).all()

    problem.target.direct_heat_integration()
    results = problem.results
    assert results is not None and results.targets
    for target in results.targets:
        assert all(
            math.isfinite(float(getattr(target, field).value))
            for field in ("Qh", "Qc", "Qr")
        )
    json.loads(results.model_dump_json())
    json.loads(json.dumps(problem.to_problem_json()))
