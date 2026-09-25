"""Pure contracts for the HPR/MVR end-to-end benchmark harness."""

from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from tests.e2e.cases import standard_problem_paths
from tests.e2e.hpr_benchmark import (
    HPR_PROFILES,
    HPROutcomeKind,
    ProblemStateSnapshot,
    SearchObservation,
    assert_atomic_outcome,
    assert_bounded_convergence,
    assert_typed_failure_contract,
    build_assignments,
    build_convergence_witness,
    classify_hpr_outcome,
    observe_hpr_search,
    profile_search_observations,
)


def test_standard_problem_corpus_is_sorted_complete_and_unique() -> None:
    paths = standard_problem_paths()

    assert len(paths) == 54
    assert len(paths) == len(set(paths))
    assert [path.name for path in paths] == sorted(path.name for path in paths)
    assert all(path.is_file() and path.match("p_*.json") for path in paths)


def test_six_profile_rotation_assigns_every_standard_problem_once() -> None:
    paths = standard_problem_paths()
    assignments = build_assignments(paths)

    assert len(HPR_PROFILES) == 6
    assert tuple(assignment.path for assignment in assignments) == paths
    assert len({assignment.parameter_id for assignment in assignments}) == 54
    assert Counter(assignment.profile.profile_id for assignment in assignments) == {
        profile.profile_id: 9 for profile in HPR_PROFILES
    }


def test_profile_search_limits_expand_only_mvr_convergence_sentinels() -> None:
    profiles = {profile.profile_id: profile for profile in HPR_PROFILES}

    for profile in HPR_PROFILES:
        assert profile.search_limits(sentinel=False) == (
            profile.maximum_evaluations,
            profile.maximum_search_observations,
        )

    direct_mvr = profiles["direct-optimized-vc-mvr-heat-pump"]
    utility_mvr = profiles["utility-optimized-vc-mvr-heat-pump"]
    assert direct_mvr.search_limits(sentinel=True) == (48, 96)
    assert utility_mvr.search_limits(sentinel=True) == (48, 96)

    non_mvr = profiles["direct-cascade-vc-heat-pump"]
    assert non_mvr.search_limits(sentinel=True) == (
        non_mvr.maximum_evaluations,
        non_mvr.maximum_search_observations,
    )


@seed(20260715)
@given(size=st.integers(min_value=1, max_value=240))
def test_profile_assignment_is_complete_repeatable_and_in_range(size: int) -> None:
    paths = tuple(Path(f"p_{index:04d}.json") for index in range(size))

    first = build_assignments(paths)
    second = build_assignments(tuple(reversed(paths)))

    assert first == second
    assert len(first) == size
    assert {assignment.path for assignment in first} == set(paths)
    assert all(assignment.profile in HPR_PROFILES for assignment in first)


def test_outcome_classification_has_only_three_public_states() -> None:
    target = object()
    error = _targeting_error()

    assert classify_hpr_outcome(target=target).kind is HPROutcomeKind.SOLVED
    assert classify_hpr_outcome(target=None).kind is HPROutcomeKind.NO_OP
    assert (
        classify_hpr_outcome(target=None, error=error).kind
        is HPROutcomeKind.TYPED_FAILURE
    )
    with pytest.raises(ValueError, match="both a target and an error"):
        classify_hpr_outcome(target=target, error=error)
    with pytest.raises(TypeError, match="HPRTargetingError"):
        classify_hpr_outcome(target=None, error=ValueError("untyped"))


def test_outcome_state_transitions_are_atomic_or_commit_once() -> None:
    before = ProblemStateSnapshot(results_json='{"targets":[]}', target_count=0)
    unchanged = ProblemStateSnapshot(results_json='{"targets":[]}', target_count=0)
    committed = ProblemStateSnapshot(results_json='{"targets":[{}]}', target_count=1)

    assert_atomic_outcome(
        before,
        unchanged,
        classify_hpr_outcome(target=None),
    )
    assert_atomic_outcome(
        before,
        unchanged,
        classify_hpr_outcome(target=None, error=_targeting_error()),
    )
    assert_atomic_outcome(
        before,
        committed,
        classify_hpr_outcome(target=object()),
    )

    with pytest.raises(AssertionError, match="leave public state unchanged"):
        assert_atomic_outcome(
            before,
            committed,
            classify_hpr_outcome(target=None, error=_targeting_error()),
        )
    with pytest.raises(AssertionError, match="exactly one target"):
        assert_atomic_outcome(
            before,
            ProblemStateSnapshot(results_json='{"targets":[{},{}]}', target_count=2),
            classify_hpr_outcome(target=object()),
        )


def test_typed_failure_contract_is_bounded_copyable_and_serializable() -> None:
    assert_typed_failure_contract(_targeting_error(), maximum_evaluations=1)

    over_budget = _targeting_error(evaluated_count=18)
    with pytest.raises(AssertionError, match="evaluation evidence exceeded"):
        assert_typed_failure_contract(over_budget, maximum_evaluations=1)


def test_convergence_deduplicates_points_and_selects_best_observed() -> None:
    observations = [
        SearchObservation(point=(0.0,), success=True, objective=12.0),
        SearchObservation(point=(0.0,), success=True, objective=9.0),
        SearchObservation(point=(1.0,), success=False, objective=None),
        SearchObservation(point=(2.0,), success=True, objective=8.0),
        SearchObservation(point=(3.0,), success=True, objective=10.0),
    ]

    witness = build_convergence_witness(
        observations,
        selected_objective=8.0,
        maximum_evaluations=4,
    )

    assert witness.distinct_evaluations == 4
    assert witness.viable_objectives == (12.0, 8.0, 10.0)
    assert witness.incumbent_objectives == (12.0, 8.0, 8.0)
    assert witness.best_observed_objective == 8.0
    assert_bounded_convergence(witness)


@seed(20260715)
@given(
    objectives=st.lists(
        st.floats(
            min_value=-1e9,
            max_value=1e9,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=30,
    )
)
def test_incumbent_sequence_matches_an_independent_prefix_minimum(
    objectives: list[float],
) -> None:
    observations = [
        SearchObservation(point=(float(index),), success=True, objective=value)
        for index, value in enumerate(objectives)
    ]
    witness = build_convergence_witness(
        observations,
        selected_objective=min(objectives),
        maximum_evaluations=len(objectives),
    )
    expected = tuple(min(objectives[: index + 1]) for index in range(len(objectives)))

    assert witness.incumbent_objectives == expected
    assert witness.best_observed_objective == min(objectives)


def test_convergence_rejects_threshold_and_selected_objective_misses() -> None:
    first = 100.0
    tolerance = 1e-8 * first
    boundary = build_convergence_witness(
        [
            SearchObservation(point=(0.0,), success=True, objective=first),
            SearchObservation(point=(1.0,), success=True, objective=first - tolerance),
        ],
        selected_objective=first - tolerance,
        maximum_evaluations=2,
    )
    with pytest.raises(AssertionError, match="materially improve"):
        assert_bounded_convergence(boundary)

    selected_miss = build_convergence_witness(
        [
            SearchObservation(point=(0.0,), success=True, objective=10.0),
            SearchObservation(point=(1.0,), success=True, objective=8.0),
        ],
        selected_objective=9.0,
        maximum_evaluations=2,
    )
    with pytest.raises(AssertionError, match="best observed"):
        assert_bounded_convergence(selected_miss)


def test_search_observer_delegates_once_and_restores_the_boundary(monkeypatch) -> None:
    import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter
    from OpenPinch.contracts.hpr import HPRBackendResult, HPREvaluationMode

    calls = []

    def fake_evaluator(**kwargs):
        calls.append(kwargs)
        return HPRBackendResult.failure(reason="bounded fake result")

    monkeypatch.setattr(adapter, "evaluate_hpr_candidate", fake_evaluator)
    before = adapter.evaluate_hpr_candidate
    with observe_hpr_search() as observations:
        result = adapter.evaluate_hpr_candidate(
            objective=lambda *args, **kwargs: None,
            point=(0.25, 0.75),
            args=object(),
            artifact_mode=HPREvaluationMode.SEARCH,
            debug=False,
        )
        assert result.success is False

    assert adapter.evaluate_hpr_candidate is before
    assert len(calls) == 1
    assert observations == [
        SearchObservation(
            point=(0.25, 0.75),
            success=False,
            objective=None,
            objective_name="<lambda>",
        )
    ]


def test_profile_trace_excludes_carnot_seed_objective() -> None:
    profile = HPR_PROFILES[0]
    observations = [
        SearchObservation(
            point=(0.0,),
            success=True,
            objective=1.0,
            objective_name="_compute_cascade_carnot_cycle_obj",
        ),
        SearchObservation(
            point=(1.0,),
            success=True,
            objective=2.0,
            objective_name=profile.objective_name,
        ),
    ]

    assert profile_search_observations(observations, profile) == (observations[1],)


def _targeting_error(*, evaluated_count: int = 0):
    from OpenPinch.contracts.hpr import (
        HPRFailureSummary,
        HPRSearchBudget,
        HPRTargetingError,
    )

    return HPRTargetingError(
        "bounded test failure",
        diagnostics=HPRFailureSummary(
            simulation_backend="coolprop",
            cycle="test",
            evaluated_count=evaluated_count,
            category_counts={},
            budget=HPRSearchBudget(maximum_iterations=1, maximum_evaluations=1),
            warm_start_evaluated=False,
            warm_start_viable=False,
        ),
    )
