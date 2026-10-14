"""Utility-placement optimization coordination tests."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pytest

import OpenPinch.optimisation.execution as execution
from OpenPinch.analysis.utility_placement.evaluation import (
    PlacementEvaluation,
    PlacementEvaluationSession,
)
from OpenPinch.analysis.utility_placement.optimisation import (
    _optimizer_point_to_physical,
    _physical_point_to_optimizer,
    coordinate_optimisation,
)
from OpenPinch.contracts.utility_placement import UtilityPlacementOptimisationMethod
from OpenPinch.optimisation.models import (
    OptimisationCandidate,
    OptimisationMethod,
    OptimisationResult,
)
from tests.analysis.utility_placement.test_codec import _model
from tests.analysis.utility_placement.test_evaluation import _Adapter, _case


@dataclass
class _Runner:
    calls: int = 0

    def __call__(self, problem, *, method, options):
        self.calls += 1
        point = tuple(problem.initial_points[0])
        objective = problem.objective(point)
        return OptimisationResult(
            method=OptimisationMethod(method),
            candidates=(
                OptimisationCandidate(objective=objective, point=point),
                OptimisationCandidate(objective=objective, point=point),
            ),
        )


def test_coordinator_calls_optimizer_once_deduplicates_and_replays_parent() -> None:
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    runner = _Runner()

    outcome = coordinate_optimisation(session=session, runner=runner)

    assert runner.calls == 1
    assert len(outcome.evaluations) == min(
        request.options.candidate_limit,
        len(set(model.initial_points)),
    )
    assert all(evaluation.feasible for evaluation in outcome.evaluations)
    assert outcome.termination.candidate_count == len(set(model.initial_points))
    assert outcome.termination.feasible_candidate_count == len(outcome.evaluations)
    assert outcome.termination.method == "cmaes"
    assert outcome.termination.evaluations >= outcome.termination.candidate_count


def test_optimizer_coordinates_are_bounded_and_preserve_physical_starts() -> None:
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )

    for physical in model.initial_points:
        bounded = _physical_point_to_optimizer(session, physical)
        assert all(0.0 <= value <= 1.0 for value in bounded)
        assert _optimizer_point_to_physical(session, bounded) == pytest.approx(physical)


def test_optimizer_transform_preserves_cross_kind_supply_interleaving() -> None:
    _, context, _ = _case()
    model = _model(isothermal_count=2, sensible_count=1)
    session = PlacementEvaluationSession(
        request=model.request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    physical = (500.0, 300.0, 350.0, 1.0)

    bounded = _physical_point_to_optimizer(session, physical)

    assert _optimizer_point_to_physical(session, bounded) == pytest.approx(physical)


def test_coordinator_ranks_by_the_same_penalized_scalar_as_the_backend(
    monkeypatch,
) -> None:
    request, context, model = _case()
    first, second = model.initial_points[:2]

    def evaluate(self, point):
        coordinates = tuple(point)
        if coordinates == first:
            return PlacementEvaluation(
                coordinates=coordinates,
                feasible=True,
                scalar_objective=0.8,
                physical_objective=1.0,
                fallback_penalty=1.0,
            )
        if coordinates == second:
            return PlacementEvaluation(
                coordinates=coordinates,
                feasible=True,
                scalar_objective=0.1,
                physical_objective=2.0,
                fallback_penalty=0.0,
            )
        return PlacementEvaluation(
            coordinates=coordinates,
            feasible=True,
            scalar_objective=0.2,
            physical_objective=2.0,
            fallback_penalty=0.0,
        )

    monkeypatch.setattr(PlacementEvaluationSession, "evaluate", evaluate)

    class RankedRunner:
        def __call__(self, problem, *, method, options):
            session = ranking_session
            return OptimisationResult(
                method=OptimisationMethod(method),
                candidates=(
                    OptimisationCandidate(
                        objective=0.1,
                        point=_physical_point_to_optimizer(session, second),
                    ),
                    OptimisationCandidate(
                        objective=0.8,
                        point=_physical_point_to_optimizer(session, first),
                    ),
                ),
            )

    ranking_session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    outcome = coordinate_optimisation(
        session=ranking_session,
        runner=RankedRunner(),
    )

    assert outcome.evaluations[0].coordinates == second
    assert outcome.evaluations[0].scalar_objective == 0.1


def test_coordinator_raises_typed_exhaustion_when_no_candidate_is_feasible() -> None:
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )

    class BadRunner:
        def __call__(self, problem, *, method, options):
            bad = tuple(upper + 10.0 for _, upper in problem.bounds)
            return OptimisationResult(
                method=OptimisationMethod(method),
                candidates=(OptimisationCandidate(objective=1.5, point=bad),),
            )

    # Initial points are intentionally part of the candidate union, so make their
    # canonical replay infeasible as well.
    session.allocation_adapter = type(
        "ShortAdapter",
        (),
        {
            "allocate": lambda self, period, placement: __import__(
                "OpenPinch.analysis.utility_placement.allocation",
                fromlist=["AllocationAdapterResult"],
            ).AllocationAdapterResult(hot_duties=(0.0, 0.0), cold_duties=(0.0, 0.0))
        },
    )()

    from OpenPinch.analysis.utility_placement.errors import NoFeasiblePlacementError

    try:
        coordinate_optimisation(session=session, runner=BadRunner())
    except NoFeasiblePlacementError as exc:
        assert exc.code == "no_feasible_placement"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("typed exhaustion was not raised")


def test_coordinator_matches_explicit_structured_grid_oracle() -> None:
    request, context, model = _case()
    start = model.initial_points[0]
    alternative = list(start)
    alternative[1] = max(model.coordinates[1].bounds.lower, alternative[1] - 10.0)
    alternative[3] = min(model.coordinates[3].bounds.upper, alternative[3] + 10.0)
    grid = (*model.initial_points, tuple(alternative))

    class GridRunner:
        def __call__(self, problem, *, method, options):
            return OptimisationResult(
                method=OptimisationMethod(method),
                candidates=tuple(
                    OptimisationCandidate(
                        objective=float(index),
                        point=_physical_point_to_optimizer(grid_session, point),
                    )
                    for index, point in enumerate(reversed(grid))
                ),
            )

    def fresh(point):
        return PlacementEvaluationSession(
            request=request,
            context=context,
            model=model,
            allocation_adapter=_Adapter(),
        ).evaluate(point)

    oracle = min(
        (fresh(point) for point in grid),
        key=lambda result: (result.physical_objective, result.coordinates),
    )
    grid_session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    outcome = coordinate_optimisation(
        session=grid_session,
        runner=GridRunner(),
    )
    assert outcome.evaluations[0].coordinates == oracle.coordinates
    assert outcome.evaluations[0].physical_objective == oracle.physical_objective


def test_tiny_real_dual_annealing_regression() -> None:
    request, context, model = _case()
    request = request.model_copy(
        update={
            "options": request.options.model_copy(
                update={
                    "iteration_limit": 1,
                    "evaluation_limit": 100,
                    "run_count": 1,
                    "candidate_limit": 2,
                }
            )
        }
    )
    outcome = coordinate_optimisation(
        session=PlacementEvaluationSession(
            request=request,
            context=context,
            model=model,
            allocation_adapter=_Adapter(),
        )
    )
    assert outcome.evaluations
    assert all(evaluation.feasible for evaluation in outcome.evaluations)


def _session_with(**options) -> PlacementEvaluationSession:
    request, context, model = _case()
    request = request.model_copy(
        update={"options": request.options.model_copy(update=options)}
    )
    return PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )


def test_task_sessions_split_the_remaining_budget_deterministically() -> None:
    session = _session_with(evaluation_limit=10)

    shares = [session.task_session(index, 3).evaluation_budget for index in range(3)]
    assert shares == [4, 3, 3]

    task = session.task_session(0, 1)
    for point in session.model.initial_points:
        task.evaluate(point)
    session.merge_task_reports([task.task_report()])

    assert session.evaluation_count == task.evaluation_count > 0
    remaining = 10 - session.evaluation_count
    shares = [session.task_session(index, 4).evaluation_budget for index in range(4)]
    assert sum(shares) == remaining


def test_task_session_budget_survives_pickling_and_reports_exhaustion() -> None:
    session = _session_with(evaluation_limit=10)
    task = pickle.loads(pickle.dumps(session.task_session(0, 10)))
    points = list(dict.fromkeys(session.model.initial_points))
    assert len(points) >= 2

    first = task.evaluate(points[0])
    second = task.evaluate(points[1])

    assert task.evaluation_budget == 1
    assert first.feasible
    assert not second.feasible
    assert second.diagnostics[0].code == "evaluation_budget_exhausted"
    session.merge_task_reports([task.task_report()])
    assert session.budget_exhausted
    assert session.evaluation_count == 1
    assert dict(session.diagnostic_counts)["evaluation_budget_exhausted"] == 1


def _run_dual_annealing(monkeypatch, *, pooled: bool):
    if not pooled:
        monkeypatch.setattr(execution, "_is_picklable", lambda obj: False)
    session = _session_with(
        method=UtilityPlacementOptimisationMethod.DUAL_ANNEALING,
        iteration_limit=50,
        evaluation_limit=12,
        run_count=3,
        candidate_limit=3,
    )
    outcome = coordinate_optimisation(session=session)
    monkeypatch.undo()
    return session, outcome


def test_evaluation_limit_is_global_across_runs_and_workers(monkeypatch) -> None:
    session, outcome = _run_dual_annealing(monkeypatch, pooled=True)
    limit = session.request.options.evaluation_limit

    # Optimiser runs and polish tasks share one budget; only the parent's
    # canonical replay of the final candidates comes on top.
    assert session.evaluation_count <= limit
    assert outcome.termination.evaluations <= (
        limit + outcome.termination.candidate_count
    )
    assert outcome.termination.status == "evaluation_limit_reached"
    assert outcome.termination.evaluation_limit == limit


def test_pooled_and_serial_runs_spend_the_same_budget(monkeypatch) -> None:
    pooled_session, pooled = _run_dual_annealing(monkeypatch, pooled=True)
    serial_session, serial = _run_dual_annealing(monkeypatch, pooled=False)

    assert pooled_session.evaluation_count == serial_session.evaluation_count
    assert pooled.termination == serial.termination
    assert [item.coordinates for item in pooled.evaluations] == [
        item.coordinates for item in serial.evaluations
    ]


def test_generous_budget_reports_complete() -> None:
    session = _session_with(
        method=UtilityPlacementOptimisationMethod.DUAL_ANNEALING,
        iteration_limit=1,
        evaluation_limit=10_000,
        run_count=1,
        candidate_limit=2,
    )

    outcome = coordinate_optimisation(session=session)

    assert outcome.termination.status == "complete"
    assert not session.budget_exhausted
