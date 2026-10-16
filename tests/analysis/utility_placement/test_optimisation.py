"""Utility-placement optimization coordination tests."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pytest

import OpenPinch.optimisation.execution as execution
from OpenPinch.analysis.utility_placement.evaluation import (
    PlacementEvaluation,
    PlacementEvaluationSession,
    polish_budget_reserve,
)
from OpenPinch.analysis.utility_placement.optimisation import (
    _optimizer_point_to_physical,
    _physical_point_to_optimizer,
    coordinate_optimisation,
)
from OpenPinch.contracts.utility_placement import UtilityPlacementOptimisationMethod
from OpenPinch.domain import progress
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

    # Search tasks split the limit less the polish reserve (2 of 10).
    shares = [session.task_session(index, 3).evaluation_budget for index in range(3)]
    assert shares == [3, 3, 2]

    task = session.task_session(0, 1)
    for point in session.model.initial_points:
        task.evaluate(point)
    session.merge_task_reports([task.task_report()])

    assert session.evaluation_count == task.evaluation_count > 0
    remaining = 10 - session.evaluation_count
    shares = [
        session.task_session(index, 4, phase="polish").evaluation_budget
        for index in range(4)
    ]
    assert sum(shares) == remaining


def test_polish_keeps_its_reserve_after_search_spends_every_share() -> None:
    session = _session_with(evaluation_limit=10)
    assert polish_budget_reserve(10) == 2

    # Charge the search batch with its full shares, as runs that hit their
    # maxfun do.
    search = [session.task_session(index, 2) for index in range(2)]
    reports = [
        task.task_report().model_copy(
            update={"evaluation_count": task.evaluation_budget}
        )
        for task in search
    ]
    session.merge_task_reports(reports)

    assert session.evaluation_count == 8
    assert not session.limit_reached
    polish = [session.task_session(index, 3, phase="polish") for index in range(3)]
    assert [task.evaluation_budget for task in polish] == [1, 1, 0]


def test_polish_also_gets_what_search_left_unspent() -> None:
    session = _session_with(evaluation_limit=10)
    search = [session.task_session(index, 2) for index in range(2)]
    # Both runs stop early, spending 1 of their 4 each.
    reports = [
        task.task_report().model_copy(update={"evaluation_count": 1}) for task in search
    ]
    session.merge_task_reports(reports)

    polish = [session.task_session(index, 2, phase="polish") for index in range(2)]
    assert sum(task.evaluation_budget for task in polish) == 8


def test_task_session_rejects_unknown_phase() -> None:
    session = _session_with(evaluation_limit=10)
    with pytest.raises(ValueError, match="phase"):
        session.task_session(0, 1, phase="refine")


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


class _SearchThenPolishRunner:
    """Search from each start, then polish the first, in this process."""

    def __call__(self, problem, *, method, options):
        objective = problem.objective
        search = objective.for_task(0, 1, phase="search")
        values = [search(point) for point in problem.initial_points]
        objective.merge_task_reports([search.task_report()])
        polish = objective.for_task(0, 1, phase="polish")
        polish(problem.initial_points[0])
        objective.merge_task_reports([polish.task_report()])
        return OptimisationResult(
            method=OptimisationMethod(method),
            candidates=tuple(
                OptimisationCandidate(objective=value, point=tuple(point))
                for value, point in zip(values, problem.initial_points, strict=True)
            ),
        )


def test_coordinator_reports_search_polish_and_replay_progress() -> None:
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    events = []

    with progress.reporting(events.append, min_interval=0):
        outcome = coordinate_optimisation(
            session=session, runner=_SearchThenPolishRunner()
        )

    starts = len(model.initial_points)
    stages = [event.stage for event in events]
    # Each batch reports again once its evaluations merge into the session.
    assert stages == (["search"] * (starts + 1) + ["polish"] * 2 + ["replay"] * starts)
    assert all(event.analysis == "utility_placement" for event in events)
    search = events[: starts + 1]
    assert [event.done for event in search] == [*range(1, starts + 1), starts]
    assert {event.total for event in search} == {request.options.evaluation_limit}
    assert {event.unit for event in search} == {"evaluations"}
    best = [event.best_objective for event in search]
    assert best == sorted(best, reverse=True)
    assert [event.done for event in events[starts + 1 : starts + 3]] == [
        starts + 1,
        starts + 1,
    ]
    replay = events[starts + 3 :]
    assert [(event.done, event.total) for event in replay] == [
        (index, starts) for index in range(1, starts + 1)
    ]
    assert replay[-1].best_objective == outcome.evaluations[0].scalar_objective


class _WorkerProcessRunner:
    """Search on two task copies as worker processes would, then merge."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def __call__(self, problem, *, method, options):
        objective = problem.objective
        tasks = [
            pickle.loads(pickle.dumps(objective.for_task(index, 2)))
            for index in range(2)
        ]
        values = []
        with self.monkeypatch.context() as patch:
            # Reports from another process are not passed on.
            patch.setattr(progress.os, "getpid", lambda: -1)
            for task, point in zip(tasks, problem.initial_points, strict=False):
                values.append(task(point))
        objective.merge_task_reports([task.task_report() for task in tasks])
        return OptimisationResult(
            method=OptimisationMethod(method),
            candidates=tuple(
                OptimisationCandidate(objective=value, point=tuple(point))
                for value, point in zip(values, problem.initial_points, strict=False)
            ),
        )


def test_coordinator_reports_a_worker_batch_when_its_reports_merge(monkeypatch):
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    events = []

    with progress.reporting(events.append, min_interval=0):
        coordinate_optimisation(
            session=session, runner=_WorkerProcessRunner(monkeypatch)
        )

    search = [event for event in events if event.stage == "search"]
    assert [event.done for event in search] == [2]
    replay = [event for event in events if event.stage == "replay"]
    assert search[0].best_objective == min(
        session.evaluate(point).scalar_objective for point in model.initial_points[:2]
    )
    assert replay
