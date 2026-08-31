"""Utility-placement optimization coordination tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from OpenPinch.analysis.utility_placement.evaluation import (
    PlacementEvaluation,
    PlacementEvaluationSession,
)
from OpenPinch.analysis.utility_placement.optimisation import (
    _optimizer_point_to_physical,
    _physical_point_to_optimizer,
    coordinate_optimisation,
)
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
