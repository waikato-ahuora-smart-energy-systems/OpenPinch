"""Utility-placement optimization coordination tests."""

from __future__ import annotations

from dataclasses import dataclass

from OpenPinch.analysis.utility_placement.evaluation import PlacementEvaluationSession
from OpenPinch.analysis.utility_placement.optimisation import coordinate_optimisation
from OpenPinch.optimisation.models import (
    OptimisationCandidate,
    OptimisationMethod,
    OptimisationResult,
)
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
    assert len(outcome.evaluations) == len(set(model.initial_points))
    assert all(evaluation.feasible for evaluation in outcome.evaluations)
    assert outcome.termination.candidate_count == len(set(model.initial_points))
    assert outcome.termination.feasible_candidate_count == len(set(model.initial_points))
    assert outcome.termination.method == "dual_annealing"


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
                    OptimisationCandidate(objective=float(index), point=point)
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
    outcome = coordinate_optimisation(
        session=PlacementEvaluationSession(
            request=request,
            context=context,
            model=model,
            allocation_adapter=_Adapter(),
        ),
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
