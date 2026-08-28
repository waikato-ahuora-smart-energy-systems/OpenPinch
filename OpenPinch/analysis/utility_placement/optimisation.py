"""Solver-neutral optimization adaptation for utility placement."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from ...contracts.utility_placement import (
    CandidateDiagnostic,
    PlacementTermination,
    UtilityPlacementOptions,
)
from ...optimisation.errors import OptimisationError
from ...optimisation.models import (
    OptimisationMethod,
    OptimisationOptions,
    OptimisationProblem,
    OptimisationResult,
)
from ...optimisation.service import run_multistart_minimisation
from .errors import NoFeasiblePlacementError, PlacementOptimisationError
from .evaluation import PlacementEvaluation, PlacementEvaluationSession

OptimisationRunner = Callable[..., OptimisationResult]


class PlacementOptimisationOutcome(BaseModel):
    """Canonical feasible evaluations and backend-independent termination."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluations: tuple[PlacementEvaluation, ...]
    termination: PlacementTermination
    diagnostics: tuple[CandidateDiagnostic, ...] = ()


def map_optimisation_options(
    options: UtilityPlacementOptions,
) -> tuple[OptimisationMethod, OptimisationOptions]:
    """Map frozen specialist options onto the existing optimizer contract."""
    return (
        OptimisationMethod(options.method.value),
        OptimisationOptions(
            n_runs=options.run_count,
            maxiter=options.iteration_limit,
            seed=options.seed,
            maxfun=options.evaluation_limit,
            cluster_tol=options.cluster_tolerance,
            max_minima=options.candidate_limit,
            local_method=options.local_method,
            backend_options=options.backend_options,
        ),
    )


def coordinate_optimisation(
    *,
    session: PlacementEvaluationSession,
    runner: OptimisationRunner = run_multistart_minimisation,
) -> PlacementOptimisationOutcome:
    """Invoke one existing optimizer, deduplicate, and fully replay in the parent."""
    method, options = map_optimisation_options(session.request.options)
    model = session.model
    problem = OptimisationProblem(
        objective=session.objective,
        bounds=tuple(
            (coordinate.bounds.lower, coordinate.bounds.upper)
            for coordinate in model.coordinates
        ),
        initial_points=model.initial_points,
    )
    try:
        backend_result = runner(problem, method=method, options=options)
    except OptimisationError as exc:
        raise PlacementOptimisationError(
            code="optimizer_failure",
            message="Utility-placement optimization failed.",
            method=method.value,
            seed=session.request.options.seed,
            details=(("reason", str(exc)),),
        ) from exc
    except Exception as exc:
        raise PlacementOptimisationError(
            code="optimizer_adapter_failure",
            message="Utility-placement optimizer adapter failed.",
            method=method.value,
            seed=session.request.options.seed,
            details=(("reason", str(exc)),),
        ) from exc

    points = tuple(model.initial_points) + tuple(
        candidate.point for candidate in backend_result.candidates
    )
    unique_points = tuple(dict.fromkeys(points))
    evaluations: list[PlacementEvaluation] = []
    diagnostics: list[CandidateDiagnostic] = []
    for point in unique_points:
        canonical = PlacementEvaluationSession(
            request=session.request.model_copy(
                update={
                    "options": session.request.options.model_copy(
                        update={
                            "evaluation_limit": max(
                                session.request.options.evaluation_limit, 1
                            )
                        }
                    )
                }
            ),
            context=session.context,
            model=session.model,
            allocation_adapter=session.allocation_adapter,
        ).evaluate(point)
        if canonical.feasible:
            evaluations.append(canonical)
        elif len(diagnostics) < 10:
            diagnostics.extend(canonical.diagnostics[: 10 - len(diagnostics)])

    evaluations.sort(
        key=lambda item: (
            item.physical_objective
            if item.physical_objective is not None
            else float("inf"),
            item.coordinates,
        )
    )
    evaluations = evaluations[: session.request.options.candidate_limit]
    if not evaluations:
        raise NoFeasiblePlacementError(
            code="no_feasible_placement",
            message="Optimization produced no placement feasible in every period.",
            method=method.value,
            seed=session.request.options.seed,
            details=(("candidate_count", len(unique_points)),),
        )

    termination = PlacementTermination(
        method=method.value,
        seed=session.request.options.seed,
        status="complete",
        code="feasible_candidates",
        message="Optimization completed with canonically replayed feasible placements.",
        iterations=None,
        evaluations=session.evaluation_count,
        candidate_count=len(unique_points),
        feasible_candidate_count=len(evaluations),
        iteration_limit=session.request.options.iteration_limit,
        evaluation_limit=session.request.options.evaluation_limit,
    )
    return PlacementOptimisationOutcome(
        evaluations=tuple(evaluations),
        termination=termination,
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "PlacementOptimisationOutcome",
    "coordinate_optimisation",
    "map_optimisation_options",
]
