"""Solver-neutral optimization adaptation for utility placement."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

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


def _coordinate_processing_order(model) -> list[int]:
    """Return coordinate indices with supply temperatures in declaration order.

    Supply ordering is chained down each side's templates as they were
    declared, which is also the order the verifier checks. Coordinates may be
    listed in another order, so they are not simply taken as listed.
    """
    declared = {
        template.key: index for index, template in enumerate(model.templates.all)
    }
    coordinates = list(model.coordinates)

    def sort_key(index: int):
        key = coordinates[index].coordinate
        is_supply = key.field.value == "supply_temperature"
        rank = declared.get(key.template_key, len(declared))
        # Supply temperatures first, by declaration; the rest keep their place.
        return (0, rank, index) if is_supply else (1, 0, index)

    return sorted(range(len(coordinates)), key=sort_key)


def _supply_family(model, key, kinds):
    side = key.template_key.side
    if model.request.uses_generated_pairs:
        return (side, kinds[key.template_key])
    return side


def _optimizer_point_to_physical(
    session: PlacementEvaluationSession,
    point,
) -> tuple[float, ...]:
    """Decode bounded optimizer coordinates with structural supply ordering."""
    model = session.model
    separation = model.envelope.minimum_separation.value
    kinds = {template.key: template.kind for template in model.templates.all}
    coordinates = list(model.coordinates)
    raw = list(point)
    if len(raw) != len(coordinates):
        raise ValueError("Optimizer point length does not match the coordinates.")
    values: dict[object, float] = {}
    previous_supply: dict[object, float] = {}
    for index in _coordinate_processing_order(model):
        coordinate = coordinates[index]
        fraction = min(max(float(raw[index]), 0.0), 1.0)
        key = coordinate.coordinate
        lower = coordinate.bounds.lower
        upper = coordinate.bounds.upper
        if key.field.value == "supply_temperature":
            family = _supply_family(model, key, kinds)
            previous = previous_supply.get(family)
            if previous is not None:
                if key.template_key.side.value == "hot":
                    upper = min(upper, previous - separation)
                elif not model.request.uses_generated_pairs:
                    lower = max(lower, previous + separation)
            previous_supply[family] = lower + fraction * max(upper - lower, 0.0)
            value = previous_supply[family]
        else:
            value = lower + fraction * (upper - lower)
        values[key] = value
    return tuple(values[item.coordinate] for item in coordinates)


def _physical_point_to_optimizer(
    session: PlacementEvaluationSession,
    point,
) -> tuple[float, ...]:
    """Encode physical coordinates into the bounded structural search space."""
    model = session.model
    separation = model.envelope.minimum_separation.value
    kinds = {template.key: template.kind for template in model.templates.all}
    coordinates = list(model.coordinates)
    raw = [float(value) for value in point]
    if len(raw) != len(coordinates):
        raise ValueError("Physical point length does not match the coordinates.")
    fractions = [0.0] * len(coordinates)
    previous_supply: dict[object, float] = {}
    for index in _coordinate_processing_order(model):
        coordinate = coordinates[index]
        value = raw[index]
        key = coordinate.coordinate
        lower = coordinate.bounds.lower
        upper = coordinate.bounds.upper
        if key.field.value == "supply_temperature":
            family = _supply_family(model, key, kinds)
            previous = previous_supply.get(family)
            if previous is not None:
                if key.template_key.side.value == "hot":
                    upper = min(upper, previous - separation)
                elif not model.request.uses_generated_pairs:
                    lower = max(lower, previous + separation)
            previous_supply[family] = value
        width = upper - lower
        fraction = (value - lower) / width if width > 0.0 else 0.0
        fractions[index] = min(max(fraction, 0.0), 1.0)
    return tuple(fractions)


def _bounded_session_objective(
    point,
    *,
    session: PlacementEvaluationSession,
) -> float:
    return session.objective(_optimizer_point_to_physical(session, point))


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
        objective=partial(_bounded_session_objective, session=session),
        bounds=((0.0, 1.0),) * len(model.coordinates),
        initial_points=tuple(
            _physical_point_to_optimizer(session, point)
            for point in model.initial_points
        ),
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
        _optimizer_point_to_physical(session, candidate.point)
        for candidate in backend_result.candidates
    )
    unique_points = tuple(dict.fromkeys(points))
    evaluations: list[PlacementEvaluation] = []
    diagnostics: list[CandidateDiagnostic] = []
    canonical_session = PlacementEvaluationSession(
        request=session.request.model_copy(
            update={
                "options": session.request.options.model_copy(
                    update={
                        "evaluation_limit": max(
                            session.request.options.evaluation_limit,
                            len(unique_points),
                        )
                    }
                )
            }
        ),
        context=session.context,
        model=session.model,
        allocation_adapter=session.allocation_adapter,
    )
    for point in unique_points:
        canonical = canonical_session.evaluate(point)
        if canonical.feasible:
            evaluations.append(canonical)
        elif len(diagnostics) < 10:
            diagnostics.extend(canonical.diagnostics[: 10 - len(diagnostics)])

    evaluations.sort(
        key=lambda item: (
            item.scalar_objective,
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
        evaluations=session.evaluation_count + canonical_session.evaluation_count,
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
