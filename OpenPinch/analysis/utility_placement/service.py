"""Detached end-to-end utility-placement optimization facade."""

from __future__ import annotations

from OpenPinch.contracts.utility_placement import (
    QuantityValue,
    TemplateBlueprintSet,
    UtilityPlacementCandidate,
    UtilityPlacementRequest,
    UtilityPlacementResult,
)
from OpenPinch.optimisation.service import run_multistart_minimisation

from .allocation import PlacementAllocationAdapter
from .codec import build_utility_placement_model, decode_placement
from .context import UtilityPlacementContext
from .evaluation import PlacementEvaluation, PlacementEvaluationSession
from .optimisation import OptimisationRunner, coordinate_optimisation


def _candidate(
    evaluation: PlacementEvaluation,
    *,
    request: UtilityPlacementRequest,
    model,
) -> UtilityPlacementCandidate:
    placement = decode_placement(model, evaluation.coordinates)
    assert evaluation.physical_objective is not None
    return UtilityPlacementCandidate(
        coordinates=evaluation.coordinates,
        hot_levels=placement.hot,
        cold_levels=placement.cold,
        period_results=evaluation.period_results,
        aggregate_objective=QuantityValue(
            value=evaluation.physical_objective,
            unit=request.units.entropy,
        ),
        thermodynamic_total=(
            QuantityValue(
                value=evaluation.thermodynamic_total,
                unit=request.units.entropy,
            )
            if evaluation.thermodynamic_total is not None
            else None
        ),
        fallback_penalty=QuantityValue(
            value=evaluation.fallback_penalty,
            unit="dimensionless",
        ),
        diagnostics=evaluation.diagnostics,
    )


def optimise_utility_placement(
    *,
    request: UtilityPlacementRequest,
    blueprints: TemplateBlueprintSet,
    context: UtilityPlacementContext,
    allocation_adapter: PlacementAllocationAdapter | None = None,
    runner: OptimisationRunner = run_multistart_minimisation,
) -> UtilityPlacementResult:
    """Build, optimize, canonically replay, and assemble a frozen result."""
    model = build_utility_placement_model(request, blueprints, context.envelope)
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=allocation_adapter,
    )
    outcome = coordinate_optimisation(session=session, runner=runner)
    candidates = tuple(
        _candidate(evaluation, request=request, model=model)
        for evaluation in outcome.evaluations
    )
    return UtilityPlacementResult(
        request=request,
        scope=context.scope,
        base_target_id=context.base_target_id,
        period_ids=tuple(period.period_id for period in context.periods),
        period_weights=tuple(period.weight for period in context.periods),
        units=request.units,
        best=candidates[0],
        alternatives=candidates[1:],
        termination=outcome.termination,
        diagnostics=outcome.diagnostics,
    )


__all__ = ["optimise_utility_placement"]
