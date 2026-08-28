"""Specialist public surface for the utility-placement pure model."""

from .allocation import (
    AllocatedUtilityLevel,
    AllocationAdapterResult,
    ExistingTargetingAllocationAdapter,
    PlacementPeriodAllocation,
    UtilityAllocationSlice,
    allocate_placement_period,
)
from .bounds import build_initial_values, derive_effective_templates
from .codec import (
    build_decision_coordinates,
    build_utility_placement_model,
    decode_placement,
    encode_placement,
    verify_candidate,
    verify_placement,
)
from .context import (
    PlacementPeriodInput,
    PlacementTargetSnapshot,
    ProcessEntropySlice,
    UtilityPlacementContext,
    build_utility_placement_context,
)
from .evaluation import PlacementEvaluation, PlacementEvaluationSession
from .normalization import (
    convert_placement_value,
    normalize_utility_placement_request,
    prepare_template_blueprints,
)
from .optimisation import (
    PlacementOptimisationOutcome,
    coordinate_optimisation,
    map_optimisation_options,
)
from .penalties import (
    aggregate_weighted_objective,
    feasible_objective_scalar,
    infeasible_objective_scalar,
)
from .service import optimise_utility_placement
from .thermodynamics import evaluate_thermodynamic_cost, stream_entropy_change

__all__ = [
    "AllocatedUtilityLevel",
    "AllocationAdapterResult",
    "ExistingTargetingAllocationAdapter",
    "PlacementPeriodAllocation",
    "PlacementEvaluation",
    "PlacementEvaluationSession",
    "PlacementOptimisationOutcome",
    "UtilityAllocationSlice",
    "allocate_placement_period",
    "aggregate_weighted_objective",
    "build_initial_values",
    "build_utility_placement_context",
    "build_decision_coordinates",
    "build_utility_placement_model",
    "convert_placement_value",
    "coordinate_optimisation",
    "derive_effective_templates",
    "decode_placement",
    "evaluate_thermodynamic_cost",
    "feasible_objective_scalar",
    "infeasible_objective_scalar",
    "encode_placement",
    "normalize_utility_placement_request",
    "optimise_utility_placement",
    "map_optimisation_options",
    "prepare_template_blueprints",
    "PlacementPeriodInput",
    "PlacementTargetSnapshot",
    "ProcessEntropySlice",
    "UtilityPlacementContext",
    "verify_candidate",
    "verify_placement",
    "stream_entropy_change",
]
