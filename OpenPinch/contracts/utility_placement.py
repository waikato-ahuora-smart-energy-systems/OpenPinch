"""Frozen contracts for utility-placement analysis."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenContract(BaseModel):
    """Base configuration shared by specialist public values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class UtilityPlacementOptimisationMethod(StrEnum):
    """Existing solver-neutral methods accepted by placement options."""

    DUAL_ANNEALING = "dual_annealing"
    CMA_ES = "cmaes"
    BAYESIAN = "bo"
    RBF = "rbf_surrogate"


class UtilityPlacementBaseTarget(StrEnum):
    """Targeting scopes available to the later application service."""

    AUTO = "auto"
    DIRECT = "direct"
    INDIRECT = "indirect"
    TOTAL_SITE = "total_site"


class UtilitySide(StrEnum):
    """Utility side relative to the process heat balance."""

    HOT = "hot"
    COLD = "cold"


class UtilityLevelKind(StrEnum):
    """Temperature behavior of a utility level."""

    ISOTHERMAL = "isothermal"
    SENSIBLE = "sensible"


class DecisionField(StrEnum):
    """Decision-coordinate field names."""

    SUPPLY_TEMPERATURE = "supply_temperature"
    TEMPERATURE_SPAN = "temperature_span"


def _finite_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return 0.0 if result == 0.0 else result


def _strict_int(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("value must be an integer")
    return value


class QuantityValue(_FrozenContract):
    """One finite magnitude with explicit unit metadata."""

    value: float
    unit: str

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> float:
        return _finite_float(value)  # type: ignore[arg-type]

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, value: str) -> str:
        unit = value.strip()
        if not unit:
            raise ValueError("unit must not be empty")
        return unit


class QuantityInterval(_FrozenContract):
    """Finite inclusive bounds in one explicit unit."""

    lower: float
    upper: float
    unit: str

    @field_validator("lower", "upper", mode="before")
    @classmethod
    def _validate_bound(cls, value: object) -> float:
        return _finite_float(value)  # type: ignore[arg-type]

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, value: str) -> str:
        unit = value.strip()
        if not unit:
            raise ValueError("unit must not be empty")
        return unit

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        return self


class PlacementTolerances(_FrozenContract):
    """Named numerical tolerances shared by all placement components."""

    absolute: float = 1e-6
    relative: float = 1e-9
    bounds: float = 1e-6
    coverage: float = 1e-6
    ordering: float = 1e-6

    @field_validator("absolute", "relative", "bounds", "coverage", "ordering")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        result = _finite_float(value)
        if result < 0.0:
            raise ValueError("tolerance must be non-negative")
        return result


class PlacementUnitSystem(_FrozenContract):
    """Canonical unit labels; conversion remains analysis-owned."""

    absolute_temperature: str = "degC"
    temperature_difference: str = "delta_degC"
    heat_flow: str = "kW"
    entropy: str = "kW/K"
    exergy: str = "kW"


class UtilityPlacementOptions(_FrozenContract):
    """Bounded optimizer-independent request options."""

    candidate_limit: int = 5
    iteration_limit: int = 500
    evaluation_limit: int = 5_000
    seed: int = 20260715
    method: UtilityPlacementOptimisationMethod = (
        UtilityPlacementOptimisationMethod.CMA_ES
    )
    run_count: int = 1
    cluster_tolerance: float = 0.01
    local_method: str = "SLSQP"
    backend_options: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    minimum_separation: QuantityValue = Field(
        default_factory=lambda: QuantityValue(value=1.0, unit="delta_degC")
    )
    minimum_sensible_span: QuantityValue = Field(
        default_factory=lambda: QuantityValue(value=0.01, unit="delta_degC")
    )
    default_isothermal_span: QuantityValue = Field(
        default_factory=lambda: QuantityValue(value=0.01, unit="delta_degC")
    )

    @field_validator(
        "candidate_limit",
        "iteration_limit",
        "evaluation_limit",
        "seed",
        "run_count",
        mode="before",
    )
    @classmethod
    def _validate_integer(cls, value: object) -> object:
        return _strict_int(value)

    @field_validator(
        "candidate_limit", "iteration_limit", "evaluation_limit", "run_count"
    )
    @classmethod
    def _validate_positive_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("limit must be positive")
        return value

    @field_validator("cluster_tolerance", mode="before")
    @classmethod
    def _validate_cluster_tolerance(cls, value: object) -> float:
        result = _finite_float(value)  # type: ignore[arg-type]
        if result < 0.0:
            raise ValueError("cluster_tolerance must be non-negative")
        return result

    @field_validator("local_method")
    @classmethod
    def _validate_local_method(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("local_method must not be empty")
        return normalized

    @field_validator("backend_options", mode="before")
    @classmethod
    def _validate_backend_options(cls, value: object):
        if value is None:
            return ()
        entries = tuple(value)  # type: ignore[arg-type]
        normalized: list[tuple[str, str | int | float | bool | None]] = []
        names: set[str] = set()
        for entry in entries:
            if not isinstance(entry, tuple | list) or len(entry) != 2:
                raise ValueError("backend options must be name/value pairs")
            raw_name, raw_value = entry
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("backend option name must not be empty")
            name = raw_name.strip()
            if name in names:
                raise ValueError("backend option names must be unique")
            names.add(name)
            if isinstance(raw_value, float):
                raw_value = _finite_float(raw_value)
            elif raw_value is not None and not isinstance(raw_value, str | int | bool):
                raise ValueError("backend option values must be JSON-safe scalars")
            normalized.append((name, raw_value))
        return tuple(sorted(normalized))

    @field_validator(
        "minimum_separation",
        "minimum_sensible_span",
        "default_isothermal_span",
    )
    @classmethod
    def _validate_positive_temperature_difference(
        cls, value: QuantityValue
    ) -> QuantityValue:
        if value.value <= 0.0:
            raise ValueError("temperature-difference option must be positive")
        return value


class UtilityLevelTemplate(_FrozenContract):
    """Caller-declared or generated utility-level specification."""

    name: str
    side: UtilitySide
    kind: UtilityLevelKind
    placement_rank: int = 0
    supply_bounds: QuantityInterval | None = None
    fixed_span: QuantityValue | None = None
    span_bounds: QuantityInterval | None = None
    fluid: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must not be empty")
        return name

    @field_validator("placement_rank", mode="before")
    @classmethod
    def _validate_rank_type(cls, value: object) -> object:
        return _strict_int(value)

    @field_validator("placement_rank")
    @classmethod
    def _validate_rank(cls, value: int) -> int:
        if value < 0:
            raise ValueError("placement_rank must be non-negative")
        return value


class TemplateKey(_FrozenContract):
    """Stable side/name identity for one utility template."""

    side: UtilitySide
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must not be empty")
        return name


class UtilityDutyLimit(_FrozenContract):
    """Canonical period-aware upper bounds for one named utility level."""

    name: str
    period_ids: tuple[str, ...]
    values: tuple[QuantityValue, ...]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must not be empty")
        return name

    @field_validator("period_ids")
    @classmethod
    def _validate_period_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(period_id.strip() for period_id in value)
        if not normalized or any(not period_id for period_id in normalized):
            raise ValueError("period_ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("period_ids must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        if len(self.period_ids) != len(self.values):
            raise ValueError("maximum-duty periods and values must align")
        if any(value.value < 0.0 for value in self.values):
            raise ValueError("maximum duties must be non-negative")
        if len({value.unit for value in self.values}) != 1:
            raise ValueError("maximum duties must use one canonical unit")
        return self

    def for_period(self, period_id: str) -> QuantityValue:
        """Return the limit for one selected period identity."""
        try:
            index = self.period_ids.index(period_id)
        except ValueError as exc:
            raise KeyError(period_id) from exc
        return self.values[index]


class UtilityTemplateBlueprint(_FrozenContract):
    """Canonical, identity-stable template before physical bounds exist."""

    key: TemplateKey
    kind: UtilityLevelKind
    placement_rank: int
    supply_bounds: QuantityInterval | None = None
    fixed_span: QuantityValue | None = None
    span_bounds: QuantityInterval | None = None
    fluid: str | None = None

    def as_template(self) -> UtilityLevelTemplate:
        """Return the equivalent normalized public template."""
        return UtilityLevelTemplate(
            name=self.key.name,
            side=self.key.side,
            kind=self.kind,
            placement_rank=self.placement_rank,
            supply_bounds=self.supply_bounds,
            fixed_span=self.fixed_span,
            span_bounds=self.span_bounds,
            fluid=self.fluid,
        )


class TemplateBlueprintSet(_FrozenContract):
    """Complete ordered hot/cold template blueprints."""

    hot: tuple[UtilityTemplateBlueprint, ...]
    cold: tuple[UtilityTemplateBlueprint, ...]

    @property
    def all(self) -> tuple[UtilityTemplateBlueprint, ...]:
        """Return every blueprint in side declaration order."""
        return self.hot + self.cold


class CoordinateKey(_FrozenContract):
    """Stable identity of one scalar decision coordinate."""

    template_key: TemplateKey
    field: DecisionField


class PhysicalCoordinateBound(_FrozenContract):
    """One period's physical interval and diagnostic provenance."""

    coordinate: CoordinateKey
    bounds: QuantityInterval
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        reason = value.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        return reason


class PlacementPeriodEnvelope(_FrozenContract):
    """Complete coordinate bounds for one weighted period."""

    period_id: str
    weight: float
    coordinate_bounds: tuple[PhysicalCoordinateBound, ...]
    residual_hot_duty: float = 0.0
    residual_cold_duty: float = 0.0

    @field_validator("period_id")
    @classmethod
    def _validate_period_id(cls, value: str) -> str:
        period_id = value.strip()
        if not period_id:
            raise ValueError("period_id must not be empty")
        return period_id

    @field_validator("weight", "residual_hot_duty", "residual_cold_duty", mode="before")
    @classmethod
    def _validate_weight(cls, value: object) -> float:
        result = _finite_float(value)  # type: ignore[arg-type]
        if result < 0.0:
            raise ValueError("period values must be non-negative")
        return result


class PlacementFeasibilityEnvelope(_FrozenContract):
    """Detached all-period physical feasibility supplied by Unit 2."""

    periods: tuple[PlacementPeriodEnvelope, ...]
    minimum_separation: QuantityValue
    approach_limits: tuple[tuple[str, str], ...] = ()
    scope: UtilityPlacementBaseTarget
    base_target_id: str
    units: PlacementUnitSystem = Field(default_factory=PlacementUnitSystem)

    @field_validator("base_target_id")
    @classmethod
    def _validate_target_id(cls, value: str) -> str:
        target_id = value.strip()
        if not target_id:
            raise ValueError("base_target_id must not be empty")
        return target_id

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if not self.periods:
            raise ValueError("periods must not be empty")
        identifiers = [period.period_id for period in self.periods]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("period identifiers must be unique")
        if not any(period.weight > 0.0 for period in self.periods):
            raise ValueError("at least one period weight must be positive")
        if self.minimum_separation.value <= 0.0:
            raise ValueError("minimum_separation must be positive")
        return self


class EffectiveUtilityTemplate(_FrozenContract):
    """Canonical template with intersected and ordered effective bounds."""

    key: TemplateKey
    kind: UtilityLevelKind
    placement_rank: int
    supply_bounds: QuantityInterval
    fixed_span: QuantityValue | None = None
    span_bounds: QuantityInterval | None = None
    fluid: str | None = None
    supply_period_bounds: tuple[PhysicalCoordinateBound, ...] = ()
    span_period_bounds: tuple[PhysicalCoordinateBound, ...] = ()
    caller_supply_bounds: QuantityInterval | None = None
    caller_span_bounds: QuantityInterval | None = None


class UtilityTemplateSet(_FrozenContract):
    """Complete effective hot and cold template families."""

    hot: tuple[EffectiveUtilityTemplate, ...]
    cold: tuple[EffectiveUtilityTemplate, ...]

    @property
    def all(self) -> tuple[EffectiveUtilityTemplate, ...]:
        """Return every effective template in side declaration order."""
        return self.hot + self.cold


class DecisionCoordinate(_FrozenContract):
    """One ordered scalar coordinate in the optimizer-facing vector schema."""

    index: int
    coordinate: CoordinateKey
    bounds: QuantityInterval


class DecodedUtilityLevel(_FrozenContract):
    """One decoded utility level with canonical temperatures."""

    template_key: TemplateKey
    kind: UtilityLevelKind
    placement_rank: int
    supply_temperature: QuantityValue
    target_temperature: QuantityValue
    temperature_span: QuantityValue


class DecodedPlacement(_FrozenContract):
    """Decoded hot/cold levels plus the exact source coordinate tuple."""

    hot: tuple[DecodedUtilityLevel, ...]
    cold: tuple[DecodedUtilityLevel, ...]
    coordinates: tuple[float, ...]

    @field_validator("coordinates")
    @classmethod
    def _validate_coordinates(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(_finite_float(item) for item in value)


class CandidateDiagnostic(_FrozenContract):
    """Machine-readable ordinary candidate constraint failure."""

    code: str
    constraint: str
    message: str
    side: UtilitySide | None = None
    template_key: TemplateKey | None = None
    period_id: str | None = None
    measured: QuantityValue | None = None
    limit: QuantityValue | None = None
    details: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    @field_validator("code", "constraint", "message")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("diagnostic text must not be empty")
        return normalized


class CandidateVerification(_FrozenContract):
    """Independent feasibility result for an ordinary candidate."""

    feasible: bool
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.feasible == bool(self.diagnostics):
            raise ValueError("feasible must be true exactly when diagnostics are empty")
        return self


class UtilityPlacementModel(_FrozenContract):
    """Complete immutable pure model consumed by the Unit 2 optimizer."""

    request: UtilityPlacementRequest
    envelope: PlacementFeasibilityEnvelope
    templates: UtilityTemplateSet
    coordinates: tuple[DecisionCoordinate, ...]
    initial_points: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def _validate_schema(self) -> Self:
        if self.request.uses_generated_pairs:
            expected_dimension = (
                self.request.isothermal_level_count
                + 2 * self.request.sensible_level_count
            )
        else:
            expected_dimension = (
                2 * self.request.isothermal_level_count
                + 4 * self.request.sensible_level_count
            )
        if len(self.coordinates) != expected_dimension:
            raise ValueError("coordinate dimension does not match request counts")
        if [coordinate.index for coordinate in self.coordinates] != list(
            range(expected_dimension)
        ):
            raise ValueError("coordinate indices must be contiguous")
        if len({coordinate.coordinate for coordinate in self.coordinates}) != len(
            self.coordinates
        ):
            raise ValueError("coordinates must be unique")
        if not self.initial_points:
            raise ValueError("at least one initial point is required")
        if any(len(point) != expected_dimension for point in self.initial_points):
            raise ValueError("initial point dimension does not match coordinate schema")
        return self


class UtilityLevelPeriodResult(_FrozenContract):
    """Solved level allocation for one period."""

    template_key: TemplateKey
    kind: UtilityLevelKind
    placement_rank: int
    supply_temperature: QuantityValue
    target_temperature: QuantityValue
    temperature_span: QuantityValue
    allocated_duty: QuantityValue
    maximum_duty: QuantityValue | None = None
    is_fallback: bool = False
    diagnostics: tuple[CandidateDiagnostic, ...] = ()


class ThermodynamicCostBreakdown(_FrozenContract):
    """Entropy and exergy terms for one result scope."""

    utility_entropy: QuantityValue
    process_entropy: QuantityValue
    total_entropy_generation: QuantityValue
    ambient_temperature: QuantityValue
    exergy_destruction: QuantityValue


class PlacementPeriodResult(_FrozenContract):
    """Complete utility placement and objective breakdown for one period."""

    period_id: str
    weight: float
    hot_levels: tuple[UtilityLevelPeriodResult, ...]
    cold_levels: tuple[UtilityLevelPeriodResult, ...]
    allocated_hot_duty: QuantityValue
    allocated_cold_duty: QuantityValue
    residual_hot_duty: QuantityValue
    residual_cold_duty: QuantityValue
    hot_coverage_residual: QuantityValue
    cold_coverage_residual: QuantityValue
    coverage_tolerance: QuantityValue
    feasible: bool
    thermodynamic: ThermodynamicCostBreakdown | None = None
    fallback_penalty: QuantityValue = Field(
        default_factory=lambda: QuantityValue(value=0.0, unit="dimensionless")
    )
    selected_objective: QuantityValue
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @field_validator("period_id")
    @classmethod
    def _validate_period_id(cls, value: str) -> str:
        period_id = value.strip()
        if not period_id:
            raise ValueError("period_id must not be empty")
        return period_id

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        result = _finite_float(value)
        if result < 0.0:
            raise ValueError("period weight must be non-negative")
        return result


class UtilityPlacementCandidate(_FrozenContract):
    """One feasible ranked candidate with nested period results."""

    coordinates: tuple[float, ...]
    hot_levels: tuple[DecodedUtilityLevel, ...]
    cold_levels: tuple[DecodedUtilityLevel, ...]
    period_results: tuple[PlacementPeriodResult, ...]
    aggregate_objective: QuantityValue
    thermodynamic_total: QuantityValue | None = None
    fallback_penalty: QuantityValue = Field(
        default_factory=lambda: QuantityValue(value=0.0, unit="dimensionless")
    )
    feasible: Literal[True] = True
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @field_validator("coordinates")
    @classmethod
    def _validate_coordinates(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(_finite_float(item) for item in value)


class PlacementTermination(_FrozenContract):
    """Backend-independent termination metadata."""

    method: str
    seed: int
    status: str
    code: str
    message: str
    iterations: int | None = None
    evaluations: int | None = None
    candidate_count: int
    feasible_candidate_count: int
    iteration_limit: int
    evaluation_limit: int

    @field_validator("method", "status", "code", "message")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("termination text must not be empty")
        return normalized

    @field_validator(
        "seed",
        "iterations",
        "evaluations",
        "candidate_count",
        "feasible_candidate_count",
        "iteration_limit",
        "evaluation_limit",
        mode="before",
    )
    @classmethod
    def _validate_integer(cls, value: object) -> object:
        if value is None:
            return value
        return _strict_int(value)

    @field_validator(
        "iterations",
        "evaluations",
        "candidate_count",
        "feasible_candidate_count",
    )
    @classmethod
    def _validate_nonnegative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("termination count must be non-negative")
        return value

    @field_validator("iteration_limit", "evaluation_limit")
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("termination limit must be positive")
        return value


class UtilityPlacementResult(_FrozenContract):
    """Complete detached result exposed by the later service and presentation."""

    request: UtilityPlacementRequest
    scope: UtilityPlacementBaseTarget
    base_target_id: str
    period_ids: tuple[str, ...]
    period_weights: tuple[float, ...]
    units: PlacementUnitSystem = Field(default_factory=PlacementUnitSystem)
    best: UtilityPlacementCandidate
    alternatives: tuple[UtilityPlacementCandidate, ...] = ()
    termination: PlacementTermination
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @field_validator("base_target_id")
    @classmethod
    def _validate_target_id(cls, value: str) -> str:
        target_id = value.strip()
        if not target_id:
            raise ValueError("base_target_id must not be empty")
        return target_id

    @field_validator("period_ids")
    @classmethod
    def _validate_period_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(period_id.strip() for period_id in value)
        if not normalized or any(not period_id for period_id in normalized):
            raise ValueError("period_ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("period_ids must be unique")
        return normalized

    @field_validator("period_weights")
    @classmethod
    def _validate_weights(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        normalized = tuple(_finite_float(weight) for weight in value)
        if any(weight < 0.0 for weight in normalized):
            raise ValueError("period weights must be non-negative")
        if not any(weight > 0.0 for weight in normalized):
            raise ValueError("at least one period weight must be positive")
        return normalized

    @model_validator(mode="after")
    def _validate_result_metadata(self) -> Self:
        if self.scope is UtilityPlacementBaseTarget.AUTO:
            raise ValueError("result scope must be resolved")
        if len(self.period_ids) != len(self.period_weights):
            raise ValueError("period identifiers and weights must align")
        return self


class UtilityPlacementRequest(_FrozenContract):
    """Detached normalized request for utility-placement analysis."""

    isothermal_level_count: int
    sensible_level_count: int = 0
    hot_templates: tuple[UtilityLevelTemplate, ...] | None = None
    cold_templates: tuple[UtilityLevelTemplate, ...] | None = None
    base_target: UtilityPlacementBaseTarget = UtilityPlacementBaseTarget.AUTO
    zone: str | None = None
    period_ids: tuple[str, ...] | None = None
    maximum_duties: tuple[UtilityDutyLimit, ...] = ()
    tolerances: PlacementTolerances = Field(default_factory=PlacementTolerances)
    options: UtilityPlacementOptions = Field(default_factory=UtilityPlacementOptions)
    units: PlacementUnitSystem = Field(default_factory=PlacementUnitSystem)

    @property
    def uses_generated_pairs(self) -> bool:
        """Return whether count-generated hot/cold levels share coordinates."""
        return self.hot_templates is None and self.cold_templates is None

    @field_validator(
        "isothermal_level_count",
        "sensible_level_count",
        mode="before",
    )
    @classmethod
    def _validate_count_type(cls, value: object) -> object:
        return _strict_int(value)

    @field_validator("isothermal_level_count")
    @classmethod
    def _validate_isothermal_count(cls, value: int) -> int:
        if value < 2:
            raise ValueError("isothermal_level_count must be at least 2")
        return value

    @field_validator("sensible_level_count")
    @classmethod
    def _validate_sensible_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sensible_level_count must be non-negative")
        return value

    @field_validator("period_ids")
    @classmethod
    def _validate_period_ids(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(period_id.strip() for period_id in value)
        if not normalized or any(not period_id for period_id in normalized):
            raise ValueError("period_ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("period_ids must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_maximum_duties(self) -> Self:
        names = tuple(limit.name for limit in self.maximum_duties)
        if len(set(names)) != len(names):
            raise ValueError("maximum-duty utility names must be unique")
        if self.period_ids is not None and any(
            limit.period_ids != self.period_ids for limit in self.maximum_duties
        ):
            raise ValueError("maximum-duty periods must match request periods")
        return self


__all__ = [
    "CandidateDiagnostic",
    "CandidateVerification",
    "CoordinateKey",
    "DecodedPlacement",
    "DecodedUtilityLevel",
    "DecisionField",
    "DecisionCoordinate",
    "EffectiveUtilityTemplate",
    "PhysicalCoordinateBound",
    "PlacementFeasibilityEnvelope",
    "PlacementPeriodResult",
    "PlacementPeriodEnvelope",
    "PlacementTermination",
    "PlacementTolerances",
    "PlacementUnitSystem",
    "QuantityInterval",
    "QuantityValue",
    "TemplateBlueprintSet",
    "TemplateKey",
    "ThermodynamicCostBreakdown",
    "UtilityLevelKind",
    "UtilityLevelPeriodResult",
    "UtilityLevelTemplate",
    "UtilityDutyLimit",
    "UtilityPlacementBaseTarget",
    "UtilityPlacementCandidate",
    "UtilityPlacementOptimisationMethod",
    "UtilityPlacementOptions",
    "UtilityPlacementRequest",
    "UtilityPlacementModel",
    "UtilityPlacementResult",
    "UtilitySide",
    "UtilityTemplateBlueprint",
    "UtilityTemplateSet",
]
