"""Deterministic candidate evaluation session with process-local exact memoization."""

from __future__ import annotations

import math
from collections import Counter
from threading import Lock

from pydantic import BaseModel, ConfigDict, field_validator

from OpenPinch.analysis.utility_placement.codec import (
    decode_placement,
    verify_candidate,
)
from OpenPinch.contracts.utility_placement import (
    CandidateDiagnostic,
    PlacementPeriodResult,
    QuantityValue,
    UtilityLevelPeriodResult,
    UtilityPlacementModel,
    UtilityPlacementRequest,
)

from .allocation import PlacementAllocationAdapter, allocate_placement_period
from .context import UtilityPlacementContext
from .errors import PlacementThermodynamicError
from .penalties import (
    aggregate_weighted_objective,
    feasible_objective_scalar,
    g_penalty,
    infeasible_objective_scalar,
    penalized_feasible_objective_scalar,
)
from .thermodynamics import evaluate_thermodynamic_cost, stream_entropy_change


class PlacementEvaluation(BaseModel):
    """Compact complete result of evaluating one exact coordinate tuple."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    coordinates: tuple[float, ...]
    feasible: bool
    scalar_objective: float
    physical_objective: float | None = None
    period_results: tuple[PlacementPeriodResult, ...] = ()
    thermodynamic_total: float | None = None
    fallback_penalty: float = 0.0
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @field_validator("coordinates", mode="before")
    @classmethod
    def _normalize_coordinates(cls, value: object) -> tuple[float, ...]:
        return tuple(0.0 if float(item) == 0.0 else float(item) for item in value)  # type: ignore[arg-type]


def _public_level(level, request: UtilityPlacementRequest) -> UtilityLevelPeriodResult:
    units = request.units
    return UtilityLevelPeriodResult(
        template_key=level.template_key,
        kind=level.kind,
        placement_rank=level.placement_rank,
        supply_temperature=QuantityValue(
            value=level.supply_temperature, unit=units.absolute_temperature
        ),
        target_temperature=QuantityValue(
            value=level.target_temperature, unit=units.absolute_temperature
        ),
        temperature_span=QuantityValue(
            value=abs(level.target_temperature - level.supply_temperature),
            unit=units.temperature_difference,
        ),
        allocated_duty=QuantityValue(value=level.allocated_duty, unit=units.heat_flow),
        maximum_duty=(
            QuantityValue(value=level.maximum_duty, unit=units.heat_flow)
            if level.maximum_duty is not None
            else None
        ),
        is_fallback=level.is_fallback,
    )


class PlacementEvaluationSession:
    """Evaluate candidates with one bounded exact-coordinate memo per process."""

    def __init__(
        self,
        *,
        request: UtilityPlacementRequest,
        context: UtilityPlacementContext,
        model: UtilityPlacementModel,
        allocation_adapter: PlacementAllocationAdapter | None = None,
    ) -> None:
        self.request = request
        self.context = context
        self.model = model
        self.allocation_adapter = allocation_adapter
        self._reset_process_state()

    def _reset_process_state(self) -> None:
        self._memo: dict[tuple[float, ...], PlacementEvaluation] = {}
        self._lock = Lock()
        self._evaluation_count = 0
        self._memo_hit_count = 0
        self._diagnostic_counts: Counter[str] = Counter()
        self._diagnostic_representatives: list[CandidateDiagnostic] = []

    def __getstate__(self):
        return {
            "request": self.request,
            "context": self.context,
            "model": self.model,
            "allocation_adapter": self.allocation_adapter,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_process_state()

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def memo_hit_count(self) -> int:
        return self._memo_hit_count

    @property
    def diagnostic_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._diagnostic_counts.items()))

    @property
    def diagnostic_representatives(self) -> tuple[CandidateDiagnostic, ...]:
        return tuple(self._diagnostic_representatives)

    def _record(self, diagnostics: tuple[CandidateDiagnostic, ...]) -> None:
        for diagnostic in diagnostics:
            self._diagnostic_counts[diagnostic.code] += 1
            if len(self._diagnostic_representatives) < 10:
                self._diagnostic_representatives.append(diagnostic)

    def _infeasible(
        self,
        key: tuple[float, ...],
        diagnostics: tuple[CandidateDiagnostic, ...],
        violation: float,
    ) -> PlacementEvaluation:
        self._record(diagnostics)
        return PlacementEvaluation(
            coordinates=key,
            feasible=False,
            scalar_objective=infeasible_objective_scalar(max(violation, 0.0)),
            diagnostics=diagnostics,
        )

    def _objective_scale(self) -> float:
        entropy_reference = math.fsum(
            period.weight
            * math.fsum(
                abs(
                    stream_entropy_change(
                        item.available_duty,
                        item.temperature_in_kelvin,
                        item.temperature_out_kelvin,
                    )
                )
                for item in period.snapshot.entropy_slices
            )
            for period in self.context.periods
        )
        return max(entropy_reference, 1e-12)

    def evaluate(self, coordinates) -> PlacementEvaluation:
        key = tuple(
            0.0 if float(value) == 0.0 else float(value) for value in coordinates
        )
        with self._lock:
            cached = self._memo.get(key)
            if cached is not None:
                self._memo_hit_count += 1
                return cached
            if self._evaluation_count >= self.request.options.evaluation_limit:
                diagnostic = CandidateDiagnostic(
                    code="evaluation_budget_exhausted",
                    constraint="evaluation_limit",
                    message="The placement evaluation budget is exhausted.",
                )
                result = self._infeasible(
                    key,
                    (diagnostic,),
                    float(self._evaluation_count + 1),
                )
                self._memo[key] = result
                return result
            self._evaluation_count += 1

        verification = verify_candidate(self.model, key)
        if not verification.feasible:
            result = self._infeasible(
                key,
                verification.diagnostics,
                float(len(verification.diagnostics)),
            )
        else:
            placement = decode_placement(self.model, key)
            period_results: list[PlacementPeriodResult] = []
            thermo_values: list[float] = []
            fallback_penalties: list[float] = []
            failure_diagnostics: list[CandidateDiagnostic] = []
            violation = 0.0
            for period in self.context.periods:
                allocation = allocate_placement_period(
                    request=self.request,
                    period=period,
                    placement=placement,
                    adapter=self.allocation_adapter,
                )
                if not allocation.feasible:
                    failure_diagnostics.extend(allocation.diagnostics)
                    violation += allocation.hot_coverage_residual / max(
                        allocation.coverage_tolerance_hot, 1e-300
                    )
                    violation += allocation.cold_coverage_residual / max(
                        allocation.coverage_tolerance_cold, 1e-300
                    )
                    continue

                fallback_penalty = g_penalty(
                    hot_fallback_duty=allocation.hot_fallback_duty,
                    cold_fallback_duty=allocation.cold_fallback_duty,
                    required_hot_duty=allocation.required_hot_duty,
                    required_cold_duty=allocation.required_cold_duty,
                )
                fallback_penalties.append(fallback_penalty)

                try:
                    thermo = evaluate_thermodynamic_cost(
                        request=self.request,
                        period=period.model_copy(
                            update={"snapshot": allocation.target_snapshot}
                        ),
                        allocation=allocation,
                    )
                except PlacementThermodynamicError as exc:
                    if exc.code not in {
                        "invalid_balanced_composite",
                        "negative_entropy_generation",
                    }:
                        raise
                    failure_diagnostics.append(
                        CandidateDiagnostic(
                            code=exc.code,
                            constraint="thermodynamic_feasibility",
                            message=str(exc),
                            period_id=period.period_id,
                            details=(("thermodynamic_error", exc.code),),
                        )
                    )
                    violation += 1.0
                    continue
                thermo_value = thermo.total_entropy_generation.value
                thermo_values.append(thermo_value)
                period_results.append(
                    PlacementPeriodResult(
                        period_id=period.period_id,
                        weight=period.weight,
                        hot_levels=tuple(
                            _public_level(level, self.request)
                            for level in allocation.hot_levels
                        ),
                        cold_levels=tuple(
                            _public_level(level, self.request)
                            for level in allocation.cold_levels
                        ),
                        allocated_hot_duty=QuantityValue(
                            value=allocation.allocated_hot_duty,
                            unit=self.request.units.heat_flow,
                        ),
                        allocated_cold_duty=QuantityValue(
                            value=allocation.allocated_cold_duty,
                            unit=self.request.units.heat_flow,
                        ),
                        residual_hot_duty=QuantityValue(
                            value=allocation.required_hot_duty,
                            unit=self.request.units.heat_flow,
                        ),
                        residual_cold_duty=QuantityValue(
                            value=allocation.required_cold_duty,
                            unit=self.request.units.heat_flow,
                        ),
                        hot_coverage_residual=QuantityValue(
                            value=allocation.hot_coverage_residual,
                            unit=self.request.units.heat_flow,
                        ),
                        cold_coverage_residual=QuantityValue(
                            value=allocation.cold_coverage_residual,
                            unit=self.request.units.heat_flow,
                        ),
                        coverage_tolerance=QuantityValue(
                            value=max(
                                allocation.coverage_tolerance_hot,
                                allocation.coverage_tolerance_cold,
                            ),
                            unit=self.request.units.heat_flow,
                        ),
                        feasible=True,
                        thermodynamic=thermo,
                        fallback_penalty=QuantityValue(
                            value=fallback_penalty,
                            unit="dimensionless",
                        ),
                        selected_objective=QuantityValue(
                            value=thermo_value, unit=self.request.units.entropy
                        ),
                    )
                )

            if failure_diagnostics:
                result = self._infeasible(
                    key, tuple(failure_diagnostics), max(violation, 1.0)
                )
            else:
                weights = tuple(period.weight for period in self.context.periods)
                thermo_total = aggregate_weighted_objective(
                    tuple(thermo_values), weights
                )
                fallback_total = aggregate_weighted_objective(
                    tuple(fallback_penalties), weights
                )
                entropy_scalar = feasible_objective_scalar(
                    thermo_total, scale=self._objective_scale()
                )
                result = PlacementEvaluation(
                    coordinates=key,
                    feasible=True,
                    scalar_objective=penalized_feasible_objective_scalar(
                        entropy_scalar,
                        fallback_total,
                    ),
                    physical_objective=thermo_total,
                    period_results=tuple(period_results),
                    thermodynamic_total=thermo_total,
                    fallback_penalty=fallback_total,
                )

        with self._lock:
            self._memo[key] = result
        return result

    def objective(self, coordinates, *args) -> float:
        """Existing optimizer-compatible scalar callback."""
        return self.evaluate(coordinates).scalar_objective


__all__ = ["PlacementEvaluation", "PlacementEvaluationSession"]
