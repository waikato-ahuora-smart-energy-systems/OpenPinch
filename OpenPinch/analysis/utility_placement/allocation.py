"""Candidate-local duty allocation through the existing targeting owner."""

from __future__ import annotations

import math
from typing import Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from OpenPinch.analysis.targeting.utilities import target_utilities_for_load_profiles
from OpenPinch.contracts.utility_placement import (
    CandidateDiagnostic,
    DecodedPlacement,
    DecodedUtilityLevel,
    QuantityValue,
    TemplateKey,
    UtilityLevelKind,
    UtilityPlacementRequest,
    UtilitySide,
)
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection

from .context import PlacementPeriodInput
from .errors import PlacementTargetingError


class _FrozenAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AllocationAdapterResult(_FrozenAllocation):
    """Raw level duties returned by one detached allocation replay."""

    hot_duties: tuple[float, ...]
    cold_duties: tuple[float, ...]
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @field_validator("hot_duties", "cold_duties", mode="before")
    @classmethod
    def _validate_duties(cls, value: object) -> tuple[float, ...]:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
        if any(not math.isfinite(item) or item < 0.0 for item in result):
            raise ValueError("allocation duties must be finite and non-negative")
        return result


class UtilityAllocationSlice(_FrozenAllocation):
    """Stable assignment of one level's duty to one target interval."""

    interval_index: int
    duty: float


class AllocatedUtilityLevel(_FrozenAllocation):
    """Decoded utility level plus its candidate-local assigned duty."""

    template_key: TemplateKey
    kind: UtilityLevelKind
    placement_rank: int
    supply_temperature: float
    target_temperature: float
    allocated_duty: float
    slices: tuple[UtilityAllocationSlice, ...] = ()


class PlacementPeriodAllocation(_FrozenAllocation):
    """Complete conservation result for one candidate and period."""

    period_id: str
    hot_levels: tuple[AllocatedUtilityLevel, ...]
    cold_levels: tuple[AllocatedUtilityLevel, ...]
    allocated_hot_duty: float
    allocated_cold_duty: float
    hot_coverage_residual: float
    cold_coverage_residual: float
    coverage_tolerance_hot: float
    coverage_tolerance_cold: float
    feasible: bool
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _validate_feasibility(self) -> Self:
        if self.feasible and self.diagnostics:
            raise ValueError("feasible allocation cannot contain diagnostics")
        if not self.feasible and not self.diagnostics:
            raise ValueError("infeasible allocation requires diagnostics")
        return self


class PlacementAllocationAdapter(Protocol):
    """Injection boundary for an existing or test allocation owner."""

    def allocate(
        self, period: PlacementPeriodInput, placement: DecodedPlacement
    ) -> AllocationAdapterResult: ...


def _build_stream(level: DecodedUtilityLevel) -> Stream:
    return Stream(
        name=level.template_key.name,
        supply_temperature=level.supply_temperature.value,
        target_temperature=level.target_temperature.value,
        heat_flow=0.0,
        is_process_stream=False,
    )


class ExistingTargetingAllocationAdapter:
    """Fresh-stream adapter over ``target_utilities_for_load_profiles``."""

    def allocate(
        self, period: PlacementPeriodInput, placement: DecodedPlacement
    ) -> AllocationAdapterResult:
        hot = StreamCollection([_build_stream(level) for level in placement.hot])
        cold = StreamCollection([_build_stream(level) for level in placement.cold])
        snapshot = period.snapshot
        try:
            targeted_hot, targeted_cold = target_utilities_for_load_profiles(
                hot_utilities=hot,
                cold_utilities=cold,
                T_vals=np.asarray(snapshot.shifted_temperatures, dtype=float),
                H_net_cold=np.asarray(snapshot.hot_load_profile, dtype=float),
                H_net_hot=np.asarray(snapshot.cold_load_profile, dtype=float),
                pinch_idx=(snapshot.hot_pinch_index, snapshot.cold_pinch_index),
                is_real_temperatures=False,
            )
        except (ArithmeticError, ValueError) as exc:
            return AllocationAdapterResult(
                hot_duties=(0.0,) * len(placement.hot),
                cold_duties=(0.0,) * len(placement.cold),
                diagnostics=(
                    CandidateDiagnostic(
                        code="targeting_infeasible",
                        constraint="utility_allocation",
                        message=str(exc) or "Utility allocation is infeasible.",
                        period_id=period.period_id,
                    ),
                ),
            )

        return AllocationAdapterResult(
            hot_duties=tuple(
                float(
                    targeted_hot.get_stream_by_name(
                        level.template_key.name
                    ).heat_flow.value
                )
                for level in placement.hot
            ),
            cold_duties=tuple(
                float(
                    targeted_cold.get_stream_by_name(
                        level.template_key.name
                    ).heat_flow.value
                )
                for level in placement.cold
            ),
        )


def _interval_index(period: PlacementPeriodInput, temperature: float) -> int:
    temperatures = period.snapshot.shifted_temperatures
    for index, (first, second) in enumerate(zip(temperatures, temperatures[1:])):
        lower, upper = sorted((first, second))
        if lower <= temperature <= upper:
            return index
    return min(
        range(len(temperatures) - 1),
        key=lambda index: abs(
            temperature - (temperatures[index] + temperatures[index + 1]) / 2.0
        ),
    )


def _allocated_levels(
    period: PlacementPeriodInput,
    levels: tuple[DecodedUtilityLevel, ...],
    duties: tuple[float, ...],
) -> tuple[AllocatedUtilityLevel, ...]:
    result = []
    for level, duty in zip(levels, duties, strict=True):
        slices = ()
        if duty > 0.0:
            interval = _interval_index(
                period,
                (level.supply_temperature.value + level.target_temperature.value) / 2.0,
            )
            slices = (UtilityAllocationSlice(interval_index=interval, duty=duty),)
        result.append(
            AllocatedUtilityLevel(
                template_key=level.template_key,
                kind=level.kind,
                placement_rank=level.placement_rank,
                supply_temperature=level.supply_temperature.value,
                target_temperature=level.target_temperature.value,
                allocated_duty=duty,
                slices=slices,
            )
        )
    return tuple(result)


def _coverage_diagnostic(
    *,
    side: UtilitySide,
    period: PlacementPeriodInput,
    residual: float,
    tolerance: float,
    heat_flow_unit: str,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        code=f"{side.value}_coverage_shortfall",
        constraint="utility_coverage",
        message=(
            f"Allocated {side.value} utility duty does not cover the residual demand."
        ),
        side=side,
        period_id=period.period_id,
        measured=QuantityValue(value=residual, unit=heat_flow_unit),
        limit=QuantityValue(value=tolerance, unit=heat_flow_unit),
    )


def allocate_placement_period(
    *,
    request: UtilityPlacementRequest,
    period: PlacementPeriodInput,
    placement: DecodedPlacement,
    adapter: PlacementAllocationAdapter | None = None,
) -> PlacementPeriodAllocation:
    """Replay one candidate and enforce exact hot/cold coverage within tolerance."""
    owner = adapter or ExistingTargetingAllocationAdapter()
    try:
        raw = owner.allocate(period, placement)
    except PlacementTargetingError:
        raise
    except Exception as exc:
        raise PlacementTargetingError(
            code="allocation_adapter_failure",
            message="Utility allocation adapter failed.",
            period_id=period.period_id,
            details=(("reason", str(exc)),),
        ) from exc

    if len(raw.hot_duties) != len(placement.hot) or len(raw.cold_duties) != len(
        placement.cold
    ):
        raise PlacementTargetingError(
            code="allocation_shape_mismatch",
            message="Allocation adapter duty count does not match utility levels.",
            period_id=period.period_id,
        )

    allocated_hot = math.fsum(raw.hot_duties)
    allocated_cold = math.fsum(raw.cold_duties)
    hot_residual = abs(period.residual_hot_duty - allocated_hot)
    cold_residual = abs(period.residual_cold_duty - allocated_cold)
    hot_tolerance = request.tolerances.coverage + request.tolerances.relative * max(
        period.residual_hot_duty, 1.0
    )
    cold_tolerance = request.tolerances.coverage + request.tolerances.relative * max(
        period.residual_cold_duty, 1.0
    )
    diagnostics = list(raw.diagnostics)
    if not diagnostics and hot_residual > hot_tolerance:
        diagnostics.append(
            _coverage_diagnostic(
                side=UtilitySide.HOT,
                period=period,
                residual=hot_residual,
                tolerance=hot_tolerance,
                heat_flow_unit=request.units.heat_flow,
            )
        )
    if not diagnostics and cold_residual > cold_tolerance:
        diagnostics.append(
            _coverage_diagnostic(
                side=UtilitySide.COLD,
                period=period,
                residual=cold_residual,
                tolerance=cold_tolerance,
                heat_flow_unit=request.units.heat_flow,
            )
        )

    return PlacementPeriodAllocation(
        period_id=period.period_id,
        hot_levels=_allocated_levels(period, placement.hot, raw.hot_duties),
        cold_levels=_allocated_levels(period, placement.cold, raw.cold_duties),
        allocated_hot_duty=allocated_hot,
        allocated_cold_duty=allocated_cold,
        hot_coverage_residual=hot_residual,
        cold_coverage_residual=cold_residual,
        coverage_tolerance_hot=hot_tolerance,
        coverage_tolerance_cold=cold_tolerance,
        feasible=not diagnostics,
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "AllocatedUtilityLevel",
    "AllocationAdapterResult",
    "ExistingTargetingAllocationAdapter",
    "PlacementAllocationAdapter",
    "PlacementPeriodAllocation",
    "UtilityAllocationSlice",
    "allocate_placement_period",
]
