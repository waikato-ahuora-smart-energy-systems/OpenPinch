"""Pure construction of validated HPR map-generation contexts."""

from __future__ import annotations

import math

from ....contracts.hpr_performance_map import HprPerformanceMapRequest
from .fluids import resolve_hpr_working_fluid
from .models import HprMapGenerationContext, HprTargetMapBasis

_SUPPORTED_CYCLE = "single_stage_vapour_compression"
_ENERGY_BALANCE_TOLERANCE = 1e-6
_TEMPERATURE_MATCH_TOLERANCE = 1e-6


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def build_hpr_map_generation_context(
    basis: HprTargetMapBasis,
    request: HprPerformanceMapRequest,
) -> HprMapGenerationContext:
    """Validate detached target facts and combine them with a Unit 1 request."""
    backend = basis.simulation_backend.strip().lower()
    if backend not in {"coolprop", "tespy"}:
        raise ValueError("simulation backend must be 'coolprop' or 'tespy'")
    if basis.mode not in {"heat_pump", "refrigeration"}:
        raise ValueError("mode must be 'heat_pump' or 'refrigeration'")
    if basis.cycle_id != _SUPPORTED_CYCLE:
        raise ValueError("target must use the supported single-stage cycle")
    if not basis.target_id.strip() or not basis.model_id.strip():
        raise ValueError("target and model identifiers must not be empty")

    nominal_evaporating = _finite(
        basis.nominal_evaporating_temperature,
        "nominal evaporating temperature",
    )
    nominal_condensing = _finite(
        basis.nominal_condensing_temperature,
        "nominal condensing temperature",
    )
    if nominal_evaporating <= -273.15 or nominal_condensing <= -273.15:
        raise ValueError("nominal temperatures must be above absolute zero")
    if nominal_condensing <= nominal_evaporating:
        raise ValueError("nominal target must have positive temperature lift")

    capacity = float(
        request.reference_capacity
        if request.reference_capacity is not None
        else basis.nominal_useful_duty
    )
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("reference capacity must be finite and positive")

    source_approach = _finite(
        basis.source_approach_temperature,
        "source approach temperature",
    )
    sink_approach = _finite(
        basis.sink_approach_temperature,
        "sink approach temperature",
    )
    if source_approach < 0.0 or sink_approach < 0.0:
        raise ValueError("approach temperatures must be nonnegative")
    for value, label in (
        (basis.superheat, "superheat"),
        (basis.subcooling, "subcooling"),
        (
            basis.internal_hx_gas_temperature_change,
            "internal heat-exchanger temperature change",
        ),
    ):
        if _finite(value, label) < 0.0:
            raise ValueError(f"{label} must be nonnegative")
    efficiency = _finite(
        basis.compressor_isentropic_efficiency,
        "compressor efficiency",
    )
    if efficiency <= 0.0 or efficiency > 1.0:
        raise ValueError("compressor efficiency must be in the interval (0, 1]")

    working_fluid = resolve_hpr_working_fluid(
        basis.refrigerant_spec,
        nominal_evaporating,
        nominal_condensing,
    )
    is_heat_pump = basis.mode == "heat_pump"
    return HprMapGenerationContext(
        basis=basis,
        working_fluid=working_fluid,
        request=request,
        reference_capacity=capacity,
        reference_capacity_basis="q_sink" if is_heat_pump else "q_source",
        cop_convention="heating" if is_heat_pump else "cooling",
        nominal_source_temperature=nominal_evaporating + source_approach,
        nominal_sink_temperature=nominal_condensing - sink_approach,
        characteristic_set_id=(
            "openpinch-steady-state-vapour-compression-v1"
            if backend == "coolprop"
            else "openpinch-single-stage-compressor-v1"
        ),
        energy_balance_tolerance=_ENERGY_BALANCE_TOLERANCE,
        temperature_match_tolerance=_TEMPERATURE_MATCH_TOLERANCE,
    )


__all__ = ["build_hpr_map_generation_context"]
