"""Immutable values shared by HPR performance-map generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ....contracts.hpr_performance_map import HprPerformanceMapRequest, JsonValue

type HprMode = Literal["heat_pump", "refrigeration"]
type HprSimulationBackend = Literal["coolprop", "tespy"]
type HprWorkingFluidKind = Literal["pure", "registered_blend", "explicit_molar_mixture"]


@dataclass(frozen=True, slots=True)
class HprTargetMapBasis:
    """Detached scalar facts extracted from one supported current HPR target."""

    target_id: str
    mode: str
    simulation_backend: str
    cycle_id: str
    model_id: str
    refrigerant_spec: str
    nominal_evaporating_temperature: float
    nominal_condensing_temperature: float
    nominal_useful_duty: float
    source_approach_temperature: float
    sink_approach_temperature: float
    compressor_isentropic_efficiency: float
    superheat: float
    subcooling: float
    internal_hx_gas_temperature_change: float
    source_provenance: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class HprWorkingFluidSpec:
    """Normalized identity for one closed-loop refrigerant specification."""

    source_spec: str
    property_backend: str
    kind: HprWorkingFluidKind
    registered_name: str | None
    components: tuple[str, ...]
    mole_fractions: tuple[float, ...]
    composition_basis: Literal["not_applicable", "provider_defined", "molar"]
    saturation_anchor: Literal["evaporation_dew_condensation_bubble"] = (
        "evaporation_dew_condensation_bubble"
    )


@dataclass(frozen=True, slots=True)
class HprMapGenerationContext:
    """Complete validated inputs for one simulator session."""

    basis: HprTargetMapBasis
    working_fluid: HprWorkingFluidSpec
    request: HprPerformanceMapRequest
    reference_capacity: float
    reference_capacity_basis: Literal["q_sink", "q_source"]
    cop_convention: Literal["heating", "cooling"]
    nominal_source_temperature: float
    nominal_sink_temperature: float
    characteristic_set_id: str
    energy_balance_tolerance: float
    temperature_match_tolerance: float


@dataclass(frozen=True, slots=True)
class HprOperatingPoint:
    """One canonical requested external-service operating coordinate."""

    ordinal: int
    source_index: int
    sink_index: int
    load_index: int
    curve_id: str
    name: str
    source_temperature: float
    sink_temperature: float
    evaporating_temperature: float
    condensing_temperature: float
    load_fraction: float
    requested_useful_duty: float
    validation_error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return whether the translated engine coordinate is physically valid."""
        return self.validation_error is None


@dataclass(frozen=True, slots=True)
class HprPointSimulation:
    """Normalized engine output before public-map validation."""

    q_source: float
    q_sink: float
    compressor_power: float
    converged: bool
    engine_details: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HprSimulatorMetadata:
    """Stable simulator preparation facts used for final provenance."""

    backend: HprSimulationBackend
    engine_version: str
    model_id: str
    characteristic_set_id: str
    design_converged: bool
    design_details: dict[str, JsonValue]
    power_boundary: Literal["compressor_only"] = "compressor_only"
    modeled_auxiliaries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HprSimulationDiagnostic:
    """Stable engine-neutral evidence for one generation failure."""

    code: str
    backend: str | None
    model_id: str | None
    point_ordinal: int | None
    curve_id: str | None
    source_temperature: float | None
    sink_temperature: float | None
    load_fraction: float | None
    message: str
    details: dict[str, JsonValue] = field(default_factory=dict)


__all__ = [
    "HprMapGenerationContext",
    "HprMode",
    "HprOperatingPoint",
    "HprPointSimulation",
    "HprSimulationBackend",
    "HprSimulationDiagnostic",
    "HprSimulatorMetadata",
    "HprTargetMapBasis",
    "HprWorkingFluidKind",
    "HprWorkingFluidSpec",
]
