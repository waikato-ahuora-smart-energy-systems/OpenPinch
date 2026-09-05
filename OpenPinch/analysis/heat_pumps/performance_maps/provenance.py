"""Deterministic plain-data provenance for generated HPR maps."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ....contracts.hpr_performance_map import JsonValue
from .models import HprMapGenerationContext, HprSimulatorMetadata


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def build_hpr_map_provenance(
    context: HprMapGenerationContext,
    metadata: HprSimulatorMetadata,
    *,
    point_count: int,
) -> dict[str, JsonValue]:
    """Assemble stable recursive JSON only after complete success."""
    fluid = context.working_fluid
    basis = context.basis
    return {
        "openpinch_version": _package_version("OpenPinch"),
        "simulation_backend": basis.simulation_backend.strip().lower(),
        "engine_version": metadata.engine_version,
        "model_id": metadata.model_id,
        "cycle_id": basis.cycle_id,
        "mode": basis.mode,
        "working_fluid": {
            "source_spec": fluid.source_spec,
            "property_backend": fluid.property_backend,
            "kind": fluid.kind,
            "registered_name": fluid.registered_name,
            "components": list(fluid.components),
            "mole_fractions": list(fluid.mole_fractions),
            "composition_basis": fluid.composition_basis,
            "saturation_anchor": fluid.saturation_anchor,
        },
        "design_condition": {
            "evaporating_temperature": basis.nominal_evaporating_temperature,
            "condensing_temperature": basis.nominal_condensing_temperature,
            "source_temperature": context.nominal_source_temperature,
            "sink_temperature": context.nominal_sink_temperature,
            "reference_capacity": context.reference_capacity,
            "reference_capacity_basis": context.reference_capacity_basis,
        },
        "assumptions": {
            "source_approach_temperature": basis.source_approach_temperature,
            "sink_approach_temperature": basis.sink_approach_temperature,
            "compressor_isentropic_efficiency": (
                basis.compressor_isentropic_efficiency
            ),
            "superheat": basis.superheat,
            "subcooling": basis.subcooling,
            "internal_hx_gas_temperature_change": (
                basis.internal_hx_gas_temperature_change
            ),
        },
        "characteristic_set_id": metadata.characteristic_set_id,
        "design_converged": metadata.design_converged,
        "design_details": dict(metadata.design_details),
        "power_boundary": metadata.power_boundary,
        "modeled_auxiliaries": list(metadata.modeled_auxiliaries),
        "point_count": point_count,
        "source_count": len(context.request.source_temperatures),
        "sink_count": len(context.request.sink_temperatures),
        "load_count": len(context.request.load_fractions),
        "source_target": dict(basis.source_provenance),
    }


__all__ = ["build_hpr_map_provenance"]
