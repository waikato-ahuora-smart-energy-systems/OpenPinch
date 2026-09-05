"""Plain winning simulation records for map-compatible HPR targets."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRParsedState,
    HprTargetSimulationRecord,
)
from .targeting_models import (
    HprFrozenJson,
    HprTargetEvaluatorMetadata,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
)


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def build_coolprop_target_simulation_record(
    *,
    args: HeatPumpTargetInputs,
    state: HPRParsedState,
    cycle: Any,
) -> HprTargetSimulationRecord | None:
    """Build a detached record from an already solved compatible CoolProp cycle."""
    if int(args.n_evap) != 1 or int(args.n_cond) != 1:
        return None
    mode = "heat_pump" if args.is_heat_pumping else "refrigeration"
    useful_duty = (
        float(np.asarray(cycle.Q_heat_arr, dtype=float).sum())
        if args.is_heat_pumping
        else float(np.asarray(cycle.Q_cool_arr, dtype=float).sum())
    )
    return HprTargetSimulationRecord(
        simulation_backend="coolprop",
        mode=mode,
        cycle_id="single_stage_vapour_compression",
        model_id="openpinch-coolprop-single-stage-v1",
        refrigerant_spec=str(args.refrigerant_ls[0]),
        nominal_evaporating_temperature=float(state.T_evap[0]),
        nominal_condensing_temperature=float(state.T_cond[0]),
        nominal_useful_duty=useful_duty,
        source_approach_temperature=float(args.dtcont_hp),
        sink_approach_temperature=float(args.dtcont_hp),
        compressor_isentropic_efficiency=float(args.eta_comp),
        superheat=_first_scalar(getattr(cycle, "dT_superheat", None), default=0.0),
        subcooling=_first_scalar(state.dT_subcool, default=0.0),
        internal_hx_gas_temperature_change=_first_scalar(
            state.dT_ihx_gas_side,
            default=0.0,
        ),
        evaporator_count=1,
        condenser_count=1,
        period_id=None,
        engine_version=_distribution_version("CoolProp"),
        assumptions={
            "target_evaluator": "existing_cascade_vapour_compression_cycle",
            "property_backend": _property_backend(str(args.refrigerant_ls[0])),
            "saturation_anchor": "evaporation_dew_condensation_bubble",
            "candidate_solve_mode": "steady_state",
            "modeled_auxiliaries": [],
        },
    )


def build_tespy_target_simulation_record(
    *,
    request: HprTargetThermodynamicRequest,
    result: HprTargetThermodynamicResult,
    metadata: HprTargetEvaluatorMetadata,
) -> HprTargetSimulationRecord:
    """Build a detached record from one accepted engine-neutral TESPy result."""
    return HprTargetSimulationRecord(
        simulation_backend="tespy",
        mode=request.mode,
        cycle_id=request.cycle_id,
        model_id=request.model_id,
        refrigerant_spec=request.working_fluid.source_spec,
        nominal_evaporating_temperature=request.evaporating_temperature,
        nominal_condensing_temperature=request.condensing_temperature,
        nominal_useful_duty=request.useful_duty,
        source_approach_temperature=request.source_approach_temperature,
        sink_approach_temperature=request.sink_approach_temperature,
        compressor_isentropic_efficiency=request.compressor_isentropic_efficiency,
        superheat=request.superheat,
        subcooling=request.subcooling,
        internal_hx_gas_temperature_change=(request.internal_hx_gas_temperature_change),
        evaporator_count=1,
        condenser_count=1,
        period_id=None,
        engine_version=result.engine_version,
        power_boundary=metadata.power_boundary,
        assumptions={
            "session": _thaw_pairs(metadata.assumptions),
            "winning_design": _thaw_pairs(result.design_details),
        },
    )


def _first_scalar(value: object, *, default: float) -> float:
    if value is None:
        return default
    array = np.asarray(value, dtype=float).reshape(-1)
    return default if array.size == 0 else float(array[0])


def _property_backend(fluid: str) -> str:
    return fluid.split("::", 1)[0].upper() if "::" in fluid else "HEOS"


def _thaw_pairs(
    value: tuple[tuple[str, HprFrozenJson], ...] | object,
) -> dict[str, object]:
    if not isinstance(value, tuple):
        return dict(value) if hasattr(value, "items") else {}
    return {key: _thaw_json(item) for key, item in value}


def _thaw_json(value: HprFrozenJson) -> object:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw_json(item) for key, item in value}
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "build_coolprop_target_simulation_record",
    "build_tespy_target_simulation_record",
]
