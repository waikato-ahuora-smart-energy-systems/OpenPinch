"""Plain winning simulation records for map-compatible HPR targets."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRParsedState,
    HPRSimulationLoopRecord,
    HPRSimulationStageRecord,
    HprTargetSimulationRecord,
    HPRTopologyIdentifier,
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
    topology_id: HPRTopologyIdentifier | str = (
        HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
    ),
) -> HprTargetSimulationRecord:
    """Build a detached record from an already solved compatible CoolProp cycle."""
    topology = HPRTopologyIdentifier(topology_id)
    mode = "heat_pump" if args.is_heat_pumping else "refrigeration"
    Q_heat = _required_vector(cycle, "Q_heat_arr")
    Q_cool = _required_vector(cycle, "Q_cool_arr")
    work = _required_vector(cycle, "work_arr")
    count = max(Q_heat.size, Q_cool.size, work.size)
    Q_heat = _broadcast_vector(Q_heat, count, name="Q_heat_arr")
    Q_cool = _broadcast_vector(Q_cool, count, name="Q_cool_arr")
    work = _broadcast_vector(work, count, name="work_arr")
    useful_values = Q_heat if args.is_heat_pumping else Q_cool
    useful_duty = float(useful_values.sum())

    T_evap = _cycle_or_state_vector(cycle, "T_evap", state.T_evap, count)
    T_cond = _cycle_or_state_vector(cycle, "T_cond", state.T_cond, count)
    superheat = _cycle_or_state_scalar(
        cycle,
        "dT_superheat",
        None,
        default=0.0,
    )
    subcooling = _cycle_or_state_scalar(
        cycle,
        "dT_subcool",
        state.dT_subcool,
        default=0.0,
    )
    ihx = _cycle_or_state_scalar(
        cycle,
        "dT_ihx_gas_side",
        state.dT_ihx_gas_side,
        default=0.0,
    )

    loops = _build_coolprop_loops(
        args=args,
        topology=topology,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        work=work,
        T_evap=T_evap,
        T_cond=T_cond,
    )
    refrigerant_specs = [loop.fluid_spec for loop in loops] or [
        str(args.refrigerant_ls[0])
    ]
    return HprTargetSimulationRecord(
        simulation_backend="coolprop",
        mode=mode,
        cycle_id=topology.value,
        topology_id=topology,
        schema_version=("1.0" if not loops else "1.1"),
        model_id=f"openpinch-coolprop-{topology.value.replace('_', '-')}-v1",
        refrigerant_spec=";".join(refrigerant_specs),
        nominal_evaporating_temperature=float(T_evap.min()),
        nominal_condensing_temperature=float(T_cond.max()),
        nominal_useful_duty=useful_duty,
        source_approach_temperature=float(args.dtcont_hp),
        sink_approach_temperature=float(args.dtcont_hp),
        compressor_isentropic_efficiency=float(args.eta_comp),
        superheat=superheat,
        subcooling=subcooling,
        internal_hx_gas_temperature_change=ihx,
        evaporator_count=int(args.n_evap),
        condenser_count=int(args.n_cond),
        period_id=None,
        engine_version=_distribution_version("CoolProp"),
        assumptions={
            "target_evaluator": f"existing_{topology.value}_cycle",
            "property_backend": _property_backend(refrigerant_specs[0]),
            "saturation_anchor": "evaporation_dew_condensation_bubble",
            "candidate_solve_mode": "steady_state",
            "modeled_auxiliaries": [],
        },
        loops=loops,
    )


def _build_coolprop_loops(
    *,
    args: HeatPumpTargetInputs,
    topology: HPRTopologyIdentifier,
    Q_heat: np.ndarray,
    Q_cool: np.ndarray,
    work: np.ndarray,
    T_evap: np.ndarray,
    T_cond: np.ndarray,
) -> tuple[HPRSimulationLoopRecord, ...]:
    if topology is HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION:
        return ()

    n_vc = Q_heat.size
    if topology is HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR:
        n_vc = max(int(args.n_cond), int(args.n_evap))
    refrigerants = [str(value) for value in args.refrigerant_ls]
    mvr_fluids = [str(value) for value in getattr(args, "mvr_fluid_ls", [])]
    loops: list[HPRSimulationLoopRecord] = []
    for ordinal in range(Q_heat.size):
        is_mvr = (
            topology is HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR and ordinal >= n_vc
        )
        role = "mvr" if is_mvr else "vapour_compression"
        fluid_index = ordinal - n_vc if is_mvr else ordinal
        fluids = mvr_fluids if is_mvr else refrigerants
        fluid = fluids[min(fluid_index, len(fluids) - 1)]
        duty = float(Q_heat[ordinal] if args.is_heat_pumping else Q_cool[ordinal])
        stage = HPRSimulationStageRecord(
            stage_id=f"{role}-{ordinal}",
            ordinal=0,
            role=role,
            fluid_spec=fluid,
            evaporating_or_suction_temperature=float(T_evap[ordinal]),
            condensing_or_discharge_temperature=float(T_cond[ordinal]),
            source_approach_temperature=float(args.dtcont_hp),
            sink_approach_temperature=float(args.dtcont_hp),
            compressor_isentropic_efficiency=float(
                args.eta_mvr_comp if is_mvr else args.eta_comp
            ),
            motor_efficiency=(float(args.eta_motor) if is_mvr else None),
            useful_duty=duty,
            compressor_work=float(work[ordinal]),
            assumptions={"power_boundary": "compressor_only"},
        )
        loops.append(
            HPRSimulationLoopRecord(
                loop_id=f"{role}-loop-{ordinal}",
                ordinal=ordinal,
                loop_role=role,
                fluid_spec=fluid,
                nominal_duty=duty,
                nominal_work=float(work[ordinal]),
                stages=(stage,),
            )
        )
    return tuple(loops)


def _required_vector(value: object, attribute: str) -> np.ndarray:
    return np.asarray(getattr(value, attribute), dtype=float).reshape(-1)


def _cycle_or_state_vector(
    cycle: object,
    attribute: str,
    state_value: object,
    size: int,
    *,
    default: float | None = None,
) -> np.ndarray:
    try:
        value = getattr(cycle, attribute)
    except AttributeError:
        value = state_value
    if value is None:
        value = default
    return _broadcast_vector(
        np.asarray(value, dtype=float).reshape(-1),
        size,
        name=attribute,
    )


def _cycle_or_state_scalar(
    cycle: object,
    attribute: str,
    state_value: object,
    *,
    default: float,
) -> float:
    try:
        value = getattr(cycle, attribute)
    except AttributeError:
        value = state_value
    return _first_scalar(value, default=default)


def _broadcast_vector(value: np.ndarray, size: int, *, name: str) -> np.ndarray:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if value.size == 1:
        return np.full(size, float(value[0]), dtype=float)
    if value.size != size:
        raise ValueError(f"{name} must have one value or one value per HPR loop")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


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
