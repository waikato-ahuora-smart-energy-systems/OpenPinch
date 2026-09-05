"""Pure compatibility validation and target-to-map-basis construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ....contracts.hpr import HprTargetSimulationRecord
from ....domain.targets import (
    DirectHeatPumpTarget,
    DirectRefrigerationTarget,
    HeatPumpTargetBase,
    IndirectHeatPumpTarget,
    IndirectRefrigerationTarget,
    SubzoneAggregateTarget,
)
from .models import HprTargetMapBasis


class HprPerformanceMapCompatibilityError(ValueError):
    """Typed rejection of a target that cannot define a scalar HPR map basis."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def build_hpr_target_map_basis(target: object) -> HprTargetMapBasis:
    """Return one detached map basis using only an existing successful target."""
    if isinstance(target, SubzoneAggregateTarget):
        _reject("aggregate_target", "Aggregate targets cannot define one HPR map.")
    if not isinstance(target, HeatPumpTargetBase):
        _reject(
            "unsupported_target_type",
            "HPR performance maps require a heat-pump or refrigeration target.",
        )
    if not bool(target.hpr_success):
        _reject("failed_target", "A failed HPR target cannot define a performance map.")

    details = target.hpr_details
    if _read(details, "period_outputs"):
        _reject(
            "aggregate_target",
            "Shared-vector multi-period HPR targets cannot define one scalar map.",
        )
    record = _read(details, "target_simulation_record")
    if record is None:
        _reject(
            "missing_simulation_record",
            "The HPR target does not carry a winning simulation record.",
        )
    if not isinstance(record, HprTargetSimulationRecord):
        _reject(
            "record_inconsistency",
            "The HPR target simulation record has an invalid type.",
        )
    if record.evaporator_count != 1 or record.condenser_count != 1:
        _reject(
            "multi_port_topology",
            "Performance-map schema 1.0 requires one evaporator and one condenser.",
        )
    if record.cycle_id != "single_stage_vapour_compression":
        _reject(
            "multi_port_topology",
            "Performance-map schema 1.0 requires a single-stage vapour cycle.",
        )

    expected_mode = _target_mode(target)
    if (
        target.hpr_simulation_backend != record.simulation_backend
        or _read(details, "simulation_backend") != record.simulation_backend
        or expected_mode != record.mode
        or (target.period_id is not None and target.period_id != record.period_id)
    ):
        _reject(
            "record_inconsistency",
            "Target identity and winning simulation record do not match.",
        )
    _validate_scalar_details(details, record)

    return HprTargetMapBasis(
        target_id=target.name,
        mode=record.mode,
        simulation_backend=record.simulation_backend,
        cycle_id=record.cycle_id,
        model_id=record.model_id,
        refrigerant_spec=record.refrigerant_spec,
        nominal_evaporating_temperature=record.nominal_evaporating_temperature,
        nominal_condensing_temperature=record.nominal_condensing_temperature,
        nominal_useful_duty=record.nominal_useful_duty,
        source_approach_temperature=record.source_approach_temperature,
        sink_approach_temperature=record.sink_approach_temperature,
        compressor_isentropic_efficiency=(record.compressor_isentropic_efficiency),
        superheat=record.superheat,
        subcooling=record.subcooling,
        internal_hx_gas_temperature_change=(record.internal_hx_gas_temperature_change),
        source_provenance={
            "engine_version": record.engine_version,
            "power_boundary": record.power_boundary,
            "period_id": record.period_id,
            "evaporator_count": record.evaporator_count,
            "condenser_count": record.condenser_count,
            "assumptions": deepcopy(record.assumptions),
        },
    )


def _target_mode(target: HeatPumpTargetBase) -> str:
    if isinstance(target, DirectHeatPumpTarget | IndirectHeatPumpTarget):
        return "heat_pump"
    if isinstance(target, DirectRefrigerationTarget | IndirectRefrigerationTarget):
        return "refrigeration"
    _reject("unsupported_target_type", "Unsupported HPR target subtype.")


def _validate_scalar_details(details: Any, record: HprTargetSimulationRecord) -> None:
    expected = {
        "T_evap": record.nominal_evaporating_temperature,
        "T_cond": record.nominal_condensing_temperature,
        "Q_heat": (record.nominal_useful_duty if record.mode == "heat_pump" else None),
        "Q_cool": (
            record.nominal_useful_duty if record.mode == "refrigeration" else None
        ),
        "dT_superheat": record.superheat,
        "dT_subcool": record.subcooling,
    }
    for name, expected_value in expected.items():
        value = _read(details, name)
        if value is None or expected_value is None:
            continue
        try:
            array = np.asarray(value, dtype=float)
        except TypeError, ValueError:
            _reject("non_scalar_nominal_data", f"Target field {name} is not scalar.")
        if array.size != 1:
            _reject("non_scalar_nominal_data", f"Target field {name} is not scalar.")
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar):
            _reject("non_scalar_nominal_data", f"Target field {name} is not finite.")
        if not np.isclose(scalar, expected_value, rtol=1e-8, atol=1e-6):
            _reject(
                "record_inconsistency",
                f"Target field {name} does not match its winning record.",
            )


def _read(owner: object, name: str) -> Any:
    if isinstance(owner, dict):
        return owner.get(name)
    return getattr(owner, name, None)


def _reject(code: str, message: str):
    raise HprPerformanceMapCompatibilityError(code, message)


__all__ = [
    "HprPerformanceMapCompatibilityError",
    "build_hpr_target_map_basis",
]
