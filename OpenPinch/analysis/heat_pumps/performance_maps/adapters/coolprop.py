"""Default HPR point simulator backed by the existing CoolProp cycle."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version

from ...cycles.vapour_compression_cycle import VapourCompressionCycle
from ..errors import HprSimulatorFailure
from ..models import (
    HprMapGenerationContext,
    HprOperatingPoint,
    HprPointSimulation,
    HprSimulatorMetadata,
)


def _coolprop_version() -> str:
    try:
        return version("CoolProp")
    except PackageNotFoundError:
        return "unknown"


class CoolPropHprPointSimulator:
    """Fresh steady-state session delegating every point to the current cycle."""

    def __init__(self) -> None:
        self._context: HprMapGenerationContext | None = None
        self._state = "fresh"

    def prepare(self, context: HprMapGenerationContext) -> HprSimulatorMetadata:
        if self._state != "fresh":
            raise HprSimulatorFailure(
                "prepare_failed",
                "CoolProp simulator can only be prepared once",
                session_fatal=True,
            )
        if context.basis.simulation_backend.strip().lower() != "coolprop":
            raise HprSimulatorFailure(
                "prepare_failed",
                "CoolProp simulator received a context for a different backend",
                session_fatal=True,
            )
        self._context = context
        self._state = "prepared"
        return HprSimulatorMetadata(
            backend="coolprop",
            engine_version=_coolprop_version(),
            model_id=context.basis.model_id,
            characteristic_set_id=context.characteristic_set_id,
            design_converged=True,
            design_details={
                "load_model": "steady_state_mass_flow_scaling",
                "property_backend": context.working_fluid.property_backend,
                "saturation_anchor": context.working_fluid.saturation_anchor,
                "cycle_input_power_unit": "W",
                "map_output_power_unit": "kW",
            },
        )

    def simulate(self, point: HprOperatingPoint) -> HprPointSimulation:
        if self._state != "prepared" or self._context is None:
            raise HprSimulatorFailure(
                "point_exception",
                "CoolProp simulator is not in a prepared state",
                session_fatal=True,
            )
        context = self._context
        is_heat_pump = context.basis.mode == "heat_pump"
        cycle = VapourCompressionCycle()
        try:
            work = cycle.solve(
                T_evap=point.evaporating_temperature,
                T_cond=point.condensing_temperature,
                dtcont=min(
                    context.basis.source_approach_temperature,
                    context.basis.sink_approach_temperature,
                ),
                dT_superheat=context.basis.superheat,
                dT_subcool=context.basis.subcooling,
                eta_comp=context.basis.compressor_isentropic_efficiency,
                refrigerant=context.working_fluid.source_spec,
                dT_ihx_gas_side=(context.basis.internal_hx_gas_temperature_change),
                Q_heat=(
                    point.requested_useful_duty * 1_000.0 if is_heat_pump else None
                ),
                Q_cool=(
                    None if is_heat_pump else point.requested_useful_duty * 1_000.0
                ),
                is_heat_pump=is_heat_pump,
            )
        except Exception as exc:
            raise HprSimulatorFailure(
                "point_exception",
                "CoolProp cycle raised at the requested operating point",
                session_fatal=False,
                cause=exc,
            ) from exc

        values = (cycle.Q_evap, cycle.Q_cond, work)
        if not cycle.solved or any(
            value is None or not math.isfinite(float(value)) for value in values
        ):
            raise HprSimulatorFailure(
                "non_converged",
                "CoolProp cycle did not return a finite solved operating point",
                session_fatal=False,
            )
        return HprPointSimulation(
            q_source=float(cycle.Q_evap) / 1_000.0,
            q_sink=float(cycle.Q_cond) / 1_000.0,
            compressor_power=float(work) / 1_000.0,
            converged=True,
            engine_details={"engine": "CoolProp"},
        )

    def close(self) -> None:
        if self._state == "closed":
            raise HprSimulatorFailure(
                "cleanup_failed",
                "CoolProp simulator was closed more than once",
                session_fatal=True,
            )
        self._context = None
        self._state = "closed"


__all__ = ["CoolPropHprPointSimulator"]
