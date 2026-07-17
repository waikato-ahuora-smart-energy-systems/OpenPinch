"""Direct process-gas MVR component solver."""

from __future__ import annotations

import numpy as np
from CoolProp.CoolProp import PropsSI

from ....domain._stream.linearisation import get_piecewise_data_points
from ....domain.configuration import tol
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection
from ....domain.value import Value
from .models import (
    DirectGasMVROutputUnits as _DirectGasMVROutputUnits,
)
from .models import (
    DirectGasMVRSettings as _DirectGasMVRSettings,
)
from .models import (
    DirectGasMVRStageResult as _DirectGasMVRStageResult,
)
from .models import (
    DirectGasMVRStreamSolveResult as _DirectGasMVRStreamSolveResult,
)
from .thermodynamics import (
    actual_outlet_at_pressure as _actual_outlet_at_pressure,
)
from .thermodynamics import (
    build_cooling_temperature_enthalpy_curve as _build_cooling_th_curve,
)
from .thermodynamics import (
    find_pressure_for_actual_discharge as _find_pressure_for_actual_discharge,
)
from .thermodynamics import source_enthalpies as _source_enthalpies
from .thermodynamics import (
    state_property_at_temperature_pressure as _state_property_at_temperature_pressure,
)
from .units import convert_value as _convert_value
from .units import profile_to_output_units as _profile_to_output_units
from .units import source_output_units as _source_output_units
from .units import stage_mass_flow as _stage_mass_flow
from .units import stage_pressure_to_pascal as _stage_pressure_to_pa
from .units import to_degrees_celsius as _to_deg_c
from .units import to_kelvin as _to_kelvin
from .units import to_kilopascal as _to_kpa
from .units import to_kilowatt as _to_kw
from .units import to_kj_per_kg as _to_kj_per_kg
from .units import to_pascal as _to_pascal
from .units import to_watt as _to_watt
from .units import value_at_index as _value

__all__ = [
    "coerce_positive_mvr_stage_count",
    "solve_direct_gas_mvr_stream",
]

DEFAULT_MVR_STAGE_T_SAT_LIFT = 20.0
DEFAULT_MVR_COMP_EFFICIENCY = 0.7
DEFAULT_MOTOR_EFFICIENCY = 0.95
DEFAULT_DIRECT_MVR_STAGES = 1
DEFAULT_TEMPERATURE_UNIT = "degC"
DEFAULT_PRESSURE_UNIT = "kPa"
DEFAULT_ENTHALPY_UNIT = "kJ/kg"
DEFAULT_HEAT_FLOW_UNIT = "kW"


def solve_direct_gas_mvr_stream(
    stream: Stream,
    *,
    settings: _DirectGasMVRSettings,
    idx: int = 0,
) -> _DirectGasMVRStreamSolveResult:
    """Solve direct gas MVR replacement streams for one source stream and period."""
    fluid = str(stream.fluid_name)
    t_supply = _value(stream.supply_temperature, idx, unit="degC")
    t_target = _value(stream.target_temperature, idx, unit="degC")
    p_supply = _value(stream.supply_pressure, idx, unit="kPa")
    heat_flow = _value(stream.heat_flow, idx, unit="kW")
    if heat_flow is None or heat_flow <= tol:
        raise ValueError(f"MVR source stream {stream.name!r} requires positive duty.")
    if p_supply is None:
        raise ValueError(f"MVR source stream {stream.name!r} requires p_supply.")
    if t_supply is None or t_target is None or t_supply <= t_target:
        raise ValueError(
            f"MVR source stream {stream.name!r} must cool from supply to target."
        )

    h_supply, h_target = _source_enthalpies(
        stream,
        fluid,
        p_supply,
        t_supply,
        t_target,
        idx,
    )
    delta_h = h_supply - h_target
    if delta_h <= tol:
        raise ValueError(
            f"MVR source stream {stream.name!r} has no positive enthalpy drop."
        )

    m_dot = _to_watt(heat_flow) / delta_h
    if m_dot <= tol:
        raise ValueError(
            f"MVR source stream {stream.name!r} has non-positive mass flow."
        )

    replacement_streams = StreamCollection()
    stage_results: list[_DirectGasMVRStageResult] = []
    p_in = _to_pascal(p_supply)
    t_in = t_supply
    compression_target = _resolve_compression_target(settings)
    n_mvr = coerce_positive_mvr_stage_count(settings.n_stages)
    output_units = _source_output_units(stream)

    eta_comp = settings.eta_mvr_comp
    eta_motor = settings.eta_motor
    liquid_injection = settings.liquid_injection
    for stage_idx in range(n_mvr):
        stage = _solve_compression_stage(
            fluid=fluid,
            source_stream=stream.name,
            stage_index=stage_idx + 1,
            m_dot=m_dot,
            p_in=p_in,
            t_in=t_in,
            t_target=t_target,
            compression_target=compression_target,
            eta_comp=eta_comp,
            eta_motor=eta_motor,
            liquid_injection=liquid_injection,
            dt_diff_max=settings.dt_diff_max,
            output_units=output_units,
        )
        replacement_streams.add(_stage_to_stream(stream, stage, idx=idx))
        stage_results.append(stage)
        p_in = _stage_pressure_to_pa(stage.p_out, stage.pressure_unit)
        t_in = t_target
        m_dot = stage.hot_mass_flow

    return _DirectGasMVRStreamSolveResult(
        replacement_streams=replacement_streams,
        stage_results=stage_results,
    )


def _resolve_compression_target(
    settings: _DirectGasMVRSettings,
) -> tuple[str, float]:
    has_stage_t_lift = settings.mvr_stage_t_lift is not None
    has_pressure_ratio = settings.mvr_stage_pressure_ratio is not None
    if has_stage_t_lift and has_pressure_ratio:
        raise ValueError(
            "Specify either mvr_stage_t_lift or mvr_stage_pressure_ratio, not both."
        )
    if has_pressure_ratio:
        pressure_ratio = float(settings.mvr_stage_pressure_ratio)
        if pressure_ratio <= 1.0:
            raise ValueError("mvr_stage_pressure_ratio must be greater than 1.0.")
        return "pressure_ratio", pressure_ratio

    stage_t_lift = (
        DEFAULT_MVR_STAGE_T_SAT_LIFT
        if settings.mvr_stage_t_lift is None
        else float(settings.mvr_stage_t_lift)
    )
    if stage_t_lift <= 0.0:
        raise ValueError("mvr_stage_t_lift must be positive.")
    return "stage_t_lift", stage_t_lift


def coerce_positive_mvr_stage_count(value, *, context: str = "Direct gas MVR") -> int:
    """Return a validated integer direct-MVR stage count."""
    message = f"{context} requires n_stages to be a positive integer."
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(message)
    stage_count = int(numeric_value)
    if stage_count < 1:
        raise ValueError(message)
    return stage_count


def _solve_compression_stage(
    *,
    fluid: str,
    source_stream: str,
    stage_index: int,
    m_dot: float,
    p_in: float,
    t_in: float,
    t_target: float,
    compression_target: tuple[str, float],
    eta_comp: float,
    eta_motor: float,
    liquid_injection: bool,
    dt_diff_max: float,
    output_units: _DirectGasMVROutputUnits,
) -> _DirectGasMVRStageResult:
    if not (0.0 < eta_comp <= 1.0) or not (0.0 < eta_motor <= 1.0):
        raise ValueError("eta_mvr_comp and eta_motor must be in the interval (0, 1].")

    t_in_k = _to_kelvin(t_in)
    h_in = _state_property_at_temperature_pressure(
        fluid=fluid,
        output="H",
        t_c=t_in,
        p_kpa=_to_kpa(p_in),
        saturated_quality=1.0,
    )
    s_in = _state_property_at_temperature_pressure(
        fluid=fluid,
        output="S",
        t_c=t_in,
        p_kpa=_to_kpa(p_in),
        saturated_quality=1.0,
    )

    target_type, target_value = compression_target
    if target_type == "pressure_ratio":
        p_out = p_in * target_value
        h_out, t_discharge_k = _actual_outlet_at_pressure(
            fluid,
            p_out,
            h_in,
            s_in,
            eta_comp,
        )
    else:
        target_discharge_k = t_in_k + target_value
        p_out, h_out, t_discharge_k = _find_pressure_for_actual_discharge(
            fluid=fluid,
            inlet_pressure=p_in,
            inlet_enthalpy=h_in,
            inlet_entropy=s_in,
            target_discharge_kelvin=target_discharge_k,
            compressor_efficiency=eta_comp,
        )
    work = _to_kw(m_dot * (h_out - h_in) / eta_motor)
    h_hot_supply = h_out
    t_hot_supply_k = t_discharge_k
    hot_m_dot = m_dot
    q_liquid_injection = 0.0
    liquid_injection_ratio = 0.0
    injection_applied = False
    h_stage_target = PropsSI("H", "T", _to_kelvin(t_target), "P", p_out, fluid)
    if liquid_injection:
        try:
            h_sat_vap = PropsSI("H", "P", p_out, "Q", 1.0, fluid)
            t_sat_vap = PropsSI("T", "P", p_out, "Q", 1.0, fluid)
        except Exception:
            h_sat_vap = np.nan
            t_sat_vap = np.nan
        if (
            np.isfinite(h_sat_vap)
            and np.isfinite(t_sat_vap)
            and _to_kelvin(t_target) < t_sat_vap < t_discharge_k
            and h_sat_vap < h_out
            and h_stage_target < h_sat_vap
        ):
            q_injection_evap = h_sat_vap - h_stage_target
            liquid_injection_ratio = (h_out - h_sat_vap) / q_injection_evap
            hot_m_dot = m_dot * (1.0 + liquid_injection_ratio)
            q_liquid_injection = _to_kw(
                m_dot * liquid_injection_ratio * q_injection_evap
            )
            h_hot_supply = h_sat_vap
            t_hot_supply_k = t_sat_vap
            injection_applied = True

    heat_flow = _to_kw(hot_m_dot * max(h_hot_supply - h_stage_target, 0.0))
    th_curve_si = _build_cooling_th_curve(
        fluid=fluid,
        outlet_pressure=p_out,
        hot_supply_enthalpy=h_hot_supply,
        target_enthalpy=h_stage_target,
    )
    linearised_profile_si = get_piecewise_data_points(
        curve=th_curve_si,
        is_hot_stream=True,
        dt_diff_max=dt_diff_max,
    )
    th_curve = _profile_to_output_units(th_curve_si, output_units)
    linearised_profile = _profile_to_output_units(linearised_profile_si, output_units)
    return _DirectGasMVRStageResult(
        source_stream=source_stream,
        stage_index=stage_index,
        p_in=_convert_value(_to_kpa(p_in), "kPa", output_units.pressure),
        p_out=_convert_value(_to_kpa(p_out), "kPa", output_units.pressure),
        t_in=_convert_value(t_in, "degC", output_units.temperature),
        t_discharge=_convert_value(
            _to_deg_c(t_discharge_k),
            "degC",
            output_units.temperature,
        ),
        t_hot_supply=_convert_value(
            _to_deg_c(t_hot_supply_k),
            "degC",
            output_units.temperature,
        ),
        t_target=_convert_value(t_target, "degC", output_units.temperature),
        heat_flow=_convert_value(heat_flow, "kW", output_units.heat_flow),
        work=_convert_value(work, "kW", output_units.heat_flow),
        h_hot_supply=_convert_value(
            _to_kj_per_kg(h_hot_supply),
            "kJ/kg",
            output_units.enthalpy,
        ),
        h_target=_convert_value(
            _to_kj_per_kg(h_stage_target),
            "kJ/kg",
            output_units.enthalpy,
        ),
        th_curve=th_curve,
        linearised_profile=linearised_profile,
        q_liquid_injection=_convert_value(
            q_liquid_injection,
            "kW",
            output_units.heat_flow,
        ),
        liquid_injection_applied=injection_applied,
        temperature_unit=output_units.temperature,
        pressure_unit=output_units.pressure,
        enthalpy_unit=output_units.enthalpy,
        heat_flow_unit=output_units.heat_flow,
        source_mass_flow=m_dot,
        hot_mass_flow=hot_m_dot,
        liquid_injection_ratio=liquid_injection_ratio,
    )


def _stage_to_stream(
    source: Stream,
    stage: _DirectGasMVRStageResult,
    *,
    idx: int,
) -> Stream:
    from ....domain._stream.linearisation import build_segmented_stream_from_profile

    return build_segmented_stream_from_profile(
        name=f"{source.name}_direct_MVR_H{stage.stage_index}",
        profile=stage.linearised_profile,
        heat_scale=_stage_mass_flow(stage),
        heat_unit=stage.heat_flow_unit,
        is_hot_stream=True,
        supply_pressure=Value(stage.p_out, stage.pressure_unit),
        target_pressure=Value(stage.p_out, stage.pressure_unit),
        delta_t_contribution=_value(source.delta_t_contribution, idx) or 0.0,
        heat_transfer_coefficient=(
            _value(source.heat_transfer_coefficient, idx) or 1.0
        ),
        is_process_stream=True,
        fluid_name=source.fluid_name,
        fluid_phase=source.fluid_phase,
    )
