"""Direct process-gas MVR component solver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from CoolProp.CoolProp import PropsSI

from ....classes.stream import Stream
from ....classes.stream_collection import StreamCollection
from ....classes.value import Value
from ....lib.config import tol
from ....utils.stream_linearisation import get_piecewise_data_points

__all__ = [
    "DirectGasMVROutputUnits",
    "DirectGasMVRSettings",
    "DirectGasMVRStageResult",
    "DirectGasMVRStreamSolveResult",
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


@dataclass(frozen=True)
class DirectGasMVROutputUnits:
    """Units used for public direct-MVR outputs."""

    temperature: str = DEFAULT_TEMPERATURE_UNIT
    pressure: str = DEFAULT_PRESSURE_UNIT
    enthalpy: str = DEFAULT_ENTHALPY_UNIT
    heat_flow: str = DEFAULT_HEAT_FLOW_UNIT


@dataclass
class DirectGasMVRSettings:
    """User-facing settings for one direct gas MVR solve."""

    n_stages: int = DEFAULT_DIRECT_MVR_STAGES
    mvr_stage_t_lift: float | None = None
    mvr_stage_pressure_ratio: float | None = None
    liquid_injection: bool = False
    eta_mvr_comp: float = DEFAULT_MVR_COMP_EFFICIENCY
    eta_motor: float = DEFAULT_MOTOR_EFFICIENCY
    dt_diff_max: float = 0.1


@dataclass
class DirectGasMVRStageResult:
    """Solved accounting for one direct gas MVR stage."""

    source_stream: str
    stage_index: int
    p_in: float
    p_out: float
    t_in: float
    t_discharge: float
    t_hot_supply: float
    t_target: float
    heat_flow: float
    work: float
    h_hot_supply: float
    h_target: float
    th_curve: np.ndarray = field(repr=False)
    linearised_profile: np.ndarray = field(repr=False)
    q_liquid_injection: float = 0.0
    liquid_injection_applied: bool = False
    temperature_unit: str = DEFAULT_TEMPERATURE_UNIT
    pressure_unit: str = DEFAULT_PRESSURE_UNIT
    enthalpy_unit: str = DEFAULT_ENTHALPY_UNIT
    heat_flow_unit: str = DEFAULT_HEAT_FLOW_UNIT
    source_mass_flow: float = 0.0
    hot_mass_flow: float = 0.0
    liquid_injection_ratio: float = 0.0


@dataclass
class DirectGasMVRStreamSolveResult:
    """Solved direct gas MVR streams for one source stream at one period index."""

    replacement_streams: StreamCollection
    stage_results: list[DirectGasMVRStageResult] = field(default_factory=list)


def solve_direct_gas_mvr_stream(
    stream: Stream,
    *,
    settings: DirectGasMVRSettings,
    idx: int = 0,
) -> DirectGasMVRStreamSolveResult:
    """Solve direct gas MVR replacement streams for one source stream and period."""
    fluid = str(stream.fluid_name)
    t_supply = _value(stream.t_supply, idx, unit="degC")
    t_target = _value(stream.t_target, idx, unit="degC")
    p_supply = _value(stream.p_supply, idx, unit="kPa")
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
    stage_results: list[DirectGasMVRStageResult] = []
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

    return DirectGasMVRStreamSolveResult(
        replacement_streams=replacement_streams,
        stage_results=stage_results,
    )


def _resolve_compression_target(
    settings: DirectGasMVRSettings,
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


def _source_enthalpies(
    stream: Stream,
    fluid: str,
    p_supply_kpa: float,
    t_supply_c: float,
    t_target_c: float,
    idx: int,
) -> tuple[float, float]:
    h_supply = _value(stream.h_supply, idx, unit="kJ/kg")
    h_target = _value(stream.h_target, idx, unit="kJ/kg")
    if h_supply is not None and h_target is not None:
        return _to_j_per_kg(h_supply), _to_j_per_kg(h_target)
    return (
        _state_property_at_temperature_pressure(
            fluid=fluid,
            output="H",
            t_c=t_supply_c,
            p_kpa=p_supply_kpa,
            saturated_quality=1.0,
        ),
        _state_property_at_temperature_pressure(
            fluid=fluid,
            output="H",
            t_c=t_target_c,
            p_kpa=p_supply_kpa,
            saturated_quality=0.0,
        ),
    )


def _state_property_at_temperature_pressure(
    *,
    fluid: str,
    output: str,
    t_c: float,
    p_kpa: float,
    saturated_quality: float,
) -> float:
    t_k = _to_kelvin(t_c)
    p_pa = _to_pascal(p_kpa)
    try:
        return PropsSI(output, "T", t_k, "P", p_pa, fluid)
    except ValueError:
        try:
            t_sat = PropsSI("T", "P", p_pa, "Q", saturated_quality, fluid)
        except Exception:
            raise
        if abs(t_k - t_sat) > 0.05:
            raise
        return PropsSI(output, "P", p_pa, "Q", saturated_quality, fluid)


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
    output_units: DirectGasMVROutputUnits,
) -> DirectGasMVRStageResult:
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
            p_in=p_in,
            h_in=h_in,
            s_in=s_in,
            target_discharge_k=target_discharge_k,
            eta_comp=eta_comp,
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
        p_out=p_out,
        h_hot_supply=h_hot_supply,
        h_target=h_stage_target,
    )
    linearised_profile_si = get_piecewise_data_points(
        curve=th_curve_si,
        is_hot_stream=True,
        dt_diff_max=dt_diff_max,
    )
    th_curve = _profile_to_output_units(th_curve_si, output_units)
    linearised_profile = _profile_to_output_units(linearised_profile_si, output_units)
    return DirectGasMVRStageResult(
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


def _find_pressure_for_actual_discharge(
    *,
    fluid: str,
    p_in: float,
    h_in: float,
    s_in: float,
    target_discharge_k: float,
    eta_comp: float,
) -> tuple[float, float, float]:
    p_lo = p_in * 1.0001
    p_hi = p_in * 1.2
    h_hi, t_hi = _actual_outlet_at_pressure(fluid, p_hi, h_in, s_in, eta_comp)
    for _ in range(80):
        if t_hi >= target_discharge_k:
            break
        p_hi *= 1.5
        h_hi, t_hi = _actual_outlet_at_pressure(fluid, p_hi, h_in, s_in, eta_comp)
    else:
        raise ValueError("Could not find a feasible MVR discharge pressure.")

    for _ in range(80):
        p_mid = 0.5 * (p_lo + p_hi)
        _, t_mid = _actual_outlet_at_pressure(fluid, p_mid, h_in, s_in, eta_comp)
        if t_mid < target_discharge_k:
            p_lo = p_mid
        else:
            p_hi = p_mid
    h_out, t_out = _actual_outlet_at_pressure(fluid, p_hi, h_in, s_in, eta_comp)
    return p_hi, h_out, t_out


def _actual_outlet_at_pressure(
    fluid: str,
    p_out: float,
    h_in: float,
    s_in: float,
    eta_comp: float,
) -> tuple[float, float]:
    h_out_s = PropsSI("H", "P", p_out, "S", s_in, fluid)
    h_out = h_in + (h_out_s - h_in) / eta_comp
    t_out = PropsSI("T", "P", p_out, "H", h_out, fluid)
    return h_out, t_out


def _build_cooling_th_curve(
    *,
    fluid: str,
    p_out: float,
    h_hot_supply: float,
    h_target: float,
) -> np.ndarray:
    if h_hot_supply <= h_target:
        raise ValueError("Compressed MVR stage has no heat-release enthalpy range.")

    h_values = _profile_enthalpy_values(
        fluid=fluid,
        p_out=p_out,
        h_hot_supply=h_hot_supply,
        h_target=h_target,
    )
    points = [[h, _to_deg_c(PropsSI("T", "P", p_out, "H", h, fluid))] for h in h_values]
    return np.asarray(points, dtype=float)


def _profile_enthalpy_values(
    *,
    fluid: str,
    p_out: float,
    h_hot_supply: float,
    h_target: float,
) -> np.ndarray:
    breakpoints = [float(h_hot_supply), float(h_target)]
    try:
        p_crit = PropsSI("PCRIT", fluid)
        if p_out < p_crit:
            h_sat_vap = PropsSI("H", "P", p_out, "Q", 1.0, fluid)
            h_sat_liq = PropsSI("H", "P", p_out, "Q", 0.0, fluid)
            for h_sat in (h_sat_vap, h_sat_liq):
                if h_target < h_sat < h_hot_supply:
                    breakpoints.append(float(h_sat))
    except Exception:
        pass

    ordered = sorted(set(breakpoints), reverse=True)
    h_values: list[float] = []
    for upper, lower in zip(ordered[:-1], ordered[1:]):
        segment = np.linspace(upper, lower, 31)
        if h_values:
            segment = segment[1:]
        h_values.extend(float(h) for h in segment)
    return np.asarray(h_values, dtype=float)


def _stage_to_stream(
    source: Stream,
    stage: DirectGasMVRStageResult,
    *,
    idx: int,
) -> Stream:
    from ....utils.stream_linearisation import build_segmented_stream_from_profile

    return build_segmented_stream_from_profile(
        name=f"{source.name}_direct_MVR_H{stage.stage_index}",
        profile=stage.linearised_profile,
        heat_scale=_stage_mass_flow(stage),
        heat_unit=stage.heat_flow_unit,
        is_hot_stream=True,
        p_supply=Value(stage.p_out, stage.pressure_unit),
        p_target=Value(stage.p_out, stage.pressure_unit),
        dt_cont=_value(source.dt_cont, idx) or 0.0,
        htc=_value(source.htc, idx) or 1.0,
        is_process_stream=True,
        fluid_name=source.fluid_name,
        fluid_phase=source.fluid_phase,
    )


def _stage_mass_flow(stage: DirectGasMVRStageResult) -> float:
    if stage.hot_mass_flow > tol:
        return stage.hot_mass_flow
    delta_h = _enthalpy_delta_to_j_per_kg(
        stage.h_hot_supply,
        stage.h_target,
        stage.enthalpy_unit,
    )
    if delta_h <= tol:
        return 0.0
    heat_flow_w = Value(stage.heat_flow, stage.heat_flow_unit).to("W").value
    return heat_flow_w / delta_h


def _stage_pressure_to_pa(value: float, pressure_unit: str) -> float:
    return float(Value(value, pressure_unit).to("Pa").value)


def _value(value, idx: int, *, unit: str | None = None) -> float | None:
    if value is None:
        return None
    try:
        selected = value[idx]
    except Exception:
        selected = value
    if isinstance(selected, Value):
        if unit is not None:
            selected = selected.to(unit)
        return float(selected.value)
    return float(selected)


def _to_kelvin(t_c: float) -> float:
    return float(Value(t_c, "degC").to("K").value)


def _to_deg_c(t_k: float) -> float:
    return float(Value(t_k, "K").to("degC").value)


def _to_pascal(p_kpa: float) -> float:
    return float(Value(p_kpa, "kPa").to("Pa").value)


def _to_kpa(p_pa: float) -> float:
    return float(Value(p_pa, "Pa").to("kPa").value)


def _to_watt(power_kw: float) -> float:
    return float(Value(power_kw, "kW").to("W").value)


def _to_kw(power_w: float) -> float:
    return float(Value(power_w, "W").to("kW").value)


def _to_j_per_kg(h_kj_per_kg: float) -> float:
    return float(Value(h_kj_per_kg, "kJ/kg").to("J/kg").value)


def _to_kj_per_kg(h_j_per_kg: float) -> float:
    return float(Value(h_j_per_kg, "J/kg").to("kJ/kg").value)


def _profile_to_output_units(
    profile: np.ndarray,
    output_units: DirectGasMVROutputUnits,
) -> np.ndarray:
    """Convert profile columns to the public stage-output units."""
    converted = np.asarray(profile, dtype=float).copy()
    converted[:, 0] = Value(converted[:, 0], "J/kg").to(output_units.enthalpy).value
    converted[:, 1] = Value(converted[:, 1], "degC").to(output_units.temperature).value
    return converted


def _convert_value(value: float, source_unit: str, target_unit: str) -> float:
    return float(Value(value, source_unit).to(target_unit).value)


def _source_output_units(stream: Stream) -> DirectGasMVROutputUnits:
    return DirectGasMVROutputUnits(
        temperature=_stream_value_unit(stream.t_supply, DEFAULT_TEMPERATURE_UNIT),
        pressure=_stream_value_unit(stream.p_supply, DEFAULT_PRESSURE_UNIT),
        enthalpy=_stream_value_unit(
            stream.h_supply or stream.h_target,
            DEFAULT_ENTHALPY_UNIT,
        ),
        heat_flow=_stream_value_unit(stream.heat_flow, DEFAULT_HEAT_FLOW_UNIT),
    )


def _stream_value_unit(value, fallback: str) -> str:
    return str(getattr(value, "unit", None) or fallback)


def _enthalpy_delta_to_j_per_kg(
    h_start: float,
    h_end: float,
    enthalpy_unit: str,
) -> float:
    delta_h = abs(float(h_start) - float(h_end))
    return float(Value(delta_h, enthalpy_unit).to("J/kg").value)
