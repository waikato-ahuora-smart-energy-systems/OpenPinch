"""The components of solved heat pump machines, for capital costing.

Each machine is its compressors and its exchangers. Internal exchangers are
sized from their own temperatures: the IHX from its end temperatures, the
cascade exchanger at ``dt_cascade_hx``. The process-side exchangers are sized
interval by interval against the background profiles they exchange with: the
condensers against the closest part of the sink below them, the evaporators
against the closest part of the source above them, with the working fluid's
and the process streams' approaches added. Heat a machine rejects to, or draws
from, ambient air goes through an air cooler or heater, sized against the air
temperature with the air side's coefficient in series: the coldest part of the
condensing curve and the warmest part of the evaporating curve. Heat the cascade
puts on the wrong side of the process (penalised) has no exchanger.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import CoolProp.CoolProp as CoolProp
import numpy as np

from ....contracts.hpr import HeatPumpTargetInputs
from ....domain.configuration import tol
from ....domain.stream_collection import StreamCollection
from ...targeting.temperature_driving_force import get_temperature_driving_forces
from ._shared.air import ambient_sink_temperature, ambient_source_temperature
from .capital import CompressorDuty, MachineComponents, matched_area

__all__ = ["machine_components", "process_side_areas"]

# A curve interval cooler or warmer by less than this is a phase change.
_ISOTHERMAL_K = 0.05
# Bisection steps placing a heat pump curve on the background curve.
_BISECTIONS = 40
# A process duty below this share of a curve's heat is cascade round-off.
_NEGLIGIBLE_PROCESS_DUTY_FRACTION = 1e-4


def process_side_areas(
    hpr_streams: StreamCollection,
    args: HeatPumpTargetInputs,
    *,
    condensing_temperatures: Sequence[float],
    air_duties: tuple[float, float] = (0.0, 0.0),
    wrong_side_duties: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    """Return the condenser, evaporator, air cooler and air heater areas (m2).

    ``air_duties`` are the condenser heat (kW) rejected to air and the
    evaporator heat drawn from air. ``wrong_side_duties`` are the condenser
    heat below the process sink and the evaporator heat above the process
    source. The rest is exchanged with the process.
    """
    dtcont_hp = float(getattr(args, "dtcont_hp", 0.0))
    extra_dt = dtcont_hp + float(getattr(args, "process_dt_cont", 0.0))
    air_dt = dtcont_hp + float(getattr(args, "dt_env_cont", 0.0))
    Q_air_cooling, Q_air_heating = (max(float(q), 0.0) for q in air_duties)
    Q_below_sink, Q_above_source = (max(float(q), 0.0) for q in wrong_side_duties)

    def is_process_duty(duty: float, total: float) -> bool:
        return duty > max(tol, _NEGLIGIBLE_PROCESS_DUTY_FRACTION * total)

    hot = hpr_streams.get_hot_utility_streams()
    condenser_u = _condenser_coefficient(
        hot, args, condensing_temperatures=condensing_temperatures
    )

    def evaporator_u(t_h1, t_h2, t_c1, t_c2):
        isothermal = np.abs(t_c1 - t_c2) < _ISOTHERMAL_K
        return np.where(isothermal, args.u_two_phase, args.u_gas)

    def with_air(u):
        return lambda *t: 1.0 / (1.0 / u(*t) + 1.0 / args.u_air)

    condenser = air_cooler = 0.0
    if len(hot):
        T_hp, H_hp = _trim(*_composite(hot, args))
        total = float(H_hp[0])
        # From the cold end: the wrong-side heat, the air's, then the process's.
        Q_wrong = min(Q_below_sink, total)
        Q_air = min(Q_air_cooling, total - Q_wrong)
        T_sink, H_sink = _trim(
            np.asarray(args.T_cold, float),
            np.asarray(args.H_cold, float) - float(np.asarray(args.H_cold)[-1]),
        )
        if is_process_duty(total - Q_wrong - Q_air, total):
            condenser = _closest_area(
                _window(T_hp, H_hp, Q_wrong + Q_air, total),
                (T_sink, H_sink),
                heat_pump_is_hot=True,
                extra_dt=extra_dt,
                heat_transfer_coefficient=condenser_u,
            )
        if Q_air > tol:
            T_air = ambient_sink_temperature(args)
            air_cooler = matched_area(
                *_window(T_hp, H_hp, Q_wrong, Q_wrong + Q_air),
                np.array([T_air, T_air]),
                np.array([Q_air, 0.0]),
                extra_dt=air_dt,
                heat_transfer_coefficient=with_air(condenser_u),
            )

    evaporator = air_heater = 0.0
    cold = hpr_streams.get_cold_utility_streams()
    if len(cold):
        T_hp, H_hp = _trim(*_composite(cold, args))
        total = float(H_hp[0])
        # From the hot end: the wrong-side heat, the air's, then the process's.
        Q_wrong = min(Q_above_source, total)
        Q_air = min(Q_air_heating, total - Q_wrong)
        H_src = np.asarray(args.H_hot, float)
        T_src, H_src = _trim(np.asarray(args.T_hot, float), H_src - H_src[-1])
        if is_process_duty(total - Q_wrong - Q_air, total):
            evaporator = _closest_area(
                _window(T_hp, H_hp, 0.0, total - Q_wrong - Q_air),
                (T_src, H_src),
                heat_pump_is_hot=False,
                extra_dt=extra_dt,
                heat_transfer_coefficient=evaporator_u,
            )
        if Q_air > tol:
            T_air = ambient_source_temperature(args)
            air_heater = matched_area(
                np.array([T_air, T_air]),
                np.array([Q_air, 0.0]),
                *_window(T_hp, H_hp, total - Q_wrong - Q_air, total - Q_wrong),
                extra_dt=air_dt,
                heat_transfer_coefficient=with_air(evaporator_u),
            )
    return condenser, evaporator, air_cooler, air_heater


def machine_components(
    machines: Sequence[Sequence[Any]],
    *,
    process_areas: tuple[float, float, float, float],
    args: HeatPumpTargetInputs,
) -> tuple[MachineComponents, ...]:
    """Return each machine's compressors and exchanger areas.

    ``machines`` lists each machine's solved stages. ``process_areas`` are the
    condenser, evaporator, air cooler and air heater areas of
    :func:`process_side_areas`; the condensing side's are shared between
    machines by their process heating, the evaporating side's by their cooling.
    """
    condenser_area, evaporator_area, air_cooler_area, air_heater_area = process_areas
    heat = [sum(_process_heat(stage) for stage in stages) for stages in machines]
    cool = [sum(_process_cooling(stage) for stage in stages) for stages in machines]
    total_heat, total_cool = sum(heat), sum(cool)
    result = []
    for stages, Q_heat, Q_cool in zip(machines, heat, cool, strict=True):
        areas = [area for stage in stages for area in _internal_areas(stage, args)]
        if total_heat > tol:
            areas.append(condenser_area * Q_heat / total_heat)
            areas.append(air_cooler_area * Q_heat / total_heat)
        if total_cool > tol:
            areas.append(evaporator_area * Q_cool / total_cool)
            areas.append(air_heater_area * Q_cool / total_cool)
        result.append(
            MachineComponents(
                compressors=tuple(_compressor(stage) for stage in stages),
                exchanger_areas=tuple(area for area in areas if area > 0.0),
            )
        )
    return tuple(result)


def _condenser_coefficient(
    streams: StreamCollection,
    args: HeatPumpTargetInputs,
    *,
    condensing_temperatures: Sequence[float],
):
    """Return the condensing side's ``U`` for each interval of a match.

    A (near) isothermal interval condenses. A sloped section is gas above its
    own stage's condensing temperature and liquid below it, so a hotter
    stage's subcooled liquid stays liquid. Sections of several stages in one
    interval add as conductances: ``1/U`` is their heat-weighted ``1/U``.
    A stage condenses at its isothermal section; a transcritical stage, which
    has none, at the highest condensing temperature within its range.
    """
    sections = []
    for stage in _stream_segments(streams, args):
        T_cond = _stage_condensing_temperature(stage, condensing_temperatures)
        for hi, lo, heat in stage:
            if hi - lo >= _ISOTHERMAL_K:
                u = args.u_gas if 0.5 * (hi + lo) > T_cond else args.u_liquid
                sections.append((hi, lo, heat / (hi - lo), u))

    def coefficient(t_h1, t_h2, t_c1, t_c2):
        middle = 0.5 * (np.asarray(t_h1, float) + np.asarray(t_h2, float))
        heat = np.zeros_like(middle)
        resistance = np.zeros_like(middle)
        for hi, lo, density, u in sections:
            inside = (middle > lo) & (middle < hi)
            heat += np.where(inside, density, 0.0)
            resistance += np.where(inside, density / u, 0.0)
        sloped = np.where(
            heat > 0.0, heat / np.where(resistance > 0.0, resistance, 1.0), 1.0
        )
        isothermal = (np.abs(np.asarray(t_h1) - np.asarray(t_h2)) < _ISOTHERMAL_K) | (
            heat <= 0.0
        )
        return np.where(isothermal, args.u_two_phase, sloped)

    return coefficient


def _stage_condensing_temperature(
    stage: list[tuple[float, float, float]], condensing_temperatures: Sequence[float]
) -> float:
    """A stage's condensing temperature, from its own sections."""
    isothermal = [hi for hi, lo, _ in stage if hi - lo < _ISOTHERMAL_K]
    if isothermal:
        return max(isothermal)
    top = max(hi for hi, _, _ in stage)
    bottom = min(lo for _, lo, _ in stage)
    within = [t for t in condensing_temperatures if bottom <= t <= top]
    return max(within, default=min(condensing_temperatures, default=np.inf))


def _stream_segments(
    streams: StreamCollection, args: HeatPumpTargetInputs
) -> list[list[tuple[float, float, float]]]:
    """Each stream's linear segments as ``(hot end, cold end, heat)``."""
    period = int(getattr(args, "period_idx", 0) or 0)

    def value(quantity) -> float:
        values = np.ravel(getattr(quantity, "value", quantity))
        return float(values[min(period, values.size - 1)])

    result = []
    for stream in streams:
        stage = []
        for segment in stream.segments if stream.has_segments else (stream,):
            a = value(segment.supply_temperature)
            b = value(segment.target_temperature)
            stage.append((max(a, b), min(a, b), abs(value(segment.heat_flow))))
        result.append(stage)
    return result


def _composite(
    streams: StreamCollection, args: HeatPumpTargetInputs
) -> tuple[np.ndarray, np.ndarray]:
    """Return a composite curve, hottest first, its heat falling to zero.

    Each linear stream segment holds its heat evenly over its temperature
    range; the composite's heat at a temperature is all the heat below it.
    """
    segments = [s for stage in _stream_segments(streams, args) for s in stage]
    T = np.unique([t for hi, lo, _ in segments for t in (hi, lo)])[::-1]
    H = np.zeros_like(T)
    for hi, lo, heat in segments:
        if hi - lo > tol:
            H += heat * np.clip((T - lo) / (hi - lo), 0.0, 1.0)
        else:
            H += np.where(T >= hi, heat, 0.0)
    return T, H


def _trim(T: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the flat ends of a curve, where it carries no heat."""
    if H.size < 2:
        return T, H
    first = int(np.flatnonzero(np.isclose(H, H[0], rtol=0.0, atol=tol))[-1])
    last = int(np.flatnonzero(np.isclose(H, H[-1], rtol=0.0, atol=tol))[0])
    if first > last:
        # A flat curve carries no heat at all.
        return T[:1], H[:1]
    return T[first : last + 1], H[first : last + 1]


def _window(
    T: np.ndarray, H: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    """The part of a curve between heat flows ``low`` and ``high``, from zero."""
    T_high = float(np.interp(high, H[::-1], T[::-1]))
    T_low = float(np.interp(low, H[::-1], T[::-1]))
    keep = (H > low) & (H < high)
    return (
        np.concatenate([[T_high], T[keep], [T_low]]),
        np.concatenate([[high], H[keep], [low]]) - low,
    )


def _closest_area(
    heat_pump: tuple[np.ndarray, np.ndarray],
    background: tuple[np.ndarray, np.ndarray],
    *,
    heat_pump_is_hot: bool,
    extra_dt: float,
    heat_transfer_coefficient,
) -> float:
    """Match a heat pump curve against the closest background segment it can.

    A condenser heats the sink just below it: its window slides down from the
    top of the sink. An evaporator cools the source just above it: its window
    slides up from the coldest end. The window is the first, by bisection on
    its offset, where the curves do not cross.
    """
    T_hp, H_hp = _trim(*heat_pump)
    T_bg, H_bg = background
    total = float(H_bg[0])
    duty = min(float(H_hp[0]), total)
    if duty <= tol:
        return 0.0
    T_hp, H_hp = _window(T_hp, H_hp, float(H_hp[0]) - duty, float(H_hp[0]))

    def window(offset: float):
        if heat_pump_is_hot:
            return _window(T_bg, H_bg, total - offset - duty, total - offset)
        return _window(T_bg, H_bg, offset, offset + duty)

    def curves(offset: float):
        T_win, H_win = window(offset)
        if heat_pump_is_hot:
            return T_hp, H_hp, T_win, H_win
        return T_win, H_win, T_hp, H_hp

    def crosses(offset: float) -> bool:
        tdf = get_temperature_driving_forces(*curves(offset))
        return float(np.min([tdf["delta_T1"], tdf["delta_T2"]])) < -tol

    low, high = 0.0, total - duty
    if crosses(high):
        return float("inf")
    if crosses(low):
        for _ in range(_BISECTIONS):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if crosses(middle) else (low, middle)
        low = high
    return matched_area(
        *curves(low),
        extra_dt=extra_dt,
        heat_transfer_coefficient=heat_transfer_coefficient,
    )


def _process_heat(stage: Any) -> float:
    if hasattr(stage, "process_heat"):
        return float(stage.process_heat)
    return float(stage.Q_heat or 0.0)


def _process_cooling(stage: Any) -> float:
    return float(getattr(stage, "Q_cool", 0.0) or 0.0)


def _compressor(stage: Any) -> CompressorDuty:
    if hasattr(stage, "suction_volume_flow"):
        return CompressorDuty(
            str(stage.refrigerant), stage.suction_volume_flow, stage.shaft_power
        )
    # An MVR stage compresses process water vapour from its state 0; its mass
    # flow is duty (kW) over specific enthalpy (J/kg), kg/s / 1000.
    # The suction is often saturated vapour, so its density comes from P and h.
    density = CoolProp.PropsSI("D", "P", stage.Ps[0], "H", stage.Hs[0], "Water")
    volume_flow = float(stage.source_m_dot) * 1000.0 / density * 3600.0
    return CompressorDuty(
        "Water", volume_flow, float(stage.work) * float(stage.eta_motor)
    )


def _internal_areas(stage: Any, args: HeatPumpTargetInputs) -> list[float]:
    """The IHX and the cascade exchanger this stage passes heat up through."""
    areas = []
    ihx_duty = float(getattr(stage, "ihx_duty", 0.0) or 0.0)
    if ihx_duty > tol:
        lmtd = stage.ihx_lmtd
        areas.append(
            ihx_duty * 1000.0 / (args.u_gas * lmtd) if lmtd > tol else float("inf")
        )
    cascade_heat = float(getattr(stage, "Q_cas_heat", 0.0) or 0.0)
    if cascade_heat > tol:
        dt = float(getattr(args, "dt_cascade_hx", 0.0))
        areas.append(
            cascade_heat * 1000.0 / (args.u_two_phase * dt)
            if dt > tol
            else float("inf")
        )
    return areas
