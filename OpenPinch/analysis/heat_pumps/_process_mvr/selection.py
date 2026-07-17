"""Source-stream selection and thermodynamic validation for Process MVR."""

from __future__ import annotations

import numpy as np
from CoolProp.CoolProp import PropsSI

from ....domain.configuration import tol
from ....domain.enums import StreamType
from ....domain.stream import Stream
from ....domain.zone import Zone
from .membership import walk_zones
from .values import (
    period_values_or_scalar,
    required_period_value,
    to_degrees_celsius,
    to_kelvin,
    to_kilopascals,
    value_at_index,
)


def normalise_source_selectors(source_streams) -> list:
    """Normalize one or many source selectors to a non-empty list."""
    if source_streams is None:
        raise ValueError("source_streams is required for process MVR.")
    if isinstance(source_streams, (str, Stream)):
        return [source_streams]
    selectors = list(source_streams)
    if not selectors:
        raise ValueError("source_streams must include at least one stream selector.")
    return selectors


def match_source_streams(root: Zone, selectors: list) -> list[Stream]:
    """Return unique active hot gas streams matched by the selectors."""
    matched: list[Stream] = []
    for selector in selectors:
        for stream in match_one_selector(root, selector):
            if not any(stream is existing for existing in matched):
                matched.append(stream)
    if not matched:
        raise ValueError("No active hot gas process streams matched source_streams.")
    return matched


def match_one_selector(root: Zone, selector) -> list[Stream]:
    """Resolve one stream object, key, or display-name selector."""
    matches: list[Stream] = []
    for zone in walk_zones(root):
        for key, stream in zone.hot_streams.items():
            if not selector_matches(selector, key, stream):
                continue
            if not any(stream is existing for existing in matches):
                matches.append(stream)
    if not matches:
        if match_cold_streams(root, selector):
            raise ValueError(f"MVR source stream {selector!r} must be a hot stream.")
        raise ValueError(f"MVR source stream {selector!r} was not found.")
    for stream in matches:
        validate_process_mvr_source(stream, selector)
    return matches


def match_cold_streams(root: Zone, selector) -> list[Stream]:
    matches: list[Stream] = []
    for zone in walk_zones(root):
        for key, stream in zone.cold_streams.items():
            if selector_matches(selector, key, stream):
                matches.append(stream)
    return matches


def selector_matches(selector, key: str, stream: Stream) -> bool:
    if isinstance(selector, Stream):
        return stream is selector
    selector_text = str(selector)
    return key == selector_text or stream.name == selector_text


def validate_process_mvr_source(stream: Stream, selector) -> None:
    """Validate source role, phase, fluid, and period thermodynamic states."""
    if not stream.is_active:
        raise ValueError(f"MVR source stream {selector!r} is not active.")
    if not stream.is_process_stream:
        raise ValueError(f"MVR source stream {selector!r} must be a process stream.")
    if stream.stream_type != StreamType.Hot.value:
        raise ValueError(f"MVR source stream {selector!r} must be a hot stream.")
    phase = str(stream.fluid_phase or "").strip().lower()
    if phase not in {"gas", "vapour", "vapor"}:
        raise ValueError(
            f"MVR source stream {selector!r} must have fluid_phase='gas' or "
            "fluid_phase='vapour'."
        )
    if stream.fluid_name is None:
        raise ValueError(f"MVR source stream {selector!r} requires fluid_name.")
    validate_process_mvr_source_phase(stream, selector)


def validate_process_mvr_source_phase(stream: Stream, selector) -> None:
    """Validate and normalize pressure values for each source period."""
    period_count = int(stream.num_periods or 1)
    supply_pressures: list[float] = []
    target_pressures: list[float] = []
    for index in range(period_count):
        supply_temperature = required_period_value(
            stream.supply_temperature,
            index,
            "t_supply",
            selector,
        )
        target_temperature = required_period_value(
            stream.target_temperature,
            index,
            "t_target",
            selector,
        )
        supply_pressure = value_at_index(stream.supply_pressure, index, unit="kPa")
        target_pressure = value_at_index(stream.target_pressure, index, unit="kPa")
        critical_temperature = to_degrees_celsius(
            coolprop_value("TCRIT", str(stream.fluid_name))
        )
        critical_pressure = to_kilopascals(
            coolprop_value("PCRIT", str(stream.fluid_name))
        )
        temperature_drop = supply_temperature - target_temperature

        if temperature_drop < -tol:
            raise ValueError(
                f"MVR source stream {selector!r} must cool from supply to target."
            )
        if temperature_drop < 1.0:
            supply_pressure = validate_vapour_state(
                selector=selector,
                fluid=str(stream.fluid_name),
                t_supply=supply_temperature,
                p_supply=supply_pressure,
                t_crit=critical_temperature,
            )
            target_pressure = (
                supply_pressure if target_pressure is None else target_pressure
            )
        else:
            if supply_pressure is None:
                raise ValueError(
                    f"MVR gas source stream {selector!r} requires p_supply."
                )
            target_pressure = (
                supply_pressure if target_pressure is None else target_pressure
            )
            validate_gas_or_supercritical_state(
                selector=selector,
                fluid=str(stream.fluid_name),
                t_c=supply_temperature,
                p_kpa=supply_pressure,
                t_crit=critical_temperature,
                p_crit=critical_pressure,
                state_label="supply",
            )
            validate_gas_or_supercritical_state(
                selector=selector,
                fluid=str(stream.fluid_name),
                t_c=target_temperature,
                p_kpa=target_pressure,
                t_crit=critical_temperature,
                p_crit=critical_pressure,
                state_label="target",
            )
        supply_pressures.append(float(supply_pressure))
        target_pressures.append(float(target_pressure))

    if stream.supply_pressure is None:
        stream.supply_pressure = period_values_or_scalar(supply_pressures)
    if stream.target_pressure is None:
        stream.target_pressure = period_values_or_scalar(target_pressures)


def validate_vapour_state(
    *,
    selector,
    fluid: str,
    t_supply: float,
    p_supply: float | None,
    t_crit: float,
) -> float:
    if t_supply >= t_crit:
        raise ValueError(
            f"MVR vapour source stream {selector!r} requires t_supply below "
            f"the critical temperature ({t_crit:.3g} degC)."
        )
    saturation_pressure = to_kilopascals(
        PropsSI("P", "T", to_kelvin(t_supply), "Q", 1.0, fluid)
    )
    if p_supply is None:
        return float(saturation_pressure)
    if not np.isclose(p_supply, saturation_pressure, rtol=1e-2, atol=0.5):
        raise ValueError(
            f"MVR vapour source stream {selector!r} requires p_supply close to "
            "saturation pressure at t_supply "
            f"({saturation_pressure:.3g} kPa)."
        )
    return float(p_supply)


def validate_gas_or_supercritical_state(
    *,
    selector,
    fluid: str,
    t_c: float,
    p_kpa: float,
    t_crit: float,
    p_crit: float,
    state_label: str,
) -> None:
    if p_kpa > p_crit:
        if t_c <= t_crit:
            raise ValueError(
                f"MVR gas source stream {selector!r} has {state_label} pressure "
                "above critical pressure but temperature below critical "
                "temperature."
            )
        return
    if t_c > t_crit:
        return
    saturation_pressure = to_kilopascals(
        PropsSI("P", "T", to_kelvin(t_c), "Q", 1.0, fluid)
    )
    if p_kpa >= saturation_pressure:
        raise ValueError(
            f"MVR gas source stream {selector!r} has {state_label} pressure "
            f"above saturation pressure ({saturation_pressure:.3g} kPa) "
            f"at {t_c:.3g} degC."
        )


def coolprop_value(output: str, fluid: str) -> float:
    try:
        return float(PropsSI(output, fluid))
    except Exception as exc:
        raise ValueError(
            f"MVR source fluid {fluid!r} is not available in CoolProp."
        ) from exc


__all__: list[str] = []
