"""Stateless core-state completion and derived stream thermodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..lib.enums import ST
from . import _stream_value_state
from .value import Value


@dataclass(frozen=True)
class CompletedCoreState:
    """Core values after defaults and the sensible minimum-span convention."""

    t_supply: Value
    t_target: Value
    dt_cont: Value
    heat_flow: Value
    htc: Value
    price: Value


@dataclass(frozen=True)
class DerivedStreamState:
    """All values calculated from one complete stream core state."""

    stream_type: str
    dt_cont_act: Value
    t_min: Value
    t_max: Value
    t_min_star: Value
    t_max_star: Value
    t_entr_mean: Value
    cp: Value
    htr: Value
    rcp_prod: Value
    cost: Value


def complete_core_state(
    *,
    t_supply: Value | None,
    t_target: Value | None,
    dt_cont: Value | None,
    heat_flow: Value | None,
    htc: Value | None,
    price: Value | None,
    value_units: Mapping[str, str],
    stream_name: str,
    state_size: int,
    temperature_equal_tol: float,
) -> CompletedCoreState:
    """Fill missing core values and preserve the existing minimum span rule."""
    if t_supply is None:
        if t_target is None:
            t_supply = Value(15, "degC").to(value_units["_t_supply"])
        else:
            t_supply = Value(t_target).to(value_units["_t_supply"])
    if t_target is None:
        t_target = Value(t_supply).to(value_units["_t_target"])
    if dt_cont is None:
        dt_cont = Value(0.0, unit=value_units["_dt_cont"])
    if heat_flow is None:
        heat_flow = Value(0.0, unit=value_units["_heat_flow"])
    if htc is None:
        htc = Value(1.0, unit=value_units["_htc"])
    if price is None:
        price = Value(0.0, unit=value_units["_price"])

    t_supply_arr = _stream_value_state.value_array(
        t_supply,
        size=state_size,
        stream_name=stream_name,
    )
    t_target_arr = _stream_value_state.value_array(
        t_target,
        size=state_size,
        stream_name=stream_name,
    )
    heat_flow_arr = _stream_value_state.value_array(
        heat_flow,
        size=state_size,
        stream_name=stream_name,
    )

    equal_mask = np.isclose(
        t_supply_arr,
        t_target_arr,
        atol=temperature_equal_tol,
        rtol=0.0,
    )
    if np.any(equal_mask):
        adjusted_target_arr = t_target_arr.copy()
        cold_mask = equal_mask & (heat_flow_arr > 0.0)
        hot_mask = equal_mask & (heat_flow_arr < 0.0)
        adjusted_target_arr[cold_mask] = t_supply_arr[cold_mask] + 0.01
        adjusted_target_arr[hot_mask] = t_supply_arr[hot_mask] - 0.01
        t_target = _stream_value_state.build_value(
            adjusted_target_arr,
            unit=t_supply.unit,
        ).to(value_units["_t_target"])

    return CompletedCoreState(
        t_supply=t_supply,
        t_target=t_target,
        dt_cont=dt_cont,
        heat_flow=heat_flow,
        htc=htc,
        price=price,
    )


def derive_stream_state(
    *,
    t_supply: Value,
    t_target: Value,
    dt_cont: Value,
    dt_cont_multiplier: float,
    heat_flow: Value,
    htc: Value,
    price: Value,
    value_units: Mapping[str, str],
    stream_name: str,
    state_size: int,
    temperature_equal_tol: float,
) -> DerivedStreamState:
    """Calculate all derived values for one complete stream core state."""
    t_supply_arr = _stream_value_state.value_array(
        t_supply,
        size=state_size,
        stream_name=stream_name,
    )
    t_target_arr = _stream_value_state.value_array(
        t_target,
        size=state_size,
        stream_name=stream_name,
    )
    heat_flow_arr = _stream_value_state.value_array(
        heat_flow,
        size=state_size,
        stream_name=stream_name,
    )
    htc_arr = _stream_value_state.value_array(
        htc,
        size=state_size,
        stream_name=stream_name,
    )
    price_arr = _stream_value_state.value_array(
        price,
        size=state_size,
        stream_name=stream_name,
    )
    dt_cont_arr = _stream_value_state.value_array(
        dt_cont,
        size=state_size,
        stream_name=stream_name,
    )

    dt_cont_act = dt_cont_arr * float(dt_cont_multiplier)
    hot_states = t_supply_arr > t_target_arr + temperature_equal_tol
    cold_states = t_supply_arr < t_target_arr - temperature_equal_tol

    if np.any(hot_states):
        stream_type = ST.Hot.value
        t_min = t_target_arr
        t_max = t_supply_arr
        t_min_star = t_min - dt_cont_act
        t_max_star = t_max - dt_cont_act
    elif np.any(cold_states):
        stream_type = ST.Cold.value
        t_min = t_supply_arr
        t_max = t_target_arr
        t_min_star = t_min + dt_cont_act
        t_max_star = t_max + dt_cont_act
    else:
        stream_type = ST.Neutral.value
        t_min = t_supply_arr
        t_max = t_target_arr
        t_min_star = t_min.copy()
        t_max_star = t_max.copy()

    delta_t = t_max - t_min
    cp = np.zeros_like(delta_t, dtype=float)
    valid_dt = np.abs(delta_t) > temperature_equal_tol
    cp[valid_dt] = heat_flow_arr[valid_dt] / delta_t[valid_dt]

    htr = np.zeros_like(htc_arr, dtype=float)
    valid_htc = htc_arr > 0.0
    htr[valid_htc] = 1.0 / htc_arr[valid_htc]

    rcp_prod = np.zeros_like(cp, dtype=float)
    rcp_prod[valid_htc] = cp[valid_htc] * htr[valid_htc]

    t_supply_k = t_supply_arr + 273.15
    t_target_k = t_target_arr + 273.15
    with np.errstate(divide="ignore", invalid="ignore"):
        t_entr_mean = (
            (t_supply_k - t_target_k) / (np.log(t_supply_k) - np.log(t_target_k))
        ) - 273.15

    build_value = _stream_value_state.build_value
    return DerivedStreamState(
        stream_type=stream_type,
        dt_cont_act=build_value(
            dt_cont_act,
            unit=value_units["_dt_cont_act"],
        ),
        t_min=build_value(t_min, unit=value_units["_t_min"]),
        t_max=build_value(t_max, unit=value_units["_t_max"]),
        t_min_star=build_value(t_min_star, unit=value_units["_t_min_star"]),
        t_max_star=build_value(t_max_star, unit=value_units["_t_max_star"]),
        t_entr_mean=build_value(t_entr_mean, unit=value_units["_t_supply"]),
        cp=build_value(cp, unit=value_units["_cp"]),
        htr=build_value(htr, unit=value_units["_htr"]),
        rcp_prod=build_value(rcp_prod, unit=value_units["_rcp_prod"]),
        cost=build_value(
            price_arr * heat_flow_arr / 1000.0,
            unit=value_units["_cost"],
        ),
    )


__all__: list[str] = []
