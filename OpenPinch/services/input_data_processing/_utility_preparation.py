"""Utility normalization and default-completion helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

import numpy as np

from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.value import Value
from ...classes.zone import Zone
from ...lib.config import Configuration
from ...lib.enums import ST, StreamLoc
from ...lib.schemas.io import UtilitySchema

_STATE_WEIGHT_RTOL = 1e-12
_STATE_WEIGHT_ATOL = 1e-12
_TEMPERATURE_EQUAL_TOL = 1e-12


def _coerce_value(data, *, unit: str | None = None) -> Value | None:
    if data is None:
        return None
    parsed = Value(data)
    if unit is None:
        return parsed
    return Value(parsed, unit=unit)


def _value_is_missing(value: Value | None) -> bool:
    if value is None:
        return True
    return bool(np.all(np.isnan(value.state_values.astype(float))))


def _resolve_state_context(
    values_by_name: dict[str, Value | None],
) -> tuple[list[str] | None, np.ndarray | None]:
    stateful_values = [
        (name, value)
        for name, value in values_by_name.items()
        if value is not None and value.state_ids is not None
    ]
    if not stateful_values:
        return None, None

    ref_name, ref_value = stateful_values[0]
    ref_state_ids = ref_value.state_ids
    ref_weights = ref_value.weights

    for name, value in stateful_values[1:]:
        if value.state_ids != ref_state_ids:
            raise ValueError(f"state_ids for {name} must align with {ref_name}.")
        if not np.allclose(
            value.weights,
            ref_weights,
            rtol=_STATE_WEIGHT_RTOL,
            atol=_STATE_WEIGHT_ATOL,
        ):
            raise ValueError(f"weights for {name} must align with {ref_name}.")

    return ref_state_ids, ref_weights


def _broadcast_magnitudes(
    value: Value | None,
    state_ids: list[str] | None,
) -> np.ndarray | None:
    if value is None:
        return None
    if state_ids is None:
        return np.asarray([float(value.value)], dtype=float)
    if value.state_ids is None:
        return np.full(len(state_ids), float(value.value), dtype=float)
    return value.state_values.astype(float)


def _value_from_array(
    magnitudes,
    *,
    unit: str | None,
    state_ids: list[str] | None,
    weights: np.ndarray | None,
) -> Value:
    arr = np.asarray(magnitudes, dtype=float).reshape(-1)
    if state_ids is None:
        return Value(float(arr[0]), unit=unit)
    return Value(values=arr, unit=unit, state_ids=state_ids, weights=weights)


def _shift_temperature_value(value: Value, delta: float) -> Value:
    unit = value.to_dict().get("unit")
    return _value_from_array(
        value.state_values.astype(float) + float(delta),
        unit=unit,
        state_ids=value.state_ids,
        weights=None if value.state_ids is None else value.weights,
    )


def _utility_temperature_arrays(
    utility: UtilitySchema,
) -> tuple[Value, Value, np.ndarray, np.ndarray, list[str] | None, np.ndarray | None]:
    t_supply_value = _coerce_value(utility.t_supply, unit="degC")
    t_target_value = _coerce_value(utility.t_target, unit="degC")
    if t_supply_value is None or t_target_value is None:
        raise ValueError(
            f"Utility '{utility.name}' is missing supply or target temperature."
        )

    state_ids, weights = _resolve_state_context(
        {"t_supply": t_supply_value, "t_target": t_target_value}
    )
    t_supply_arr = _broadcast_magnitudes(t_supply_value, state_ids)
    t_target_arr = _broadcast_magnitudes(t_target_value, state_ids)
    return (
        t_supply_value,
        t_target_value,
        t_supply_arr,
        t_target_arr,
        state_ids,
        weights,
    )


def _orient_utility_temperatures(
    utility: UtilitySchema,
    utility_type: str,
) -> tuple[Value, Value]:
    (
        t_supply_value,
        t_target_value,
        t_supply_arr,
        t_target_arr,
        _state_ids,
        _weights,
    ) = _utility_temperature_arrays(utility)

    if utility_type == ST.Hot.value:
        if np.all(t_supply_arr >= t_target_arr - _TEMPERATURE_EQUAL_TOL):
            return t_supply_value, t_target_value
        if np.all(t_supply_arr <= t_target_arr + _TEMPERATURE_EQUAL_TOL):
            return t_target_value, t_supply_value
    elif utility_type == ST.Cold.value:
        if np.all(t_supply_arr <= t_target_arr + _TEMPERATURE_EQUAL_TOL):
            return t_supply_value, t_target_value
        if np.all(t_supply_arr >= t_target_arr - _TEMPERATURE_EQUAL_TOL):
            return t_target_value, t_supply_value

    raise ValueError(
        f"Utility '{utility.name}' temperatures cannot be oriented consistently as "
        f"'{utility_type}' across all states."
    )


def _get_hot_and_cold_utilities(
    utilities: List[UtilitySchema],
    hu_t_min: float,
    cu_t_max: float,
    zone_config: Configuration,
    dt_cont_multiplier: float = 1.0,
) -> Tuple[StreamCollection, StreamCollection]:
    """Extract all utility data into class instances."""
    utilities_filled, add_default_hu, add_default_cu = _complete_utility_data(
        deepcopy(list(utilities)),
        zone_config=zone_config,
        hu_t_min=hu_t_min,
        cu_t_max=cu_t_max,
        dt_cont_multiplier=dt_cont_multiplier,
    )
    utilities_with_defaults = _add_default_utilities(
        utilities=utilities_filled,
        zone_config=zone_config,
        add_default_hu=add_default_hu,
        add_default_cu=add_default_cu,
        hu_t_min=hu_t_min,
        cu_t_max=cu_t_max,
        dt_cont_multiplier=dt_cont_multiplier,
    )
    hot_utilities, utilities_left = _create_utilities_list(
        utilities=utilities_with_defaults,
        utility_type=ST.Hot.value,
        dt_cont_multiplier=dt_cont_multiplier,
    )
    cold_utilities, _ = _create_utilities_list(
        utilities=utilities_left,
        utility_type=ST.Cold.value,
        dt_cont_multiplier=dt_cont_multiplier,
    )
    return hot_utilities, cold_utilities


def _complete_utility_data(
    utilities: List[UtilitySchema],
    zone_config: Configuration,
    hu_t_min: float,
    cu_t_max: float,
    dt_cont_multiplier: float = 1.0,
) -> Tuple[List[UtilitySchema], bool, bool]:
    """Complete utility data and add default utilities where needed."""
    add_default_hu = True
    add_default_cu = True

    for utility in utilities:
        t_supply_value = _coerce_value(utility.t_supply, unit="degC")
        t_target_value = _coerce_value(utility.t_target, unit="degC")
        if _value_is_missing(t_target_value):
            delta = (
                -zone_config.DT_PHASE_CHANGE
                if utility.type in [ST.Hot.value, ST.Both.value]
                else zone_config.DT_PHASE_CHANGE
            )
            utility.t_target = _shift_temperature_value(t_supply_value, delta).to_dict()
            t_target_value = _coerce_value(utility.t_target, unit="degC")
        else:
            state_ids, _weights = _resolve_state_context(
                {"t_supply": t_supply_value, "t_target": t_target_value}
            )
            t_supply_arr = _broadcast_magnitudes(t_supply_value, state_ids)
            t_target_arr = _broadcast_magnitudes(t_target_value, state_ids)
            if np.allclose(
                t_supply_arr,
                t_target_arr,
                atol=_TEMPERATURE_EQUAL_TOL,
                rtol=0.0,
            ):
                delta = (
                    -zone_config.DT_PHASE_CHANGE
                    if utility.type in [ST.Hot.value, ST.Both.value]
                    else zone_config.DT_PHASE_CHANGE
                )
                utility.t_target = _shift_temperature_value(
                    t_supply_value,
                    delta,
                ).to_dict()
                t_target_value = _coerce_value(utility.t_target, unit="degC")

        dt_cont_value = _coerce_value(utility.dt_cont, unit="delta_degC")
        if _value_is_missing(dt_cont_value):
            utility.dt_cont = zone_config.DT_CONT
            dt_cont_value = _coerce_value(utility.dt_cont, unit="delta_degC")

        price_value = _coerce_value(utility.price, unit="USD/MWh")
        if _value_is_missing(price_value):
            utility.price = zone_config.UTILITY_PRICE * zone_config.ANNUAL_OP_TIME

        htc_value = _coerce_value(utility.htc, unit="kW/m^2/K")
        if _value_is_missing(htc_value):
            utility.htc = zone_config.HTC

        state_ids, _weights = _resolve_state_context(
            {
                "t_supply": t_supply_value,
                "t_target": t_target_value,
                "dt_cont": dt_cont_value,
            }
        )
        t_supply_arr = _broadcast_magnitudes(t_supply_value, state_ids)
        t_target_arr = _broadcast_magnitudes(t_target_value, state_ids)
        dt_cont_arr = _broadcast_magnitudes(dt_cont_value, state_ids)
        effective_dt_cont_arr = dt_cont_arr * float(dt_cont_multiplier)

        if (
            utility.type in [ST.Hot.value, ST.Both.value]
            and utility.active
            and np.min(np.minimum(t_supply_arr, t_target_arr) - effective_dt_cont_arr)
            >= hu_t_min - zone_config.DT_PHASE_CHANGE
        ):
            add_default_hu = False
        if (
            utility.type in [ST.Cold.value, ST.Both.value]
            and utility.active
            and np.max(np.maximum(t_supply_arr, t_target_arr) + effective_dt_cont_arr)
            <= cu_t_max + zone_config.DT_PHASE_CHANGE
        ):
            add_default_cu = False
    return utilities, add_default_hu, add_default_cu


def _add_default_utilities(
    utilities: List[UtilitySchema],
    zone_config: Configuration,
    add_default_hu: bool,
    add_default_cu: bool,
    hu_t_min: float,
    cu_t_max: float,
    dt_cont_multiplier: float = 1.0,
) -> List[UtilitySchema]:
    """Add default hot and cold utilities to the list of utilities."""
    if add_default_hu:
        utilities.append(
            _create_default_utility(
                "HU",
                ST.Hot.value,
                hu_t_min,
                zone_config,
                dt_cont_multiplier=dt_cont_multiplier,
            )
        )
    if add_default_cu:
        utilities.append(
            _create_default_utility(
                "CU",
                ST.Cold.value,
                cu_t_max,
                zone_config,
                dt_cont_multiplier=dt_cont_multiplier,
            )
        )
    return utilities


def _create_default_utility(
    name: str,
    utility_type: str,
    temperature: float,
    zone_config: Configuration,
    dt_cont_multiplier: float = 1.0,
) -> UtilitySchema:
    """Construct a default utility entry anchored to the process temperature."""
    direction = 1 if utility_type == ST.Hot.value else -1
    dt_cont = zone_config.DT_CONT
    dt_cont_act = dt_cont * dt_cont_multiplier
    return UtilitySchema.model_validate(
        {
            "name": name,
            "type": utility_type,
            "t_supply": temperature + dt_cont_act * direction,
            "t_target": temperature
            + (dt_cont_act - zone_config.DT_PHASE_CHANGE) * direction,
            "heat_flow": 0,
            "dt_cont": dt_cont,
            "price": zone_config.UTILITY_PRICE,
            "htc": zone_config.HTC,
        }
    )


def _create_utilities_list(
    utilities: List[UtilitySchema],
    utility_type: str,
    dt_cont_multiplier: float = 1.0,
) -> Tuple[StreamCollection, List[UtilitySchema]]:
    """Create a sorted list of hot or cold Stream objects based on type."""
    created_utilities = StreamCollection()
    created_utilities.set_sort_key(lambda stream: stream.t_supply.max, reverse=True)
    unassigned_utilities = utilities

    candidates: list[tuple[float, UtilitySchema, Value, Value]] = []
    for selected in unassigned_utilities:
        if not (selected.active and selected.type in ["Both", utility_type]):
            continue

        supply_value, target_value = _orient_utility_temperatures(selected, utility_type)
        order = float(supply_value.max_value().value)
        candidates.append((order, selected, supply_value, target_value))

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for _order, selected, supply_value, target_value in candidates:
        if selected.type == utility_type:
            selected.active = False

        key = (
            ".".join([StreamLoc.HotU.value, selected.name])
            if utility_type == ST.Hot.value
            else ".".join([StreamLoc.ColdU.value, selected.name])
        )

        created_utilities.add(
            Stream(
                name=selected.name,
                t_supply=supply_value,
                t_target=target_value,
                dt_cont=selected.dt_cont,
                htc=selected.htc,
                price=selected.price,
                is_process_stream=False,
            ),
            key,
        )
        created_utilities[key].dt_cont_multiplier = dt_cont_multiplier

    return created_utilities, unassigned_utilities


def _set_utilities_for_zone_and_subzones(
    zone: Zone,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
) -> Zone:
    """Add utilities to a zone and its subzones using effective multipliers."""
    zone.hot_utilities = hot_utilities.copy(deep=True)
    zone.cold_utilities = cold_utilities.copy(deep=True)
    for subzone in zone.subzones.values():
        _set_utilities_for_zone_and_subzones(
            zone=subzone,
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
        )
    return zone
