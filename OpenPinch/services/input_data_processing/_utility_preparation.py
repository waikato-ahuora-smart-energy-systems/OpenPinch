"""Utility normalization and default-completion helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

import numpy as np

from ...classes._stream.value_state import resolve_period_weights
from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.value import Value
from ...classes.zone import Zone
from ...lib.config import Configuration
from ...lib.enums import ST, StreamLoc
from ...lib.schemas.io import UtilitySchema
from ...lib.unit_system import standardise_input_value
from ._stream_segment_preparation import _create_segmented_utility_stream

__all__ = [
    "_get_hot_and_cold_utilities",
    "_set_utilities_for_zone_and_subzones",
]

_TEMPERATURE_EQUAL_TOL = 1e-12


def _get_hot_and_cold_utilities(
    utilities: List[UtilitySchema],
    hu_t_min: float,
    cu_t_max: float,
    config: Configuration,
    dt_cont_multiplier: float = 1.0,
) -> StreamCollection:
    """Extract all utility data into class instances."""
    utilities_filled, add_default_hu, add_default_cu = _complete_utility_data(
        deepcopy(list(utilities)),
        config=config,
        hu_t_min=hu_t_min,
        cu_t_max=cu_t_max,
        dt_cont_multiplier=dt_cont_multiplier,
    )
    utilities_with_defaults = _add_default_utilities(
        utilities=utilities_filled,
        config=config,
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
        config=config,
    )
    cold_utilities, _ = _create_utilities_list(
        utilities=utilities_left,
        utility_type=ST.Cold.value,
        dt_cont_multiplier=dt_cont_multiplier,
        config=config,
    )
    prepared = hot_utilities + cold_utilities
    period_ids = {
        str(period_id): index
        for index, period_id in enumerate(config.problem.period_ids)
    }
    weights = resolve_period_weights(period_ids, config.problem.period_weights)
    prepared.set_period_context(
        period_ids=period_ids,
        weights=weights,
        num_periods=len(period_ids),
    )
    return prepared


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


def _value_is_missing(value: Value | None) -> bool:
    if value is None:
        return True
    return bool(np.all(np.isnan(value.period_values.astype(float))))


def _shift_temperature_value(value: Value, delta: float) -> Value:
    return value + Value(delta, unit="delta_degC")


def _utility_temperature_arrays(
    utility: UtilitySchema,
    config: Configuration,
    *,
    allow_missing_target: bool = False,
) -> tuple[Value, Value]:
    if utility.segments is not None:
        authoritative_supply = utility.segments[0].t_supply
        authoritative_target = utility.segments[-1].t_target
    elif utility.profile is not None:
        authoritative_supply = utility.profile.points[0].temperature
        authoritative_target = utility.profile.points[-1].temperature
    else:
        authoritative_supply = utility.t_supply
        authoritative_target = utility.t_target
    t_supply = standardise_input_value(
        authoritative_supply,
        field_name="t_supply",
        config=config,
    )
    t_target = standardise_input_value(
        authoritative_target,
        field_name="t_target",
        config=config,
    )
    is_nested = utility.segments is not None or utility.profile is not None
    if t_supply is None or (
        t_target is None and (is_nested or not allow_missing_target)
    ):
        raise ValueError(
            f"Utility '{utility.name}' is missing supply or target temperature."
        )
    if is_nested:
        for field_name, supplied, authoritative in (
            ("t_supply", utility.t_supply, t_supply),
            ("t_target", utility.t_target, t_target),
        ):
            if supplied is None:
                continue
            supplied_value = standardise_input_value(
                supplied,
                field_name=field_name,
                config=config,
            ).to(authoritative.unit)
            supplied_values = supplied_value.period_values
            authoritative_values = authoritative.period_values
            if supplied_values.size == 1 and authoritative_values.size > 1:
                supplied_values = np.full(
                    authoritative_values.size,
                    supplied_values[0],
                )
            if authoritative_values.size == 1 and supplied_values.size > 1:
                authoritative_values = np.full(
                    supplied_values.size,
                    authoritative_values[0],
                )
            if not np.allclose(
                supplied_values,
                authoritative_values,
                atol=_TEMPERATURE_EQUAL_TOL,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Segmented utility {utility.name!r} supplied {field_name} "
                    "does not match the authoritative profile."
                )
    return t_supply, t_target


def _orient_utility_temperatures(
    utility: UtilitySchema,
    utility_type: str,
    config: Configuration,
) -> tuple[Value, Value, bool]:
    t_supply, t_target = _utility_temperature_arrays(utility, config)
    t_supply_arr = t_supply.period_values
    t_target_arr = t_target.period_values

    if utility_type == ST.Hot.value:
        if np.all(t_supply_arr >= t_target_arr - _TEMPERATURE_EQUAL_TOL):
            return t_supply, t_target, False
        if np.all(t_supply_arr <= t_target_arr + _TEMPERATURE_EQUAL_TOL):
            return t_target, t_supply, True
    elif utility_type == ST.Cold.value:
        if np.all(t_supply_arr <= t_target_arr + _TEMPERATURE_EQUAL_TOL):
            return t_supply, t_target, False
        if np.all(t_supply_arr >= t_target_arr - _TEMPERATURE_EQUAL_TOL):
            return t_target, t_supply, True

    raise ValueError(
        f"Utility '{utility.name}' temperatures cannot be oriented consistently as "
        f"'{utility_type}' across all states."
    )


def _complete_utility_data(
    utilities: List[UtilitySchema],
    config: Configuration,
    hu_t_min: float,
    cu_t_max: float,
    dt_cont_multiplier: float = 1.0,
) -> Tuple[List[UtilitySchema], bool, bool]:
    """Complete utility data and add default utilities where needed."""
    add_default_hu = True
    add_default_cu = True
    thermal = config.thermal

    for utility in utilities:
        t_supply, t_target = _utility_temperature_arrays(
            utility,
            config,
            allow_missing_target=True,
        )
        if _value_is_missing(t_target):
            delta = (
                -thermal.dt_phase_change
                if utility.type in [ST.Hot.value, ST.Both.value]
                else thermal.dt_phase_change
            )
            utility.t_target = _shift_temperature_value(t_supply, delta).to_dict()
            t_target = standardise_input_value(
                utility.t_target,
                field_name="t_target",
                config=config,
            )
        else:
            if np.allclose(
                t_supply.period_values,
                t_target.period_values,
                atol=_TEMPERATURE_EQUAL_TOL,
                rtol=0.0,
            ):
                delta = (
                    -thermal.dt_phase_change
                    if utility.type in [ST.Hot.value, ST.Both.value]
                    else thermal.dt_phase_change
                )
                utility.t_target = _shift_temperature_value(
                    t_supply,
                    delta,
                ).to_dict()
                t_target = standardise_input_value(
                    utility.t_target,
                    field_name="t_target",
                    config=config,
                )

        dt_cont = standardise_input_value(
            utility.dt_cont,
            field_name="dt_cont",
            config=config,
        )
        if _value_is_missing(dt_cont):
            utility.dt_cont = thermal.dt_cont
            dt_cont = standardise_input_value(
                utility.dt_cont,
                field_name="dt_cont",
                config=config,
            )

        price_value = standardise_input_value(
            utility.price,
            field_name="price",
            config=config,
        )
        if _value_is_missing(price_value):
            utility.price = config.costing.utility_price * config.costing.annual_op_time

        htc_value = standardise_input_value(
            utility.htc,
            field_name="htc",
            config=config,
        )
        if _value_is_missing(htc_value):
            utility.htc = thermal.htc

        t_supply_arr = t_supply.period_values
        t_target_arr = t_target.period_values
        dt_cont_arr = dt_cont.period_values

        effective_dt_cont_arr = dt_cont_arr * float(dt_cont_multiplier)

        if (
            utility.type in [ST.Hot.value, ST.Both.value]
            and utility.active
            and (
                np.min(np.minimum(t_supply_arr, t_target_arr) - effective_dt_cont_arr)
                >= hu_t_min - thermal.dt_phase_change
            )
        ):
            add_default_hu = False
        if (
            utility.type in [ST.Cold.value, ST.Both.value]
            and utility.active
            and (
                np.max(np.maximum(t_supply_arr, t_target_arr) + effective_dt_cont_arr)
                <= cu_t_max + thermal.dt_phase_change
            )
        ):
            add_default_cu = False
    return utilities, add_default_hu, add_default_cu


def _add_default_utilities(
    utilities: List[UtilitySchema],
    config: Configuration,
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
                config,
                dt_cont_multiplier=dt_cont_multiplier,
            )
        )
    if add_default_cu:
        utilities.append(
            _create_default_utility(
                "CU",
                ST.Cold.value,
                cu_t_max,
                config,
                dt_cont_multiplier=dt_cont_multiplier,
            )
        )
    return utilities


def _create_default_utility(
    name: str,
    utility_type: str,
    temperature: float,
    config: Configuration,
    dt_cont_multiplier: float = 1.0,
) -> UtilitySchema:
    """Construct a default utility entry anchored to the process temperature."""
    direction = 1 if utility_type == ST.Hot.value else -1
    dt_cont = config.thermal.dt_cont
    dt_cont_act = dt_cont * dt_cont_multiplier
    return UtilitySchema.model_validate(
        {
            "name": name,
            "type": utility_type,
            "t_supply": temperature + dt_cont_act * direction,
            "t_target": temperature
            + (dt_cont_act - config.thermal.dt_phase_change) * direction,
            "heat_flow": 0,
            "dt_cont": dt_cont,
            "price": config.costing.utility_price,
            "htc": config.thermal.htc,
        }
    )


def _create_utilities_list(
    utilities: List[UtilitySchema],
    utility_type: str,
    dt_cont_multiplier: float = 1.0,
    config: Configuration = Configuration(),
) -> Tuple[StreamCollection, List[UtilitySchema]]:
    """Create a sorted list of hot or cold Stream objects based on type."""
    created_utilities = StreamCollection()
    unassigned_utilities = utilities

    candidates: list[tuple[UtilitySchema, Value, Value, bool]] = []
    for selected in unassigned_utilities:
        if not (selected.active and selected.type in ["Both", utility_type]):
            continue

        supply_value, target_value, endpoints_swapped = _orient_utility_temperatures(
            selected,
            utility_type,
            config,
        )
        candidates.append((selected, supply_value, target_value, endpoints_swapped))

    for selected, supply_value, target_value, endpoints_swapped in candidates:
        if selected.type == utility_type:
            selected.active = False

        key = (
            ".".join([StreamLoc.HotU.value, selected.name])
            if utility_type == ST.Hot.value
            else ".".join([StreamLoc.ColdU.value, selected.name])
        )

        p_supply = selected.p_target if endpoints_swapped else selected.p_supply
        p_target = selected.p_supply if endpoints_swapped else selected.p_target
        h_supply = selected.h_target if endpoints_swapped else selected.h_supply
        h_target = selected.h_supply if endpoints_swapped else selected.h_target
        if selected.segments is not None or selected.profile is not None:
            created = _create_segmented_utility_stream(
                selected,
                utility_type=utility_type,
                config=config,
                dt_cont_multiplier=dt_cont_multiplier,
            )
        else:
            created = Stream(
                name=selected.name,
                t_supply=supply_value,
                t_target=target_value,
                p_supply=standardise_input_value(
                    p_supply,
                    field_name="p_supply",
                    config=config,
                ),
                p_target=standardise_input_value(
                    p_target,
                    field_name="p_target",
                    config=config,
                ),
                h_supply=standardise_input_value(
                    h_supply,
                    field_name="h_supply",
                    config=config,
                ),
                h_target=standardise_input_value(
                    h_target,
                    field_name="h_target",
                    config=config,
                ),
                dt_cont=standardise_input_value(
                    selected.dt_cont,
                    field_name="dt_cont",
                    config=config,
                ),
                htc=standardise_input_value(
                    selected.htc,
                    field_name="htc",
                    config=config,
                ),
                price=standardise_input_value(
                    selected.price,
                    field_name="price",
                    config=config,
                ),
                is_process_stream=False,
                fluid_name=selected.fluid_name,
                fluid_phase=selected.fluid_phase,
            )
        created_utilities.add(created, key)
        created_utilities[key].dt_cont_multiplier = dt_cont_multiplier

    return created_utilities, unassigned_utilities
