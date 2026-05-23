"""Utility normalization and default-completion helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.zone import Zone
from ...lib.config import Configuration
from ...lib.enums import ST, StreamLoc
from ...lib.schemas.io import UtilitySchema
from ...utils.miscellaneous import get_value


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
        utility.t_supply = get_value(utility.t_supply)

        t_target = get_value(utility.t_target)
        if t_target is None or t_target == utility.t_supply:
            delta = (
                -zone_config.DT_PHASE_CHANGE
                if utility.type in [ST.Hot.value, ST.Both.value]
                else zone_config.DT_PHASE_CHANGE
            )
            utility.t_target = utility.t_supply + delta
        else:
            utility.t_target = t_target

        dt_cont = get_value(utility.dt_cont)
        base_dt_cont = zone_config.DT_CONT if dt_cont is None else dt_cont
        utility.dt_cont = base_dt_cont
        effective_dt_cont = base_dt_cont * dt_cont_multiplier

        price = get_value(utility.price)
        utility.price = (
            zone_config.UTILITY_PRICE * zone_config.ANNUAL_OP_TIME
            if price is None
            else price
        )

        htc = get_value(utility.htc)
        utility.htc = zone_config.HTC if not htc else htc

        if (
            utility.type in [ST.Hot.value, ST.Both.value]
            and utility.active
            and (
                min(utility.t_supply, utility.t_target) - effective_dt_cont
                >=
                hu_t_min - zone_config.DT_PHASE_CHANGE
            )
        ):
            add_default_hu = False
        if (
            utility.type in [ST.Cold.value, ST.Both.value]
            and utility.active
            and (
                max(utility.t_supply, utility.t_target) + effective_dt_cont
                <=
                cu_t_max + zone_config.DT_PHASE_CHANGE
            )
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
    unassigned_utilities = utilities

    def _sort_key(selected: UtilitySchema):
        order = selected.t_supply
        return -order if utility_type == ST.Hot.value else order

    candidates = sorted(
        (
            u
            for u in unassigned_utilities
            if u.active and u.type in ["Both", utility_type]
        ),
        key=_sort_key,
    )

    for selected in candidates:
        if selected.type == utility_type:
            selected.active = False

        if utility_type == ST.Hot.value:
            t_supply = max(selected.t_supply, selected.t_target)
            t_target = min(selected.t_supply, selected.t_target)
        else:
            t_supply = min(selected.t_supply, selected.t_target)
            t_target = max(selected.t_supply, selected.t_target)

        key = (
            ".".join([StreamLoc.HotU.value, selected.name])
            if utility_type == ST.Hot.value
            else ".".join([StreamLoc.ColdU.value, selected.name])
        )

        created_utilities.add(
            Stream(
                name=selected.name,
                t_supply=t_supply,
                t_target=t_target,
                dt_cont=selected.dt_cont,
                dt_cont_act=selected.dt_cont * dt_cont_multiplier,
                htc=selected.htc,
                price=selected.price,
                is_process_stream=False,
            ),
            key,
        )

    return created_utilities, unassigned_utilities


def _set_utilities_for_zone_and_subzones(
    zone: Zone,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
) -> Zone:
    """Add utilities to a zone and its subzones using effective multipliers."""
    zone.hot_utilities = deepcopy(hot_utilities)
    zone.cold_utilities = deepcopy(cold_utilities)
    for subzone in zone.subzones.values():
        _set_utilities_for_zone_and_subzones(
            zone=subzone,
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
        )
    return zone
