"""Construct validated :class:`Zone` trees and attach stream/utility data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....contracts.input import StreamSchema, UtilitySchema, ZoneTreeSchema
from ....contracts.units import standardise_input_value
from ....domain._value.resolution import resolve_value_array
from ....domain.configuration import tol
from ....domain.enums import StreamType
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection
from ....domain.value import Value
from ....domain.zone import Zone
from . import canonicalization as _canonicalization
from .segments import _create_segmented_process_stream
from .utilities import (
    _get_hot_and_cold_utilities,
    _set_utilities_for_zone_and_subzones,
)

__all__ = ["prepare_problem"]


def prepare_problem(
    streams: Optional[List[StreamSchema]] = None,
    utilities: Optional[List[UtilitySchema]] = None,
    options: Optional[Dict[str, Any]] = None,
    project_name: str = "Site",
    zone_tree: ZoneTreeSchema = None,
) -> Zone:
    """Build the top-level zone hierarchy for analysis."""
    streams = [] if streams is None else list(streams)
    utilities = [] if utilities is None else list(utilities)

    top_zone_name, top_zone_identifier = _canonicalization._get_validated_zone_info(
        zone_tree,
        project_name,
    )
    config = _canonicalization._build_zone_config(
        options=options,
        top_zone_name=top_zone_name,
        top_zone_identifier=top_zone_identifier,
    )
    zone_tree, streams, utilities, config = _canonicalization._validate_input_data(
        zone_tree=zone_tree,
        streams=streams,
        utilities=utilities,
        config=config,
    )
    master_zone = Zone(
        name=config.problem.top_zone_name,
        type=config.problem.top_zone_identifier,
        config=config,
    )
    master_zone = _canonicalization._create_nested_zones(
        parent_zone=master_zone,
        zone_tree=zone_tree,
        config=master_zone.config,
    )
    # Zone multipliers must be set before the streams are built: each stream
    # takes its zone's multiplier, and the default utilities are placed from
    # the shifted process temperatures.
    master_zone = _canonicalization._apply_zone_dt_cont_multiplier(
        parent_zone=master_zone,
        zone_tree=zone_tree,
    )
    prepared_streams, process_zone_paths = _build_prepared_stream_collection(
        master_zone=master_zone,
        streams=sorted(streams, key=lambda stream: stream.name),
        utilities=utilities,
    )
    master_zone = _assign_process_streams_to_subzones(
        master_zone=master_zone,
        process_streams=prepared_streams.get_process_streams(),
        process_zone_paths=process_zone_paths,
    )
    master_zone.import_hot_and_cold_streams_from_sub_zones()
    master_zone = _set_utilities_for_zone_and_subzones(
        zone=master_zone,
        hot_utilities=prepared_streams.get_hot_utility_streams(),
        cold_utilities=prepared_streams.get_cold_utility_streams(),
    )
    master_zone = _canonicalization._apply_zone_dt_cont_multiplier(
        parent_zone=master_zone,
        zone_tree=zone_tree,
    )
    return master_zone


def _build_prepared_stream_collection(
    master_zone: Zone,
    streams: List[StreamSchema],
    utilities: List[UtilitySchema],
) -> tuple[StreamCollection, Dict[str, str]]:
    """Build one canonical collection of prepared process and utility streams."""
    process_streams = StreamCollection()
    process_streams.set_period_context(
        period_ids=master_zone.period_ids,
        weights=master_zone.weights,
        num_periods=master_zone.num_periods,
    )
    process_zone_paths: Dict[str, str] = {}

    for stream_schema in streams:
        if stream_schema.dt_cont is None:
            # A stream without its own dt_cont uses THERMAL_DT_CONT, as
            # utilities do.
            stream_schema = stream_schema.model_copy(
                update={"dt_cont": master_zone.config.thermal.dt_cont}
            )
        zone = master_zone.get_subzone(stream_schema.zone)
        if zone is None:
            raise ValueError(
                f"Validated stream '{stream_schema.name}' could not resolve zone "
                f"'{stream_schema.zone}'."
            )
        stream_obj = _create_process_stream(
            stream=stream_schema,
            zone=zone,
        )
        stream_key = _build_process_stream_key(
            zone_path=zone.address,
            stream_obj=stream_obj,
        )
        resolved_key = process_streams.add(
            stream=stream_obj,
            key=stream_key,
            prevent_overwrite=True,
        )
        process_zone_paths[resolved_key] = zone.address

    hu_t_min, cu_t_max = _find_extreme_process_temperatures(
        hot_streams=process_streams.get_hot_process_streams(),
        cold_streams=process_streams.get_cold_process_streams(),
    )
    utility_streams = _get_hot_and_cold_utilities(
        utilities=utilities,
        hu_t_min=hu_t_min,
        cu_t_max=cu_t_max,
        config=master_zone.config,
        dt_cont_multiplier=master_zone.dt_cont_multiplier,
        placement_multiplier=_largest_dt_cont_multiplier(master_zone),
    )
    prepared_streams = process_streams + utility_streams
    return prepared_streams, process_zone_paths


def _assign_process_streams_to_subzones(
    master_zone: Zone,
    process_streams: StreamCollection,
    process_zone_paths: Dict[str, str],
) -> Zone:
    """Attach prepared process streams to their owning zones by reference."""
    for stream_key, stream_obj in process_streams.items():
        zone_path = process_zone_paths.get(stream_key)
        if zone_path is None:
            raise RuntimeError(
                f"Prepared process stream '{stream_key}' is missing a zone mapping."
            )

        zone = master_zone.get_subzone(zone_path)
        if zone is None:
            raise ValueError(
                f"Prepared process stream '{stream_obj.name}' could not resolve zone "
                f"'{zone_path}'."
            )

        if stream_obj.stream_type == StreamType.Hot.value:
            zone.hot_streams.add(stream_obj, key=stream_key, prevent_overwrite=False)
        elif stream_obj.stream_type == StreamType.Cold.value:
            zone.cold_streams.add(
                stream_obj,
                key=stream_key,
                prevent_overwrite=False,
            )
        else:
            raise ValueError(
                f"Process stream '{stream_obj.name}' must classify as Hot or Cold, "
                f"got '{stream_obj.stream_type}'."
            )
    return master_zone


def _validate_stream_temperatures(stream: StreamSchema, config=None):
    """Validate that supply and target temperatures align with stream type."""
    # Compare in canonical units: 100 K and 100 degC are different temperatures.
    t_supply = standardise_input_value(
        stream.t_supply, field_name="t_supply", config=config
    ).period_values
    t_target = standardise_input_value(
        stream.t_target, field_name="t_target", config=config
    ).period_values
    heat_flow = resolve_value_array(stream.heat_flow)
    if np.all((abs(t_supply - t_target) < tol) * (heat_flow != 0.0)):
        raise ValueError(
            f"Process stream '{stream.name}' must classify as Hot or Cold."
        )


def _create_process_stream(stream: StreamSchema, zone: Zone) -> Stream:
    """Create a process :class:`Stream` from one validated schema record."""
    if stream.segments is not None or stream.profile is not None:
        return _create_segmented_process_stream(stream, zone)
    _validate_stream_temperatures(stream, zone.config)
    supply_temperature = standardise_input_value(
        stream.t_supply,
        field_name="t_supply",
        config=zone.config,
    )
    target_temperature = _widen_near_isothermal_target(
        supply_temperature,
        standardise_input_value(
            stream.t_target,
            field_name="t_target",
            config=zone.config,
        ),
        minimum_span=zone.config.thermal.dt_phase_change,
    )
    stream_obj = Stream(
        name=stream.name,
        supply_temperature=supply_temperature,
        target_temperature=target_temperature,
        supply_pressure=standardise_input_value(
            stream.p_supply,
            field_name="p_supply",
            config=zone.config,
        ),
        target_pressure=standardise_input_value(
            stream.p_target,
            field_name="p_target",
            config=zone.config,
        ),
        supply_enthalpy=standardise_input_value(
            stream.h_supply,
            field_name="h_supply",
            config=zone.config,
        ),
        target_enthalpy=standardise_input_value(
            stream.h_target,
            field_name="h_target",
            config=zone.config,
        ),
        heat_flow=standardise_input_value(
            stream.heat_flow,
            field_name="heat_flow",
            config=zone.config,
        ),
        delta_t_contribution=standardise_input_value(
            stream.dt_cont,
            field_name="dt_cont",
            config=zone.config,
        ),
        delta_t_contribution_multiplier=zone.dt_cont_multiplier,
        heat_transfer_coefficient=standardise_input_value(
            stream.htc,
            field_name="htc",
            config=zone.config,
        ),
        is_process_stream=True,
        fluid_name=stream.fluid_name,
        fluid_phase=stream.fluid_phase,
    )
    return stream_obj


def _widen_near_isothermal_target(
    supply: Value, target: Value, *, minimum_span: float
) -> Value:
    """Widen a stream's temperature span to at least ``minimum_span``.

    A near-isothermal stream (e.g. condensing steam) narrower than the problem
    table's interval tolerance would drop out of every interval and lose its
    duty. Its target moves away from the supply to ``minimum_span``
    (``THERMAL_DT_PHASE_CHANGE``); the duty is unchanged.
    """
    if minimum_span <= 0.0:
        return target
    supply_values = np.asarray(supply.period_values, dtype=float)
    target_values = np.asarray(target.period_values, dtype=float)
    supply_values, target_values = np.broadcast_arrays(supply_values, target_values)
    delta = target_values - supply_values
    narrow = np.isfinite(delta) & (np.abs(delta) > 0.0) & (np.abs(delta) < minimum_span)
    if not narrow.any():
        return target
    widened = target_values.copy()
    widened[narrow] = supply_values[narrow] + np.sign(delta[narrow]) * minimum_span
    unit = target.to_dict()["unit"]
    if widened.size == 1:
        return Value({"value": float(widened[0]), "unit": unit})
    return Value({"values": widened.tolist(), "unit": unit})


def _build_process_stream_key(zone_path: str, stream_obj: Stream) -> str:
    """Build a stable canonical key for one prepared process stream."""
    return ".".join([zone_path, stream_obj.name])


def _largest_dt_cont_multiplier(zone: Zone) -> float:
    """Largest ``dt_cont`` multiplier in ``zone`` and its subzones.

    Every zone gets its own copy of the utilities, shifted by that zone's
    multiplier, so the default utilities are placed for the largest one; in
    every other zone they then sit beyond the process temperatures.
    """
    return max(
        [float(zone.dt_cont_multiplier)]
        + [_largest_dt_cont_multiplier(sub) for sub in zone.subzones.values()]
    )


def _find_extreme_process_temperatures(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
) -> Tuple[float, float]:
    """Find cold-stream high and hot-stream low target temperatures."""
    if len(hot_streams) == 0 and len(cold_streams) == 0:
        return 20, 20
    hu_t_min: float = None
    cu_t_max: float = None
    stream: Stream
    for stream in hot_streams:
        thermal_segments = stream.segments or (stream,)
        for segment in thermal_segments:
            segment_min = segment.shifted_minimum_temperature.min
            if cu_t_max is None or cu_t_max > segment_min:
                cu_t_max = segment_min
    for stream in cold_streams:
        thermal_segments = stream.segments or (stream,)
        for segment in thermal_segments:
            segment_max = segment.shifted_maximum_temperature.max
            if hu_t_min is None or hu_t_min < segment_max:
                hu_t_min = segment_max
    if hu_t_min is None:
        hu_t_min = cu_t_max
    if cu_t_max is None:
        cu_t_max = hu_t_min
    return float(hu_t_min), float(cu_t_max)
