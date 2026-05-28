"""Construct validated :class:`Zone` trees and attach stream/utility data."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.zone import Zone
from ...lib.enums import ST, StreamLoc
from ...lib.schemas.io import StreamSchema, UtilitySchema, ZoneTreeSchema
from ...utils.miscellaneous import get_value
from ._canonicalization import (
    _apply_zone_dt_cont_multiplier,
    _build_zone_config,
    _create_nested_zones,
    _get_validated_zone_info,
    _validate_config_data_completed,
    _validate_input_data,
    _validate_zone_tree_structure,
)
from ._utility_preparation import (
    _get_hot_and_cold_utilities,
    _set_utilities_for_zone_and_subzones,
)

__all__ = [
    "prepare_problem",
    "_apply_zone_dt_cont_multiplier",
    "_build_zone_config",
    "_create_nested_zones",
    "_get_validated_zone_info",
    "_validate_config_data_completed",
    "_validate_input_data",
    "_validate_zone_tree_structure",
]


def prepare_problem(
    streams: Optional[List[StreamSchema]] = None,
    utilities: Optional[List[UtilitySchema]] = None,
    options=None,
    project_name: str = "Site",
    zone_tree: ZoneTreeSchema = None,
) -> Zone:
    """Build the top-level zone hierarchy for analysis."""
    streams = [] if streams is None else list(streams)
    utilities = [] if utilities is None else list(utilities)

    top_zone_name, top_zone_identifier = _get_validated_zone_info(
        zone_tree,
        project_name,
    )
    zone_config = _build_zone_config(
        options=options,
        top_zone_name=top_zone_name,
        top_zone_identifier=top_zone_identifier,
    )
    zone_tree, streams, utilities, zone_config = _validate_input_data(
        zone_tree=zone_tree,
        streams=streams,
        utilities=utilities,
        zone_config=zone_config,
    )
    master_zone = Zone(
        name=zone_config.TOP_ZONE_NAME,
        type=zone_config.TOP_ZONE_IDENTIFIER,
        zone_config=zone_config,
    )
    master_zone = _create_nested_zones(
        parent_zone=master_zone,
        zone_tree=zone_tree,
        zone_config=master_zone.config,
    )
    prepared_streams, process_zone_paths = _build_prepared_stream_collection(
        master_zone=master_zone,
        streams=sorted(streams, key=lambda stream: stream.name),
        utilities=utilities,
    )
    prepared_streams.validate_state_alignment()
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
    master_zone = _apply_zone_dt_cont_multiplier(
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
    prepared_streams = StreamCollection()
    process_zone_paths: Dict[str, str] = {}

    for stream_schema in streams:
        zone = master_zone.get_subzone(stream_schema.zone)
        if zone is None:
            raise ValueError(
                f"Validated stream '{stream_schema.name}' could not resolve zone "
                f"'{stream_schema.zone}'."
            )

        stream_obj = _create_process_stream(stream_schema, zone)
        stream_key = _build_process_stream_key(zone.address, stream_obj)
        resolved_key = _add_stream_to_collection(
            prepared_streams,
            stream_obj,
            key=stream_key,
        )
        process_zone_paths[resolved_key] = zone.address

    hu_t_min, cu_t_max = _find_extreme_process_temperatures(
        hot_streams=prepared_streams.get_hot_process_streams(),
        cold_streams=prepared_streams.get_cold_process_streams(),
    )
    hot_utilities, cold_utilities = _get_hot_and_cold_utilities(
        utilities=utilities,
        hu_t_min=hu_t_min,
        cu_t_max=cu_t_max,
        zone_config=master_zone.config,
        dt_cont_multiplier=master_zone.dt_cont_multiplier,
    )
    _extend_stream_collection(prepared_streams, hot_utilities)
    _extend_stream_collection(prepared_streams, cold_utilities)
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

        if stream_obj.type == ST.Hot.value:
            zone.hot_streams.add(stream_obj, key=stream_key, prevent_overwrite=False)
        elif stream_obj.type == ST.Cold.value:
            zone.cold_streams.add(
                stream_obj,
                key=stream_key,
                prevent_overwrite=False,
            )
        else:
            raise ValueError(
                f"Process stream '{stream_obj.name}' must classify as Hot or Cold, "
                f"got '{stream_obj.type}'."
            )
    return master_zone


def _create_process_stream(stream: StreamSchema, zone: Zone) -> Stream:
    """Create a process :class:`Stream` from one validated schema record."""
    dt_cont = stream.dt_cont
    dt_cont_value = get_value(dt_cont)
    if dt_cont_value is None or not isinstance(dt_cont_value, (int, float)):
        dt_cont = 0.0
    elif not math.isfinite(dt_cont_value):
        dt_cont = 0.0

    htc = stream.htc
    htc_value = get_value(htc)
    if (
        not hasattr(htc, "state_ids")
        and (
            htc_value is None
            or not isinstance(htc_value, (int, float))
            or not math.isfinite(htc_value)
            or htc_value <= 0.0
        )
    ):
        htc = zone.config.HTC

    stream_obj = Stream(
        name=stream.name,
        t_supply=stream.t_supply,
        t_target=stream.t_target,
        heat_flow=stream.heat_flow,
        dt_cont=dt_cont,
        htc=htc,
        is_process_stream=True,
    )
    stream_obj.dt_cont_multiplier = zone.dt_cont_multiplier
    return stream_obj


def _build_process_stream_key(zone_path: str, stream_obj: Stream) -> str:
    """Build a stable canonical key for one prepared process stream."""
    if stream_obj.type == ST.Hot.value:
        stream_loc = StreamLoc.HotS.value
    elif stream_obj.type == ST.Cold.value:
        stream_loc = StreamLoc.ColdS.value
    else:
        raise ValueError(
            f"Process stream '{stream_obj.name}' must classify as Hot or Cold, "
            f"got '{stream_obj.type}'."
        )
    return ".".join([zone_path, stream_loc, stream_obj.name])


def _add_stream_to_collection(
    collection: StreamCollection,
    stream_obj: Stream,
    *,
    key: str | None = None,
    prevent_overwrite: bool = True,
) -> str:
    """Add a stream and return the resolved collection key."""
    return collection.add(stream_obj, key=key, prevent_overwrite=prevent_overwrite)


def _extend_stream_collection(
    destination: StreamCollection,
    source: StreamCollection,
) -> StreamCollection:
    """Copy stream references from ``source`` into ``destination`` using stable keys."""
    for key, stream_obj in source.items():
        destination.add(stream_obj, key=key, prevent_overwrite=False)
    return destination


def _find_extreme_process_temperatures(
    hot_streams: StreamCollection, cold_streams: StreamCollection
) -> Tuple[float, float]:
    """Find highest TT of a cold stream and lowest TT of a hot stream."""
    hu_t_min: float = None
    cu_t_max: float = None
    stream: Stream
    for stream in hot_streams:
        stream_min = stream.t_min_star.min
        if cu_t_max is None or cu_t_max > stream_min:
            cu_t_max = stream_min
    for stream in cold_streams:
        stream_max = stream.t_max_star.max
        if hu_t_min is None or hu_t_min < stream_max:
            hu_t_min = stream_max
    if hu_t_min is None:
        hu_t_min = cu_t_max
    if cu_t_max is None:
        cu_t_max = hu_t_min
    return hu_t_min, cu_t_max
