"""Construct validated :class:`Zone` trees and attach stream/utility data."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from ...classes.stream import Stream
from ...classes.zone import Zone
from ...lib.enums import ST, StreamLoc
from ...lib.schemas.io import StreamSchema, UtilitySchema, ZoneTreeSchema
from ...utils.miscellaneous import get_value
from ._canonicalization import (
    _build_zone_config,
    _create_nested_zones,
    _get_validated_zone_info,
    _resolve_zone_dt_cont_multiplier,
    _validate_config_data_completed,
    _validate_input_data,
    _validate_zone_tree_structure,
)
from ._utility_preparation import _set_utilities_for_zone_and_subzones

__all__ = [
    "prepare_problem",
    "_build_zone_config",
    "_create_nested_zones",
    "_get_validated_zone_info",
    "_resolve_zone_dt_cont_multiplier",
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
        zone_tree,
        streams,
        utilities,
        zone_config,
    )
    master_zone = Zone(
        name=zone_config.TOP_ZONE_NAME,
        type=zone_config.TOP_ZONE_IDENTIFIER,
        zone_config=zone_config,
        dt_cont_multiplier=_resolve_zone_dt_cont_multiplier(zone_tree, None),
        lock_dt_cont_multiplier=True,
    )
    master_zone = _create_nested_zones(master_zone, zone_tree, master_zone.config)
    master_zone = _get_process_streams_in_each_subzone(
        master_zone,
        sorted(streams, key=lambda stream: stream.name),
    )
    master_zone.import_hot_and_cold_streams_from_sub_zones()
    return _set_utilities_for_zone_and_subzones(master_zone, utilities)


def _get_process_streams_in_each_subzone(
    master_zone: Zone,
    streams: List[StreamSchema],
) -> Zone:
    """Create stream objects, subzones, and zone attachments for the hierarchy."""
    streams_by_full_path = defaultdict(list)
    streams_by_relative_path = defaultdict(list)
    for stream in streams:
        zone_path = getattr(stream, "zone", None)
        if not zone_path:
            continue
        streams_by_full_path[zone_path].append(stream)
        path_components = zone_path.split("/")
        for index in range(1, len(path_components)):
            relative_key = "/".join(path_components[index:])
            streams_by_relative_path[relative_key].append(stream)
        streams_by_relative_path[zone_path].append(stream)

    def iter_zones(parent_zone: Zone):
        yield parent_zone
        for subzone in parent_zone.subzones.values():
            yield from iter_zones(subzone)

    def zone_path_from_child(child_zone: Zone, delimiter="/") -> str:
        path_parts = []
        current = child_zone
        while current is not None:
            path_parts.append(current.name)
            current = current.parent_zone
        return delimiter.join(reversed(path_parts))

    for zone in iter_zones(master_zone):
        zone_path = zone_path_from_child(zone)
        relative_zone_path = (
            zone_path.split("/", 1)[1] if "/" in zone_path else zone_path
        )

        matched_streams: List[StreamSchema] = []
        seen = set()

        for candidate in streams_by_full_path.get(zone_path, ()):
            candidate_id = id(candidate)
            if candidate_id not in seen:
                matched_streams.append(candidate)
                seen.add(candidate_id)

        for candidate in streams_by_relative_path.get(relative_zone_path, ()):
            candidate_id = id(candidate)
            if candidate_id not in seen:
                matched_streams.append(candidate)
                seen.add(candidate_id)

        if not matched_streams:
            continue

        for stream_schema in matched_streams:
            stream_obj = _create_process_stream(stream_schema, zone)
            if stream_obj.type == ST.Hot.value:
                key = ".".join(
                    [stream_schema.zone, StreamLoc.HotS.value, stream_schema.name]
                )
                zone.hot_streams.add(stream_obj, key)
            else:
                zone.cold_streams.add(stream_obj)
    return master_zone


def _create_process_stream(stream: StreamSchema, zone: Zone) -> Stream:
    """Create a process :class:`Stream` from one validated schema record."""
    base_dt_cont = get_value(stream.dt_cont)
    if base_dt_cont is None:
        base_dt_cont = 0.0

    return Stream(
        name=stream.name,
        t_supply=get_value(stream.t_supply),
        t_target=get_value(stream.t_target),
        heat_flow=get_value(stream.heat_flow),
        dt_cont=base_dt_cont,
        dt_cont_act=base_dt_cont * zone.dt_cont_multiplier,
        htc=get_value(stream.htc),
        is_process_stream=True,
    )
