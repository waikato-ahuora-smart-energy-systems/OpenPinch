"""Construct validated :class:`Zone` trees and attach stream/utility data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.enums import ST
from ...lib.schemas.io import StreamSchema, UtilitySchema, ZoneTreeSchema
from ...lib.unit_system import standardise_input_value
from ...utils.value_resolution import resolve_value_array
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
    options: Optional[Dict[str, Any]] = None,
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
    config = _build_zone_config(
        options=options,
        top_zone_name=top_zone_name,
        top_zone_identifier=top_zone_identifier,
    )
    zone_tree, streams, utilities, config = _validate_input_data(
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
    master_zone = _create_nested_zones(
        parent_zone=master_zone,
        zone_tree=zone_tree,
        config=master_zone.config,
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
    process_streams = StreamCollection()
    process_streams.set_state_context(
        state_ids=master_zone.state_ids,
        weights=master_zone.weights,
        num_states=master_zone.num_states,
    )
    process_zone_paths: Dict[str, str] = {}

    for stream_schema in streams:
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


def _validate_stream_temperatures(stream: StreamSchema):
    """Validate that supply and target temperatures align with stream type."""
    t_supply = resolve_value_array(stream.t_supply)
    t_target = resolve_value_array(stream.t_target)
    heat_flow = resolve_value_array(stream.heat_flow)
    if np.all((abs(t_supply - t_target) < tol) * (heat_flow != 0.0)):
        raise ValueError(
            f"Process stream '{stream.name}' must classify as Hot or Cold."
        )


def _create_process_stream(stream: StreamSchema, zone: Zone) -> Stream:
    """Create a process :class:`Stream` from one validated schema record."""
    _validate_stream_temperatures(stream)
    stream_obj = Stream(
        name=stream.name,
        t_supply=standardise_input_value(
            stream.t_supply,
            field_name="t_supply",
            config=zone.config,
        ),
        t_target=standardise_input_value(
            stream.t_target,
            field_name="t_target",
            config=zone.config,
        ),
        p_supply=standardise_input_value(
            stream.p_supply,
            field_name="p_supply",
            config=zone.config,
        ),
        p_target=standardise_input_value(
            stream.p_target,
            field_name="p_target",
            config=zone.config,
        ),
        h_supply=standardise_input_value(
            stream.h_supply,
            field_name="h_supply",
            config=zone.config,
        ),
        h_target=standardise_input_value(
            stream.h_target,
            field_name="h_target",
            config=zone.config,
        ),
        heat_flow=standardise_input_value(
            stream.heat_flow,
            field_name="heat_flow",
            config=zone.config,
        ),
        dt_cont=standardise_input_value(
            stream.dt_cont,
            field_name="dt_cont",
            config=zone.config,
        ),
        dt_cont_multiplier=zone.dt_cont_multiplier,
        htc=standardise_input_value(
            stream.htc,
            field_name="htc",
            config=zone.config,
        ),
        is_process_stream=True,
        fluid_name=stream.fluid_name,
        fluid_phase=stream.fluid_phase,
    )
    return stream_obj


def _build_process_stream_key(zone_path: str, stream_obj: Stream) -> str:
    """Build a stable canonical key for one prepared process stream."""
    return ".".join([zone_path, stream_obj.name])


def _find_extreme_process_temperatures(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
) -> Tuple[float, float]:
    """Find highest TT of a cold stream and lowest TT of a hot stream."""
    if len(hot_streams) == 0 and len(cold_streams) == 0:
        return 20, 20
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
    return float(hu_t_min), float(cu_t_max)
