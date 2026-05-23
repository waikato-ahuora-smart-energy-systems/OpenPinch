"""Canonical payload and zone-tree preparation helpers."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, List, Optional, Tuple

from ...classes.zone import Zone
from ...lib.config import Configuration
from ...lib.enums import ZT
from ...lib.schemas.io import StreamSchema, TargetInput, UtilitySchema, ZoneTreeSchema


def canonical_problem_payload(
    input_data: TargetInput,
    *,
    project_name: str,
) -> dict[str, Any]:
    """Build a canonical mutable payload with an explicit normalized zone tree."""
    payload = copy.deepcopy(input_data.model_dump(mode="python"))
    stream_models = [stream.model_copy(deep=True) for stream in input_data.streams]
    zone_tree = (
        input_data.zone_tree.model_copy(deep=True)
        if input_data.zone_tree is not None
        else None
    )
    canonical_zone_tree = _validate_zone_tree_structure(
        zone_tree,
        stream_models,
        project_name,
    )
    payload["streams"] = [stream.model_dump(mode="python") for stream in stream_models]
    payload["zone_tree"] = canonical_zone_tree.model_dump(mode="python")
    return payload


def _build_zone_config(
    *,
    options: Configuration | dict | None,
    top_zone_name: str,
    top_zone_identifier: str,
) -> Configuration:
    """Construct a zone config without discarding caller-provided settings."""
    if options is None:
        return Configuration(
            top_zone_name=top_zone_name,
            top_zone_identifier=top_zone_identifier,
        )

    if isinstance(options, Configuration):
        zone_config = copy.deepcopy(options)
        zone_config.TOP_ZONE_NAME = top_zone_name
        zone_config.TOP_ZONE_IDENTIFIER = top_zone_identifier
        return zone_config

    if isinstance(options, dict):
        return Configuration(
            options=options,
            top_zone_name=top_zone_name,
            top_zone_identifier=top_zone_identifier,
        )

    raise TypeError("options must be a Configuration, dict, or None.")


def _get_validated_zone_info(
    zone_tree: ZoneTreeSchema, project_name: str = None, depth: int = 0
) -> Tuple[str, str]:
    """Get from input data (zone_tree) the type for the top level zone."""
    if isinstance(zone_tree, ZoneTreeSchema):
        normalized_type = (zone_tree.type or "").strip()
        type_map = {
            "Zone": ZT.P.value,
            "Sub-Zone": ZT.P.value,
            "Process Zone": ZT.P.value,
            "Unit Operation": ZT.O.value,
            "Site": ZT.S.value,
            "Community": ZT.C.value,
            "Region": ZT.R.value,
            "Utility Zone": ZT.U.value,
        }
        if normalized_type:
            try:
                zone_type = type_map[normalized_type]
            except KeyError as exc:
                raise ValueError(
                    "Zone name and type could not be identified correctly."
                ) from exc
        else:
            zone_type = None

        if not normalized_type or normalized_type == "Zone":
            if depth == 0:
                zone_type = ZT.S.value
            elif depth == 1:
                zone_type = ZT.P.value
            else:
                zone_type = ZT.O.value

        zone_name = zone_tree.name
    else:
        zone_type = ZT.S.value
        zone_name = project_name
    return zone_name, zone_type


def _validate_input_data(
    zone_tree: ZoneTreeSchema = None,
    streams: Optional[List[StreamSchema]] = None,
    utilities: Optional[List[UtilitySchema]] = None,
    zone_config: Optional[Configuration] = None,
):
    """Check and complete input data where safe defaults are available."""
    streams_list = [] if streams is None else list(streams)
    utilities_list = [] if utilities is None else list(utilities)
    cfg = zone_config or Configuration()

    streams_list = _validate_streams_passed_in(streams_list)
    utilities_list = _validate_utilities_passed_in(utilities_list)
    cfg = _validate_config_data_completed(cfg)
    zone_tree = _validate_zone_tree_structure(
        zone_tree, streams_list, cfg.TOP_ZONE_NAME
    )
    return zone_tree, streams_list, utilities_list, cfg


def _create_nested_zones(
    parent_zone: Zone, zone_tree: ZoneTreeSchema, zone_config: Configuration
) -> Zone:
    """Recursively construct a Zone hierarchy from a ZoneTreeSchema."""
    if not zone_tree.children:
        return parent_zone

    for child_schema in zone_tree.children:
        child_zone = Zone(
            name=child_schema.name,
            type=child_schema.type,
            zone_config=zone_config,
            parent_zone=parent_zone,
        )
        parent_zone.add_zone(child_zone, sub=True)
        _create_nested_zones(child_zone, child_schema, zone_config)

    return parent_zone


def _resolve_zone_dt_cont_multiplier(
    zone_tree: ZoneTreeSchema | None,
    inherited_multiplier: float | None,
) -> float:
    """Resolve a zone's effective ``dt_cont`` multiplier with inheritance."""
    if zone_tree is None or zone_tree.dt_cont_multiplier is None:
        return 1.0 if inherited_multiplier is None else float(inherited_multiplier)
    value = float(zone_tree.dt_cont_multiplier)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("dt_cont_multiplier must be a finite non-negative value.")
    return value


def _apply_zone_dt_cont_multiplier(
    parent_zone: Zone,
    zone_tree: ZoneTreeSchema,
    inherited_multiplier: float | None = None,
) -> Zone:
    """Recursively apply dt_cont_multiplier values from the zone tree to the Zone hierarchy."""
    parent_zone.dt_cont_multiplier = _resolve_zone_dt_cont_multiplier(
        zone_tree=zone_tree, 
        inherited_multiplier=inherited_multiplier,
    )
    if zone_tree.children is None:
        return parent_zone
    
    for child_schema in zone_tree.children:
        try:
            child_zone = parent_zone.get_subzone(loc=child_schema.name)
        except:
            child_zone = parent_zone
        _apply_zone_dt_cont_multiplier(
            parent_zone=child_zone,
            zone_tree=child_schema,
            inherited_multiplier=parent_zone.dt_cont_multiplier,
        )
    return parent_zone


def _rewrite_stream_zones_from_tree(
    zone_tree: ZoneTreeSchema, streams: Optional[List[StreamSchema]]
) -> None:
    """Rewrite stream.zone values to the fully-qualified zone-tree path."""
    if not streams:
        return

    all_paths: List[List[str]] = []
    path_to_node: dict[Tuple[str, ...], ZoneTreeSchema] = {}

    def _collect_paths(node: ZoneTreeSchema, prefix: List[str]) -> None:
        current_path = prefix + [node.name]
        all_paths.append(current_path)
        path_to_node[tuple(current_path)] = node
        for child in node.children or []:
            _collect_paths(child, current_path)

    _collect_paths(zone_tree, [])

    canonical_paths = {"/".join(path) for path in all_paths}
    root_name = zone_tree.name
    if zone_tree.children is None:
        zone_tree.children = []
    root_child_names = {child.name for child in zone_tree.children}

    for stream in streams:
        zone_label = getattr(stream, "zone", None)
        if not zone_label:
            continue
        components = [part.strip() for part in zone_label.split("/") if part.strip()]
        if not components:
            continue

        label_joined = "/".join(components)
        components_tuple = tuple(components)

        if len(components_tuple) == 1 and components_tuple[0] == root_name:
            base_name = stream.name or f"{root_name}_Process"
            process_name = base_name
            counter = 1
            while process_name in root_child_names or process_name == root_name:
                counter += 1
                process_name = f"{base_name}_{counter}"

            new_node = ZoneTreeSchema(name=process_name, type=ZT.P.value, children=None)
            zone_tree.children.append(new_node)
            root_child_names.add(process_name)

            new_path = (root_name, process_name)
            path_to_node[new_path] = new_node
            all_paths.append(list(new_path))
            canonical_paths.add("/".join(new_path))

            stream.zone = "/".join(new_path)
            continue

        if label_joined in canonical_paths:
            stream.zone = label_joined
            continue

        candidate_paths: List[Tuple[str, ...]] = []
        for path_tuple in path_to_node.keys():
            if len(path_tuple) >= len(components_tuple) and list(path_tuple)[
                -len(components_tuple) :
            ] == list(components_tuple):
                candidate_paths.append(path_tuple)

        if len(candidate_paths) == 1:
            stream.zone = "/".join(candidate_paths[0])


def _validate_zone_tree_structure(
    zone_tree: ZoneTreeSchema = None,
    streams: Optional[List[StreamSchema]] = None,
    top_zone_name: str = None,
) -> ZoneTreeSchema:
    """Normalise a provided zone tree or synthesise one from stream zone paths."""
    if isinstance(zone_tree, ZoneTreeSchema):
        if zone_tree.type == ZT.U.value:
            raise ValueError("Pinch analysis does not apply to Utility Zones.")

        def _check_zone_tree(
            parent_schema: ZoneTreeSchema, depth: int = 0
        ) -> ZoneTreeSchema:
            zone_name, zone_type = _get_validated_zone_info(parent_schema, depth=depth)
            parent_schema.name = zone_name
            parent_schema.type = zone_type
            if parent_schema.dt_cont_multiplier is not None:
                parent_schema.dt_cont_multiplier = _resolve_zone_dt_cont_multiplier(
                    parent_schema,
                    inherited_multiplier=None,
                )

            if not parent_schema.children:
                return parent_schema

            for child_schema in parent_schema.children:
                _check_zone_tree(child_schema, depth + 1)

            return parent_schema

        zone_tree = _check_zone_tree(zone_tree, depth=0)
        _rewrite_stream_zones_from_tree(zone_tree, streams)
        return zone_tree

    if not isinstance(top_zone_name, str):
        top_zone_name = ZT.S.value

    root = {"name": top_zone_name, "type": ZT.S.value, "children": {}}
    stream_iter = [stream for stream in (streams or []) if stream.zone]

    def _split_zone_name(name: str):
        if "/" in name:
            return [z.strip() for z in name.split("/") if z.strip()]
        return [name]

    zone_counters = defaultdict(int)

    def _build_full_path(path_components: List[str], operation_zone_name: str) -> str:
        if path_components:
            return "/".join([root["name"], *path_components, operation_zone_name])
        return "/".join([root["name"], operation_zone_name])

    for stream in sorted(stream_iter, key=lambda item: (item.zone, item.name)):
        z_path = _split_zone_name(stream.zone)
        current = root
        path_components: List[str] = []
        for zone_name in z_path:
            if zone_name not in current["children"]:
                current["children"][zone_name] = {
                    "name": zone_name,
                    "type": ZT.P.value,
                    "children": {},
                }
            current = current["children"][zone_name]
            path_components.append(zone_name)

        zone_key = tuple(path_components)
        zone_counters[zone_key] += 1

        operation_zone_name = f"O{zone_counters[zone_key]}"
        while operation_zone_name in current["children"]:
            zone_counters[zone_key] += 1
            operation_zone_name = f"O{zone_counters[zone_key]}"

        current["children"][operation_zone_name] = {
            "name": operation_zone_name,
            "type": ZT.O.value,
            "children": {},
        }
        stream.zone = _build_full_path(path_components, operation_zone_name)

    def _build_tree(node_dict):
        children = [_build_tree(child) for child in node_dict["children"].values()]
        return ZoneTreeSchema(
            name=node_dict["name"],
            type=node_dict["type"],
            children=children if children else None,
        )

    return ZoneTreeSchema.model_validate(_build_tree(root))


def _validate_streams_passed_in(streams: List[StreamSchema]) -> list:
    """Raises an error if no streams are passed in."""
    if len(streams) == 0:
        raise ValueError("At least one stream is required")
    return streams


def _validate_utilities_passed_in(utilities: List[UtilitySchema]) -> list:
    """Check if any utilities are passed in."""
    return [] if utilities is None else utilities


def _validate_config_data_completed(zone_config: Configuration) -> Configuration:
    """Validates that the configuration settings make logical sense."""
    if (
        not isinstance(zone_config.ANNUAL_OP_TIME, (int, float))
        or zone_config.ANNUAL_OP_TIME == 0
    ):
        zone_config.ANNUAL_OP_TIME = 365 * 24
    if zone_config.DO_TURBINE_WORK and zone_config.TURB_P_IN > 220:
        zone_config.TURB_P_IN = 200
    if zone_config.DT_PHASE_CHANGE <= 0:
        zone_config.DT_PHASE_CHANGE = 0.01
    if zone_config.DT_CONT < 0:
        zone_config.DT_CONT = 0.0
    return zone_config
