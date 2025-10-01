import copy
from typing import List, Tuple
from ..lib import *
from .support_methods import get_value
from ..classes import Zone, Stream, StreamCollection


__all__ = ["prepare_problem_struture"]

#######################################################################################################
# Public API
#######################################################################################################

def prepare_problem_struture(
    streams: List[StreamSchema] = [],
    utilities: List[UtilitySchema] = [],
    options: Configuration = None,
    project_name: str = "Site",
    zone_tree: ZoneTreeSchema = None,
):
    """Build the top-level :class:`OpenPinch.classes.zone.Zone` hierarchy for analysis.

    Parameters
    ----------
    streams:
        Iterable of validated :class:`OpenPinch.lib.schema.StreamSchema` objects describing
        the process streams to analyse.
    utilities:
        Iterable of :class:`OpenPinch.lib.schema.UtilitySchema` describing candidate hot
        and cold utilities.
    options:
        Optional :class:`OpenPinch.lib.config.Configuration` overrides.  When omitted the
        defaults from ``Configuration()`` are used.
    project_name:
        Human-friendly label applied to the root zone when no explicit zone tree is supplied.
    zone_tree:
        Optional :class:`OpenPinch.lib.schema.ZoneTreeSchema` describing the desired zone
        hierarchy (i.e., Zone A encompasses Zones B and C).

    Returns
    -------
    Zone
        Fully initialised zone tree with streams and utilities attached to each node.
    """
    top_zone_name, top_zone_identifier = _get_validated_zone_info(zone_tree, project_name)
    config = Configuration(
        options=options, 
        top_zone_name=top_zone_name, 
        top_zone_identifier=top_zone_identifier,
        )
    zone_tree, streams, utilities, config = _validate_input_data(zone_tree, streams, utilities, config)
    master_zone = Zone(name=config.TOP_ZONE_NAME, identifier=config.TOP_ZONE_IDENTIFIER, config=config)
    master_zone = _create_nested_zones(master_zone, zone_tree, master_zone.config)
    master_zone = _get_process_streams_in_each_subzone(master_zone, sorted(streams, key=lambda x: x.name))
    master_zone.import_hot_and_cold_streams_from_sub_zones()
    hot_utilities, cold_utilities = _get_hot_and_cold_utilities(utilities, master_zone.hot_streams, master_zone.cold_streams, master_zone.config)
    master_zone = _set_utilities_for_zone_and_subzones(master_zone, hot_utilities, cold_utilities)
    return master_zone

#######################################################################################################
# Helper Functions
#######################################################################################################

def _get_validated_zone_info(zone_tree: ZoneTreeSchema, project_name: str = None) -> Tuple[str, str]:
    """Get from input data (zone_tree) the identifier/type for the top level zone."""
    if isinstance(zone_tree, ZoneTreeSchema):
        if zone_tree.type in ["Zone", "Sub-Zone", "Process Zone"]:
            zone_type = ZoneType.P.value
        elif zone_tree.type == "Site":
            zone_type = ZoneType.S.value
        elif zone_tree.type == "Community":
            zone_type = ZoneType.C.value
        elif zone_tree.type == "Region":
            zone_type = ZoneType.R.value
        elif zone_tree.type == "Utility Zone":
            zone_type = ZoneType.U.value
        else:
            raise ValueError("Zone name and type could not be identified correctly.")
        zone_name = zone_tree.name
    else: 
        zone_type = ZoneType.S.value
        zone_name = project_name
    return zone_name, zone_type


def _validate_input_data(zone_tree: ZoneTreeSchema = None, streams: List[StreamSchema] = [], utilities: List[UtilitySchema] = [], config: Configuration = Configuration()):
    """Checks for logic and completeness of the input data. Where possible, fills in the gaps with general assumptions."""
    streams = _validate_streams_passed_in(streams)
    utilities = _validate_utilities_passed_in(utilities)
    config = _validate_config_data_completed(config)
    zone_tree = _validate_zone_tree_structure(zone_tree, streams, config.TOP_ZONE_NAME)
    return zone_tree, streams, utilities, config


def _create_nested_zones(parent_zone: Zone, zone_tree: ZoneTreeSchema, config: Configuration) -> Zone:
    """Recursively construct a Zone hierarchy from a ZoneTreeSchema."""
    if not zone_tree.children:
        return parent_zone

    for child_schema in zone_tree.children:
        child_zone = Zone(
            name=child_schema.name,
            identifier=child_schema.type,
            config=config,
            parent_zone=parent_zone,
        )
        parent_zone.add_zone(child_zone, sub=True)
        _create_nested_zones(child_zone, child_schema, config)

    return parent_zone


def _get_process_streams_in_each_subzone(master_zone: Zone, streams: List[StreamSchema]) -> Zone:
    """Extracts all stream data into class instances, creates the required subzones and adds these to the parent zone."""
    
    def _flatten_zone_hierarchy(parent_zone: Zone) -> List[Zone]:
        """Recursively flattens the zone tree starting from parent_zone into a list of all Zone objects."""
        zones = [parent_zone]
        for subzone in parent_zone.subzones.values():
            zones.extend(_flatten_zone_hierarchy(subzone))
        return zones
    
    flat_zones = _flatten_zone_hierarchy(master_zone)
    for z in flat_zones:
        _add_process_streams_under_zones(z, streams)
    return master_zone


def _create_process_stream(stream: StreamSchema) -> Stream:
    """Creates a Stream instance from StreamSchema."""
    # Create and initialise stream
    return Stream(
        name=stream.name,
        t_supply = get_value(stream.t_supply),
        t_target = get_value(stream.t_target),
        heat_flow = get_value(stream.heat_flow),
        dt_cont = get_value(stream.dt_cont),
        htc = get_value(stream.htc),
        is_process_stream = True,
    )


def _add_process_streams_under_zones(z: Zone, streams: List[StreamSchema]) -> Zone:
    """Adds hot and cold streams to the given zone."""
    stream_j: Stream

    def _get_zone_path_from_child(child_zone: Zone, delimiter="/") -> str:
        """Constructs the zone path from a child Zone back to the master zone using parent_zone links."""
        path_parts = []
        current = child_zone
        while current is not None:
            path_parts.append(current.name)
            current = current.parent_zone
        return delimiter.join(reversed(path_parts))

    zone_path = _get_zone_path_from_child(z)

    for s in streams:
        stream_zone_name = s.zone.split("/")[-1]
        if stream_zone_name == z.name or s.zone == z.name or s.zone == zone_path: #or z.name == TargetType.DI.value
            # Create Stream from Data
            stream_j = _create_process_stream(s)
            if stream_j.type==StreamType.Hot.value:
                key = ".".join([s.zone, StreamLoc.HotStr.value, s.name])
                z.hot_streams.add(stream_j, key)
            else:
                key = ".".join([s.zone, StreamLoc.ColdStr.value, s.name])
                z.cold_streams.add(stream_j)
    return z


def _get_hot_and_cold_utilities(utilities: List[UtilitySchema], hot_streams: List[Stream], cold_streams: List[Stream], config: Configuration) -> Tuple[List[Stream], List[Stream]]:
    """Extracts all utility data into class instances."""
    HU_T_min, CU_T_max = _find_extreme_process_temperatures(hot_streams, cold_streams)
    utilities, addDefaultHU, addDefaultCU = _complete_utility_data(utilities, config, HU_T_min, CU_T_max)
    utilities = _add_default_utilities(utilities, config, addDefaultHU, addDefaultCU, HU_T_min, CU_T_max)
    hot_utilities, utilities = _create_utilities_list(utilities, utility_type=StreamType.Hot.value)
    cold_utilities, utilities = _create_utilities_list(utilities, utility_type=StreamType.Cold.value)
    return hot_utilities, cold_utilities


def _find_extreme_process_temperatures(hot_streams: List[Stream], cold_streams: List[Stream]) -> Tuple[float, float]:
    """Find highest TT of a cold stream and lowest TT of a hot stream."""
    HU_T_min: float = -1e9
    CU_T_max: float = 1e9
    s: Stream
    for s in hot_streams:
        if CU_T_max > s.t_min_star:
            CU_T_max = s.t_min_star
    for s in cold_streams:
        if HU_T_min < s.t_max_star:
            HU_T_min = s.t_max_star
    return HU_T_min, CU_T_max


def _complete_utility_data(utilities: List[UtilitySchema], config: Configuration, HU_T_min: float, CU_T_max: float) -> Tuple[List[UtilitySchema], bool, bool]:
    """Completes the utility data with default values and adds default utilities if needed."""
    utility: UtilitySchema
    
    # Fill in any missing data
    addDefaultHU = True
    addDefaultCU = True
    
    # Set Defaults
    for utility in utilities:
        utility.t_supply = get_value(utility.t_supply)
        if get_value(utility.t_target) == None or get_value(utility.t_target) == get_value(utility.t_supply):
            utility.t_target = utility.t_supply - config.DTGLIDE if utility.type == "Hot" else utility.t_supply + config.DTGLIDE
        else:
            utility.t_target = get_value(utility.t_target)
        if get_value(utility.dt_cont) == None:
            utility.dt_cont = config.DTCONT
        else:
            utility.dt_cont = get_value(utility.dt_cont)
        if get_value(utility.price) == None:
            utility.price = config.UTILITY_PRICE * config.ANNUAL_OP_TIME
        else:
            utility.price = get_value(utility.price)
        if get_value(utility.htc) == None or get_value(utility.htc) == 0:
            utility.htc = config.HTC
        else:
            utility.htc = get_value(utility.htc)
        if (utility.type in ["Hot", "Both"] and utility.active and min(utility.t_supply, utility.t_target) - utility.dt_cont >= HU_T_min):
            addDefaultHU = False
        if (utility.type in ["Cold", "Both"] and utility.active and max(utility.t_supply, utility.t_target) - utility.dt_cont <= CU_T_max):
            addDefaultCU = False
    return utilities, addDefaultHU, addDefaultCU


def _add_default_utilities(utilities: List[UtilitySchema], config: Configuration, addDefaultHU: bool, addDefaultCU: bool,  HU_T_min: float, CU_T_max: float) -> List[UtilitySchema]:
    """Adds default hot and cold utilities to the list of utilities."""
    # Add default hot and cold utilities
    if addDefaultHU:
        utilities.append(
            _create_default_utility("HU", "Hot", HU_T_min, config)
        )
    if addDefaultCU:
        utilities.append(
            _create_default_utility("CU", "Cold", CU_T_max, config)
        )
    return utilities


def _create_default_utility(name: str, ut_type: str, T: float, config: Configuration) -> UtilitySchema:
    """Construct a default utility entry anchored to the extreme process temperature."""
    a = 1 if ut_type == "Hot" else -1
    return UtilitySchema.model_validate(
        {
            "name": name,
            "type": ut_type,
            "t_supply": T + (config.DTCONT) * a,
            "t_target": T + (config.DTCONT - config.DTGLIDE) * a,
            "heat_flow": 0,
            "dt_cont": config.DTCONT,
            "price": config.UTILITY_PRICE,
            "htc": config.HTC,
        }
    )


def _create_utilities_list(utilities: List[UtilitySchema], utility_type: str) -> Tuple[List[Stream], List[UtilitySchema]]:
    """Creates a sorted list of hot or cold Stream objects based on type."""
    created_utilities = StreamCollection()
    T_prev = 1e9

    # Find the first utility of the specified type
    for selected in utilities:
        if selected.type in ["Both", utility_type] and selected.active:
            break

    prev_selected_name = ""

    for _ in range(len(utilities)):
        #Cycle through all utilities as candidates, comparing each candidate agaist the previous utility selected and the best found (U)
        for candidate in utilities:
            is_valid = (
                candidate.type in ["Both", utility_type]                
                and candidate.t_supply < T_prev
                and (candidate.t_supply >= selected.t_supply or selected.name == prev_selected_name)
                and candidate.active
            )
            if is_valid:
                selected = candidate
                
        # If no different utility is identified, break
        if selected.name == prev_selected_name: 
            break

        T_prev = selected.t_supply
        if selected.type in [utility_type]:
            selected.active = False

        if utility_type == "Hot":
            t_supply = max(selected.t_supply, selected.t_target)
            t_target = min(selected.t_supply, selected.t_target)
        else:
            t_supply = min(selected.t_supply, selected.t_target)
            t_target = max(selected.t_supply, selected.t_target)

        # Create utility
        key = ".".join([StreamLoc.HotU.value, selected.name]) if utility_type == StreamType.Hot.value else ".".join([StreamLoc.ColdU.value, selected.name])
        created_utilities.add(
            Stream(
                selected.name,
                t_supply,
                t_target,
                selected.dt_cont,
                htc=selected.htc,
                price=selected.price,
                is_process_stream=False,
            ),
            key
        )
        prev_selected_name = selected.name

    return created_utilities, utilities


def _set_utilities_for_zone_and_subzones(zone: Zone, hot_utilities: List[Stream], cold_utilities: List[Stream]) -> Zone:
    """Adds hot and cold utilities to the zone and each subzone under zone."""
    zone.hot_utilities.add_many(copy.deepcopy(hot_utilities))
    zone.cold_utilities.add_many(copy.deepcopy(cold_utilities))
    for subzone in zone.subzones.values():
        subzone = _set_utilities_for_zone_and_subzones(subzone, hot_utilities, cold_utilities)
    return zone


def _validate_zone_tree_structure(zone_tree: ZoneTreeSchema = None, streams: List[StreamSchema] = [], top_zone_name: str = None) -> ZoneTreeSchema:
    """Normalise a provided zone tree or synthesise one from stream zone paths."""
    if isinstance(zone_tree, ZoneTreeSchema):
        if zone_tree.type == ZoneType.U.value:
            raise ValueError("Pinch analysis does not apply to Utility Zones.")
        
        def _check_zone_tree(parent_schema: ZoneTreeSchema) -> ZoneTreeSchema:
            """Recursively construct a Zone hierarchy from a ZoneTreeSchema."""
            zone_name, zone_type = _get_validated_zone_info(parent_schema)
            parent_schema.name = zone_name
            parent_schema.type = zone_type

            if not parent_schema.children:
                return parent_schema

            for child_schema in parent_schema.children:
                child_schema = _check_zone_tree(child_schema)

            return parent_schema

        zone_tree = _check_zone_tree(zone_tree)
        return zone_tree

    # Build zone tree from stream zone names
    if not isinstance(top_zone_name, str):
        top_zone_name = ZoneType.S.value

    root = {"name": top_zone_name, "type": ZoneType.S.value, "children": {}}
    zone_names = sorted(set(stream.zone for stream in streams if stream.zone))  # Filter empty/null zones

    def _split_zone_name(name: str):
        """Split hierarchical zone strings ("Site/Area/Unit") into clean path components."""
        if "/" in name:
            return [z.strip() for z in name.split("/") if z.strip()]
        return [name]

    for zone_name in zone_names:
        z_path = _split_zone_name(zone_name)
        current = root
        for i, z_name in enumerate(z_path):
            if z_name not in current["children"]:
                current["children"][z_name] = {
                    "name": z_name,
                    "type": ZoneType.P.value,
                    "children": {}
                }
            current = current["children"][z_name]

    def _build_tree(node_dict):
        """Recursively convert the intermediate dict representation into Pydantic models."""
        children = [
            _build_tree(child) for child in node_dict["children"].values()
        ]
        return ZoneTreeSchema(
            name=node_dict["name"],
            type=node_dict["type"],
            children=children if children else None
        )

    return ZoneTreeSchema.model_validate(
        _build_tree(root)
    )


def _validate_streams_passed_in(streams: List[StreamSchema]) -> list:
    """Raises an error if no streams are passed in."""
    if len(streams) == 0:
        raise ValueError("At least one stream is required")
    return streams

    
def _validate_utilities_passed_in(utilities: List[UtilitySchema]) -> list:
    """Check if any utilities are passed in"""
    return [] if utilities is None else utilities


def _validate_config_data_completed(config: Configuration) -> Configuration:
    """Validates that the configuration settings make logical sense."""
    # Check if annual operation time is set    
    if isinstance(config.ANNUAL_OP_TIME, float | int) == False or config.ANNUAL_OP_TIME == 0:
        config.ANNUAL_OP_TIME = 365 * 24 #h/y
    # Ensures the inlet pressure to the turbine is below the critical pressure   
    # TODO: Add units to the turbine pressure     
    if config.TURBINE_WORK_BUTTON is True and config.P_TURBINE_BOX > 220:
        config.P_TURBINE_BOX = 200 
    if config.DTGLIDE <= 0:
        config.DTGLIDE = 0.01
    if config.DTCONT < 0:
        config.DTCONT = 0.0
    return config
