"""High-level orchestration for running an OpenPinch analysis.

The functions in this module wire together data validation, pinch targeting,
and output formatting.  They act as the main entry points used by the
``PinchProblem`` helper class as well as external callers embedding OpenPinch
in larger workflows.
"""

from typing import Any, List

from .lib import *
from .utils import *
from .classes import *
from .analysis.data_preparation import *
from .analysis.graph_data import *
from .analysis.direct_integration_entry import *
from .analysis.indirect_integration_entry import *

__all__ = ["pinch_analysis_service", "get_targets", "get_visualise", "extract_results"]


#######################################################################################################
# Public API
#######################################################################################################


@timing_decorator
def pinch_analysis_service(data: Any, project_name: str = "Project", is_return_full_results: bool = False) -> TargetOutput:
    """Validate user data, run the targeting workflow, and return structured results.

    Parameters
    ----------
    data:
        Raw request payload matching :class:`OpenPinch.lib.schema.TargetInput`.
        Dictionaries, Pydantic models, and dataclass-like objects are accepted.
    project_name:
        Optional label used in generated graphs and result files.

    Returns
    -------
    TargetOutput
        Pydantic model containing site, zone, and utility targets plus summary
        tables ready for serialisation.
    """
    # Validate request data using Pydantic model
    request_data = TargetInput.model_validate(data)

    # Formulate the top level zone with all subzones and approperiate input data
    master_zone = prepare_problem(
        project_name=project_name,
        streams=request_data.streams,
        utilities=request_data.utilities,
        options=request_data.options,
        zone_tree=request_data.zone_tree,
    )

    # Perform advanced targeting analysis on the master zone and all subzones
    master_zone = get_targets(master_zone)

    # Extract the core results from the master zone
    return_data = extract_results(master_zone)

    # Validate response data
    validated_data = TargetOutput.model_validate(return_data)

    # Return data
    return validated_data if not is_return_full_results else (validated_data, master_zone)


def get_targets(master_zone: Zone) -> dict:
    """Conduct core Pinch Analysis and total site targeting.

    This function is a lower-level hook compared to :func:`pinch_analysis_service`.
    It expects already validated option blocks and stream/utility schemas, and it
    returns a dictionary ready for conversion into :class:`TargetOutput`.
    """

    handler = _TARGET_HANDLERS.get(master_zone.identifier)
    if handler is None:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")

    return handler(master_zone)


def extract_results(master_zone: Zone) -> dict:
    """Serializes results data into a dictionaty from options."""
    return {
        "name": master_zone.name,
        "targets": _get_report(master_zone),
        "utilities": _get_utilities(master_zone),
        "graphs": get_output_graph_data(master_zone),
    }


########### TODO: This function is untested and not updated since the overhaul of OpenPinch. Broken, most likely.#####
def get_visualise(data) -> dict:
    """Build graph payloads directly from legacy problem-table structures.

    Notes
    -----
    This helper predates the class-based refactor and retains compatibility with
    older tooling.  It is currently untested and should be considered provisional.
    """
    r_data = {"graphs": []}
    z: Zone
    for z in data:
        graph_set = {"name": f"{z.name}", "graphs": []}
        for graph in z.graphs:
            visualise_graphs(graph_set, graph)
        r_data["graphs"].append(graph_set)
    return r_data


#######################################################################################################
# Helper functions
#######################################################################################################


def _get_unit_operation_targets(zone: Zone):
    """Populate a ``Zone`` with detailed unit operation-level pinch targets.
    """
    if zone.config.DO_DIRECT_OPERATION_TARGETING:
        if len(zone.subzones) > 0:
            z: Zone
            for z in zone.subzones.values():
                if z.identifier == ZoneType.O.value:
                    if zone.config.DO_DIRECT_OPERATION_TARGETING:
                        compute_direct_integration_targets(z)
                else:
                    raise ValueError("Invalid zone nesting. Unit operation zones can only contain other operation zones.")

        compute_direct_integration_targets(zone)

    return zone


def _get_process_targets(zone: Zone):
    """Populate a ``Zone`` with detailed process-level pinch targets.
    """
    
    if len(zone.subzones) > 0:
        z: Zone
        for z in zone.subzones.values():
            if z.identifier == ZoneType.O.value:
                z = _get_unit_operation_targets(z)
            elif z.identifier == ZoneType.P.value:
                z = _get_process_targets(z)
            else:
                raise ValueError("Invalid zone nesting. Process zones can only contain other process zones and operation zones.")

        if zone.config.DO_INDIRECT_PROCESS_TARGETING:
            compute_indirect_integration_targets(zone)

    compute_direct_integration_targets(zone)

    return zone


def _get_site_targets(zone: Zone):
    """Targets heat integration using Total Site Anlysis,
    by systematically analysing individual zones and then performing
    site-level indirect integration through the utility system.
    """

    # if zone.config.DO_DIRECT_SITE_TARGETING:
    # Totally integrated analysis for a site zone
    compute_direct_integration_targets(zone)

    if len(zone.subzones) > 0:
        # Targets process level energy requirements
        z: Zone
        for z in zone.subzones.values():
            _get_process_targets(z)
            if z.identifier == ZoneType.O.value:
                _get_unit_operation_targets(z)
            elif z.identifier == ZoneType.P.value:
                _get_process_targets(z)
            elif z.identifier == ZoneType.S.value:
                _get_site_targets(z)                
            else:
                raise ValueError("Invalid zone nesting. Sites zones can only contain site, process and operation zones.")

        # Calculates TS targets based on different approaches
        compute_indirect_integration_targets(zone)

    return zone


def _get_community_targets(zone: Zone):
    """Targets a Community Zone."""
    z: Zone
    for z in zone.subzones.values():
        z = _get_site_targets(z)
    return zone


def _get_regional_targets(zone: Zone):
    """Targets a Regional Zone."""
    z: Zone
    for z in zone.subzones.values():
        z = _get_community_targets(z)
    return zone


def _get_report(zone: Zone) -> dict:
    """Creates the database summary of zone targets."""
    targets: List[dict] = []

    for t in zone.targets.values():
        t: EnergyTarget
        targets.append(t.serialize_json())

    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            z: Zone
            targets.extend(_get_report(z))

    return targets


def _get_utilities(zone: Zone) -> List[Stream]:
    """Gets a list of any default utilities generated during the analysis."""
    utilities: List[Stream] = zone.hot_utilities + zone.cold_utilities
    default_hu: Stream = next((u for u in utilities if u.name == "HU"), None)
    default_cu: Stream = next((u for u in utilities if u.name == "CU"), None)
    return [default_hu, default_cu]


_TARGET_HANDLERS = {
    ZoneType.R.value: _get_regional_targets,
    ZoneType.C.value: _get_community_targets,
    ZoneType.S.value: _get_site_targets,
    ZoneType.P.value: _get_process_targets,
    ZoneType.O.value: _get_unit_operation_targets,
}
