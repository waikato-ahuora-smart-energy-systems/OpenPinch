"""TODO: Write docstring
"""

from typing import List

from ..classes import Zone
from ..lib import *
from ..utils import *
from ..analysis import (
    prepare_problem_struture,
    visualise_graphs,
    output_response,
)
from . import (
    get_unit_operation_targets,
    get_process_targets,
    get_site_targets,
    get_community_targets,
    get_regional_targets,
)

__all__ = ["pinch_analysis_service", "get_targets", "get_visualise"]


@timing_decorator
def get_targets(
    streams: List[StreamSchema],
    utilities: List[UtilitySchema],
    options: List[Options],
    name: str = "Project",
    zone_tree: ZoneTreeSchema = None,
) -> dict:
    """Conduct core Pinch Analysis and total site targeting.

    This function is a lower-level hook compared to :func:`pinch_analysis_service`.
    It expects already validated option blocks and stream/utility schemas, and it
    returns a dictionary ready for conversion into :class:`TargetOutput`.
    """
    master_zone = prepare_problem_struture(streams, utilities, options, name, zone_tree)
    if master_zone.identifier == ZoneType.R.value:
        master_zone = get_regional_targets(master_zone)
    elif master_zone.identifier == ZoneType.C.value:
        master_zone = get_community_targets(master_zone)
    elif master_zone.identifier == ZoneType.S.value:
        master_zone = get_site_targets(master_zone)
    elif master_zone.identifier == ZoneType.P.value:
        master_zone = get_process_targets(master_zone)
    elif master_zone.identifier == ZoneType.O.value:
        master_zone = get_unit_operation_targets(master_zone) 
    else:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")
    
    return output_response(master_zone)    


########### TODO: This function is untested and not updated since the overhaul of OpenPinch. Broken, most likely.#####
@timing_decorator
def get_visualise(data) -> dict:
    """Build graph payloads directly from legacy problem-table structures.

    Notes
    -----
    This helper predates the class-based refactor and retains compatibility with
    older tooling.  It is currently untested and should be considered provisional.
    """
    r_data = {'graphs': []}
    z: Zone
    for z in data:
        graph_set = {'name': f"{z.name}", 'graphs': []}
        for graph in z.graphs:
            visualise_graphs(graph_set, graph)
        r_data['graphs'].append(graph_set)
    return r_data
