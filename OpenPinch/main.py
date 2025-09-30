"""High-level orchestration for running an OpenPinch analysis.

The functions in this module wire together data validation, pinch targeting,
and output formatting.  They act as the main entry points used by the
``PinchProblem`` helper class as well as external callers embedding OpenPinch
in larger workflows.
"""

from typing import Any, List

from .classes import Zone
from .lib import *
from .utils import *
from .analysis import (
    prepare_problem_struture,
    get_site_targets,
    get_process_pinch_targets,
    get_regional_targets,
    visualise_graphs,
    output_response,
)

__all__ = ["pinch_analysis_service", "get_targets", "get_visualise"]


def pinch_analysis_service(data: Any, project_name: str = "Project") -> TargetOutput:
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

    # Perform advanced pinch analysis and total site analysis
    return_data = get_targets(
        zone_tree=request_data.zone_tree,
        streams=request_data.streams, 
        utilities=request_data.utilities, 
        options=request_data.options,
        name=project_name,
    )

    # Validate response data
    validated_data = TargetOutput.model_validate(return_data)

    # Return data
    return validated_data


@timing_decorator
def get_targets(
    streams: List[StreamSchema],
    utilities: List[UtilitySchema],
    options: List[Options],
    name: str = "Project",
    zone_tree: ZoneTreeSchema = None,
) -> dict:
    """Conduct core pinch analysis and total site targeting.

    This function is a lower-level hook compared to :func:`pinch_analysis_service`.
    It expects already validated option blocks and stream/utility schemas, and it
    returns a dictionary ready for conversion into :class:`TargetOutput`.
    """
    master_zone = prepare_problem_struture(streams, utilities, options, name, zone_tree)
    if master_zone.identifier in [ZoneType.R.value, ZoneType.C.value]:
        master_zone = get_regional_targets(master_zone)
    elif master_zone.identifier == ZoneType.S.value:
        master_zone = get_site_targets(master_zone)
    elif master_zone.identifier == ZoneType.P.value:
        master_zone = get_process_pinch_targets(master_zone)
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
