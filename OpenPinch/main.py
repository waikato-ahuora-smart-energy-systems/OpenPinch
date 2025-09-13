from .classes import Zone
from .lib import *
from .utils import *
from .analysis import (
    prepare_problem_struture, 
    get_site_targets, 
    get_process_pinch_targets, 
    get_regional_targets, 
    visualise_graphs, 
    output_response
)

__all__ = ["pinch_analysis_service", "get_targets", "visualise"]


def pinch_analysis_service(data: Any, parent_fs_name: str ='Project') -> TargetResponse:
    """Calculates targets and outputs from inputs and options"""
    # Validate request data using Pydantic model
    request_data = TargetRequest.model_validate(data)

    # Perform advanced pinch analysis and total site analysis
    return_data = get_targets(
        zone_tree=request_data.zone_tree,
        streams=request_data.streams, 
        utilities=request_data.utilities, 
        options=request_data.options,
        name=parent_fs_name,
    )

    # Validate response data
    validated_data = TargetResponse.model_validate(return_data)

    # Return data
    return validated_data


@timing_decorator
def get_targets(streams: List[StreamSchema], utilities: List[UtilitySchema], options: List[Options], name: str ='Project', zone_tree: ZoneTreeSchema = None) -> dict:
    """Conduct advanced pinch analysis and total site analysis on the given streams and utilities."""
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
def visualise(data) -> dict:
    """Function for building graphs from problem tables as opposed to class instances."""
    r_data = {'graphs': []}
    z: Zone
    for z in data:
        graph_set = {'name': f"{z.name}", 'graphs': []}
        for graph in z.graphs:
            visualise_graphs(graph_set, graph)
        r_data['graphs'].append(graph_set)
    return r_data


########### TODO ends ########### 
