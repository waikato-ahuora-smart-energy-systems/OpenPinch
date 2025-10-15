"""High-level orchestration utility for executing multi-scale pinch analysis runs.

Provides the public entry points that select the appropriate targeting workflow for
each zone type. Functions here coordinate the heavier lifting performed within the
submodules under :mod:`OpenPinch.scales` and :mod:`OpenPinch.analysis`.
"""

from ..analysis import visualise_graphs
from ..classes import Zone
from ..lib import *
from ..utils import *
from . import (
    get_community_targets,
    get_process_targets,
    get_regional_targets,
    get_site_targets,
    get_unit_operation_targets,
)

__all__ = ["get_targets", "get_visualise"]

_TARGET_HANDLERS = {
    ZoneType.R.value: get_regional_targets,
    ZoneType.C.value: get_community_targets,
    ZoneType.S.value: get_site_targets,
    ZoneType.P.value: get_process_targets,
    ZoneType.O.value: get_unit_operation_targets,
}


@timing_decorator
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


########### TODO: This function is untested and not updated since the overhaul of OpenPinch. Broken, most likely.#####
@timing_decorator
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
