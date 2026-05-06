"""
"""

from typing import Callable, List

from ..lib.enums import ZT
from ..classes.zone import Zone
from ..classes.energy_target import EnergyTarget
from ..classes.energy_target import EnergyTarget
from ..classes.stream import Stream
from ..classes.stream_collection import StreamCollection
from ..services.common.graph_data import get_output_graph_data
from ..services.services_entry import (
    direct_heat_integration_service,
    indirect_heat_integration_service,
)


def get_targets(
    zone: Zone, 
    direct_service_func: Callable = direct_heat_integration_service, 
    indirect_service_func: Callable = indirect_heat_integration_service, 
):
    """Dispatch a prepared zone tree to the appropriate targeting routine.

    This function is a lower-level hook compared to
    :func:`pinch_analysis_service`. It expects a fully prepared
    :class:`~OpenPinch.classes.zone.Zone` hierarchy and returns the same zone
    tree after the relevant direct and indirect integration targets have been
    populated.
    """

    handler = _TARGET_HANDLERS.get(zone.type)
    if handler is None:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")

    return handler(zone, direct_service_func, indirect_service_func)


def extract_results(zone: Zone) -> dict:
    """Serialise solved targets, generated utilities, and graph payloads."""
    return {
        "name": zone.name,
        "targets": _get_report(zone),
        "utilities": _get_utilities(zone),
        "graphs": get_output_graph_data(zone),
    }


#######################################################################################################
# Helper functions
#######################################################################################################


def _get_unit_operation_targets(
    zone: Zone, 
    direct_service_func: Callable = direct_heat_integration_service, 
    indirect_service_func: Callable = indirect_heat_integration_service, 
):
    """Populate a ``Zone`` with detailed unit operation-level pinch targets."""
    if zone.config.DO_DIRECT_OPERATION_TARGETING:
        if len(zone.subzones) > 0:
            z: Zone
            for z in zone.subzones.values():
                if z.type == ZT.O.value:
                    if zone.config.DO_DIRECT_OPERATION_TARGETING:
                        direct_heat_integration_service(z)
                else:
                    raise ValueError(
                        "Invalid zone nesting. Unit operation zones can only contain other operation zones."
                    )

        direct_heat_integration_service(zone)

    return zone


def _get_process_targets(
    zone: Zone, 
    direct_service_func: Callable = direct_heat_integration_service, 
    indirect_service_func: Callable = indirect_heat_integration_service, 
):
    """Populate a ``Zone`` with detailed process-level pinch targets."""

    if len(zone.subzones) > 0:
        z: Zone
        for z in zone.subzones.values():
            if z.type == ZT.O.value:
                z = _get_unit_operation_targets(z)
            elif z.type == ZT.P.value:
                z = _get_process_targets(z)
            else:
                raise ValueError(
                    "Invalid zone nesting. Process zones can only contain other process zones and operation zones."
                )

        if zone.config.DO_INDIRECT_PROCESS_TARGETING:
            indirect_heat_integration_service(zone)

    direct_heat_integration_service(zone)

    return zone


def _get_site_targets(
    zone: Zone, 
    direct_service_func: Callable = direct_heat_integration_service, 
    indirect_service_func: Callable = indirect_heat_integration_service, 
):
    """Targets heat integration using Total Site Anlysis,
    by systematically analysing individual zones and then performing
    site-level indirect integration through the utility system.
    """

    # Totally integrated analysis for a site zone
    direct_heat_integration_service(zone)

    # Targets sub-zone energy requirements
    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            if z.type == ZT.O.value:
                _get_unit_operation_targets(z)
            elif z.type == ZT.P.value:
                _get_process_targets(z)
            elif z.type == ZT.S.value:
                _get_site_targets(z)
            else:
                raise ValueError(
                    "Invalid zone nesting. Sites zones can only contain site, process and operation zones."
                )

        # Calculates TS targets based on different approaches
        indirect_heat_integration_service(zone)

    return zone


def _get_community_targets(
    zone: Zone, 
    direct_service_func: Callable = direct_heat_integration_service, 
    indirect_service_func: Callable = indirect_heat_integration_service, 
):
    """Targets a Community Zone."""
    z: Zone
    for z in zone.subzones.values():
        z = _get_site_targets(z)
    return zone


def _get_regional_targets(
    zone: Zone, 
    direct_service_func: Callable = direct_heat_integration_service, 
    indirect_service_func: Callable = indirect_heat_integration_service, 
):
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


def _get_utilities(zone: Zone) -> StreamCollection:
    """Gets a list of any default utilities generated during the analysis."""
    utilities: StreamCollection = zone.hot_utilities + zone.cold_utilities
    default_hu: Stream = next((u for u in utilities if u.name == "HU"), None)
    default_cu: Stream = next((u for u in utilities if u.name == "CU"), None)
    return [default_hu, default_cu]


_TARGET_HANDLERS = {
    ZT.R.value: _get_regional_targets,
    ZT.C.value: _get_community_targets,
    ZT.S.value: _get_site_targets,
    ZT.P.value: _get_process_targets,
    ZT.O.value: _get_unit_operation_targets,
}