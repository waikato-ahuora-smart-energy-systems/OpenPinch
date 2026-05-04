"""High-level orchestration for running an OpenPinch analysis.

The functions in this module wire together data validation, pinch targeting,
and output formatting.  They act as the main entry points used by the
``PinchProblem`` helper class as well as external callers embedding OpenPinch
in larger workflows.
"""

from typing import Any, List

from .services.common.graph_data import get_output_graph_data
from .services.services_entry import (
    data_preprocessing_service,
    direct_heat_integration_service,
    indirect_heat_integration_service,
)
from .classes.energy_target import EnergyTarget
from .classes.stream import Stream
from .classes.stream_collection import StreamCollection
from .classes.zone import Zone
from .lib.enums import Z, ZoneType
from .lib.schema import TargetInput, TargetOutput
from .utils.decorators import timing_decorator

__all__ = ["pinch_analysis_service", "get_targets", "extract_results"]


#######################################################################################################
# Public API
#######################################################################################################


@timing_decorator
def pinch_analysis_service(
    data: Any,
    project_name: str = "Project",
    is_return_full_results: bool = False,
) -> TargetOutput | tuple[TargetOutput, Zone]:
    """Validate user data, run the targeting workflow, and return structured results.

    Parameters
    ----------
    data:
        Raw request payload matching :class:`OpenPinch.lib.schema.TargetInput`.
        Dictionaries, Pydantic models, and dataclass-like objects are accepted.
    project_name:
        Optional label used in generated graphs and result files.
    is_return_full_results:
        When ``True``, return both the validated
        :class:`~OpenPinch.lib.schema.TargetOutput` and the solved
        :class:`~OpenPinch.classes.zone.Zone` hierarchy.

    Returns
    -------
    TargetOutput or tuple[TargetOutput, Zone]
        Validated response payload, optionally paired with the in-memory zone
        tree for advanced inspection and post-processing.
    """

    # Formulate the top level zone with all subzones and approperiate input data
    master_zone = data_preprocessing_service(
        project_name=project_name,
        input_data=data,
    )

    # Perform advanced targeting analysis on the master zone and all subzones
    master_zone = get_targets(master_zone)

    # Extract the core results from the master zone
    return_data = extract_results(master_zone)

    # Validate response data
    validated_data = TargetOutput.model_validate(return_data)

    # Return data
    return (
        validated_data if not is_return_full_results else (validated_data, master_zone)
    )


def get_targets(master_zone: Zone) -> Zone:
    """Dispatch a prepared zone tree to the appropriate targeting routine.

    This function is a lower-level hook compared to
    :func:`pinch_analysis_service`. It expects a fully prepared
    :class:`~OpenPinch.classes.zone.Zone` hierarchy and returns the same zone
    tree after the relevant direct and indirect integration targets have been
    populated.
    """

    handler = _TARGET_HANDLERS.get(master_zone.type)
    if handler is None:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")

    return handler(master_zone)


def extract_results(master_zone: Zone) -> dict:
    """Serialise solved targets, generated utilities, and graph payloads."""
    return {
        "name": master_zone.name,
        "targets": _get_report(master_zone),
        "utilities": _get_utilities(master_zone),
        "graphs": get_output_graph_data(master_zone),
    }


#######################################################################################################
# Helper functions
#######################################################################################################


def _get_unit_operation_targets(zone: Zone):
    """Populate a ``Zone`` with detailed unit operation-level pinch targets."""
    if zone.config.DO_DIRECT_OPERATION_TARGETING:
        if len(zone.subzones) > 0:
            z: Zone
            for z in zone.subzones.values():
                if z.type == ZoneType.O.value:
                    if zone.config.DO_DIRECT_OPERATION_TARGETING:
                        direct_heat_integration_service(z)
                else:
                    raise ValueError(
                        "Invalid zone nesting. Unit operation zones can only contain other operation zones."
                    )

        direct_heat_integration_service(zone)

    return zone


def _get_process_targets(zone: Zone):
    """Populate a ``Zone`` with detailed process-level pinch targets."""

    if len(zone.subzones) > 0:
        z: Zone
        for z in zone.subzones.values():
            if z.type == ZoneType.O.value:
                z = _get_unit_operation_targets(z)
            elif z.type == ZoneType.P.value:
                z = _get_process_targets(z)
            else:
                raise ValueError(
                    "Invalid zone nesting. Process zones can only contain other process zones and operation zones."
                )

        if zone.config.DO_INDIRECT_PROCESS_TARGETING:
            indirect_heat_integration_service(zone)

    direct_heat_integration_service(zone)

    return zone


def _get_site_targets(zone: Zone):
    """Targets heat integration using Total Site Anlysis,
    by systematically analysing individual zones and then performing
    site-level indirect integration through the utility system.
    """

    # Totally integrated analysis for a site zone
    direct_heat_integration_service(zone)

    # Targets sub-zone energy requirements
    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            if z.type == Z.O.value:
                _get_unit_operation_targets(z)
            elif z.type == Z.P.value:
                _get_process_targets(z)
            elif z.type == Z.S.value:
                _get_site_targets(z)
            else:
                raise ValueError(
                    "Invalid zone nesting. Sites zones can only contain site, process and operation zones."
                )

        # Calculates TS targets based on different approaches
        indirect_heat_integration_service(zone)

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


def _get_utilities(zone: Zone) -> StreamCollection:
    """Gets a list of any default utilities generated during the analysis."""
    utilities: StreamCollection = zone.hot_utilities + zone.cold_utilities
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
