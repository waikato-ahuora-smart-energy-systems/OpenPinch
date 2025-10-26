from ..classes import *
from ..lib import *
from ..utils import *
from ..analysis.direct_integration_entry import *
from ..analysis.indirect_integration_entry import *
from .unit_operation_analysis import get_unit_operation_targets

__all__ = ["get_process_targets"]


#######################################################################################################
# Public API
#######################################################################################################


def get_process_targets(zone: Zone):
    """Populate a ``Zone`` with detailed process-level pinch targets.
    """
    
    if len(zone.subzones) > 0:
        z: Zone
        for z in zone.subzones.values():
            if z.identifier == ZoneType.O.value:
                z = get_unit_operation_targets(z)
            elif z.identifier == ZoneType.P.value:
                z = get_process_targets(z)
            else:
                raise ValueError("Invalid zone nesting. Process zones can only contain other process zones and operation zones.")

        if zone.config.DO_INDIRECT_PROCESS_TARGETING:
            compute_indirect_integration_targets(zone)

    compute_direct_integration_targets(zone)

    return zone