from ..classes import *
from ..lib import *
from ..utils import *
from ..analysis.direct_integration_entry import *
from ..analysis.indirect_integration_entry import *
from .unit_operation_analysis import get_unit_operation_targets
from .process_analysis import get_process_targets

__all__ = ["get_site_targets"]


#######################################################################################################
# Public API
#######################################################################################################


def get_site_targets(zone: Zone):
    """Targets heat integration using Total Site Anlysis,
    by systematically analysing individual zones and then performing
    site-level indirect integration through the utility system.
    """

    if zone.config.DO_DIRECT_SITE_TARGETING:
        # Totally integrated analysis for a site zone
        compute_direct_integration_targets(zone)

    if len(zone.subzones) > 0:
        # Targets process level energy requirements
        z: Zone
        for z in zone.subzones.values():
            get_process_targets(z)
            if z.identifier == ZoneType.O.value:
                z = get_unit_operation_targets(z)
            elif z.identifier == ZoneType.P.value:
                z = get_process_targets(z)
            elif z.identifier == ZoneType.S.value:
                z = get_site_targets(z)                
            else:
                raise ValueError("Invalid zone nesting. Sites zones can only contain site, process and operation zones.")

        # Calculates TS targets based on different approaches
        compute_indirect_integration_targets(zone)

    return zone

