from ..classes import *
from .site_analysis import get_site_targets

__all__ = ["get_regional_targets"]


#######################################################################################################
# Public API --- TODO
#######################################################################################################

def get_regional_targets(region: Zone):
    """Targets a Regional Zone."""
    # Targets site level energy & exergy requirements
    s: Zone
    for s in region.subzones.values():
        s = get_site_targets(s)
    return region


#######################################################################################################
# Helper Functions
#######################################################################################################

