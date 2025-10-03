from ..classes import *
from .community_analysis import get_community_targets

__all__ = ["get_regional_targets"]


#######################################################################################################
# Public API --- TODO
#######################################################################################################

def get_regional_targets(region: Zone):
    """Targets a Regional Zone."""
    s: Zone
    for s in region.subzones.values():
        s = get_community_targets(s)
    return region


#######################################################################################################
# Helper Functions
#######################################################################################################

