from ..classes import *
from .site_analysis import get_site_targets

__all__ = ["get_community_targets"]


#######################################################################################################
# Public API --- TODO
#######################################################################################################

def get_community_targets(community: Zone):
    """Targets a Community Zone."""
    s: Zone
    for s in community.subzones.values():
        s = get_site_targets(s)
    return community


#######################################################################################################
# Helper Functions
#######################################################################################################

