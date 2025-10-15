from __future__ import annotations
from typing import TYPE_CHECKING

from .site_analysis import get_site_targets

if TYPE_CHECKING:
    from ..classes import Zone

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
