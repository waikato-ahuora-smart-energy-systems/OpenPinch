"""Analysis submodules implementing OpenPinch recursive analysis of different zones.

This package exposes the most commonly used entry points for analyse different 
scales of zones: unit, process, site, community, and region.
"""

from .unit_operation_analysis import get_unit_operation_targets
from .process_analysis import get_process_targets
from .site_analysis import get_site_targets
from .community_analysis import get_community_targets
from .region_analysis import get_regional_targets
from .entry import get_targets, get_visualise