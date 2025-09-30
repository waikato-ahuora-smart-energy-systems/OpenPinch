from typing import List
from .graphs import get_output_graphs
from .support_methods import *
from ..classes import *

__all__ = ["output_response"]


#######################################################################################################
# Public API
#######################################################################################################

def output_response(site: Zone) -> dict:
    """Serializes results data into a dictionaty from options."""
    results = {}
    results['name'] = site.name
    results['targets'] = _report(site)
    results['utilities'] = _get_default_utilities(site)
    results['graphs'] = get_output_graphs(site)
    return results


#######################################################################################################
# Helper Functions
#######################################################################################################

def _report(zone: Zone) -> dict:
    """Creates the database summary of zone targets."""
    targets: List[dict] = []

    for t in zone.targets.values():
        t: EnergyTarget
        targets.append(t.serialize_json())        
    
    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            z: Zone
            targets.extend(
                _report(z)
            )
    
    return targets


def _get_default_utilities(site: Zone) -> List[Stream]:
    """Gets a list of any default utilities generated during the analysis."""
    utilities: List[Stream] = site.hot_utilities + site.cold_utilities
    default_hu: Stream = next((u for u in utilities if u.name == 'HU'), None)
    default_cu: Stream = next((u for u in utilities if u.name == 'CU'), None)
    return [default_hu, default_cu]