"""Aggregate analysis outputs into serialisable response payloads."""

from typing import List

from ..classes import *
from ..analysis.graphs import get_output_graphs

__all__ = ["extract_results"]


#######################################################################################################
# Public API
#######################################################################################################


def extract_results(master_zone: Zone) -> dict:
    """Serializes results data into a dictionaty from options."""
    return {
        "name": master_zone.name,
        "targets": _get_report(master_zone),
        "utilities": _get_utilities(master_zone),
        "graphs": get_output_graphs(master_zone),
    }


#######################################################################################################
# Helper Functions
#######################################################################################################


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


def _get_utilities(master_zone: Zone) -> List[Stream]:
    """Gets a list of any default utilities generated during the analysis."""
    utilities: List[Stream] = master_zone.hot_utilities + master_zone.cold_utilities
    default_hu: Stream = next((u for u in utilities if u.name == "HU"), None)
    default_cu: Stream = next((u for u in utilities if u.name == "CU"), None)
    return [default_hu, default_cu]
