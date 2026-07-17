"""Helpers for converting solved zone trees into serializable result data."""

from __future__ import annotations

from typing import List

from ....analysis.graphs.service import get_output_graph_data
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection
from ....domain.zone import Zone
from ....presentation.reporting.results import serialize_target

__all__ = ["extract_results"]


def extract_results(zone: Zone, period_id: str | None = None) -> dict:
    """Serialise solved targets, generated utilities, and graph data."""
    return {
        "name": zone.name,
        "period_id": period_id,
        "targets": _get_report(zone, period_id=period_id),
        "utilities": _get_utilities(zone),
        "graphs": get_output_graph_data(zone),
    }


def _get_report(zone: Zone, period_id: str | None = None) -> dict:
    """Create the report data from one zone and all nested subzones."""
    targets: List[dict] = []

    for target in zone.targets.values():
        target_data = serialize_target(target)
        if period_id is not None:
            target_data["period_id"] = period_id
        targets.append(target_data)

    if len(zone.subzones) > 0:
        for subzone in zone.subzones.values():
            targets.extend(_get_report(subzone, period_id=period_id))

    return targets


def _get_utilities(zone: Zone) -> StreamCollection:
    """Get any default utilities generated during the analysis."""
    utilities: StreamCollection = zone.hot_utilities + zone.cold_utilities
    default_hu: Stream = next((u for u in utilities if u.name == "HU"), None)
    default_cu: Stream = next((u for u in utilities if u.name == "CU"), None)
    return [default_hu, default_cu]
