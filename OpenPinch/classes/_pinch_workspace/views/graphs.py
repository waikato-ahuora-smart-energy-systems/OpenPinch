"""Workspace graph catalog and payload view shaping."""

from __future__ import annotations

from typing import Any

from ....lib.schemas.workspace import GraphCatalogEntry, GraphDataEntry
from .common import json_safe, maybe_string


def graph_catalog_entries(graph_data: dict[str, Any]) -> list[GraphCatalogEntry]:
    """Flatten graph metadata into a catalog view."""
    entries = []
    for graph_set_id, graph_set in graph_data.items():
        target_name = str(graph_set.get("name", graph_set_id))
        target_id = target_name
        zone_name = maybe_string(graph_set.get("zone_name"))
        zone_address = maybe_string(graph_set.get("zone_address"))
        target_type = maybe_string(graph_set.get("target_type"))
        for index, graph in enumerate(graph_set.get("graphs", [])):
            graph_type = maybe_string(graph.get("type"))
            graph_name = str(graph.get("name") or graph_type or f"Graph {index + 1}")
            entries.append(
                GraphCatalogEntry(
                    graph_id=f"{graph_set_id}::{index}::{graph_type or 'graph'}",
                    graph_set_id=str(graph_set_id),
                    target_id=target_id,
                    target_name=target_name,
                    zone_name=zone_name,
                    zone_address=zone_address,
                    target_type=target_type,
                    graph_type=graph_type,
                    graph_name=graph_name,
                    index=index,
                )
            )
    return entries


def graph_data_entries(graph_data: dict[str, Any]) -> list[GraphDataEntry]:
    """Flatten raw graph data into stable graph entry records."""
    entries = []
    for graph_set_id, graph_set in graph_data.items():
        target_name = str(graph_set.get("name", graph_set_id))
        target_id = target_name
        for index, graph in enumerate(graph_set.get("graphs", [])):
            graph_type = maybe_string(graph.get("type"))
            graph_name = str(graph.get("name") or graph_type or f"Graph {index + 1}")
            entries.append(
                GraphDataEntry(
                    graph_id=f"{graph_set_id}::{index}::{graph_type or 'graph'}",
                    graph_set_id=str(graph_set_id),
                    target_id=target_id,
                    target_name=target_name,
                    graph_type=graph_type,
                    graph_name=graph_name,
                    graph_data=json_safe(graph),
                )
            )
    return entries
