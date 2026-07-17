"""Workspace input and configuration view shaping."""

from __future__ import annotations

from typing import Any, Optional

from ....contracts.workspace import (
    InputRecordView,
    ZoneNodeView,
)
from .serialization import json_safe, maybe_float, maybe_string


def zone_tree_view(zone_tree: Any) -> list[ZoneNodeView]:
    """Flatten a nested zone tree into frontend-friendly node records."""
    if not isinstance(zone_tree, dict):
        return []

    nodes: list[ZoneNodeView] = []

    def walk(node: dict[str, Any], parent_path: Optional[str]) -> None:
        name = str(node.get("name", "Zone"))
        path = name if parent_path is None else f"{parent_path}/{name}"
        nodes.append(
            ZoneNodeView(
                zone_id=path,
                path=path,
                name=name,
                zone_type=maybe_string(node.get("type")),
                parent_id=parent_path,
                dt_cont_multiplier=maybe_float(node.get("dt_cont_multiplier")),
            )
        )
        children = node.get("children") or []
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child, path)

    walk(zone_tree, None)
    return nodes


def record_views(records: Any, *, section: str) -> list[InputRecordView]:
    """Convert stream/utility input rows into editable frontend records."""
    if not isinstance(records, list):
        return []
    views = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        path = f"{section}[{index}]"
        views.append(
            InputRecordView(
                record_id=path,
                path=path,
                section=section,
                index=index,
                name=maybe_string(record.get("name")),
                zone=maybe_string(record.get("zone")),
                data=json_safe(record),
            )
        )
    return views
