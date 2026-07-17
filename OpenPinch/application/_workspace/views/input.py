"""Workspace input and configuration view shaping."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from ....contracts.workspace import (
    ConfigurationFieldMetadata,
    InputRecordView,
    ZoneNodeView,
)
from ....domain.configuration_fields import (
    CONFIG_FIELD_SPECS,
    configuration_field_support_level,
)
from .serialization import json_safe, maybe_float, maybe_string


def configuration_field_metadata() -> list[ConfigurationFieldMetadata]:
    """Return declarative metadata for editable configuration fields."""
    fields = []
    for name, spec in CONFIG_FIELD_SPECS.items():
        enum_cls = spec.enum_cls
        field_type, multiple = annotation_metadata(spec.annotation, enum_cls=enum_cls)
        fields.append(
            ConfigurationFieldMetadata(
                name=name,
                label=name.replace("_", " ").title(),
                field_type=field_type,
                group=spec.group,
                config_path=list(spec.config_path),
                support_level=configuration_field_support_level(name),
                runtime_status=spec.runtime_status,
                enum_choices=[str(item.value) for item in enum_cls] if enum_cls else [],
                numeric_min=spec.numeric_min,
                numeric_max=spec.numeric_max,
                multiple=multiple,
            )
        )
    return fields


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


def annotation_metadata(
    annotation: Any,
    *,
    enum_cls: Optional[type[Enum]],
) -> tuple[str, bool]:
    """Map Python annotations to simple frontend input field kinds."""
    if enum_cls is not None:
        return "enum", False
    if annotation in {bool}:
        return "boolean", False
    if annotation in {int, float}:
        return "number", False
    if annotation in {str}:
        return "string", False
    if annotation in {list[str], list}:
        return "string_list", True

    text = str(annotation)
    if "List" in text or "list[" in text:
        return "string_list", True
    if "bool" in text:
        return "boolean", False
    if "float" in text or "int" in text:
        return "number", False
    if "dict" in text:
        return "object", False
    return "string", False
