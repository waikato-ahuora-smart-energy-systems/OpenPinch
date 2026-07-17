"""Presentation metadata for editable OpenPinch configuration."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..contracts.workspace import ConfigurationFieldMetadata
from ..domain.configuration_fields import (
    USER_CONFIG_FIELD_SPECS,
    configuration_field_support_level,
)


def configuration_options() -> list[ConfigurationFieldMetadata]:
    """Return report-friendly metadata for supported configuration fields."""
    return configuration_field_metadata()


def configuration_field_metadata() -> list[ConfigurationFieldMetadata]:
    """Return declarative metadata for editable configuration fields."""
    fields = []
    for name, spec in USER_CONFIG_FIELD_SPECS.items():
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


def annotation_metadata(
    annotation: Any,
    *,
    enum_cls: type[Enum] | None,
) -> tuple[str, bool]:
    """Map Python annotations to frontend input field kinds."""
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


__all__ = [
    "annotation_metadata",
    "configuration_field_metadata",
    "configuration_options",
]
