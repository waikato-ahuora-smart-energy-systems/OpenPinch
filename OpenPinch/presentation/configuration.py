"""Presentation metadata for editable OpenPinch configuration."""

from __future__ import annotations

import types
from enum import Enum
from typing import Any, Union, get_args, get_origin

from ..contracts.workspace import ConfigurationFieldMetadata
from ..domain.configuration_fields import (
    USER_CONFIG_FIELD_SPECS,
    configuration_field_support_level,
)
from .configuration_text import CONFIGURATION_FIELD_TEXT


def configuration_options() -> list[ConfigurationFieldMetadata]:
    """Return report-friendly metadata for supported configuration fields."""
    return configuration_field_metadata()


def configuration_field_metadata() -> list[ConfigurationFieldMetadata]:
    """Return declarative metadata for editable configuration fields."""
    fields = []
    for name, spec in USER_CONFIG_FIELD_SPECS.items():
        enum_cls = spec.enum_cls
        field_type, multiple = annotation_metadata(spec.annotation, enum_cls=enum_cls)
        text = CONFIGURATION_FIELD_TEXT[name]
        fields.append(
            ConfigurationFieldMetadata(
                name=name,
                label=text.label,
                unit=text.unit,
                description=text.description,
                default=spec.default,
                field_type=field_type,
                group=spec.group,
                config_path=list(spec.config_path),
                support_level=configuration_field_support_level(name),
                runtime_status=spec.runtime_status,
                enum_choices=[str(item.value) for item in enum_cls] if enum_cls else [],
                numeric_min=spec.numeric_min,
                numeric_max=spec.numeric_max,
                # positive=True fields reject their minimum (0) itself.
                numeric_min_exclusive=spec.positive,
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
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return annotation_metadata(members[0], enum_cls=None)
    if annotation is list or origin is list:
        items = get_args(annotation)
        if items and items[0] in {int, float}:
            return "number_list", True
        return "string_list", True
    if annotation is dict or origin is dict:
        return "object", False
    if annotation in {bool}:
        return "boolean", False
    if annotation in {int, float}:
        return "number", False
    return "string", False


__all__ = [
    "annotation_metadata",
    "configuration_field_metadata",
    "configuration_options",
]
