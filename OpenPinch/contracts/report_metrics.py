"""Explicit per-field reporting units and period aggregation policies."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from pydantic import Field
from pydantic_core import PydanticUndefined

from .units import OUTPUT_UNIT_RULES


class AggregationPolicy(StrEnum):
    WEIGHTED_MEAN = "weighted_mean"
    MAXIMUM = "maximum"
    CONSENSUS = "consensus"
    DERIVED = "derived"
    EXCLUDE = "exclude"


@dataclass(frozen=True)
class MetricSpecification:
    name: str
    aggregation: AggregationPolicy
    unit: str | None
    representation: str


def report_field(
    name,
    policy,
    default=PydanticUndefined,
    *,
    representation="quantity",
    default_factory=None,
):
    rule = OUTPUT_UNIT_RULES.get(name)
    metadata = {
        "aggregation": policy,
        "unit": None if rule is None else rule.default_unit,
        "representation": representation,
    }
    if default_factory is not None:
        return Field(default_factory=default_factory, json_schema_extra=metadata)
    return Field(default=default, json_schema_extra=metadata)


def metric_specifications(model):
    """Reject unclassified fields instead of silently copying a first-period value."""
    specs = {}
    for name, field in model.model_fields.items():
        metadata = field.json_schema_extra or {}
        if not isinstance(metadata, dict) or "aggregation" not in metadata:
            raise ValueError(f"Report field {name!r} has no aggregation policy.")
        specs[name] = MetricSpecification(
            name,
            AggregationPolicy(metadata["aggregation"]),
            metadata.get("unit"),
            metadata["representation"],
        )
    return MappingProxyType(specs)
