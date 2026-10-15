"""JSON Schema for the public contracts, for typed clients such as TypeScript."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from ..domain.stream_collection import StreamCollection
from ..domain.value import Value
from .heat_recovery_dt_min import HeatRecoveryDtMinResult
from .hpr_performance_map import HprPerformanceMap, HprPerformanceMapRequest
from .input import TargetInput
from .output import TargetOutput
from .synthesis.result import HeatExchangerNetworkSynthesisResult
from .utility_placement import UtilityPlacementResult
from .workspace import (
    ConfigurationFieldMetadata,
    PinchWorkspaceBundle,
    ValidationReport,
)

# What Value.to_dict() emits: a scalar, or one value per operating period.
VALUE_JSON_SCHEMA: dict[str, Any] = {
    "title": "Value",
    "description": "A value with its unit; period-valued when it has 'values'.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "value": {"type": ["number", "null"]},
                "unit": {"type": "string"},
            },
            "required": ["value", "unit"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}},
                "unit": {"type": "string"},
                "weights": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["values", "unit"],
            "additionalProperties": False,
        },
    ],
}


def _column(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


_NUMBER_COLUMN = _column({"type": ["number", "null"]})

# What StreamCollection.to_dict() emits: one array per column, one entry per
# stream, in reporting order. NaN magnitudes serialise as null.
STREAM_COLLECTION_JSON_SCHEMA: dict[str, Any] = {
    "title": "StreamCollection",
    "description": "Streams by column, as StreamCollection.to_dict().",
    "type": "object",
    "properties": {
        "name": _column({"type": "string"}),
        "category": _column(
            {
                "enum": [
                    "hot_stream",
                    "cold_stream",
                    "hot_utility",
                    "cold_utility",
                    "other",
                ]
            }
        ),
        "type": _column({"type": "string"}),
        "is_process_stream": _column({"type": "boolean"}),
        "t_supply": _NUMBER_COLUMN,
        "t_target": _NUMBER_COLUMN,
        "heat_flow": _NUMBER_COLUMN,
        "dt_cont": _NUMBER_COLUMN,
        "dt_cont_multiplier": _NUMBER_COLUMN,
        "htc": _NUMBER_COLUMN,
        "active": _column({"type": "boolean"}),
    },
}


class _ContractSchemaGenerator(GenerateJsonSchema):
    """Describes the domain types that contracts carry by their JSON form."""

    def is_instance_schema(self, schema):
        cls = schema["cls"]
        if issubclass(cls, Value):
            return dict(VALUE_JSON_SCHEMA)
        if issubclass(cls, StreamCollection):
            return dict(STREAM_COLLECTION_JSON_SCHEMA)
        return super().is_instance_schema(schema)


# Public contracts by name. Inputs use the validation schema (what callers may
# send); results use the serialization schema (``model_dump(mode="json")``).
CONTRACTS: dict[str, tuple[type[BaseModel], str]] = {
    "TargetInput": (TargetInput, "validation"),
    "TargetOutput": (TargetOutput, "serialization"),
    "PinchWorkspaceBundle": (PinchWorkspaceBundle, "validation"),
    "ValidationReport": (ValidationReport, "serialization"),
    "ConfigurationFieldMetadata": (ConfigurationFieldMetadata, "serialization"),
    "HprPerformanceMapRequest": (HprPerformanceMapRequest, "validation"),
    "HprPerformanceMap": (HprPerformanceMap, "serialization"),
    "UtilityPlacementResult": (UtilityPlacementResult, "serialization"),
    "HeatRecoveryDtMinResult": (HeatRecoveryDtMinResult, "serialization"),
    "HeatExchangerNetworkSynthesisResult": (
        HeatExchangerNetworkSynthesisResult,
        "serialization",
    ),
}


def contract_json_schemas() -> dict[str, dict[str, Any]]:
    """Return the JSON Schema of each public contract, by contract name."""
    return {
        name: model.model_json_schema(
            mode=mode, schema_generator=_ContractSchemaGenerator
        )
        for name, (model, mode) in CONTRACTS.items()
    }


__all__ = ["CONTRACTS", "VALUE_JSON_SCHEMA", "contract_json_schemas"]
