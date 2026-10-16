"""JSON Schema export of the public contracts, for typed clients (TypeScript)."""

from __future__ import annotations

import json

import jsonschema
import pytest

from OpenPinch import PinchProblem
from OpenPinch.contracts.schema import (
    STREAM_COLLECTION_JSON_SCHEMA,
    VALUE_JSON_SCHEMA,
    contract_json_schemas,
)


def test_problem_input_schema_lists_streams_and_utilities():
    schema = contract_json_schemas()["TargetInput"]

    assert {"streams", "utilities", "options"} <= set(schema["properties"])


def test_problem_output_schema_lists_targets_and_graphs():
    schema = contract_json_schemas()["TargetOutput"]

    assert {"targets", "graphs"} <= set(schema["properties"])


def test_stream_collection_fields_describe_their_columns():
    definitions = contract_json_schemas()["TargetOutput"]["$defs"]
    report = definitions["TargetResults"]
    hpr_hot = report["properties"]["hpr_hot_streams"]["anyOf"][0]

    assert hpr_hot == {"$ref": "#/$defs/StreamCollection"}
    assert definitions["StreamCollection"]["properties"]["t_supply"]["items"][
        "type"
    ] == ["number", "null"]


def test_stream_collection_schema_requires_exactly_its_columns():
    streams = json.loads(
        json.dumps(PinchProblem("basic_pinch.json").hot_streams.to_dict())
    )
    missing = {key: value for key, value in streams.items() if key != "t_supply"}
    extra = {**streams, "unknown": []}

    jsonschema.validate(streams, STREAM_COLLECTION_JSON_SCHEMA)
    for payload in (missing, extra):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, STREAM_COLLECTION_JSON_SCHEMA)


def test_workspace_bundle_schema_is_json():
    schema = json.loads(json.dumps(contract_json_schemas()["PinchWorkspaceBundle"]))

    assert schema["type"] == "object"


def test_validation_report_schema_lists_issues():
    schema = contract_json_schemas()["ValidationReport"]

    assert "issues" in schema["properties"]


def test_option_metadata_schema_describes_editable_fields():
    schema = contract_json_schemas()["ConfigurationFieldMetadata"]

    assert {"name", "label", "field_type", "group"} <= set(schema["properties"])


def test_performance_map_request_schema_lists_operating_grid():
    schema = contract_json_schemas()["HprPerformanceMapRequest"]

    assert {"source_temperatures", "sink_temperatures"} <= set(schema["properties"])


def test_performance_map_schema_lists_points_and_units():
    schema = contract_json_schemas()["HprPerformanceMap"]

    assert {"points", "units", "thermodynamic_backend"} <= set(schema["properties"])


def test_utility_placement_result_schema_lists_best_candidate():
    schema = contract_json_schemas()["UtilityPlacementResult"]

    assert {"best", "period_ids", "request"} <= set(schema["properties"])


def test_dt_min_result_schema_lists_dt_min_and_limit():
    schema = contract_json_schemas()["HeatRecoveryDtMinResult"]

    assert {"dt_min", "thermodynamic_limit"} <= set(schema["properties"])


def test_network_synthesis_result_schema_lists_network():
    schema = contract_json_schemas()["HeatExchangerNetworkSynthesisResult"]

    assert {"network", "run_id"} <= set(schema["properties"])


def _subschemas(schema):
    yield schema
    for item in schema.values() if isinstance(schema, dict) else schema:
        if isinstance(item, dict | list):
            yield from _subschemas(item)


@pytest.mark.parametrize(
    ("title", "definition"),
    [
        ("Value", VALUE_JSON_SCHEMA),
        ("StreamCollection", STREAM_COLLECTION_JSON_SCHEMA),
    ],
)
def test_domain_types_are_one_shared_definition(title, definition):
    referencing = []
    for name, schema in contract_json_schemas().items():
        refs = [
            item
            for item in _subschemas(schema)
            if isinstance(item, dict) and item.get("$ref") == f"#/$defs/{title}"
        ]
        if not refs:
            continue
        referencing.append(name)
        assert schema["$defs"][title] == definition, name
        inline = [
            item
            for item in _subschemas(schema)
            if isinstance(item, dict) and item.get("title") == title
        ]
        assert inline == [schema["$defs"][title]], name
    assert "TargetOutput" in referencing
