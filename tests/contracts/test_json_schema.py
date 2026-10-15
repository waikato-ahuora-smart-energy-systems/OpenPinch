"""JSON Schema export of the public contracts, for typed clients (TypeScript)."""

from __future__ import annotations

import json

from OpenPinch.contracts.schema import contract_json_schemas


def test_problem_input_schema_lists_streams_and_utilities():
    schema = contract_json_schemas()["TargetInput"]

    assert {"streams", "utilities", "options"} <= set(schema["properties"])


def test_problem_output_schema_lists_targets_and_graphs():
    schema = contract_json_schemas()["TargetOutput"]

    assert {"targets", "graphs"} <= set(schema["properties"])


def test_stream_collection_fields_describe_their_columns():
    report = contract_json_schemas()["TargetOutput"]["$defs"]["TargetResults"]
    hpr_hot = report["properties"]["hpr_hot_streams"]["anyOf"][0]

    assert hpr_hot["properties"]["t_supply"]["items"]["type"] == ["number", "null"]


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
