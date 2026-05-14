"""Tests for the script-native PinchWorkspace surface."""

from __future__ import annotations

import json
from pathlib import Path

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.lib.enums import HPRcycle
from OpenPinch.resources import read_sample_case


def _basic_payload() -> dict:
    return json.loads(read_sample_case("basic_pinch.json"))


def _chocolate_payload() -> dict:
    return json.loads(read_sample_case("chocolate_factory.json"))


def test_pinch_workspace_returns_real_cases_and_delegates_to_pinchproblem():
    workspace = PinchWorkspace.from_payload(_basic_payload(), project_name="Demo")

    baseline = workspace.case("baseline")
    summary = workspace.summary_frame()
    direct_target_name = summary.loc[
        summary["Target"].str.endswith("/Direct Integration"),
        "Target",
    ].iloc[0]

    assert isinstance(baseline, PinchProblem)
    assert workspace.active_case_name == "baseline"
    assert "zone_tree" in workspace.get_case_payload("baseline")
    assert not summary.empty
    assert not workspace.plot.catalog().empty

    workspace.copy_case("baseline", "wide_dt", activate=False)
    workspace.set_dt_cont_multiplier(2.0, case_name="wide_dt")
    workspace.use_case("wide_dt")

    comparison = workspace.compare_cases(
        "baseline",
        "wide_dt",
        target_name=direct_target_name,
    )

    assert workspace.active_case_name == "wide_dt"
    assert "Change" in comparison.index


def test_pinch_workspace_updates_case_options_and_roundtrips_bundles(tmp_path: Path):
    workspace = PinchWorkspace.from_payload(_basic_payload(), project_name="Demo")
    workspace.copy_case("baseline", "wide_dt", activate=False)
    workspace.update_options({"DT_CONT": 15}, case_name="wide_dt")

    bundle_path = workspace.save_bundle(tmp_path / "pinch_workspace_bundle.json")
    reloaded = PinchWorkspace.load_bundle(bundle_path)

    assert workspace.get_case_payload("wide_dt")["options"]["DT_CONT"] == 15
    assert bundle_path.exists()
    assert reloaded.list_cases() == ["baseline", "wide_dt"]
    assert reloaded.get_case_payload("wide_dt")["options"]["DT_CONT"] == 15


def test_pinch_workspace_exposes_frontend_views_and_comparison_payloads():
    workspace = PinchWorkspace.from_payload(_basic_payload(), project_name="Demo")

    payload_view = workspace.payload_view("baseline")
    assert payload_view.variant_name == "baseline"
    assert payload_view.streams
    assert payload_view.streams[0].record_id == "streams[0]"

    workspace.set_variant_payload(
        "wide_dt",
        {"options": {"DT_CONT": 15}},
        base="baseline",
    )
    baseline_view = workspace.solve_variant("baseline")
    variant_view = workspace.solve_variant("wide_dt")
    comparison = workspace.compare_variants(["baseline", "wide_dt"])

    assert baseline_view.status == "solved"
    assert baseline_view.summary_table is not None
    assert baseline_view.graph_catalog
    assert baseline_view.graph_payloads
    assert any(table.table_kind == "shifted" for table in baseline_view.problem_tables)
    assert variant_view.status == "solved"
    assert comparison.metric_deltas


def test_pinch_workspace_validation_and_configuration_metadata():
    payload = _basic_payload()
    del payload["streams"][0]["t_target"]

    workspace = PinchWorkspace.from_payload(payload)
    report = workspace.validate_variant("baseline")
    invalid_view = workspace.solve_variant("baseline")
    fields = PinchWorkspace.configuration_field_metadata()
    by_name = {field.name: field for field in fields}

    assert report.valid is False
    assert any(issue.path == "streams[0].t_target" for issue in report.issues)
    assert invalid_view.status == "invalid"
    assert by_name["HPR_TYPE"].field_type == "enum"
    assert by_name["ELE_PRICE"].group == "costing"


def test_pinch_workspace_supports_advanced_workflows_on_real_cases():
    workspace = PinchWorkspace.from_payload(_chocolate_payload(), project_name="Site")
    case = workspace.copy_case("baseline", "direct_hp_full_load", activate=False)
    case.update_options(
        {
            "HPR_TYPE": HPRcycle.MultiTempCarnot.value,
            "HPR_LOAD_VALUE": 1.0,
            "MAX_HP_MULTISTART": 10,
            "N_COND": 3,
            "N_EVAP": 3,
            "REFRIGERANTS": "water;ammonia",
        }
    )

    target = case.target.direct_heat_pump()

    assert target.name.endswith("Direct Heat Pump")
    assert workspace.get_case_payload("direct_hp_full_load")["options"]["HPR_LOAD_VALUE"] == 1.0
    assert not case.plot.catalog().empty


def test_pinch_workspace_accepts_packaged_sample_case_name_as_source():
    workspace = PinchWorkspace(
        source="crude_preheat_train.json",
        project_name="crude_preheat_train",
    )

    case = workspace.case("baseline")
    summary = workspace.summary_frame()
    payload = workspace.get_case_payload("baseline")

    assert workspace.active_case_name == "baseline"
    assert case.project_name == "crude_preheat_train"
    assert payload["zone_tree"] is not None
    assert not summary.empty
