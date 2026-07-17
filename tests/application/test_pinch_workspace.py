"""Tests for the script-native PinchWorkspace surface."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
from pydantic import ValidationError

from OpenPinch.application._workspace import case_inputs
from OpenPinch.application._workspace.case_inputs import (
    canonical_case_input_from_source,
    normalise_case_input,
    project_name_from_case_input,
)
from OpenPinch.application._workspace.execution import (
    WorkspaceExecutionError,
    normalise_workflow_name,
    run_problem_workflow,
    workflow_support_level,
    workflow_warnings,
)
from OpenPinch.application._workspace.views import comparison as workspace_variants
from OpenPinch.application._workspace.views import input as workspace_inputs
from OpenPinch.application._workspace.views import (
    problem_table as workspace_problem_tables,
)
from OpenPinch.application._workspace.views import serialization as workspace_common
from OpenPinch.application._workspace.views.comparison import summary_metric_deltas
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.contracts.input import TargetInput
from OpenPinch.contracts.workspace import (
    PinchWorkspaceBundle,
    ProblemTableView,
    ScenarioVariantView,
    TableView,
    ValidationReport,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, HPRcycle, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.presentation.configuration import annotation_metadata
from OpenPinch.presentation.configuration import configuration_options as config_options
from OpenPinch.resources import read_sample_case


def _basic_payload() -> dict:
    return json.loads(read_sample_case("basic_pinch.json"))


def _chocolate_payload() -> dict:
    return json.loads(read_sample_case("chocolate_factory.json"))


def _network_payload() -> dict:
    network = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Process,
                stage=1,
                period_states=(
                    HeatExchangerPeriodState(
                        period_id="base",
                        period_idx=0,
                        duty=10.0,
                    ),
                ),
            ),
        )
    )
    return network.model_dump(mode="json")


def test_workspace_returns_real_cases_and_delegates_to_pinchproblem():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    baseline = workspace.case("baseline")
    summary = workspace.summary_frame()
    direct_target_name = summary.loc[
        summary["Target"].str.endswith("/Direct Integration"),
        "Target",
    ].iloc[0]

    assert isinstance(baseline, PinchProblem)
    assert workspace.active_case_name == "baseline"
    assert "zone_tree" in workspace.get_case_input("baseline")
    assert not summary.empty
    assert not workspace.plot.catalog().empty

    workspace.copy_case(source_name="baseline", new_name="wide_dt", activate=False)
    workspace.set_dt_cont_multiplier(2.0, case_name="wide_dt")
    workspace.use_case("wide_dt")

    comparison = workspace.compare_cases(
        "baseline",
        "wide_dt",
        target_name=direct_target_name,
    )

    assert workspace.active_case_name == "wide_dt"
    assert "Change" in comparison.index


def test_workspace_updates_case_options_and_roundtrips_bundles(tmp_path: Path):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace.copy_case(source_name="baseline", new_name="wide_dt", activate=False)
    workspace.update_options({"THERMAL_DT_CONT": 15}, case_name="wide_dt")

    bundle_path = workspace.save_bundle(tmp_path / "pinch_workspace_bundle.json")
    saved_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    reloaded = PinchWorkspace.load_bundle(bundle_path)

    assert workspace.get_case_input("wide_dt")["options"]["THERMAL_DT_CONT"] == 15
    assert bundle_path.exists()
    assert saved_bundle["schema_version"] == "2"
    assert "case_input" in saved_bundle["variants"]["baseline"]
    assert reloaded.list_cases() == ["baseline", "wide_dt"]
    assert reloaded.get_case_input("wide_dt")["options"]["THERMAL_DT_CONT"] == 15


def test_workspace_roundtrips_a_serialized_heat_exchanger_network(
    tmp_path: Path,
) -> None:
    payload = _basic_payload()
    network_payload = _network_payload()
    payload["network"] = network_payload
    workspace = PinchWorkspace(payload, project_name="Demo")

    bundle_path = workspace.save_bundle(tmp_path / "hen_workspace_bundle.json")
    saved_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    reloaded = PinchWorkspace.load_bundle(bundle_path)

    assert saved_bundle["variants"]["baseline"]["case_input"]["network"] == (
        network_payload
    )
    assert reloaded.get_case_input("baseline")["network"] == network_payload


def test_workspace_bundle_rejects_v1_payload_and_unknown_versions():
    base = {
        "project_name": "Demo",
        "baseline_name": "baseline",
        "variants": {"baseline": {"case_input": _basic_payload()}},
    }

    valid = PinchWorkspaceBundle.model_validate({"schema_version": "2", **base})
    assert valid.schema_version == "2"

    with pytest.raises(ValidationError, match="Unsupported workspace schema_version"):
        PinchWorkspaceBundle.model_validate({"schema_version": "1", **base})
    with pytest.raises(ValidationError, match="Unsupported workspace schema_version"):
        PinchWorkspaceBundle.model_validate({"schema_version": "future", **base})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PinchWorkspaceBundle.model_validate(
            {
                "schema_version": "2",
                **base,
                "variants": {"baseline": {"payload": _basic_payload()}},
            }
        )


def test_workspace_scenario_helper_and_report_delegates():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    scenario = workspace.scenario(
        "wide_dt",
        options={"THERMAL_DT_CONT": 15},
        dt_cont_multiplier=2.0,
        activate=True,
        solve=True,
    )
    report = workspace.report(case_name="wide_dt", solve=False)
    metrics = workspace.metrics(case_name="wide_dt")
    validation = workspace.validation_report("wide_dt")

    assert isinstance(scenario, PinchProblem)
    assert workspace.active_case_name == "wide_dt"
    assert workspace.get_case_input("wide_dt")["options"]["THERMAL_DT_CONT"] == 15
    assert validation.valid is True
    assert report.solved is True
    assert any(metric.metric == "Hot Utility Target" for metric in metrics)


def test_workspace_exposes_frontend_views_and_comparison_payloads():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    input_view = workspace.input_view("baseline")
    assert input_view.variant_name == "baseline"
    assert input_view.streams
    assert input_view.streams[0].record_id == "streams[0]"

    workspace.set_variant_input(
        "wide_dt",
        {"options": {"THERMAL_DT_CONT": 15}},
        base="baseline",
    )
    baseline_view = workspace.solve_variant("baseline")
    variant_view = workspace.solve_variant("wide_dt")
    comparison = workspace.compare_variants(["baseline", "wide_dt"])

    assert baseline_view.status == "solved"
    assert baseline_view.summary_table is not None
    assert baseline_view.graph_catalog
    assert baseline_view.graph_data_entries
    assert any(table.table_kind == "shifted" for table in baseline_view.problem_tables)
    assert variant_view.status == "solved"
    assert comparison.metric_deltas
    assert any(card.value for card in baseline_view.summary_cards)
    assert any(delta.variant_value for delta in comparison.metric_deltas)


def test_workspace_metric_deltas_require_matching_units():
    base_view = ScenarioVariantView(
        variant_name="baseline",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="solved",
        support_level="supported",
        validation=ValidationReport(valid=True),
        summary_table=TableView(
            columns=["Target", "Hot Utility Target", "Hot Utility Target (unit)"],
            rows=[
                {
                    "Target": "Plant/Direct Integration",
                    "Hot Utility Target": 100.0,
                    "Hot Utility Target (unit)": "kW",
                }
            ],
        ),
    )
    variant_view = ScenarioVariantView(
        variant_name="scenario",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="solved",
        support_level="supported",
        validation=ValidationReport(valid=True),
        summary_table=TableView(
            columns=["Target", "Hot Utility Target", "Hot Utility Target (unit)"],
            rows=[
                {
                    "Target": "Plant/Direct Integration",
                    "Hot Utility Target": 120.0,
                    "Hot Utility Target (unit)": "MW",
                }
            ],
        ),
    )

    deltas = summary_metric_deltas("baseline", base_view, "scenario", variant_view)
    hot_delta = next(delta for delta in deltas if delta.metric == "Hot Utility Target")

    assert hot_delta.unit == "kW"
    assert hot_delta.delta is None


def test_workspace_validation_and_configuration_metadata():
    payload = _basic_payload()
    del payload["streams"][0]["t_target"]

    workspace = PinchWorkspace(payload)
    report = workspace.validate_variant("baseline")
    invalid_view = workspace.solve_variant("baseline")
    fields = PinchWorkspace.configuration_field_metadata()
    root_fields = config_options()
    by_name = {field.name: field for field in fields}

    assert root_fields[0].name == fields[0].name
    assert report.valid is False
    assert any(issue.path == "streams[0].t_target" for issue in report.issues)
    assert invalid_view.status == "invalid"
    assert by_name["HPR_TYPE"].field_type == "enum"
    assert by_name["DIRECT_ASSISTED_HT_DT"].group == "direct"
    assert by_name["COSTING_HPR_ELE_PRICE"].group == "costing"
    assert by_name["TARGETING_EXERGY_ENABLED"].runtime_status == "experimental"
    assert by_name["HENS_SOLVER_EVM"].config_path == ["hens", "solver_evm"]


def test_workspace_reports_error_category_for_unsupported_workflow():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    view = workspace.solve_variant("baseline", workflow="not_a_real_workflow")

    assert view.status == "error"
    assert view.error_category == "unsupported_workflow"


def test_workspace_supports_advanced_workflows_on_real_cases():
    workspace = PinchWorkspace(_chocolate_payload(), project_name="Site")
    case = workspace.copy_case(
        source_name="baseline", new_name="direct_hp_full_load", activate=False
    )
    case.update_options(
        {
            "HPR_TYPE": HPRcycle.CascadeCarnot.value,
            "HPR_LOAD_MODE": "fraction",
            "HPR_LOAD_FRACTION": 1.0,
            "HPR_MAX_MULTISTART": 10,
            "HPR_N_COND": 3,
            "HPR_N_EVAP": 3,
            "HPR_REFRIGERANTS": ["water", "ammonia"],
        }
    )

    target = case.target.direct_heat_pump()

    assert target.name.endswith("Direct Heat Pump")
    assert (
        workspace.get_case_input("direct_hp_full_load")["options"]["HPR_LOAD_FRACTION"]
        == 1.0
    )
    assert not case.plot.catalog().empty


def test_workspace_accepts_existing_problem_as_constructor_source():
    problem = PinchProblem(source=_basic_payload(), project_name="Demo")
    workspace = PinchWorkspace(problem)

    assert workspace.active_case_name == "baseline"
    assert workspace.case("baseline").project_name == "Demo"
    assert workspace.get_case_input("baseline")["zone_tree"] is not None


def test_workspace_case_input_helpers_accept_target_input_and_missing_project_names(
    monkeypatch,
):
    target_input = TargetInput(streams=[])
    case_input = normalise_case_input(target_input)

    assert case_input["streams"] == []
    with pytest.raises(TypeError, match="dict or TargetInput"):
        normalise_case_input(["not", "a", "case"])
    assert project_name_from_case_input({}) is None
    assert project_name_from_case_input({"zone_tree": {"name": ""}}) is None

    normalized, project_name = canonical_case_input_from_source(
        target_input,
        project_name=None,
        workspace_project_name="WorkspaceSite",
    )

    assert normalized["streams"] == []
    assert project_name == "WorkspaceSite"

    class RaisingProblem:
        def __init__(self, source, project_name):
            raise ValueError("invalid source")

    monkeypatch.setattr(case_inputs, "PinchProblem", RaisingProblem)
    with pytest.raises(ValueError, match="invalid source"):
        canonical_case_input_from_source(
            "not-a-case-source",
            project_name=None,
            workspace_project_name=None,
        )


def test_workspace_accepts_packaged_sample_case_name_as_source():
    workspace = PinchWorkspace(
        source="crude_preheat_train.json",
        project_name="crude_preheat_train",
    )

    case = workspace.case("baseline")
    summary = workspace.summary_frame()
    payload = workspace.get_case_input("baseline")

    assert workspace.active_case_name == "baseline"
    assert case.project_name == "crude_preheat_train"
    assert payload["zone_tree"] is not None
    assert not summary.empty


def test_workspace_view_helpers_handle_empty_and_mixed_inputs():
    assert (
        workspace_problem_tables.problem_table_views(SimpleNamespace(master_zone=None))
        == []
    )
    assert workspace_variants.summary_rows_by_target(None) == {}
    assert workspace_variants.count_changed_cells(None, None, []) is None
    assert workspace_inputs.zone_tree_view("not-a-tree") == []
    assert (
        workspace_inputs.record_views({"name": "not-a-list"}, section="streams") == []
    )

    records = workspace_inputs.record_views(
        ["skip-me", {"name": "Feed", "zone": ""}],
        section="streams",
    )

    assert len(records) == 1
    assert records[0].record_id == "streams[1]"
    assert records[0].name == "Feed"
    assert records[0].zone is None


def test_workspace_problem_table_views_skip_empty_generated_tables():
    empty_table = SimpleNamespace(data=None, columns=[])
    target = SimpleNamespace(
        name="Target",
        pt=empty_table,
        pt_real=empty_table,
    )
    zone = SimpleNamespace(
        targets={"Target": target},
        subzones={},
    )

    assert (
        workspace_problem_tables.problem_table_views(SimpleNamespace(master_zone=zone))
        == []
    )


def test_workspace_diff_helpers_count_cell_changes_and_missing_tables():
    base_table = ProblemTableView(
        table_id="target::shifted",
        target_id="target",
        target_name="Target",
        table_kind="shifted",
        table=TableView(
            columns=["temperature", "duty"],
            rows=[{"temperature": 100.0, "duty": 10.0}],
        ),
    )
    variant_table = ProblemTableView(
        table_id="target::shifted",
        target_id="target",
        target_name="Target",
        table_kind="shifted",
        table=TableView(
            columns=["temperature", "duty"],
            rows=[{"temperature": 105.0, "duty": 10.0}],
        ),
    )

    assert (
        workspace_variants.count_changed_cells(
            base_table,
            variant_table,
            ["temperature", "duty"],
        )
        == 1
    )

    base_view = ScenarioVariantView(
        variant_name="baseline",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="solved",
        support_level="supported",
        validation=ValidationReport(valid=True),
        problem_tables=[base_table],
    )
    variant_view = ScenarioVariantView(
        variant_name="variant",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="solved",
        support_level="supported",
        validation=ValidationReport(valid=True),
        problem_tables=[variant_table],
    )
    missing_view = ScenarioVariantView(
        variant_name="missing",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="solved",
        support_level="supported",
        validation=ValidationReport(valid=True),
        problem_tables=[],
    )

    changed = workspace_variants.problem_table_diffs(
        "baseline",
        base_view,
        "variant",
        variant_view,
    )
    missing = workspace_variants.problem_table_diffs(
        "baseline",
        base_view,
        "missing",
        missing_view,
    )

    assert changed[0].changed_cells == 1
    assert changed[0].shape_changed is False
    assert missing[0].changed_cells is None
    assert missing[0].shape_changed is True


def test_workspace_scalar_metadata_helpers_cover_annotation_and_numeric_edges():
    class Color(Enum):
        RED = "red"

    assert annotation_metadata(str, enum_cls=Color) == ("enum", False)
    assert annotation_metadata(List[str], enum_cls=None) == (
        "string_list",
        True,
    )
    assert annotation_metadata(list[str], enum_cls=None) == (
        "string_list",
        True,
    )
    assert annotation_metadata("bool option", enum_cls=None) == (
        "boolean",
        False,
    )
    assert annotation_metadata("dict option", enum_cls=None) == (
        "object",
        False,
    )
    assert annotation_metadata(object(), enum_cls=None) == (
        "string",
        False,
    )
    assert workspace_common.numeric_delta(True, 1) is None
    assert workspace_common.numeric_delta(2.5, 5) == 2.5
    assert workspace_common.maybe_float(None) is None
    assert workspace_common.maybe_float("3.5") == 3.5
    assert workspace_common.maybe_float("not-a-number") is None
    assert workspace_common.maybe_float(float("inf")) is None


def test_workspace_json_safe_handles_enum_paths_and_model_like_objects():
    class Color(Enum):
        RED = "red"

    class ItemValue:
        def item(self):
            return 12

    class FailingItem:
        def item(self):
            raise RuntimeError("cannot unwrap")

        def __str__(self):
            return "failing-item"

    class DumpValue:
        def model_dump(self, *, mode):
            return {"mode": mode, "path": Path("artifact.json")}

    class DictValue:
        def to_dict(self):
            return {"value": 3}

    class FailingDictValue:
        def to_dict(self):
            raise RuntimeError("cannot convert")

        def __str__(self):
            return "failing-dict"

    assert workspace_common.json_safe(Color.RED) == "red"
    assert workspace_common.json_safe(Path("data.json")) == "data.json"
    assert workspace_common.json_safe(ItemValue()) == 12
    assert workspace_common.json_safe(FailingItem()) == "failing-item"
    assert workspace_common.json_safe(workspace_common.pd.Timestamp("2024-01-02")) == (
        "2024-01-02T00:00:00"
    )
    assert workspace_common.json_safe(workspace_common.pd.NA) is None
    assert workspace_common.json_safe(DumpValue()) == {
        "mode": "python",
        "path": "artifact.json",
    }
    assert workspace_common.json_safe(DictValue()) == {"value": 3}
    assert workspace_common.json_safe(FailingDictValue()) == "failing-dict"


def test_workspace_json_safe_ignores_isna_errors(monkeypatch):
    class IsnaError:
        def __str__(self):
            return "isna-error"

    def raise_isna(_value):
        raise RuntimeError("cannot inspect")

    monkeypatch.setattr(workspace_common.pd, "isna", raise_isna)

    assert workspace_common.json_safe(IsnaError()) == "isna-error"


def test_workspace_from_json_repr_and_load_none_delegation():
    workspace = PinchWorkspace.from_json(
        _basic_payload(),
        baseline_name="base",
        project_name="Demo",
    )

    loaded_case = workspace.load(None)
    representation = repr(workspace)

    assert loaded_case is workspace.case("base")
    assert "PinchWorkspace" in representation
    assert "active_case='base'" in representation
    assert workspace.to_problem_json(case_name="base") == workspace.get_case_input(
        "base"
    )


def test_workspace_set_variant_input_sets_active_case_when_empty():
    workspace = PinchWorkspace()

    stored = workspace.set_variant_input("scenario", _basic_payload())

    assert workspace.active_case_name == "scenario"
    assert stored["streams"]


def test_workspace_solve_variant_reports_unexpected_errors(monkeypatch):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    monkeypatch.setattr(
        workspace,
        "case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    view = workspace.solve_variant("baseline")

    assert view.status == "error"
    assert view.error_category == "unexpected_error"
    assert "boom" in view.error_message


def test_workspace_compare_variants_empty_and_base_insertion(monkeypatch):
    workspace = PinchWorkspace()
    with pytest.raises(ValueError, match="At least one variant"):
        workspace.compare_variants([])

    workspace._variant_inputs = {
        "baseline": _basic_payload(),
        "scenario": _basic_payload(),
    }
    solved_view = ScenarioVariantView(
        variant_name="baseline",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="solved",
        support_level="supported",
        validation=ValidationReport(valid=True),
        graph_catalog=[],
        problem_tables=[],
    )
    monkeypatch.setattr(workspace, "_ensure_solved_view", lambda name: solved_view)

    comparison = workspace.compare_variants(["scenario"])

    assert comparison.variant_names == ["baseline", "scenario"]


def test_workspace_active_case_delegates_and_compare_to(monkeypatch, tmp_path: Path):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    other_workspace = PinchWorkspace(_basic_payload(), project_name="Other")
    case = workspace.case()
    captured = {}

    monkeypatch.setattr(case, "export_excel", lambda results_dir: Path(results_dir))
    monkeypatch.setattr(
        case,
        "show_dashboard",
        lambda **kwargs: captured.setdefault("dashboard", kwargs),
    )

    def fake_compare_to(other, **kwargs):
        captured["compare"] = (other, kwargs)

    monkeypatch.setattr(case, "compare_to", fake_compare_to)

    assert workspace.target is not None
    assert workspace.plot is not None
    assert workspace.problem_data is not None
    assert workspace.problem_filepath is None
    assert workspace.results is None
    assert workspace.master_zone is case.master_zone
    assert workspace.validate().streams
    assert workspace.export_excel(tmp_path) == tmp_path

    workspace.show_dashboard(page_title="Dash")
    assert captured["dashboard"]["page_title"] == "Dash"

    workspace.compare_to(other_workspace, other_case_name="baseline")
    assert captured["compare"][0] is other_workspace.case("baseline")

    other_problem = PinchProblem(source=_basic_payload(), project_name="Other")
    workspace.compare_to(other_problem)
    assert captured["compare"][0] is other_problem


def test_workspace_case_resolution_and_default_fallback_guards():
    empty = PinchWorkspace()
    with pytest.raises(KeyError, match="No cases"):
        empty.case()

    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    with pytest.raises(KeyError, match="Unknown case"):
        workspace.case("missing")
    with pytest.raises(KeyError, match="Unknown variant"):
        workspace._get_variant_input("missing")

    fallback = PinchWorkspace(baseline_name="baseline")
    fallback._variant_inputs = {"scenario": _basic_payload()}

    assert fallback._default_case_name() == "scenario"
    assert fallback.active_case_name == "scenario"


def test_workspace_ensure_solved_view_and_sync_cache_paths(monkeypatch):
    workspace = PinchWorkspace()
    workspace._variant_inputs = {"baseline": _basic_payload()}
    invalid_view = ScenarioVariantView(
        variant_name="baseline",
        period_id=None,
        workflow="target",
        workflow_options={},
        status="invalid",
        support_level="supported",
        validation=ValidationReport(valid=False),
    )
    monkeypatch.setattr(
        workspace, "solve_variant", lambda *args, **kwargs: invalid_view
    )

    with pytest.raises(ValueError, match="not solved"):
        workspace._ensure_solved_view("baseline")

    workspace._case_cache["baseline"] = SimpleNamespace(
        canonical_problem_json=lambda: {"changed": True}
    )
    workspace._cached_views["baseline"] = invalid_view

    workspace._sync_case_input("baseline")

    assert workspace.get_case_input("baseline", canonical=False) == {"changed": True}
    assert "baseline" not in workspace._cached_views


def test_workspace_execution_helpers_normalise_and_warn_for_support_levels():
    assert normalise_workflow_name(" Direct Heat-Pump ") == "direct_heat_pump"
    assert workflow_support_level("target") == "stable"
    assert workflow_support_level("Pinch Design Method") == "advanced"
    assert workflow_support_level("not real") == "unsupported"
    assert workflow_warnings("target", "stable") == []
    assert "advanced" in workflow_warnings("pinch_design_method", "advanced")[0]
    assert "not a supported" in workflow_warnings("custom", "unsupported")[0]
    assert (
        str(
            WorkspaceExecutionError(
                category="workflow_runtime",
                message="boom",
            )
        )
        == "boom"
    )


def test_run_problem_workflow_covers_target_design_and_error_paths():
    calls = []

    class TargetAccessor:
        def __call__(self):
            calls.append(("target", None))

        def direct_heat_integration(self, **kwargs):
            calls.append(("direct_heat_integration", kwargs))

        def failing_target(self, **_kwargs):
            raise RuntimeError("target failure")

        def raises_structured(self, **_kwargs):
            raise WorkspaceExecutionError(
                category="structured",
                message="structured failure",
            )

    class DesignAccessor:
        def pinch_design_method(self, **kwargs):
            calls.append(("pinch_design_method", kwargs))

    problem = SimpleNamespace(target=TargetAccessor(), design=DesignAccessor())

    run_problem_workflow(problem, "target", {})
    run_problem_workflow(problem, "direct_heat_integration", {"period_id": "0"})
    run_problem_workflow(
        problem,
        "pinch_design_method",
        {"solver": "fake"},
        workspace_variant="variant_a",
    )

    assert calls == [
        ("target", None),
        ("target", None),
        ("direct_heat_integration", {"period_id": "0"}),
        (
            "pinch_design_method",
            {"solver": "fake", "workspace_variant": "variant_a"},
        ),
    ]

    with pytest.raises(WorkspaceExecutionError, match="Unknown design workflow") as exc:
        run_problem_workflow(problem, "open_hens_method", {})
    assert exc.value.category == "unsupported_workflow"

    problem.design = SimpleNamespace(
        open_hens_method=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("design failure")
        )
    )
    with pytest.raises(WorkspaceExecutionError, match="design failure") as exc:
        run_problem_workflow(problem, "open_hens_method", {})
    assert exc.value.category == "workflow_runtime"

    problem.design = SimpleNamespace(
        open_hens_method=lambda **_kwargs: (_ for _ in ()).throw(
            WorkspaceExecutionError(
                category="structured_design",
                message="structured design failure",
            )
        )
    )
    with pytest.raises(
        WorkspaceExecutionError,
        match="structured design failure",
    ) as exc:
        run_problem_workflow(problem, "open_hens_method", {})
    assert exc.value.category == "structured_design"

    with pytest.raises(WorkspaceExecutionError, match="Unknown workflow") as exc:
        run_problem_workflow(problem, "not_real", {})
    assert exc.value.category == "unsupported_workflow"

    with pytest.raises(WorkspaceExecutionError, match="target failure") as exc:
        run_problem_workflow(problem, "failing_target", {})
    assert exc.value.category == "workflow_runtime"

    with pytest.raises(WorkspaceExecutionError, match="structured failure") as exc:
        run_problem_workflow(problem, "raises_structured", {})
    assert exc.value.category == "structured"
