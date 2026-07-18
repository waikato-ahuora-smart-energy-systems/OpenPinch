"""Tests for the script-native PinchWorkspace surface."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from OpenPinch.application._workspace import case_inputs
from OpenPinch.application._workspace.case_inputs import (
    canonical_case_input_from_source,
    normalise_case_input,
    project_name_from_case_input,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.contracts.input import TargetInput
from OpenPinch.contracts.workspace import (
    PinchWorkspaceBundle,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
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


_INVALID_CASE_NAMES = (
    "",
    " baseline",
    "baseline ",
    ".",
    "..",
    "a/b",
    "a\\b",
    "bad\nname",
    "a:b",
    'a"b',
    "a|b",
    "a?b",
    "a*b",
    "a<b",
    "a>b",
    "trailing.",
    "CON",
    "con.txt",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM9.log",
    "LPT1",
)

_VALID_CASE_NAMES = (
    "baseline",
    "Retrofit 2026",
    "unité",
    "case.v2",
    "wide_dt",
    "case-study",
)

_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _is_generated_portable_case_name(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and not value.endswith(".")
        and value.split(".", 1)[0].upper() not in _WINDOWS_DEVICE_NAMES
    )


_GENERATED_VALID_CASE_NAMES = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _.-é",
    min_size=1,
    max_size=24,
).filter(_is_generated_portable_case_name)

_GENERATED_INVALID_CASE_NAMES = st.one_of(
    st.sampled_from(_INVALID_CASE_NAMES),
    st.text(min_size=0, max_size=12).map(lambda value: f"{value}/case"),
    st.text(min_size=0, max_size=12).map(lambda value: f"case\\{value}"),
)


def test_workspace_returns_real_cases_and_delegates_to_pinchproblem():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    baseline = workspace.case("baseline")
    baseline.target.all_heat_integration()
    summary = workspace.summary_frame()
    direct_target_name = summary.loc[
        summary["Target"].str.endswith("/Direct Integration"),
        "Target",
    ].iloc[0]

    assert isinstance(baseline, PinchProblem)
    assert workspace.active_case_name == "baseline"
    assert "zone_tree" in workspace.to_problem_json(case_name="baseline")
    assert not summary.empty
    assert not workspace.plot.catalog().empty

    workspace.scenario("wide_dt", base="baseline", activate=False)
    workspace.set_dt_cont_multiplier(2.0, case_name="wide_dt")
    workspace.use_case("wide_dt")
    workspace.case("wide_dt").target.all_heat_integration()

    comparison = workspace.compare_cases(
        "baseline",
        "wide_dt",
        target_name=direct_target_name,
    )

    assert workspace.active_case_name == "wide_dt"
    assert "Change" in comparison.index


def test_workspace_updates_case_options_and_roundtrips_bundles(tmp_path: Path):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace.scenario("wide_dt", base="baseline", activate=False)
    workspace.update_options({"THERMAL_DT_CONT": 15}, case_name="wide_dt")

    bundle_path = workspace.save_bundle(tmp_path / "pinch_workspace_bundle.json")
    saved_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    reloaded = PinchWorkspace.load_bundle(bundle_path)

    assert (
        workspace.to_problem_json(case_name="wide_dt")["options"]["THERMAL_DT_CONT"]
        == 15
    )
    assert bundle_path.exists()
    assert saved_bundle["schema_version"] == "3"
    assert "case_input" in saved_bundle["cases"]["baseline"]
    assert reloaded.list_cases() == ["baseline", "wide_dt"]
    assert (
        reloaded.to_problem_json(case_name="wide_dt")["options"]["THERMAL_DT_CONT"]
        == 15
    )


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

    assert saved_bundle["cases"]["baseline"]["case_input"]["network"] == (
        network_payload
    )
    assert reloaded.to_problem_json(case_name="baseline")["network"] == network_payload


def test_workspace_bundle_rejects_old_payloads_and_unknown_versions():
    base = {
        "project_name": "Demo",
        "baseline_name": "baseline",
        "cases": {"baseline": {"case_input": _basic_payload()}},
    }

    valid = PinchWorkspaceBundle.model_validate({"schema_version": "3", **base})
    assert valid.schema_version == "3"

    with pytest.raises(ValidationError, match="Unsupported workspace schema_version"):
        PinchWorkspaceBundle.model_validate(base)
    with pytest.raises(ValidationError, match="Unsupported workspace schema_version"):
        PinchWorkspaceBundle.model_validate({"schema_version": "1", **base})
    with pytest.raises(ValidationError, match="Unsupported workspace schema_version"):
        PinchWorkspaceBundle.model_validate({"schema_version": "future", **base})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PinchWorkspaceBundle.model_validate(
            {
                "schema_version": "3",
                **base,
                "cases": {"baseline": {"payload": _basic_payload()}},
            }
        )


def test_workspace_bundle_validates_generic_mapping_case_keys():
    payload = MappingProxyType(
        {
            "schema_version": "3",
            "baseline_name": "baseline",
            "cases": MappingProxyType({"../escape": {"case_input": _basic_payload()}}),
        }
    )

    with pytest.raises(ValidationError, match="case name"):
        PinchWorkspaceBundle.model_validate(payload)


def test_workspace_scenario_helper_and_report_delegates():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    scenario = workspace.scenario(
        "wide_dt",
        options={"THERMAL_DT_CONT": 15},
        dt_cont_multiplier=2.0,
        activate=True,
    )
    scenario.target.all_heat_integration()
    report = workspace.report(case_name="wide_dt")
    metrics = workspace.metrics(case_name="wide_dt")
    validation = workspace.validation_report("wide_dt")

    assert isinstance(scenario, PinchProblem)
    assert workspace.active_case_name == "wide_dt"
    assert (
        workspace.to_problem_json(case_name="wide_dt")["options"]["THERMAL_DT_CONT"]
        == 15
    )
    assert validation.valid is True
    assert report.solved is True
    assert any(metric.metric == "Hot Utility Target" for metric in metrics)


def test_workspace_cases_batch_preserves_order_and_returns_results():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace.scenario("wide_dt", options={"THERMAL_DT_CONT": 15})

    outcome = workspace.cases(["wide_dt", "baseline"]).target.direct_heat_integration()

    assert list(outcome.results) == ["wide_dt", "baseline"]
    assert outcome.errors == {}


def test_workspace_validation_uses_case_vocabulary():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    assert workspace.validation_report("baseline").valid is True
    assert not hasattr(workspace, "validate_variant")
    assert not hasattr(workspace, "configuration_field_metadata")


def test_workspace_has_no_workflow_string_dispatch_surface():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    assert not hasattr(workspace, "solve_variant")
    assert not hasattr(workspace, "compare_variants")


def test_workspace_supports_advanced_workflows_on_real_cases():
    workspace = PinchWorkspace(_chocolate_payload(), project_name="Site")
    case = workspace.scenario("direct_hp_full_load", base="baseline", activate=False)
    original_options = case.to_problem_json()["options"]

    target = case.target.carnot_heat_pump(
        load_fraction=0.25,
        maximum_restarts=1,
        condensers=1,
        evaporators=1,
    )

    assert target.name.endswith("Direct Heat Pump")
    assert case.to_problem_json()["options"] == original_options
    assert not case.plot.catalog().empty


def test_workspace_accepts_existing_problem_as_constructor_source():
    problem = PinchProblem(source=_basic_payload(), project_name="Demo")
    workspace = PinchWorkspace(problem)

    assert workspace.active_case_name == "baseline"
    assert workspace.case("baseline").project_name == "Demo"
    assert workspace.to_problem_json(case_name="baseline")["zone_tree"] is not None


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
    case.target.all_heat_integration()
    summary = workspace.summary_frame()
    payload = workspace.to_problem_json(case_name="baseline")

    assert workspace.active_case_name == "baseline"
    assert case.project_name == "crude_preheat_train"
    assert payload["zone_tree"] is not None
    assert not summary.empty


def test_workspace_constructor_repr_and_load_none_delegation():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")

    assert "baseline" in repr(workspace)
    assert workspace.load(None) is workspace.case("baseline")
    assert not hasattr(PinchWorkspace, "from_json")


def test_workspace_load_sets_active_case_when_empty():
    workspace = PinchWorkspace()

    stored = workspace.load(_basic_payload(), case_name="scenario")

    assert isinstance(stored, PinchProblem)
    assert workspace.active_case_name == "scenario"
    assert not hasattr(workspace, "set_variant_input")


def test_workspace_batch_isolates_case_failures(monkeypatch):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace.scenario("broken")
    broken = workspace.case("broken")
    monkeypatch.setattr(
        broken,
        "_execute_targeting",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    outcome = workspace.cases(["baseline", "broken"]).target.direct_heat_integration()

    assert list(outcome.results) == ["baseline"]
    assert isinstance(outcome.errors["broken"], RuntimeError)


def test_workspace_batch_namespaces_expose_only_valid_workflows():
    batch = PinchWorkspace(_basic_payload(), project_name="Demo").cases()

    target_methods = {
        name
        for name in dir(batch.target)
        if not name.startswith("_") and callable(getattr(batch.target, name))
    }
    all_period_methods = {
        name
        for name in dir(batch.target.all_periods)
        if not name.startswith("_")
        and callable(getattr(batch.target.all_periods, name))
    }
    design_methods = {
        name
        for name in dir(batch.design)
        if not name.startswith("_") and callable(getattr(batch.design, name))
    }

    assert "direct_heat_integration" in target_methods
    assert "heat_exchanger_network" not in target_methods
    assert "heat_exchanger_network" in design_methods
    assert "direct_heat_integration" not in design_methods
    assert "direct_heat_integration" in all_period_methods
    assert "brayton_heat_pump" not in all_period_methods


def test_workspace_batch_mirrors_observation_and_export_surfaces(
    monkeypatch, tmp_path: Path
):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace.scenario("retrofit")
    batch = workspace.cases(["baseline", "retrofit"])

    for name in batch.names:
        case = workspace.case(name)
        monkeypatch.setattr(case, "summary_frame", lambda **_kwargs: "summary")
        monkeypatch.setattr(case, "metrics", lambda **_kwargs: "metrics")
        monkeypatch.setattr(case, "report", lambda **_kwargs: "report")
        monkeypatch.setattr(
            case,
            "export_excel",
            lambda destination, **_kwargs: Path(destination),
        )

    assert list(batch.summary_frames().results) == ["baseline", "retrofit"]
    assert set(batch.metrics().results.values()) == {"metrics"}
    assert set(batch.reports().results.values()) == {"report"}
    exported = batch.export_excel(tmp_path)
    assert exported.results["baseline"] == tmp_path / "baseline"
    assert exported.results["retrofit"] == tmp_path / "retrofit"


def test_workspace_compare_cases_uses_explicit_solved_cases():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    scenario = workspace.scenario("scenario", dt_cont_multiplier=2.0)
    workspace.case("baseline").target.direct_heat_integration()
    scenario.target.direct_heat_integration()

    comparison = workspace.compare_cases("baseline", "scenario")

    assert "Change" in comparison.index


def test_workspace_active_case_delegates_and_compare_to(monkeypatch, tmp_path: Path):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    other_workspace = PinchWorkspace(_basic_payload(), project_name="Other")
    case = workspace.case()
    captured = {}

    monkeypatch.setattr(
        case, "export_excel", lambda destination, **kwargs: Path(destination)
    )
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
    with pytest.raises(KeyError, match="Unknown case"):
        workspace._get_case_input("missing")

    fallback = PinchWorkspace(baseline_name="baseline")
    fallback._case_inputs = {"scenario": _basic_payload()}

    assert fallback._default_case_name() == "scenario"
    assert fallback.active_case_name == "scenario"


def test_workspace_sync_case_input_uses_canonical_problem_json():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    case = workspace.case("baseline")
    case.update_options({"THERMAL_DT_CONT": 15})

    workspace._sync_case_input("baseline")

    assert (
        workspace.to_problem_json(case_name="baseline")["options"]["THERMAL_DT_CONT"]
        == 15
    )


def test_workspace_problem_data_returns_detached_active_case_snapshot():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    snapshot = workspace.problem_data

    assert isinstance(snapshot, dict)
    snapshot["streams"][0]["t_supply"]["value"] = 99.0

    assert (
        workspace.to_problem_json(case_name="baseline")["streams"][0]["t_supply"][
            "value"
        ]
        == 20.0
    )


@pytest.mark.parametrize("case_name", _INVALID_CASE_NAMES)
def test_workspace_rejects_invalid_case_names_at_runtime_boundaries(case_name):
    with pytest.raises(ValueError, match="case name"):
        PinchWorkspace(baseline_name=case_name)

    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    with pytest.raises(ValueError, match="case name"):
        workspace.load(_basic_payload(), case_name=case_name)
    with pytest.raises(ValueError, match="case name"):
        workspace.scenario(case_name)


@pytest.mark.parametrize("case_name", _VALID_CASE_NAMES)
def test_workspace_preserves_valid_case_names(case_name):
    workspace = PinchWorkspace(
        _basic_payload(),
        project_name="Demo",
        baseline_name=case_name,
    )
    bundle = PinchWorkspaceBundle.model_validate(
        {
            "schema_version": "3",
            "baseline_name": case_name,
            "cases": {case_name: {"case_input": _basic_payload()}},
        }
    )

    assert workspace.list_cases() == [case_name]
    assert bundle.baseline_name == case_name
    assert list(bundle.cases) == [case_name]


@pytest.mark.parametrize("case_name", _INVALID_CASE_NAMES)
def test_workspace_bundle_rejects_invalid_case_names(case_name):
    with pytest.raises(ValidationError, match="case name"):
        PinchWorkspaceBundle.model_validate(
            {
                "schema_version": "3",
                "baseline_name": case_name,
                "cases": {"baseline": {"case_input": _basic_payload()}},
            }
        )

    with pytest.raises(ValidationError, match="case name"):
        PinchWorkspaceBundle.model_validate(
            {
                "schema_version": "3",
                "baseline_name": "baseline",
                "cases": {case_name: {"case_input": _basic_payload()}},
            }
        )


@given(_GENERATED_VALID_CASE_NAMES)
def test_generated_valid_case_names_remain_single_workspace_keys(case_name):
    workspace = PinchWorkspace(baseline_name=case_name)

    assert workspace.baseline_name == case_name
    assert Path(case_name).name == case_name


@given(_GENERATED_INVALID_CASE_NAMES)
def test_generated_invalid_case_names_are_rejected(case_name):
    with pytest.raises(ValueError, match="case name"):
        PinchWorkspace(baseline_name=case_name)


def test_batch_export_rejects_corrupted_unsafe_case_without_calling_exporter(
    monkeypatch,
    tmp_path: Path,
):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace._case_inputs["../outside"] = _basic_payload()
    called: list[Path] = []

    def fake_export(_problem, destination, **_kwargs):
        called.append(Path(destination))
        return Path(destination)

    monkeypatch.setattr(PinchProblem, "export_excel", fake_export)

    exported = workspace.cases(["baseline", "../outside"]).export_excel(tmp_path)

    assert list(exported.results) == ["baseline"]
    assert list(exported.errors) == ["../outside"]
    assert "case name" in str(exported.errors["../outside"])
    assert called == [tmp_path.resolve() / "baseline"]
    assert called[0].is_relative_to(tmp_path.resolve())


def test_batch_export_rejects_existing_case_symlink_outside_root(
    monkeypatch,
    tmp_path: Path,
):
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    export_root = tmp_path / "exports"
    outside = tmp_path / "outside"
    export_root.mkdir()
    outside.mkdir()
    (export_root / "baseline").symlink_to(outside, target_is_directory=True)
    called = False

    def fake_export(_problem, destination, **_kwargs):
        nonlocal called
        called = True
        return Path(destination)

    monkeypatch.setattr(PinchProblem, "export_excel", fake_export)

    exported = workspace.cases(["baseline"]).export_excel(export_root)

    assert exported.results == {}
    assert list(exported.errors) == ["baseline"]
    assert "outside the batch export destination" in str(exported.errors["baseline"])
    assert called is False
