"""Process components survive JSON export, bundles and workspace copies."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from OpenPinch.application._problem.targeting.provenance import input_fingerprint
from OpenPinch.application.commands import run_command
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.contracts.input import TargetInput

pytest.importorskip("CoolProp")

SAMPLE_CASE = "process_mvr.json"
SOURCE = "Evaporator vapour"
# Replaying the same call on the same input is the same arithmetic.
TARGET_REL_TOL = 1e-9


def _vapour(zone: str, *, heat_flow: float = 100.0) -> dict:
    return {
        "zone": zone,
        "name": SOURCE,
        "t_supply": 120.0,
        "t_target": 80.0,
        "p_supply": 101.325,
        "heat_flow": heat_flow,
        "dt_cont": 0.0,
        "htc": 1.0,
        "fluid_name": "Air",
        "fluid_phase": "gas",
    }


def _product(zone: str) -> dict:
    return {
        "zone": zone,
        "name": f"Product {zone}",
        "t_supply": 30.0,
        "t_target": 100.0,
        "heat_flow": 120.0,
        "dt_cont": 0.0,
        "htc": 1.0,
    }


def _two_train_payload() -> dict:
    return {
        "streams": [
            _vapour("Site/Train A"),
            _product("Site/Train A"),
            _vapour("Site/Train B", heat_flow=60.0),
            _product("Site/Train B"),
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [
                {"name": "Train A", "type": "Process Zone"},
                {"name": "Train B", "type": "Process Zone"},
            ],
        },
        "options": {},
    }


def _with_mvr(**kwargs) -> PinchProblem:
    problem = PinchProblem(SAMPLE_CASE)
    problem.components.add_process_mvr(
        SOURCE, **{"mvr_stage_t_lift": 10.0, "liquid_injection": False, **kwargs}
    )
    return problem


def _direct_targets(problem: PinchProblem) -> tuple[float, float, float]:
    target = problem.target.direct_heat_integration()
    return (
        float(target.hot_utility_target),
        float(target.cold_utility_target),
        # A case without components leaves this unset.
        float(target.process_component_work_target or 0.0),
    )


def _assert_same_targets(copied: PinchProblem, original: PinchProblem) -> None:
    assert _direct_targets(copied) == pytest.approx(
        _direct_targets(original), rel=TARGET_REL_TOL
    )


def _reloaded(problem: PinchProblem) -> PinchProblem:
    """Pass the export through real JSON text, as a saved file does."""
    return PinchProblem(json.loads(json.dumps(problem.to_problem_json())))


def test_json_round_trip_keeps_the_component_and_its_targets():
    original = _with_mvr(mvr_id="evaporator", n_stages=2)

    copied = _reloaded(original)

    assert list(copied.process_components) == ["evaporator"]
    component = copied.process_components["evaporator"]
    assert component.problem is copied
    assert component.active is True
    assert component.settings == original.process_components["evaporator"].settings
    _assert_same_targets(copied, original)
    # Without the MVR the same case targets different utilities.
    assert _direct_targets(PinchProblem(SAMPLE_CASE))[:2] != pytest.approx(
        _direct_targets(original)[:2], rel=TARGET_REL_TOL
    )


def test_export_records_the_call_in_json_values():
    record = _with_mvr(
        mvr_id="evaporator",
        compressor_efficiency=0.73,
    ).to_problem_json()["process_components"][0]

    assert record == {
        "type": "process_mvr",
        "id": "evaporator",
        "source_streams": record["source_streams"],
        "n_stages": 1,
        "liquid_injection": False,
        "mvr_stage_t_lift": 10.0,
        "mvr_stage_pressure_ratio": None,
        "compressor_efficiency": 0.73,
        "motor_efficiency": None,
        "active": True,
    }
    assert [source["name"] for source in record["source_streams"]] == [SOURCE]


def test_round_trip_keeps_pressure_ratio_and_stream_object_selector():
    original = PinchProblem(SAMPLE_CASE)
    source = next(s for s in original.hot_streams if s.name == SOURCE)
    original.components.add_process_mvr(
        source, mvr_stage_pressure_ratio=1.5, liquid_injection=True
    )

    copied = _reloaded(original)

    assert copied.process_components["mvr_1"].settings == (
        original.process_components["mvr_1"].settings
    )
    _assert_same_targets(copied, original)


def test_efficiencies_left_unset_still_follow_the_options_after_a_round_trip():
    copied = _reloaded(_with_mvr(motor_efficiency=0.9))

    copied.update_options({"PROCESS_MVR_ETA_COMP": 0.6, "PROCESS_MVR_ETA_MOTOR": 0.5})

    settings = copied.process_components["mvr_1"].settings
    assert settings.eta_mvr_comp == pytest.approx(0.6)
    assert settings.eta_motor == pytest.approx(0.9)


def test_round_trip_picks_the_same_stream_when_display_names_repeat():
    original = PinchProblem(_two_train_payload())
    original.components.add_process_mvr(
        "Site/Train B.Evaporator vapour", mvr_stage_t_lift=10.0, mvr_id="b_only"
    )
    original.components.add_process_mvr(
        "Site/Train A.Evaporator vapour", mvr_stage_t_lift=5.0, mvr_id="a_only"
    )

    copied = _reloaded(original)

    assert list(copied.process_components) == ["b_only", "a_only"]
    for key, component in original.process_components.items():
        assert copied.process_components[key].affected_zone_paths == (
            component.affected_zone_paths
        )
    _assert_same_targets(copied, original)


def test_round_trip_keeps_every_stream_a_name_selector_matched():
    original = PinchProblem(_two_train_payload())
    original.components.add_process_mvr(SOURCE, mvr_stage_t_lift=10.0)

    copied = _reloaded(original)

    assert len(copied.process_components["mvr_1"].stream_records) == 2
    _assert_same_targets(copied, original)


def test_round_trip_finds_duplicate_names_in_one_zone():
    payload = _two_train_payload()
    payload["streams"].append(_vapour("Site/Train A", heat_flow=40.0))
    original = PinchProblem(payload)
    original.components.add_process_mvr("Evaporator vapour_1", mvr_stage_t_lift=10.0)

    copied = _reloaded(original)

    record = copied.process_components["mvr_1"].stream_records[0]
    assert record.original_stream.name == "Evaporator vapour_1"
    _assert_same_targets(copied, original)


def test_round_trip_keeps_a_later_component_on_a_released_source():
    original = _with_mvr(mvr_id="first")
    original.process_components["first"].deactivate()
    original.components.add_process_mvr(
        SOURCE, mvr_id="second", mvr_stage_t_lift=5.0, liquid_injection=False
    )

    copied = _reloaded(original)

    assert copied.process_components["first"].active is False
    assert copied.process_components["second"].active is True
    _assert_same_targets(copied, original)


def test_round_trip_keeps_a_component_fed_by_another_component():
    original = _with_mvr(mvr_id="first")
    replacement = original.process_components["first"].replacement_streams[0]
    original.components.add_process_mvr(
        replacement.name, mvr_id="second", mvr_stage_t_lift=5.0, liquid_injection=False
    )

    copied = _reloaded(original)

    assert list(copied.process_components) == ["first", "second"]
    _assert_same_targets(copied, original)


def test_export_of_a_reloaded_case_is_stable():
    first = _reloaded(_with_mvr())
    second = _reloaded(first)

    assert second.to_problem_json() == first.to_problem_json()
    assert input_fingerprint(second) == input_fingerprint(first)


def test_multiperiod_round_trip_matches_each_period():
    payload = _two_train_payload()
    payload["options"] = {"PROBLEM_PERIOD_IDS": ["low", "high"]}
    payload["streams"][0]["heat_flow"] = {"values": [100.0, 130.0], "unit": "kW"}
    original = PinchProblem(payload)
    original.components.add_process_mvr(
        "Site/Train A.Evaporator vapour", mvr_stage_t_lift=10.0
    )

    copied = _reloaded(original)

    for period_id in ("low", "high"):
        work = [
            problem.target.direct_heat_integration(
                period_id=period_id
            ).process_component_work_target
            for problem in (copied, original)
        ]
        assert work[0] == pytest.approx(work[1], rel=TARGET_REL_TOL)
        assert work[0] > 0.0


def test_saving_a_loaded_file_keeps_the_component(tmp_path):
    original = _with_mvr()
    path = tmp_path / "case.json"
    path.write_text(json.dumps(original.to_problem_json()), encoding="utf-8")

    copied = PinchProblem(path)

    assert list(copied.process_components) == ["mvr_1"]
    _assert_same_targets(copied, original)


def test_a_target_input_source_keeps_the_component():
    original = _with_mvr()

    copied = PinchProblem(TargetInput.model_validate(original.to_problem_json()))

    assert list(copied.process_components) == ["mvr_1"]
    _assert_same_targets(copied, original)


@pytest.mark.parametrize("read", [lambda p: p.problem_data, lambda p: p.validate()])
def test_input_snapshots_can_rebuild_the_case_with_its_components(read):
    original = _with_mvr()

    copied = PinchProblem(read(original))

    assert list(copied.process_components) == ["mvr_1"]
    _assert_same_targets(copied, original)


def test_option_change_after_a_round_trip_keeps_one_component_handle():
    copied = _reloaded(_with_mvr())
    handle = copied.process_components["mvr_1"]

    copied.update_options({"ENV_TEMPERATURE": 30.0})

    assert list(copied.process_components) == ["mvr_1"]
    assert copied.process_components["mvr_1"] is handle
    assert len(copied.to_problem_json()["process_components"]) == 1


def test_bundle_round_trip_keeps_the_component(tmp_path):
    original = _with_mvr()
    workspace = PinchWorkspace(project_name="Site")
    workspace.load(original, case_name="baseline")
    path = workspace.save_bundle(tmp_path / "study.json")

    reopened = PinchWorkspace.load_bundle(path)

    case = reopened.case("baseline")
    assert list(case.process_components) == ["mvr_1"]
    _assert_same_targets(case, original)


def test_a_deleted_source_stream_fails_the_load_and_keeps_the_prior_case():
    exported = _with_mvr().to_problem_json()
    exported["process_components"][0]["source_streams"][0]["name"] = "Missing"
    problem = PinchProblem(SAMPLE_CASE)
    before = problem.to_problem_json()

    with pytest.raises(ValueError, match="Missing"):
        problem.load(exported)

    assert problem.process_components == {}
    assert problem.to_problem_json() == before


def test_utility_placement_result_keeps_the_component():
    original = _with_mvr()

    optimized = original.target.utility_placement(
        isothermal=2,
        options={"iteration_limit": 2, "evaluation_limit": 20, "seed": 20260715},
    )

    assert list(optimized.process_components) == ["mvr_1"]


def test_bundle_keeps_a_component_added_to_a_live_case():
    workspace = PinchWorkspace(project_name="Site")
    workspace.load(PinchProblem(SAMPLE_CASE), case_name="baseline")
    workspace.case("baseline").components.add_process_mvr(
        SOURCE, mvr_stage_t_lift=10.0, liquid_injection=False
    )

    reopened = PinchWorkspace.from_bundle(workspace.to_bundle())

    assert list(reopened.case("baseline").process_components) == ["mvr_1"]
    _assert_same_targets(reopened.case("baseline"), workspace.case("baseline"))


def test_a_case_without_components_exports_as_before():
    problem = PinchProblem(SAMPLE_CASE)

    assert "process_components" not in problem.to_problem_json()
    assert "process_components" not in problem.validate().model_dump()
    assert "process_components" not in problem.problem_data
    assert problem.process_components == {}


def test_session_export_and_reopen_keep_the_component():
    session = run_command({"command": "open", "problem": {"sample_case": SAMPLE_CASE}})[
        "session"
    ]
    run_command(
        {
            "command": "components.add_process_mvr",
            "session": session,
            "arguments": {"source_streams": SOURCE, "mvr_stage_t_lift": 10.0},
        }
    )
    data = run_command({"command": "to_problem_json", "session": session})["value"]

    reopened = run_command({"command": "open", "problem": {"data": data}})["session"]

    inventory = run_command({"command": "components.inventory", "session": reopened})
    assert list(inventory["value"]) == ["mvr_1"]


def test_a_missing_source_stream_is_a_validation_report_error():
    exported = _with_mvr().to_problem_json()
    exported["process_components"][0]["source_streams"][0]["zone"] = "Nowhere"

    report = PinchProblem.source_validation_report(exported)

    assert report.valid is False


def test_process_component_records_reject_unknown_fields():
    exported = _with_mvr().to_problem_json()
    exported["process_components"][0]["unknown"] = 1

    with pytest.raises(ValidationError):
        TargetInput.model_validate(exported)


def test_export_after_a_project_rename_still_reloads():
    original = _with_mvr()
    original.project_name = "Renamed"

    copied = PinchProblem(original.to_problem_json(), project_name="Renamed")

    assert list(copied.process_components) == ["mvr_1"]
    _assert_same_targets(copied, original)


def test_scenario_keeps_the_base_case_component():
    original = _with_mvr()
    workspace = PinchWorkspace(project_name="Site")
    workspace.load(original, case_name="baseline")

    scenario = workspace.scenario("copy", base="baseline")

    assert list(scenario.process_components) == ["mvr_1"]
    assert scenario.process_components["mvr_1"].problem is scenario
    _assert_same_targets(scenario, original)


def test_scenario_option_change_re_solves_the_copied_component_only():
    workspace = PinchWorkspace(project_name="Site")
    workspace.load(_with_mvr(), case_name="baseline")

    scenario = workspace.scenario(
        "efficient", base="baseline", options={"PROCESS_MVR_ETA_COMP": 0.5}
    )

    base_settings = workspace.case("baseline").process_components["mvr_1"].settings
    scenario_settings = scenario.process_components["mvr_1"].settings
    assert scenario_settings.eta_mvr_comp == pytest.approx(0.5)
    assert base_settings.eta_mvr_comp != pytest.approx(0.5)


def test_a_file_written_before_this_field_loads_without_components():
    exported = PinchProblem(SAMPLE_CASE).to_problem_json()

    reloaded = PinchProblem(exported)

    assert reloaded.process_components == {}
    assert reloaded.to_problem_json() == exported


def test_round_trip_keeps_an_inactive_component_inactive():
    original = _with_mvr()
    original.process_components["mvr_1"].deactivate()

    copied = _reloaded(original)

    component = copied.process_components["mvr_1"]
    assert component.active is False
    assert all(stream.is_active for stream in component.original_streams)
    assert not any(stream.is_active for stream in component.replacement_streams)
    _assert_same_targets(copied, original)
    assert _direct_targets(copied)[2] == pytest.approx(0.0, abs=1e-12)
