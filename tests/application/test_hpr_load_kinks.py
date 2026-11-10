"""Application orchestration for HPR load kinks."""

from __future__ import annotations

import json
import pickle
from copy import deepcopy
from pathlib import Path

import pytest

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.contracts.hpr_load_kinks import HprLoadKinkKind, HprLoadKinksResult

_CASES = Path(__file__).parents[1] / "fixtures" / "hpr_load_kinks"


def _case(name: str) -> PinchProblem:
    payload = json.loads((_CASES / f"{name}_openpinch.json").read_text())
    return PinchProblem(payload)


def _multiperiod_payload() -> dict:
    return {
        "streams": [
            {
                "zone": "Site/Process",
                "name": "Hot",
                "t_supply": {"values": [200.0, 220.0], "unit": "degC"},
                "t_target": {"values": [100.0, 120.0], "unit": "degC"},
                "heat_flow": {"values": [100.0, 200.0], "unit": "kW"},
                "dt_cont": 5.0,
                "htc": 1.0,
            },
            {
                "zone": "Site/Process",
                "name": "Low cold",
                "t_supply": {"values": [50.0, 60.0], "unit": "degC"},
                "t_target": {"values": [150.0, 170.0], "unit": "degC"},
                "heat_flow": {"values": [100.0, 200.0], "unit": "kW"},
                "dt_cont": 5.0,
                "htc": 1.0,
            },
            {
                "zone": "Site/Process",
                "name": "High cold",
                "t_supply": 230.0,
                "t_target": 240.0,
                "heat_flow": 50.0,
                "dt_cont": 5.0,
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Process", "type": "Process Zone"}],
        },
        "options": {"PROBLEM_PERIOD_IDS": ["0", "peak"]},
    }


def _no_sink_payload() -> dict:
    payload = _multiperiod_payload()
    payload["streams"] = payload["streams"][:1]
    return payload


def _logical(result: HprLoadKinksResult) -> list:
    return [round(fraction, 3) for fraction in result.logical_load_fractions]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("chocolate", [0.765, 1.0]),
        ("ethylbenzene", [0.227, 0.672, 1.0]),
        ("brewery", [0.725, 1.0]),
    ],
)
def test_heat_pump_loads_match_the_published_case_kinks(case, expected) -> None:
    result = _case(case).target.hpr_load_kinks()

    assert isinstance(result, HprLoadKinksResult)
    assert result.scope == "Site"
    assert result.is_refrigeration is False
    assert _logical(result) == expected
    assert all(
        vertex.kind is HprLoadKinkKind.STEP
        for vertex in result.vertices
        if vertex.load_fraction in result.logical_load_fractions
    )


def test_brewery_flattening_vertex_is_not_a_logical_load() -> None:
    result = _case("brewery").target.hpr_load_kinks()

    flattening = [
        vertex for vertex in result.vertices if vertex.kind is HprLoadKinkKind.FLATTENS
    ]
    assert [round(vertex.load_fraction, 3) for vertex in flattening] == [0.761]
    assert 0.761 not in _logical(result)
    step = next(v for v in result.vertices if v.kind is HprLoadKinkKind.STEP)
    assert step.duty.value == pytest.approx(728.24, abs=0.01)
    assert step.duty.unit == "kW"
    assert step.temperature.value == pytest.approx(67.85, abs=0.01)
    assert step.temperature.unit == "degC"
    assert step.slope_before.unit == "delta_degC"


def test_output_unit_overrides_apply_to_every_quantity() -> None:
    payload = json.loads((_CASES / "chocolate_openpinch.json").read_text())
    payload["options"].update(
        OUTPUT_UNIT_HEAT_FLOW="MW", OUTPUT_UNIT_TEMPERATURE="degF"
    )

    result = PinchProblem(payload).target.hpr_load_kinks()

    step = next(v for v in result.vertices if v.kind is HprLoadKinkKind.STEP)
    assert result.maximum_duty.unit == "MW"
    assert step.duty.unit == "MW"
    assert step.duty.value == pytest.approx(0.340, abs=1e-3)
    assert step.temperature.unit == "degF"
    assert step.temperature.value == pytest.approx(140.0, abs=1e-6)
    assert step.slope_before.unit == "delta_degF"


def test_refrigeration_walks_the_source_and_spans_the_utility_targets() -> None:
    problem = _case("chocolate")
    direct = problem.target.direct_heat_integration()

    heat_pump = problem.target.hpr_load_kinks()
    refrigeration = problem.target.hpr_load_kinks(is_refrigeration=True)

    assert heat_pump.maximum_duty.value == pytest.approx(
        float(direct.hot_utility_target), rel=1e-9
    )
    assert refrigeration.maximum_duty.value == pytest.approx(
        float(direct.cold_utility_target), rel=1e-9
    )
    assert refrigeration.is_refrigeration is True
    temperatures = [vertex.temperature.value for vertex in refrigeration.vertices]
    assert temperatures == sorted(temperatures, reverse=True)
    assert [round(f, 3) for f in refrigeration.logical_load_fractions] == [0.999, 1.0]


@pytest.mark.parametrize(
    "steepening", [1.0, 0.5, float("nan"), float("inf"), True, "2"]
)
def test_steepening_must_be_a_finite_number_above_one(steepening) -> None:
    with pytest.raises((TypeError, ValueError), match="steepening"):
        _case("chocolate").target.hpr_load_kinks(steepening=steepening)


def test_is_refrigeration_must_be_a_bool() -> None:
    with pytest.raises(TypeError, match="is_refrigeration"):
        _case("chocolate").target.hpr_load_kinks(is_refrigeration="yes")


def test_profile_without_duty_names_the_scope_and_period() -> None:
    problem = PinchProblem(_no_sink_payload(), project_name="Site")

    with pytest.raises(ValueError, match="no heat pump sink.*'Site/Process'.*'peak'"):
        problem.target.hpr_load_kinks(zone="Site/Process", period_id="peak")


def test_provenance_records_method_scope_period_and_settings() -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")

    result = problem.target.hpr_load_kinks(
        zone="Site/Process", period_id="peak", steepening=3.0
    )

    assert result.scope == "Site/Process"
    assert result.period_id == "peak"
    assert result.steepening == 3.0
    assert result.provenance.method_id == "target.hpr_load_kinks"
    assert result.provenance.zone_address == "Site/Process"
    assert result.provenance.period_ids == ("peak",)
    settings = json.loads(result.provenance.effective_settings_json)
    assert settings["steepening"] == 3.0
    assert settings["is_refrigeration"] is False


@pytest.mark.parametrize("zone", ["Region", "Region/Community"])
def test_community_and_region_scopes_are_rejected_with_guidance(zone) -> None:
    payload = _multiperiod_payload()
    payload["zone_tree"] = {
        "name": "Region",
        "type": "Region",
        "children": [
            {
                "name": "Community",
                "type": "Community",
                "children": [{"name": "Site", "type": "Site"}],
            }
        ],
    }
    for stream in payload["streams"]:
        stream["zone"] = "Region/Community/Site"
    problem = PinchProblem(payload, project_name="Region")

    with pytest.raises(ValueError, match="HPR load kinks require a Site"):
        problem.target.hpr_load_kinks(zone=zone)


def test_all_periods_match_selected_calls_in_order_and_in_parallel() -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")

    sequential = problem.target.all_periods.hpr_load_kinks(
        zone="Site/Process", steepening=3.0
    )
    parallel = problem.target.all_periods.hpr_load_kinks(
        zone="Site/Process", steepening=3.0, workers=2
    )

    assert list(sequential) == ["0", "peak"]
    assert [result.period_id for result in sequential.values()] == ["0", "peak"]
    assert sequential["peak"] == problem.target.hpr_load_kinks(
        zone="Site/Process", period_id="peak", steepening=3.0
    )
    assert parallel == sequential
    with pytest.raises(ValueError, match="positive integer"):
        problem.target.all_periods.hpr_load_kinks(workers=0)


def test_selected_and_all_period_calls_are_non_mutating() -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")
    problem.target.direct_heat_integration(period_id="peak")
    problem.target.all_periods.direct_heat_integration()
    before = {
        "input": problem.to_problem_json(),
        "results": problem.results.model_dump(mode="json"),
        "period_results": {
            key: value.model_dump(mode="json")
            for key, value in problem.period_results.items()
        },
        "run_spec": deepcopy(problem._last_target_run_spec),
        "targets": pickle.dumps(problem._master_zone.targets),
        "target_ids": {
            key: id(value) for key, value in problem._master_zone.targets.items()
        },
    }

    problem.target.hpr_load_kinks(period_id="0")
    problem.target.hpr_load_kinks(period_id="peak", steepening=3.0)
    problem.target.all_periods.hpr_load_kinks(workers=2)

    assert problem.to_problem_json() == before["input"]
    assert problem.results.model_dump(mode="json") == before["results"]
    assert {
        key: value.model_dump(mode="json")
        for key, value in problem.period_results.items()
    } == before["period_results"]
    assert problem._last_target_run_spec == before["run_spec"]
    assert pickle.dumps(problem._master_zone.targets) == before["targets"]
    assert {
        key: id(value) for key, value in problem._master_zone.targets.items()
    } == before["target_ids"]


def test_workspace_active_case_and_batch_failure_isolation() -> None:
    workspace = PinchWorkspace(_multiperiod_payload(), project_name="Site")
    workspace.load(_no_sink_payload(), case_name="no-sink", activate=False)

    active = workspace.target.hpr_load_kinks(zone="Site/Process", period_id="0")
    batch = workspace.cases().target.hpr_load_kinks(zone="Site/Process", period_id="0")
    batch_periods = workspace.cases().target.all_periods.hpr_load_kinks(
        zone="Site/Process", steepening=3.0
    )

    assert active.period_id == "0"
    assert batch.results[workspace.baseline_name] == active
    assert list(batch.errors) == ["no-sink"]
    assert list(batch_periods.results[workspace.baseline_name]) == ["0", "peak"]
    assert list(batch_periods.errors) == ["no-sink"]
