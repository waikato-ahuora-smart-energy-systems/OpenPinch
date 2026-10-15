"""JSON-in/JSON-out engine commands shared by the browser and backend engines."""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

from OpenPinch.application._problem.accessors.design import (
    HeatExchangerNetworkDesignView,
    _DesignAccessor,
)
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.application.commands import run_command, run_command_json
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.workspace import ValidationReport
from OpenPinch.tutorials import sample_cases


def test_target_command_returns_problem_results_as_json():
    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )

    assert response["results"]["targets"]


def test_target_command_passes_arguments_to_the_method():
    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {"options": {"DIRECT_VERTICAL_GCC_ENABLED": True}},
        }
    )

    provenance = response["results"]["targets"][0]["provenance"]
    settings = json.loads(provenance["effective_settings_json"])
    assert settings["DIRECT_VERTICAL_GCC_ENABLED"] is True


def test_command_returns_the_contract_the_method_returns_as_value():
    response = run_command(
        {
            "command": "target.heat_recovery_dt_min",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {"heat_recovery": 4000.0},
        }
    )

    assert "dt_min" in response["value"]


def test_design_command_returns_the_design_result_as_value(monkeypatch):
    report = ValidationReport(valid=True, issues=[])
    monkeypatch.setattr(
        _DesignAccessor,
        "pinch_design",
        lambda self, **_kwargs: HeatExchangerNetworkDesignView(report),
    )

    response = run_command(
        {
            "command": "design.pinch_design",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )

    assert response["value"] == {"valid": True, "issues": []}


@pytest.mark.parametrize(
    "command",
    [
        "target._execute",
        "target.missing",
        "load.anything",
        "target",
        "target.all_periods._run",
        "target._problem.load",
    ],
)
def test_command_rejects_names_that_are_not_public_engine_methods(command):
    with pytest.raises(ValueError, match="Unknown command"):
        run_command(
            {"command": command, "problem": {"sample_case": "basic_pinch.json"}}
        )


def test_sample_case_must_name_a_packaged_case_not_a_file_path(tmp_path):
    case = tmp_path / "basic_pinch.json"
    case.write_text(
        (files(sample_cases) / "basic_pinch.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Unknown OpenPinch sample case"):
        run_command(
            {
                "command": "target.direct_heat_integration",
                "problem": {"sample_case": str(case)},
            }
        )


def test_problem_data_must_be_a_json_object_not_a_file_path(tmp_path):
    case = tmp_path / "basic_pinch.json"
    case.write_text(
        (files(sample_cases) / "basic_pinch.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSON object"):
        run_command(
            {
                "command": "target.direct_heat_integration",
                "problem": {"data": str(case)},
            }
        )


def test_json_text_entry_point_round_trips_a_command():
    request = {
        "command": "target.direct_heat_integration",
        "problem": {"sample_case": "basic_pinch.json"},
    }

    response = json.loads(run_command_json(json.dumps(request)))

    assert response["results"]["targets"]


def test_target_command_accepts_problem_data_inline():
    data = json.loads(
        (files(sample_cases) / "basic_pinch.json").read_text(encoding="utf-8")
    )

    response = run_command(
        {"command": "target.direct_heat_integration", "problem": {"data": data}}
    )

    assert response["results"]["targets"]


def test_commands_in_one_session_share_the_problem():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    run_command({"command": "target.direct_heat_integration", "session": session})
    response = run_command(
        {"command": "target.indirect_heat_integration", "session": session}
    )

    assert len(response["results"]["targets"]) == 2


def test_all_periods_command_returns_each_period_output():
    response = run_command(
        {
            "command": "target.all_periods.direct_heat_integration",
            "problem": {"sample_case": "crude_preheat_train_multiperiod.json"},
        }
    )

    assert set(response["value"]) == {"turndown", "base", "peak"}
    assert response["value"]["base"]["targets"]


def test_session_runs_multiperiod_design_after_all_period_targeting(monkeypatch):
    class Synthesised(Exception):
        pass

    def synthesise(self, **_kwargs):
        raise Synthesised

    monkeypatch.setattr(_DesignAccessor, "heat_exchanger_network", synthesise)
    session = run_command(
        {
            "command": "open",
            "problem": {"sample_case": "crude_preheat_train_multiperiod.json"},
        }
    )["session"]
    run_command(
        {"command": "target.all_periods.direct_heat_integration", "session": session}
    )

    with pytest.raises(Synthesised):
        run_command(
            {"command": "design.multiperiod_heat_exchanger_network", "session": session}
        )


def _optimised_case() -> PinchProblem:
    data = json.loads(
        (files(sample_cases) / "chocolate_factory.json").read_text(encoding="utf-8")
    )
    return PinchProblem(data)


def test_command_returning_a_case_returns_its_problem_data(monkeypatch):
    case = _optimised_case()
    monkeypatch.setattr(
        _TargetAccessor, "utility_placement", lambda self, **_kwargs: case
    )

    response = run_command(
        {
            "command": "target.utility_placement",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )

    assert response["value"] == {"data": case.to_problem_json()}


def test_case_returned_in_a_session_opens_as_a_new_session(monkeypatch):
    monkeypatch.setattr(
        _TargetAccessor, "utility_placement", lambda self, **_kwargs: _optimised_case()
    )
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    value = run_command({"command": "target.utility_placement", "session": session})[
        "value"
    ]
    response = run_command(
        {"command": "target.direct_heat_integration", "session": value["session"]}
    )

    assert value["session"] != session
    assert response["results"]["targets"]


@pytest.fixture(scope="module")
def heat_pump_session():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    response = run_command(
        {
            "command": "target.carnot_heat_pump",
            "session": session,
            "arguments": {
                "load_fraction": 0.25,
                "condensers": 1,
                "evaporators": 1,
                "maximum_restarts": 1,
            },
        }
    )
    return session, response["value"]["ref"]


def test_target_returned_in_a_session_can_be_passed_as_a_base_target(
    heat_pump_session,
):
    session, heat_pump = heat_pump_session

    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "session": session,
            "arguments": {"base_target": {"$ref": heat_pump}},
        }
    )

    assert response["value"]["ref"] != heat_pump


def test_target_reference_needs_the_session_that_returned_it(heat_pump_session):
    _session, heat_pump = heat_pump_session

    with pytest.raises(ValueError, match="Unknown target reference"):
        run_command(
            {
                "command": "target.direct_heat_integration",
                "problem": {"sample_case": "basic_pinch.json"},
                "arguments": {"base_target": {"$ref": heat_pump}},
            }
        )


def test_closed_session_cannot_run_commands():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    run_command({"command": "close", "session": session})

    with pytest.raises(ValueError, match="Unknown session"):
        run_command(
            {"command": "target.direct_heat_integration", "session": session}
        )
