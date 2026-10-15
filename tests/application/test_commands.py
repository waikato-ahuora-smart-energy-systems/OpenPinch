"""JSON-in/JSON-out engine commands shared by the browser and backend engines."""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

from OpenPinch.application._problem.accessors.design import (
    HeatExchangerNetworkDesignView,
    _DesignAccessor,
)
from OpenPinch.application.commands import run_command, run_command_json
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
    "command", ["target._execute", "target.missing", "load.anything", "target"]
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


def test_closed_session_cannot_run_commands():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    run_command({"command": "close", "session": session})

    with pytest.raises(ValueError, match="Unknown session"):
        run_command(
            {"command": "target.direct_heat_integration", "session": session}
        )
