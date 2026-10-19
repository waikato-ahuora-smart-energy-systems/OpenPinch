"""JSON-in/JSON-out engine commands shared by the browser and backend engines."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import warnings
from importlib.resources import files

import jsonschema
import pytest

from OpenPinch.adapters.io.problem_sources import load_packaged_sample_case
from OpenPinch.application._problem.accessors.design import (
    HeatExchangerNetworkDesignView,
    _DesignAccessor,
)
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.application.commands import run_command, run_command_json
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.hpr import (
    HPRFailureSummary,
    HPRSearchBudget,
    HPRTargetingError,
)
from OpenPinch.contracts.hpr_performance_map import HprPerformanceMapRequest
from OpenPinch.contracts.workspace import ValidationReport
from OpenPinch.domain import progress
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.tutorials import sample_cases
from tests.support.paths import REPOSITORY_ROOT


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
    # The stand-in result has no networks to view; the view is tested below.
    monkeypatch.setattr(HeatExchangerNetworkDesignView, "views", lambda self: (report,))

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


@pytest.mark.parametrize(
    "request_",
    [
        {"command": "target.direct_heat_integration", "problem": "basic_pinch.json"},
        {"command": "validate", "problem": "basic_pinch.json"},
        {"command": "open", "problem": "basic_pinch.json"},
        {"command": "open", "workspace": {"problem": "basic_pinch.json"}},
        {"command": "validate", "problem": {"name": "basic_pinch.json"}},
    ],
    ids=["target", "validate", "open", "workspace", "no_source"],
)
def test_problem_must_be_a_data_or_sample_case_object(request_):
    with pytest.raises(ValueError, match=r'\{"data": .*\} or \{"sample_case": '):
        run_command(request_)


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


def test_residual_target_from_a_base_target_is_returned_by_value(
    heat_pump_session,
):
    # The residual target is solved on a detached residual case, so it is not
    # in the session problem's results; the value is the only place it shows.
    session, heat_pump = heat_pump_session

    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "session": session,
            "arguments": {"base_target": {"$ref": heat_pump}},
        }
    )

    assert response["value"]["target"]["integration_type"] == "Residual"


def test_contract_arguments_are_validated_from_json(monkeypatch):
    def echo(self, *, request: HprPerformanceMapRequest) -> HprPerformanceMapRequest:
        return request

    monkeypatch.setattr(_TargetAccessor, "hpr_performance_map", echo)

    response = run_command(
        {
            "command": "target.hpr_performance_map",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {
                "request": {
                    "map_id": "map",
                    "source_temperatures": [20.0, 5.0],
                    "sink_temperatures": [70.0],
                    "load_fractions": [1.0],
                }
            },
        }
    )

    assert response["value"]["source_temperatures"] == [5.0, 20.0]


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

    response = run_command(
        {"command": "target.direct_heat_integration", "session": session}
    )

    assert response["value"] is None
    assert response["error"]["type"] == "unknown_session"
    assert response["error"]["details"] == {"session": session}


def test_an_unknown_session_argument_returns_an_unknown_session_error():
    response = run_command(
        {
            "command": "with_utilities_from",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {"other_problem": {"$session": "missing"}},
        }
    )

    assert response["error"]["type"] == "unknown_session"
    assert response["error"]["details"] == {"session": "missing"}


def test_problem_source_names_the_project():
    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "problem": {"sample_case": "basic_pinch.json", "project_name": "Mill"},
        }
    )

    assert response["results"]["targets"][0]["scope"] == "Mill"


def test_validate_returns_the_normalised_input_and_the_report():
    value = run_command(
        {"command": "validate", "problem": {"sample_case": "basic_pinch.json"}}
    )["value"]

    assert value["report"]["valid"] is True
    assert value["input"]["streams"]


def test_validate_reports_invalid_inline_input_instead_of_raising():
    data = json.loads(
        (files(sample_cases) / "basic_pinch.json").read_text(encoding="utf-8")
    )
    data["streams"][0]["t_supply"] = "warm"

    response = run_command({"command": "validate", "problem": {"data": data}})

    assert response["value"]["input"] is None
    report = response["value"]["report"]
    assert report["valid"] is False
    assert report["issues"][0]["path"].startswith("streams[0].t_supply")


def test_validate_reports_an_undefined_unit_instead_of_raising():
    data = load_packaged_sample_case("basic_pinch.json")
    data["streams"][0]["t_supply"] = {"value": 200.0, "unit": "bogus"}

    response = run_command({"command": "validate", "problem": {"data": data}})

    report = response["value"]["report"]
    assert report["valid"] is False
    assert any(
        issue["path"] == "streams[0].t_supply" and "bogus" in issue["message"]
        for issue in report["issues"]
    )


def test_validate_reports_input_that_passes_the_schema_but_cannot_be_built():
    # A zone type the schema accepts as text but problem construction rejects.
    data = json.loads(
        (files(sample_cases) / "basic_pinch.json").read_text(encoding="utf-8")
    )
    data["zone_tree"] = {"name": "Site", "type": "Bogus"}

    response = run_command({"command": "validate", "problem": {"data": data}})

    report = response["value"]["report"]
    assert response["value"]["input"] is None
    assert report["valid"] is False
    assert "Zone name and type" in report["issues"][0]["message"]


def test_options_returns_fields_effective_values_and_options_set():
    data = load_packaged_sample_case("basic_pinch.json")
    data["options"] = {"THERMAL_DT_CONT": 5.0}

    value = run_command({"command": "options", "problem": {"data": data}})["value"]

    fields = {field["name"]: field for field in value["fields"]}
    assert fields["THERMAL_DT_CONT"]["unit"] == "delta_degC"
    assert value["values"]["THERMAL_DT_CONT"] == 5.0
    assert value["values"]["THERMAL_HTC"] == 1.0
    # Set to its default, yet set: the editor tells it from an unset option.
    assert value["set"] == {"THERMAL_DT_CONT": 5.0}


_STREAMS_CSV = """Zone,Stream,Ts,Tt,Q,dTcont,HTC
,,degC,degC,kW,degC,kW/m^2/degC
Plant,H1,180.0,60.0,240.0,4.0,2.0
Plant,C1,20.0,135.0,230.0,4.0,2.0
"""
_UTILITIES_CSV = """Utility,Type,Ts,Tt,dTcont,Price,HTC,Q
,,degC,degC,degC,$/MWh,kW/m^2/degC,kW
HP Steam,Hot,250.0,249.0,5.0,40.0,1.0,
Cooling Water,Cold,15.0,25.0,5.0,5.0,1.0,
"""


def _file(name: str, text: str, role: str) -> dict[str, str]:
    content = base64.b64encode(text.encode()).decode("ascii")
    return {"name": name, "role": role, "content_base64": content}


def test_import_reads_csv_files_by_their_named_roles():
    # Neither name says which table it is: the roles do.
    value = run_command(
        {
            "command": "import",
            "arguments": {
                "files": [
                    _file("a.csv", _UTILITIES_CSV, "utilities"),
                    _file("b.csv", _STREAMS_CSV, "streams"),
                ]
            },
        }
    )["value"]

    assert value["source_kind"] == "csv"
    assert [stream["name"] for stream in value["data"]["streams"]] == ["H1", "C1"]
    assert [utility["name"] for utility in value["data"]["utilities"]] == [
        "HP Steam",
        "Cooling Water",
    ]


def test_import_keeps_role_files_apart_when_their_names_match():
    # Files picked from two folders can share a name.
    value = run_command(
        {
            "command": "import",
            "arguments": {
                "files": [
                    _file("table.csv", _STREAMS_CSV, "streams"),
                    _file("table.csv", _UTILITIES_CSV, "utilities"),
                ]
            },
        }
    )["value"]

    assert [stream["name"] for stream in value["data"]["streams"]] == ["H1", "C1"]


def test_import_reads_a_problem_file_by_its_suffix():
    workbook = REPOSITORY_ROOT / "examples" / "OpenPinchWkbs" / "Ahmad (example 1).xlsb"
    content = base64.b64encode(workbook.read_bytes()).decode("ascii")

    value = run_command(
        {
            "command": "import",
            "arguments": {
                "files": [
                    {
                        "name": workbook.name,
                        "role": "problem",
                        "content_base64": content,
                    }
                ]
            },
        }
    )["value"]

    assert value["source_kind"] == "excel"
    assert value["project_name"] == "Ahmad (example 1)"
    assert value["data"]["streams"]


@pytest.mark.parametrize(
    "roles",
    [
        ["streams"],
        ["streams", "streams"],
        ["problem", "streams"],
        ["problem", "problem"],
        ["stream"],
        [],
    ],
)
def test_import_needs_one_problem_file_or_a_streams_and_a_utilities_file(roles):
    files = [
        _file(f"{index}.csv", _STREAMS_CSV, role) for index, role in enumerate(roles)
    ]

    with pytest.raises(ValueError, match="one 'problem' file, or a 'streams'"):
        run_command({"command": "import", "arguments": {"files": files}})


def test_to_problem_json_returns_the_problem_input():
    value = run_command(
        {"command": "to_problem_json", "problem": {"sample_case": "basic_pinch.json"}}
    )["value"]

    assert PinchProblem(value).to_problem_json() == value


def test_streams_returns_prepared_streams_and_utilities_with_segments():
    value = run_command(
        {"command": "streams", "problem": {"sample_case": "basic_pinch.json"}}
    )["value"]

    assert value["hot_streams"]["segment_index"]
    assert set(value) == {
        "hot_streams",
        "cold_streams",
        "hot_utilities",
        "cold_utilities",
    }


def test_load_returns_the_input_prepared_streams_and_zone_tree():
    value = run_command(
        {"command": "load", "problem": {"sample_case": "zonal_site.json"}}
    )["value"]

    assert PinchProblem(value["input"]).to_problem_json() == value["input"]
    assert value["streams"]["hot_utilities"]["segment_index"]
    site = value["zones"]
    assert (site["name"], site["type"], site["address"]) == ("Site", "Site", "Site")
    plant = site["subzones"][0]
    assert plant["address"] == "Site/Chemical Plant[S]"
    assert plant["subzones"][0] == {
        "name": "O1",
        "type": "Unit Operation",
        "address": "Site/Chemical Plant[S]/O1",
        "hot_streams": ["A1"],
        "cold_streams": [],
        "subzones": [],
    }


def test_report_returns_metrics_report_and_summary_rows_for_solved_targets():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    run_command({"command": "target.direct_heat_integration", "session": session})

    value = run_command({"command": "report", "session": session})["value"]

    assert value["metrics"]
    assert value["report"]["project_name"] == "Site"
    assert value["summary"]["data"]


def test_adding_a_process_mvr_in_a_session_returns_the_component():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "process_mvr.json"}}
    )["session"]

    value = run_command(
        {
            "command": "components.add_process_mvr",
            "session": session,
            "arguments": {"source_streams": "Evaporator vapour"},
        }
    )["value"]

    assert value["id"] == "mvr_1"
    assert value["original_streams"] == ["Evaporator vapour"]


def test_process_component_can_be_deactivated_and_reactivated_by_id():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "process_mvr.json"}}
    )["session"]
    run_command(
        {
            "command": "components.add_process_mvr",
            "session": session,
            "arguments": {"source_streams": "Evaporator vapour"},
        }
    )

    off = run_command(
        {
            "command": "components.deactivate",
            "session": session,
            "arguments": {"id": "mvr_1"},
        }
    )["value"]
    on = run_command(
        {
            "command": "components.activate",
            "session": session,
            "arguments": {"id": "mvr_1"},
        }
    )["value"]

    assert (off["active"], on["active"]) == (False, True)


def test_component_inventory_lists_the_session_components_by_id():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "process_mvr.json"}}
    )["session"]
    run_command(
        {
            "command": "components.add_process_mvr",
            "session": session,
            "arguments": {"source_streams": "Evaporator vapour"},
        }
    )

    value = run_command({"command": "components.inventory", "session": session})[
        "value"
    ]

    assert list(value) == ["mvr_1"]


def test_residual_utility_derives_a_case_from_a_session_target(heat_pump_session):
    session, heat_pump = heat_pump_session

    value = run_command(
        {
            "command": "residual_utility",
            "session": session,
            "arguments": {"base_target": {"$ref": heat_pump}},
        }
    )["value"]

    # A residual case keeps the heat pump's residual, not the process streams.
    assert value["data"]["residual_basis"]
    assert value["session"] != session


def test_with_utilities_from_takes_the_donor_problem_from_another_session():
    receiver = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    donor = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    value = run_command(
        {
            "command": "with_utilities_from",
            "session": receiver,
            "arguments": {"other_problem": {"$session": donor}},
        }
    )["value"]

    assert value["data"]["utilities"]


def test_compare_returns_summary_rows_of_two_solved_sessions():
    sessions = []
    for case in ("basic_pinch.json", "basic_pinch.json"):
        session = run_command({"command": "open", "problem": {"sample_case": case}})[
            "session"
        ]
        run_command({"command": "target.direct_heat_integration", "session": session})
        sessions.append(session)

    value = run_command(
        {
            "command": "compare",
            "session": sessions[0],
            "arguments": {"other_problem": {"$session": sessions[1]}},
        }
    )["value"]

    assert value["columns"] and value["data"]


def test_returned_case_carries_its_placement_evidence(monkeypatch):
    case = _optimised_case()
    case._utility_placement_result = ValidationReport(valid=True, issues=[])
    monkeypatch.setattr(
        _TargetAccessor, "utility_placement", lambda self, **_kwargs: case
    )

    value = run_command(
        {
            "command": "target.utility_placement",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )["value"]

    assert value["utility_placement_result"] == {"valid": True, "issues": []}


def test_returned_case_carries_its_own_results_once_solved(monkeypatch):
    case = _optimised_case()
    case.target.direct_heat_integration()
    monkeypatch.setattr(
        _TargetAccessor, "utility_placement", lambda self, **_kwargs: case
    )

    value = run_command(
        {
            "command": "target.utility_placement",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )["value"]

    assert value["results"] == case.results.model_dump(mode="json")


def test_heat_pump_targeting_failure_is_returned_as_an_error(monkeypatch):
    diagnostics = HPRFailureSummary(
        simulation_backend="coolprop",
        cycle="single-stage",
        evaluated_count=0,
        category_counts={},
        representative_failures=(),
        budget=HPRSearchBudget(maximum_iterations=1, maximum_evaluations=1),
        warm_start_evaluated=False,
        warm_start_viable=False,
    )

    def infeasible(self, **_kwargs):
        raise HPRTargetingError("bounded failure", diagnostics=diagnostics)

    monkeypatch.setattr(_TargetAccessor, "carnot_heat_pump", infeasible)

    response = run_command(
        {
            "command": "target.carnot_heat_pump",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )

    assert response["error"]["type"] == "HPRTargetingError"
    assert response["error"]["details"]["status"] == "typed infeasible"


def test_missing_tespy_is_returned_as_an_error(monkeypatch):
    # An engine without the optional extra, such as the browser engine.
    monkeypatch.setattr(
        "OpenPinch.analysis.heat_pumps.performance_maps.targeting.find_spec",
        lambda _name: None,
    )

    response = run_command(
        {
            "command": "target.vapour_compression_heat_pump",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {
                "simulation_backend": "tespy",
                "condensers": 1,
                "evaporators": 1,
            },
        }
    )

    assert response["value"] is None
    assert response["error"]["type"] == "HprTargetCompatibilityError"
    assert response["error"]["details"] == {"code": "dependency_unavailable"}


def test_orc_targeting_failure_is_returned_as_an_error():
    # 200 degC is above the basic case's surplus, so no ORC can condense.
    response = run_command(
        {
            "command": "target.carnot_orc",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {"condensing_temperature": 200.0},
        }
    )

    assert response["value"] is None
    assert response["error"]["type"] == "OrcTargetingError"
    assert "too cold" in response["error"]["message"]
    assert response["error"]["details"] is None


def test_plot_catalog_lists_the_solved_graphs():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    run_command({"command": "target.direct_heat_integration", "session": session})

    value = run_command({"command": "plot_catalog", "session": session})["value"]

    assert value["columns"] and value["data"]


def test_excel_export_returns_the_workbook_as_a_base64_file():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    run_command({"command": "target.direct_heat_integration", "session": session})

    files = run_command({"command": "export.excel", "session": session})["value"][
        "files"
    ]

    (workbook,) = files
    assert workbook["name"].endswith(".xlsx")
    assert base64.b64decode(workbook["content_base64"])[:2] == b"PK"


def test_excel_export_after_an_all_periods_run_returns_the_workbook():
    session = run_command(
        {
            "command": "open",
            "problem": {"sample_case": "crude_preheat_train_multiperiod.json"},
        }
    )["session"]
    run_command(
        {"command": "target.all_periods.direct_heat_integration", "session": session}
    )

    response = run_command({"command": "export.excel", "session": session})

    (workbook,) = response["value"]["files"]
    assert base64.b64decode(workbook["content_base64"])[:2] == b"PK"


def test_plot_export_returns_html_files_for_the_named_plot():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    run_command({"command": "target.direct_heat_integration", "session": session})

    files = run_command(
        {
            "command": "export.plots",
            "session": session,
            "arguments": {"plot": "grand_composite_curve", "gallery": True},
        }
    )["value"]["files"]

    names = [file["name"] for file in files]
    assert "index.html" in names
    assert all(name.endswith(".html") for name in names)


def test_plot_export_can_share_one_plotly_js_file():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    run_command({"command": "target.direct_heat_integration", "session": session})

    files = run_command(
        {
            "command": "export.plots",
            "session": session,
            "arguments": {"gallery": True, "include_plotlyjs": "directory"},
        }
    )["value"]["files"]

    names = [file["name"] for file in files]
    assert names.count("plotly.min.js") == 1


def test_workspace_session_runs_commands_on_a_named_case():
    session = run_command(
        {
            "command": "open",
            "workspace": {"problem": {"sample_case": "basic_pinch.json"}},
        }
    )["session"]

    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "session": session,
            "case": "baseline",
        }
    )

    assert response["results"]["targets"]


def _workspace_session() -> str:
    return run_command(
        {
            "command": "open",
            "workspace": {"problem": {"sample_case": "basic_pinch.json"}},
        }
    )["session"]


def test_workspace_scenario_adds_a_case_that_commands_can_target():
    session = _workspace_session()

    value = run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "tight_dt", "dt_cont_multiplier": 0.75},
        }
    )["value"]
    tight = run_command(
        {
            "command": "target.direct_heat_integration",
            "session": session,
            "case": "tight_dt",
        }
    )["results"]["targets"][0]
    baseline = run_command(
        {
            "command": "target.direct_heat_integration",
            "session": session,
            "case": "baseline",
        }
    )["results"]["targets"][0]

    assert value["cases"] == ["baseline", "tight_dt"]
    assert tight["Qh"]["value"] < baseline["Qh"]["value"]


def test_workspace_use_case_makes_a_case_active_for_case_free_commands():
    session = _workspace_session()
    run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "wide_dt", "dt_cont_multiplier": 2.0},
        }
    )

    value = run_command(
        {
            "command": "workspace.use_case",
            "session": session,
            "arguments": {"name": "wide_dt"},
        }
    )["value"]
    data = run_command({"command": "to_problem_json", "session": session})["value"]

    assert value["active_case"] == "wide_dt"
    assert data["zone_tree"]["dt_cont_multiplier"] == pytest.approx(2.0)


def test_workspace_update_options_changes_one_case_input():
    session = _workspace_session()
    run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "wide_dt"},
        }
    )

    run_command(
        {
            "command": "workspace.update_options",
            "session": session,
            "arguments": {"options": {"THERMAL_DT_CONT": 15}, "case_name": "wide_dt"},
        }
    )
    data = run_command(
        {"command": "to_problem_json", "session": session, "case": "wide_dt"}
    )["value"]

    assert data["options"]["THERMAL_DT_CONT"] == 15


def test_workspace_set_case_input_replaces_a_case_in_the_session():
    session = _workspace_session()
    run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "b"},
        }
    )
    run_command({"command": "target.direct_heat_integration", "session": session})
    data = load_packaged_sample_case("basic_pinch.json")
    data["options"] = {"THERMAL_DT_CONT": 20.0}

    value = run_command(
        {
            "command": "workspace.set_case_input",
            "session": session,
            "arguments": {"case_name": "b", "source": data},
        }
    )["value"]
    baseline = run_command({"command": "to_problem_json", "session": session})

    assert value["cases"] == ["baseline", "b"]
    assert value["report"]["valid"] is True
    assert baseline["results"]["targets"]


def test_workspace_set_case_input_refuses_a_host_path(tmp_path):
    session = _workspace_session()
    path = tmp_path / "case.json"
    path.write_text(json.dumps(load_packaged_sample_case("basic_pinch.json")))

    with pytest.raises(ValueError, match=r'\{"data": \.\.\.\}'):
        run_command(
            {
                "command": "workspace.set_case_input",
                "session": session,
                "arguments": {"case_name": "baseline", "source": str(path)},
            }
        )


def test_workspace_set_case_input_takes_a_data_source():
    session = _workspace_session()
    data = load_packaged_sample_case("basic_pinch.json")
    data["options"] = {"THERMAL_DT_CONT": 20.0}

    value = run_command(
        {
            "command": "workspace.set_case_input",
            "session": session,
            "arguments": {"case_name": "baseline", "source": {"data": data}},
        }
    )["value"]
    stored = run_command({"command": "to_problem_json", "session": session})

    assert value["report"]["valid"] is True
    assert stored["value"]["options"]["THERMAL_DT_CONT"] == 20.0


def test_workspace_sets_a_case_dt_cont_multiplier():
    session = _workspace_session()

    run_command(
        {
            "command": "workspace.set_dt_cont_multiplier",
            "session": session,
            "arguments": {"value": 1.5, "case_name": "baseline"},
        }
    )
    data = run_command({"command": "to_problem_json", "session": session})["value"]

    assert data["zone_tree"]["dt_cont_multiplier"] == pytest.approx(1.5)


def test_workspace_compare_cases_returns_comparison_rows():
    session = _workspace_session()
    run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "tight_dt", "dt_cont_multiplier": 0.75},
        }
    )
    for case in ("baseline", "tight_dt"):
        run_command(
            {
                "command": "target.direct_heat_integration",
                "session": session,
                "case": case,
            }
        )

    value = run_command(
        {
            "command": "workspace.compare_cases",
            "session": session,
            "arguments": {"base_case": "baseline", "other_case": "tight_dt"},
        }
    )["value"]

    assert value["columns"] and value["data"]


def test_workspace_bundle_round_trips_through_json():
    session = _workspace_session()
    run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "tight_dt", "dt_cont_multiplier": 0.75},
        }
    )

    bundle = run_command({"command": "workspace.to_bundle", "session": session})[
        "value"
    ]
    reopened = run_command({"command": "open", "workspace": {"bundle": bundle}})[
        "session"
    ]
    data = run_command(
        {"command": "to_problem_json", "session": reopened, "case": "tight_dt"}
    )["value"]

    assert data["zone_tree"]["dt_cont_multiplier"] == pytest.approx(0.75)


def test_workspace_add_registers_another_session_problem_as_a_case():
    session = _workspace_session()
    other = run_command(
        {"command": "open", "problem": {"sample_case": "chocolate_factory.json"}}
    )["session"]

    value = run_command(
        {
            "command": "workspace.add",
            "session": session,
            "arguments": {"case": {"$session": other}, "name": "factory"},
        }
    )["value"]

    assert value["cases"] == ["baseline", "factory"]


def test_a_workspace_session_passed_as_a_problem_is_its_active_case():
    workspace = _workspace_session()
    receiver = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    value = run_command(
        {
            "command": "with_utilities_from",
            "session": receiver,
            "arguments": {"other_problem": {"$session": workspace}},
        }
    )["value"]

    assert value["data"]["utilities"]


def test_workspace_batch_runs_one_command_on_each_case():
    session = _workspace_session()
    run_command(
        {
            "command": "workspace.scenario",
            "session": session,
            "arguments": {"name": "tight_dt", "dt_cont_multiplier": 0.75},
        }
    )

    value = run_command(
        {
            "command": "workspace.batch",
            "session": session,
            "arguments": {"command": "target.direct_heat_integration"},
        }
    )["value"]

    assert list(value["results"]) == ["baseline", "tight_dt"]
    assert value["errors"] == {}
    assert value["results"]["tight_dt"]["results"]["targets"]


def test_workspace_batch_reports_a_failing_case_and_runs_the_rest():
    session = _workspace_session()

    value = run_command(
        {
            "command": "workspace.batch",
            "session": session,
            "arguments": {
                "command": "target.direct_heat_integration",
                "cases": ["missing", "baseline"],
            },
        }
    )["value"]

    assert list(value["results"]) == ["baseline"]
    assert value["errors"]["missing"]["type"] == "KeyError"


def test_an_empty_workspace_session_can_be_filled_and_saved():
    bundle = {
        "schema_version": "3",
        "project_name": "Site",
        "baseline_name": "baseline",
        "cases": {},
    }
    session = run_command({"command": "open", "workspace": {"bundle": bundle}})[
        "session"
    ]
    other = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]

    value = run_command(
        {
            "command": "workspace.add",
            "session": session,
            "arguments": {"case": {"$session": other}, "name": "baseline"},
        }
    )["value"]

    assert value["cases"] == ["baseline"]


def test_every_response_has_a_warnings_list():
    opened = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )
    targeted = run_command(
        {"command": "target.direct_heat_integration", "session": opened["session"]}
    )
    closed = run_command({"command": "close", "session": opened["session"]})
    unknown = run_command({"command": "close", "session": opened["session"]})

    for response in (opened, targeted, closed, unknown):
        assert response["warnings"] == []


def test_warnings_raised_by_a_command_are_returned(monkeypatch):
    target = _TargetAccessor.direct_heat_integration

    def warning_target(self, **kwargs):
        warnings.warn("The utilities fall short.", UserWarning, stacklevel=1)
        warnings.warn("The utilities fall short.", UserWarning, stacklevel=1)
        return target(self, **kwargs)

    monkeypatch.setattr(_TargetAccessor, "direct_heat_integration", warning_target)

    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )

    assert response["warnings"] == [
        {
            "type": "UserWarning",
            "message": "The utilities fall short.",
            "details": None,
        }
    ]


def test_concurrent_commands_each_return_only_their_own_warnings(monkeypatch):
    # The audit saw warnings cross between 24 concurrent responses, and the
    # global warning filters left modified, when calls ran in worker threads.
    target = _TargetAccessor.direct_heat_integration
    calls = 24
    barrier = threading.Barrier(calls, timeout=60)

    def warning_target(self, **kwargs):
        name = threading.current_thread().name
        warnings.warn(f"{name} before", UserWarning, stacklevel=1)
        barrier.wait()  # every call is now inside run_command at once
        warnings.warn(f"{name} after", UserWarning, stacklevel=1)
        return target(self, **kwargs)

    monkeypatch.setattr(_TargetAccessor, "direct_heat_integration", warning_target)
    filters = list(warnings.filters)
    showwarning = warnings.showwarning
    responses: dict[str, dict] = {}

    def call():
        responses[threading.current_thread().name] = run_command(
            {
                "command": "target.direct_heat_integration",
                "problem": {"sample_case": "basic_pinch.json"},
            }
        )

    threads = [threading.Thread(target=call, name=f"call-{i}") for i in range(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(responses) == calls
    for name, response in responses.items():
        assert [w["message"] for w in response["warnings"]] == [
            f"{name} before",
            f"{name} after",
        ]
    assert warnings.filters == filters
    assert warnings.showwarning is showwarning


def test_a_warning_raised_outside_a_command_is_shown_as_before(monkeypatch):
    target = _TargetAccessor.direct_heat_integration
    inside, outside = threading.Event(), threading.Event()
    shown = []

    def waiting_target(self, **kwargs):
        inside.set()
        assert outside.wait(timeout=60)
        return target(self, **kwargs)

    def outside_command():
        assert inside.wait(timeout=60)
        warnings.warn("Not from a command.", UserWarning, stacklevel=1)
        outside.set()

    monkeypatch.setattr(_TargetAccessor, "direct_heat_integration", waiting_target)
    monkeypatch.setattr(
        warnings, "showwarning", lambda message, *args: shown.append(str(message))
    )
    thread = threading.Thread(target=outside_command)
    thread.start()
    response = run_command(
        {
            "command": "target.direct_heat_integration",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )
    thread.join()

    assert response["warnings"] == []
    assert shown == ["Not from a command."]


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="context-aware warnings need Python 3.14"
)
def test_concurrent_commands_keep_their_warnings_with_context_aware_warnings():
    # With -X context_aware_warnings the filters are per context, and TESPy
    # (imported with the package) replaces showwarning, so recording through
    # catch_warnings alone would return no warnings.
    node = (
        "tests/application/test_commands.py::"
        "test_concurrent_commands_each_return_only_their_own_warnings"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "context_aware_warnings=1",
            "-m",
            "pytest",
            node,
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_validation_warnings_are_returned_one_per_issue():
    data = load_packaged_sample_case("basic_pinch.json")
    data["streams"][0]["dt_cont"] = {"value": -1.0, "unit": "delta_degC"}

    response = run_command({"command": "to_problem_json", "problem": {"data": data}})

    [warning] = response["warnings"]
    assert warning["type"] == "validation"
    assert warning["details"]["path"] == "streams[0].dt_cont"
    assert warning["message"] == warning["details"]["message"]


def test_default_utilities_added_on_loading_are_returned_as_warnings():
    data = load_packaged_sample_case("basic_pinch.json")
    data["utilities"] = []

    opened = run_command({"command": "open", "problem": {"data": data}})
    targeted = run_command(
        {"command": "target.direct_heat_integration", "session": opened["session"]}
    )

    defaults = [w for w in opened["warnings"] if w["type"] == "default_utility"]
    assert {w["details"]["kind"] for w in defaults} == {"hot", "cold"}
    assert all(w["details"]["name"] in w["message"] for w in defaults)
    # Reported when the problem loads, not again by later commands.
    assert targeted["warnings"] == []


def test_non_finite_numbers_are_returned_as_null_so_the_json_is_strict(monkeypatch):
    # Browsers' JSON.parse rejects NaN; a missing magnitude must come back null.
    monkeypatch.setattr(
        StreamCollection,
        "to_dict",
        lambda self, **_kwargs: {"t_supply": [float("nan"), float("inf")]},
    )

    text = run_command_json(
        json.dumps(
            {"command": "streams", "problem": {"sample_case": "basic_pinch.json"}}
        )
    )

    value = json.loads(text, parse_constant=_reject)["value"]
    assert value["hot_streams"]["t_supply"] == [None, None]


def _reject(constant: str):
    raise AssertionError(f"Non-standard JSON constant {constant}")


def test_design_command_returns_the_design_view_beside_the_result(monkeypatch):
    from OpenPinch.contracts.schema import contract_json_schemas
    from tests.analysis.heat_exchanger_networks.test_design_workflow import (
        _public_example_payload,
        _use_fake_default_executor,
    )

    _use_fake_default_executor(monkeypatch)

    response = run_command(
        {
            "command": "design.heat_exchanger_network",
            "problem": {"data": _public_example_payload()},
        }
    )

    view = response["view"]
    jsonschema.validate(view, contract_json_schemas()["HeatExchangerNetworkView"])
    assert view["rank"] == 1
    assert [row["rank"] for row in view["ranking"]] == [1]
    assert view["total_heat_recovery"] == pytest.approx(
        sum(
            state["duty"]
            for exchanger in response["value"]["network"]["exchangers"]
            if exchanger["kind"] == "recovery"
            for state in exchanger["period_states"]
        ),
        rel=1e-12,
    )


def _two_period_hen_payload() -> dict:
    """Tutorial 17's case: the four-stream example in two weighted periods."""
    from tests.analysis.heat_exchanger_networks.test_design_workflow import (
        _public_example_payload,
    )

    payload = _public_example_payload()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["base", "turndown"]
    payload["options"]["PROBLEM_PERIOD_WEIGHTS"] = [0.7, 0.3]
    for stream in payload["streams"]:
        duty = stream["heat_flow"]
        stream["heat_flow"] = {
            "values": [duty["value"], 0.8 * duty["value"]],
            "unit": duty["unit"],
        }
        stream["heat_capacity_flowrate"] = None
    return payload


def _use_two_period_fake_executor(monkeypatch) -> None:
    """The fake executor, its networks spread over a base and turndown period."""
    from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
        FakeSynthesisExecutor,
    )

    def two_periods(network):
        data = network.model_dump()
        for exchanger in data["exchangers"]:
            (state,) = exchanger["period_states"]
            turndown = {**state, "period_id": "turndown", "period_idx": 1}
            turndown["duty"] = 0.8 * state["duty"]
            exchanger["period_states"] = ({**state, "period_id": "base"}, turndown)
        return type(network).model_validate(data)

    class TwoPeriodExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, **kwargs):
            return tuple(
                outcome.model_copy(update={"network": two_periods(outcome.network)})
                if outcome.network is not None
                else outcome
                for outcome in super().execute(tasks, **kwargs)
            )

    from OpenPinch.analysis.heat_exchanger_networks.targeting import (
        network_evolution_method,
        open_hens_method,
        pinch_design_method,
        thermal_derivative_method,
    )

    for module in (
        open_hens_method,
        network_evolution_method,
        pinch_design_method,
        thermal_derivative_method,
    ):
        monkeypatch.setattr(module, "LocalSynthesisExecutor", TwoPeriodExecutor)


def test_multiperiod_design_command_returns_one_view_per_period(monkeypatch):
    from OpenPinch.contracts.schema import contract_json_schemas

    _use_two_period_fake_executor(monkeypatch)
    session = run_command(
        {"command": "open", "problem": {"data": _two_period_hen_payload()}}
    )["session"]
    run_command(
        {"command": "target.all_periods.all_heat_integration", "session": session}
    )

    response = run_command(
        {"command": "design.multiperiod_heat_exchanger_network", "session": session}
    )

    views = response["views"]
    schema = contract_json_schemas()["HeatExchangerNetworkView"]
    for view in views:
        jsonschema.validate(view, schema)
    assert [view["period_id"] for view in views] == ["base", "turndown"]
    assert [view["grid"]["period_id"] for view in views] == ["base", "turndown"]
    assert response["view"] == views[0]
    for view in views:
        assert view["total_heat_recovery"] == pytest.approx(
            sum(
                state["duty"]
                for exchanger in response["value"]["network"]["exchangers"]
                if exchanger["kind"] == "recovery"
                for state in exchanger["period_states"]
                if state["period_id"] == view["period_id"] and state["active"]
            ),
            rel=1e-12,
        )


def _design_reporting_progress(self, **_kwargs):
    progress.emit(
        analysis="hen_synthesis", stage="pinch_design_method", done=0, total=1
    )
    progress.emit(
        analysis="hen_synthesis", stage="pinch_design_method", done=1, total=1
    )
    return HeatExchangerNetworkDesignView(ValidationReport(valid=True, issues=[]))


def test_progress_callback_receives_json_events_during_a_command(monkeypatch):
    monkeypatch.setattr(_DesignAccessor, "pinch_design", _design_reporting_progress)
    # The stand-in result has no networks to view; only the progress matters.
    monkeypatch.setattr(
        HeatExchangerNetworkDesignView,
        "views",
        lambda self: (ValidationReport(valid=True, issues=[]),),
    )
    request = {
        "command": "design.pinch_design",
        "problem": {"sample_case": "basic_pinch.json"},
    }
    events = []

    response = run_command(request, progress=events.append)

    assert [(event["stage"], event["done"], event["total"]) for event in events] == [
        ("pinch_design_method", 0, 1),
        ("pinch_design_method", 1, 1),
    ]
    assert events[-1]["message"] == "Pinch design method: 1 of 1"
    assert json.loads(json.dumps(events)) == events
    assert response == run_command(request)


def test_json_text_entry_point_passes_progress_as_json_text(monkeypatch):
    monkeypatch.setattr(_DesignAccessor, "pinch_design", _design_reporting_progress)
    # The stand-in result has no networks to view; only the progress matters.
    monkeypatch.setattr(
        HeatExchangerNetworkDesignView,
        "views",
        lambda self: (ValidationReport(valid=True, issues=[]),),
    )
    request = {
        "command": "design.pinch_design",
        "problem": {"sample_case": "basic_pinch.json"},
    }
    events = []

    run_command_json(json.dumps(request), events.append)

    assert [json.loads(event)["done"] for event in events] == [0, 1]


class _Cancel(Exception):
    pass


def _cancel(_event):
    raise _Cancel


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        (
            "target.carnot_heat_pump",
            {"load_fraction": 0.25, "condensers": 1, "evaporators": 1},
        ),
        ("target.utility_placement", {"isothermal": 2}),
        ("design.pinch_design", {}),
    ],
)
def test_a_progress_callback_cancels_a_command_and_leaves_the_session(
    command, arguments
):
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    before = run_command(
        {"command": "target.direct_heat_integration", "session": session}
    )["results"]

    with pytest.raises(_Cancel):
        run_command(
            {"command": command, "session": session, "arguments": arguments},
            progress=_cancel,
        )

    after = run_command({"command": "to_problem_json", "session": session})
    assert after["results"] == before


def test_all_periods_progress_carries_each_period_as_its_scope(monkeypatch):
    solve = _TargetAccessor.direct_heat_integration

    def reporting_solve(self, **kwargs):
        progress.emit(analysis="a", stage="search", done=1, total=1)
        return solve(self, **kwargs)

    monkeypatch.setattr(_TargetAccessor, "direct_heat_integration", reporting_solve)
    events = []

    run_command(
        {
            "command": "target.all_periods.direct_heat_integration",
            "problem": {"sample_case": "crude_preheat_train_multiperiod.json"},
            "arguments": {"workers": 2},
        },
        progress=events.append,
    )

    assert sorted(event["scope"] for event in events) == ["base", "peak", "turndown"]
