"""Fixture and private-array adapter tests for the OpenHENS migration."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import OpenPinch
from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.schemas.io import TargetInput
from OpenPinch.lib.schemas.synthesis import HeatExchangerNetworkSynthesisManifest
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.arrays import (
    problem_to_solver_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
REGRESSION_ARTIFACT_ROOT = FIXTURE_ROOT / "regression_artifacts"
ADAPTER_ARRAY_SNAPSHOT_ROOT = REGRESSION_ARTIFACT_ROOT / "adapter_array_snapshots"
CASE_DTMIN = {
    "Four-stream-Yee-and-Grossmann-1990-1": 14.0,
    "Nine-stream-Linnhoff-and-Ahmad-1999-1": 18.0,
}
AXIS_ARRAY_NAMES = {
    "cold_process_streams": {
        "T_c_cont",
        "T_c_in",
        "T_c_out",
        "c_cost",
        "cold_names",
        "f_c",
        "htc_c",
    },
    "cold_utilities": {
        "T_cu_in",
        "T_cu_out",
        "cu_cost",
        "htc_cu",
    },
    "hot_process_streams": {
        "T_h_cont",
        "T_h_in",
        "T_h_out",
        "f_h",
        "h_cost",
        "hot_names",
        "htc_h",
    },
    "hot_utilities": {
        "T_hu_in",
        "T_hu_out",
        "htc_hu",
        "hu_cost",
    },
}


@pytest.mark.parametrize(
    ("case_id", "stream_count", "utility_count"),
    [
        ("Four-stream-Yee-and-Grossmann-1990-1", 4, 2),
        ("Nine-stream-Linnhoff-and-Ahmad-1999-1", 9, 2),
    ],
)
def test_converted_openhens_json_loads_through_prepared_pinch_problem(
    case_id: str,
    stream_count: int,
    utility_count: int,
) -> None:
    fixture_path = FIXTURE_ROOT / f"{case_id}.json"
    payload = json.loads(fixture_path.read_text())

    target_input = TargetInput.model_validate(payload)
    problem = PinchProblem(source=fixture_path)
    arrays = problem_to_solver_arrays(problem, CASE_DTMIN[case_id])

    assert len(target_input.streams) == stream_count
    assert len(target_input.utilities) == utility_count
    assert all(stream.t_supply.unit == "K" for stream in target_input.streams)
    assert all(stream.t_target.unit == "K" for stream in target_input.streams)
    assert all("price" not in stream for stream in payload["streams"])
    assert all(utility.price.unit == "$/MWh" for utility in target_input.utilities)
    assert isinstance(problem.master_zone, Zone)
    assert isinstance(problem.master_zone.process_streams, StreamCollection)
    assert all(isinstance(stream, Stream) for stream in problem.hot_streams)
    assert all(isinstance(stream, Stream) for stream in problem.cold_streams)
    assert arrays.preparation["prepared_zone_class"] == "Zone"
    assert arrays.configuration["HENS_RUN_ID"] == case_id


def test_four_stream_adapter_snapshot_matches_openhens_source_arrays() -> None:
    case_id = "Four-stream-Yee-and-Grossmann-1990-1"
    fixture_path = FIXTURE_ROOT / f"{case_id}.json"
    snapshot_path = ADAPTER_ARRAY_SNAPSHOT_ROOT / case_id / "dTmin-14.json"
    snapshot = json.loads(snapshot_path.read_text())
    payload = problem_to_solver_arrays(PinchProblem(source=fixture_path), 14.0)
    current = payload.to_json_dict()

    assert current["array_shapes"] == snapshot["array_shapes"]
    _assert_axis_identities_match(current, snapshot)

    for name, expected in snapshot["arrays"].items():
        _assert_array_matches_by_identity(name, current, expected, snapshot)

    for name, expected in snapshot["source_openhens_arrays"].items():
        _assert_array_matches_by_identity(name, current, expected, snapshot)


def test_nine_stream_adapter_uses_openhens_order_and_real_utilities() -> None:
    case_id = "Nine-stream-Linnhoff-and-Ahmad-1999-1"
    payload = problem_to_solver_arrays(
        PinchProblem(source=FIXTURE_ROOT / f"{case_id}.json"),
        10.0,
    )

    assert payload.stream_identities["cold_process_streams"] == [
        "Process A.Cold 1 N",
        "Process A.Cold 2 N",
        "Process A.Cold 3 N",
        "Process A.Cold 4 N",
        "Process A.Cold 5 N",
    ]
    assert payload.utility_identities["hot_utilities"] == ["Hot Utility.HPS"]
    np.testing.assert_allclose(
        payload.arrays["T_c_in"], [373.15, 308.15, 358.15, 333.15, 413.15]
    )
    np.testing.assert_allclose(payload.arrays["f_c"], [100.0, 70.0, 350.0, 60.0, 200.0])
    np.testing.assert_allclose(payload.arrays["hu_cost"], [60.0])


def test_adapter_prefers_input_heat_capacity_flowrate_when_supplied() -> None:
    problem = PinchProblem(
        source={
            "streams": [
                {
                    "zone": "Site/Process A",
                    "name": "H1",
                    "t_supply": {"value": 500.0, "unit": "K"},
                    "t_target": {"value": 370.0, "unit": "K"},
                    "heat_flow": {"value": 999.0, "unit": "kW"},
                    "heat_capacity_flowrate": {
                        "value": 14.8,
                        "unit": "kW/delta_degC",
                    },
                },
                {
                    "zone": "Site/Process A",
                    "name": "C1",
                    "t_supply": {"value": 310.0, "unit": "K"},
                    "t_target": {"value": 470.0, "unit": "K"},
                    "heat_flow": {"value": 888.0, "unit": "kW"},
                    "flow_heat_capacity": 6.1,
                },
            ],
            "utilities": [
                {
                    "name": "HPS",
                    "type": "Hot",
                    "t_supply": {"value": 520.0, "unit": "K"},
                    "t_target": {"value": 520.0, "unit": "K"},
                    "heat_flow": None,
                    "htc": {"value": 5.0, "unit": "kW/m^2/K"},
                    "price": {"value": 60.0, "unit": "$/MWh"},
                },
                {
                    "name": "CW",
                    "type": "Cold",
                    "t_supply": {"value": 290.0, "unit": "K"},
                    "t_target": {"value": 310.0, "unit": "K"},
                    "heat_flow": None,
                    "htc": {"value": 1.0, "unit": "kW/m^2/K"},
                    "price": {"value": 10.0, "unit": "$/MWh"},
                },
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {"name": "Process A", "type": "Process Zone", "children": None}
                ],
            },
        }
    )

    payload = problem_to_solver_arrays(problem, 10.0)

    assert float(problem.hot_streams[0].CP) != pytest.approx(14.8)
    assert float(problem.cold_streams[0].CP) != pytest.approx(6.1)
    assert payload.arrays["f_h"].tolist() == [14.8]
    assert payload.arrays["f_c"].tolist() == [6.1]


def test_adapter_converts_absolute_temperatures_to_kelvin_for_solver_arrays() -> None:
    case_id = "Four-stream-Yee-and-Grossmann-1990-1"
    fixture_path = FIXTURE_ROOT / f"{case_id}.json"
    kelvin_arrays = problem_to_solver_arrays(PinchProblem(source=fixture_path), 14.0)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    celsius_payload = deepcopy(payload)
    for record in celsius_payload["streams"] + celsius_payload["utilities"]:
        for field in ("t_supply", "t_target"):
            record[field]["value"] -= 273.15
            record[field]["unit"] = "degC"

    celsius_arrays = problem_to_solver_arrays(
        PinchProblem(source=celsius_payload), 14.0
    )

    assert celsius_arrays.unit_conventions["temperature"] == "K"
    for array_name in (
        "T_c_in",
        "T_c_out",
        "T_cu_in",
        "T_cu_out",
        "T_h_in",
        "T_h_out",
        "T_hu_in",
        "T_hu_out",
    ):
        np.testing.assert_allclose(
            celsius_arrays.arrays[array_name],
            kelvin_arrays.arrays[array_name],
        )


@pytest.mark.parametrize("case_id", CASE_DTMIN)
def test_reordered_fixtures_change_private_solver_axis_order(case_id: str) -> None:
    base = problem_to_solver_arrays(
        PinchProblem(source=FIXTURE_ROOT / f"{case_id}.json"),
        CASE_DTMIN[case_id],
    )
    reordered = problem_to_solver_arrays(
        PinchProblem(source=FIXTURE_ROOT / f"{case_id}.reordered.json"),
        CASE_DTMIN[case_id],
    )

    assert set(reordered.axis_maps) == set(base.axis_maps)
    assert reordered.utility_identities == base.utility_identities
    assert set(reordered.stream_identities) == set(base.stream_identities)
    assert reordered.stream_identities["hot_process_streams"] == list(
        reversed(base.stream_identities["hot_process_streams"])
    )
    assert reordered.stream_identities["cold_process_streams"] == list(
        reversed(base.stream_identities["cold_process_streams"])
    )
    assert reordered.arrays["hot_names"].tolist() == list(
        reversed(base.arrays["hot_names"].tolist())
    )
    assert reordered.arrays["cold_names"].tolist() == list(
        reversed(base.arrays["cold_names"].tolist())
    )


def test_adapter_requires_prepared_pinch_problem_and_rejects_bypass_payloads() -> None:
    good_problem = PinchProblem(
        source=FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    cached_arrays = problem_to_solver_arrays(good_problem, 14.0)
    manifest = HeatExchangerNetworkSynthesisManifest(
        run_id="run-1",
        approach_temperatures=(14.0,),
        derivative_thresholds=(0.5,),
        stage_selection=(1,),
    )
    bypass_values = [
        [{"Designation": "Hot"}],
        TargetInput(streams=[]),
        manifest,
        cached_arrays,
        cached_arrays.to_json_dict(),
    ]

    with pytest.raises(RuntimeError, match="requires PinchProblem.load"):
        problem_to_solver_arrays(PinchProblem(), 14.0)

    for value in bypass_values:
        with pytest.raises(TypeError, match="prepared PinchProblem"):
            problem_to_solver_arrays(value, 14.0)  # type: ignore[arg-type]


def test_adapter_rejects_non_positive_dtmin() -> None:
    problem = PinchProblem(
        source=FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )

    with pytest.raises(ValueError, match="dTmin"):
        problem_to_solver_arrays(problem, -1.0)


def test_conversion_errors_include_row_and_field_context(tmp_path: Path) -> None:
    converter = _load_converter_module()
    source = tmp_path / "bad-openhens-case.csv"
    source.write_text(
        "\n".join(
            [
                "Number;Subsystem;Description;Designation;Supply Temp;Target Temp;Flow heat capacity;HTC;Stream cost",
                ";;;;T_in;T_out;f;htc;_cost",
                ";;;;K;K;kW/K;kW/m2-K;$/kW-y",
                "1;Process A;Hot A;Hot;650;370;not-a-number;1;0",
                "1;Process A;Cold A;Cold;410;650;15;1;0",
                "1;Utility;HPS;Hot Utility;680;680;;5;80",
                "1;Utility;CW;Cold Utility;300;320;;1;15",
                ";;;;;;;;",
                "Number;Subsystem;Description;Designation;HX unit cost;HX area coefficient;HX area exponent;;",
                ";;;;unit_cost;_coeff;A_exp;;",
                "1;Process A;Process-process HX;Exchange;5500;150;1;;",
                "1;Utility;Process-heater HX;Heating;5500;150;1;;",
                "1;Utility;Process-cooler HX;Cooling;5500;150;1;;",
            ]
        )
    )

    with pytest.raises(
        converter.ConversionError,
        match="process streams row 4: Flow heat capacity must be numeric",
    ):
        converter.parse_openhens_csv(source)


def test_openhens_csv_conversion_is_not_public_runtime_api() -> None:
    assert not hasattr(OpenPinch, "convert_openhens_fixtures")
    assert not hasattr(OpenPinch, "problem_to_solver_arrays")


def _load_converter_module():
    converter_path = REPO_ROOT / "scripts" / "convert_openhens_fixtures.py"
    spec = importlib.util.spec_from_file_location(
        "_openhens_fixture_converter",
        converter_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load converter script module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_axis_identities_match(current: dict, snapshot: dict) -> None:
    assert set(current["axis_maps"]) == set(snapshot["axis_maps"])
    for axis_name, current_axis in current["axis_maps"].items():
        assert current_axis == snapshot["axis_maps"][axis_name]

    for identity_group in ("stream_identities", "utility_identities"):
        assert set(current[identity_group]) == set(snapshot[identity_group])
        for axis_name, identities in current[identity_group].items():
            assert identities == snapshot[identity_group][axis_name]


def _assert_array_matches_by_identity(
    name: str,
    current: dict,
    expected: list,
    snapshot: dict,
) -> None:
    actual = current["arrays"][name]
    expected = _reordered_expected_array(name, current, expected, snapshot)
    if name.endswith("_names"):
        assert _normalised_names(actual) == _normalised_names(expected)
    else:
        np.testing.assert_allclose(actual, expected)


def _reordered_expected_array(
    name: str,
    current: dict,
    expected: list,
    snapshot: dict,
) -> list:
    for axis_name, array_names in AXIS_ARRAY_NAMES.items():
        if name not in array_names:
            continue
        current_identities = current["stream_identities"].get(
            axis_name,
            current["utility_identities"].get(axis_name, []),
        )
        snapshot_axis = snapshot["axis_maps"][axis_name]
        return [expected[snapshot_axis[identity]] for identity in current_identities]
    return expected


def _normalised_names(values: list) -> list[str]:
    return [str(value).strip() for value in values]
