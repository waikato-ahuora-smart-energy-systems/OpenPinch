"""Regression tests for wkbook to json utility helpers."""

from pathlib import Path
import pandas as pd
import pytest
from OpenPinch.utils.wkbook_to_json import (
    _validate_stream_data,
    get_problem_from_excel,
    get_results_from_excel,
)
import numpy as np
from OpenPinch.utils import wkbook_to_json


def _write_test_workbook(path: Path):
    """
    Create an Excel file with the three sheets expected by the module:
      - 'Stream Data' (header row, units row, data rows)
      - 'Utility Data'
      - 'Summary'     (custom layout used by get_results_from_excel)
    All sheets are written with header=None, matching the reader.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # -------------------- Stream Data --------------------
        # Row 0 (names) is ignored by the parser for this sheet; it supplies its own column names.
        # Row 1 (units) is used.
        stream_units = [
            None,  # zone (string)
            None,  # name (string)
            "degC",  # t_supply
            "degC",  # t_target
            "kW",  # heat_flow
            "K",  # dt_cont
            "kW/mK",  # htc  (avoid '2' so the simple replace doesn't mutate it)
            None,  # loc (string)
            None,  # index (int but without unit string -> stored raw)
        ]
        stream_data = [["Zone A", "H1", 150.0, 60.0, 100.0, 10.0, 0.5, "IN", 1]]
        df_stream = pd.DataFrame(
            [["names-ignored"] * len(stream_units), stream_units] + stream_data
        )
        df_stream.to_excel(xw, sheet_name="Stream Data", header=None, index=False)

        # -------------------- Utility Data --------------------
        util_units = [
            None,  # name
            None,  # type
            "degC",  # t_supply
            "degC",  # t_target
            "K",  # dt_cont
            "$/MWh",  # price
            "kW/mK",  # htc
            "kW",  # heat_flow
        ]
        util_data = [
            ["HP Steam", "Hot", 250.0, 200.0, 10.0, 50.0, 0.8, 75.0],
            ["Cooling Water", "Cold", 20.0, 30.0, 5.0, 5.0, 0.3, 60.0],
        ]
        df_util = pd.DataFrame([["ignored"] * len(util_units), util_units] + util_data)
        df_util.to_excel(xw, sheet_name="Utility Data", header=None, index=False)

        # -------------------- Summary --------------------
        # The parser expects:
        #   row 0: "section" labels to detect switching from HU to CU ("Cold Utility" or "Heat Receiver Utility")
        #   row 1: utility names ("Default HU"/"Default CU" turn into "HU"/"CU"; else used verbatim)
        #   row 2: units (used to attach to values)
        #   row 4+: data rows
        # Columns 0..5 are fixed: name, temp_pinch, Qh, Qc, Qr, degree_of_integration
        # Then utility columns (index >= 6) until last column, which is utility_cost.
        row0 = [
            "Meta",
            "Meta",
            "Meta",
            "Meta",
            "Meta",
            "Meta",
            "Hot Utility",
            "Hot Utility",
            "Cold Utility",
            "Cold Utility",
            "Meta",
        ]
        row1 = [
            "name",
            "temp_pinch",
            "Qh",
            "Qc",
            "Qr",
            "degree_of_integration",
            "Default HU",
            "MP Steam",
            "Default CU",
            "Chilled Water",
            "utility_cost",
        ]
        row2 = [None, "degC", "kW", "kW", "kW", "%", "kW", "kW", "kW", "kW", "$/h"]

        # Data rows start at row index 4 (row_data=4)
        # Two rows: a "Total Site Targets" row and a zone-level row
        # temp_pinch format: "cold; hot"
        data1 = [
            "Total Site Targets",
            "85; 125",
            100.0,
            80.0,
            20.0,
            0.70,  # degree_of_integration -> multiplied by 100 in parser
            75.0,
            55.0,
            60.0,
            30.0,
            1234.0,
        ]
        data2 = [
            "Zone A",
            "60; 120",
            90.0,
            70.0,
            15.0,
            0.50,
            50.0,
            40.0,
            45.0,
            25.0,
            999.0,
        ]
        # Add one filler row (index 3) so data begins at index 4
        df_summary = pd.DataFrame(
            [row0, row1, row2, ["filler"] * len(row2), data1, data2]
        )
        df_summary.to_excel(xw, sheet_name="Summary", header=None, index=False)


def test_get_problem_from_excel_happy_path(tmp_path: Path):
    xlsx = tmp_path / "problem.xlsx"
    _write_test_workbook(xlsx)

    out = get_problem_from_excel(xlsx)

    # Basic structure
    assert set(out.keys()) == {"streams", "utilities", "options"}
    assert isinstance(out["streams"], list) and isinstance(out["utilities"], list)

    # One stream row
    s = out["streams"][0]
    # String fields are kept raw
    assert s["zone"] == "Zone A"
    assert s["name"] == "H1"
    # Unit-backed fields become dicts {value, units}
    assert s["t_supply"]["value"] == 150.0 and s["t_supply"]["units"] == "degC"
    assert s["t_target"]["value"] == 60.0 and s["t_target"]["units"] == "degC"
    assert s["heat_flow"]["value"] == 100.0 and s["heat_flow"]["units"] == "kW"
    # Field with no unit string is stored raw (not a dict)
    assert s["index"] == 1

    # Utilities preserved similarly
    u0 = out["utilities"][0]
    assert u0["name"] == "HP Steam"
    assert u0["type"] == "Hot"
    assert u0["heat_flow"]["value"] == 75.0 and u0["heat_flow"]["units"] == "kW"

    # # Options present with expected shape (don’t overfit exact values)
    # opts = out["options"]
    # assert (
    #     "turbine" in opts
    #     and isinstance(opts["turbine"], list)
    #     and len(opts["turbine"]) > 0
    # )


def test_get_results_from_excel_parses_summary(tmp_path: Path):
    xlsx = tmp_path / "results.xlsx"
    _write_test_workbook(xlsx)

    out = get_results_from_excel(xlsx, output_json=None, project_name="Proj")

    assert set(out.keys()) == {"targets"}
    targets = out["targets"]
    assert isinstance(targets, list) and len(targets) == 2

    # Row derived from "Total Site Targets"
    ts = next(t for t in targets if t["name"].startswith("Proj/Total Site Target"))
    # degree_of_integration gets scaled by 100
    assert ts["degree_of_integration"]["value"] == pytest.approx(70.0)
    assert ts["degree_of_integration"]["units"] == "%"
    # temp pinch split into cold/hot floats with units
    assert ts["temp_pinch"]["cold_temp"]["value"] == pytest.approx(85.0)
    assert ts["temp_pinch"]["cold_temp"]["units"] == "degC"
    assert ts["temp_pinch"]["hot_temp"]["value"] == pytest.approx(125.0)
    # utility cost becomes a normal numeric field wrapped with units
    assert ts["utility_cost"]["value"] == pytest.approx(1234.0)
    # hot & cold utilities are flattened with names from the header rows
    hu_names = {h["name"] for h in ts["hot_utilities"]}
    cu_names = {c["name"] for c in ts["cold_utilities"]}
    assert "HU" in hu_names  # from "Default HU"
    assert "MP Steam" in hu_names
    assert "CU" in cu_names  # from "Default CU"
    assert "Chilled Water" in cu_names

    # Zone-level row
    z = next(t for t in targets if t["name"] == "Zone A/Direct Integration")
    assert z["Qh"]["value"] == pytest.approx(90.0)
    # Utility cost present & wrapped with units
    assert z["utility_cost"]["value"] == pytest.approx(999.0)
    assert z["utility_cost"]["units"] == "$/h"
    # Work a sample utility value
    hu0 = z["hot_utilities"][0]
    assert (
        hu0["heat_flow"]["value"] in (50.0, 40.0) and hu0["heat_flow"]["units"] == "kW"
    )


def test_get_results_from_excel_writes_json_when_path_given(tmp_path: Path):
    xlsx = tmp_path / "results.xlsx"
    _write_test_workbook(xlsx)
    out_json = tmp_path / "out.json"

    out = get_results_from_excel(xlsx, output_json=str(out_json), project_name="Proj")

    # Function returns the dict and writes the JSON file
    assert out_json.exists()
    assert out["targets"]  # non-empty


def test_validate_stream_data_inherits_missing_zone_from_previous_stream():
    streams = [
        {"zone": "Zone A", "name": "H1"},
        {"zone": None, "name": "C1"},
        {"zone": "", "name": "H2"},
        {"zone": "2", "name": "3"},
        {"zone": None, "name": "4"},
    ]

    out = _validate_stream_data(streams)

    assert [s["zone"] for s in out] == ["Zone A", "Zone A", "Zone A", "Z2", "Z2"]
    assert [s["name"] for s in out] == ["H1", "C1", "H2", "S3", "S4"]


def test_validate_stream_data_defaults_first_missing_zone_then_inherits():
    streams = [
        {"zone": None, "name": "H1"},
        {"zone": "", "name": "C1"},
        {"zone": "Zone B", "name": "H2"},
        {"zone": None, "name": "C2"},
    ]

    out = _validate_stream_data(streams)

    assert [s["zone"] for s in out] == [
        "Process Zone",
        "Process Zone",
        "Zone B",
        "Zone B",
    ]


def test_validate_stream_data_defaults_name_when_zone_and_name_missing():
    streams = [
        {"zone": None, "name": None, "t_supply": {"value": 150.0, "units": "degC"}},
        {"zone": "", "name": "", "t_supply": {"value": 140.0, "units": "degC"}},
        {"zone": "Zone C", "name": "H3", "t_supply": {"value": 130.0, "units": "degC"}},
        {"zone": None, "name": None, "t_supply": {"value": 120.0, "units": "degC"}},
    ]

    out = _validate_stream_data(streams)

    assert [s["zone"] for s in out] == [
        "Process Zone",
        "Process Zone",
        "Zone C",
        "Zone C",
    ]
    assert [s["name"] for s in out] == ["S1", "S2", "H3", "S3"]


# ===== Merged from test_wkbook_to_json_extra.py =====
"""Additional coverage tests for workbook-to-JSON helpers."""


def test_get_problem_from_excel_writes_json(tmp_path: Path):
    xlsx = tmp_path / "problem.xlsx"
    out_json = tmp_path / "problem.json"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["ignored"] * 9,
                [None, None, "degC", "degC", "kW", "K", "kW/mK", None, None],
                ["Zone A", "H1", 150.0, 60.0, 100.0, 10.0, 0.5, "IN", 1],
            ]
        ).to_excel(writer, sheet_name="Stream Data", header=None, index=False)
        pd.DataFrame(
            [
                ["ignored"] * 8,
                [None, None, "degC", "degC", "K", "$/MWh", "kW/mK", "kW"],
                ["HP Steam", "Hot", 250.0, 200.0, 10.0, 50.0, 0.8, 75.0],
            ]
        ).to_excel(writer, sheet_name="Utility Data", header=None, index=False)

    out = wkbook_to_json.get_problem_from_excel(xlsx, output_json=str(out_json))
    assert out_json.exists()
    assert out["streams"][0]["name"] == "H1"


def test_get_column_names_and_units_pads_units_for_short_stream_sheet():
    df = pd.DataFrame([["ignored"] * 5, [None] * 5])
    names, units = wkbook_to_json._get_column_names_and_units(df, "Stream Data")
    assert len(names) == 9
    assert len(units) == 9


def test_parse_sheet_with_units_handles_extra_and_missing_columns(monkeypatch):
    captured = {}

    def _capture(df_data, units_map):
        captured["df"] = df_data
        captured["units"] = units_map
        return []

    monkeypatch.setattr(wkbook_to_json, "_write_problem_to_dict_and_list", _capture)

    extra_df = pd.DataFrame(
        [
            ["ignored"] * 11,
            [None, None, "degC", "degC", "kW", "K", "kW/mK", None, None, None, None],
            ["Zone A", "H1", 150, 60, 100, 10, 0.5, "IN", 1, "x", "y"],
        ]
    )
    monkeypatch.setattr(
        wkbook_to_json.pd, "read_excel", lambda *args, **kwargs: extra_df.copy()
    )
    wkbook_to_json._parse_sheet_with_units("dummy.xlsx", sheet_name="Stream Data")
    assert len(captured["df"].columns) == 9

    missing_df = pd.DataFrame(
        [
            ["ignored"] * 7,
            [None, None, "degC", "degC", "kW", "K", "kW/mK"],
            ["Zone A", "H1", 150, 60, 100, 10, 0.5],
        ]
    )
    monkeypatch.setattr(
        wkbook_to_json.pd, "read_excel", lambda *args, **kwargs: missing_df.copy()
    )
    wkbook_to_json._parse_sheet_with_units("dummy.xlsx", sheet_name="Stream Data")
    assert len(captured["df"].columns) == 9
    assert captured["df"].iloc[0]["loc"] == 0
    assert captured["df"].iloc[0]["index"] == 0


def test_parse_options_sheet_filters_comments_and_numpy_scalars(tmp_path: Path):
    xlsx = tmp_path / "options.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["### comment", 100],
                ["alpha", np.int64(5)],
                ["beta", np.float64(2.5)],
                ["gamma", None],
            ]
        ).to_excel(writer, sheet_name="Options", header=None, index=False)

    parsed = wkbook_to_json._parse_options_sheet(xlsx)
    assert parsed["alpha"] == 5
    assert parsed["beta"] == 2.5
    assert "gamma" not in parsed


def test_parse_options_sheet_empty_after_filter_returns_empty_dict(tmp_path: Path):
    xlsx = tmp_path / "options_empty.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame([["### only comment", 1], ["another", None]]).to_excel(
            writer,
            sheet_name="Options",
            header=None,
            index=False,
        )

    assert wkbook_to_json._parse_options_sheet(xlsx) == {}


def test_parse_options_sheet_to_python_none_and_item_exception_paths(monkeypatch):
    class _BadItem:
        def item(self):
            raise RuntimeError("item failed")

    class _FakeExcel:
        sheet_names = ["Options"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    df = pd.DataFrame([["alpha", "sentinel"], ["beta", _BadItem()]])
    monkeypatch.setattr(wkbook_to_json.pd, "ExcelFile", lambda excel_file: _FakeExcel())
    monkeypatch.setattr(
        wkbook_to_json.pd, "read_excel", lambda xls, sheet_name, header=None: df.copy()
    )
    monkeypatch.setattr(
        wkbook_to_json.pd,
        "isna",
        lambda value: value == "sentinel",
    )

    parsed = wkbook_to_json._parse_options_sheet("dummy.xlsx")
    assert parsed["alpha"] is None
    assert isinstance(parsed["beta"], _BadItem)


def test_write_problem_to_dict_and_list_empty_and_missing_columns():
    assert wkbook_to_json._write_problem_to_dict_and_list(pd.DataFrame(), {}) == []

    df = pd.DataFrame([{"name": "S1", "heat_flow": 10.0}])
    rows = wkbook_to_json._write_problem_to_dict_and_list(
        df_data=df,
        units_map={"heat_flow": "kW", "t_supply": "degC"},
    )
    assert rows[0]["name"] == "S1"
    assert rows[0]["heat_flow"]["value"] == 10.0


def test_write_targets_to_dict_and_list_special_name_and_none_fields():
    df = pd.DataFrame(
        [
            {
                "name": "Total Process Targets",
                "temp_pinch": "70; 100",
                "Qh": 10.0,
                "Qc": 8.0,
                "Qr": 2.0,
                "degree_of_integration": None,
                "HU::HU": 3.0,
                "CU::CU": 1.0,
                "utility_cost": 50.0,
            }
        ]
    )
    units = {
        "temp_pinch": "degC",
        "Qh": "kW",
        "Qc": "kW",
        "Qr": "kW",
        "degree_of_integration": "%",
        "HU::HU": "kW",
        "CU::CU": "kW",
        "utility_cost": "$/h",
    }

    out = wkbook_to_json._write_targets_to_dict_and_list(df, units, project_name="Proj")

    assert out[0]["name"] == "Proj/Total Process Target"
    assert out[0]["temp_pinch"]["cold_temp"]["value"] is None
    assert out[0]["temp_pinch"]["hot_temp"]["value"] is None
    assert out[0]["degree_of_integration"]["value"] is None


def test_write_targets_to_dict_and_list_empty_paths():
    empty_columns = ["name", "temp_pinch", "Qh", "Qc", "Qr", "degree_of_integration"]
    empty_df = pd.DataFrame(columns=empty_columns)
    assert (
        wkbook_to_json._write_targets_to_dict_and_list(
            empty_df, {}, project_name="Proj"
        )
        == []
    )

    filtered_out_df = pd.DataFrame(
        [
            {"name": "Individual Process Targets", "temp_pinch": "50; 70"},
            {"name": 123, "temp_pinch": "50; 70"},
        ]
    )
    assert (
        wkbook_to_json._write_targets_to_dict_and_list(
            filtered_out_df, {}, project_name="Proj"
        )
        == []
    )


def test_validate_utilities_data_alias_executes():
    out = wkbook_to_json._validate_utilities_data(
        [
            {"name": "HP Steam", "type": "Hot"},
            {"name": "Cooling Water", "type": "Cold"},
        ]
    )
    assert out[0]["name"] == "HP Steam"
