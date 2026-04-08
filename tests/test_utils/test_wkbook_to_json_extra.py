"""Additional coverage tests for workbook-to-JSON helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from OpenPinch.utils import wkbook_to_json


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
