from OpenPinch.utils.wkbook_to_json import (
    get_problem_from_excel,
    get_results_from_excel,
)

from pathlib import Path
from datetime import datetime
import pandas as pd
import pytest


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
            None,            # zone (string)
            None,            # name (string)
            "degC",          # t_supply
            "degC",          # t_target
            "kW",            # heat_flow
            "K",             # dt_cont
            "kW/mK",         # htc  (avoid '2' so the simple replace doesn't mutate it)
            None,            # loc (string)
            None,            # index (int but without unit string -> stored raw)
        ]
        stream_data = [
            ["Zone A", "H1", 150.0, 60.0, 100.0, 10.0, 0.5, "IN", 1]
        ]
        df_stream = pd.DataFrame([["names-ignored"] * len(stream_units), stream_units] + stream_data)
        df_stream.to_excel(xw, sheet_name="Stream Data", header=None, index=False)

        # -------------------- Utility Data --------------------
        util_units = [
            None,            # name
            None,            # type
            "degC",          # t_supply
            "degC",          # t_target
            "K",             # dt_cont
            "$/MWh",         # price
            "kW/mK",         # htc
            "kW",            # heat_flow
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
        row0 = ["Meta", "Meta", "Meta", "Meta", "Meta", "Meta",
                "Hot Utility", "Hot Utility", "Cold Utility", "Cold Utility", "Meta"]
        row1 = ["name", "temp_pinch", "Qh", "Qc", "Qr", "degree_of_integration",
                "Default HU", "MP Steam", "Default CU", "Chilled Water", "utility_cost"]
        row2 = [None, "degC", "kW", "kW", "kW", "%", "kW", "kW", "kW", "kW", "$/h"]

        # Data rows start at row index 4 (row_data=4)
        # Two rows: a "Total Site Targets" row and a zone-level row
        # temp_pinch format: "cold; hot"
        data1 = [
            "Total Site Targets", "85; 125",
            100.0, 80.0, 20.0, 0.70,      # degree_of_integration -> multiplied by 100 in parser
            75.0, 55.0, 60.0, 30.0, 1234.0
        ]
        data2 = [
            "Zone A", "60; 120",
            90.0, 70.0, 15.0, 0.50,
            50.0, 40.0, 45.0, 25.0, 999.0
        ]
        # Add one filler row (index 3) so data begins at index 4
        df_summary = pd.DataFrame([row0, row1, row2, ["filler"] * len(row2), data1, data2])
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

    # Options present with expected shape (don’t overfit exact values)
    opts = out["options"]
    assert "turbine" in opts and isinstance(opts["turbine"], list) and len(opts["turbine"]) > 0


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
    assert hu0["heat_flow"]["value"] in (50.0, 40.0) and hu0["heat_flow"]["units"] == "kW"


def test_get_results_from_excel_writes_json_when_path_given(tmp_path: Path):
    xlsx = tmp_path / "results.xlsx"
    _write_test_workbook(xlsx)
    out_json = tmp_path / "out.json"

    out = get_results_from_excel(xlsx, output_json=str(out_json), project_name="Proj")

    # Function returns the dict and writes the JSON file
    assert out_json.exists()
    assert out["targets"]  # non-empty
