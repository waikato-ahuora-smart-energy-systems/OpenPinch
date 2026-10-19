"""Regression tests for csv to json utility helpers."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest

import OpenPinch.adapters.io.csv as csv_to_json
import OpenPinch.adapters.io.tabular as _tabular_input
from OpenPinch.adapters.io.csv import get_problem_from_csv


def _write_stream_csv(path: Path) -> None:
    """Write stream csv data used by this test module."""
    rows = [
        ["ignored"] * 9,
        [None, None, "degC", "degC", "kW", "K", "kW/mK", None, None],
        ["Zone A", "H1", 150.0, 60.0, 100.0, 10.0, 0.5, "IN", 1],
        [None, None, 140.0, 55.0, 90.0, 10.0, 0.4, "IN", 2],
        ["", "", 130.0, 50.0, 80.0, 10.0, 0.3, "IN", 3],
    ]
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def _write_utility_csv(path: Path) -> None:
    """Write utility csv data used by this test module."""
    rows = [
        ["ignored"] * 8,
        [None, None, "degC", "degC", "K", "$/MWh", "kW/mK", "kW"],
        [None, "Hot", 260.0, 210.0, 10.0, 60.0, 0.9, 70.0],
        ["", "Cold", 15.0, 25.0, 5.0, 5.0, 0.3, 40.0],
        ["HP Steam", "Hot", 250.0, 200.0, 10.0, 50.0, 0.8, 75.0],
    ]
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_get_problem_from_csv_normalises_missing_zone_and_name(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    _write_stream_csv(streams_csv)
    _write_utility_csv(utilities_csv)

    out = get_problem_from_csv(streams_csv, utilities_csv, output_json=None)

    zones = [s["zone"] for s in out["streams"]]
    names = [s["name"] for s in out["streams"]]
    utility_names = [u["name"] for u in out["utilities"]]

    assert zones == ["Zone A", "Zone A", "Zone A"]
    assert names == ["H1", "S1", "S2"]
    assert utility_names == ["HP Steam"]


# ===== Merged from test_csv_to_json_extra.py =====
"""Additional coverage tests for CSV ingestion helpers."""


def _write_csv(path: Path, rows: list[list]):
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def _stream_rows() -> list[list]:
    return [
        ["ignored"] * 9,
        [None, None, "degC", "degC", "kW", "K", "kW/mK", None, None],
        ["Zone A", "H1", 150.0, 60.0, 100.0, 10.0, 0.5, "IN", 1],
    ]


def _utility_rows() -> list[list]:
    return [
        ["ignored"] * 8,
        [None, None, "degC", "degC", "K", "$/MWh", "kW/mK", "kW"],
        ["HP Steam", "Hot", 250.0, 200.0, 10.0, 50.0, 0.8, 75.0],
    ]


def _summary_rows() -> list[list]:
    row0 = [
        "Meta",
        "Meta",
        "Meta",
        "Meta",
        "Meta",
        "Meta",
        "Hot Utility",
        "Cold Utility",
        "Meta",
    ]
    row1 = [
        "name",
        "pinch_temp",
        "Qh",
        "Qc",
        "Qr",
        "degree_of_integration",
        "Default HU",
        "Default CU",
        "utility_cost",
    ]
    row2 = [None, "degC", "kW", "kW", "kW", "%", "kW", "kW", "$/h"]
    row3 = ["filler"] * 9
    row4 = ["Total Site Targets", "85; 125", 100.0, 80.0, 20.0, 0.7, 50.0, 40.0, 999.0]
    return [row0, row1, row2, row3, row4]


def test_get_problem_from_csv_writes_json(tmp_path: Path, capsys):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    output_json = tmp_path / "problem.json"
    _write_csv(streams_csv, _stream_rows())
    _write_csv(utilities_csv, _utility_rows())

    out = csv_to_json.get_problem_from_csv(
        streams_csv,
        utilities_csv,
        output_json=str(output_json),
    )

    assert output_json.exists()
    assert json.loads(output_json.read_text()) == out
    assert capsys.readouterr().out == ""


def test_get_results_from_csv_writes_json(tmp_path: Path, capsys):
    summary_csv = tmp_path / "summary.csv"
    output_json = tmp_path / "results.json"
    _write_csv(summary_csv, _summary_rows())

    out = csv_to_json.get_results_from_csv(
        summary_csv=summary_csv,
        output_json=str(output_json),
        project_name="Proj",
    )

    assert output_json.exists()
    assert json.loads(output_json.read_text()) == out
    assert out["targets"][0]["scope"] == "Proj"
    assert out["targets"][0]["zone_type"] == "Site"
    assert out["targets"][0]["integration_type"] == "Utility"
    assert out["targets"][0]["target_method"] == "Heat Exchange"
    assert "name" not in out["targets"][0]
    assert capsys.readouterr().out == ""


def test_get_results_from_csv_wraps_missing_and_invalid_summary_inputs(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="CSV summary file not found"):
        csv_to_json.get_results_from_csv(tmp_path / "missing.csv", None, "Site")

    invalid = tmp_path / "invalid_summary.csv"
    invalid.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to read CSV summary input"):
        csv_to_json.get_results_from_csv(invalid, None, "Site")


def test_parse_csv_with_units_drops_extra_columns(monkeypatch):
    rows = [
        ["ignored"] * 11,
        [None, None, "degC", "degC", "kW", "K", "kW/mK", None, None, None, None],
        ["Zone A", "H1", 150, 60, 100, 10, 0.5, "IN", 1, "x", "y"],
    ]
    captured = {}

    def _capture(df_data, units_map):
        captured["df"] = df_data
        captured["units"] = units_map
        return []

    monkeypatch.setattr(csv_to_json, "_write_problem_to_dict_and_list", _capture)

    csv_to_json._parse_csv_with_units(
        csv_file=io.StringIO(pd.DataFrame(rows).to_csv(header=False, index=False)),
        kind="Stream Data",
        row_units=1,
        row_data=2,
    )

    assert list(captured["df"].columns) == [
        "zone",
        "name",
        "t_supply",
        "t_target",
        "heat_flow",
        "dt_cont",
        "htc",
    ]


def test_parse_csv_with_units_rejects_empty_dataframe(monkeypatch):
    monkeypatch.setattr(
        csv_to_json.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame()
    )

    with pytest.raises(ValueError, match="CSV is empty"):
        csv_to_json._parse_csv_with_units(io.StringIO(""), kind="Stream Data")


def test_tabular_helpers_reject_unknown_sheet_and_invalid_pinch_temperature():
    with pytest.raises(ValueError, match="Unsupported tabular sheet kind"):
        _tabular_input.get_column_names_and_units(pd.DataFrame(), "Not A Sheet")

    assert _tabular_input._parse_temp_pinch("") == (None, None)
    assert _tabular_input._parse_temp_pinch("80") == (80.0, 80.0)

    with pytest.raises(ValueError, match="Invalid pinch_temp"):
        _tabular_input._parse_temp_pinch("1;2;3")


def test_parse_csv_with_units_leaves_missing_columns_absent_and_converts_blanks(
    monkeypatch,
):
    rows = [
        ["ignored"] * 5,
        [None, None, "degC", "degC", "kW"],
        ["Zone A", "H1", "  ", 60, 100],
    ]
    captured = {}

    def _capture(df_data, units_map):
        captured["df"] = df_data
        return []

    monkeypatch.setattr(csv_to_json, "_write_problem_to_dict_and_list", _capture)

    csv_to_json._parse_csv_with_units(
        csv_file=io.StringIO(pd.DataFrame(rows).to_csv(header=False, index=False)),
        kind="Stream Data",
        row_units=1,
        row_data=2,
    )

    df = captured["df"]
    # Absent dt_cont/htc columns must not be padded (a 0 would override the
    # configured defaults); the input contract fills them in.
    assert list(df.columns) == ["zone", "name", "t_supply", "t_target", "heat_flow"]
    assert df.iloc[0]["t_supply"] is None


def test_get_problem_from_csv_reports_short_stream_csv(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    _write_csv(streams_csv, [["ignored"] * 9])
    _write_csv(utilities_csv, _utility_rows())

    with pytest.raises(ValueError, match="Failed to read CSV problem inputs"):
        csv_to_json.get_problem_from_csv(streams_csv, utilities_csv)


def test_get_problem_from_csv_reports_missing_file(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    _write_csv(streams_csv, _stream_rows())

    with pytest.raises(FileNotFoundError, match="CSV input file not found"):
        csv_to_json.get_problem_from_csv(streams_csv, tmp_path / "missing.csv")


def test_get_problem_from_csv_skips_blank_and_whitespace_rows(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    streams_csv.write_text(
        "zone,name,t_supply,t_target,heat_flow,dt_cont,htc\n"
        ",,degC,degC,kW,K,kW/mK\n"
        "Zone A,H1,150,60,100,10,0.5\n"
        ",,,,,,\n"
        "  , ,  ,,   ,,\n"
        "Zone A,C1,40,120,90,10,0.5\n",
        encoding="utf-8",
    )
    _write_csv(utilities_csv, _utility_rows())

    out = get_problem_from_csv(streams_csv, utilities_csv)

    assert [s["name"] for s in out["streams"]] == ["H1", "C1"]


def test_get_problem_from_csv_maps_columns_by_header_name(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    streams_csv.write_text(
        "Zone,Name,T_target,T_supply,Heat flow,Comment\n"
        ",,degC,degF,kW,\n"
        "Zone A,H1,50,392,100,first\n",
        encoding="utf-8",
    )
    utilities_csv.write_text(
        "Type,Name,Price,T_supply,T_target\n,,$/MWh,degC,degC\nHot,HU,60,260,259\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="'Comment'"):
        out = get_problem_from_csv(streams_csv, utilities_csv)

    stream = out["streams"][0]
    assert stream["t_supply"] == {"value": 392.0, "unit": "degF"}
    assert stream["t_target"] == {"value": 50.0, "unit": "degC"}
    assert "Comment" not in stream and "comment" not in stream
    utility = out["utilities"][0]
    assert utility["name"] == "HU"
    assert utility["type"] == "Hot"
    assert utility["price"] == {"value": 60.0, "unit": "$/MWh"}


def test_get_problem_from_csv_rejects_duplicate_header_columns(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    streams_csv.write_text(
        "Zone,Name,T_supply,TS,T_target,Heat flow\n"
        ",,degC,degC,degC,kW\n"
        "Zone A,H1,150,150,60,100\n",
        encoding="utf-8",
    )
    _write_csv(utilities_csv, _utility_rows())

    with pytest.raises(ValueError, match="t_supply"):
        get_problem_from_csv(streams_csv, utilities_csv)
