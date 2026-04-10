"""Regression tests for csv to json utility helpers."""

from pathlib import Path
import pandas as pd
from OpenPinch.utils.csv_to_json import get_problem_from_csv
import io
import json
from OpenPinch.utils import csv_to_json


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
        "temp_pinch",
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


def test_get_problem_from_csv_writes_json(tmp_path: Path):
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


def test_get_results_from_csv_writes_json(tmp_path: Path):
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
    assert out["targets"][0]["name"].startswith("Proj/")


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

    assert len(captured["df"].columns) == 9
    assert "index" in captured["df"].columns


def test_parse_csv_with_units_pads_missing_columns_and_converts_blanks(monkeypatch):
    rows = [
        ["ignored"] * 7,
        [None, None, "degC", "degC", "kW", "K", "kW/mK"],
        ["Zone A", "H1", "  ", 60, 100, 10, 0.5],
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
    assert len(df.columns) == 9
    assert df.iloc[0]["t_supply"] is None
    assert df.iloc[0]["loc"] == 0
    assert df.iloc[0]["index"] == 0
