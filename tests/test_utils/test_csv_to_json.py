from pathlib import Path

import pandas as pd

from OpenPinch.utils.csv_to_json import get_problem_from_csv


def _write_stream_csv(path: Path) -> None:
    rows = [
        ["ignored"] * 9,
        [None, None, "degC", "degC", "kW", "K", "kW/mK", None, None],
        ["Zone A", "H1", 150.0, 60.0, 100.0, 10.0, 0.5, "IN", 1],
        [None, None, 140.0, 55.0, 90.0, 10.0, 0.4, "IN", 2],
        ["", "", 130.0, 50.0, 80.0, 10.0, 0.3, "IN", 3],
    ]
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def _write_utility_csv(path: Path) -> None:
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
