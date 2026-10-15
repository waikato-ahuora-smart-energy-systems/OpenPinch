"""Regression tests that load real tabular problem sources end to end.

The adapter unit tests build synthetic frames, so they missed that every
bundled workbook and CSV directory failed the input contract (the reader
padded ``loc``/``index`` columns that ``StreamSchema`` forbids).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from OpenPinch import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

WORKBOOK_DIR = REPOSITORY_ROOT / "examples" / "OpenPinchWkbs"

# Its Utility Data dt_cont cells hold a literal "%" placeholder, which the
# reader rejects as non-numeric rather than guessing a value.
PERCENT_DT_CONT_WORKBOOK = "Bjork and Pettersson.xlsb"

BUNDLED_WORKBOOKS = sorted(WORKBOOK_DIR.glob("*.xlsb"))
LOADABLE_WORKBOOKS = [
    path for path in BUNDLED_WORKBOOKS if path.name != PERCENT_DT_CONT_WORKBOOK
]


def _load(source) -> PinchProblem:
    # Example data triggers expected naming/default warnings; only errors matter.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return PinchProblem(source=source)


def _magnitude(value) -> float:
    return float(getattr(value, "value", value))


def test_bundled_workbooks_are_discovered():
    assert len(BUNDLED_WORKBOOKS) > 40
    assert PERCENT_DT_CONT_WORKBOOK in {path.name for path in BUNDLED_WORKBOOKS}


@pytest.mark.parametrize("workbook", LOADABLE_WORKBOOKS, ids=lambda path: path.stem)
def test_bundled_workbook_loads(workbook: Path):
    problem = _load(workbook)

    assert len(problem.hot_streams) + len(problem.cold_streams) > 0


def test_workbook_with_percent_dt_cont_reports_the_bad_cell():
    with pytest.raises(ValueError, match=r"Column 'dt_cont' has a non-numeric"):
        _load(WORKBOOK_DIR / PERCENT_DT_CONT_WORKBOOK)


def _write_csv(path: Path, rows: list[list]) -> None:
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def _write_csv_directory(directory: Path, *, stream_columns: int) -> Path:
    header = ["Zone", "Stream", "Ts", "Tt", "Q", "dTcont", "HTC"]
    units = [None, None, "degC", "degC", "kW", "degC", "kW/m^2/degC"]
    streams = [
        ["Plant", "H1", 180.0, 60.0, 240.0, 4.0, 2.0],
        ["Plant", "H2", 150.0, 30.0, 180.0, 4.0, 2.0],
        ["Plant", "C1", 20.0, 135.0, 230.0, 4.0, 2.0],
        ["Plant", "C2", 80.0, 140.0, 240.0, 4.0, 2.0],
    ]
    _write_csv(
        directory / "streams.csv",
        [header[:stream_columns], units[:stream_columns]]
        + [row[:stream_columns] for row in streams],
    )
    _write_csv(
        directory / "utilities.csv",
        [
            ["Utility", "Type", "Ts", "Tt", "dTcont", "Price", "HTC", "Q"],
            [None, None, "degC", "degC", "degC", "$/MWh", "kW/m^2/degC", "kW"],
            ["HP Steam", "Hot", 250.0, 249.0, 5.0, 40.0, 1.0, None],
            ["Cooling Water", "Cold", 15.0, 25.0, 5.0, 5.0, 1.0, None],
        ],
    )
    return directory


def test_seven_column_csv_directory_loads(tmp_path: Path):
    problem = _load(_write_csv_directory(tmp_path, stream_columns=7))

    streams = {stream.name: stream for stream in problem.hot_streams}
    streams.update({stream.name: stream for stream in problem.cold_streams})
    assert set(streams) == {"H1", "H2", "C1", "C2"}
    assert _magnitude(streams["H1"].delta_t_contribution) == pytest.approx(4.0)
    assert _magnitude(streams["H1"].heat_transfer_coefficient) == pytest.approx(2.0)


def test_csv_directory_without_dt_cont_and_htc_uses_configured_defaults(
    tmp_path: Path,
):
    problem = _load(_write_csv_directory(tmp_path, stream_columns=5))

    stream = next(iter(problem.hot_streams))
    assert _magnitude(stream.delta_t_contribution) == pytest.approx(
        _magnitude(problem.config["THERMAL_DT_CONT"])
    )
    assert _magnitude(stream.heat_transfer_coefficient) > 0
