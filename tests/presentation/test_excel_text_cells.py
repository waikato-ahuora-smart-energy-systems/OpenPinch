"""Text cells in every Excel export are written as plain strings."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
import pytest

import OpenPinch.presentation.dashboard.exports as dashboard_exports
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.presentation.reporting.problem_table import export_problem_table

LEADING_CHARACTERS = ("=", "+", "-", "@", "\t", "\r")


def _text_cells(path: Path) -> dict:
    workbook = openpyxl.load_workbook(path)
    return {
        (sheet.title, cell.coordinate): cell
        for sheet in workbook.worksheets
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    }


def _assert_plain_text(path: Path, expected: set[tuple[str, str]]) -> None:
    """No formula cells; ``expected`` and every risky text cell are quoted."""
    cells = _text_cells(path)
    assert [key for key, cell in cells.items() if cell.data_type == "f"] == []
    # XML reads a carriage return back as a newline.
    risky = {
        key
        for key, cell in cells.items()
        if cell.value.startswith((*LEADING_CHARACTERS, "\n"))
    }
    assert expected <= risky
    assert all(cells[key].quotePrefix for key in risky)


def test_summary_and_problem_table_sheets_write_names_as_plain_text(tmp_path):
    problem = PinchProblem(source="basic_pinch.json", project_name="=1+1")
    problem.target.direct_heat_integration()

    path = problem.export_excel(tmp_path / "summary.xlsx")

    cells = _text_cells(path)
    named = {key for key, cell in cells.items() if cell.value.startswith("=1+1")}
    assert ("Summary", "A2") in named
    _assert_plain_text(path, named)


def test_problem_table_export_writes_text_as_plain_strings(tmp_path):
    table = ProblemTable({"=A": [1.0], "+B": [2.0]}, add_default_labels=False)

    path = export_problem_table(table, output_dir=tmp_path)

    _assert_plain_text(path, {("ProblemTable", "A1"), ("ProblemTable", "B1")})


class _Streamlit:
    def __init__(self, destination: str) -> None:
        self.destination = destination
        self.errors: list[str] = []
        self.successes: list[str] = []

    def text_input(self, *_args, **_kwargs) -> str:
        return self.destination

    def button(self, *_args, **_kwargs) -> bool:
        return True

    def error(self, message: str) -> None:
        self.errors.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)


@pytest.mark.parametrize("prefix", LEADING_CHARACTERS)
def test_dashboard_table_export_writes_text_as_plain_strings(tmp_path, prefix):
    destination = tmp_path / "table.xlsx"
    text = f"{prefix}SUM(1,1)"
    frame = pd.DataFrame({"Stream": [text, "plain"], "T": [1.0, -2.0]})

    dashboard_exports.render_table_export(
        _Streamlit(str(destination)),
        str(destination),
        dashboard_key="d",
        target_name="t",
        frame=frame,
        table_kind="real",
    )

    _assert_plain_text(destination, {("Problem Table", "A2")})
