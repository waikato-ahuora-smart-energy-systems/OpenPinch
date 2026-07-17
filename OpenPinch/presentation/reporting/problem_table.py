"""Tabular and workbook presentation for domain problem tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...domain.problem_table import ProblemTable


def problem_table_frame(
    table: ProblemTable | None,
    *,
    round_decimals: int | None = None,
) -> pd.DataFrame:
    """Return a detached dataframe view of a problem table."""
    if table is None or table.data is None:
        return pd.DataFrame()
    if table.data.size == 0 or len(table.columns) == 0:
        return pd.DataFrame(columns=table.columns)
    frame = pd.DataFrame(table.data.copy(), columns=table.columns)
    if round_decimals is not None and not frame.empty:
        numeric_columns = frame.select_dtypes(include="number").columns
        frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].round(
            round_decimals
        )
    return frame


def export_problem_table(
    table: ProblemTable,
    filename: str = "problem_table",
    sheet_name: str = "ProblemTable",
    include_index: bool = False,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Write a problem table workbook and return the resulting path."""
    if table.data is None:
        raise ValueError("Cannot export an uninitialised ProblemTable.")
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    name = Path(filename).stem or "problem_table"
    output_path = output_dir / f"{name}.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        problem_table_frame(table).to_excel(
            writer,
            sheet_name=sheet_name,
            index=include_index,
        )
    return output_path


__all__ = ["export_problem_table", "problem_table_frame"]
