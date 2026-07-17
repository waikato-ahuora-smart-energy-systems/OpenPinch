"""CSV ingestion helpers that mirror the workbook-to-JSON conversion flow."""

from __future__ import annotations

from typing import IO, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from .json import write_json
from .records import (
    validate_stream_data,
    validate_utility_data,
)
from .tabular import (
    get_column_names_and_units as _get_column_names_and_units,
)
from .tabular import (
    problem_records_from_frame as _write_problem_to_dict_and_list,
)
from .tabular import (
    target_records_from_frame as _write_targets_to_dict_and_list,
)

__all__ = ["get_problem_from_csv", "get_results_from_csv"]


def get_problem_from_csv(
    streams_csv: Union[str, IO],
    utilities_csv: Union[str, IO],
    output_json: str | None = None,
    *,
    row_units: int = 1,
    row_data: int = 2,
    encoding: str = "utf-8-sig",
):
    """Read stream and utility CSV files into one OpenPinch input mapping."""
    try:
        streams_data = _parse_csv_with_units(
            streams_csv,
            kind="Stream Data",
            row_units=row_units,
            row_data=row_data,
            encoding=encoding,
        )
        streams_data = validate_stream_data(streams_data)
        utilities_data = _parse_csv_with_units(
            utilities_csv,
            kind="Utility Data",
            row_units=row_units,
            row_data=row_data,
            encoding=encoding,
        )
        utilities_data = validate_utility_data(utilities_data)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV input file not found: {exc.filename}") from exc
    except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Failed to read CSV problem inputs: {exc}") from exc

    output_dict = {
        "streams": streams_data,
        "utilities": utilities_data,
        "options": {},
    }

    if isinstance(output_json, str):
        write_json(output_json, output_dict)

    return output_dict


def get_results_from_csv(
    summary_csv: Union[str, IO],
    output_json: str | None,
    project_name: str,
    *,
    row_units: int = 2,
    row_data: int = 4,
    encoding: str = "utf-8-sig",
):
    """Read one summary CSV file into structured target JSON."""
    try:
        results_data = _parse_csv_with_units(
            summary_csv,
            kind="Summary",
            row_units=row_units,
            row_data=row_data,
            project_name=project_name,
            encoding=encoding,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV summary file not found: {exc.filename}") from exc
    except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Failed to read CSV summary input: {exc}") from exc

    output_dict = {"targets": results_data}

    if isinstance(output_json, str):
        write_json(output_json, output_dict)

    return output_dict


def _parse_csv_with_units(
    csv_file: Union[str, IO],
    *,
    kind: str,
    row_units: int = 1,
    row_data: int = 2,
    project_name: str | None = None,
    encoding: str = "utf-8-sig",
):
    """Read one CSV source and convert it using the shared tabular helpers."""
    df_full = pd.read_csv(csv_file, header=None, encoding=encoding, dtype=object)
    if df_full.empty:
        raise ValueError(f"{kind} CSV is empty.")
    min_required_rows = max(row_units, row_data) + 1
    if len(df_full.index) < min_required_rows:
        raise ValueError(
            f"{kind} CSV must include at least {min_required_rows} rows "
            "(names, units, and data rows)."
        )

    col_names, col_units = _get_column_names_and_units(
        df_full,
        sheet_name=kind,
        row_units=row_units,
    )
    df_data: pd.DataFrame = df_full.iloc[row_data:].copy()

    for index in range(len(df_data.columns), len(col_names), -1):
        df_data = df_data.drop(columns=index - 1)
    df_data.columns = col_names[: len(df_data.columns)]
    for index in range(len(df_data.columns), len(col_names)):
        df_data[col_names[index]] = 0

    def to_number_maybe(value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            try:
                if "." in stripped or "e" in stripped.lower():
                    return float(stripped)
                return int(stripped)
            except ValueError:
                return value
        return value

    df_data = df_data.map(to_number_maybe)
    units_map = dict(zip(col_names, col_units))
    if kind == "Summary":
        return _write_targets_to_dict_and_list(df_data, units_map, project_name)
    return _write_problem_to_dict_and_list(df_data, units_map)
