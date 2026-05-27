"""Excel workbook ingestion helpers for OpenPinch problem/result payloads."""

from __future__ import annotations

import json

import pandas as pd
from pandas.errors import EmptyDataError

from ._tabular_input import (
    get_column_names_and_units as _get_column_names_and_units,
)
from ._tabular_input import (
    problem_records_from_frame as _write_problem_to_dict_and_list,
)
from ._tabular_input import (
    target_records_from_frame as _write_targets_to_dict_and_list,
)
from .input_validation import validate_stream_data, validate_utility_data

__all__ = ["get_problem_from_excel", "get_results_from_excel"]


def get_problem_from_excel(excel_file, output_json=None):
    """Read workbook stream/utility sheets and return OpenPinch problem JSON."""
    try:
        streams_data = _parse_sheet_with_units(excel_file, sheet_name="Stream Data")
        streams_data = validate_stream_data(streams_data)

        utilities_data = _parse_sheet_with_units(excel_file, sheet_name="Utility Data")
        utilities_data = validate_utility_data(utilities_data)

        options_data = _parse_options_sheet(excel_file, sheet_name="Options")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Workbook file not found: {exc.filename}") from exc
    except (ValueError, EmptyDataError) as exc:
        raise ValueError(f"Failed to read workbook problem inputs: {exc}") from exc

    output_dict = {
        "streams": streams_data,
        "utilities": utilities_data,
        "options": options_data,
    }

    if isinstance(output_json, str):
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(output_dict, handle, indent=4)

    return output_dict


def get_results_from_excel(excel_file, output_json, project_name):
    """Read workbook summary results and return structured target JSON."""
    try:
        results_data = _parse_sheet_with_units(
            excel_file,
            sheet_name="Summary",
            row_units=2,
            row_data=4,
            project_name=project_name,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Workbook file not found: {exc.filename}") from exc
    except (ValueError, EmptyDataError) as exc:
        raise ValueError(f"Failed to read workbook summary inputs: {exc}") from exc

    output_dict = {"targets": results_data}

    if isinstance(output_json, str):
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(output_dict, handle, indent=4)

    return output_dict


def _parse_sheet_with_units(
    excel_file,
    sheet_name,
    row_units=1,
    row_data=2,
    project_name=None,
):
    """Read one workbook sheet and convert it to row dictionaries."""
    with pd.ExcelFile(excel_file) as workbook:
        if sheet_name not in workbook.sheet_names:
            available = ", ".join(workbook.sheet_names)
            raise ValueError(
                f"Workbook is missing required sheet '{sheet_name}'. "
                f"Available sheets: {available or '<none>'}."
            )
        df_full = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

    if df_full.empty:
        raise ValueError(f"Worksheet '{sheet_name}' is empty.")
    min_required_rows = max(row_units, row_data) + 1
    if len(df_full.index) < min_required_rows:
        raise ValueError(
            f"Worksheet '{sheet_name}' must include at least {min_required_rows} rows "
            "(names, units, and data rows)."
        )

    col_names, col_units = _get_column_names_and_units(
        df_full,
        sheet_name,
        row_units,
    )
    df_data: pd.DataFrame = df_full.iloc[row_data:].copy()

    for index in range(len(df_data.columns), len(col_names), -1):
        df_data = df_data.drop(columns=index - 1)
    df_data.columns = col_names[: len(df_data.columns)]
    for index in range(len(df_data.columns), len(col_names)):
        df_data[col_names[index]] = 0

    units_map = dict(zip(col_names, col_units))
    if sheet_name != "Summary":
        keep = ["t_supply", "t_target", "heat_flow", "dt_cont", "htc"]
        units_map = {key: units_map[key] for key in keep if key in units_map}

    if sheet_name == "Summary":
        return _write_targets_to_dict_and_list(df_data, units_map, project_name)
    return _write_problem_to_dict_and_list(df_data, units_map)


def _parse_options_sheet(
    excel_file,
    sheet_name: str = "Options",
):
    """Read key/value option rows from one workbook sheet."""
    with pd.ExcelFile(excel_file) as workbook:
        if sheet_name not in workbook.sheet_names:
            return {}
        df_full = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

    col0 = df_full.iloc[:, 0]
    col1 = df_full.iloc[:, 1]
    df_full = df_full.loc[col1.notna() & ~col0.astype(str).str.startswith("###")]
    if df_full.empty:
        return {}

    keys = df_full.iloc[:, 0].tolist()
    values = df_full.iloc[:, 1].tolist()

    def to_python(value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except AttributeError, RuntimeError, TypeError, ValueError:
                pass
        return value

    return {
        (key.strip() if isinstance(key, str) else key): to_python(value)
        for key, value in zip(keys, values)
    }
