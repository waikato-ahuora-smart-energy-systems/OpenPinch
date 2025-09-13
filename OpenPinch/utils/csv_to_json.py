import pandas as pd
import json
from typing import Union, IO
from .wkbook_to_json import (
    _set_options, 
    _get_column_names_and_units, 
    _write_problem_to_dict_and_list, 
    _write_targets_to_dict_and_list,
)

#######################################################################################################
# Public API
#######################################################################################################

def get_problem_from_csv(
    streams_csv: Union[str, IO],
    utilities_csv: Union[str, IO],
    output_json: str | None = None,
    *,
    row_units: int = 1,
    row_data: int = 2,
    encoding: str = "utf-8-sig",
):
    """
    Reads two CSV files (streams & utilities) with the same structure you use in Excel:
      - Row 1 = column names
      - Row 2 = column units
      - Row 3 onward = data

    Parameters
    ----------
    streams_csv : path or file-like
        CSV for the "Stream Data" sheet.
    utilities_csv : path or file-like
        CSV for the "Utility Data" sheet.
    output_json : str | None
        If provided, write the combined JSON to this path.
    row_units : int
        Zero-based row index containing units (defaults to 1 -> the 2nd row).
    row_data : int
        Zero-based first row of data (defaults to 2 -> the 3rd row).
    encoding : str
        CSV text encoding (default 'utf-8-sig' to gracefully handle BOMs).

    Returns
    -------
    dict
        {
          "streams": [...],
          "utilities": [...],
          "options": {...}
        }
    """
    streams_data = _parse_csv_with_units(
        streams_csv, kind="Stream Data", row_units=row_units, row_data=row_data, encoding=encoding
    )
    utilities_data = _parse_csv_with_units(
        utilities_csv, kind="Utility Data", row_units=row_units, row_data=row_data, encoding=encoding
    )
    options_data = _set_options()

    output_dict = {
        "streams": streams_data,
        "utilities": utilities_data,
        "options": options_data,
    }

    if isinstance(output_json, str):
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
        print(f"JSON data successfully written to {output_json}")

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
    """
    Reads a CSV equivalent of your 'Summary' sheet with:
      - Row 1 = column names
      - Row 2 = column units
      - Row 3 onward = data

    (Defaults mirror your Excel call where units were on row 2 and data started at row 4.)

    Parameters
    ----------
    summary_csv : path or file-like
        CSV for the 'Summary' sheet.
    output_json : str | None
        If provided, write the JSON to this path.
    project_name : str
        Used to prefix/rename totals exactly like your Excel version.
    row_units : int
        Zero-based units row index (default 2).
    row_data : int
        Zero-based first data row (default 4).
    encoding : str
        CSV text encoding.

    Returns
    -------
    dict
        {
          "targets": [...]
        }
    """
    results_data = _parse_csv_with_units(
        summary_csv,
        kind="Summary",
        row_units=row_units,
        row_data=row_data,
        project_name=project_name,
        encoding=encoding,
    )

    output_dict = {
        "targets": results_data,
    }

    if isinstance(output_json, str):
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
        print(f"JSON data successfully written to {output_json}")

    return output_dict


#######################################################################################################
# Helpers
#######################################################################################################

def _parse_csv_with_units(
    csv_file: Union[str, IO],
    *,
    kind: str,
    row_units: int = 1,
    row_data: int = 2,
    project_name: str | None = None,
    encoding: str = "utf-8-sig",
):
    """
    CSV analogue of _parse_sheet_with_units(...) for Excel.
    Reads the entire CSV without headers, then delegates to your existing
    _get_column_names_and_units(...) and writer helpers.
    """
    # Read as-is (no header row); keep object dtype so mixed cells don't get mangled.
    df_full = pd.read_csv(csv_file, header=None, encoding=encoding, dtype=object)

    # Build column names & units using your existing logic keyed by 'kind'
    col_names, col_units = _get_column_names_and_units(df_full, sheet_name=kind, row_units=row_units)

    # Slice data rows
    df_data: pd.DataFrame = df_full.iloc[row_data:].copy()

    # Trim or pad columns to match expected 'col_names'
    for i in range(len(df_data.columns), len(col_names), -1):
        df_data = df_data.drop(columns=i-1)
    df_data.columns = col_names[:len(df_data.columns)]
    for i in range(len(df_data.columns), len(col_names)):
        df_data[col_names[i]] = 0

    # Convert numeric-looking cells to numbers where appropriate
    # (Excel reader often infers numerics; CSV keeps strings—this helps parity.)
    def _to_number_maybe(x):
        if isinstance(x, str):
            xs = x.strip()
            if xs == "":
                return None
            try:
                # int vs float
                if "." in xs or "e" in xs.lower():
                    return float(xs)
                return int(xs)
            except ValueError:
                return x
        return x

    df_data = df_data.applymap(_to_number_maybe)

    units_map = dict(zip(col_names, col_units))

    if kind == "Summary":
        return _write_targets_to_dict_and_list(df_data, units_map, project_name)
    else:
        return _write_problem_to_dict_and_list(df_data, units_map)
