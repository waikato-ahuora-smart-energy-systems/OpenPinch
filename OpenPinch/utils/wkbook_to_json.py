import json
import pandas as pd

#######################################################################################################
# Public API
#######################################################################################################


def get_problem_from_excel(excel_file, output_json=None):
    """
    Reads the 'Streams' and 'Utilities' sheets from `excel_file`, assuming
    each has:
      - Row 1 = column names
      - Row 2 = column units
      - Row 3 onward = data
    Then writes a JSON file (similar to the provided example) containing:
      {
        "streams": [...],
        "utilities": [...],
        "options": [...],
      }
    """

    streams_data = _parse_sheet_with_units(excel_file, sheet_name="Stream Data")
    utilities_data = _parse_sheet_with_units(excel_file, sheet_name="Utility Data")
    options_data = _parse_options_sheet(excel_file, sheet_name="Options")

    # Build the final dictionary. If you have more sections (e.g. "options"), read them similarly.
    output_dict = {
        "streams": streams_data,
        "utilities": utilities_data,
        "options": options_data,
    }

    if isinstance(output_json, str):
        # Write the final JSON structure to file
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
        print(f"JSON data successfully written to {output_json}")

    return output_dict


def get_results_from_excel(excel_file, output_json, project_name):
    """
    Reads the 'Results' sheet from `excel_file`, assuming
    each has:
      - Row 1 = column names
      - Row 2 = column units
      - Row 3 onward = data
    Then writes a JSON file (similar to the provided example) containing:
      {
        "graphs": [...]
    """
    results_data = _parse_sheet_with_units(
        excel_file,
        sheet_name="Summary",
        row_units=2,
        row_data=4,
        project_name=project_name,
    )

    # Build the final dictionary. If you have more sections (e.g. "options"), read them similarly.
    output_dict = {
        "targets": results_data,
    }

    if isinstance(output_json, str):
        # Write the final JSON structure to file
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
        print(f"JSON data successfully written to {output_json}")

    return output_dict


#######################################################################################################
# Helper functions
#######################################################################################################


def _get_column_names_and_units(df_full, sheet_name, row_units=1):
    """Derive canonical column names and unit strings for a workbook sheet."""
    # First row = column names
    # Second row = units
    if sheet_name == "Stream Data":
        col_names = [
            "zone",
            "name",
            "t_supply",
            "t_target",
            "heat_flow",
            "dt_cont",
            "htc",
            "loc",
            "index",
        ]
    elif sheet_name == "Utility Data":
        col_names = [
            "name",
            "type",
            "t_supply",
            "t_target",
            "dt_cont",
            "price",
            "htc",
            "heat_flow",
        ]
    elif sheet_name == "Summary":
        col_names = ["name", "temp_pinch", "Qh", "Qc", "Qr", "degree_of_integration"]
        prefix = "HU::"
        for i in range(6, len(df_full.columns) - 1):
            if (
                df_full[i][0] == "Cold Utility"
                or df_full[i][0] == "Heat Receiver Utility"
            ):
                prefix = "CU::"
            if df_full[i][1] == "Default HU":
                col_names.append(prefix + "HU")
            elif df_full[i][1] == "Default CU":
                col_names.append(prefix + "CU")
            else:
                col_names.append(prefix + df_full[i][1])
        col_names.append("utility_cost")

    col_units = df_full.iloc[row_units].tolist()

    for _ in range(len(col_units), len(col_names)):
        col_units.append(None)
    col_units = [
        s.replace("°", "deg") if isinstance(s, str) else None for s in col_units
    ]
    col_units = [
        s.replace("2", "^2") if isinstance(s, str) else None for s in col_units
    ]

    return col_names, col_units


def _parse_sheet_with_units(
    excel_file, sheet_name, row_units=1, row_data=2, project_name=None
):
    """
    Read key/value options from `sheet_name` and return them as a dictionary.
    """
    # Read the entire sheet as-is (no header)
    df_full = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

    # Get the column names and units from the first two rows
    col_names, col_units = _get_column_names_and_units(df_full, sheet_name, row_units)

    # The actual data starts from the third row
    df_data: pd.DataFrame = df_full.iloc[row_data:].copy()
    # print(df_full.columns.to_list())
    # Rename the DataFrame columns to the column names from Row 1
    for i in range(len(df_data.columns), len(col_names), -1):
        df_data = df_data.drop(columns=i - 1)

    df_data.columns = col_names[: len(df_data.columns)]
    for i in range(len(df_data.columns), len(col_names)):
        df_data[col_names[i]] = 0

    # Build a mapping from column -> unit (from Row 2)
    units_map = dict(zip(col_names, col_units))

    if sheet_name == "Summary":
        return _write_targets_to_dict_and_list(df_data, units_map, project_name)
    else:
        return _write_problem_to_dict_and_list(df_data, units_map)


def _parse_options_sheet(
    excel_file, 
    sheet_name: str = "Options",
):
    """
    Reads a sheet from `excel_file` in which:
      - Row 1 = column names
      - Row 2 = column units
      - Row 3 onward = data

    Returns:
      A list of dictionaries, where each dictionary represents a row of data.
      Numeric columns are stored as:
         column_name: {"value": <number>, "units": <unit string>}
      String columns (like 'name' or 'zone') remain as simple keys and values.
    """
    # Read the entire sheet as-is (no header)

    with pd.ExcelFile(excel_file) as xls:
        if sheet_name not in xls.sheet_names:
            return {}
        df_full = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    col0 = df_full.iloc[:, 0]
    col1 = df_full.iloc[:, 1]
    df_full = df_full.loc[col1.notna() & ~col0.astype(str).str.startswith("###")]
    if df_full.empty:
        return {}

    keys = df_full.iloc[:, 0].tolist()
    values = df_full.iloc[:, 1].tolist()

    def _to_python(val):
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        return val

    options_dict = {
        (k.strip() if isinstance(k, str) else k): _to_python(v)
        for k, v in zip(keys, values)
    }
    return options_dict


def _write_problem_to_dict_and_list(df_data: pd.DataFrame, units_map: dict) -> list:
    """Convert stream/utility worksheets into JSON-ready row dictionaries with units."""
    if df_data.empty:
        return []

    clean_df = df_data.replace({pd.NA: None}).reset_index(drop=True)
    records = clean_df.to_dict(orient="records")

    for col, unit in units_map.items():
        if col not in clean_df.columns:
            continue

        column_values = clean_df[col]
        if isinstance(unit, str) and unit.strip():
            numeric_values = pd.to_numeric(column_values, errors="coerce")
            payloads = [
                {"value": (None if pd.isna(val) else float(val)), "units": unit}
                for val in numeric_values
            ]
            for record, payload in zip(records, payloads):
                record[col] = payload
        else:
            for record, value in zip(records, column_values):
                record[col] = value if not pd.isna(value) else None

    return records


def _write_targets_to_dict_and_list(
    df_data: pd.DataFrame, units_map: dict, project_name: str
) -> list:
    """Convert summary worksheet into structured target records, including utility splits."""
    if df_data.empty:
        return []

    name_series = df_data["name"]
    mask_valid = name_series.map(lambda x: isinstance(x, str))
    mask_skip = name_series.eq("Individual Process Targets")
    filtered = df_data.loc[mask_valid & ~mask_skip].reset_index(drop=True).copy()

    if filtered.empty:
        return []

    original_names = filtered["name"].copy()
    filtered.loc[original_names == "Total Site Targets", "name"] = (
        f"{project_name}/Total Site Target"
    )
    filtered.loc[original_names == "Total Process Targets", "name"] = (
        f"{project_name}/Total Process Target"
    )
    filtered.loc[original_names == "Total Integrated Targets", "name"] = (
        f"{project_name}/Direct Integration"
    )
    mask_direct = ~original_names.isin(
        ["Total Site Targets", "Total Process Targets", "Total Integrated Targets"]
    )
    filtered.loc[mask_direct, "name"] = (
        original_names[mask_direct].astype(str) + "/Direct Integration"
    )

    hot_cols = [col for col in filtered.columns if col.startswith("HU::")]
    cold_cols = [col for col in filtered.columns if col.startswith("CU::")]
    other_cols = [
        col
        for col in filtered.columns
        if col not in hot_cols + cold_cols
    ]

    clean_df = filtered.replace({pd.NA: None})
    clean_df = clean_df.where(~clean_df.isna(), None)
    base_records = clean_df[other_cols].to_dict(orient="records")

    hot_units = {col: units_map[col] for col in hot_cols}
    cold_units = {col: units_map[col] for col in cold_cols}

    hot_matrix = clean_df[hot_cols].to_numpy(dtype=float) if hot_cols else None
    cold_matrix = clean_df[cold_cols].to_numpy(dtype=float) if cold_cols else None

    results = []
    for idx, record in enumerate(base_records):
        entry = {
            "hot_utilities": [],
            "cold_utilities": [],
        }

        if hot_matrix is not None:
            entry["hot_utilities"] = []
            for col_idx, col in enumerate(hot_cols):
                entry["hot_utilities"].append(
                    {
                        "name": col[4:],
                        "heat_flow": {
                            "value": hot_matrix[idx, col_idx],
                            "units": hot_units[col],
                        },
                    }
                )

        if cold_matrix is not None:
            entry["cold_utilities"] = []
            for col_idx, col in enumerate(cold_cols):
                entry["cold_utilities"].append(
                    {
                        "name": col[4:],
                        "heat_flow": {
                            "value": cold_matrix[idx, col_idx],
                            "units": cold_units[col],
                        },
                    }
                )

        for col, value in record.items():
            unit = units_map.get(col)
            if col == "temp_pinch":
                if value not in (None, ""):
                    parts = str(value).split("; ")
                is_process_target = record["name"] == f"{project_name}/Total Process Target"
                if is_process_target:
                    parts[0] = None
                    parts[-1] = None
                else:
                    parts[0] = float(parts[0])
                    parts[-1] = float(parts[-1])
                entry[col] = {
                    "cold_temp": {"value": parts[0], "units": unit},
                    "hot_temp": {"value": parts[-1], "units": unit},
                }
            elif col == "degree_of_integration":
                if value is None or pd.isna(value):
                    entry[col] = {"value": None, "units": unit}
                else:
                    entry[col] = {"value": float(value) * 100, "units": unit}
            elif (
                pd.api.types.is_number(value)
                and isinstance(unit, str)
                and unit
            ):
                entry[col] = {"value": float(value), "units": unit}
            else:
                entry[col] = value if value is not None else None

        items = list(entry.items())
        results.append(dict(items[2:] + items[:2]))

    return results
