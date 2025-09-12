import pandas as pd
import json
import os
import openpyxl, pyxlsb

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
        "utilities": [...]
      }
    """
    
    streams_data = _parse_sheet_with_units(excel_file, sheet_name="Stream Data")
    utilities_data = _parse_sheet_with_units(excel_file, sheet_name="Utility Data")
    options_data = _set_options()

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
    results_data = _parse_sheet_with_units(excel_file, sheet_name="Summary", row_units=2, row_data=4, project_name=project_name)

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
    # First row = column names
    # Second row = units
    if sheet_name == "Stream Data":
        col_names = ['zone', 'name', 't_supply', 't_target', 'heat_flow', 'dt_cont', 'htc', 'loc', 'index']  
    elif sheet_name == "Utility Data":
        col_names = ['name', 'type', 't_supply', 't_target', 'dt_cont', 'price', 'htc', 'heat_flow']
    elif sheet_name == "Summary":
        col_names =['name', 'temp_pinch', 'Qh', 'Qc', 'Qr', 'degree_of_integration']
        prefix = "HU::"
        for i in range(6, len(df_full.columns) - 1):
            if df_full[i][0] == "Cold Utility" or df_full[i][0] == "Heat Receiver Utility":
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
    col_units = [s.replace('°', 'deg') if isinstance(s, str) else None for s in col_units]
    col_units = [s.replace('2', '^2') if isinstance(s, str) else None for s in col_units]
    
    return col_names, col_units


def _parse_sheet_with_units(excel_file, sheet_name, row_units=1, row_data=2, project_name=None):
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
    df_full = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

    # Get the column names and units from the first two rows
    col_names, col_units = _get_column_names_and_units(df_full, sheet_name, row_units)

    # The actual data starts from the third row
    df_data: pd.DataFrame = df_full.iloc[row_data:].copy()
    # print(df_full.columns.to_list())
    # Rename the DataFrame columns to the column names from Row 1
    for i in range(len(df_data.columns), len(col_names), -1):
        df_data = df_data.drop(columns=i-1)

    df_data.columns = col_names[:len(df_data.columns)]
    for i in range(len(df_data.columns), len(col_names)):
        df_data[col_names[i]] = 0
    
    # Build a mapping from column -> unit (from Row 2)
    units_map = dict(zip(col_names, col_units))

    if sheet_name == "Summary":
        return _write_targets_to_dict_and_list(df_data, units_map, project_name)
    else:
        return _write_problem_to_dict_and_list(df_data, units_map)


def _write_problem_to_dict_and_list(df_data: pd.DataFrame, units_map: dict) -> list:
    # Convert each row to a dictionary, attaching units where appropriate
    records = []
    for _, row in df_data.iterrows():
        entry = {}
        for col in df_data.columns.to_list():
            value = row[col]
            unit = units_map[col]

            # If the column is numeric, we store a dict with { value, units }
            # You can refine this check, e.g. exclude columns known to be text.
            if isinstance(unit, str) and unit.strip():
                entry[col] = {
                    "value": value if pd.api.types.is_number(value) else None,
                    "units": unit
                }
            else:
                # If it's not numeric or there's no unit string, just store the raw data
                entry[col] = value if not(pd.isna(value)) else None
        records.append(entry)

    return records


def _write_targets_to_dict_and_list(df_data: pd.DataFrame, units_map: dict, project_name: str) -> list:
    # Convert each row to a dictionary, attaching units where appropriate
    records = []
    for _, row in df_data.iterrows():
        if not isinstance(row['name'], str) or row['name'] == "Individual Process Targets":
            continue

        if row["name"] == "Total Site Targets":
            row["name"] = f"{project_name}/Total Site Target"
        elif row["name"] == "Total Process Targets":
            row["name"] = f"{project_name}/Total Process Target"
        elif row["name"] == "Total Integrated Targets":
            row["name"] = f"{project_name}/Direct Integration"
        else:
            zone_name = row["name"]
            row["name"] = f"{zone_name}/Direct Integration"
        
        entry = {}
        entry["hot_utilities"] = []
        entry["cold_utilities"] = []
        for col in df_data.columns.to_list():
            value = row[col]
            unit = units_map[col]

            # If the column is numeric, we store a dict with { value, units }
            # You can refine this check, e.g. exclude columns known to be text.
            if col[0:4] == "HU::":
                entry["hot_utilities"].append({
                    "name": col[4:len(col)],
                    "heat_flow": {
                        "value": float(value),
                        "units": unit,
                    }
                })
            elif col[0:4] == "CU::":
                entry["cold_utilities"].append({
                    "name": col[4:len(col)],
                    "heat_flow": {
                        "value": float(value),
                        "units": unit,
                    }
                })
            elif col == "temp_pinch":
                val = str(value).split("; ")
                if row["name"] == "Total Process Target":
                    val[0] = None
                    val[-1] = None
                else:
                    val[0] = float(val[0])
                    val[-1] = float(val[-1])
                entry[col] = {
                    "cold_temp": {
                        "value": val[0],
                        "units": unit,
                    }
                    ,
                    "hot_temp": {
                        "value": val[-1],
                        "units": unit,
                    },
                }
            elif col == "degree_of_integration":
                if pd.isna(value):
                    value = None
                else:
                    value = float(value) * 100
                entry[col] = {
                    "value": value,
                    "units": unit
                }
                                    
            elif pd.api.types.is_number(value) and isinstance(unit, str) and unit.strip():
                entry[col] = {
                    "value": float(value),
                    "units": unit
                }
            else:
                # If it's not numeric or there's no unit string, just store the raw data
                entry[col] = value if not(pd.isna(value)) else None
        records.append(entry)
    
    reordered = []
    for r in records:
        items = list(r.items())
        reordered.append(dict(items[2:] + items[:2]))
    return reordered


def _set_options():
    """
    Create and set default options.
    """
    return {
            "main": [],
            "turbine": [
                {"key": "PROP_TOP_0", "value": 450},
                {"key": "PROP_TOP_1", "value": 90},
                {"key": "PROP_TOP_2", "value": 0.1},
                {"key": "PROP_TOP_3", "value": 100},
                {"key": "PROP_TOP_4", "value": 1},
                {"key": "PROP_TOP_5", "value": 1},
                {"key": "PROP_TOP_6", "value": "Medina-Flores et al. (2010)"},
                {"key": "PROP_TOP_7", "value": True},
                {"key": "PROP_TOP_8", "value": False},
                {"key": "PROP_TOP_9", "value": False}
            ]
    }
