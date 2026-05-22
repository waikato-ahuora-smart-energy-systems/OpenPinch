"""Shared tabular parsing helpers for workbook and CSV ingestion."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def get_column_names_and_units(df_full, sheet_name, row_units=1):
    """Derive canonical column names and unit strings for a workbook sheet."""
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
            if df_full[i][0] in {"Cold Utility", "Heat Receiver Utility"}:
                prefix = "CU::"
            if df_full[i][1] == "Default HU":
                col_names.append(prefix + "HU")
            elif df_full[i][1] == "Default CU":
                col_names.append(prefix + "CU")
            else:
                col_names.append(prefix + df_full[i][1])
        col_names.append("utility_cost")
    else:
        raise ValueError(f"Unsupported tabular sheet kind: {sheet_name}")

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


def problem_records_from_frame(df_data: pd.DataFrame, units_map: dict) -> list:
    """Convert stream/utility tables into JSON-ready row dictionaries with units."""
    if df_data.empty:
        return []

    clean_df = df_data.replace({pd.NA: None}).reset_index(drop=True)
    records = clean_df.to_dict(orient="records")

    for column_name, unit in units_map.items():
        if column_name not in clean_df.columns:
            continue

        column_values = clean_df[column_name]
        if isinstance(unit, str) and unit.strip():
            numeric_values = pd.to_numeric(column_values, errors="coerce")
            payloads = [
                {"value": (None if pd.isna(val) else float(val)), "units": unit}
                for val in numeric_values
            ]
            for record, payload in zip(records, payloads):
                record[column_name] = payload
        else:
            for record, value in zip(records, column_values):
                record[column_name] = value if not pd.isna(value) else None

    return records


def target_records_from_frame(
    df_data: pd.DataFrame,
    units_map: dict,
    project_name: str,
) -> list:
    """Convert one summary table into structured target records."""
    filtered = _filtered_summary_frame(df_data, project_name=project_name)
    if filtered.empty:
        return []

    hot_cols, cold_cols, other_cols = _partition_summary_columns(filtered.columns)
    clean_df = filtered.replace({pd.NA: None}).where(~filtered.isna(), None)
    base_records = clean_df[other_cols].to_dict(orient="records")

    results = []
    for row_index, record in enumerate(base_records):
        entry = {
            "hot_utilities": _utility_records_from_row(
                clean_df,
                row_index,
                hot_cols,
                units_map=units_map,
            ),
            "cold_utilities": _utility_records_from_row(
                clean_df,
                row_index,
                cold_cols,
                units_map=units_map,
            ),
        }
        for column_name, value in record.items():
            unit = units_map.get(column_name)
            entry[column_name] = _summary_field_payload(
                column_name,
                value,
                unit=unit,
                project_name=project_name,
                record_name=str(record.get("name") or ""),
            )
        items = list(entry.items())
        results.append(dict(items[2:] + items[:2]))

    return results


def _filtered_summary_frame(
    df_data: pd.DataFrame,
    *,
    project_name: str,
) -> pd.DataFrame:
    if df_data.empty:
        return df_data

    name_series = df_data["name"]
    mask_valid = name_series.map(lambda value: isinstance(value, str))
    mask_skip = name_series.eq("Individual Process Targets")
    filtered = df_data.loc[mask_valid & ~mask_skip].reset_index(drop=True).copy()
    if filtered.empty:
        return filtered

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
    return filtered


def _partition_summary_columns(columns) -> tuple[list[str], list[str], list[str]]:
    hot_cols = [col for col in columns if str(col).startswith("HU::")]
    cold_cols = [col for col in columns if str(col).startswith("CU::")]
    other_cols = [col for col in columns if col not in hot_cols + cold_cols]
    return hot_cols, cold_cols, other_cols


def _utility_records_from_row(
    frame: pd.DataFrame,
    row_index: int,
    columns: list[str],
    *,
    units_map: dict,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column_name in columns:
        value = frame.iloc[row_index][column_name]
        numeric_value = None if value is None or pd.isna(value) else float(value)
        records.append(
            {
                "name": str(column_name)[4:],
                "heat_flow": {
                    "value": numeric_value,
                    "units": units_map.get(column_name),
                },
            }
        )
    return records


def _summary_field_payload(
    column_name: str,
    value: Any,
    *,
    unit: Optional[str],
    project_name: str,
    record_name: str,
) -> Any:
    if column_name == "temp_pinch":
        return _temp_pinch_payload(
            value,
            unit=unit,
            is_total_process_target=(
                record_name == f"{project_name}/Total Process Target"
            ),
        )
    if column_name == "degree_of_integration":
        if value is None or pd.isna(value):
            return {"value": None, "units": unit}
        return {"value": float(value) * 100, "units": unit}
    if pd.api.types.is_number(value) and isinstance(unit, str) and unit:
        return {"value": float(value), "units": unit}
    return value if value is not None else None


def _temp_pinch_payload(
    value: Any,
    *,
    unit: Optional[str],
    is_total_process_target: bool,
) -> dict[str, dict[str, Any]]:
    cold_temp, hot_temp = _parse_temp_pinch(value)
    if is_total_process_target:
        cold_temp = None
        hot_temp = None
    return {
        "cold_temp": {"value": cold_temp, "units": unit},
        "hot_temp": {"value": hot_temp, "units": unit},
    }


def _parse_temp_pinch(value: Any) -> tuple[Optional[float], Optional[float]]:
    if value in (None, ""):
        return None, None
    parts = [part.strip() for part in str(value).split(";")]
    if len(parts) != 2:
        raise ValueError(f"Invalid temp_pinch value: {value!r}")
    cold_temp = None if parts[0] == "" else float(parts[0])
    hot_temp = None if parts[1] == "" else float(parts[1])
    return cold_temp, hot_temp
