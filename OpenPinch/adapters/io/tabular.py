"""Shared tabular parsing helpers for workbook and CSV ingestion."""

from __future__ import annotations

import re
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from ...domain.enums import IntegrationType, TargetMethod, ZoneType


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
        col_names = ["name", "pinch_temp", "Qh", "Qc", "Qr", "degree_of_integration"]
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
    # Exponents such as "m2" are handled by the unit normaliser, which only
    # treats a "2" after a letter as a power; a blind replace broke "kW/m^2/K".
    return col_names, col_units


# Header spellings accepted for each column, compared after lower-casing,
# reading a Greek delta as "d", dropping any bracketed unit and removing
# everything but letters and digits.
_HEADER_ALIASES = {
    "Stream Data": {
        "zone": ("process zone", "zone name"),
        "name": ("stream", "stream name"),
        "t_supply": ("ts", "tsupply", "supply temperature", "tin"),
        "t_target": ("tt", "ttarget", "target temperature", "tout"),
        "heat_flow": ("dh", "heat load", "duty", "q"),
        "heat_capacity_flowrate": ("cp", "mcp", "fcp", "heat capacity flowrate"),
        "dt_cont": ("dtcont", "dt contribution"),
        "htc": ("heat transfer coefficient",),
    },
    "Utility Data": {
        "name": ("utility", "utility name"),
        "type": ("utility type",),
        "t_supply": ("ts", "tsupply", "supply temperature", "tin"),
        "t_target": ("tt", "ttarget", "target temperature", "tout"),
        "dt_cont": ("dtcont", "dt contribution"),
        "price": ("cost", "unit price"),
        "htc": ("heat transfer coefficient",),
        "heat_flow": ("dh", "heat load", "duty", "q"),
    },
}


def _header_key(text: str) -> str:
    key = re.sub(r"\(.*?\)|\[.*?\]", "", text).lower().replace("δ", "d")
    return re.sub(r"[^a-z0-9]", "", key)


def _map_header_row(header: list, sheet_name: str) -> Optional[list]:
    """Return the field name for each column, or None when no header is known."""
    aliases = {
        _header_key(alias): field
        for field, names in _HEADER_ALIASES[sheet_name].items()
        for alias in (field, *names)
    }
    labels = [
        cell.strip() if isinstance(cell, str) and cell.strip() else None
        for cell in header
    ]
    fields = [aliases.get(_header_key(label)) if label else None for label in labels]
    if not any(fields):
        return None
    for field in set(filter(None, fields)):
        columns = [label for label, f in zip(labels, fields) if f == field]
        if len(columns) > 1:
            raise ValueError(
                f"{sheet_name} has more than one column for {field!r}: "
                f"{', '.join(repr(column) for column in columns)}."
            )
    unknown = [label for label, f in zip(labels, fields) if label and not f]
    if unknown:
        warnings.warn(
            f"{sheet_name} column(s) {', '.join(repr(u) for u in unknown)} "
            "are not recognised and were ignored.",
            UserWarning,
            stacklevel=4,
        )
    return fields


def problem_frame_from_sheet(
    df_full: pd.DataFrame,
    sheet_name: str,
    row_units: int,
    row_data: int,
) -> tuple[pd.DataFrame, dict]:
    """Select a stream or utility table's columns and their units.

    Columns are matched by the header text in the first row. A header with no
    recognised column name falls back to the fixed column order.
    """
    col_names, _ = get_column_names_and_units(df_full, sheet_name, row_units)
    raw_units = [
        unit.replace("°", "deg") if isinstance(unit, str) else None
        for unit in df_full.iloc[row_units].tolist()
    ]
    fields = _map_header_row(df_full.iloc[0].tolist(), sheet_name)
    if fields is None:
        warnings.warn(
            f"{sheet_name} header row has no recognised column names; columns "
            f"are read in the order {', '.join(col_names)}.",
            UserWarning,
            stacklevel=3,
        )
        count = min(len(df_full.columns), len(col_names))
        fields = col_names[:count] + [None] * (len(df_full.columns) - count)
    positions = [index for index, field in enumerate(fields) if field]
    df_data = df_full.iloc[row_data:, positions].copy()
    df_data.columns = [fields[index] for index in positions]
    units_map = {fields[index]: raw_units[index] for index in positions}
    return df_data, units_map


def summary_frame_from_sheet(
    df_full: pd.DataFrame,
    row_units: int,
    row_data: int,
) -> tuple[pd.DataFrame, dict]:
    """Name a summary table's columns by position and collect their units."""
    col_names, col_units = get_column_names_and_units(df_full, "Summary", row_units)
    df_data = df_full.iloc[row_data:, : len(col_names)].copy()
    df_data.columns = col_names[: len(df_data.columns)]
    return df_data, dict(zip(col_names, col_units))


def problem_records_from_frame(df_data: pd.DataFrame, units_map: dict) -> list:
    """Convert stream/utility tables into JSON-ready row dictionaries with units."""
    if df_data.empty:
        return []

    clean_df = df_data.replace({pd.NA: None}).reset_index(drop=True)
    # Blank separator rows (empty or whitespace-only cells) carry no data; drop
    # them before numeric cells are wrapped as {"value": None, ...} records.
    clean_df = clean_df.loc[~clean_df.map(_is_blank_cell).all(axis=1)]
    if clean_df.empty:
        return []
    records = clean_df.to_dict(orient="records")

    for column_name, unit in units_map.items():
        if column_name not in clean_df.columns:
            continue

        column_values = clean_df[column_name]
        if isinstance(unit, str) and unit.strip():
            numeric_values = pd.to_numeric(column_values, errors="coerce")
            missing = column_values.isna() | column_values.map(
                lambda value: isinstance(value, str) and not value.strip()
            )
            invalid = numeric_values.isna() & ~missing
            invalid |= column_values.map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            if invalid.any():
                row = invalid[invalid].index[0]
                raise ValueError(
                    f"Column {column_name!r} has a non-numeric value at "
                    f"data row {row + 1}: {column_values.loc[row]!r}."
                )
            value_entries = [
                {"value": (None if pd.isna(val) else float(val)), "unit": unit}
                for val in numeric_values
            ]
            for record, value_entry in zip(records, value_entries):
                record[column_name] = value_entry
        else:
            for record, value in zip(records, column_values):
                record[column_name] = value if not pd.isna(value) else None

    return records


def _is_blank_cell(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    return value is None or bool(pd.isna(value))


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
            entry[column_name] = _summary_field_data(
                column_name,
                value,
                unit=unit,
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
    has_total_site = name_series.eq("Total Site Targets").any()
    mask_skip = name_series.eq("Individual Process Targets")
    if has_total_site:
        mask_skip |= name_series.eq("Total Process Targets")
    filtered = df_data.loc[mask_valid & ~mask_skip].reset_index(drop=True).copy()
    if filtered.empty:
        return filtered

    original_names = filtered["name"].copy()
    total_names = {
        "Total Site Targets",
        "Total Process Targets",
        "Total Integrated Targets",
    }
    mask_total = original_names.isin(total_names)
    mask_utility = original_names.isin(["Total Site Targets", "Total Process Targets"])
    filtered["scope"] = original_names.astype(str)
    filtered.loc[mask_total, "scope"] = project_name
    filtered.loc[~mask_total, "scope"] = (
        project_name + "/" + original_names[~mask_total].astype(str)
    )
    filtered["zone_type"] = ZoneType.P.value
    filtered.loc[mask_total, "zone_type"] = ZoneType.S.value
    filtered["integration_type"] = IntegrationType.Process.value
    filtered.loc[mask_utility, "integration_type"] = IntegrationType.Utility.value
    filtered["target_method"] = TargetMethod.HeatExchange.value
    filtered = filtered.drop(columns=["name"])
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
                    "unit": units_map.get(column_name),
                },
            }
        )
    return records


def _summary_field_data(
    column_name: str,
    value: Any,
    *,
    unit: Optional[str],
) -> Any:
    if column_name == "pinch_temp":
        return _temp_pinch_data(
            value,
            unit=unit,
            is_total_process_target=False,
        )
    if column_name == "degree_of_integration":
        if value is None or pd.isna(value):
            return {"value": None, "unit": unit}
        return {"value": float(value) * 100, "unit": unit}
    if pd.api.types.is_number(value) and isinstance(unit, str) and unit:
        return {"value": float(value), "unit": unit}
    return value if value is not None else None


def _temp_pinch_data(
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
        "cold_temp": {"value": cold_temp, "unit": unit},
        "hot_temp": {"value": hot_temp, "unit": unit},
    }


def _parse_temp_pinch(value: Any) -> tuple[Optional[float], Optional[float]]:
    if value in (None, ""):
        return None, None
    parts = [part.strip() for part in str(value).split(";")]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    elif len(parts) != 2:
        raise ValueError(f"Invalid pinch_temp value: {value!r}")
    cold_temp = None if parts[0] == "" else float(parts[0])
    hot_temp = None if parts[1] == "" else float(parts[1])
    return cold_temp, hot_temp
