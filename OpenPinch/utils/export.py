import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from ..classes import Zone
    from ..lib import TargetOutput

#######################################################################################################
# Public API
#######################################################################################################


def export_target_summary_to_excel_with_units(
    target_response: "TargetOutput", 
    master_zone: "Zone", 
    out_dir: str = ".",
) -> str:
    """
    Export TargetOutput to an Excel workbook with values and units.
      Sheet 'Summary'       : One row per TargetResults with value/unit columns
      Sheet 'PinchTemps'    : Hot/Cold pinch temps per target
      Sheet 'Utilities'     : Flattened hot & cold utilities (value+unit)

    Returns the path to the saved workbook.
    """
    df_summary = _build_summary_dataframe(target_response.targets)

    out_path = _compose_output_path(
        project_name=getattr(target_response, "name", "Project"),
        out_dir=out_dir,
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        _write_summary_sheet(df_summary, xw)
        _write_problem_tables(master_zone, xw)

    return str(out_path)


#######################################################################################################
# Helpers
#######################################################################################################


def _split_vu(x: Any) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, units) for either a float or ValueWithUnit/None."""
    if x is None:
        return None, None
    # If it's a pydantic model with attributes
    if hasattr(x, "value") and hasattr(x, "units"):
        return x.value, x.units
    # plain number
    try:
        return float(x), None
    except Exception:
        return None, None


def _autosize_columns(df: pd.DataFrame, ws, start_col: int = 1, header_row: int = 1):
    """Best-effort column width:  max(len(header), max len cell)."""
    for i, col in enumerate(df.columns, start=start_col):
        max_len = len(str(col))
        for val in df[col].astype(str):
            max_len = max(max_len, len(val))
        ws.column_dimensions[ws.cell(row=header_row, column=i).column_letter].width = min(
            max_len + 2, 40
        )


def _safe_name(name: str) -> str:
    """Make a filesystem-safe project name (keep letters, numbers, - _ .)."""
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)  # replace forbidden characters
    name = re.sub(r"\s+", "_", name)  # spaces -> underscore
    return name or "Project"


def _build_summary_dataframe(targets) -> pd.DataFrame:
    """Convert TargetResults objects into a tabular dataframe with value/unit columns."""
    rows = []
    for target in targets:
        rows.append(_make_summary_row(target))
    return pd.DataFrame(rows)


def _make_summary_row(t) -> dict:
    cold_val, cold_unit = _split_vu(getattr(t.temp_pinch, "cold_temp", None))
    hot_val, hot_unit = _split_vu(getattr(t.temp_pinch, "hot_temp", None))

    Qh_val, Qh_unit = _split_vu(t.Qh)
    Qc_val, Qc_unit = _split_vu(t.Qc)
    Qr_val, Qr_unit = _split_vu(t.Qr)

    deg_val, deg_unit = _split_vu(t.degree_of_integration)

    base_columns = {
        "Target": t.name,
        "Cold Pinch (value)": cold_val,
        "Cold Pinch (unit)": cold_unit,
        "Hot Pinch (value)": hot_val,
        "Hot Pinch (unit)": hot_unit,
        "Qh (value)": Qh_val,
        "Qh (unit)": Qh_unit,
        "Qc (value)": Qc_val,
        "Qc (unit)": Qc_unit,
        "Qr (value)": Qr_val,
        "Qr (unit)": Qr_unit,
        "Degree of Integration (value)": deg_val,
        "Degree of Integration (unit)": deg_unit,
    }

    utility_columns = _utility_columns(t.hot_utilities, t.cold_utilities)

    util_cost_val, util_cost_unit = _split_vu(t.utility_cost)
    area_val, area_unit = _split_vu(t.area)

    work_val, work_unit = _split_vu(t.work_target)
    turb_eff_val, turb_eff_unit = _split_vu(t.turbine_efficiency_target)

    ex_src_val, ex_src_unit = _split_vu(t.exergy_sources)
    ex_sink_val, ex_sink_unit = _split_vu(t.exergy_sinks)
    ex_req_val, ex_req_unit = _split_vu(t.exergy_req_min)
    ex_des_val, ex_des_unit = _split_vu(t.exergy_des_min)

    tail_columns = {
        "Utility Cost (value)": util_cost_val,
        "Utility Cost (unit)": util_cost_unit,
        "Area (value)": area_val,
        "Area (unit)": area_unit,
        "Num Units": t.num_units,
        "Capital Cost": t.capital_cost,
        "Total Cost": t.total_cost,
        "Work Target (value)": work_val,
        "Work Target (unit)": work_unit,
        "Turbine Eff Target (value)": turb_eff_val,
        "Turbine Eff Target (unit)": turb_eff_unit,
        "ETE": t.ETE,
        "Exergy Sources (value)": ex_src_val,
        "Exergy Sources (unit)": ex_src_unit,
        "Exergy Sinks (value)": ex_sink_val,
        "Exergy Sinks (unit)": ex_sink_unit,
        "Exergy Req Min (value)": ex_req_val,
        "Exergy Req Min (unit)": ex_req_unit,
        "Exergy Des Min (value)": ex_des_val,
        "Exergy Des Min (unit)": ex_des_unit,
    }

    return base_columns | utility_columns | tail_columns


def _utility_columns(hot_utils: Optional[Iterable], cold_utils: Optional[Iterable]) -> dict:
    """Return flattened value/unit columns for the provided utilities."""
    columns: dict[str, Any] = {}

    def emit(utils):
        for u in utils or []:
            hf_val, hf_unit = _split_vu(u.heat_flow)
            columns[f"{u.name} (value)"] = hf_val
            columns[f"{u.name} (unit)"] = hf_unit

    emit(hot_utils)
    emit(cold_utils)
    return columns


def _compose_output_path(project_name: str, out_dir: str) -> Path:
    project = _safe_name(project_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{project}_{timestamp}.xlsx"
    return Path(out_dir) / filename


def _write_summary_sheet(df_summary: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    _autosize_columns(df_summary, writer.sheets["Summary"])


def _write_problem_tables(master_zone: Optional["Zone"], writer: pd.ExcelWriter) -> None:
    """Emit shifted/real problem tables for master zone and all subzones."""
    if master_zone is None:
        return

    from ..streamlit_webviewer.web_graphing import problem_table_to_dataframe

    used_sheet_names: set[str] = set()

    for zone in _iter_zones(master_zone):
        for target_name, target in zone.targets.items():
            table_specs = (
                (f"{zone.name} - {target_name} (Shifted)", getattr(target, "pt", None)),
                (f"{zone.name} - {target_name} (Real)", getattr(target, "pt_real", None)),
            )
            for sheet_label, table in table_specs:
                df = problem_table_to_dataframe(table, round_decimals=2)
                if df.empty:
                    continue
                sheet_name = _unique_sheet_name(sheet_label, used_sheet_names)
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    startcol=0,
                    startrow=2,
                )
                ws = writer.sheets[sheet_name]
                ws["A1"] = target.name
                _autosize_columns(df, ws, start_col=1, header_row=3)


def _iter_zones(zone: "Zone"):
    """Yield ``zone`` and all nested subzones depth-first."""
    stack = [zone]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.subzones.values())


def _unique_sheet_name(base: str, used: set[str]) -> str:
    """Return an Excel-safe, unique sheet name capped at 31 chars."""
    cleaned = _sanitize_sheet_name(base)
    candidate = cleaned[:31] or "Sheet"
    if candidate not in used:
        used.add(candidate)
        return candidate

    for idx in range(2, 1000):
        suffix = f" ({idx})"
        trimmed = candidate[: 31 - len(suffix)] if len(candidate) + len(suffix) > 31 else candidate
        alt = f"{trimmed}{suffix}"
        if alt not in used:
            used.add(alt)
            return alt

    raise ValueError("Unable to allocate unique sheet name.")


def _sanitize_sheet_name(name: str) -> str:
    """Replace characters Excel forbids in sheet names and strip trailing apostrophes."""
    cleaned = re.sub(r"[:/?*\\\[\]]", "_", name).strip().rstrip("'")
    return cleaned or "Sheet"
