import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import pandas as pd

#######################################################################################################
# Public API
#######################################################################################################


def export_target_summary_to_excel_with_units(
    target_response, out_dir: str = "."
) -> str:
    """
    Export TargetOutput to an Excel workbook with values and units.
      Sheet 'Summary'       : One row per TargetResults with value/unit columns
      Sheet 'PinchTemps'    : Hot/Cold pinch temps per target
      Sheet 'Utilities'     : Flattened hot & cold utilities (value+unit)

    Returns the path to the saved workbook.
    """
    # -------- Summary sheet (values + units) --------
    rows = []

    for t in target_response.targets:
        cold_val, cold_unit = _split_vu(getattr(t.temp_pinch, "cold_temp", None))
        hot_val, hot_unit = _split_vu(getattr(t.temp_pinch, "hot_temp", None))

        Qh_val, Qh_unit = _split_vu(t.Qh)
        Qc_val, Qc_unit = _split_vu(t.Qc)
        Qr_val, Qr_unit = _split_vu(t.Qr)

        deg_val, deg_unit = _split_vu(t.degree_of_integration)

        row_front = {
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

        row_mid = {}

        def _emit_utils(utils: Iterable):
            """Populate summary row with value/unit pairs for each utility."""
            for u in utils or []:
                hf_val, hf_unit = _split_vu(u.heat_flow)
                row_mid[u.name + " (value)"] = hf_val
                row_mid[u.name + " (unit)"] = hf_unit

        _emit_utils(t.hot_utilities)
        _emit_utils(t.cold_utilities)

        util_cost_val, util_cost_unit = _split_vu(t.utility_cost)
        area_val, area_unit = _split_vu(t.area)

        work_val, work_unit = _split_vu(t.work_target)
        turb_eff_val, turb_eff_unit = _split_vu(t.turbine_efficiency_target)

        ex_src_val, ex_src_unit = _split_vu(t.exergy_sources)
        ex_sink_val, ex_sink_unit = _split_vu(t.exergy_sinks)
        ex_req_val, ex_req_unit = _split_vu(t.exergy_req_min)
        ex_des_val, ex_des_unit = _split_vu(t.exergy_des_min)

        row_end = {
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
        rows.append(row_front | row_mid | row_end)

    df_summary = pd.DataFrame(rows)

    # -------- File name: <project name>_<YYYYmmdd_HHMMSS>.xlsx --------
    project = _safe_name(getattr(target_response, "name", "Project"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{project}_{timestamp}.xlsx"
    out_path = Path(out_dir) / filename

    # -------- Write workbook --------
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        # Summary
        df_summary.to_excel(xw, sheet_name="Summary", index=False)
        _autosize_columns(df_summary, xw.sheets["Summary"])

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


def _autosize_columns(df: pd.DataFrame, ws):
    """Best-effort column width:  max(len(header), max len cell)."""
    for i, col in enumerate(df.columns, start=1):
        max_len = len(str(col))
        for val in df[col].astype(str):
            max_len = max(max_len, len(val))
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(
            max_len + 2, 40
        )


def _safe_name(name: str) -> str:
    """Make a filesystem-safe project name (keep letters, numbers, - _ .)."""
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)  # replace forbidden characters
    name = re.sub(r"\s+", "_", name)  # spaces -> underscore
    return name or "Project"
