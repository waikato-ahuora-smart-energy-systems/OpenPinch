"""Excel export utilities for OpenPinch targeting outputs."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import pandas as pd

from ..lib.schemas.report_units import split_report_value
from ..streamlit_webviewer.web_graphing import problem_table_to_dataframe

if TYPE_CHECKING:
    from ..classes.zone import Zone
    from ..lib.schemas.io import TargetOutput

__all__ = ["export_target_summary_to_excel_with_units"]

################################################################################
# Public API
################################################################################


def export_target_summary_to_excel_with_units(
    target_response: "TargetOutput",
    master_zone: "Zone",
    out_dir: str = ".",
) -> str:
    """Export solved targets and problem tables to an Excel workbook.

    Parameters
    ----------
    target_response:
        Structured response returned by the high-level targeting service.
    master_zone:
        Solved zone hierarchy used to export shifted and real problem tables for
        the master zone and all subzones. May be ``None`` when only the summary
        sheet is required.
    out_dir:
        Destination directory for the workbook.

    Returns
    -------
    str
        Absolute or relative path to the workbook that was written.

    Notes
    -----
    The workbook currently includes a summary sheet plus one or more problem-
    table sheets. Value-with-unit objects are flattened into adjacent
    ``(value)`` and ``(unit)`` columns for easy review in Excel.
    """
    df_summary = build_summary_dataframe(target_response.targets)

    out_path = _compose_output_path(
        project_name=getattr(target_response, "name", "Project"),
        out_dir=out_dir,
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        _write_summary_sheet(df_summary, xw)
        _write_problem_tables(master_zone, xw)

    return str(out_path)


def build_summary_dataframe(targets) -> pd.DataFrame:
    """Convert ``TargetResults`` objects into a value/unit dataframe."""
    rows = []
    for target in targets:
        rows.append(_make_summary_row(target))
    return pd.DataFrame(rows)


################################################################################
# Helpers
################################################################################


def _value_unit_columns(
    label: str,
    value: Any,
    *,
    idx: int | None = None,
) -> dict[str, Any]:
    resolved_value, resolved_unit = split_report_value(value, idx=idx)
    return {
        f"{label} (value)": resolved_value,
        f"{label} (unit)": resolved_unit,
    }


def _autosize_columns(df: pd.DataFrame, ws, start_col: int = 1, header_row: int = 1):
    """Best-effort column width:  max(len(header), max len cell)."""
    for offset, col in enumerate(df.columns):
        i = start_col + offset
        max_len = len(str(col))
        column_values = df.iloc[:, offset]
        for val in column_values:
            text = "" if pd.isna(val) else str(val)
            max_len = max(max_len, len(text))
        ws.column_dimensions[
            ws.cell(row=header_row, column=i).column_letter
        ].width = min(max_len + 2, 40)


def _safe_name(name: str) -> str:
    """Make a filesystem-safe project name (keep letters, numbers, - _ .)."""
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)  # replace forbidden characters
    name = re.sub(r"\s+", "_", name)  # spaces -> underscore
    return name or "Project"


def _make_summary_row(t) -> dict:
    state_id = getattr(t, "state_id", None)
    idx = getattr(t, "idx", None)
    base_columns = {
        "Target": t.name,
        "State ID": state_id,
        **_value_unit_columns(
            "Cold Pinch",
            getattr(t.pinch_temp, "cold_temp", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "Hot Pinch",
            getattr(t.pinch_temp, "hot_temp", None),
            idx=idx,
        ),
        **_value_unit_columns("Qh", t.Qh, idx=idx),
        **_value_unit_columns("Qc", t.Qc, idx=idx),
        **_value_unit_columns("Qr", t.Qr, idx=idx),
        **_value_unit_columns(
            "Degree of Integration",
            t.degree_of_integration,
            idx=idx,
        ),
    }

    utility_columns = _utility_columns(
        t.hot_utilities,
        t.cold_utilities,
        idx=idx,
    )

    tail_columns = {
        **_value_unit_columns("Utility Cost", t.utility_cost, idx=idx),
        **_value_unit_columns("Area", t.area, idx=idx),
        "Num Units": t.num_units,
        **_value_unit_columns("Capital Cost", t.capital_cost, idx=idx),
        **_value_unit_columns("Total Cost", t.total_cost, idx=idx),
        **_value_unit_columns("Work Target", t.work_target, idx=idx),
        **_value_unit_columns(
            "Process Component Work",
            getattr(t, "process_component_work_target", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "Turbine Eff Target",
            t.turbine_efficiency_target,
            idx=idx,
        ),
        **_value_unit_columns("ETE", t.ETE, idx=idx),
        **_value_unit_columns("Exergy Sources", t.exergy_sources, idx=idx),
        **_value_unit_columns("Exergy Sinks", t.exergy_sinks, idx=idx),
        **_value_unit_columns("Exergy Req Min", t.exergy_req_min, idx=idx),
        **_value_unit_columns("Exergy Des Min", t.exergy_des_min, idx=idx),
        "HPR Cycle": getattr(t, "hpr_cycle", None),
        "HPR Success": getattr(t, "hpr_success", None),
        **_value_unit_columns(
            "HPR Utility Total",
            getattr(t, "hpr_utility_total", None),
            idx=idx,
        ),
        **_value_unit_columns("HPR Work", getattr(t, "hpr_work", None), idx=idx),
        **_value_unit_columns(
            "HPR External Utility",
            getattr(t, "hpr_external_utility", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Ambient Hot",
            getattr(t, "hpr_ambient_hot", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Ambient Cold",
            getattr(t, "hpr_ambient_cold", None),
            idx=idx,
        ),
        **_value_unit_columns("HPR COP", getattr(t, "hpr_cop", None), idx=idx),
        **_value_unit_columns("HPR Eta HE", getattr(t, "hpr_eta_he", None), idx=idx),
        **_value_unit_columns(
            "HPR Operating Cost",
            getattr(t, "hpr_operating_cost", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Capital Cost",
            getattr(t, "hpr_capital_cost", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Annualized Capital Cost",
            getattr(t, "hpr_annualized_capital_cost", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Total Annualized Cost",
            getattr(t, "hpr_total_annualized_cost", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Compressor Capital Cost",
            getattr(t, "hpr_compressor_capital_cost", None),
            idx=idx,
        ),
        **_value_unit_columns(
            "HPR Heat Exchanger Capital Cost",
            getattr(t, "hpr_heat_exchanger_capital_cost", None),
            idx=idx,
        ),
    }

    return base_columns | utility_columns | tail_columns


def _utility_columns(
    hot_utils: Optional[Iterable],
    cold_utils: Optional[Iterable],
    *,
    idx: int | None = None,
) -> dict:
    """Return flattened value/unit columns for the provided utilities."""
    columns: dict[str, Any] = {}

    def emit(utils):
        for u in utils or []:
            hf_val, hf_unit = split_report_value(u.heat_flow, idx=idx)
            columns[f"{u.name} (value)"] = hf_val
            columns[f"{u.name} (unit)"] = hf_unit

    emit(hot_utils)
    emit(cold_utils)
    return columns


def _compose_output_path(project_name: str, out_dir: str) -> Path:
    project = _safe_name(project_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{project}_{timestamp}.xlsx"
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _write_summary_sheet(df_summary: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    _autosize_columns(df_summary, writer.sheets["Summary"])


def _write_problem_tables(
    master_zone: Optional["Zone"], writer: pd.ExcelWriter
) -> None:
    """Emit shifted and real temperature Problem Tables for every solved zone."""
    if master_zone is None:
        return

    used_sheet_names: set[str] = set()

    for zone in _iter_zones(master_zone):
        for target_name, target in zone.targets.items():
            table_specs = (
                (f"{zone.name} - {target_name} (Shifted)", getattr(target, "pt", None)),
                (
                    f"{zone.name} - {target_name} (Real)",
                    getattr(target, "pt_real", None),
                ),
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
        trimmed = (
            candidate[: 31 - len(suffix)]
            if len(candidate) + len(suffix) > 31
            else candidate
        )
        alt = f"{trimmed}{suffix}"
        if alt not in used:
            used.add(alt)
            return alt

    raise ValueError("Unable to allocate unique sheet name.")


def _sanitize_sheet_name(name: str) -> str:
    """Replace Excel-forbidden sheet-name characters and trailing apostrophes."""
    cleaned = re.sub(r"[:/?*\\\[\]]", "_", name).strip().rstrip("'")
    return cleaned or "Sheet"
