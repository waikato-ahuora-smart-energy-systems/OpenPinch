"""Excel export utilities for OpenPinch targeting outputs."""

from __future__ import annotations

import os
import re
from contextlib import suppress
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import pandas as pd

from ...contracts.report_units import split_report_value
from .problem_table import problem_table_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...contracts.output import TargetOutput
    from ...domain.zone import Zone

__all__ = ["export_target_summary_to_excel_with_units"]

################################################################################
# Public API
################################################################################


def export_target_summary_to_excel_with_units(
    target_response: "TargetOutput",
    master_zone: "Zone",
    out_dir: str = ".",
    *,
    period_zones: Mapping[str, Zone] | None = None,
) -> Path:
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
        Destination directory for the workbook, or a path ending in ``.xlsx``
        to write that file.
    period_zones:
        For all-period exports, each period's solved zone hierarchy. Each
        period's problem tables are written with the period id in the sheet
        name, instead of the tables of ``master_zone``.

    Returns
    -------
    pathlib.Path
        Absolute or relative path to the workbook that was written.

    Notes
    -----
    The workbook currently includes a summary sheet plus one or more problem-
    table sheets. Value-with-unit objects are flattened into adjacent
    ``(value)`` and ``(unit)`` columns for easy review in Excel.
    """
    df_summary = build_summary_dataframe(target_response.targets)

    out_path = _reserve_workbook_path(
        project_name=getattr(target_response, "name", "Project"),
        out_dir=out_dir,
    )

    writer: pd.ExcelWriter | None = None
    completed = False
    try:
        writer = pd.ExcelWriter(out_path, engine="openpyxl")
        xw = writer
        _write_summary_sheet(df_summary, xw)
        if period_zones:
            used_sheet_names: set[str] = set()
            for period_id, zone in period_zones.items():
                _write_problem_tables(
                    zone, xw, prefix=f"[{period_id}] ", used=used_sheet_names
                )
        else:
            _write_problem_tables(master_zone, xw)
        writer.close()
        completed = True
    finally:
        if not completed:
            if writer is not None:
                with suppress(Exception):
                    writer._handles.close()
            out_path.unlink(missing_ok=True)

    return out_path


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
    period_idx: int | None = None,
) -> dict[str, Any]:
    resolved_value, resolved_unit = split_report_value(value, period_idx=period_idx)
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
    period_id = getattr(t, "period_id", None)
    period_idx = getattr(t, "period_idx", None)
    base_columns = {
        "Scope": t.scope,
        "Zone Type": t.zone_type,
        "Integration Type": t.integration_type,
        "Target Method": t.target_method,
        "Period ID": period_id,
        **_value_unit_columns(
            "Cold Pinch",
            getattr(t.pinch_temp, "cold_temp", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "Hot Pinch",
            getattr(t.pinch_temp, "hot_temp", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns("Qh", t.Qh, period_idx=period_idx),
        **_value_unit_columns("Qc", t.Qc, period_idx=period_idx),
        **_value_unit_columns("Qr", t.Qr, period_idx=period_idx),
        **_value_unit_columns(
            "Degree of Integration",
            t.degree_of_integration,
            period_idx=period_idx,
        ),
    }

    utility_columns = _utility_columns(
        t.hot_utilities,
        t.cold_utilities,
        period_idx=period_idx,
    )

    tail_columns = {
        **_value_unit_columns("Utility Cost", t.utility_cost, period_idx=period_idx),
        **_value_unit_columns("Area", t.area, period_idx=period_idx),
        "Num Units": t.num_units,
        **_value_unit_columns("Capital Cost", t.capital_cost, period_idx=period_idx),
        **_value_unit_columns("Total Cost", t.total_cost, period_idx=period_idx),
        **_value_unit_columns("Work Target", t.work_target, period_idx=period_idx),
        **_value_unit_columns(
            "Process Component Work",
            getattr(t, "process_component_work_target", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "Turbine Eff Target",
            t.turbine_efficiency_target,
            period_idx=period_idx,
        ),
        **_value_unit_columns("ETE", t.ETE, period_idx=period_idx),
        **_value_unit_columns(
            "Exergy Sources", t.exergy_sources, period_idx=period_idx
        ),
        **_value_unit_columns("Exergy Sinks", t.exergy_sinks, period_idx=period_idx),
        **_value_unit_columns(
            "Exergy Req Min", t.exergy_req_min, period_idx=period_idx
        ),
        **_value_unit_columns(
            "Exergy Des Min", t.exergy_des_min, period_idx=period_idx
        ),
        "HPR Cycle": getattr(t, "hpr_cycle", None),
        "HPR Success": getattr(t, "hpr_success", None),
        **_value_unit_columns(
            "HPR Utility Total",
            getattr(t, "hpr_utility_total", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Work", getattr(t, "hpr_work", None), period_idx=period_idx
        ),
        **_value_unit_columns(
            "HPR External Utility",
            getattr(t, "hpr_external_utility", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Ambient Hot",
            getattr(t, "hpr_ambient_hot", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Ambient Cold",
            getattr(t, "hpr_ambient_cold", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR COP", getattr(t, "hpr_cop", None), period_idx=period_idx
        ),
        **_value_unit_columns(
            "HPR Eta HE", getattr(t, "hpr_eta_he", None), period_idx=period_idx
        ),
        **_value_unit_columns(
            "HPR Operating Cost",
            getattr(t, "hpr_operating_cost", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Capital Cost",
            getattr(t, "hpr_capital_cost", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Annualized Capital Cost",
            getattr(t, "hpr_annualized_capital_cost", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Hot Utility Annualized Capital Cost",
            getattr(t, "hpr_hot_utility_annualized_capital_cost", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Refrigeration Annualized Capital Cost",
            getattr(t, "hpr_refrigeration_annualized_capital_cost", None),
            period_idx=period_idx,
        ),
        **_value_unit_columns(
            "HPR Utility Annualized Capital Cost",
            getattr(t, "hpr_utility_annualized_capital_cost", None),
            period_idx=period_idx,
        ),
        "HPR Machine Capital Costs ($)": _joined(
            getattr(t, "hpr_machine_capital_costs", None)
        ),
        "HPR Machine Annualized Capital Costs ($/y)": _joined(
            getattr(t, "hpr_machine_annualized_capital_costs", None)
        ),
        **_value_unit_columns(
            "HPR Total Annualized Cost",
            getattr(t, "hpr_total_annualized_cost", None),
            period_idx=period_idx,
        ),
    }

    return base_columns | utility_columns | tail_columns


def _joined(values) -> str | None:
    """Render a per-machine tuple as one cell, e.g. ``"1200; 800"``."""
    if not values:
        return None
    return "; ".join(f"{float(value):.6g}" for value in values)


def _utility_columns(
    hot_utils: Optional[Iterable],
    cold_utils: Optional[Iterable],
    *,
    period_idx: int | None = None,
) -> dict:
    """Return flattened value/unit columns for the provided utilities."""
    columns: dict[str, Any] = {}

    # Prefix with the side, so a utility named like a built-in column (or a
    # hot and a cold utility sharing a name) never overwrites another column.
    def emit(utils, side):
        for u in utils or []:
            hf_val, hf_unit = split_report_value(u.heat_flow, period_idx=period_idx)
            columns[f"{side}: {u.name} (value)"] = hf_val
            columns[f"{side}: {u.name} (unit)"] = hf_unit

    emit(hot_utils, "HU")
    emit(cold_utils, "CU")
    return columns


def _reserve_workbook_path(project_name: str, out_dir: str | Path) -> Path:
    """Atomically reserve a unique workbook path for one export invocation."""
    requested = Path(out_dir)
    if requested.suffix.lower() == ".xlsx":
        # An explicit workbook file name is written as given.
        requested.parent.mkdir(parents=True, exist_ok=True)
        return requested
    project = _safe_name(project_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = requested
    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in count(1):
        suffix = "" if attempt == 1 else f"_{attempt}"
        candidate = output_dir / f"{project}_{timestamp}{suffix}.xlsx"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            continue
        try:
            os.close(descriptor)
        except BaseException:
            candidate.unlink(missing_ok=True)
            raise
        return candidate


def _write_summary_sheet(df_summary: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    _autosize_columns(df_summary, writer.sheets["Summary"])


def _write_problem_tables(
    master_zone: Optional["Zone"],
    writer: pd.ExcelWriter,
    *,
    prefix: str = "",
    used: set[str] | None = None,
) -> None:
    """Emit shifted and real temperature Problem Tables for every solved zone."""
    if master_zone is None:
        return

    used_sheet_names: set[str] = set() if used is None else used

    for zone in _iter_zones(master_zone):
        for target_name, target in zone.targets.items():
            table_specs = (
                (
                    f"{prefix}{zone.name} - {target_name} (Shifted)",
                    getattr(target, "pt", None),
                ),
                (
                    f"{prefix}{zone.name} - {target_name} (Real)",
                    getattr(target, "pt_real", None),
                ),
            )
            for sheet_label, table in table_specs:
                df = problem_table_frame(table, round_decimals=2)
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
    # Excel sheet names are case-insensitive, so compare them that way.
    used_folded = {name.casefold() for name in used}
    cleaned = _sanitize_sheet_name(base)
    candidate = cleaned[:31] or "Sheet"
    if candidate.casefold() not in used_folded:
        used.add(candidate)
        return candidate

    for period_idx in range(2, 1000):
        suffix = f" ({period_idx})"
        trimmed = (
            candidate[: 31 - len(suffix)]
            if len(candidate) + len(suffix) > 31
            else candidate
        )
        alt = f"{trimmed}{suffix}"
        if alt.casefold() not in used_folded:
            used.add(alt)
            return alt

    raise ValueError("Unable to allocate unique sheet name.")


def _sanitize_sheet_name(name: str) -> str:
    """Replace Excel-forbidden sheet-name characters and trailing apostrophes."""
    cleaned = re.sub(r"[:/?*\\\[\]]", "_", name).strip().rstrip("'")
    return cleaned or "Sheet"
