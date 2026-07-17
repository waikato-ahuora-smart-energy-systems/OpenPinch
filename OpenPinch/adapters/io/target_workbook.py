"""Workbook output adapter for solved target results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...utils.export import export_target_summary_to_excel_with_units

if TYPE_CHECKING:
    from ...contracts.output import TargetOutput
    from ...domain.zone import Zone


def export_target_workbook(
    target_response: "TargetOutput",
    master_zone: "Zone | None",
    output_directory: Any,
) -> Path:
    """Write a solved target workbook and return its filesystem path."""
    return Path(
        export_target_summary_to_excel_with_units(
            target_response=target_response,
            master_zone=master_zone,
            out_dir=output_directory,
        )
    )


__all__ = ["export_target_workbook"]
