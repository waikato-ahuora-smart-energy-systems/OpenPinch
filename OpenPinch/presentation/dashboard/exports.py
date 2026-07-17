"""Streamlit controls for exporting dashboard tables."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .dependencies import _require_openpyxl


def render_table_export(
    streamlit: Any,
    default_path: str,
    *,
    dashboard_key: str,
    target_name: str,
    frame: pd.DataFrame,
    table_kind: str,
) -> None:
    """Render controls that save one problem-table frame as Excel."""
    save_path = streamlit.text_input(
        "Save location",
        default_path,
        key=f"save_path_{dashboard_key}_{target_name}_{table_kind}",
    )
    if not streamlit.button(
        "Save table as Excel",
        key=f"save_button_{dashboard_key}_{target_name}_{table_kind}",
    ):
        return

    destination = save_path.strip()
    if not destination:
        streamlit.error("Please provide a file path to save the table.")
        return

    buffer = BytesIO()
    engine_name = _require_openpyxl().__name__
    with pd.ExcelWriter(buffer, engine=engine_name) as writer:
        frame.to_excel(writer, index=False, sheet_name="Problem Table")
    try:
        Path(destination).write_bytes(buffer.getvalue())
    except OSError as exc:
        streamlit.error(f"Failed to save file: {exc}")
    else:
        streamlit.success(f"Saved table to {destination}")


__all__ = ["render_table_export"]
