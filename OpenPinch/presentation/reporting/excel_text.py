"""Write Excel text cells as plain strings, never as formulas."""

from __future__ import annotations

from typing import Any

# Text starting with one of these can be read by a spreadsheet as a formula.
FORMULA_LEADING_CHARACTERS = ("=", "+", "-", "@", "\t", "\r")


def write_text_as_plain_strings(workbook: Any) -> None:
    """Store every text cell of an openpyxl workbook as a plain string.

    openpyxl stores text starting with ``=`` as a formula. Each text cell is
    stored as a string instead, and text that a spreadsheet could read as a
    formula also gets Excel's quote prefix, so editing the cell keeps it text.
    """
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if not isinstance(cell.value, str):
                    continue
                if cell.data_type == "f":
                    cell.data_type = "s"
                if cell.value.startswith(FORMULA_LEADING_CHARACTERS):
                    cell.quotePrefix = True


__all__ = ["FORMULA_LEADING_CHARACTERS", "write_text_as_plain_strings"]
