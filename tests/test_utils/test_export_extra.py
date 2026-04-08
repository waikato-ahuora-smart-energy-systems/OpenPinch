"""Additional branch coverage tests for export helpers."""

from pathlib import Path

import pandas as pd
import pytest

import OpenPinch.utils.export as export_mod
from OpenPinch.classes import EnergyTarget, Zone


def test_write_problem_tables_skips_empty_dataframes(monkeypatch, tmp_path: Path):
    zone = Zone("Plant")
    target = EnergyTarget(name="Plant/DI")
    target.pt = object()
    target.pt_real = object()
    zone.targets["DI"] = target

    monkeypatch.setattr(
        "OpenPinch.streamlit_webviewer.web_graphing.problem_table_to_dataframe",
        lambda table, round_decimals=2: pd.DataFrame(),
    )

    out_path = tmp_path / "out.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame({"keep": [1]}).to_excel(writer, sheet_name="Keep", index=False)
        export_mod._write_problem_tables(zone, writer)
        assert set(writer.sheets.keys()) == {"Keep"}


def test_unique_sheet_name_suffix_and_exhaustion_paths():
    used = {"Sheet"}
    second = export_mod._unique_sheet_name("Sheet", used)
    assert second == "Sheet (2)"

    exhausted = {"Sheet"} | {f"Sheet ({i})" for i in range(2, 1000)}
    with pytest.raises(ValueError, match="Unable to allocate unique sheet name"):
        export_mod._unique_sheet_name("Sheet", exhausted)
