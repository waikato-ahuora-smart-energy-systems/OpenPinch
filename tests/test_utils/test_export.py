from OpenPinch.utils.export import (
    export_target_summary_to_excel_with_units,
    _split_vu,
    _safe_name,
    _autosize_columns,
)

from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import pandas as pd
import pytest


# --------------------------------------------------------------------------------------
# Fixtures & tiny helpers
# --------------------------------------------------------------------------------------

class VU:
    """Simple value-with-units stub."""
    def __init__(self, value, units):
        self.value = value
        self.units = units


def _make_target(
    name="Base",
    pinch_cold=VU(85.0, "°C"),
    pinch_hot=VU(125.0, "°C"),
    Qh=VU(100.0, "kW"),
    Qc=VU(80.0, "kW"),
    Qr=VU(20.0, "kW"),
    degree_of_integration=VU(0.7, "-"),
    utility_cost=VU(1234.0, "$/h"),
    area=VU(456.0, "m²"),
    work_target=VU(10.0, "kW"),
    turbine_eff_target=VU(0.8, "-"),
    exergy_sources=VU(200.0, "kW"),
    exergy_sinks=VU(180.0, "kW"),
    exergy_req_min=VU(15.0, "kW"),
    exergy_des_min=VU(12.0, "kW"),
    num_units=7,
    capital_cost=111.0,
    total_cost=222.0,
    ETE=0.93,
    hot_utils=None,
    cold_utils=None,
):
    """Builds a single TargetResults-like stub compatible with export.py access pattern."""
    if hot_utils is None:
        hot_utils = [SimpleNamespace(name="HP Steam", heat_flow=VU(75.0, "kW"))]
    if cold_utils is None:
        cold_utils = [SimpleNamespace(name="Cooling Water", heat_flow=VU(60.0, "kW"))]

    return SimpleNamespace(
        name=name,
        temp_pinch=SimpleNamespace(cold_temp=pinch_cold, hot_temp=pinch_hot),
        Qh=Qh, Qc=Qc, Qr=Qr,
        degree_of_integration=degree_of_integration,
        hot_utilities=hot_utils,
        cold_utilities=cold_utils,
        utility_cost=utility_cost,
        area=area,
        work_target=work_target,
        turbine_efficiency_target=turbine_eff_target,
        exergy_sources=exergy_sources,
        exergy_sinks=exergy_sinks,
        exergy_req_min=exergy_req_min,
        exergy_des_min=exergy_des_min,
        num_units=num_units,
        capital_cost=capital_cost,
        total_cost=total_cost,
        ETE=ETE,
    )


# --------------------------------------------------------------------------------------
# export_target_summary_to_excel_with_units
# --------------------------------------------------------------------------------------

def test_export_writes_expected_excel(tmp_path: Path, monkeypatch):
    """
    End-to-end: ensure the Excel file is written with the right filename pattern
    and the Summary sheet contains expected columns/values (values + units).
    """
    # Freeze timestamp used in filename
    class _FixedDT:
        @staticmethod
        def now():
            return datetime(2025, 1, 2, 3, 4, 5)

        @staticmethod
        def strptime(*a, **kw):
            return datetime.strptime(*a, **kw)

    # Patch the datetime reference inside the export module
    import OpenPinch.utils.export as export_mod
    monkeypatch.setattr(export_mod, "datetime", _FixedDT, raising=True)

    # Project name exercises _safe_name (forbidden chars + spaces)
    target_response = SimpleNamespace(
        name="Proj: Foo/Bar",  # becomes Proj__Foo_Bar
        targets=[
            _make_target(name="Base"),
            _make_target(
                name="Alt A",
                hot_utils=[SimpleNamespace(name="MP Steam", heat_flow=VU(55.0, "kW"))],
                cold_utils=[SimpleNamespace(name="Chilled Water", heat_flow=VU(30.0, "kW"))],
            ),
        ],
    )

    out = export_target_summary_to_excel_with_units(target_response, out_dir=tmp_path)
    out_path = Path(out)
    assert out_path.exists()

    # Filename pattern check
    assert out_path.name == "Proj__Foo_Bar_20250102_030405.xlsx"

    # Verify Summary sheet contents round-trip
    df = pd.read_excel(out_path, sheet_name="Summary")
    # Expect at least our known columns
    expected_cols = {
        "Target",
        "Cold Pinch (value)", "Cold Pinch (unit)",
        "Hot Pinch (value)", "Hot Pinch (unit)",
        "Qh (value)", "Qh (unit)",
        "Qc (value)", "Qc (unit)",
        "Qr (value)", "Qr (unit)",
        "Degree of Integration (value)", "Degree of Integration (unit)",
        "Utility Cost (value)", "Utility Cost (unit)",
        "Area (value)", "Area (unit)",
        "Num Units", "Capital Cost", "Total Cost",
        "Work Target (value)", "Work Target (unit)",
        "Turbine Eff Target (value)", "Turbine Eff Target (unit)",
        "ETE",
        "Exergy Sources (value)", "Exergy Sources (unit)",
        "Exergy Sinks (value)", "Exergy Sinks (unit)",
        "Exergy Req Min (value)", "Exergy Req Min (unit)",
        "Exergy Des Min (value)", "Exergy Des Min (unit)",
        # Utility columns come from names:
        "HP Steam (value)", "HP Steam (unit)",
        "Cooling Water (value)", "Cooling Water (unit)",
        "MP Steam (value)", "MP Steam (unit)",
        "Chilled Water (value)", "Chilled Water (unit)",
    }
    assert expected_cols.issubset(set(df.columns))

    # Row for "Base"
    base = df.loc[df["Target"] == "Base"].iloc[0]
    assert pytest.approx(base["Cold Pinch (value)"]) == 85.0
    assert base["Cold Pinch (unit)"] == "°C"
    assert pytest.approx(base["Qh (value)"]) == 100.0
    assert base["Qh (unit)"] == "kW"
    assert pytest.approx(base["HP Steam (value)"]) == 75.0
    assert base["HP Steam (unit)"] == "kW"
    assert pytest.approx(base["Cooling Water (value)"]) == 60.0
    assert base["Cooling Water (unit)"] == "kW"
    assert int(base["Num Units"]) == 7
    assert pytest.approx(base["ETE"]) == 0.93

    # Row for "Alt A"
    alt = df.loc[df["Target"] == "Alt A"].iloc[0]
    assert pytest.approx(alt["MP Steam (value)"]) == 55.0
    assert alt["MP Steam (unit)"] == "kW"
    assert pytest.approx(alt["Chilled Water (value)"]) == 30.0
    assert alt["Chilled Water (unit)"] == "kW"


# --------------------------------------------------------------------------------------
# _split_vu
# --------------------------------------------------------------------------------------

def test_split_vu_handles_none_plain_number_and_object():
    assert _split_vu(None) == (None, None)
    assert _split_vu(3) == (3.0, None)
    assert _split_vu(3.14) == (3.14, None)
    assert _split_vu(VU(12.3, "kg/s")) == (12.3, "kg/s")

def test_split_vu_non_numeric_unparsable():
    # Strings that cannot be coerced to float return (None, None)
    assert _split_vu("not-a-number") == (None, None)


# --------------------------------------------------------------------------------------
# _safe_name
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (" Project  X  ", "Project_X"),
        ("data:run/01", "data_run_01"),
        ("a\\b|c*<d>?\"e", "a_b_c_d_e"),
        ("", "Project"),
    ],
)
def test_safe_name_sanitization(raw, expected):
    assert _safe_name(raw) == expected

# --------------------------------------------------------------------------------------
# _autosize_columns
# --------------------------------------------------------------------------------------

def test_autosize_columns_sets_reasonable_widths(tmp_path: Path):
    # Prepare a small df with long header + long cell to exercise both code paths
    df = pd.DataFrame({
        "A" * 50: ["x" * 10, "y" * 30],  # long header, varying cell content
        "B": ["short", "also short"],
    })

    # Create a worksheet via openpyxl (used under the hood by pandas writer)
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active

    # Write header row so ws.cell(row=1, column=...) has a real cell
    for j, col in enumerate(df.columns, start=1):
        ws.cell(row=1, column=j, value=col)

    # Call autosize; should not raise and should cap widths at 40
    _autosize_columns(df, ws)

    # Check widths
    # First column width should be capped at 40 due to very long header
    first_col_letter = ws.cell(row=1, column=1).column_letter
    assert ws.column_dimensions[first_col_letter].width == 40

    # Second column should be at least len("B") + 2 = 3 and not exceed 40
    second_col_letter = ws.cell(row=1, column=2).column_letter
    w = ws.column_dimensions[second_col_letter].width
    assert 3 <= w <= 40
