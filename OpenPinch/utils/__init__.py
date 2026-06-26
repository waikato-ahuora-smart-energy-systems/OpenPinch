"""Utility helpers used across OpenPinch analyses.

This module aggregates reusable conversion utilities, workbook import/export
helpers, numerical shortcuts, and the timing decorator used to measure
performance critical routines.
"""

from .blackbox_minimisers import multiminima
from .costing import (
    compute_annual_capital_cost,
    compute_annual_energy_cost,
    compute_capital_cost,
    compute_capital_recovery_factor,
)
from .csv_to_json import get_problem_from_csv, get_results_from_csv
from .decorators import timing_decorator
from .export import export_target_summary_to_excel_with_units
from .heat_exchanger import (
    HX_NTU,
    CalcAreaUE,
    Coth,
    CrossflowUnmixedEff1,
    CrossflowUnmixedEff2,
    HX_Eff,
    HX_NTU_Numerical,
    MultiPassEff,
    MultiPassNTU,
    compute_LMTD_from_dts,
    compute_LMTD_from_ts,
    eNTU_slope_Numerical,
)
from .input_validation import (
    validate_stream_data,
    validate_utility_data,
)
from .plots import graph_simple_cc_plot
from .value_resolution import (
    evaluate_value_spec,
    get_period_value,
    get_scalar_value,
    resolve_value_array,
)
from .water_properties import (
    Tsat_p,
    fromSIunit_h,
    fromSIunit_p,
    fromSIunit_s,
    fromSIunit_T,
    h_ps,
    h_pT,
    hL_p,
    hV_p,
    psat_T,
    s_ph,
    toSIunit_h,
    toSIunit_p,
    toSIunit_s,
    toSIunit_T,
)
from .wkbook_to_json import get_problem_from_excel, get_results_from_excel

__all__ = [
    "compute_capital_recovery_factor",
    "compute_capital_cost",
    "compute_annual_capital_cost",
    "compute_annual_energy_cost",
    "get_problem_from_csv",
    "get_results_from_csv",
    "timing_decorator",
    "export_target_summary_to_excel_with_units",
    "CalcAreaUE",
    "Coth",
    "CrossflowUnmixedEff1",
    "CrossflowUnmixedEff2",
    "HX_Eff",
    "HX_NTU",
    "HX_NTU_Numerical",
    "MultiPassEff",
    "MultiPassNTU",
    "compute_LMTD_from_dts",
    "compute_LMTD_from_ts",
    "eNTU_slope_Numerical",
    "validate_stream_data",
    "validate_utility_data",
    "graph_simple_cc_plot",
    "get_scalar_value",
    "get_period_value",
    "evaluate_value_spec",
    "resolve_value_array",
    "multiminima",
    "Tsat_p",
    "fromSIunit_T",
    "fromSIunit_h",
    "fromSIunit_p",
    "fromSIunit_s",
    "hL_p",
    "hV_p",
    "h_pT",
    "h_ps",
    "psat_T",
    "s_ph",
    "toSIunit_T",
    "toSIunit_h",
    "toSIunit_p",
    "toSIunit_s",
    "get_problem_from_excel",
    "get_results_from_excel",
]
