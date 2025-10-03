"""Shared numerical helpers used by multiple analysis submodules."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import math
from typing import List, Tuple, Union
from ..lib import *

if TYPE_CHECKING:
    from ..classes import *


def get_pinch_loc(pt: ProblemTable, col=PT.H_NET.value) -> tuple[int, int, bool]:
    """Returns the row indices of the Hot and Cold Pinch Temperatures."""
    # Pull out the 1-D profile as a NumPy array
    h_net = np.asarray(pt.col[col])
    n     = h_net.size

    abs_arr    = np.abs(h_net)
    zeros_mask = abs_arr < tol

    if np.all(zeros_mask) == False and np.any(zeros_mask):
        # ---------- Hot-pinch (scan top-down) ----------
        first_zero = np.flatnonzero(zeros_mask)[0]
        if first_zero > 0:                          # zero not on the first row
            row_h = first_zero
        else:                                       # zero *is* the first row
            nz_after = np.flatnonzero(~zeros_mask)  # first genuinely non-zero row
            row_h    = nz_after[0] - 1 if nz_after.size else n - 1
        
        # ---------- Cold-pinch (scan bottom-up) ----------
        last_zero = np.flatnonzero(zeros_mask)[-1]
        if last_zero < n - 1:                       # zero not on the last row
            row_c = last_zero
        else:                                       # zero *is* the last row
            # find first non-zero when walking *up* from the bottom
            nz_before_rev = np.flatnonzero(~zeros_mask[::-1])
            row_c = n - nz_before_rev[0] if nz_before_rev.size else 0
    else:
        row_h = n - 1
        row_c = 0

    valid = row_h <= row_c
    return row_h, row_c, valid


def get_pinch_temperatures(pt: ProblemTable, col_T: str =PT.T.value, col_H=PT.H_NET.value) -> tuple[float | None, float | None]:
    """Determines the hottest hot Pinch Temperature and coldest cold Pinch Temperature and return both values."""
    h_loc, c_loc, valid = get_pinch_loc(pt, col_H)
    if valid:
        return pt.loc[h_loc, col_T], pt.loc[c_loc, col_T]
    else:
        return None, None


def shift_heat_cascade(pt: ProblemTable, dh: float, col: Union[int, str, Enum]) -> ProblemTable:
    """Shifts a column in a heat cascade DataFrame by dH."""
    pt.col[col] += dh
    return pt.copy


def insert_temperature_interval_into_pt(pt: ProblemTable, T_ls: List[float] | float) -> Tuple[ProblemTable, int]:
    """Efficient insert into a ProblemTable assuming strictly descending T column."""
    if isinstance(T_ls, float):
        T_ls = [T_ls]

    for T_new in T_ls:
        col = pt.col_index
        T_col = pt.data[:, col[PT.T.value]]

        # Vectorized scan for insert index
        insert_index = None
        for i in range(1, len(T_col)):
            if T_col[i - 1] - tol > T_new > T_col[i] + tol:
                insert_index = i
                break

        if insert_index is None:
            return pt, 0  # already exists
        
        row_top = pt.data[insert_index - 1]
        row_bot = pt.data[insert_index]
        
        cp_hot = row_bot[col[PT.CP_HOT.value]]
        cp_cold = row_bot[col[PT.CP_COLD.value]]
        mcp_net = row_bot[col[PT.MCP_NET.value]]

        delta_above = row_top[col[PT.T.value]] - T_new
        delta_below = T_new - row_bot[col[PT.T.value]]

        row_dict = {
            PT.T.value: T_new,
            PT.DELTA_T.value: delta_above,
            PT.CP_HOT.value: cp_hot,
            PT.DELTA_H_HOT.value: delta_above * cp_hot,
            PT.CP_COLD.value: cp_cold,
            PT.DELTA_H_COLD.value: delta_above * cp_cold,
            PT.MCP_NET.value: mcp_net,
            PT.DELTA_H_NET.value: delta_above * mcp_net,
        }

        icol_T = col[PT.T.value]
        for key in [
            PT.H_HOT.value, PT.H_COLD.value, PT.H_NET.value,
            PT.H_NET_NP.value, PT.H_NET_A.value, PT.H_NET_V.value,
        ]:
            i = col[key]
            if not np.isnan(row_bot[i]):
                row_dict[key] = linear_interpolation(
                    T_new,
                    row_bot[icol_T],
                    row_top[icol_T],
                    row_bot[i],
                    row_top[i]
                )

        # Insert and update next row
        pt.insert(row_dict, insert_index)
        
        pt.data[insert_index + 1, col[PT.DELTA_T.value]] = delta_below
        pt.data[insert_index + 1, col[PT.DELTA_H_HOT.value]] = delta_below * cp_hot
        pt.data[insert_index + 1, col[PT.DELTA_H_COLD.value]] = delta_below * cp_cold
        pt.data[insert_index + 1, col[PT.DELTA_H_NET.value]] = delta_below * mcp_net

    return pt, 1


def key_name(zone_name: str, target_type: str = TargetType.DI.value):
    """Compose the canonical dictionary key for storing zone targets."""
    return f"{zone_name}/{target_type}"


def get_value(val: Union[float, dict, ValueWithUnit]) -> float:
    """Extract a numeric value from raw floats, dict payloads, or :class:`ValueWithUnit`."""
    if isinstance(val, float):
        return val
    elif isinstance(val, dict):
        return val['value']
    elif isinstance(val, ValueWithUnit):
        return val.value
    else:
        raise TypeError(f"Unsupported type: {type(val)}. Expected float, dict, or ValueWithUnit.")


def find_LMTD(T_hot_in: float, T_hot_out: float, T_cold_in: float, T_cold_out: float) -> float:
    """Returns the log mean temperature difference (LMTD) for a counterflow heat exchanger."""
    # Check temperature directions for counter-current assumption
    if T_hot_in < T_hot_out:
        raise ValueError("Hot fluid must cool down (T_hot_in > T_hot_out)")
    if T_cold_out < T_cold_in:
        raise ValueError("Cold fluid must heat up (T_cold_out > T_cold_in)")

    delta_T1 = T_hot_in - T_cold_out   # Inlet diff (hottest hot - hottest cold)
    delta_T2 = T_hot_out - T_cold_in   # Outlet diff (coldest hot - coldest cold)

    if delta_T1 <= 0 or delta_T2 <= 0:
        raise ValueError(f"Invalid temperature differences: ΔT1={delta_T1}, ΔT2={delta_T2}")

    if math.isclose(delta_T1, delta_T2, rel_tol=1e-6):
        return delta_T1  # or delta_T2 — they're equal

    return (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)


def compute_capital_recovery_factor(interest_rate: float, years: int) -> float:
    """Calculates the Capital Recovery Factor (CRF), also known as the annualisation factor."""
    i = interest_rate
    n = years
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def compute_capital_cost(area: float, num_units: int, config: Configuration) -> float:
    """Determine the capital cost of a heat exchanger."""
    return num_units * config.FC + num_units * config.VC * (area / num_units) ** config.EXP


def compute_annual_capital_cost(area: float, num_units: int, config: Configuration) -> float:
    """Determine the annualised capital cost of heat exchanger."""
    return compute_capital_cost(area, num_units, config) * compute_capital_recovery_factor(config.DISCOUNT_RATE, config.SERV_LIFE)


def compute_exergetic_temperature(T: float, T_ref_in_C: float = 15.0, units_of_T: str = "C") -> float:
    """Calculate the exergetic temperature difference relative to T_ref (in °C or K)."""
    # Marmolejo-Correa, D., Gundersen, T., 2013. New Graphical Representation of Exergy Applied to Low Temperature Process Design. 
    # Industrial & Engineering Chemistry Research 52, 7145–7156. https://doi.org/10.1021/ie302541e
    if units_of_T not in ("C", "K"):
        raise ValueError("units must be either 'C' or 'K'")

    T_amb = T_ref_in_C + C_to_K  # Convert reference to Kelvin
    T_K = T + C_to_K if units_of_T == "C" else T

    if T_K <= 0:
        raise ValueError("Absolute temperature must be > 0 K")

    ratio = T_K / T_amb
    return T_amb * (ratio - 1 - math.log(ratio))


def linear_interpolation(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Performs linear interpolation to estimate y at a given x, using two known points (x1, y1) and (x2, y2)."""
    if x1 == x2:
        raise ValueError("Cannot perform interpolation when x1 == x2 (undefined slope).")
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    return m * x + c
