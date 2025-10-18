"""Shared numerical helpers used by multiple analysis submodules."""

import math
from enum import Enum
from typing import List, Tuple, Union

from ..classes import ProblemTable
from ..lib import *


def get_pinch_loc(pt: ProblemTable, col=PT.H_NET.value) -> tuple[int, int, bool]:
    """Returns the row indices of the Hot and Cold Pinch Temperatures."""
    return pt.get_pinch_loc(col)


def get_pinch_temperatures(
    pt: ProblemTable, col_T: str = PT.T.value, col_H=PT.H_NET.value
) -> tuple[float | None, float | None]:
    """Determines the hottest hot Pinch Temperature and coldest cold Pinch Temperature and return both values."""
    return pt.get_pinch_temperatures(col_T=col_T, col_H=col_H)


def shift_heat_cascade(
    pt: ProblemTable, dh: float, col: Union[int, str, Enum]
) -> ProblemTable:
    """Shifts a column in a heat cascade DataFrame by dH."""
    return pt.shift_heat_cascade(dh, col)


def insert_temperature_interval_into_pt(
    pt: ProblemTable, T_ls: List[float] | float
) -> Tuple[ProblemTable, int]:
    """Efficient insert into a ProblemTable assuming strictly descending T column."""
    return pt.insert_temperature_interval(T_ls)


def key_name(zone_name: str, target_type: str = TargetType.DI.value):
    """Compose the canonical dictionary key for storing zone targets."""
    return f"{zone_name}/{target_type}"


def get_value(val: Union[float, dict, ValueWithUnit]) -> float:
    """Extract a numeric value from raw floats, dict payloads, or :class:`ValueWithUnit`."""
    if isinstance(val, float):
        return val
    elif isinstance(val, dict):
        return val["value"]
    elif isinstance(val, ValueWithUnit):
        return val.value
    else:
        raise TypeError(
            f"Unsupported type: {type(val)}. Expected float, dict, or ValueWithUnit."
        )


def find_LMTD(
    T_hot_in: float, T_hot_out: float, T_cold_in: float, T_cold_out: float
) -> float:
    """Returns the log mean temperature difference (LMTD) for a counterflow heat exchanger."""
    # Check temperature directions for counter-current assumption
    if T_hot_in < T_hot_out:
        raise ValueError("Hot fluid must cool down (T_hot_in > T_hot_out)")
    if T_cold_out < T_cold_in:
        raise ValueError("Cold fluid must heat up (T_cold_out > T_cold_in)")

    delta_T1 = T_hot_in - T_cold_out  # Inlet diff (hottest hot - hottest cold)
    delta_T2 = T_hot_out - T_cold_in  # Outlet diff (coldest hot - coldest cold)

    if delta_T1 <= 0 or delta_T2 <= 0:
        raise ValueError(
            f"Invalid temperature differences: ΔT1={delta_T1}, ΔT2={delta_T2}"
        )

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
    return (
        num_units * config.FC + num_units * config.VC * (area / num_units) ** config.EXP
    )


def compute_annual_capital_cost(
    area: float, num_units: int, config: Configuration
) -> float:
    """Determine the annualised capital cost of heat exchanger."""
    return compute_capital_cost(
        area, num_units, config
    ) * compute_capital_recovery_factor(config.DISCOUNT_RATE, config.SERV_LIFE)


def compute_exergetic_temperature(
    T: float, T_ref_in_C: float = 15.0, units_of_T: str = "C"
) -> float:
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
    return ProblemTable.linear_interpolation(x, x1, x2, y1, y2)
