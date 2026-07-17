"""Area targeting methods."""

from __future__ import annotations

import numpy as np

from ...domain._problem_table.types import ProblemTableUpdateKwargs
from ...domain.configuration import Configuration, tol
from ...domain.enums import PT
from ...domain.stream_collection import StreamCollection
from ...utils.costing import compute_annual_capital_cost, compute_capital_cost
from ...utils.heat_exchanger import compute_LMTD_from_dts
from .graph_data import clean_composite_curve_ends
from .temperature_driving_force import get_temperature_driving_forces

__all__ = [
    "get_balanced_CC",
    "get_capital_cost_targets",
    "get_area_targets",
    "get_min_number_hx",
]


################################################################################
# Public API
################################################################################


def get_capital_cost_targets(
    area: float,
    num_units: int,
    config: Configuration,
) -> dict:
    """Estimate equipment and annualized capital costs from area/unit targets.

    Parameters
    ----------
    area:
        Total heat-transfer area target from balanced composite curves.
    num_units:
        Minimum exchanger count estimate for the same targeting scenario.
    config:
        Active configuration containing fixed/variable cost coefficients,
        capital exponent, discount rate, and service life assumptions.

    Returns
    -------
    tuple[float, float]
        ``(capital_cost, annual_capital_cost)``.
    """
    costing = config.costing
    capital_cost = compute_capital_cost(
        area,
        num_units,
        costing.hx_unit_cost,
        costing.hx_area_coeff,
        costing.hx_area_exp,
    )
    annual_capital_cost = compute_annual_capital_cost(
        capital_cost,
        costing.discount_rate,
        costing.service_life,
    )
    return capital_cost, annual_capital_cost


def get_balanced_CC(
    T_col: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    H_hot_ut: np.ndarray,
    H_cold_ut: np.ndarray,
    dT_vals: np.ndarray = None,
    RCP_hot: np.ndarray = None,
    RCP_cold: np.ndarray = None,
    RCP_hot_ut: np.ndarray = None,
    RCP_cold_ut: np.ndarray = None,
) -> ProblemTableUpdateKwargs:
    """Create the balanced composite curve using process and utility streams."""
    H_hot_bal = H_hot + H_hot_ut
    H_cold_bal = H_cold + H_cold_ut
    res = {
        PT.H_HOT_BAL: H_hot_bal,
        PT.H_COLD_BAL: H_cold_bal,
    }

    if (
        RCP_hot is not None
        and RCP_cold is not None
        and RCP_hot_ut is not None
        and RCP_cold_ut is not None
    ):
        dH_hot_bal = np.insert(
            H_hot_bal[:-1] - H_hot_bal[1:],
            0,
            0,
        )
        dH_cold_bal = np.insert(
            H_cold_bal[:-1] - H_cold_bal[1:],
            0,
            0,
        )
        R_hot_bal = np.zeros_like(dH_hot_bal, dtype=float)
        mask_hot = dH_hot_bal > tol
        np.divide(
            (RCP_hot + RCP_hot_ut) * dT_vals,
            dH_hot_bal,
            out=R_hot_bal,
            where=mask_hot,
        )
        R_cold_bal = np.zeros_like(dH_cold_bal, dtype=float)
        mask_cold = dH_cold_bal > tol
        np.divide(
            (RCP_cold + RCP_cold_ut) * dT_vals,
            dH_cold_bal,
            out=R_cold_bal,
            where=mask_cold,
        )
        res.update(
            {
                PT.H_HOT_BAL: H_hot + H_hot_ut,
                PT.H_COLD_BAL: H_cold + H_cold_ut,
                PT.RCP_HOT_BAL: RCP_hot + RCP_hot_ut,
                PT.RCP_COLD_BAL: RCP_cold + RCP_cold_ut,
                PT.R_HOT_BAL: R_hot_bal,
                PT.R_COLD_BAL: R_cold_bal,
            }
        )
    return {"T_col": T_col, "updates": res}


def get_area_targets(
    T_vals: np.ndarray,
    H_hot_bal: np.ndarray,
    H_cold_bal: np.ndarray,
    R_hot_bal: np.ndarray,
    R_cold_bal: np.ndarray,
) -> dict:
    """Estimate heat-transfer area targets with vectorised counter-current logic."""
    if abs((H_hot_bal[0] - H_hot_bal[-1]) - (H_cold_bal[0] - H_cold_bal[-1])) > tol:
        # Raise an error because heat-flow balance is required for this analysis.
        raise ValueError(
            "The temperature driving force plot requires the inputted "
            "composite curves to be balanced."
        )

    # Shift the hot and cold cascades to start from zero at the lowest temperature.
    if abs(H_hot_bal[0]) > tol:
        H_hot_bal = H_hot_bal - H_hot_bal[-1]
    if abs(H_cold_bal[0]) > tol:
        H_cold_bal = H_cold_bal - H_cold_bal[-1]

    Th, Hh = clean_composite_curve_ends(T_vals, H_hot_bal)
    Tc, Hc = clean_composite_curve_ends(T_vals, H_cold_bal)

    tdf = get_temperature_driving_forces(Th, Hh, Tc, Hc)
    dt_lm_i = compute_LMTD_from_dts(
        tdf["delta_T1"],
        tdf["delta_T2"],
    )
    Q_i: np.ndarray = tdf["dh_vals"]
    R_i = _map_interval_resistances_to_tdf(
        T_vals,
        R_hot_bal,
        R_cold_bal,
        tdf["t_h1"],
        tdf["t_h2"],
        tdf["t_c1"],
        tdf["t_c2"],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        U_i = np.where(R_i > tol, 1.0 / R_i, 1.0)
    if not (Q_i.shape == U_i.shape == dt_lm_i.shape):
        raise ValueError("Shape of heat exchanger area calculation arrays are unequal.")
    area_i = Q_i / (U_i * dt_lm_i)
    return area_i.sum()


def get_min_number_hx(
    T_vals: np.ndarray,
    H_hot_bal: np.ndarray,
    H_cold_bal: np.ndarray,
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    idx: int | None = None,
) -> int:
    """Estimate the minimum number of exchangers using vectorised interval logic."""
    H_net_bal = H_cold_bal - H_hot_bal
    mask = np.isclose(H_net_bal, 0.0, atol=tol)
    mask_true_positions = np.flatnonzero(mask)
    if mask_true_positions.size < 2:
        return 0

    adjacent = mask_true_positions[:-1] + 1 < mask_true_positions[1:]
    if not np.any(adjacent):
        return 0

    upper_indices = mask_true_positions[:-1][adjacent]
    lower_indices = mask_true_positions[1:][adjacent]
    t_high = np.asarray(T_vals, dtype=float)[upper_indices]
    t_low = np.asarray(T_vals, dtype=float)[lower_indices]

    num_hx = (
        _count_crossing_ranges(t_low, t_high, hot_streams, idx=idx)
        + _count_crossing_ranges(t_low, t_high, cold_streams, idx=idx)
        + _count_utility_range_containers(t_low, t_high, hot_utilities, idx=idx)
        + _count_utility_range_containers(t_low, t_high, cold_utilities, idx=idx)
    )
    return int(num_hx - t_low.size)


################################################################################
# Helper functions
################################################################################


def _map_interval_resistances_to_tdf(
    T_vals: np.ndarray,
    R_hot_bal: np.ndarray,
    R_cold_bal: np.ndarray,
    t_h1: np.ndarray,
    t_h2: np.ndarray,
    t_c1: np.ndarray,
    t_c2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align total resistances with temperature-driving-force intervals.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(total_resistance, mask)`` where ``total_resistance`` has the
            same length as the temperature-driving-force intervals and
            ``mask`` maps each interval to a corresponding temperature band in
            ``T_vals``.
    """
    interval_lower = T_vals[1:]
    interval_upper = T_vals[:-1]

    active_hot = (t_h1[np.newaxis, :] >= interval_lower[:, np.newaxis] - tol) & (
        t_h2[np.newaxis, :] <= interval_upper[:, np.newaxis] + tol
    )
    active_cold = (t_c1[np.newaxis, :] >= interval_lower[:, np.newaxis] - tol) & (
        t_c2[np.newaxis, :] <= interval_upper[:, np.newaxis] + tol
    )
    R_hot_mat = np.ones(shape=active_hot.shape) * R_hot_bal[1:, np.newaxis]
    R_cold_mat = np.ones(shape=active_hot.shape) * R_cold_bal[1:, np.newaxis]
    Rh = (active_hot * R_hot_mat).sum(axis=0)
    Rc = (active_cold * R_cold_mat).sum(axis=0)
    return Rh + Rc


def _count_crossing(
    T_low: float,
    T_high: float,
    streams: StreamCollection,
    idx: int | None = None,
):
    """Count process streams intersecting interval ``[T_low, T_high]``."""
    return int(
        _count_crossing_ranges(
            np.asarray([T_low], dtype=float),
            np.asarray([T_high], dtype=float),
            streams,
            idx=idx,
        )
    )


def _count_crossing_ranges(
    T_low: np.ndarray,
    T_high: np.ndarray,
    streams: StreamCollection,
    idx: int | None = None,
) -> int:
    """Count process stream crossings across several temperature intervals."""
    if len(streams) == 0 or T_low.size == 0:
        return 0
    numeric = streams.segment_numeric_view(idx)
    t_max = numeric.t_max_star[:, np.newaxis]
    t_min = numeric.t_min_star[:, np.newaxis]
    T_low = T_low[np.newaxis, :]
    T_high = T_high[np.newaxis, :]
    crossings = (
        ((t_max > T_low + tol) & (t_max <= T_high + tol))
        | ((t_min >= T_low - tol) & (t_min < T_high - tol))
        | ((t_min < T_low - tol) & (t_max > T_high + tol))
    )
    return sum(
        int(np.unique(numeric.parent_index[crossings[:, interval]]).size)
        for interval in range(crossings.shape[1])
    )


def _count_utility_range_container(
    T_low: float,
    T_high: float,
    utilities: StreamCollection,
    idx: int | None = None,
):
    """Count utility streams intersecting interval ``[T_low, T_high]``."""
    return int(
        _count_utility_range_containers(
            np.asarray([T_low], dtype=float),
            np.asarray([T_high], dtype=float),
            utilities,
            idx=idx,
        )
    )


def _count_utility_range_containers(
    T_low: np.ndarray,
    T_high: np.ndarray,
    utilities: StreamCollection,
    idx: int | None = None,
) -> int:
    """Count active utilities contained by several temperature intervals."""
    if len(utilities) == 0 or T_low.size == 0:
        return 0
    numeric = utilities.numeric_view(idx)
    t_max = numeric.t_max_star[:, np.newaxis]
    t_min = numeric.t_min_star[:, np.newaxis]
    active = numeric.heat_flow[:, np.newaxis] > tol
    T_low = T_low[np.newaxis, :]
    T_high = T_high[np.newaxis, :]
    contained = (t_min >= T_low - tol) & (t_max <= T_high + tol) & active
    return int(np.sum(contained))
