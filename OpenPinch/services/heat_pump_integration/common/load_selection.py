"""Load selection helpers for heat pump and refrigeration targeting."""

from __future__ import annotations

import numpy as np

from ....lib.config import Configuration

__all__ = ["resolve_hpr_target_load"]


def resolve_hpr_target_load(
    *,
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    is_heat_pumping: bool = False,
    is_refrigeration: bool = False,
    config: Configuration | None = None,
    period_id: str | None = None,
    period_idx: int | None = None,
) -> float:
    """Return the requested HPR load capped by the available period duty."""
    if config is None:
        raise ValueError("config must be provided for HPR targeting.")

    if is_heat_pumping:
        load_values = H_net_cold
    elif is_refrigeration:
        load_values = H_net_hot
    else:
        return 0.0
    Q_max = float(np.nanmax(np.abs(load_values), initial=0.0))

    hpr = config.hpr
    if hpr.load_mode == "fraction":
        return Q_max * float(hpr.load_fraction)
    if hpr.load_mode == "duty":
        return min(float(hpr.load_duty), Q_max)
    if hpr.load_mode == "period_values":
        return min(
            _resolve_hpr_period_load(
                hpr.load_period_values,
                period_id=period_id,
                period_idx=period_idx,
            ),
            Q_max,
        )
    raise ValueError(f"Unsupported HPR_LOAD_MODE {hpr.load_mode!r}.")


def _resolve_hpr_period_load(
    load_by_period: dict[str, float],
    *,
    period_id: str | None,
    period_idx: int | None,
) -> float:
    if period_id is not None and str(period_id) in load_by_period:
        return float(load_by_period[str(period_id)])
    if period_idx is not None and str(period_idx) in load_by_period:
        return float(load_by_period[str(period_idx)])
    available = ", ".join(sorted(str(key) for key in load_by_period)) or "<none>"
    raise ValueError(
        "HPR_LOAD_PERIOD_VALUES does not define a load for the selected period. "
        f"Expected period_id {period_id!r} or period index {period_idx!r}; "
        f"available keys: {available}."
    )
