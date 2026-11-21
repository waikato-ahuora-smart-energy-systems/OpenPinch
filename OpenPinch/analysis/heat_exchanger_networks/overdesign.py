"""Overdesign margin on a network's heat-recovery areas.

A margin of OD per cent installs (1 + OD/100) times each recovery
exchanger's area; heaters and coolers are not scaled. A rating solve, at
those fixed areas, then re-optimises every period's duties for the least
period-weighted utility cost, with bypass where a period needs less area.
``"optimise"`` picks OD in [0, ``maximum``] for the least total annual cost
(annualised capital on the installed areas plus utility cost) with a bounded
one-dimensional search over rating solves. A positive margin whose rating
fails is left out of the search; after a failed maximum, bisection finds
the largest margin that rates, which then bounds the search. When none
could be rated the margin is 0 % with a ``UserWarning``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

from scipy.optimize import minimize_scalar

from ...domain.enums import HeatExchangerKind
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)

OPTIMISE = "optimise"
# The bounded search stops within this many percentage points of the optimum,
# after at most this many rating solves besides the two ends.
_SEARCH_TOLERANCE = 0.5
_SEARCH_SOLVES = 15


def validate_margin(margin) -> float | str | None:
    """``None``, a percentage of at least 0, or ``"optimise"``."""

    if margin is None or margin == OPTIMISE:
        return margin
    if isinstance(margin, bool) or not isinstance(margin, int | float):
        raise ValueError(
            "overdesign_percentage must be None, a percentage of at least 0 or "
            f"'optimise'; got {margin!r}."
        )
    if not margin >= 0.0:
        raise ValueError(f"overdesign_percentage must be at least 0; got {margin!r}.")
    return float(margin)


def with_scaled_recovery_areas(
    network: HeatExchangerNetwork, factor: float
) -> HeatExchangerNetwork:
    """``network`` with every recovery exchanger's area times ``factor``."""

    return network.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(update={"area": exchanger.area * factor})
                if exchanger.kind is HeatExchangerKind.RECOVERY
                else exchanger
                for exchanger in network.exchangers
            )
        }
    )


def apply_overdesign(
    network: HeatExchangerNetwork,
    margin: float | str,
    *,
    maximum: float,
    rate: Callable[[HeatExchangerNetwork], Any],
):
    """Rate ``network`` with its recovery areas enlarged by ``margin``.

    ``rate`` runs the rating solve on a network carrying installed areas and
    returns a result whose ``selected_network`` is the rated network. Returns
    ``(result, metrics)`` for the chosen margin; ``metrics`` holds the mode,
    the percentage, each period's recovery and utility before and after, the
    total annual cost with and without the margin, every evaluated
    ``(percentage, cost)`` point, and every margin of an ``"optimise"``
    search whose rating failed (``ValueError`` or ``RuntimeError``; the
    rating at 0 % must succeed).
    """

    evaluated: dict[float, Any] = {}
    failed: list[float] = []

    def rated(percentage: float):
        percentage = float(percentage)
        if percentage not in evaluated:
            evaluated[percentage] = rate(
                with_scaled_recovery_areas(network, 1.0 + percentage / 100.0)
            )
        return evaluated[percentage]

    def cost(percentage: float) -> float:
        return float(rated(percentage).selected_network.total_annual_cost)

    def searched(percentage: float) -> float:
        """The cost at a positive margin, infinite where its rating fails."""
        percentage = float(percentage)
        if percentage in failed:
            return math.inf
        try:
            return cost(percentage)
        except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
            raise
        except (ValueError, RuntimeError):
            failed.append(percentage)
            return math.inf

    if margin == OPTIMISE:
        cost(0.0)
        if maximum > 0.0:
            # After a failed maximum, bisection finds the largest margin that
            # rates (0 % always does), which bounds the search, so small
            # feasible margins are not missed.
            upper, failed_above = 0.0, float(maximum)
            if not math.isinf(searched(failed_above)):
                upper = failed_above
            else:
                while failed_above - upper > _SEARCH_TOLERANCE:
                    middle = 0.5 * (upper + failed_above)
                    if math.isinf(searched(middle)):
                        failed_above = middle
                    else:
                        upper = middle
            if upper > 0.0:
                minimize_scalar(
                    searched,
                    bounds=(0.0, upper),
                    method="bounded",
                    options={"xatol": _SEARCH_TOLERANCE, "maxiter": _SEARCH_SOLVES},
                )
            if failed and all(point == 0.0 for point in evaluated):
                warnings.warn(
                    "no positive overdesign margin could be rated (every rating "
                    f"from {min(failed):g} to {max(failed):g} % failed); the "
                    "network keeps 0 %, which is not shown to be optimal.",
                    UserWarning,
                    stacklevel=2,
                )
        percentage = min(evaluated, key=lambda point: (cost(point), point))
    else:
        percentage = float(margin)
        cost(0.0)
    result = rated(percentage)
    before = _period_totals(network)
    after = _period_totals(result.selected_network)
    metrics = {
        "mode": "optimised" if margin == OPTIMISE else "fixed",
        "percentage": percentage,
        "periods": {
            period: {
                "recovery_before": before[period]["recovery"],
                "recovery_after": after[period]["recovery"],
                "utility_before": before[period]["utility"],
                "utility_after": after[period]["utility"],
            }
            for period in before
        },
        "cost_without_margin": cost(0.0),
        "cost": cost(percentage),
        "search": [[point, cost(point)] for point in sorted(evaluated)],
        "failed_margins": sorted(set(failed)),
    }
    return result, metrics


def _period_totals(network: HeatExchangerNetwork) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for exchanger in network.exchangers:
        key = "recovery" if exchanger.kind is HeatExchangerKind.RECOVERY else "utility"
        for state in exchanger.period_states:
            period = totals.setdefault(
                state.period_id, {"recovery": 0.0, "utility": 0.0}
            )
            period[key] += state.duty
    return totals


__all__ = [
    "OPTIMISE",
    "apply_overdesign",
    "validate_margin",
    "with_scaled_recovery_areas",
]
