"""Operate an installed heat pump in one period (design-and-operate).

A shared multiperiod design is one machine, its sizes installed once. In a
period it runs at the operation that costs least to run, with every
compressor and exchanger needing no more than its installed size; its
capital is sunk, so the period's objective is its operating cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ....contracts.hpr import HeatPumpTargetInputs, HPRBackendResult
from ..common.capital import MachineComponents
from ..common.shared import hpr_penalty_cost_scale
from ..optimisation_adapter import evaluate_hpr_candidate, run_hpr_candidate_search

__all__ = ["DOES_NOT_FIT_REASON", "operate_installed_machine"]

DOES_NOT_FIT_REASON = (
    "No operation of the installed heat pump found in this period fits its "
    "compressors, motors and exchangers, and its compressors' turndown."
)

# A component needing more than installed is priced, per unit of its
# relative excess, at this many times the no-heat-pump cost of the service.
_OVERSIZE_PENALTY = 10.0
# The summed relative shortfall an operation may leave and still fit.
_SHORTFALL_TOLERANCE = 1e-6


def operate_installed_machine(
    installed: Sequence[MachineComponents],
    *,
    objective,
    initial_points: Sequence[float] | np.ndarray | None,
    bounds: Sequence[Sequence[float]],
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Return the period's cheapest operation of ``installed``.

    ``objective`` is the cycle family's period objective; each candidate is
    ranked on its operating cost and feasibility penalty, plus a penalty for
    every component needing more than its installed size.
    """
    scale = hpr_penalty_cost_scale(args)
    turndown = float(getattr(args, "turndown", 0.0) or 0.0)

    def shortfall(result: HPRBackendResult) -> float:
        needed = result.hpr_machine_components
        return _oversize(needed, installed) + _turndown_gap(needed, installed, turndown)

    def operating(x, period_args, **kwargs) -> HPRBackendResult:
        result = objective(x, period_args, **kwargs)
        if not result.success or result.hpr_operating_cost is None:
            return result
        return result.with_updates(
            obj=float(result.hpr_operating_cost.to("$/y").value)
            + float(result.feasibility_penalty)
            + _OVERSIZE_PENALTY * scale * shortfall(result)
        )

    candidates = run_hpr_candidate_search(
        objective=operating,
        initial_points=initial_points,
        bounds=bounds,
        args=args,
    )
    best = min(candidates, key=lambda candidate: candidate.objective)
    result = evaluate_hpr_candidate(objective=objective, point=best.point, args=args)
    if result.success and shortfall(result) > _SHORTFALL_TOLERANCE:
        # The penalty guides the search; an operation still outside the
        # machine's sizes or turndown is none.
        return HPRBackendResult.failure(reason=DOES_NOT_FIT_REASON)
    return result


def _turndown_gap(
    needed: Sequence[MachineComponents] | None,
    installed: Sequence[MachineComponents],
    turndown: float,
) -> float:
    """How far running compressors sit inside their turndown, as shares of
    their installed volume flow.

    A compressor runs at no less than ``turndown`` of its installed volume
    flow, or is off. The gap of each is its distance to the nearer of the
    two, so switching a compressor off is never penalised.
    """
    if not needed or turndown <= 0.0:
        return 0.0
    gap = 0.0
    for machine, limit in zip(needed, installed, strict=True):
        for need, have in zip(machine.compressors, limit.compressors, strict=True):
            if have.volume_flow <= 0.0:
                continue
            share = need.volume_flow / have.volume_flow
            if 0.0 < share < turndown:
                gap += min(share, turndown - share)
    return float(gap)


def _oversize(
    needed: Sequence[MachineComponents] | None,
    installed: Sequence[MachineComponents],
) -> float:
    """The summed relative excess of each component over its installed size."""
    if not needed:
        return 0.0
    excess = 0.0
    for machine, limit in zip(needed, installed, strict=True):
        pairs = [
            pair
            for need, have in zip(machine.compressors, limit.compressors, strict=True)
            for pair in (
                (need.volume_flow, have.volume_flow),
                (need.shaft_power, have.shaft_power),
            )
        ] + list(zip(machine.exchanger_areas, limit.exchanger_areas, strict=True))
        for need, have in pairs:
            if need <= have:
                continue
            excess += need / have - 1.0 if have > 0.0 else 1.0 + need
    return float(excess)
