"""Search for the ORC design with the lowest total annual cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...optimisation.errors import NoOptimisationCandidatesError
from ...optimisation.models import OptimisationOptions, OptimisationProblem
from ...optimisation.service import run_multistart_minimisation
from .carnot import OrcDesign, carnot_orc_design, orc_temperature_window
from .costing import OrcCosts, orc_costs
from .inputs import OrcTargetInputs
from .profile import OrcHeatSource

__all__ = ["OrcSearchResult", "OrcTargetingError", "optimise_carnot_orc"]

# An ORC must save at least this much ($/y) to count as beneficial.
_MIN_BENEFIT = 1.0


class OrcTargetingError(ValueError):
    """No ORC design pays, or the search found no valid design."""


@dataclass(frozen=True)
class OrcSearchResult:
    """The selected design, its costs and the decision vector behind it."""

    design: OrcDesign
    costs: OrcCosts
    x: tuple[float, ...]


def evaluate_carnot_orc(
    x: np.ndarray,
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
) -> tuple[OrcDesign, OrcCosts] | None:
    """Decode ``x`` and cost the design; ``None`` if no design is possible."""
    design = carnot_orc_design(x, source, inputs)
    if design is None:
        return None
    costs = orc_costs(
        W_net=np.asarray(design.W_net),
        Q_in=design.Q_in_total,
        Q_out=design.Q_out_total,
        inputs=inputs,
    )
    return design, costs


def _objective_scale(source: OrcHeatSource, inputs: OrcTargetInputs) -> float:
    """$/y of power from converting the whole surplus, to keep values O(1)."""
    return max(source.surplus * max(inputs.power_value, 1e-9), 1.0)


def _carnot_objective(x, source, inputs, scale) -> float:
    evaluated = evaluate_carnot_orc(x, source, inputs)
    if evaluated is None:
        return 0.0
    return evaluated[1].total_annualized_cost_change / scale


def _initial_points(n: int) -> tuple[tuple[float, ...], ...]:
    """Evenly spaced units taking all the heat available to them."""
    points = []
    for first in (0.25, 0.5, 0.75):
        # Spread the remaining units evenly over the rest of the window.
        temps = [first] + [1.0 / (n - i + 1) for i in range(1, n)]
        points.append(tuple(temps) + (1.0,) * n)
    return tuple(points)


def optimise_carnot_orc(
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
) -> OrcSearchResult:
    """Return the Carnot ORC design with the lowest total annual cost.

    Raises ``OrcTargetingError`` when the surplus is too cold for an ORC or
    when no design saves money.
    """
    if orc_temperature_window(source, inputs) is None:
        raise OrcTargetingError(
            "The process surplus below the pinch is too cold for an ORC at the "
            f"{inputs.T_cond:g} degC condensing temperature."
        )
    n = int(inputs.n_stages)
    scale = _objective_scale(source, inputs)
    starts = _initial_points(n)
    problem = OptimisationProblem(
        objective=_carnot_objective,
        bounds=((0.0, 1.0),) * (2 * n),
        initial_points=starts,
        args=(source, inputs, scale),
    )
    options = OptimisationOptions(
        n_runs=max(int(inputs.max_multistart), 1),
        maxiter=max(int(inputs.maximum_iterations), 1),
        seed=int(inputs.seed),
        max_minima=4,
        local_method="SLSQP",
    )
    candidates = [np.asarray(point) for point in starts]
    try:
        result = run_multistart_minimisation(
            problem, method=inputs.bb_minimiser, options=options
        )
        candidates.extend(np.asarray(c.point) for c in result.candidates)
    except NoOptimisationCandidatesError:
        pass

    best: OrcSearchResult | None = None
    for x in candidates:
        evaluated = evaluate_carnot_orc(x, source, inputs)
        if evaluated is None:
            continue
        design, costs = evaluated
        if best is None or (
            costs.total_annualized_cost_change < best.costs.total_annualized_cost_change
        ):
            best = OrcSearchResult(
                design=design,
                costs=costs,
                x=tuple(float(v) for v in np.clip(x, 0.0, 1.0)),
            )
    if best is None or best.costs.total_annualized_cost_change > -_MIN_BENEFIT:
        raise OrcTargetingError(
            "No beneficial ORC: no design's power is worth more than its "
            "annualised capital and cooling."
        )
    return best
