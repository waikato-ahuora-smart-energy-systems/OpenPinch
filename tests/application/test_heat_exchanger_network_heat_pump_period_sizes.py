"""On several periods the sizes tried are every period's load kinks.

The net load profile differs from period to period, and so do its kinks.
A size worth trying in one period is tried over all of them: the sweep
takes the kinks of every period, not of whichever period comes first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


# The solver lane: six sizes are designed and fitted to three periods, 15 s.
pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.solver,
    pytest.mark.skipif(not _highs_available(), reason="needs highspy"),
]


def test_the_sizes_of_a_multiperiod_design_are_the_kinks_of_every_period():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )
    by_period = {
        period: {round(fraction, 4) for fraction in result.logical_load_fractions}
        for period, result in problem.target.all_periods.hpr_load_kinks().items()
    }
    kinks = set().union(*by_period.values())

    design = problem.design.heat_exchanger_network(
        heat_pump={
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )

    sizes = {
        round(outcome.network.heat_pumps[0].load_fraction, 4)
        for outcome in design.top(1000)
        if outcome.network.heat_pumps
    }
    # Turndown, base and peak each have kinks the others lack, and a size
    # from each is in the ranking. Not every size need be: one at which no
    # network is designed is left out with a warning (0.6241 here).
    assert len(kinks) == 6
    assert sizes <= kinks
    for period, own in by_period.items():
        others = set().union(*(k for p, k in by_period.items() if p != period))
        assert (own - others) & sizes, period
