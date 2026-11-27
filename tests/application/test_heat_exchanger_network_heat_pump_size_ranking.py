"""The ranking orders heat pump sizes by what the steam they save is worth.

Basic pinch sample, Carnot heat pump at its load kink (0.4 of the load)
and at full load, compressor at 200 $/kW, electricity at 100 $/MWh. Below
a steam price of about 37 $/MWh less heat pump is cheaper, above it more:
the best network of each size falls in that order, and the selected
design is the one without a heat pump or the one at full load.

The part-load size is the cheapest only between 37.35 and 37.46 $/MWh
here, because a kW of heat pump costs almost the same at either size
(310 against 311 $/y), so no price is tested inside that window.

The heat pump is targeted at the steam's own price (since 2026-11-27,
when it was the electricity price): at 30 $/MWh the target finds that no
heat pump pays, so no heat pump size is designed at all and the design
without one is the only one.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.application.heat_pump_stream_cases import basic_source


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = pytest.mark.synthesis


def _best_by_size(steam_price: float) -> dict[float, float]:
    """The cheapest network's total annual cost at each heat pump size
    (0.0: no heat pump), cheapest size first."""

    source = basic_source()
    for utility in source["utilities"]:
        if utility["type"] == "Hot":
            utility["price"] = {"value": steam_price, "unit": "$/MWh"}
    design = PinchProblem(
        source=source, project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )
    best: dict[float, float] = {}
    for outcome in design.top(1000):
        pumps = outcome.network.heat_pumps
        size = round(pumps[0].load_fraction, 6) if pumps else 0.0
        best.setdefault(size, outcome.network.total_annual_cost)
    return best


# The solver lane: each design searches two heat pump targets, 6 s in all.
@pytest.mark.solver
@pytest.mark.skipif(not _highs_available(), reason="needs highspy")
@pytest.mark.parametrize(
    ("steam_price", "cheapest_first"),
    [(30.0, [0.0]), (50.0, [1.0, 0.4, 0.0])],
)
def test_sizes_rank_by_the_worth_of_the_steam_they_save(steam_price, cheapest_first):
    assert list(_best_by_size(steam_price)) == cheapest_first
