"""A fitted multiperiod design's manifest is its synthesis's.

A design on the median basis synthesises on the basis case and fits each
network to every period. The result's manifest described the fit, one
network evolution task at the selected multiplier with width 1 and depth 1,
not the search the call asked for (review finding 37).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.test_site_heat_exchanger_network import _solvers_available


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_fitted_designs_manifest_is_the_search_asked_for():
    problem = PinchProblem("crude_preheat_train_multiperiod.json")

    view = problem.design.heat_exchanger_network(width=0, depth=0)

    manifest = view.result.manifest
    assert (manifest.search_width, manifest.search_depth) == (0, 0)
