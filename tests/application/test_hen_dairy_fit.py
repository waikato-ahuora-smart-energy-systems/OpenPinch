"""The packaged dairy chilling case designs at its own approach.

Its Ice-water return and Glycol pair needs only 3.5 K, but stream-end
padding at twice THERMAL_DT_CONT (10 K) dropped that cooler, so the fit to
every period failed and every network fell back to 1 K; overdesign then
rated that fallback network at the design approach and failed.
"""

from __future__ import annotations

import warnings

import pytest

from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
    DEFAULT_FEASIBILITY_APPROACH,
)
from OpenPinch.application.problem import PinchProblem

_OPTIONS = {"HENS_MAX_PARALLEL": 1}


def _fit_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if "fit to every period" in str(w.message)]


@pytest.mark.solver
def test_the_default_design_keeps_its_approach_without_a_fallback():
    problem = PinchProblem("dairy_chilling_multiperiod.json")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view = problem.design.heat_exchanger_network(width=0, depth=0, options=_OPTIONS)

    assert _fit_warnings(caught) == []
    approach = view.selected_network.summary_metrics.get("approach_temperature")
    assert approach != DEFAULT_FEASIBILITY_APPROACH


@pytest.mark.solver
def test_a_fixed_overdesign_margin_rates_the_default_design():
    problem = PinchProblem("dairy_chilling_multiperiod.json")

    view = problem.design.heat_exchanger_network(
        width=0, depth=0, overdesign_percentage=10, options=_OPTIONS
    )

    assert view.selected_network.overdesign.percentage == pytest.approx(10.0)
