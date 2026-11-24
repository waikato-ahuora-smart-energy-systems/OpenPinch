"""The as-piped heat pump of a network of several periods is one machine.

Its design serves the coolers and heaters of every period, each as it runs
there (``as_piped_period_profiles``), so its streams are known in each
period: the target holds every period's outputs.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.context import ensure_target_results
from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
)
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as case,
)

# One evaluation takes the search's starting point as it is.
BUDGET = {"maximum_evaluations": 1}


def test_the_as_piped_heat_pump_of_two_periods_holds_each_periods_outputs():
    problem = case.two_period_problem()
    ensure_target_results(problem, {}, None)

    machine = as_piped_heat_pump(problem, case.two_period_network(), args=BUDGET)

    assert set(machine.period_outputs or {}) == set(case.PERIODS)
