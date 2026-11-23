"""The heuristic designs the heat pump case split at the process pinch.

Half load at multiplier 0.75 is the one row of the heat pump table that
neither method could design while the case was split at the condenser
level. Split at the process pinch, the above side holds the condenser
level as a second pinch, which the heuristic's regions handle.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod
from tests.analysis.heat_exchanger_networks.test_heat_pump_case_pinch_split import (
    MULTIPLIER,
    _half_load_case,
    _highs_available,
)

pytestmark = pytest.mark.synthesis


@pytest.mark.skipif(not _highs_available(), reason="needs highspy")
def test_the_heuristic_designs_a_heat_pump_case_whose_condenser_is_the_hot_pinch():
    design = PinchProblem(
        source=_half_load_case(), project_name="half"
    ).design.heat_exchanger_network(
        width=0,
        depth=0,
        dt_cont_multipliers=[MULTIPLIER],
        max_parallel=1,
        options={"HENS_PDM_MILP": False},
    )

    best = design.top(1)[0]
    assert best.method is HeatExchangerNetworkDesignMethod.HeuristicPinchDesign
    assert best.network.summary_metrics["hot_utility_load"] == pytest.approx(
        325.0, abs=0.5
    )
