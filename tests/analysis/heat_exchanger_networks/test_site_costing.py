"""Site cost of a site network: its capital plus the site utility balance.

A site network's heaters and coolers are grouped by process and level; the
site utility accounting balances the levels, so steam raised on a main that
another process uses is credited instead of charged as a cooler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import OpenPinch
from OpenPinch.analysis.heat_exchanger_networks.site.costing import (
    priced_site_network,
    site_loads,
    site_utility_levels,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.site import (
    LevelLoad,
    SiteUtilityLevel,
    SiteUtilityLevels,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

ZONAL_SITE = (
    Path(OpenPinch.__file__).parent / "tutorials" / "sample_cases" / "zonal_site.json"
)


def _state(duty: float) -> HeatExchangerPeriodState:
    return HeatExchangerPeriodState(period_id="0", period_idx=0, duty=duty)


def _heater(utility: str, cold: str, duty: float) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"hot-utility:{utility}->{cold}",
        kind=HeatExchangerKind.HOT_UTILITY,
        source_stream=utility,
        sink_stream=cold,
        source_stream_role=StreamID.Utility,
        sink_stream_role=StreamID.Process,
        period_states=(_state(duty),),
    )


def _cooler(hot: str, utility: str, duty: float) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"cold-utility:{hot}->{utility}",
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream=hot,
        sink_stream=utility,
        source_stream_role=StreamID.Process,
        sink_stream_role=StreamID.Utility,
        period_states=(_state(duty),),
    )


def test_site_levels_are_the_problems_utilities_with_their_annual_prices():
    """zonal_site: VHPS is bought, four mains (HPS, MPS, LPS, HW) are both
    used and raised, and cooling water takes heat. Every price is 30 $/MWh,
    249 $/kW/y over 8300 h."""
    arrays = problem_to_solver_arrays(PinchProblem(source=str(ZONAL_SITE)), 10.0)

    levels = site_utility_levels(arrays)

    kinds = {level.name: level.kind for level in levels.levels}
    assert kinds == {
        "VHPS": "hot",
        "HPS": "both",
        "MPS": "both",
        "LPS": "both",
        "HW": "both",
        "CW": "cold",
    }
    hw = levels.level("HW")
    assert (hw.t_high, hw.t_low) == pytest.approx((90.0, 70.0), abs=1e-9)
    assert hw.import_price == pytest.approx((249.0,), rel=1e-12)
    assert hw.dump_price == pytest.approx((249.0,), rel=1e-12)
    assert levels.period_weights == pytest.approx((1.0,), rel=1e-12)


def test_a_site_networks_utility_duties_are_grouped_by_process_and_level():
    network = HeatExchangerNetwork(
        exchangers=(
            _heater("Hot Utility.LPS", "Site/A.C1", 10.0),
            _heater("Hot Utility.LPS", "Site/A.C2", 5.0),
            _cooler("Site/B.H1", "Cold Utility.LPS", 25.0),
            _cooler("Site/B.H2", "Cold Utility.CW", 3.0),
        )
    )

    loads = site_loads(network, ("Site/A", "Site/B"))

    assert loads == {
        "Site/A": {"LPS": LevelLoad(use=(15.0,), generation=(0.0,))},
        "Site/B": {
            "LPS": LevelLoad(use=(0.0,), generation=(25.0,)),
            "CW": LevelLoad(use=(0.0,), generation=(3.0,)),
        },
    }


def _levels() -> SiteUtilityLevels:
    return SiteUtilityLevels(
        levels=(
            SiteUtilityLevel(
                name="LPS",
                kind="both",
                t_high=150.1,
                t_low=150.0,
                import_price=(30.0,),
                dump_price=(30.0,),
            ),
            SiteUtilityLevel(
                name="CW", kind="cold", t_high=25.0, t_low=20.0, dump_price=(5.0,)
            ),
        )
    )


def test_a_site_network_costs_its_capital_plus_the_site_balance():
    """A uses 30 kW of LPS and B raises 10: priced as heaters and coolers
    that is 40 kW at 30 $; on the site B's steam meets A's use, so LPS buys
    20 kW (600 $), and the network's capital (1000 $) is added."""
    network = HeatExchangerNetwork(
        exchangers=(
            _heater("Hot Utility.LPS", "Site/A.C1", 30.0),
            _cooler("Site/B.H1", "Cold Utility.LPS", 10.0),
        ),
        total_annual_cost=2200.0,
        utility_cost=1200.0,
        capital_cost=1000.0,
    )

    priced = priced_site_network(network, ("Site/A", "Site/B"), _levels())

    metrics = priced.summary_metrics
    assert metrics["site_operating_cost"] == pytest.approx(600.0, abs=1e-6)
    assert metrics["site_capital_cost"] == pytest.approx(1000.0, abs=1e-9)
    assert metrics["site_total_annual_cost"] == pytest.approx(1600.0, abs=1e-6)
    balance = priced.source_metadata["site_utility_balance"]
    assert balance.level("LPS").imported == pytest.approx(20.0, abs=1e-9)
    assert priced.total_annual_cost == pytest.approx(2200.0, abs=1e-9)
