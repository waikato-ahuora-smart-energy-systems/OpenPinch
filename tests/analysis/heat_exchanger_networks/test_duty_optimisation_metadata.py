"""A duty-optimised network records each period's stream data.

Verification checks each stream's heat balance per period from the
network's ``*_by_period`` metadata. Duty optimisation wrote only the first
period's temperatures and duties, so on a multiperiod network that check
found nothing to compare and passed a stream left short of its target.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
    _source_metadata,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

PERIODS = ("low", "high")


def _metadata(hot_htc=(1.0, 1.5)):
    """One hot stream H1, 100 to 50 C, CP 10 then 12 kW/K (500 and 600 kW)."""
    arrays = {
        "f_h_period": [[10.0], [12.0]],
        "f_c_period": [[], []],
        "htc_h_period": [[hot_htc[0]], [hot_htc[1]]],
        "htc_c_period": [[], []],
        "htc_hu_period": [[], []],
        "htc_cu_period": [[1.0], [1.0]],
        "T_h_in_period": [[100.0], [100.0]],
        "T_h_out_period": [[50.0], [50.0]],
        "T_c_in_period": [[], []],
        "T_c_out_period": [[], []],
    }
    model = SimpleNamespace(
        name="fit",
        recovery_dt=[],
        hot_profiles=[
            [SimpleNamespace(total=500.0)],
            [SimpleNamespace(total=600.0)],
        ],
        cold_profiles=[[], []],
    )
    return _source_metadata(model, SimpleNamespace(arrays=arrays))


def _cooler(duties, area=None):
    return HeatExchanger(
        exchanger_id="cooler",
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream="H1",
        sink_stream="CW",
        source_stream_role=StreamID.Process,
        sink_stream_role=StreamID.Utility,
        area=area,
        period_states=tuple(
            HeatExchangerPeriodState(
                period_id=period,
                period_idx=n,
                duty=duty,
                source_inlet_temperature=100.0,
                source_outlet_temperature=50.0,
                sink_inlet_temperature=20.0,
                sink_outlet_temperature=30.0,
            )
            for n, (period, duty) in enumerate(zip(PERIODS, duties, strict=True))
        ),
    )


def _network(duties, *, area=None, hot_htc=(1.0, 1.5)):
    return HeatExchangerNetwork(
        exchangers=(_cooler(duties, area),),
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_process_streams": {},
                "hot_utilities": {},
                "cold_utilities": {"CW": 0},
            }
        },
        source_metadata=_metadata(hot_htc),
    )


def test_the_metadata_holds_each_periods_temperatures_duties_and_coefficients():
    metadata = _metadata()

    assert metadata["hot_stream_total_duties_by_period"] == [[500.0], [600.0]]
    assert metadata["hot_stream_supply_temperatures_by_period"] == [[100.0], [100.0]]
    assert metadata["hot_stream_target_temperatures_by_period"] == [[50.0], [50.0]]
    assert metadata["hot_stream_heat_transfer_coefficients_by_period"] == [
        [1.0],
        [1.5],
    ]


def test_a_multiperiod_stream_short_of_its_target_fails_verification():
    failures = verify_network_feasibility(_network((500.0, 200.0)))

    assert any("H1" in failure and "'high'" in failure for failure in failures)


def test_the_area_check_uses_each_periods_heat_transfer_coefficients():
    """The cooler's area serves period "low" (h = 1, U = 0.5); with h = 0.25
    in "high" (U = 0.2) it is short there, whatever period 0's h is."""
    sized_for_low = 500.0 / 0.5 / ((70.0 - 30.0) / math.log(70.0 / 30.0))

    failures = verify_network_feasibility(
        _network((500.0, 500.0), area=sized_for_low * 1.01, hot_htc=(1.0, 0.25))
    )

    assert any("below required area" in failure for failure in failures)
