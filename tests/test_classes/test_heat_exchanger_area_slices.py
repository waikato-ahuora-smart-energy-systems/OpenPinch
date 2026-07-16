from __future__ import annotations

import pytest
from hypothesis import given, settings

from OpenPinch.classes._heat_exchanger.area import HeatExchangerAreaSlice
from OpenPinch.classes._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.classes.heat_exchanger import HeatExchanger
from OpenPinch.lib.enums import HeatExchangerKind, HeatExchangerStreamRole
from tests.strategies.heat_exchangers import heat_exchangers_with_area_slices


def _slice(*, period: str, duty: float, area: float) -> HeatExchangerAreaSlice:
    return HeatExchangerAreaSlice(
        period=period,
        hot_segment_identity="hot.S1",
        cold_segment_identity="cold.S1",
        duty=duty,
        hot_inlet_temperature=400.0,
        hot_outlet_temperature=380.0,
        cold_inlet_temperature=300.0,
        cold_outlet_temperature=320.0,
        hot_htc=2.0,
        cold_htc=2.0,
        overall_htc=1.0,
        lmtd=duty / area,
        area=area,
    )


def _exchanger(*, slices=(), area=None) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id="recovery:hot->cold:S1",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="hot",
        sink_stream="cold",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        period_states=(
            HeatExchangerPeriodState(period_id="0", period_idx=0, duty=10.0),
        ),
        area=area,
        segment_area_contributions=tuple(slices),
    )


def test_segment_area_totals_and_design_area_are_parent_properties():
    exchanger = _exchanger(
        slices=(
            _slice(period="normal", duty=4.0, area=2.0),
            _slice(period="normal", duty=6.0, area=3.0),
            _slice(period="peak", duty=10.0, area=7.0),
        )
    )

    assert exchanger.has_segment_area_contributions
    assert exchanger.segment_duty_by_period == {"normal": 10.0, "peak": 10.0}
    assert exchanger.segment_area_by_period == {"normal": 5.0, "peak": 7.0}
    assert exchanger.segment_design_area == 7.0
    assert exchanger.area == 7.0


def test_segment_area_rejects_inconsistent_parent_area():
    with pytest.raises(ValueError, match="maximum period-total segment area"):
        _exchanger(
            slices=(_slice(period="normal", duty=10.0, area=5.0),),
            area=6.0,
        )


def test_constant_cp_exchanger_area_behavior_is_unchanged():
    exchanger = _exchanger(area=6.0)

    assert not exchanger.has_segment_area_contributions
    assert exchanger.segment_duty_by_period == {}
    assert exchanger.segment_area_by_period == {}
    assert exchanger.segment_design_area is None
    assert exchanger.area == 6.0


@given(heat_exchangers_with_area_slices())
@settings(max_examples=30)
def test_area_sliced_exchanger_json_round_trip_and_aggregation_invariants(exchanger):
    restored = HeatExchanger.model_validate_json(exchanger.model_dump_json())

    assert restored == exchanger
    assert tuple(restored.segment_area_contributions) == tuple(
        exchanger.segment_area_contributions
    )
    assert restored.segment_duty_by_period == exchanger.segment_duty_by_period
    assert restored.segment_area_by_period == exchanger.segment_area_by_period
    assert restored.segment_design_area == pytest.approx(
        max(restored.segment_area_by_period.values())
    )
    assert restored.area == pytest.approx(restored.segment_design_area)
