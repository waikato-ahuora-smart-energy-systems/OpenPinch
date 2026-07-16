from __future__ import annotations

from hypothesis import strategies as st

from OpenPinch.classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerAreaSlice,
    HeatExchangerPeriodState,
)
from OpenPinch.lib.enums import HeatExchangerKind, HeatExchangerStreamRole

_POSITIVE_FLOATS = st.floats(
    min_value=0.1,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def heat_exchangers_with_area_slices(draw) -> HeatExchanger:
    """Generate valid recovery exchangers with ordered multiperiod area slices."""

    period_count = draw(st.integers(min_value=1, max_value=3))
    slices = []
    period_duties: list[float] = []
    for period_index in range(period_count):
        slice_count = draw(st.integers(min_value=1, max_value=4))
        period_duty = 0.0
        for slice_index in range(slice_count):
            duty = draw(_POSITIVE_FLOATS)
            hot_htc = draw(_POSITIVE_FLOATS)
            cold_htc = draw(_POSITIVE_FLOATS)
            overall_htc = 1.0 / (1.0 / hot_htc + 1.0 / cold_htc)
            lmtd = draw(_POSITIVE_FLOATS)
            hot_outlet = draw(
                st.floats(
                    min_value=250.0,
                    max_value=600.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            cold_inlet = draw(
                st.floats(
                    min_value=150.0,
                    max_value=hot_outlet - 0.1,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            slices.append(
                HeatExchangerAreaSlice(
                    period=f"period-{period_index}",
                    hot_segment_identity=f"hot.S{slice_index + 1}",
                    cold_segment_identity=f"cold.S{slice_index + 1}",
                    duty=duty,
                    hot_inlet_temperature=hot_outlet + 10.0,
                    hot_outlet_temperature=hot_outlet,
                    cold_inlet_temperature=cold_inlet,
                    cold_outlet_temperature=cold_inlet + 10.0,
                    hot_htc=hot_htc,
                    cold_htc=cold_htc,
                    overall_htc=overall_htc,
                    lmtd=lmtd,
                    area=duty / overall_htc / lmtd,
                )
            )
            period_duty += duty
        period_duties.append(period_duty)

    return HeatExchanger(
        exchanger_id="generated-recovery",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="hot",
        sink_stream="cold",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        period_states=tuple(
            HeatExchangerPeriodState(
                period_id=f"period-{period_index}",
                period_idx=period_index,
                duty=period_duty,
            )
            for period_index, period_duty in enumerate(period_duties)
        ),
        segment_area_contributions=tuple(slices),
    )
