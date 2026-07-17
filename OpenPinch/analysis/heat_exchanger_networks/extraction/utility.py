"""Extract process-to-utility heat exchangers."""

from __future__ import annotations

from typing import Any

from ....domain.enums import HeatExchangerKind, HeatExchangerStreamRole
from ....domain.heat_exchanger import HeatExchanger
from .metadata import _allowed, _index, _optional_float, _optional_int
from .period_state import _cold_utility_period_state, _hot_utility_period_state
from .segment_area import _model_utility_segment_contributions


def _hot_utility_exchangers(
    solved_model: Any,
    *,
    hot_utility: str,
    cold_streams: tuple[str, ...],
    period_ids: tuple[str, ...],
    tolerance: float,
    include_inactive: bool,
) -> tuple[HeatExchanger, ...]:
    q_values = getattr(solved_model, "Q_h", None)
    q_values_by_period = getattr(solved_model, "Q_h_by_period", None)
    if q_values is None and q_values_by_period is None:
        return ()
    exchangers: list[HeatExchanger] = []
    for j, cold_stream in enumerate(cold_streams):
        states = tuple(
            _hot_utility_period_state(
                solved_model,
                period_id=period_id,
                period_idx=period_idx,
                cold_index=j,
                q_values=q_values,
                q_values_by_period=q_values_by_period,
                tolerance=tolerance,
            )
            for period_idx, period_id in enumerate(period_ids)
        )
        if not any(state.active for state in states) and not include_inactive:
            continue
        segment_contributions = _model_utility_segment_contributions(
            solved_model,
            "segment_area_hu_contributions_by_period",
            j,
        )
        area = _optional_float(_index(getattr(solved_model, "area_hu", None), j))
        if segment_contributions:
            area = None
        exchangers.append(
            HeatExchanger(
                exchanger_id=f"hot-utility:{hot_utility}->{cold_stream}",
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream=hot_utility,
                sink_stream=cold_stream,
                source_stream_role=HeatExchangerStreamRole.UTILITY,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                period_states=states,
                area=area,
                segment_area_contributions=segment_contributions,
                match_allowed=_allowed(
                    _index(getattr(solved_model, "z_hu_allowed", None), j)
                ),
            )
        )
    return tuple(exchangers)


def _cold_utility_exchangers(
    solved_model: Any,
    *,
    cold_utility: str,
    hot_streams: tuple[str, ...],
    period_ids: tuple[str, ...],
    tolerance: float,
    include_inactive: bool,
) -> tuple[HeatExchanger, ...]:
    q_values = getattr(solved_model, "Q_c", None)
    q_values_by_period = getattr(solved_model, "Q_c_by_period", None)
    if q_values is None and q_values_by_period is None:
        return ()
    exchangers: list[HeatExchanger] = []
    last_stage = _optional_int(getattr(solved_model, "S", None))
    for i, hot_stream in enumerate(hot_streams):
        states = tuple(
            _cold_utility_period_state(
                solved_model,
                period_id=period_id,
                period_idx=period_idx,
                hot_index=i,
                last_stage=last_stage,
                q_values=q_values,
                q_values_by_period=q_values_by_period,
                tolerance=tolerance,
            )
            for period_idx, period_id in enumerate(period_ids)
        )
        if not any(state.active for state in states) and not include_inactive:
            continue
        segment_contributions = _model_utility_segment_contributions(
            solved_model,
            "segment_area_cu_contributions_by_period",
            i,
        )
        area = _optional_float(_index(getattr(solved_model, "area_cu", None), i))
        if segment_contributions:
            area = None
        exchangers.append(
            HeatExchanger(
                exchanger_id=f"cold-utility:{hot_stream}->{cold_utility}",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream=hot_stream,
                sink_stream=cold_utility,
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.UTILITY,
                period_states=states,
                area=area,
                segment_area_contributions=segment_contributions,
                match_allowed=_allowed(
                    _index(getattr(solved_model, "z_cu_allowed", None), i)
                ),
            )
        )
    return tuple(exchangers)
