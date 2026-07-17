"""Extract process-to-process recovery exchangers."""

from __future__ import annotations

from typing import Any

from ....domain.enums import HeatExchangerKind, HeatExchangerStreamRole
from ....domain.heat_exchanger import HeatExchanger
from ..solver.arrays import PreparedSolverArrays
from .metadata import _allowed, _index, _optional_float, _optional_int, _third_dimension
from .period_state import _recovery_period_state
from .segment_area import _recovery_segment_contributions


def _recovery_exchangers(
    solved_model: Any,
    *,
    solver_arrays: PreparedSolverArrays,
    hot_streams: tuple[str, ...],
    cold_streams: tuple[str, ...],
    stage_total: int | None,
    period_ids: tuple[str, ...],
    tolerance: float,
    include_inactive: bool,
) -> tuple[HeatExchanger, ...]:
    q_values = getattr(solved_model, "Q_r", None)
    q_values_by_period = getattr(solved_model, "Q_r_by_period", None)
    if q_values is None and q_values_by_period is None:
        return ()
    stages = stage_total or _optional_int(getattr(solved_model, "S", None))
    if stages is None:
        stages = _third_dimension(q_values)
    if not stages and q_values_by_period is not None:
        stages = _third_dimension(_index(q_values_by_period, 0))
    exchangers: list[HeatExchanger] = []
    for i, hot_stream in enumerate(hot_streams):
        for j, cold_stream in enumerate(cold_streams):
            for k in range(stages):
                states = tuple(
                    _recovery_period_state(
                        solved_model,
                        period_id=period_id,
                        period_idx=period_idx,
                        hot_index=i,
                        cold_index=j,
                        stage_index=k,
                        q_values=q_values,
                        q_values_by_period=q_values_by_period,
                        tolerance=tolerance,
                    )
                    for period_idx, period_id in enumerate(period_ids)
                )
                if not any(state.active for state in states) and not include_inactive:
                    continue
                stage = k + 1
                segment_contributions = _recovery_segment_contributions(
                    solved_model,
                    solver_arrays,
                    hot_index=i,
                    cold_index=j,
                    stage_index=k,
                    tolerance=tolerance,
                )
                aggregate_area = _optional_float(
                    _index(getattr(solved_model, "area_r", None), i, j, k)
                )
                if segment_contributions:
                    aggregate_area = None
                exchangers.append(
                    HeatExchanger(
                        exchanger_id=f"recovery:{hot_stream}->{cold_stream}:S{stage}",
                        kind=HeatExchangerKind.RECOVERY,
                        source_stream=hot_stream,
                        sink_stream=cold_stream,
                        source_stream_role=HeatExchangerStreamRole.PROCESS,
                        sink_stream_role=HeatExchangerStreamRole.PROCESS,
                        stage=stage,
                        period_states=states,
                        area=aggregate_area,
                        segment_area_contributions=segment_contributions,
                        match_allowed=_allowed(
                            _index(getattr(solved_model, "z_allowed", None), i, j, k)
                        ),
                    )
                )
    return tuple(exchangers)
