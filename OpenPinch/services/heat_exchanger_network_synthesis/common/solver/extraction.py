"""Extract private solver arrays into OpenPinch heat exchanger network records."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .....classes._heat_exchanger.period_state import HeatExchangerPeriodState
from .....classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerStreamRole,
)
from .....classes.heat_exchanger_network import HeatExchangerNetwork
from .....lib.config import tol
from .....lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from ..indexing import ordered_mapping_keys
from .arrays import SEGMENT_PROFILE_VERSION, PreparedSolverArrays
from .piecewise import (
    duty_aligned_area_contributions,
    profile_from_solver_arrays,
)

_SOLVER_NEGATIVE_DUTY_NOISE = 1e-4


def extract_heat_exchanger_network(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
    *,
    run_id: str,
    task_id: str | None = None,
    method: str | None = None,
    stage_count: int | None = None,
    objective_value: float | None = None,
    tolerance: float = tol,
    include_inactive: bool = False,
) -> HeatExchangerNetwork:
    """Convert a solved model into identity-labelled OpenPinch records."""

    hot_streams = _identities_by_axis(solver_arrays, "hot_process_streams")
    cold_streams = _identities_by_axis(solver_arrays, "cold_process_streams")
    hot_utilities = _identities_by_axis(solver_arrays, "hot_utilities")
    cold_utilities = _identities_by_axis(solver_arrays, "cold_utilities")
    stage_total = stage_count or _optional_int(getattr(solved_model, "S", None))
    period_ids = _period_ids(solved_model, solver_arrays)

    exchangers: list[HeatExchanger] = []
    exchangers.extend(
        _recovery_exchangers(
            solved_model,
            solver_arrays=solver_arrays,
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            stage_total=stage_total,
            period_ids=period_ids,
            tolerance=tolerance,
            include_inactive=include_inactive,
        )
    )
    exchangers.extend(
        _hot_utility_exchangers(
            solved_model,
            hot_utility=_single_utility(hot_utilities, "hot utility"),
            cold_streams=cold_streams,
            period_ids=period_ids,
            tolerance=tolerance,
            include_inactive=include_inactive,
        )
    )
    exchangers.extend(
        _cold_utility_exchangers(
            solved_model,
            cold_utility=_single_utility(cold_utilities, "cold utility"),
            hot_streams=hot_streams,
            period_ids=period_ids,
            tolerance=tolerance,
            include_inactive=include_inactive,
        )
    )

    total_annual_cost = _optional_float(getattr(solved_model, "TAC", None))
    objective = (
        objective_value
        if objective_value is not None
        else _optional_float(getattr(solved_model, "TAC_model", total_annual_cost))
    )
    return HeatExchangerNetwork(
        exchangers=tuple(exchangers),
        run_id=run_id,
        task_id=task_id,
        method=method,
        stage_count=stage_total,
        objective_value=objective,
        total_annual_cost=total_annual_cost,
        utility_cost=_utility_cost(solved_model),
        capital_cost=_capital_cost(solved_model),
        summary_metrics=_summary_metrics(solved_model),
        solver_axis_metadata={
            "axis_maps": solver_arrays.axis_maps,
            "source_array_names": (
                "Q_r",
                "Q_h",
                "Q_c",
                "T_h",
                "T_c",
                "T_h_out_x",
                "T_c_out_y",
                "area_r",
                "area_hu",
                "area_cu",
                "z",
                "z_hu",
                "z_cu",
            ),
        },
        source_metadata={
            "solver_model_class": type(solved_model).__name__,
            "solver_model_name": getattr(solved_model, "name", None),
            "solver_framework": getattr(solved_model, "framework", None),
            "solver_non_isothermal_model": bool(
                getattr(solved_model, "non_isothermal_model", False)
            ),
            "solver_dTmin": _optional_float(getattr(solved_model, "dTmin", None)),
            "segment_profile_version": SEGMENT_PROFILE_VERSION,
            "operating_periods": _operating_state_metadata(solved_model),
            "hot_stream_heat_capacity_flowrates": _float_list(
                getattr(solved_model, "f_h", None)
            ),
            "hot_stream_heat_capacity_flowrates_by_period": _float_matrix(
                getattr(solved_model, "f_h_period", None)
            ),
            "cold_stream_heat_capacity_flowrates": _float_list(
                getattr(solved_model, "f_c", None)
            ),
            "cold_stream_heat_capacity_flowrates_by_period": _float_matrix(
                getattr(solved_model, "f_c_period", None)
            ),
            "hot_stream_heat_transfer_coefficients": _float_list(
                getattr(solved_model, "htc_h", None)
            ),
            "cold_stream_heat_transfer_coefficients": _float_list(
                getattr(solved_model, "htc_c", None)
            ),
            "hot_utility_heat_transfer_coefficients": _float_list(
                getattr(solved_model, "htc_hu", None)
            ),
            "cold_utility_heat_transfer_coefficients": _float_list(
                getattr(solved_model, "htc_cu", None)
            ),
            "hot_stream_supply_temperatures": _float_list(
                getattr(solved_model, "T_h_in", None)
            ),
            "hot_stream_target_temperatures": _float_list(
                getattr(solved_model, "T_h_out", None)
            ),
            "cold_stream_supply_temperatures": _float_list(
                getattr(solved_model, "T_c_in", None)
            ),
            "cold_stream_target_temperatures": _float_list(
                getattr(solved_model, "T_c_out", None)
            ),
            "hot_stream_total_duties": _segment_parent_total_duties(
                solver_arrays, "hot"
            ),
            "cold_stream_total_duties": _segment_parent_total_duties(
                solver_arrays, "cold"
            ),
            "hot_stage_boundary_temperatures": _boundary_temperature_matrix(
                getattr(solved_model, "T_h", None),
                rows=len(hot_streams),
                columns=(stage_total or 0) + 1,
            ),
            "cold_stage_boundary_temperatures": _boundary_temperature_matrix(
                getattr(solved_model, "T_c", None),
                rows=len(cold_streams),
                columns=(stage_total or 0) + 1,
            ),
        },
    )


def _segment_parent_total_duties(
    solver_arrays: PreparedSolverArrays,
    side: str,
) -> list[float]:
    values = solver_arrays.arrays.get(f"{side}_segment_duty_period")
    if values is None:
        return []
    return np.sum(np.asarray(values, dtype=float)[0], axis=1).tolist()


def extract_network_synthesis_result(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
    *,
    run_id: str,
    task_id: str | None = None,
    problem_id: str | None = None,
    workspace_variant: str | None = None,
    period_id: str | None = None,
    solver_name: str | None = None,
    solver_status: str | None = None,
    method: str | None = None,
    stage_count: int | None = None,
    objective_value: float | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Build the result data that may cross the OpenPinch service boundary."""

    result_method = _result_method(method)
    network = extract_heat_exchanger_network(
        solved_model,
        solver_arrays,
        run_id=run_id,
        task_id=task_id,
        method=result_method,
        stage_count=stage_count,
        objective_value=objective_value,
    )
    objective_values = {}
    if network.total_annual_cost is not None:
        objective_values["total_annual_cost"] = network.total_annual_cost
    if network.objective_value is not None:
        objective_values["model_objective_value"] = network.objective_value
    return HeatExchangerNetworkSynthesisResult(
        network=network,
        run_id=run_id,
        task_id=task_id,
        problem_id=problem_id,
        workspace_variant=workspace_variant,
        period_id=period_id,
        solver_name=solver_name,
        solver_status=solver_status,
        method=result_method,
        stage_count=network.stage_count,
        objective_values=objective_values,
    )


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


def _recovery_period_state(
    solved_model: Any,
    *,
    period_id: str,
    period_idx: int,
    hot_index: int,
    cold_index: int,
    stage_index: int,
    q_values: Any,
    q_values_by_period: Any,
    tolerance: float,
) -> HeatExchangerPeriodState:
    duty_source = (
        _index(
            q_values_by_period,
            period_idx,
            hot_index,
            cold_index,
            stage_index,
        )
        if q_values_by_period is not None
        else _index(q_values, hot_index, cold_index, stage_index)
    )
    duty = _normalized_solver_duty(duty_source, tolerance=tolerance)
    active = _is_active(
        duty,
        _index(getattr(solved_model, "z", None), hot_index, cold_index, stage_index),
        tolerance,
    )
    hot_boundaries = getattr(solved_model, "T_h_by_period", None)
    cold_boundaries = getattr(solved_model, "T_c_by_period", None)
    source_inlet = _optional_float(
        _index(hot_boundaries, period_idx, hot_index, stage_index)
        if hot_boundaries is not None
        else _index(getattr(solved_model, "T_h", None), hot_index, stage_index)
    )
    sink_inlet = _optional_float(
        _index(cold_boundaries, period_idx, cold_index, stage_index + 1)
        if cold_boundaries is not None
        else _index(
            getattr(solved_model, "T_c", None),
            cold_index,
            stage_index + 1,
        )
    )
    return HeatExchangerPeriodState(
        period_id=period_id,
        period_idx=period_idx,
        duty=duty,
        active=active,
        approach_temperatures=_approach_temperatures(
            solved_model,
            hot_index,
            cold_index,
            stage_index,
            period_idx=period_idx,
        ),
        source_split_fraction=_recovery_split_fraction(
            solved_model,
            side="hot",
            period_idx=period_idx,
            hot_index=hot_index,
            cold_index=cold_index,
            stage_index=stage_index,
            tolerance=tolerance,
        ),
        sink_split_fraction=_recovery_split_fraction(
            solved_model,
            side="cold",
            period_idx=period_idx,
            hot_index=hot_index,
            cold_index=cold_index,
            stage_index=stage_index,
            tolerance=tolerance,
        ),
        source_inlet_temperature=source_inlet,
        source_outlet_temperature=_hot_recovery_outlet(
            solved_model,
            hot_index,
            cold_index,
            stage_index,
            duty,
            period_idx=period_idx,
        ),
        sink_inlet_temperature=sink_inlet,
        sink_outlet_temperature=_cold_recovery_outlet(
            solved_model,
            hot_index,
            cold_index,
            stage_index,
            duty,
            period_idx=period_idx,
        ),
    )


def _recovery_segment_contributions(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
    *,
    hot_index: int,
    cold_index: int,
    stage_index: int,
    tolerance: float,
):
    model_contributions = _model_recovery_segment_contributions(
        solved_model,
        hot_index=hot_index,
        cold_index=cold_index,
        stage_index=stage_index,
    )
    if model_contributions:
        return model_contributions
    arrays = solver_arrays.arrays
    required = {"hot_segment_count", "cold_segment_count"}
    if not required.issubset(arrays):
        return ()
    if (
        int(arrays["hot_segment_count"][hot_index]) == 1
        and int(arrays["cold_segment_count"][cold_index]) == 1
    ):
        return ()

    period_ids = [str(value) for value in arrays.get("period_ids", ["0"])]
    q_by_period = getattr(solved_model, "Q_r_by_period", None)
    hot_temperatures = getattr(solved_model, "T_h_by_period", None)
    cold_temperatures = getattr(solved_model, "T_c_by_period", None)
    contributions = []
    for period_index, period_id in enumerate(period_ids):
        duty_value = (
            _index(q_by_period, period_index, hot_index, cold_index, stage_index)
            if q_by_period is not None
            else _index(
                getattr(solved_model, "Q_r", None),
                hot_index,
                cold_index,
                stage_index,
            )
        )
        duty = _optional_float(duty_value) or 0.0
        if duty <= tolerance:
            continue
        hot_inlet = _optional_float(
            _index(hot_temperatures, period_index, hot_index, stage_index)
            if hot_temperatures is not None
            else _index(getattr(solved_model, "T_h", None), hot_index, stage_index)
        )
        cold_inlet = _optional_float(
            _index(cold_temperatures, period_index, cold_index, stage_index + 1)
            if cold_temperatures is not None
            else _index(getattr(solved_model, "T_c", None), cold_index, stage_index + 1)
        )
        if hot_inlet is None or cold_inlet is None:
            raise ValueError(
                "Segment-aware HEN extraction requires stage boundary temperatures."
            )
        contributions.extend(
            duty_aligned_area_contributions(
                profile_from_solver_arrays(
                    solver_arrays,
                    side="hot",
                    parent_index=hot_index,
                    period_index=period_index,
                ),
                profile_from_solver_arrays(
                    solver_arrays,
                    side="cold",
                    parent_index=cold_index,
                    period_index=period_index,
                ),
                duty=duty,
                hot_inlet_temperature=hot_inlet,
                cold_inlet_temperature=cold_inlet,
                period=period_id,
                tolerance=tolerance,
            )
        )
    return tuple(contributions)


def _model_recovery_segment_contributions(
    solved_model: Any,
    *,
    hot_index: int,
    cold_index: int,
    stage_index: int,
):
    grid = getattr(solved_model, "segment_area_contributions_by_period", None)
    if grid is None:
        return ()
    contributions = []
    for period_index in range(len(grid)):
        values = _index(
            grid,
            period_index,
            hot_index,
            cold_index,
            stage_index,
        )
        if values:
            contributions.extend(values)
    return tuple(contributions)


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


def _hot_utility_period_state(
    solved_model: Any,
    *,
    period_id: str,
    period_idx: int,
    cold_index: int,
    q_values: Any,
    q_values_by_period: Any,
    tolerance: float,
) -> HeatExchangerPeriodState:
    duty = _normalized_solver_duty(
        _index(q_values_by_period, period_idx, cold_index)
        if q_values_by_period is not None
        else _index(q_values, cold_index),
        tolerance=tolerance,
    )
    source_inlet = _period_value(
        solved_model,
        "T_hu_in_period",
        "T_hu_in",
        period_idx,
        0,
    )
    source_outlet = _utility_outlet_temperature(
        solved_model,
        side="hot",
        period_idx=period_idx,
        match_index=cold_index,
        duty=duty,
    )
    sink_inlet = _period_value(
        solved_model,
        "T_c_by_period",
        "T_c",
        period_idx,
        cold_index,
        0,
    )
    sink_outlet = _period_value(
        solved_model,
        "T_c_out_period",
        "T_c_out",
        period_idx,
        cold_index,
    )
    active = _is_active(
        duty,
        _index(getattr(solved_model, "z_hu", None), cold_index),
        tolerance,
    )
    return HeatExchangerPeriodState(
        period_id=period_id,
        period_idx=period_idx,
        duty=duty,
        active=active,
        approach_temperatures=(),
        source_inlet_temperature=source_inlet,
        source_outlet_temperature=source_outlet,
        sink_inlet_temperature=sink_inlet,
        sink_outlet_temperature=sink_outlet,
    )


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


def _cold_utility_period_state(
    solved_model: Any,
    *,
    period_id: str,
    period_idx: int,
    hot_index: int,
    last_stage: int | None,
    q_values: Any,
    q_values_by_period: Any,
    tolerance: float,
) -> HeatExchangerPeriodState:
    duty = _normalized_solver_duty(
        _index(q_values_by_period, period_idx, hot_index)
        if q_values_by_period is not None
        else _index(q_values, hot_index),
        tolerance=tolerance,
    )
    source_inlet = _period_value(
        solved_model,
        "T_h_by_period",
        "T_h",
        period_idx,
        hot_index,
        last_stage,
    )
    source_outlet = _period_value(
        solved_model,
        "T_h_out_period",
        "T_h_out",
        period_idx,
        hot_index,
    )
    sink_inlet = _period_value(
        solved_model,
        "T_cu_in_period",
        "T_cu_in",
        period_idx,
        0,
    )
    sink_outlet = _utility_outlet_temperature(
        solved_model,
        side="cold",
        period_idx=period_idx,
        match_index=hot_index,
        duty=duty,
    )
    active = _is_active(
        duty,
        _index(getattr(solved_model, "z_cu", None), hot_index),
        tolerance,
    )
    return HeatExchangerPeriodState(
        period_id=period_id,
        period_idx=period_idx,
        duty=duty,
        active=active,
        approach_temperatures=(),
        source_inlet_temperature=source_inlet,
        source_outlet_temperature=source_outlet,
        sink_inlet_temperature=sink_inlet,
        sink_outlet_temperature=sink_outlet,
    )


def _model_utility_segment_contributions(
    solved_model: Any,
    attribute_name: str,
    parent_index: int,
):
    grid = getattr(solved_model, attribute_name, None)
    if grid is None:
        return ()
    contributions = []
    for period_values in grid:
        values = _index(period_values, parent_index)
        if values:
            contributions.extend(values)
    return tuple(contributions)


def _identities_by_axis(
    solver_arrays: PreparedSolverArrays,
    axis_name: str,
) -> tuple[str, ...]:
    axis_map = solver_arrays.axis_maps.get(axis_name, {})
    if axis_map:
        return ordered_mapping_keys(axis_map)
    return tuple(solver_arrays.stream_identities.get(axis_name, ())) or tuple(
        solver_arrays.utility_identities.get(axis_name, ())
    )


def _single_utility(identities: tuple[str, ...], label: str) -> str:
    if not identities:
        raise ValueError(
            f"solved heat exchanger network extraction requires at least one {label}."
        )
    return identities[0]


def _hot_recovery_outlet(
    solved_model: Any,
    i: int,
    j: int,
    k: int,
    duty: float,
    *,
    period_idx: int = 0,
) -> float | None:
    explicit = _optional_float(
        _index(
            getattr(solved_model, "T_h_out_x_by_period", None),
            period_idx,
            i,
            j,
            k,
        )
    )
    if explicit is None:
        explicit = _optional_float(
            _index(getattr(solved_model, "T_h_out_x", None), i, j, k)
        )
    if explicit is not None:
        return explicit
    inlet = _period_value(
        solved_model,
        "T_h_by_period",
        "T_h",
        period_idx,
        i,
        k,
    )
    heat_capacity = _period_value(
        solved_model,
        "f_h_period",
        "f_h",
        period_idx,
        i,
    )
    if inlet is not None and heat_capacity is not None and heat_capacity > 0.0:
        return inlet - duty / heat_capacity
    return _period_value(
        solved_model,
        "T_h_by_period",
        "T_h",
        period_idx,
        i,
        k + 1,
    )


def _recovery_split_fraction(
    solved_model: Any,
    *,
    side: str,
    period_idx: int,
    hot_index: int,
    cold_index: int,
    stage_index: int,
    tolerance: float,
) -> float | None:
    if side == "hot":
        explicit = _optional_float(
            _index(
                getattr(solved_model, "T_h_out_x_by_period", None),
                period_idx,
                hot_index,
                cold_index,
                stage_index,
            )
        )
        if explicit is None:
            explicit = _optional_float(
                _index(
                    getattr(solved_model, "T_h_out_x", None),
                    hot_index,
                    cold_index,
                    stage_index,
                )
            )
        if explicit is None:
            return None
        value = _period_value(
            solved_model,
            "X_by_period",
            "X",
            period_idx,
            hot_index,
            cold_index,
            stage_index,
        )
    elif side == "cold":
        explicit = _optional_float(
            _index(
                getattr(solved_model, "T_c_out_y_by_period", None),
                period_idx,
                cold_index,
                hot_index,
                stage_index,
            )
        )
        if explicit is None:
            explicit = _optional_float(
                _index(
                    getattr(solved_model, "T_c_out_y", None),
                    cold_index,
                    hot_index,
                    stage_index,
                )
            )
        if explicit is None:
            return None
        value = _period_value(
            solved_model,
            "Y_by_period",
            "Y",
            period_idx,
            cold_index,
            hot_index,
            stage_index,
        )
    else:
        raise ValueError("side must be 'hot' or 'cold'")
    if value is None:
        return None
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(
            f"solved {side} split fraction {value:.6g} is outside zero to one"
        )
    return min(max(value, 0.0), 1.0)


def _cold_recovery_outlet(
    solved_model: Any,
    i: int,
    j: int,
    k: int,
    duty: float,
    *,
    period_idx: int = 0,
) -> float | None:
    explicit = _optional_float(
        _index(
            getattr(solved_model, "T_c_out_y_by_period", None),
            period_idx,
            j,
            i,
            k,
        )
    )
    if explicit is None:
        explicit = _optional_float(
            _index(getattr(solved_model, "T_c_out_y", None), j, i, k)
        )
    if explicit is not None:
        return explicit
    inlet = _period_value(
        solved_model,
        "T_c_by_period",
        "T_c",
        period_idx,
        j,
        k + 1,
    )
    heat_capacity = _period_value(
        solved_model,
        "f_c_period",
        "f_c",
        period_idx,
        j,
    )
    if inlet is not None and heat_capacity is not None and heat_capacity > 0.0:
        return inlet + duty / heat_capacity
    return _period_value(
        solved_model,
        "T_c_by_period",
        "T_c",
        period_idx,
        j,
        k,
    )


def _period_ids(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
) -> tuple[str, ...]:
    raw_ids = getattr(solved_model, "period_ids", None)
    if raw_ids is None:
        raw_ids = solver_arrays.arrays.get("period_ids")
    period_ids = (
        tuple(str(value) for value in list(raw_ids)) if raw_ids is not None else ()
    )
    period_count = _optional_int(getattr(solved_model, "N_periods", None))
    if period_count is None:
        period_count = len(period_ids) or 1
    if not period_ids:
        period_ids = tuple(str(index) for index in range(period_count))
    if len(period_ids) != period_count:
        raise ValueError("solver period_ids must match N_periods during extraction")
    if len(set(period_ids)) != len(period_ids):
        raise ValueError("solver period_ids must be unique during extraction")
    return period_ids


def _period_value(
    solved_model: Any,
    period_attribute: str,
    scalar_attribute: str,
    period_idx: int,
    *indexes: int | None,
) -> float | None:
    period_values = getattr(solved_model, period_attribute, None)
    if period_values is not None:
        return _optional_float(_index(period_values, period_idx, *indexes))
    return _optional_float(
        _index(getattr(solved_model, scalar_attribute, None), *indexes)
    )


def _utility_outlet_temperature(
    solved_model: Any,
    *,
    side: str,
    period_idx: int,
    match_index: int,
    duty: float,
) -> float | None:
    resolver = getattr(solved_model, "_utility_solved_outlet_temperature", None)
    if callable(resolver):
        return _optional_float(resolver(side, period_idx, match_index, duty))
    if side == "hot":
        return _period_value(
            solved_model,
            "T_hu_out_period",
            "T_hu_out",
            period_idx,
            0,
        )
    return _period_value(
        solved_model,
        "T_cu_out_period",
        "T_cu_out",
        period_idx,
        0,
    )


def _boundary_temperature_matrix(
    values: Any,
    *,
    rows: int,
    columns: int,
) -> list[list[float | None]]:
    if values is None or rows <= 0 or columns <= 0:
        return []
    return [
        [_optional_float(_index(values, row, column)) for column in range(columns)]
        for row in range(rows)
    ]


def _approach_temperatures(
    solved_model: Any,
    i: int,
    j: int,
    k: int,
    *,
    period_idx: int = 0,
) -> tuple[float, ...]:
    theta_1_period = getattr(solved_model, "theta_1_by_period", None)
    theta_2_period = getattr(solved_model, "theta_2_by_period", None)
    values = (
        _optional_float(
            _index(theta_1_period, period_idx, i, j, k)
            if theta_1_period is not None
            else _index(getattr(solved_model, "theta_1", None), i, j, k)
        ),
        _optional_float(
            _index(theta_2_period, period_idx, i, j, k)
            if theta_2_period is not None
            else _index(getattr(solved_model, "theta_2", None), i, j, k)
        ),
    )
    return tuple(value for value in values if value is not None)


def _utility_cost(solved_model: Any) -> float | None:
    hot = _optional_float(getattr(solved_model, "hu_cost_total", None))
    cold = _optional_float(getattr(solved_model, "cu_cost_total", None))
    if hot is None and cold is None:
        return None
    return (hot or 0.0) + (cold or 0.0)


def _capital_cost(solved_model: Any) -> float | None:
    explicit = _optional_float(getattr(solved_model, "capital_cost_value", None))
    if explicit is None:
        explicit = _optional_float(getattr(solved_model, "capital_cost_total", None))
    if explicit is not None:
        return explicit

    total = 0.0
    found = False
    for name in (
        "recovery_area_cost_total",
        "hu_area_cost_total",
        "cu_area_cost_total",
        "unit_cost_total",
        "utility_unit_cost_total",
    ):
        value = _optional_float(getattr(solved_model, name, None))
        if value is not None:
            total += value
            found = True

    tac = _optional_float(getattr(solved_model, "TAC", None))
    utility = _utility_cost(solved_model)
    implied = None
    if tac is not None and utility is not None and tac >= utility:
        implied = tac - utility
    if implied is not None and (
        not found or not math.isclose(total, implied, rel_tol=1e-4, abs_tol=1.0)
    ):
        return implied
    return total if found else None


def _summary_metrics(solved_model: Any) -> dict[str, float | int | str | bool | None]:
    fields = {
        "total_units": "n_units",
        "recovery_units": "n_recovery_units",
        "hot_utility_units": "n_hu_units",
        "cold_utility_units": "n_cu_units",
        "hot_utility_load": "Q_hu_total",
        "cold_utility_load": "Q_cu_total",
        "recovery_load": "Q_r_total",
    }
    metrics: dict[str, float | int | str | bool | None] = {}
    for label, attribute in fields.items():
        value = getattr(solved_model, attribute, None)
        if label.endswith("units"):
            metrics[label] = _optional_int(value)
        else:
            metrics[label] = _optional_float(value)
    return {name: value for name, value in metrics.items() if value is not None}


def _operating_state_metadata(solved_model: Any) -> dict[str, Any]:
    period_count = _optional_int(getattr(solved_model, "N_periods", None)) or 1
    if period_count <= 1:
        return {}
    period_ids = [str(item) for item in list(getattr(solved_model, "period_ids", ()))]
    period_weights = [
        float(item) for item in list(getattr(solved_model, "period_weights", ()))
    ]
    return {
        "period_ids": period_ids,
        "period_weights": period_weights,
        "hot_utility_load_by_period": _float_list(
            getattr(solved_model, "Q_hu_total_by_period", None)
        ),
        "cold_utility_load_by_period": _float_list(
            getattr(solved_model, "Q_cu_total_by_period", None)
        ),
        "recovery_load_by_period": _float_list(
            getattr(solved_model, "Q_r_total_by_period", None)
        ),
        "operating_cost_by_period": _float_list(
            getattr(solved_model, "operating_cost_by_period", None)
        ),
        "weighted_operating_cost": _optional_float(
            getattr(solved_model, "weighted_operating_cost_value", None)
        ),
        "shared_capital_cost": _optional_float(
            getattr(solved_model, "capital_cost_value", None)
        ),
    }


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [
        float_value
        for item in values
        if (float_value := _optional_float(item)) is not None
    ]


def _float_matrix(values: Any) -> list[list[float]]:
    if values is None:
        return []
    return [_float_list(row) for row in values]


def _result_method(method: str | None):
    return {
        "PDM": "pinch_design_method",
        "TDM": "thermal_derivative_method",
        "ESM": "network_evolution_method",
    }.get(method, method)


def _is_active(duty: float, binary_value: Any, tolerance: float) -> bool:
    binary = _optional_float(binary_value)
    if binary is not None:
        return binary > tolerance and duty > tolerance
    return duty > tolerance


def _normalized_solver_duty(value: Any, *, tolerance: float) -> float:
    duty = _optional_float(value) or 0.0
    negative_tolerance = max(tolerance, _SOLVER_NEGATIVE_DUTY_NOISE)
    if duty < -negative_tolerance:
        raise ValueError(
            f"solved exchanger duty {duty:.6g} is below zero beyond tolerance"
        )
    return max(duty, 0.0)


def _allowed(value: Any) -> bool:
    flag = _optional_float(value)
    if flag is None:
        return True
    return flag > 0.0


def _third_dimension(values: Any) -> int:
    try:
        return len(values[0][0])
    except TypeError, IndexError, KeyError:
        return 0


def _index(values: Any, *indexes: int | None) -> Any:
    if values is None:
        return None
    current = values
    for index in indexes:
        if index is None:
            return None
        try:
            current = current[index]
        except TypeError, IndexError, KeyError:
            return None
    return current


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        pass
    try:
        return _optional_float(value[0])
    except TypeError, IndexError, KeyError:
        pass
    value_attr = getattr(value, "value", None)
    if value_attr is not None:
        return _optional_float(value_attr)
    value_attr = getattr(value, "VALUE", None)
    if value_attr is not None:
        return _optional_float(value_attr)
    return None


def _optional_int(value: Any) -> int | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    return int(round(numeric))
