"""Translate solved HEN model state into domain and contract results."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ....domain.configuration import tol
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger import HeatExchanger
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..solver.arrays import SEGMENT_PROFILE_VERSION, PreparedSolverArrays
from ..solver.utility_slices import utility_slices
from .metadata import (
    _boundary_temperature_matrix,
    _capital_cost,
    _float_list,
    _float_matrix,
    _identities_by_axis,
    _operating_state_metadata,
    _optional_float,
    _optional_int,
    _period_ids,
    _result_method,
    _summary_metrics,
    _utilities,
    _utility_cost,
)
from .recovery import _recovery_exchangers
from .utility import (
    _cold_utility_exchangers,
    _hot_utility_exchangers,
    _staged_cold_utility_exchangers,
    _staged_hot_utility_exchangers,
)


def _float_tensor(values) -> list[list[list[float]]]:
    """Return a [period][row][column] array as nested float lists."""
    if values is None:
        return []
    return [_float_matrix(matrix) for matrix in values]


def _end_utilities_in_parallel(
    exchangers: list[HeatExchanger], period_ids: tuple[str, ...]
) -> list[HeatExchanger]:
    """Split a stream between its stream-end heaters (or coolers).

    The stage-wise model holds each end heater's approaches and sizes its
    area against the stream's whole end span, so several on one stream run
    in parallel, each on the share of the stream's flow its duty takes.
    """

    for period_idx, period_id in enumerate(period_ids):
        groups: dict[tuple[HeatExchangerKind, str], list[int]] = {}
        for index, exchanger in enumerate(exchangers):
            if exchanger.stage is not None or exchanger.kind not in (
                HeatExchangerKind.HOT_UTILITY,
                HeatExchangerKind.COLD_UTILITY,
            ):
                continue
            if not exchanger.state(period_id).active:
                continue
            hot = exchanger.kind is HeatExchangerKind.HOT_UTILITY
            stream = exchanger.sink_stream if hot else exchanger.source_stream
            groups.setdefault((exchanger.kind, stream), []).append(index)
        for (kind, _stream), members in groups.items():
            total = sum(exchangers[index].state(period_id).duty for index in members)
            if len(members) < 2 or total <= 0.0:
                continue
            field = (
                "sink_split_fraction"
                if kind is HeatExchangerKind.HOT_UTILITY
                else "source_split_fraction"
            )
            for index in members:
                exchanger = exchangers[index]
                share = exchanger.state(period_id).duty / total
                exchangers[index] = exchanger.model_copy(
                    update={
                        "period_states": tuple(
                            item.model_copy(update={field: share})
                            if n == period_idx
                            else item
                            for n, item in enumerate(exchanger.period_states)
                        )
                    }
                )
    return exchangers


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
    dt_cont_multiplier: float | None = None,
) -> HeatExchangerNetwork:
    """Convert a solved model into identity-labelled OpenPinch records.

    ``dt_cont_multiplier``, the synthesis task's multiplier, is recorded in
    the network's summary metrics and source metadata when given.
    """

    hot_streams = _identities_by_axis(solver_arrays, "hot_process_streams")
    cold_streams = _identities_by_axis(solver_arrays, "cold_process_streams")
    hot_utilities = _identities_by_axis(solver_arrays, "hot_utilities")
    cold_utilities = _identities_by_axis(solver_arrays, "cold_utilities")
    # A utility slice (``../solver/utility_slices.py``) is reported under its
    # parent utility; its exchanger id keeps the slice.
    parents = {
        label: parent
        for side in utility_slices(solver_arrays).values()
        for label, (parent, _share) in side.items()
    }
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
    for u, hot_utility in enumerate(_utilities(hot_utilities, "hot utility")):
        exchangers.extend(
            _staged_hot_utility_exchangers(
                solved_model,
                hot_utility=hot_utility,
                utility_name=parents.get(hot_utility),
                utility_index=u,
                cold_streams=cold_streams,
                period_ids=period_ids,
                tolerance=tolerance,
                include_inactive=include_inactive,
            )
        )
        exchangers.extend(
            _hot_utility_exchangers(
                solved_model,
                hot_utility=hot_utility,
                utility_name=parents.get(hot_utility),
                utility_index=u,
                cold_streams=cold_streams,
                period_ids=period_ids,
                tolerance=tolerance,
                include_inactive=include_inactive,
            )
        )
    for u, cold_utility in enumerate(_utilities(cold_utilities, "cold utility")):
        exchangers.extend(
            _staged_cold_utility_exchangers(
                solved_model,
                cold_utility=cold_utility,
                utility_name=parents.get(cold_utility),
                utility_index=u,
                hot_streams=hot_streams,
                period_ids=period_ids,
                tolerance=tolerance,
                include_inactive=include_inactive,
            )
        )
        exchangers.extend(
            _cold_utility_exchangers(
                solved_model,
                cold_utility=cold_utility,
                utility_name=parents.get(cold_utility),
                utility_index=u,
                hot_streams=hot_streams,
                period_ids=period_ids,
                tolerance=tolerance,
                include_inactive=include_inactive,
            )
        )

    exchangers = _end_utilities_in_parallel(exchangers, period_ids)

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
        summary_metrics=_summary_metrics(solved_model, dt_cont_multiplier),
        solver_axis_metadata={
            "axis_maps": solver_arrays.axis_maps,
            "utility_parents": parents,
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
            "dt_cont_multiplier": _optional_float(dt_cont_multiplier),
            # Each recovery pair's own minimum approach (hot + cold stream
            # contributions), by period: [period][hot][cold].
            "recovery_minimum_approach_by_period": _float_tensor(
                getattr(solved_model, "dT_r_period", None)
            ),
            # Each heater's and cooler's least approach, by period:
            # [period][hot utility][cold] and [period][hot][cold utility].
            "hot_utility_minimum_approach_by_period": _float_tensor(
                getattr(solved_model, "dT_hu_min_period", None)
            ),
            "cold_utility_minimum_approach_by_period": _float_tensor(
                getattr(solved_model, "dT_cu_min_period", None)
            ),
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
            # The same stream data per operating period ([period][stream]), so
            # multi-period networks are checked against each period's loads.
            "hot_stream_supply_temperatures_by_period": _period_matrix(
                solved_model, solver_arrays, "T_h_in_period"
            ),
            "hot_stream_target_temperatures_by_period": _period_matrix(
                solved_model, solver_arrays, "T_h_out_period"
            ),
            "cold_stream_supply_temperatures_by_period": _period_matrix(
                solved_model, solver_arrays, "T_c_in_period"
            ),
            "cold_stream_target_temperatures_by_period": _period_matrix(
                solved_model, solver_arrays, "T_c_out_period"
            ),
            # Verification sizes each period with its own film coefficients;
            # the scalar ones above are the first period's.
            "hot_stream_heat_transfer_coefficients_by_period": _period_matrix(
                solved_model, solver_arrays, "htc_h_period"
            ),
            "cold_stream_heat_transfer_coefficients_by_period": _period_matrix(
                solved_model, solver_arrays, "htc_c_period"
            ),
            "hot_utility_heat_transfer_coefficients_by_period": _period_matrix(
                solved_model, solver_arrays, "htc_hu_period"
            ),
            "cold_utility_heat_transfer_coefficients_by_period": _period_matrix(
                solved_model, solver_arrays, "htc_cu_period"
            ),
            "hot_stream_total_duties_by_period": _segment_parent_total_duties_by_period(
                solver_arrays, "hot"
            ),
            "cold_stream_total_duties_by_period": (
                _segment_parent_total_duties_by_period(solver_arrays, "cold")
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


def _period_matrix(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
    name: str,
) -> list[list[float]]:
    """Return a solver [period][stream] array from the model or its inputs."""
    values = getattr(solved_model, name, None)
    if values is None:
        values = solver_arrays.arrays.get(name)
    return _float_matrix(values) if values is not None else []


def _segment_parent_total_duties_by_period(
    solver_arrays: PreparedSolverArrays,
    side: str,
) -> list[list[float]]:
    values = solver_arrays.arrays.get(f"{side}_segment_duty_period")
    if values is None:
        return []
    return np.sum(np.asarray(values, dtype=float), axis=2).tolist()


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
