"""Extract private solver arrays into OpenPinch heat exchanger network records."""

from __future__ import annotations

import math
from typing import Any

from .....classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerStreamRole,
)
from .....classes.heat_exchanger_network import HeatExchangerNetwork
from .....lib.config import tol
from .....lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from .arrays import PreparedSolverArrays


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

    exchangers: list[HeatExchanger] = []
    exchangers.extend(
        _recovery_exchangers(
            solved_model,
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            stage_total=stage_total,
            tolerance=tolerance,
            include_inactive=include_inactive,
        )
    )
    exchangers.extend(
        _hot_utility_exchangers(
            solved_model,
            hot_utility=_single_utility(hot_utilities, "hot utility"),
            cold_streams=cold_streams,
            tolerance=tolerance,
            include_inactive=include_inactive,
        )
    )
    exchangers.extend(
        _cold_utility_exchangers(
            solved_model,
            cold_utility=_single_utility(cold_utilities, "cold utility"),
            hot_streams=hot_streams,
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
            "operating_states": _operating_state_metadata(solved_model),
            "hot_stream_heat_capacity_flowrates": _float_list(
                getattr(solved_model, "f_h", None)
            ),
            "cold_stream_heat_capacity_flowrates": _float_list(
                getattr(solved_model, "f_c", None)
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


def extract_network_synthesis_result(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
    *,
    run_id: str,
    task_id: str | None = None,
    problem_id: str | None = None,
    workspace_variant: str | None = None,
    state_id: str | None = None,
    solver_name: str | None = None,
    solver_status: str | None = None,
    method: str | None = None,
    stage_count: int | None = None,
    objective_value: float | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Build the result payload that may cross the OpenPinch service boundary."""

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
        state_id=state_id,
        solver_name=solver_name,
        solver_status=solver_status,
        method=result_method,
        stage_count=network.stage_count,
        objective_values=objective_values,
    )


def _recovery_exchangers(
    solved_model: Any,
    *,
    hot_streams: tuple[str, ...],
    cold_streams: tuple[str, ...],
    stage_total: int | None,
    tolerance: float,
    include_inactive: bool,
) -> tuple[HeatExchanger, ...]:
    q_values = getattr(solved_model, "Q_r", None)
    if q_values is None:
        return ()
    stages = stage_total or _third_dimension(q_values)
    exchangers: list[HeatExchanger] = []
    for i, hot_stream in enumerate(hot_streams):
        for j, cold_stream in enumerate(cold_streams):
            for k in range(stages):
                duty = _optional_float(_index(q_values, i, j, k)) or 0.0
                active = _is_active(
                    duty,
                    _index(getattr(solved_model, "z", None), i, j, k),
                    tolerance,
                )
                if not active and not include_inactive:
                    continue
                stage = k + 1
                exchangers.append(
                    HeatExchanger(
                        exchanger_id=f"recovery:{hot_stream}->{cold_stream}:S{stage}",
                        kind=HeatExchangerKind.RECOVERY,
                        source_stream=hot_stream,
                        sink_stream=cold_stream,
                        source_stream_role=HeatExchangerStreamRole.PROCESS,
                        sink_stream_role=HeatExchangerStreamRole.PROCESS,
                        stage=stage,
                        duty=duty,
                        area=_optional_float(
                            _index(getattr(solved_model, "area_r", None), i, j, k)
                        ),
                        active=active,
                        match_allowed=_allowed(
                            _index(getattr(solved_model, "z_allowed", None), i, j, k)
                        ),
                        approach_temperatures=_approach_temperatures(
                            solved_model,
                            i,
                            j,
                            k,
                        ),
                        source_inlet_temperature=_optional_float(
                            _index(getattr(solved_model, "T_h", None), i, k)
                        ),
                        source_outlet_temperature=_hot_recovery_outlet(
                            solved_model,
                            i,
                            j,
                            k,
                            duty,
                        ),
                        sink_inlet_temperature=_optional_float(
                            _index(getattr(solved_model, "T_c", None), j, k + 1)
                        ),
                        sink_outlet_temperature=_cold_recovery_outlet(
                            solved_model,
                            i,
                            j,
                            k,
                            duty,
                        ),
                    )
                )
    return tuple(exchangers)


def _hot_utility_exchangers(
    solved_model: Any,
    *,
    hot_utility: str,
    cold_streams: tuple[str, ...],
    tolerance: float,
    include_inactive: bool,
) -> tuple[HeatExchanger, ...]:
    q_values = getattr(solved_model, "Q_h", None)
    if q_values is None:
        return ()
    exchangers: list[HeatExchanger] = []
    for j, cold_stream in enumerate(cold_streams):
        duty = _optional_float(_index(q_values, j)) or 0.0
        active = _is_active(
            duty,
            _index(getattr(solved_model, "z_hu", None), j),
            tolerance,
        )
        if not active and not include_inactive:
            continue
        exchangers.append(
            HeatExchanger(
                exchanger_id=f"hot-utility:{hot_utility}->{cold_stream}",
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream=hot_utility,
                sink_stream=cold_stream,
                source_stream_role=HeatExchangerStreamRole.UTILITY,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                duty=duty,
                area=_optional_float(_index(getattr(solved_model, "area_hu", None), j)),
                active=active,
                match_allowed=_allowed(
                    _index(getattr(solved_model, "z_hu_allowed", None), j)
                ),
                source_inlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_hu_in", None), 0)
                ),
                source_outlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_hu_out", None), 0)
                ),
                sink_inlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_c", None), j, 0)
                ),
                sink_outlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_c_out", None), j)
                ),
            )
        )
    return tuple(exchangers)


def _cold_utility_exchangers(
    solved_model: Any,
    *,
    cold_utility: str,
    hot_streams: tuple[str, ...],
    tolerance: float,
    include_inactive: bool,
) -> tuple[HeatExchanger, ...]:
    q_values = getattr(solved_model, "Q_c", None)
    if q_values is None:
        return ()
    exchangers: list[HeatExchanger] = []
    last_stage = _optional_int(getattr(solved_model, "S", None))
    for i, hot_stream in enumerate(hot_streams):
        duty = _optional_float(_index(q_values, i)) or 0.0
        active = _is_active(
            duty,
            _index(getattr(solved_model, "z_cu", None), i),
            tolerance,
        )
        if not active and not include_inactive:
            continue
        exchangers.append(
            HeatExchanger(
                exchanger_id=f"cold-utility:{hot_stream}->{cold_utility}",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream=hot_stream,
                sink_stream=cold_utility,
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.UTILITY,
                duty=duty,
                area=_optional_float(_index(getattr(solved_model, "area_cu", None), i)),
                active=active,
                match_allowed=_allowed(
                    _index(getattr(solved_model, "z_cu_allowed", None), i)
                ),
                source_inlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_h", None), i, last_stage)
                ),
                source_outlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_h_out", None), i)
                ),
                sink_inlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_cu_in", None), 0)
                ),
                sink_outlet_temperature=_optional_float(
                    _index(getattr(solved_model, "T_cu_out", None), 0)
                ),
            )
        )
    return tuple(exchangers)


def _identities_by_axis(
    solver_arrays: PreparedSolverArrays,
    axis_name: str,
) -> tuple[str, ...]:
    axis_map = solver_arrays.axis_maps.get(axis_name, {})
    if axis_map:
        return tuple(
            key for key, _ in sorted(axis_map.items(), key=lambda item: item[1])
        )
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
) -> float | None:
    inlet = _optional_float(_index(getattr(solved_model, "T_h", None), i, k))
    heat_capacity = _optional_float(_index(getattr(solved_model, "f_h", None), i))
    if inlet is not None and heat_capacity is not None and heat_capacity > 0.0:
        return inlet - duty / heat_capacity
    explicit = _optional_float(
        _index(getattr(solved_model, "T_h_out_x", None), i, j, k)
    )
    if explicit is not None:
        return explicit
    return _optional_float(_index(getattr(solved_model, "T_h", None), i, k + 1))


def _cold_recovery_outlet(
    solved_model: Any,
    i: int,
    j: int,
    k: int,
    duty: float,
) -> float | None:
    inlet = _optional_float(_index(getattr(solved_model, "T_c", None), j, k + 1))
    heat_capacity = _optional_float(_index(getattr(solved_model, "f_c", None), j))
    if inlet is not None and heat_capacity is not None and heat_capacity > 0.0:
        return inlet + duty / heat_capacity
    explicit = _optional_float(
        _index(getattr(solved_model, "T_c_out_y", None), j, i, k)
    )
    if explicit is not None:
        return explicit
    return _optional_float(_index(getattr(solved_model, "T_c", None), j, k))


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
) -> tuple[float, ...]:
    values = (
        _optional_float(_index(getattr(solved_model, "theta_1", None), i, j, k)),
        _optional_float(_index(getattr(solved_model, "theta_2", None), i, j, k)),
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
    state_count = _optional_int(getattr(solved_model, "N_states", None)) or 1
    if state_count <= 1:
        return {}
    state_ids = [str(item) for item in list(getattr(solved_model, "state_ids", ()))]
    state_weights = [
        float(item) for item in list(getattr(solved_model, "state_weights", ()))
    ]
    return {
        "state_ids": state_ids,
        "state_weights": state_weights,
        "hot_utility_load_by_state": _float_list(
            getattr(solved_model, "Q_hu_total_by_state", None)
        ),
        "cold_utility_load_by_state": _float_list(
            getattr(solved_model, "Q_cu_total_by_state", None)
        ),
        "recovery_load_by_state": _float_list(
            getattr(solved_model, "Q_r_total_by_state", None)
        ),
        "operating_cost_by_state": _float_list(
            getattr(solved_model, "operating_cost_by_state", None)
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
