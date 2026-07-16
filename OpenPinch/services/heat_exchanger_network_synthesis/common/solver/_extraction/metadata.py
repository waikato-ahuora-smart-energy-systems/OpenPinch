"""Solver extraction period, utility, summary, and metadata helpers."""

from __future__ import annotations

import math
from typing import Any

from ..arrays import PreparedSolverArrays

_SOLVER_NEGATIVE_DUTY_NOISE = 1e-4


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
