"""Extract period-native exchanger state and temperatures."""

from __future__ import annotations

from typing import Any

from ....domain._heat_exchanger.period_state import HeatExchangerPeriodState
from .metadata import (
    _index,
    _is_active,
    _normalized_solver_duty,
    _optional_float,
)


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
