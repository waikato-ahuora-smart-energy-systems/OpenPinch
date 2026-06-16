"""Private adapter from prepared :class:`PinchProblem` state to solver arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...classes.pinch_problem import PinchProblem
from ...classes.stream import Stream
from ...classes.zone import Zone
from ...lib.config import tol


@dataclass(frozen=True)
class PreparedSolverArrays:
    """Immutable private solver-array payload for migrated equation models."""

    arrays: dict[str, np.ndarray]
    axis_maps: dict[str, dict[str, int]]
    unit_conventions: dict[str, str]
    stream_identities: dict[str, list[str]]
    utility_identities: dict[str, list[str]]
    configuration: dict[str, Any]
    preparation: dict[str, Any]

    @property
    def array_shapes(self) -> dict[str, list[int]]:
        """Return JSON-compatible shapes for each private array."""
        return {name: list(values.shape) for name, values in self.arrays.items()}

    def to_json_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "arrays": {
                name: values.tolist()
                for name, values in sorted(
                    self.arrays.items(), key=lambda item: item[0]
                )
            },
            "array_shapes": {
                name: self.array_shapes[name] for name in sorted(self.array_shapes)
            },
            "axis_maps": self.axis_maps,
            "configuration": self.configuration,
            "preparation": self.preparation,
            "stream_identities": self.stream_identities,
            "unit_conventions": self.unit_conventions,
            "utility_identities": self.utility_identities,
        }


def problem_to_solver_arrays(
    problem: PinchProblem,
    dTmin: float,
) -> PreparedSolverArrays:
    """Build the private solver-array payload from a prepared ``PinchProblem``.

    The adapter intentionally accepts only a live ``PinchProblem`` so raw fixture
    rows, raw ``TargetInput`` payloads, HEN schemas, cached array dictionaries,
    and standalone DTOs cannot bypass OpenPinch validation and preparation.
    """

    if not isinstance(problem, PinchProblem):
        raise TypeError(
            "problem_to_solver_arrays requires a prepared PinchProblem; "
            f"got {type(problem).__name__}."
        )
    if not isinstance(problem.master_zone, Zone):
        raise RuntimeError(
            "problem_to_solver_arrays requires PinchProblem.load(...) and "
            "prepare_problem(...) to create a prepared root Zone first."
        )
    if not np.isfinite(dTmin) or dTmin <= 0.0:
        raise ValueError("dTmin must be finite and positive.")

    zone = problem.master_zone
    hot_items = list(zone.hot_streams.items())
    cold_items = list(zone.cold_streams.items())
    hot_utility_items = list(zone.hot_utilities.items())
    cold_utility_items = list(zone.cold_utilities.items())

    if not hot_items or not cold_items:
        raise ValueError("prepared PinchProblem must contain hot and cold streams.")
    if not hot_utility_items or not cold_utility_items:
        raise ValueError("prepared PinchProblem must contain hot and cold utilities.")

    config = zone.config
    arrays = {
        "A_coeff": _float_array([config.VARIABLE_COST]),
        "A_exp": _float_array([config.COST_EXP]),
        "T_c_cont": _float_array(
            _temperature_contribution(stream, dTmin) for _, stream in cold_items
        ),
        "T_c_in": _float_array(
            _value(stream.t_supply, "K") for _, stream in cold_items
        ),
        "T_c_out": _float_array(
            _value(stream.t_target, "K") for _, stream in cold_items
        ),
        "T_cu_in": _float_array(
            _value(stream.t_supply, "K") for _, stream in cold_utility_items
        ),
        "T_cu_out": _float_array(
            _utility_solver_target(stream, zone) for _, stream in cold_utility_items
        ),
        "T_h_cont": _float_array(
            _temperature_contribution(stream, dTmin) for _, stream in hot_items
        ),
        "T_h_in": _float_array(_value(stream.t_supply, "K") for _, stream in hot_items),
        "T_h_out": _float_array(
            _value(stream.t_target, "K") for _, stream in hot_items
        ),
        "T_hu_in": _float_array(
            _value(stream.t_supply, "K") for _, stream in hot_utility_items
        ),
        "T_hu_out": _float_array(
            _utility_solver_target(stream, zone) for _, stream in hot_utility_items
        ),
        "c_cost": _float_array(
            _value(stream.price, "$/MW/h") for _, stream in cold_items
        ),
        "cold_names": _str_array(stream.name for _, stream in cold_items),
        "cu_coeff": _float_array([config.VARIABLE_COST]),
        "cu_cost": _float_array(
            _value(stream.price, "$/MW/h") for _, stream in cold_utility_items
        ),
        "cu_exp": _float_array([config.COST_EXP]),
        "cu_unit_cost": _float_array([config.FIXED_COST]),
        "f_c": _float_array(
            _value(stream.CP, "kW/delta_degC") for _, stream in cold_items
        ),
        "f_h": _float_array(
            _value(stream.CP, "kW/delta_degC") for _, stream in hot_items
        ),
        "h_cost": _float_array(
            _value(stream.price, "$/MW/h") for _, stream in hot_items
        ),
        "hot_names": _str_array(stream.name for _, stream in hot_items),
        "htc_c": _float_array(
            _value(stream.htc, "kW/m^2/delta_degC") for _, stream in cold_items
        ),
        "htc_cu": _float_array(
            _value(stream.htc, "kW/m^2/delta_degC") for _, stream in cold_utility_items
        ),
        "htc_h": _float_array(
            _value(stream.htc, "kW/m^2/delta_degC") for _, stream in hot_items
        ),
        "htc_hu": _float_array(
            _value(stream.htc, "kW/m^2/delta_degC") for _, stream in hot_utility_items
        ),
        "hu_coeff": _float_array([config.VARIABLE_COST]),
        "hu_cost": _float_array(
            _value(stream.price, "$/MW/h") for _, stream in hot_utility_items
        ),
        "hu_exp": _float_array([config.COST_EXP]),
        "hu_unit_cost": _float_array([config.FIXED_COST]),
        "unit_cost": _float_array([config.FIXED_COST]),
    }

    axis_maps = {
        "cold_process_streams": {
            key: index for index, (key, _) in enumerate(cold_items)
        },
        "cold_utilities": {
            key: index for index, (key, _) in enumerate(cold_utility_items)
        },
        "hot_process_streams": {key: index for index, (key, _) in enumerate(hot_items)},
        "hot_utilities": {
            key: index for index, (key, _) in enumerate(hot_utility_items)
        },
        "stages": {
            str(stage): index
            for index, stage in enumerate(getattr(config, "HENS_STAGE_SELECTION", ()))
        },
    }

    return PreparedSolverArrays(
        arrays=arrays,
        axis_maps=axis_maps,
        configuration={
            "HENS_APPROACH_TEMPERATURES": list(config.HENS_APPROACH_TEMPERATURES),
            "HENS_BEST_SOLUTIONS_TO_SAVE": config.HENS_BEST_SOLUTIONS_TO_SAVE,
            "HENS_DERIVATIVE_THRESHOLDS": list(config.HENS_DERIVATIVE_THRESHOLDS),
            "HENS_ESM_SOLVER": config.HENS_ESM_SOLVER,
            "HENS_LOG_LEVEL": config.HENS_LOG_LEVEL,
            "HENS_MAX_PARALLEL": config.HENS_MAX_PARALLEL,
            "HENS_METHOD_SEQUENCE": list(config.HENS_METHOD_SEQUENCE),
            "HENS_OUTPUT_FOLDER": config.HENS_OUTPUT_FOLDER,
            "HENS_OUTPUT_FORMATS": list(config.HENS_OUTPUT_FORMATS),
            "HENS_PDM_SOLVER": config.HENS_PDM_SOLVER,
            "HENS_RUN_ID": config.HENS_RUN_ID,
            "HENS_SOLVE_TOLERANCE": config.HENS_SOLVE_TOLERANCE,
            "HENS_STAGE_SELECTION": list(config.HENS_STAGE_SELECTION),
            "HENS_TDM_SOLVER": config.HENS_TDM_SOLVER,
            "active_dTmin": float(dTmin),
            "costing": {
                "COST_EXP": config.COST_EXP,
                "FIXED_COST": config.FIXED_COST,
                "VARIABLE_COST": config.VARIABLE_COST,
            },
        },
        preparation={
            "pinch_problem_class": type(problem).__name__,
            "prepared_zone_class": type(zone).__name__,
            "process_stream_collection_class": type(zone.process_streams).__name__,
            "stream_class": Stream.__name__,
            "zone_name": zone.name,
        },
        stream_identities={
            "cold_process_streams": [key for key, _ in cold_items],
            "hot_process_streams": [key for key, _ in hot_items],
        },
        unit_conventions={
            "cost_coefficients": (
                "OpenPinch Configuration FIXED_COST, VARIABLE_COST, COST_EXP"
            ),
            "heat_capacity_flowrate": "kW/K",
            "heat_transfer_coefficient": "kW/m^2/K",
            "temperature": "K",
            "temperature_contribution": (
                "K, falling back to dTmin / 2 when prepared contribution is absent"
            ),
            "utility_price": (
                "numeric UtilitySchema.price passed through for OpenHENS "
                "annual cost equations"
            ),
        },
        utility_identities={
            "cold_utilities": [key for key, _ in cold_utility_items],
            "hot_utilities": [key for key, _ in hot_utility_items],
        },
    )


def _temperature_contribution(stream: Stream, dTmin: float) -> float:
    contribution = _value(stream.dt_cont, "delta_degC")
    if contribution > tol:
        return contribution
    return float(dTmin) / 2.0


def _utility_solver_target(stream: Stream, zone: Zone) -> float:
    supply = _value(stream.t_supply, "K")
    target = _value(stream.t_target, "K")
    if abs(supply - target) <= zone.config.DT_PHASE_CHANGE + tol:
        return supply
    return target


def _value(value: Any, unit: str) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "to"):
        return float(value.to(unit).value)
    return float(value)


def _float_array(values) -> np.ndarray:
    return np.array(list(values), dtype=float)


def _str_array(values) -> np.ndarray:
    return np.array(list(values), dtype=str)


__all__ = ["PreparedSolverArrays", "problem_to_solver_arrays"]
