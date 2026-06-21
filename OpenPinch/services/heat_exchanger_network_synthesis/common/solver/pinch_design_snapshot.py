"""Private pinch-design-method contract for migrated heat exchanger network PDM fields.

This module is intentionally internal to the synthesis service package. It
accepts a prepared :class:`PinchProblem`, uses source-compatible pinch target
arithmetic for the private PDM scalars, and returns the structural fields that
the migrated pinch-design-method model needs before solver construction. Later
migration slices may route PDM setup through this contract only after HENS-04
parity remains green for the required source examples and grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .....classes.pinch_problem import PinchProblem
from .....lib.config import Configuration, tol
from .arrays import PreparedSolverArrays, problem_to_solver_arrays

PinchLocation = Literal["above", "below"]
StageSelection = Literal["automated"] | tuple[int, int]


@dataclass(frozen=True)
class PinchTargetSnapshot:
    """Semantic OpenPinch target values used by PDM decomposition."""

    hot_utility_target: float
    cold_utility_target: float
    heat_recovery_target: float
    hot_pinch: float | None
    cold_pinch: float | None
    shifted_pinch_temperature: float | None
    target_access_contract: tuple[str, ...]


@dataclass(frozen=True)
class PinchDecompositionSnapshot:
    """OpenPinch-native PDM fields that shape downstream TDM/ESM tasks."""

    pinch_location: PinchLocation
    target: PinchTargetSnapshot
    z_i_active: tuple[int, ...]
    z_j_active: tuple[int, ...]
    clipped_hot_supply_temperatures: tuple[float, ...]
    clipped_hot_target_temperatures: tuple[float, ...]
    clipped_cold_supply_temperatures: tuple[float, ...]
    clipped_cold_target_temperatures: tuple[float, ...]
    S: int
    K: int
    manual_stage_selection: tuple[int, int] | None
    hot_stream_identities: tuple[str, ...]
    cold_stream_identities: tuple[str, ...]
    unit_conventions: dict[str, str]
    dt_cont_convention: str


def build_pinch_design_method_snapshot(
    problem: PinchProblem,
    dTmin: float,
    *,
    pinch_location: PinchLocation,
    stage_selection: StageSelection = "automated",
) -> PinchDecompositionSnapshot:
    """Return PDM decomposition fields from OpenPinch-owned problem state.

    The stable contract is prepared ``PinchProblem`` in, copied and
    convention-normalized ``Zone`` targeting, and identity-preserving structural
    PDM fields out. The helper is private so it cannot become a public synthesis
    entry point or bypass the migration replacement gate.
    """

    if pinch_location not in {"above", "below"}:
        raise ValueError("pinch_location must be 'above' or 'below'.")

    arrays = problem_to_solver_arrays(problem, dTmin)
    target = _calculate_openpinch_targets(problem, arrays)
    return _build_decomposition_snapshot(
        arrays=arrays,
        target=target,
        dTmin=float(dTmin),
        pinch_location=pinch_location,
        stage_selection=stage_selection,
    )


def _calculate_openpinch_targets(
    problem: PinchProblem,
    arrays: PreparedSolverArrays,
) -> PinchTargetSnapshot:
    target = _source_style_target_snapshot(
        arrays,
        reference_temperature=_problem_reference_temperature(problem),
    )
    shifted_pinch_temperature = _shifted_pinch_temperature(
        hot_utility_target=target["hot_utility_target"],
        cold_utility_target=target["cold_utility_target"],
        hot_pinch=target["hot_pinch"],
        cold_pinch=target["cold_pinch"],
    )
    return PinchTargetSnapshot(
        hot_utility_target=target["hot_utility_target"],
        cold_utility_target=target["cold_utility_target"],
        heat_recovery_target=target["heat_recovery_target"],
        hot_pinch=target["hot_pinch"],
        cold_pinch=target["cold_pinch"],
        shifted_pinch_temperature=shifted_pinch_temperature,
        target_access_contract=(
            "source-style shifted process heat cascade",
            "PreparedSolverArrays.T_h_cont",
            "PreparedSolverArrays.T_c_cont",
        ),
    )


def _problem_reference_temperature(problem: PinchProblem) -> float:
    master_zone = problem.master_zone
    if master_zone is None:
        return float(Configuration.T_ENV)
    return float(getattr(master_zone.config, "T_ENV", Configuration.T_ENV))


def _source_style_target_snapshot(
    arrays: PreparedSolverArrays,
    *,
    reference_temperature: float | None = None,
) -> dict[str, float]:
    if reference_temperature is None:
        reference_temperature = float(Configuration.T_ENV)

    values = arrays.arrays
    T_h_in = np.array(values["T_h_in"], dtype=np.float64)
    T_h_out = np.array(values["T_h_out"], dtype=np.float64)
    T_c_in = np.array(values["T_c_in"], dtype=np.float64)
    T_c_out = np.array(values["T_c_out"], dtype=np.float64)
    f_h = np.array(values["f_h"], dtype=np.float64)
    f_c = np.array(values["f_c"], dtype=np.float64)
    T_h_cont = np.array(values["T_h_cont"], dtype=np.float64)
    T_c_cont = np.array(values["T_c_cont"], dtype=np.float64)

    hot_min_star, hot_max_star, hot_cp = _source_hot_stream_table_values(
        T_h_in=T_h_in,
        T_h_out=T_h_out,
        T_h_cont=T_h_cont,
        f_h=f_h,
    )
    cold_min_star, cold_max_star, cold_cp = _source_cold_stream_table_values(
        T_c_in=T_c_in,
        T_c_out=T_c_out,
        T_c_cont=T_c_cont,
        f_c=f_c,
    )
    shifted_temperatures = _source_temperature_intervals(
        [
            *hot_min_star.tolist(),
            *hot_max_star.tolist(),
            *cold_min_star.tolist(),
            *cold_max_star.tolist(),
            reference_temperature,
        ]
    )
    shifted_table = _source_problem_table_algorithm(
        shifted_temperatures=shifted_temperatures,
        hot_min_star=hot_min_star,
        hot_max_star=hot_max_star,
        cold_min_star=cold_min_star,
        cold_max_star=cold_max_star,
        hot_cp=hot_cp,
        cold_cp=cold_cp,
    )

    hot_utility_target = shifted_table[10][0]
    cold_utility_target = shifted_table[10][-1]
    heat_recovery_target = shifted_table[4][0] - cold_utility_target
    hot_pinch, cold_pinch = _source_style_pinch_temperatures(
        shifted_temperatures=shifted_temperatures,
        shifted_cascade=shifted_table[10][1:],
        hot_utility_target=hot_utility_target,
        cold_utility_target=cold_utility_target,
    )
    return {
        "hot_utility_target": hot_utility_target,
        "cold_utility_target": cold_utility_target,
        "heat_recovery_target": heat_recovery_target,
        "hot_pinch": hot_pinch,
        "cold_pinch": cold_pinch,
    }


def _source_hot_stream_table_values(
    *,
    T_h_in: np.ndarray,
    T_h_out: np.ndarray,
    T_h_cont: np.ndarray,
    f_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supply = T_h_in - 273.15
    target = T_h_out - 273.15
    t_min = target
    t_max = supply
    cp = _source_stream_cp(T_h_in, T_h_out, t_min, t_max, f_h)
    return t_min - T_h_cont, t_max - T_h_cont, cp


def _source_cold_stream_table_values(
    *,
    T_c_in: np.ndarray,
    T_c_out: np.ndarray,
    T_c_cont: np.ndarray,
    f_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supply = T_c_in - 273.15
    target = T_c_out - 273.15
    t_min = supply
    t_max = target
    cp = _source_stream_cp(T_c_in, T_c_out, t_min, t_max, f_c)
    return t_min + T_c_cont, t_max + T_c_cont, cp


def _source_stream_cp(
    supply: np.ndarray,
    target: np.ndarray,
    t_min: np.ndarray,
    t_max: np.ndarray,
    flow_capacity: np.ndarray,
) -> np.ndarray:
    heat_flow = np.abs(target - supply) * flow_capacity
    return heat_flow / (t_max - t_min)


def _source_temperature_intervals(values: list[float]) -> list[float]:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return []
    deduplicated = [sorted_values[0]]
    for value in sorted_values[1:]:
        if abs(deduplicated[-1] - value) > tol:
            deduplicated.append(value)
    deduplicated.reverse()
    return deduplicated


def _source_problem_table_algorithm(
    *,
    shifted_temperatures: list[float],
    hot_min_star: np.ndarray,
    hot_max_star: np.ndarray,
    cold_min_star: np.ndarray,
    cold_max_star: np.ndarray,
    hot_cp: np.ndarray,
    cold_cp: np.ndarray,
) -> list[list[Any]]:
    table = [[0 for _ in shifted_temperatures] for _row in range(11)]
    table[0] = [value for value in shifted_temperatures]
    for row in range(1, 10):
        if row not in {4, 7} and table[row]:
            table[row][0] = None

    minimum_heat = np.float64(0.0)
    for index in range(1, len(table[0])):
        upper = table[0][index - 1]
        lower = table[0][index]
        interval = upper - lower
        hot_sum = _source_interval_cp_sum(
            upper=upper,
            lower=lower,
            stream_min=hot_min_star,
            stream_max=hot_max_star,
            cp=hot_cp,
        )
        cold_sum = _source_interval_cp_sum(
            upper=upper,
            lower=lower,
            stream_min=cold_min_star,
            stream_max=cold_max_star,
            cp=cold_cp,
        )

        table[1][index] = interval
        table[2][index] = hot_sum
        table[3][index] = np.float64(0.0)
        table[4][index] = table[1][index] * table[2][index] + table[4][index - 1]
        table[5][index] = cold_sum
        table[6][index] = np.float64(0.0)
        table[7][index] = table[1][index] * table[5][index] + table[7][index - 1]
        table[8][index] = table[2][index] - table[5][index]
        table[9][index] = (
            table[1][index] * table[2][index] - table[1][index] * table[5][index]
        )
        table[10][index] = table[9][index] + table[10][index - 1]
        if table[10][index] < minimum_heat:
            minimum_heat = table[10][index]

    hot_maximum = table[4][-1] if table[4] else np.float64(0.0)
    cold_minimum = table[10][-1] - minimum_heat if table[10] else np.float64(0.0)
    cold_maximum = table[7][-1] + cold_minimum if table[7] else np.float64(0.0)

    for index in range(len(table[0])):
        table[4][index] = hot_maximum - table[4][index]
        table[7][index] = cold_maximum - table[7][index]
        table[10][index] = table[10][index] - minimum_heat
        for row in (4, 7, 10):
            if abs(table[row][index]) < tol:
                table[row][index] = 0
    return table


def _source_interval_cp_sum(
    *,
    upper: float,
    lower: float,
    stream_min: np.ndarray,
    stream_max: np.ndarray,
    cp: np.ndarray,
) -> np.float64:
    total = np.float64(0.0)
    for index in range(len(cp)):
        if stream_max[index] > lower + tol and stream_min[index] < upper - tol:
            total = total + cp[index]
    return total


def _source_style_pinch_temperatures(
    *,
    shifted_temperatures: list[float],
    shifted_cascade: list[np.float64],
    hot_utility_target: np.float64,
    cold_utility_target: np.float64,
) -> tuple[float | None, float | None]:
    if not shifted_temperatures:
        return None, None

    pinch_temperatures = [
        shifted_temperatures[index + 1]
        for index, load in enumerate(shifted_cascade)
        if abs(float(load)) <= tol
    ]
    if pinch_temperatures:
        pinch = float(pinch_temperatures[0])
        return pinch, pinch

    if abs(float(hot_utility_target)) <= tol:
        return float(shifted_temperatures[-1]), float(shifted_temperatures[0])
    if abs(float(cold_utility_target)) <= tol:
        pinch = float(shifted_temperatures[-1])
        return pinch, pinch
    return None, None


def _shifted_pinch_temperature(
    *,
    hot_utility_target: float,
    cold_utility_target: float,
    hot_pinch: float | None,
    cold_pinch: float | None,
) -> float | None:
    if abs(hot_utility_target) <= tol:
        selected_pinch = cold_pinch
    elif abs(cold_utility_target) <= tol:
        selected_pinch = hot_pinch
    else:
        selected_pinch = hot_pinch
    if selected_pinch is None:
        return None
    return float(selected_pinch) + 273.15


def _build_decomposition_snapshot(
    *,
    arrays: PreparedSolverArrays,
    target: PinchTargetSnapshot,
    dTmin: float,
    pinch_location: PinchLocation,
    stage_selection: StageSelection,
) -> PinchDecompositionSnapshot:
    if target.shifted_pinch_temperature is None:
        raise ValueError("Cannot build PDM fields without a shifted pinch temperature.")

    T_h_in = _copy_float_array(arrays.arrays["T_h_in"])
    T_h_out = _copy_float_array(arrays.arrays["T_h_out"])
    T_c_in = _copy_float_array(arrays.arrays["T_c_in"])
    T_c_out = _copy_float_array(arrays.arrays["T_c_out"])
    z_i_active, z_j_active = _clip_stream_temperatures(
        T_h_in=T_h_in,
        T_h_out=T_h_out,
        T_c_in=T_c_in,
        T_c_out=T_c_out,
        shifted_pinch_temperature=target.shifted_pinch_temperature,
        dTmin=dTmin,
        pinch_location=pinch_location,
    )
    manual_stage_selection = _manual_stage_selection(stage_selection)
    S = _stage_count(
        pinch_location=pinch_location,
        stage_selection=stage_selection,
        z_i_active=z_i_active,
        z_j_active=z_j_active,
    )

    return PinchDecompositionSnapshot(
        pinch_location=pinch_location,
        target=target,
        z_i_active=z_i_active,
        z_j_active=z_j_active,
        clipped_hot_supply_temperatures=tuple(T_h_in.tolist()),
        clipped_hot_target_temperatures=tuple(T_h_out.tolist()),
        clipped_cold_supply_temperatures=tuple(T_c_in.tolist()),
        clipped_cold_target_temperatures=tuple(T_c_out.tolist()),
        S=S,
        K=S + 1,
        manual_stage_selection=manual_stage_selection,
        hot_stream_identities=tuple(arrays.stream_identities["hot_process_streams"]),
        cold_stream_identities=tuple(arrays.stream_identities["cold_process_streams"]),
        unit_conventions={
            "process_temperatures": "K in solver arrays; degC in ProblemTable rows",
            "pinch_temperatures": (
                "hot_pinch/cold_pinch in degC; shifted_pinch_temperature in K"
            ),
            "utility_targets": "kW",
        },
        dt_cont_convention=(
            "OpenHENS PDM fallback uses dTmin / 2 per process stream when "
            "prepared dt_cont is zero."
        ),
    )


def _clip_stream_temperatures(
    *,
    T_h_in: np.ndarray,
    T_h_out: np.ndarray,
    T_c_in: np.ndarray,
    T_c_out: np.ndarray,
    shifted_pinch_temperature: float,
    dTmin: float,
    pinch_location: PinchLocation,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    hot_threshold = shifted_pinch_temperature + dTmin / 2.0
    cold_threshold = shifted_pinch_temperature - dTmin / 2.0
    if pinch_location == "above":
        z_i_active = tuple(1 if value > hot_threshold else 0 for value in T_h_in)
        z_j_active = tuple(1 if value > cold_threshold else 0 for value in T_c_out)
        for index, is_active in enumerate(z_i_active):
            if is_active:
                T_h_out[index] = max(T_h_out[index], hot_threshold)
            else:
                T_h_in[index] = 0.0
                T_h_out[index] = 0.0
        for index, is_active in enumerate(z_j_active):
            if is_active:
                T_c_in[index] = max(T_c_in[index], cold_threshold)
            else:
                T_c_in[index] = 0.0
                T_c_out[index] = 0.0
    else:
        z_i_active = tuple(1 if value < hot_threshold else 0 for value in T_h_out)
        z_j_active = tuple(1 if value < cold_threshold else 0 for value in T_c_in)
        for index, is_active in enumerate(z_i_active):
            if is_active:
                T_h_in[index] = min(T_h_in[index], hot_threshold)
            else:
                T_h_in[index] = 0.0
                T_h_out[index] = 0.0
        for index, is_active in enumerate(z_j_active):
            if is_active:
                T_c_out[index] = min(T_c_out[index], cold_threshold)
            else:
                T_c_in[index] = 0.0
                T_c_out[index] = 0.0
    return z_i_active, z_j_active


def _stage_count(
    *,
    pinch_location: PinchLocation,
    stage_selection: StageSelection,
    z_i_active: tuple[int, ...],
    z_j_active: tuple[int, ...],
) -> int:
    manual_stage_selection = _manual_stage_selection(stage_selection)
    if manual_stage_selection is None:
        return max(sum(z_i_active), sum(z_j_active))
    return manual_stage_selection[0 if pinch_location == "above" else 1]


def _manual_stage_selection(
    stage_selection: StageSelection,
) -> tuple[int, int] | None:
    if stage_selection == "automated":
        return None
    if len(stage_selection) != 2:
        raise ValueError(
            "manual stage_selection must contain exactly two stage counts."
        )
    if any(type(stage) is not int or stage <= 0 for stage in stage_selection):
        raise ValueError(
            "manual stage_selection stage counts must be positive integers."
        )
    return tuple(stage_selection)


def _copy_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).copy()


__all__ = [
    "PinchDecompositionSnapshot",
    "PinchTargetSnapshot",
    "build_pinch_design_method_snapshot",
]
