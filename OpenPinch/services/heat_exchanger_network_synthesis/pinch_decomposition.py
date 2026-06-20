"""Private pinch-decomposition contract for migrated heat exchanger network PDM fields.

This module is intentionally internal to the synthesis service package. It
accepts a prepared :class:`PinchProblem`, uses OpenPinch targeting as the source
of target semantics, and returns the structural fields that the migrated
pinch-decomposition model needs before solver construction. Later migration
slices may route PDM setup through this contract only after HENS-04 parity
remains green for the required source examples and grids.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ...classes.pinch_problem import PinchProblem
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.schemas.targets import DirectIntegrationTarget
from ...services.direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from .array_adapter import PreparedSolverArrays, problem_to_solver_arrays

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


def build_pinch_decomposition_snapshot(
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
    zone = _copy_zone_with_solver_temperature_contributions(problem, arrays)
    target = compute_direct_integration_targets(zone)
    return _target_snapshot_from_direct_target(target)


def _copy_zone_with_solver_temperature_contributions(
    problem: PinchProblem,
    arrays: PreparedSolverArrays,
) -> Zone:
    zone = deepcopy(problem.master_zone)
    if not isinstance(zone, Zone):
        raise RuntimeError("PinchProblem must be loaded before pinch decomposition.")

    for stream_key, index in arrays.axis_maps["hot_process_streams"].items():
        zone.hot_streams[stream_key].dt_cont = float(arrays.arrays["T_h_cont"][index])
    for stream_key, index in arrays.axis_maps["cold_process_streams"].items():
        zone.cold_streams[stream_key].dt_cont = float(arrays.arrays["T_c_cont"][index])
    return zone


def _target_snapshot_from_direct_target(
    target: DirectIntegrationTarget,
) -> PinchTargetSnapshot:
    hot_utility_target = float(target.hot_utility_target)
    cold_utility_target = float(target.cold_utility_target)
    heat_recovery_target = float(target.heat_recovery_target)
    shifted_pinch_temperature = _shifted_pinch_temperature(
        hot_utility_target=hot_utility_target,
        cold_utility_target=cold_utility_target,
        hot_pinch=target.hot_pinch,
        cold_pinch=target.cold_pinch,
    )
    return PinchTargetSnapshot(
        hot_utility_target=hot_utility_target,
        cold_utility_target=cold_utility_target,
        heat_recovery_target=heat_recovery_target,
        hot_pinch=None if target.hot_pinch is None else float(target.hot_pinch),
        cold_pinch=None if target.cold_pinch is None else float(target.cold_pinch),
        shifted_pinch_temperature=shifted_pinch_temperature,
        target_access_contract=(
            "DirectIntegrationTarget.hot_utility_target",
            "DirectIntegrationTarget.cold_utility_target",
            "DirectIntegrationTarget.heat_recovery_target",
            "DirectIntegrationTarget.hot_pinch",
            "DirectIntegrationTarget.cold_pinch",
        ),
    )


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
    "build_pinch_decomposition_snapshot",
]
