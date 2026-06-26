"""Private pinch-design-method decomposition contract.

This module adapts an OpenPinch :class:`PinchProblem` into the validated
above-/below-pinch fields required by the migrated PDM equation model. It keeps
the solver-array ordering contract at the synthesis boundary, but delegates
pinch target arithmetic to the OpenPinch direct-integration service operating
on copied :class:`Zone` and :class:`Stream` objects.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .....classes.pinch_problem import PinchProblem
from .....classes.stream import Stream
from .....classes.zone import Zone
from .....lib.config import tol
from .....services.direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from .arrays import PreparedSolverArrays, problem_to_solver_arrays

PinchLocation = Literal["above", "below"]
StageSelection = Literal["automated"] | tuple[int, int]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class PinchDesignTarget(BaseModel):
    """Semantic OpenPinch target values used by PDM decomposition."""

    model_config = _MODEL_CONFIG

    hot_utility_target: float
    cold_utility_target: float
    heat_recovery_target: float
    hot_pinch: float | None
    cold_pinch: float | None
    shifted_pinch_temperature: float | None
    target_access_contract: tuple[str, ...] = Field(
        default=(
            "OpenPinch direct_heat_integration on copied Zone",
            "Zone.process_streams",
            "Stream.dt_cont clamped to dTmin / 2 minimum",
        )
    )

    @field_validator(
        "hot_utility_target",
        "cold_utility_target",
        "heat_recovery_target",
    )
    @classmethod
    def _finite_target(cls, value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("pinch target values must be finite")
        return value

    @field_validator("hot_pinch", "cold_pinch", "shifted_pinch_temperature")
    @classmethod
    def _optional_finite_target(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("pinch temperatures must be finite when supplied")
        return value


class PinchDesignDecomposition(BaseModel):
    """OpenPinch-native PDM fields that shape downstream TDM/ESM tasks."""

    model_config = _MODEL_CONFIG

    pinch_location: PinchLocation
    target: PinchDesignTarget
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

    @field_validator("z_i_active", "z_j_active")
    @classmethod
    def _binary_flags(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        flags = tuple(int(item) for item in value)
        if any(item not in {0, 1} for item in flags):
            raise ValueError("PDM active-stream flags must be binary")
        return flags

    @field_validator(
        "clipped_hot_supply_temperatures",
        "clipped_hot_target_temperatures",
        "clipped_cold_supply_temperatures",
        "clipped_cold_target_temperatures",
    )
    @classmethod
    def _finite_temperature_tuple(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        temperatures = tuple(float(item) for item in value)
        if any(not np.isfinite(item) for item in temperatures):
            raise ValueError("PDM clipped stream temperatures must be finite")
        return temperatures

    @model_validator(mode="after")
    def _consistent_stage_count(self) -> "PinchDesignDecomposition":
        if self.S <= 0:
            raise ValueError("PDM stage count S must be positive")
        if self.K != self.S + 1:
            raise ValueError("PDM boundary count K must equal S + 1")
        if len(self.z_i_active) != len(self.hot_stream_identities):
            raise ValueError("hot active flags must match hot stream identities")
        if len(self.z_j_active) != len(self.cold_stream_identities):
            raise ValueError("cold active flags must match cold stream identities")
        return self


def build_pinch_design_decomposition(
    problem: PinchProblem,
    dTmin: float,
    *,
    pinch_location: PinchLocation,
    stage_selection: StageSelection = "automated",
) -> PinchDesignDecomposition:
    """Return PDM decomposition fields from OpenPinch-owned problem state.

    The stable contract is prepared ``PinchProblem`` in, copied and
    convention-normalized ``Zone`` targeting, and identity-preserving structural
    PDM fields out. The helper is private so it cannot become a public synthesis
    entry point or bypass the migration replacement gate.
    """

    if pinch_location not in {"above", "below"}:
        raise ValueError("pinch_location must be 'above' or 'below'.")

    arrays = problem_to_solver_arrays(problem, dTmin)
    target = _calculate_openpinch_targets(problem, dTmin=float(dTmin))
    return _build_decomposition(
        arrays=arrays,
        target=target,
        dTmin=float(dTmin),
        pinch_location=pinch_location,
        stage_selection=stage_selection,
    )


def _calculate_openpinch_targets(
    problem: PinchProblem,
    *,
    dTmin: float,
) -> PinchDesignTarget:
    zone = _zone_with_hen_dt_contribution(problem, dTmin=dTmin)
    target = compute_direct_integration_targets(zone)
    shifted_pinch_temperature = _shifted_pinch_temperature(
        hot_utility_target=target.hot_utility_target,
        cold_utility_target=target.cold_utility_target,
        hot_pinch=target.hot_pinch,
        cold_pinch=target.cold_pinch,
    )
    return PinchDesignTarget(
        hot_utility_target=target.hot_utility_target,
        cold_utility_target=target.cold_utility_target,
        heat_recovery_target=target.heat_recovery_target,
        hot_pinch=target.hot_pinch,
        cold_pinch=target.cold_pinch,
        shifted_pinch_temperature=shifted_pinch_temperature,
    )


def _zone_with_hen_dt_contribution(problem: PinchProblem, *, dTmin: float) -> Zone:
    master_zone = problem.master_zone
    if master_zone is None:
        raise ValueError("PDM decomposition requires a prepared PinchProblem.")
    zone = deepcopy(master_zone)
    _apply_hen_dt_cont_convention(zone, dTmin=dTmin)
    return zone


def _apply_hen_dt_cont_convention(zone: Zone, *, dTmin: float) -> None:
    """Apply the HEN dTmin convention to copied streams before targeting."""

    minimum_dt_cont = float(dTmin) / 2.0
    zone.dt_cont_multiplier = 1.0
    for stream in zone.all_streams:
        stream.dt_cont = _stream_dt_cont_with_minimum(
            stream,
            minimum_dt_cont=minimum_dt_cont,
        )


def _stream_dt_cont_with_minimum(
    stream: Stream,
    *,
    minimum_dt_cont: float,
) -> dict[str, float | list[float] | str]:
    current = getattr(stream, "_dt_cont", None)
    if current is None:
        return {"value": minimum_dt_cont, "unit": "delta_degC"}

    converted = current.to("delta_degC")
    if converted.num_states > 1:
        return {
            "values": np.maximum(converted.state_values, minimum_dt_cont).tolist(),
            "unit": "delta_degC",
        }
    return {
        "value": max(float(converted.value), minimum_dt_cont),
        "unit": "delta_degC",
    }


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


def _build_decomposition(
    *,
    arrays: PreparedSolverArrays,
    target: PinchDesignTarget,
    dTmin: float,
    pinch_location: PinchLocation,
    stage_selection: StageSelection,
) -> PinchDesignDecomposition:
    if target.shifted_pinch_temperature is None:
        raise ValueError("Cannot build PDM fields without a shifted pinch temperature.")

    values = arrays.arrays
    T_h_in = _copy_state_row(values, "T_h_in_state")
    T_h_out = _copy_state_row(values, "T_h_out_state")
    T_c_in = _copy_state_row(values, "T_c_in_state")
    T_c_out = _copy_state_row(values, "T_c_out_state")
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

    return PinchDesignDecomposition(
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
            "Copied stream dt_cont values are max(prepared stream dt_cont, "
            "dTmin / 2), and the copied zone dt_cont_multiplier is set to 1.0."
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


def _copy_state_row(
    values: dict[str, np.ndarray],
    key: str,
    state_idx: int = 0,
) -> np.ndarray:
    return np.asarray(values[key], dtype=float)[state_idx].copy()


__all__ = [
    "PinchDesignDecomposition",
    "PinchDesignTarget",
    "PinchLocation",
    "StageSelection",
    "build_pinch_design_decomposition",
]
