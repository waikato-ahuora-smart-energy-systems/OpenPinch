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
from ..indexing import ordered_mapping_keys
from .arrays import PreparedSolverArrays, problem_to_solver_arrays

PinchLocation = Literal["above", "below"]
StageSelection = Literal["automated"] | tuple[int, int]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class PinchDesignTarget(BaseModel):
    """Semantic OpenPinch target values used by PDM decomposition."""

    model_config = _MODEL_CONFIG

    period_id: str
    period_idx: int
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

    @field_validator("period_idx")
    @classmethod
    def _non_negative_period_idx(cls, value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("period_idx must be non-negative")
        return value

    @field_validator("period_id")
    @classmethod
    def _non_empty_period_id(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("period_id must be non-empty")
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
    period_targets: tuple[PinchDesignTarget, ...]
    z_i_active: tuple[int, ...]
    z_j_active: tuple[int, ...]
    z_i_active_by_period: tuple[tuple[int, ...], ...]
    z_j_active_by_period: tuple[tuple[int, ...], ...]
    clipped_hot_supply_temperatures_by_period: tuple[tuple[float, ...], ...]
    clipped_hot_target_temperatures_by_period: tuple[tuple[float, ...], ...]
    clipped_cold_supply_temperatures_by_period: tuple[tuple[float, ...], ...]
    clipped_cold_target_temperatures_by_period: tuple[tuple[float, ...], ...]
    S: int
    K: int
    manual_stage_selection: tuple[int, int] | None
    hot_stream_identities: tuple[str, ...]
    cold_stream_identities: tuple[str, ...]
    unit_conventions: dict[str, str]
    dt_cont_convention: str

    @field_validator(
        "z_i_active",
        "z_j_active",
        "z_i_active_by_period",
        "z_j_active_by_period",
    )
    @classmethod
    def _binary_flags(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        nested = bool(value and isinstance(value[0], tuple))
        rows = value if nested else (value,)
        flags = tuple(tuple(int(item) for item in row) for row in rows)
        if any(item not in {0, 1} for row in flags for item in row):
            raise ValueError("PDM active-stream flags must be binary")
        return flags if nested else flags[0]

    @field_validator(
        "clipped_hot_supply_temperatures_by_period",
        "clipped_hot_target_temperatures_by_period",
        "clipped_cold_supply_temperatures_by_period",
        "clipped_cold_target_temperatures_by_period",
    )
    @classmethod
    def _finite_temperature_tuple(
        cls,
        value: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        temperatures = tuple(tuple(float(item) for item in row) for row in value)
        if any(not np.isfinite(item) for row in temperatures for item in row):
            raise ValueError("PDM clipped stream temperatures must be finite")
        return temperatures

    @model_validator(mode="after")
    def _consistent_stage_count(self) -> "PinchDesignDecomposition":
        if self.S < 0:
            raise ValueError("PDM stage count S must be non-negative")
        if self.K != self.S + 1:
            raise ValueError("PDM boundary count K must equal S + 1")
        if len(self.z_i_active) != len(self.hot_stream_identities):
            raise ValueError("hot active flags must match hot stream identities")
        if len(self.z_j_active) != len(self.cold_stream_identities):
            raise ValueError("cold active flags must match cold stream identities")
        period_count = len(self.period_targets)
        if period_count == 0:
            raise ValueError("PDM period_targets must be non-empty")
        period_fields = (
            self.z_i_active_by_period,
            self.z_j_active_by_period,
            self.clipped_hot_supply_temperatures_by_period,
            self.clipped_hot_target_temperatures_by_period,
            self.clipped_cold_supply_temperatures_by_period,
            self.clipped_cold_target_temperatures_by_period,
        )
        if any(len(field) != period_count for field in period_fields):
            raise ValueError("PDM period-indexed fields must match period_targets")
        if tuple(target.period_idx for target in self.period_targets) != tuple(
            range(period_count)
        ):
            raise ValueError("PDM period targets must be ordered by period_idx")
        if len({target.period_id for target in self.period_targets}) != period_count:
            raise ValueError("PDM period target identities must be unique")
        if any(
            len(row) != len(self.hot_stream_identities)
            for row in (
                *self.z_i_active_by_period,
                *self.clipped_hot_supply_temperatures_by_period,
                *self.clipped_hot_target_temperatures_by_period,
            )
        ):
            raise ValueError("hot period fields must match hot stream identities")
        if any(
            len(row) != len(self.cold_stream_identities)
            for row in (
                *self.z_j_active_by_period,
                *self.clipped_cold_supply_temperatures_by_period,
                *self.clipped_cold_target_temperatures_by_period,
            )
        ):
            raise ValueError("cold period fields must match cold stream identities")
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
    period_targets = _calculate_openpinch_targets(problem, dTmin=float(dTmin))
    return _build_decomposition(
        arrays=arrays,
        period_targets=period_targets,
        dTmin=float(dTmin),
        pinch_location=pinch_location,
        stage_selection=stage_selection,
    )


def _calculate_openpinch_targets(
    problem: PinchProblem,
    *,
    dTmin: float,
) -> tuple[PinchDesignTarget, ...]:
    zone = _zone_with_hen_dt_contribution(problem, dTmin=dTmin)
    period_ids = ordered_mapping_keys(zone.period_ids or {"0": 0})
    targets = []
    for period_idx, period_id in enumerate(period_ids):
        target = compute_direct_integration_targets(
            zone,
            args={"period_id": period_id},
        )
        shifted_pinch_temperature = _shifted_pinch_temperature(
            hot_utility_target=target.hot_utility_target,
            cold_utility_target=target.cold_utility_target,
            hot_pinch=target.hot_pinch,
            cold_pinch=target.cold_pinch,
        )
        targets.append(
            PinchDesignTarget(
                period_id=str(period_id),
                period_idx=period_idx,
                hot_utility_target=target.hot_utility_target,
                cold_utility_target=target.cold_utility_target,
                heat_recovery_target=target.heat_recovery_target,
                hot_pinch=target.hot_pinch,
                cold_pinch=target.cold_pinch,
                shifted_pinch_temperature=shifted_pinch_temperature,
            )
        )
    return tuple(targets)


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
        if stream.has_segments:
            _apply_segment_dt_cont_minimum(
                stream,
                minimum_dt_cont=minimum_dt_cont,
            )
        else:
            stream.dt_cont = _stream_dt_cont_with_minimum(
                stream,
                minimum_dt_cont=minimum_dt_cont,
            )


def _apply_segment_dt_cont_minimum(
    stream: Stream,
    *,
    minimum_dt_cont: float,
) -> None:
    """Apply the HEN contribution to every child used by numeric targeting."""

    stream.update_segments(
        {
            segment_index: {
                "dt_cont": _stream_dt_cont_with_minimum(
                    segment,
                    minimum_dt_cont=minimum_dt_cont,
                )
            }
            for segment_index, segment in enumerate(stream.segments)
        }
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
    if converted.num_periods > 1:
        return {
            "values": np.maximum(converted.period_values, minimum_dt_cont).tolist(),
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
    period_targets: tuple[PinchDesignTarget, ...],
    dTmin: float,
    pinch_location: PinchLocation,
    stage_selection: StageSelection,
) -> PinchDesignDecomposition:
    values = arrays.arrays
    period_count = len(period_targets)
    if period_count == 0:
        raise ValueError("PDM period_targets must be non-empty.")
    array_period_ids = tuple(str(value) for value in values["period_ids"])
    target_period_ids = tuple(target.period_id for target in period_targets)
    if target_period_ids != array_period_ids:
        raise ValueError("PDM period target identities must match solver arrays.")
    state_names = (
        "T_h_in_period",
        "T_h_out_period",
        "T_c_in_period",
        "T_c_out_period",
    )
    state_rows = {
        name: np.asarray(values[name], dtype=float).copy() for name in state_names
    }
    if any(rows.shape[0] != period_count for rows in state_rows.values()):
        raise ValueError("PDM period targets must match solver-array periods.")
    z_i_active_period = []
    z_j_active_period = []
    for period_idx, target in enumerate(period_targets):
        if target.shifted_pinch_temperature is None:
            raise ValueError(
                f"Cannot build PDM fields without a shifted pinch temperature for "
                f"period_id {target.period_id!r}."
            )
        z_i_active, z_j_active = _clip_stream_temperatures(
            T_h_in=state_rows["T_h_in_period"][period_idx],
            T_h_out=state_rows["T_h_out_period"][period_idx],
            T_c_in=state_rows["T_c_in_period"][period_idx],
            T_c_out=state_rows["T_c_out_period"][period_idx],
            shifted_pinch_temperature=target.shifted_pinch_temperature,
            dTmin=dTmin,
            pinch_location=pinch_location,
        )
        z_i_active_period.append(z_i_active)
        z_j_active_period.append(z_j_active)
    z_i_active = tuple(
        int(any(row[index] for row in z_i_active_period))
        for index in range(len(z_i_active_period[0]))
    )
    z_j_active = tuple(
        int(any(row[index] for row in z_j_active_period))
        for index in range(len(z_j_active_period[0]))
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
        period_targets=period_targets,
        z_i_active=z_i_active,
        z_j_active=z_j_active,
        z_i_active_by_period=tuple(z_i_active_period),
        z_j_active_by_period=tuple(z_j_active_period),
        clipped_hot_supply_temperatures_by_period=tuple(
            tuple(row) for row in state_rows["T_h_in_period"]
        ),
        clipped_hot_target_temperatures_by_period=tuple(
            tuple(row) for row in state_rows["T_h_out_period"]
        ),
        clipped_cold_supply_temperatures_by_period=tuple(
            tuple(row) for row in state_rows["T_c_in_period"]
        ),
        clipped_cold_target_temperatures_by_period=tuple(
            tuple(row) for row in state_rows["T_c_out_period"]
        ),
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
            "Copied stream and explicit segment dt_cont values are "
            "max(prepared dt_cont, dTmin / 2), and the copied zone "
            "dt_cont_multiplier is set to 1.0."
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
    period_idx: int = 0,
) -> np.ndarray:
    return np.asarray(values[key], dtype=float)[period_idx].copy()


__all__ = [
    "PinchDesignDecomposition",
    "PinchDesignTarget",
    "PinchLocation",
    "StageSelection",
    "build_pinch_design_decomposition",
]
