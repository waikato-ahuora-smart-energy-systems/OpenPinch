"""Normalisation helpers for optimisation vectors used in HP targeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ....contracts.hpr import HPRParsedState

__all__ = [
    "DutyAllocation",
    "DutyAllocationRequest",
    "StageDutyRequest",
    "allocate_stage_duties",
    "decode_available_fractions",
    "encode_available_fractions",
    "limit_available_duty",
    "map_x_arr_to_T_arr",
    "map_T_arr_to_x_arr",
    "map_x_arr_to_DT_arr",
    "map_DT_arr_to_x_arr",
    "map_x_arr_to_Q_arr",
    "map_Q_arr_to_x_arr",
    "require_stage_duty_allocation",
]



@dataclass(frozen=True)
class DutyAllocation:
    """Stage duties decoded as fractions of each stage's available duty."""

    Q_base: float
    Q_available: np.ndarray
    Q_model: np.ndarray


def decode_available_fractions(
    x_split: np.ndarray,
    Q_available: np.ndarray,
) -> np.ndarray:
    """Return stage duties as fractions of each stage's available duty.

    Every fraction vector maps to a distinct feasible duty vector: no stage can
    request more than it can deliver, so nothing is capped and the objective
    stays sensitive to every fraction whose stage has duty available.
    """
    fractions = np.clip(np.asarray(x_split, dtype=float), 0.0, 1.0)
    return fractions * np.maximum(np.asarray(Q_available, dtype=float), 0.0)


def limit_available_duty(Q_available: np.ndarray, capacity: float) -> np.ndarray:
    """Scale stage availabilities so together they cannot exceed ``capacity``.

    The limit applies to availability, not to decoded duties, so each fraction
    still acts on its own stage alone and every fraction vector stays a
    distinct design. Use it where the total duty is capped, for example by the
    selected heat-pump load when ambient air adds sink capacity beyond it.
    """
    Q_available = np.maximum(np.asarray(Q_available, dtype=float), 0.0)
    total = float(Q_available.sum())
    capacity = max(float(capacity), 0.0)
    if total <= capacity or total <= 0.0:
        return Q_available
    return Q_available * (capacity / total)


def encode_available_fractions(
    Q_request: np.ndarray,
    Q_available: np.ndarray,
) -> np.ndarray:
    """Encode seed stage duties as fractions of each stage's available duty.

    Duties above availability encode as 1; stages with nothing available
    encode as 0.
    """
    Q_request = np.asarray(Q_request, dtype=float)
    Q_request = np.where(np.isfinite(Q_request), np.maximum(Q_request, 0.0), 0.0)
    Q_available = np.maximum(np.asarray(Q_available, dtype=float), 0.0)
    if Q_request.shape != Q_available.shape:
        raise ValueError("Q_request and Q_available must have the same shape.")
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = np.where(Q_available > 0.0, Q_request / Q_available, 0.0)
    return np.clip(fractions, 0.0, 1.0)


def allocate_stage_duties(
    x_split: np.ndarray,
    Q_available: np.ndarray,
) -> DutyAllocation:
    """Decode per-stage duties from availability fractions."""
    Q_available = np.maximum(np.asarray(Q_available, dtype=float), 0.0)
    if Q_available.size != np.asarray(x_split).size:
        raise ValueError(
            "Q_available must have the same length as the duty split vector."
        )
    Q_model = decode_available_fractions(x_split, Q_available)
    return DutyAllocation(
        Q_base=float(Q_model.sum()),
        Q_available=Q_available,
        Q_model=Q_model,
    )


def require_stage_duty_allocation(
    *,
    x_split: np.ndarray | None,
    Q_available: np.ndarray | None,
    duty_name: str,
) -> DutyAllocation:
    """Validate and allocate one split/availability duty input set."""
    if x_split is None or Q_available is None:
        raise ValueError(
            f"Q_{duty_name}_base requires x_{duty_name}_split "
            f"and Q_{duty_name}_available."
        )
    return allocate_stage_duties(x_split, Q_available)


@dataclass(frozen=True)
class StageDutyRequest:
    """Availability fractions and available duty for one process side.

    ``Q_base`` is the total decoded duty; ``None`` means no allocation is
    requested for this side.
    """

    Q_base: float | None = None
    x_split: np.ndarray | None = None
    Q_available: np.ndarray | None = None

    def allocate(self, duty_name: str) -> DutyAllocation:
        """Validate and allocate this side's stage duties."""
        return require_stage_duty_allocation(
            x_split=self.x_split,
            Q_available=self.Q_available,
            duty_name=duty_name,
        )


@dataclass(frozen=True)
class DutyAllocationRequest:
    """Heat-side and cool-side duty-allocation inputs for a cycle solve."""

    heat: StageDutyRequest = field(default_factory=StageDutyRequest)
    cool: StageDutyRequest = field(default_factory=StageDutyRequest)

    @classmethod
    def from_state(cls, state: HPRParsedState) -> DutyAllocationRequest:
        """Collect the duty-allocation fields of a parsed optimisation state."""
        return cls(
            heat=StageDutyRequest(
                Q_base=state.Q_heat_base,
                x_split=state.x_heat_split,
                Q_available=state.Q_heat_available,
            ),
            cool=StageDutyRequest(
                Q_base=state.Q_cool_base,
                x_split=state.x_cool_split,
                Q_available=state.Q_cool_available,
            ),
        )


def map_x_arr_to_T_arr(
    x: np.ndarray,
    T_0: float,
    T_1: float,
) -> np.ndarray:
    """Map cumulative optimisation fractions onto descending stage temperatures."""
    temp = []
    for i in range(x.size):
        temp.append(T_0 - x[i] * (T_0 - T_1))
        T_0 = temp[-1]
    return np.sort(np.array(temp).flatten())[::-1]


def map_T_arr_to_x_arr(
    T_arr: np.ndarray,
    T_0: float,
    T_1: float,
) -> np.ndarray:
    """Encode descending stage temperatures as cumulative optimisation fractions."""
    temp = []
    for i in range(T_arr.size):
        temp.append((T_0 - T_arr[i]) / (T_0 - T_1) if T_0 != T_1 else 0.0)
        T_0 = T_arr[i]
    return np.array(temp)


def map_x_arr_to_DT_arr(
    x: np.ndarray,
    T_arr: np.ndarray,
    T_last: float,
) -> np.ndarray:
    """Scale optimisation fractions into temperature differences."""
    return x * np.abs(T_arr - T_last)


def map_DT_arr_to_x_arr(
    DT_arr: np.ndarray,
    T_arr: np.ndarray,
    T_last: float,
) -> np.ndarray:
    """Normalise temperature differences back into optimisation fractions."""
    return np.where(
        T_arr != T_last,
        DT_arr / np.abs(T_arr - T_last),
        0.0,
    )


def map_x_arr_to_Q_arr(
    x: np.ndarray,
    Q_max: float,
) -> np.ndarray:
    """Scale optimisation fractions into heat duties."""
    return x * Q_max


def map_Q_arr_to_x_arr(
    Q_arr: np.ndarray,
    Q_max: float,
) -> np.ndarray:
    """Normalise heat duties back into optimisation fractions."""
    return np.where(Q_max != 0, Q_arr / Q_max, 0.0)
