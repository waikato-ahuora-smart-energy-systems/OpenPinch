"""Normalisation helpers for optimisation vectors used in HP targeting."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "AMBIENT_X_BOUNDS",
    "DutyAllocation",
    "allocate_stage_duties",
    "decode_duty_splits",
    "encode_base_and_duty_splits",
    "encode_duty_splits",
    "map_x_arr_to_T_arr",
    "map_T_arr_to_x_arr",
    "map_x_arr_to_DT_arr",
    "map_DT_arr_to_x_arr",
    "map_x_arr_to_Q_arr",
    "map_Q_arr_to_x_arr",
    "map_x_to_Q_amb",
    "map_Q_amb_to_x",
    "require_stage_duty_allocation",
]

MAX_AMBIENT_X_ABS = 0.999
AMBIENT_X_BOUNDS = (-MAX_AMBIENT_X_ABS, MAX_AMBIENT_X_ABS)


@dataclass(frozen=True)
class DutyAllocation:
    """Decoded stage-duty allocation from one base duty and split vector."""

    Q_base: float
    Q_request: np.ndarray
    Q_available: np.ndarray
    Q_model: np.ndarray
    Q_excess: np.ndarray


def decode_duty_splits(x_split: np.ndarray, Q_base: float) -> np.ndarray:
    """Decode stick-breaking split fractions into nonnegative stage duties."""
    x_split = np.asarray(x_split, dtype=float).reshape(-1)
    remaining = max(float(Q_base), 0.0)
    Q_request = np.zeros_like(x_split, dtype=float)
    for i, fraction in enumerate(np.clip(x_split, 0.0, 1.0)):
        Q_request[i] = fraction * remaining
        remaining -= Q_request[i]
    return Q_request


def encode_duty_splits(Q_request: np.ndarray, Q_base: float) -> np.ndarray:
    """Encode nonnegative stage duties as bounded stick-breaking fractions."""
    Q_request = np.maximum(np.asarray(Q_request, dtype=float).reshape(-1), 0.0)
    remaining = max(float(Q_base), 0.0)
    x_split = np.zeros_like(Q_request, dtype=float)
    for i, duty in enumerate(Q_request):
        if remaining <= 0.0:
            break
        duty = min(float(duty), remaining)
        x_split[i] = duty / remaining
        remaining -= duty
    return x_split


def encode_base_and_duty_splits(
    Q_request: np.ndarray,
    Q_limit: float,
) -> tuple[float, float, np.ndarray]:
    """Encode a seed duty vector into base-duty scale and split fractions."""
    Q_request = np.asarray(Q_request, dtype=float).reshape(-1)
    Q_request = np.where(np.isfinite(Q_request), np.maximum(Q_request, 0.0), 0.0)
    Q_limit = float(Q_limit) if np.isfinite(Q_limit) and Q_limit > 0.0 else 0.0
    Q_request_sum = float(Q_request.sum())
    if Q_request_sum > Q_limit and Q_request_sum > 0.0:
        Q_request = Q_request * (Q_limit / Q_request_sum)
        Q_request_sum = float(Q_request.sum())
    Q_base = min(Q_request_sum, Q_limit)
    x_base = Q_base / Q_limit if Q_limit > 0.0 else 0.0
    return Q_base, x_base, encode_duty_splits(Q_request, Q_base)


def allocate_stage_duties(
    Q_base: float,
    x_split: np.ndarray,
    Q_available: np.ndarray,
) -> DutyAllocation:
    """Decode and availability-limit per-stage model duties."""
    Q_request = decode_duty_splits(x_split, Q_base)
    Q_available = np.maximum(np.asarray(Q_available, dtype=float).reshape(-1), 0.0)
    if Q_available.size != Q_request.size:
        raise ValueError(
            "Q_available must have the same length as the duty split vector."
        )
    Q_model = np.minimum(Q_request, Q_available)
    Q_excess = np.maximum(Q_request - Q_available, 0.0)
    return DutyAllocation(
        Q_base=max(float(Q_base), 0.0),
        Q_request=Q_request,
        Q_available=Q_available,
        Q_model=Q_model,
        Q_excess=Q_excess,
    )


def require_stage_duty_allocation(
    *,
    Q_base: float,
    x_split: np.ndarray | None,
    Q_available: np.ndarray | None,
    duty_name: str,
) -> DutyAllocation:
    """Validate and allocate one base/split/availability duty payload."""
    if x_split is None or Q_available is None:
        raise ValueError(
            f"Q_{duty_name}_base requires x_{duty_name}_split "
            f"and Q_{duty_name}_available."
        )
    return allocate_stage_duties(
        Q_base,
        np.asarray(x_split, dtype=float).reshape(-1),
        Q_available,
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


def map_x_to_Q_amb(
    x: float,
    scale: float,
) -> Tuple[float, float]:
    """Split one signed bounded ambient variable into hot and cold duties.

    ``x`` is interpreted on the open interval ``(-1, 1)`` and decoded through
    ``atanh`` so the mapping stays close to linear around zero while ambient
    duties remain unbounded.
    """
    if scale <= 0.0:
        return 0.0, 0.0

    x_arr = np.asarray(x, dtype=float)
    x_clip = np.clip(x_arr, -MAX_AMBIENT_X_ABS, MAX_AMBIENT_X_ABS)
    q_signed = scale * np.arctanh(x_clip)
    q_hot = np.maximum(-q_signed, 0.0)
    q_cold = np.maximum(q_signed, 0.0)
    return float(q_hot), float(q_cold)


def map_Q_amb_to_x(
    Q_amb_hot: float,
    Q_amb_cold: float,
    scale: float,
) -> float:
    """Encode ambient duties back into one bounded signed decision variable."""
    if scale <= 0.0:
        return 0.0

    q_signed = float(Q_amb_cold) - float(Q_amb_hot)
    return float(np.tanh(q_signed / scale))
