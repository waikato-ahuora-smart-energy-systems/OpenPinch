"""Normalisation helpers for optimisation vectors used in HP targeting."""

from typing import Tuple

import numpy as np

__all__ = [
    "MAX_AMBIENT_X_ABS",
    "map_x_arr_to_T_arr",
    "map_T_arr_to_x_arr",
    "map_x_arr_to_DT_arr",
    "map_DT_arr_to_x_arr",
    "map_x_arr_to_Q_arr",
    "map_Q_arr_to_x_arr",
    "map_x_to_Q_amb",
    "map_Q_amb_to_x",
]

MAX_AMBIENT_X_ABS = 0.999


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
