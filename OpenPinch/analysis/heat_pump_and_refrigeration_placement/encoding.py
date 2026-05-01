"""Normalisation helpers for optimisation vectors used in HP targeting."""

from typing import Tuple

import numpy as np


__all__ = [
    "map_x_arr_to_T_arr",
    "map_T_arr_to_x_arr",
    "map_x_arr_to_DT_arr",
    "map_DT_arr_to_x_arr",
    "map_x_arr_to_Q_arr",
    "map_Q_arr_to_x_arr",
    "map_x_to_Q_amb",
    "map_Q_amb_to_x",
]


def map_x_arr_to_T_arr(
    x: np.ndarray,
    T_0: float,
    T_1: float,
) -> np.ndarray:
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
    return x * np.abs(T_arr - T_last)


def map_DT_arr_to_x_arr(
    DT_arr: np.ndarray,
    T_arr: np.ndarray,
    T_last: float,
) -> np.ndarray:
    return np.where(
        T_arr != T_last,
        DT_arr / np.abs(T_arr - T_last),
        0.0,
    )


def map_x_arr_to_Q_arr(
    x: np.ndarray,
    Q_max: float,
) -> np.ndarray:
    return x * Q_max


def map_Q_arr_to_x_arr(
    Q_arr: np.ndarray,
    Q_max: float,
) -> np.ndarray:
    return np.where(Q_max != 0, Q_arr / Q_max, 0.0)


def map_x_to_Q_amb(
    x: float,
    scale: float,
) -> Tuple[float, float]:
    Q_amb_hot = max(-scale * x, 0.0)
    Q_amb_cold = max(scale * x, 0.0)
    return Q_amb_hot, Q_amb_cold


def map_Q_amb_to_x(
    Q_amb_hot: float,
    Q_amb_cold: float,
    scale: float,
) -> float:
    return (Q_amb_cold - Q_amb_hot) / scale
