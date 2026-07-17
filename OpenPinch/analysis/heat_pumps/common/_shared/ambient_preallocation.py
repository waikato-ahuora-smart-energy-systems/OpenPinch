"""Ambient-air preallocation helpers for simulated HPR targeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....contracts.hpr import HeatPumpTargetInputs
from .....domain.configuration import tol
from .....domain.stream_collection import StreamCollection
from ..preprocessing import _create_stream_collection_of_background_profile
from .streams import get_Q_vals_at_T_hpr_from_bckgrd_profile


@dataclass(frozen=True)
class DirectAmbientPreallocation:
    Q_amb_hot: float
    Q_amb_cold: float
    Q_amb_hot_direct: float
    Q_amb_cold_direct: float
    Q_amb_hot_residual: float
    Q_amb_cold_residual: float
    T_hot_residual: np.ndarray
    H_hot_residual: np.ndarray
    T_cold_residual: np.ndarray
    H_cold_residual: np.ndarray
    bckgrd_hot_streams: StreamCollection
    bckgrd_cold_streams: StreamCollection

    @property
    def Q_heat_max_residual(self) -> float:
        return max(float(self.H_cold_residual[0]), 0.0)

    @property
    def Q_cool_max_residual(self) -> float:
        return max(float(-self.H_hot_residual[-1]), 0.0)

    @property
    def Q_heat_capacity(self) -> float:
        return self.Q_heat_max_residual + self.Q_amb_cold_residual

    @property
    def Q_cool_capacity(self) -> float:
        return self.Q_cool_max_residual + self.Q_amb_hot_residual

    def H_hot_with_residual_ambient(self, args: HeatPumpTargetInputs) -> np.ndarray:
        z_amb_hot = _interp_profile_values(
            self.T_hot_residual,
            args.T_hot,
            args.z_amb_hot,
        )
        return self.H_hot_residual + z_amb_hot * self.Q_amb_hot_residual

    def H_cold_with_residual_ambient(self, args: HeatPumpTargetInputs) -> np.ndarray:
        z_amb_cold = _interp_profile_values(
            self.T_cold_residual,
            args.T_cold,
            args.z_amb_cold,
        )
        return self.H_cold_residual + z_amb_cold * self.Q_amb_cold_residual


def preallocate_direct_ambient_duties(
    *,
    args: HeatPumpTargetInputs,
    Q_amb_hot: float,
    Q_amb_cold: float,
) -> DirectAmbientPreallocation:
    """Allocate ambient duties directly to the opposite background profiles."""
    Q_amb_hot = max(float(Q_amb_hot), 0.0)
    Q_amb_cold = max(float(Q_amb_cold), 0.0)
    if Q_amb_cold > tol:
        T_amb_sink = _ambient_sink_temperature(args)
        available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
            np.array([T_amb_sink]),
            args.T_hot,
            args.H_hot,
            is_cond=False,
        )
        Q_amb_cold_direct = min(
            Q_amb_cold,
            max(float(available[0]), 0.0),
        )
        T_hot_residual, H_hot_residual = _remove_direct_ambient_sink_from_hot_profile(
            args.T_hot,
            args.H_hot,
            Q_amb_cold_direct,
            T_amb_sink,
        )
    else:
        Q_amb_cold_direct = 0.0
        T_hot_residual = np.asarray(args.T_hot, dtype=float)
        H_hot_residual = np.asarray(args.H_hot, dtype=float)

    if Q_amb_hot > tol:
        T_amb_source = _ambient_source_temperature(args)
        available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
            np.array([T_amb_source]),
            args.T_cold,
            args.H_cold,
            is_cond=True,
        )
        Q_amb_hot_direct = min(
            Q_amb_hot,
            max(float(available[0]), 0.0),
        )
        T_cold_residual, H_cold_residual = (
            _remove_direct_ambient_source_from_cold_profile(
                args.T_cold,
                args.H_cold,
                Q_amb_hot_direct,
                T_amb_source,
            )
        )
    else:
        Q_amb_hot_direct = 0.0
        T_cold_residual = np.asarray(args.T_cold, dtype=float)
        H_cold_residual = np.asarray(args.H_cold, dtype=float)

    return DirectAmbientPreallocation(
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        Q_amb_hot_direct=Q_amb_hot_direct,
        Q_amb_cold_direct=Q_amb_cold_direct,
        Q_amb_hot_residual=max(Q_amb_hot - Q_amb_hot_direct, 0.0),
        Q_amb_cold_residual=max(Q_amb_cold - Q_amb_cold_direct, 0.0),
        T_hot_residual=T_hot_residual,
        H_hot_residual=H_hot_residual,
        T_cold_residual=T_cold_residual,
        H_cold_residual=H_cold_residual,
        bckgrd_hot_streams=_create_stream_collection_of_background_profile(
            T_hot_residual,
            H_hot_residual,
        ),
        bckgrd_cold_streams=_create_stream_collection_of_background_profile(
            T_cold_residual,
            H_cold_residual,
        ),
    )


def _ambient_source_temperature(args: HeatPumpTargetInputs) -> float:
    return (
        float(args.T_env)
        - float(getattr(args, "dt_env_cont", 0.0))
        + float(getattr(args, "dtcont_hp", 0.0))
    )


def _ambient_sink_temperature(args: HeatPumpTargetInputs) -> float:
    return (
        float(args.T_env)
        + float(getattr(args, "dt_env_cont", 0.0))
        - float(getattr(args, "dtcont_hp", 0.0))
    )


def _remove_direct_ambient_sink_from_hot_profile(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    duty: float,
    T_amb: float,
) -> tuple[np.ndarray, np.ndarray]:
    T_hot = np.asarray(T_hot, dtype=float)
    H_hot = np.asarray(H_hot, dtype=float)
    duty = max(float(duty), 0.0)
    if duty <= tol:
        return T_hot, H_hot

    T_amb_profile = np.clip(float(T_amb), T_hot[-1], T_hot[0])
    H_amb = float(_interp_profile_values(np.array([T_amb_profile]), T_hot, H_hot)[0])
    duty = min(duty, max(-H_amb, 0.0))
    if duty <= tol:
        return T_hot, H_hot

    H_cut = H_amb + duty
    T_cut = float(np.interp(H_cut, H_hot[::-1], T_hot[::-1]))
    T_residual, H_original = _insert_profile_temperatures(
        T_hot,
        H_hot,
        T_amb_profile,
        T_cut,
    )
    H_residual = H_original.copy()
    in_removed_zone = (T_residual <= T_cut + tol) & (T_residual >= T_amb_profile - tol)
    below_ambient = T_residual < T_amb_profile - tol
    H_residual[in_removed_zone] = H_cut
    H_residual[below_ambient] = H_original[below_ambient] + duty
    H_residual[np.abs(H_residual) < tol] = 0.0
    return T_residual, H_residual


def _remove_direct_ambient_source_from_cold_profile(
    T_cold: np.ndarray,
    H_cold: np.ndarray,
    duty: float,
    T_amb: float,
) -> tuple[np.ndarray, np.ndarray]:
    T_cold = np.asarray(T_cold, dtype=float)
    H_cold = np.asarray(H_cold, dtype=float)
    duty = max(float(duty), 0.0)
    if duty <= tol:
        return T_cold, H_cold

    T_amb_profile = np.clip(float(T_amb), T_cold[-1], T_cold[0])
    H_amb = float(_interp_profile_values(np.array([T_amb_profile]), T_cold, H_cold)[0])
    duty = min(duty, max(H_amb, 0.0))
    if duty <= tol:
        return T_cold, H_cold

    T_cut = float(np.interp(duty, H_cold[::-1], T_cold[::-1]))
    T_residual, H_original = _insert_profile_temperatures(
        T_cold,
        H_cold,
        T_amb_profile,
        T_cut,
    )
    H_residual = np.maximum(H_original - duty, 0.0)
    H_residual[T_residual < T_cut - tol] = 0.0
    H_residual[np.abs(H_residual) < tol] = 0.0
    return T_residual, H_residual


def _insert_profile_temperatures(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *T_insert: float,
) -> tuple[np.ndarray, np.ndarray]:
    T_vals = np.asarray(T_vals, dtype=float)
    H_vals = np.asarray(H_vals, dtype=float)
    T_new = [float(T) for T in T_vals]
    for T in T_insert:
        T = float(T)
        if T < T_vals[-1] - tol or T > T_vals[0] + tol:
            continue
        if not np.any(np.isclose(np.asarray(T_new), T, atol=tol, rtol=0.0)):
            T_new.append(T)
    T_out = np.array(sorted(T_new, reverse=True), dtype=float)
    H_out = _interp_profile_values(T_out, T_vals, H_vals)
    return T_out, H_out


def _interp_profile_values(
    T_new: np.ndarray,
    T_vals: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(
        np.asarray(T_new, dtype=float),
        np.asarray(T_vals, dtype=float)[::-1],
        np.asarray(values, dtype=float)[::-1],
    )
