"""Piecewise parent heat coordinates and duty-aligned exchanger area slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....classes.heat_exchanger import HeatExchangerAreaSlice
from .....lib.config import tol
from .....utils.heat_exchanger import compute_LMTD_from_dts
from .arrays import PreparedSolverArrays


@dataclass(frozen=True)
class PiecewiseThermalProfile:
    """One parent stream's ordered local thermal intervals for one period."""

    identities: tuple[str, ...]
    temperatures_in: np.ndarray
    temperatures_out: np.ndarray
    duties: np.ndarray
    heat_capacity_flowrates: np.ndarray
    heat_transfer_coefficients: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.temperatures_in,
            self.temperatures_out,
            self.duties,
            self.heat_capacity_flowrates,
            self.heat_transfer_coefficients,
        )
        lengths = {len(values) for values in arrays}
        if not self.identities or lengths != {len(self.identities)}:
            raise ValueError("Piecewise profile arrays must have one common length.")
        if any(np.asarray(values).ndim != 1 for values in arrays):
            raise ValueError("Piecewise profile arrays must be one-dimensional.")
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("Piecewise profile values must be finite.")
        if np.any(self.duties <= tol):
            raise ValueError("Piecewise profile duties must be positive.")
        if np.any(self.heat_capacity_flowrates <= tol):
            raise ValueError("Piecewise profile heat capacities must be positive.")
        if np.any(self.heat_transfer_coefficients <= tol):
            raise ValueError("Piecewise profile HTCs must be positive.")
        temperature_steps = self.temperatures_out - self.temperatures_in
        if len(self.identities) > 1 and (
            np.any(np.abs(temperature_steps) <= tol)
            or not np.all(np.sign(temperature_steps) == np.sign(temperature_steps[0]))
        ):
            raise ValueError("Piecewise profile segments must have one direction.")
        if len(self.identities) > 1 and not np.allclose(
            self.temperatures_out[:-1],
            self.temperatures_in[1:],
            atol=tol,
            rtol=0.0,
        ):
            raise ValueError(
                "Piecewise profile segment temperatures must be continuous."
            )

    @property
    def cumulative_duties(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.duties)))

    @property
    def total_duty(self) -> float:
        return float(np.sum(self.duties))

    def segment_index_at_heat(self, heat_coordinate: float) -> int:
        q = min(max(float(heat_coordinate), 0.0), self.total_duty)
        index = int(np.searchsorted(self.cumulative_duties, q, side="right") - 1)
        return min(max(index, 0), len(self.duties) - 1)

    def temperature_at_heat(self, heat_coordinate: float) -> float:
        q = min(max(float(heat_coordinate), 0.0), self.total_duty)
        index = self.segment_index_at_heat(q)
        q_start = self.cumulative_duties[index]
        fraction = (q - q_start) / self.duties[index]
        return float(
            self.temperatures_in[index]
            + fraction * (self.temperatures_out[index] - self.temperatures_in[index])
        )

    def heat_at_temperature(self, temperature: float) -> float:
        value = float(temperature)
        lower = min(float(self.temperatures_in[0]), float(self.temperatures_out[-1]))
        upper = max(float(self.temperatures_in[0]), float(self.temperatures_out[-1]))
        if value < lower - tol or value > upper + tol:
            raise ValueError(
                f"Temperature {value} is outside the piecewise profile range "
                f"[{lower}, {upper}]."
            )
        value = min(max(value, lower), upper)
        for index, (t_in, t_out) in enumerate(
            zip(self.temperatures_in, self.temperatures_out)
        ):
            lower, upper = sorted((float(t_in), float(t_out)))
            if lower - tol <= value <= upper + tol:
                if abs(t_out - t_in) <= tol:
                    return float(self.cumulative_duties[index])
                fraction = (value - t_in) / (t_out - t_in)
                return float(
                    self.cumulative_duties[index] + fraction * self.duties[index]
                )
        raise ValueError(f"Temperature {value} could not be mapped to the profile.")

    def clipped(
        self,
        supply_temperature: float,
        target_temperature: float,
    ) -> "PiecewiseThermalProfile":
        """Return the ordered profile portion between two parent temperatures."""
        q_start = self.heat_at_temperature(supply_temperature)
        q_end = self.heat_at_temperature(target_temperature)
        if q_end < q_start:
            q_start, q_end = q_end, q_start
        if q_end - q_start <= tol:
            raise ValueError("Clipped segment profile must retain positive duty.")
        identities = []
        temperatures_in = []
        temperatures_out = []
        duties = []
        cps = []
        htcs = []
        for index in range(len(self.duties)):
            local_start = max(q_start, self.cumulative_duties[index])
            local_end = min(q_end, self.cumulative_duties[index + 1])
            if local_end - local_start <= tol:
                continue
            identities.append(self.identities[index])
            temperatures_in.append(self.temperature_at_heat(local_start))
            temperatures_out.append(self.temperature_at_heat(local_end))
            duties.append(local_end - local_start)
            cps.append(self.heat_capacity_flowrates[index])
            htcs.append(self.heat_transfer_coefficients[index])
        return PiecewiseThermalProfile(
            identities=tuple(identities),
            temperatures_in=np.asarray(temperatures_in, dtype=float),
            temperatures_out=np.asarray(temperatures_out, dtype=float),
            duties=np.asarray(duties, dtype=float),
            heat_capacity_flowrates=np.asarray(cps, dtype=float),
            heat_transfer_coefficients=np.asarray(htcs, dtype=float),
        )


def profile_from_solver_arrays(
    arrays: PreparedSolverArrays,
    *,
    side: str,
    parent_index: int,
    period_index: int,
) -> PiecewiseThermalProfile:
    """Read one unpadded parent profile from prepared solver tensors."""
    count = int(arrays.arrays[f"{side}_segment_count"][parent_index])
    prefix = f"{side}_segment"
    identities = arrays.arrays[f"{prefix}_identities"][parent_index, :count]
    return PiecewiseThermalProfile(
        identities=tuple(str(value) for value in identities),
        temperatures_in=arrays.arrays[f"{prefix}_t_in_period"][
            period_index, parent_index, :count
        ].astype(float),
        temperatures_out=arrays.arrays[f"{prefix}_t_out_period"][
            period_index, parent_index, :count
        ].astype(float),
        duties=arrays.arrays[f"{prefix}_duty_period"][
            period_index, parent_index, :count
        ].astype(float),
        heat_capacity_flowrates=arrays.arrays[f"{prefix}_cp_period"][
            period_index, parent_index, :count
        ].astype(float),
        heat_transfer_coefficients=arrays.arrays[f"{prefix}_htc_period"][
            period_index, parent_index, :count
        ].astype(float),
    )


def utility_thermal_profile(
    *,
    identity: str,
    inlet_temperature: float,
    outlet_temperature: float,
    duty: float,
    heat_transfer_coefficient: float,
) -> PiecewiseThermalProfile:
    """Represent one utility duty as a flat virtual segment for local slicing."""
    delta = abs(float(inlet_temperature) - float(outlet_temperature))
    cp = float(duty) / delta if delta > tol else float(duty) / tol
    return PiecewiseThermalProfile(
        identities=(identity,),
        temperatures_in=np.array([inlet_temperature], dtype=float),
        temperatures_out=np.array([outlet_temperature], dtype=float),
        duties=np.array([duty], dtype=float),
        heat_capacity_flowrates=np.array([cp], dtype=float),
        heat_transfer_coefficients=np.array([heat_transfer_coefficient], dtype=float),
    )


def duty_aligned_area_contributions(
    hot_profile: PiecewiseThermalProfile,
    cold_profile: PiecewiseThermalProfile,
    *,
    duty: float,
    hot_inlet_temperature: float,
    cold_inlet_temperature: float,
    period: str,
    tolerance: float = tol,
) -> tuple[HeatExchangerAreaSlice, ...]:
    """Split a parent exchanger at each traversed hot or cold segment boundary."""
    remaining = float(duty)
    if remaining <= tolerance:
        return ()
    hot_q = hot_profile.heat_at_temperature(hot_inlet_temperature)
    cold_inlet_q = cold_profile.heat_at_temperature(cold_inlet_temperature)
    cold_q = cold_inlet_q + remaining
    if hot_q + remaining > hot_profile.total_duty + tolerance or cold_q > (
        cold_profile.total_duty + tolerance
    ):
        raise ValueError("Exchanger duty exceeds an available parent segment profile.")
    contributions = []
    while remaining > tolerance:
        hot_index = hot_profile.segment_index_at_heat(hot_q)
        cold_index = int(
            np.searchsorted(cold_profile.cumulative_duties, cold_q, side="left") - 1
        )
        cold_index = min(max(cold_index, 0), len(cold_profile.duties) - 1)
        hot_boundary = hot_profile.cumulative_duties[hot_index + 1]
        cold_boundary = cold_profile.cumulative_duties[cold_index]
        slice_duty = min(
            remaining,
            max(hot_boundary - hot_q, 0.0),
            max(cold_q - cold_boundary, 0.0),
        )
        if slice_duty <= tolerance:
            if hot_boundary - hot_q <= tolerance:
                hot_q = hot_boundary
            if cold_q - cold_boundary <= tolerance:
                cold_q = cold_boundary
            if hot_q >= hot_profile.total_duty or cold_q <= 0.0:
                break
            continue

        hot_out = hot_profile.temperature_at_heat(hot_q + slice_duty)
        cold_out = cold_profile.temperature_at_heat(cold_q)
        hot_in = hot_profile.temperature_at_heat(hot_q)
        cold_in = cold_profile.temperature_at_heat(cold_q - slice_duty)
        delta_1 = hot_in - cold_out
        delta_2 = hot_out - cold_in
        if delta_1 <= tolerance or delta_2 <= tolerance:
            raise ValueError(
                "Segmented exchanger slice has a non-positive terminal "
                "temperature difference."
            )
        lmtd = float(compute_LMTD_from_dts(delta_1, delta_2))
        hot_htc = float(hot_profile.heat_transfer_coefficients[hot_index])
        cold_htc = float(cold_profile.heat_transfer_coefficients[cold_index])
        overall_htc = 1.0 / (1.0 / hot_htc + 1.0 / cold_htc)
        area = slice_duty / overall_htc / lmtd
        contributions.append(
            HeatExchangerAreaSlice(
                period=str(period),
                hot_segment_identity=hot_profile.identities[hot_index],
                cold_segment_identity=cold_profile.identities[cold_index],
                duty=slice_duty,
                hot_inlet_temperature=hot_in,
                hot_outlet_temperature=hot_out,
                cold_inlet_temperature=cold_in,
                cold_outlet_temperature=cold_out,
                hot_htc=hot_htc,
                cold_htc=cold_htc,
                overall_htc=overall_htc,
                lmtd=lmtd,
                area=area,
            )
        )
        remaining -= slice_duty
        hot_q += slice_duty
        cold_q -= slice_duty

    if remaining > tolerance:
        raise ValueError("Exchanger duty exceeds an available parent segment profile.")
    return tuple(contributions)


def add_piecewise_temperature_mapping(
    model,
    heat_coordinate,
    temperature,
    profile: PiecewiseThermalProfile,
    *,
    name: str,
    integer_capable: bool,
    initial_segment: int = 0,
):
    """Constrain ``temperature`` to ordered ``T(Q)`` without average CP."""
    q_points = profile.cumulative_duties
    t_points = np.concatenate(([profile.temperatures_in[0]], profile.temperatures_out))
    if len(profile.duties) == 1:
        slope = (t_points[1] - t_points[0]) / (q_points[1] - q_points[0])
        model.Equation(temperature == t_points[0] + slope * heat_coordinate)
        return None

    if integer_capable:
        lambdas = [
            model.Var(
                value=1.0 if index == 0 else 0.0,
                lb=0.0,
                ub=1.0,
                name=f"{name}_lambda_{index}",
            )
            for index in range(len(q_points))
        ]
        intervals = [
            model.Var(
                value=1 if index == initial_segment else 0,
                lb=0,
                ub=1,
                integer=True,
                name=f"{name}_interval_{index}",
            )
            for index in range(len(profile.duties))
        ]
        model.Equation(sum(lambdas) == 1.0)
        model.Equation(sum(intervals) == 1.0)
        model.Equation(
            heat_coordinate
            == sum(lambdas[index] * q_points[index] for index in range(len(q_points)))
        )
        model.Equation(
            temperature
            == sum(lambdas[index] * t_points[index] for index in range(len(t_points)))
        )
        model.Equation(lambdas[0] <= intervals[0])
        model.Equation(lambdas[-1] <= intervals[-1])
        for index in range(1, len(lambdas) - 1):
            model.Equation(lambdas[index] <= intervals[index - 1] + intervals[index])
        return None

    active_segment = min(max(initial_segment, 0), len(profile.duties) - 1)
    selectors = [
        model.Param(
            value=1.0 if index == active_segment else 0.0, name=f"{name}_active_{index}"
        )
        for index in range(len(profile.duties))
    ]
    for index, selector in enumerate(selectors):
        q_start = q_points[index]
        slope = (t_points[index + 1] - t_points[index]) / (
            q_points[index + 1] - q_start
        )
        # The active line is deliberately allowed to extrapolate. After each
        # continuous solve the owning model selects the interval containing Q
        # and resolves; constraining Q to the current interval would prevent
        # that active-set iteration from ever crossing a breakpoint.
        model.Equation(
            selector
            * (temperature - t_points[index] - slope * (heat_coordinate - q_start))
            == 0.0
        )
    return {
        "heat_coordinate": heat_coordinate,
        "profile": profile,
        "selectors": selectors,
        "active_segment": active_segment,
    }


__all__ = [
    "PiecewiseThermalProfile",
    "add_piecewise_temperature_mapping",
    "duty_aligned_area_contributions",
    "profile_from_solver_arrays",
    "utility_thermal_profile",
]
