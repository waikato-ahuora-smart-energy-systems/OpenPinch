"""Per-stream decision data for one pinch side of the heuristic method."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..solver.piecewise import profile_from_solver_arrays

# The decomposition clips a stream that crosses the pinch to its pinch
# temperature exactly; a stream ending within this of it touches the pinch.
PINCH_TOLERANCE = 1e-2


@dataclass(frozen=True)
class SideStream:
    """One active process stream on one pinch side.

    Values are per period. ``heats`` and ``temperatures`` give the stream's
    temperature against the heat counted from its pinch end, so a match that
    takes the slice ``[q, q + Q]`` meets the stream between those two
    temperatures. A stream absent from the side in a period has zero load
    there and no profile.
    """

    kind: Literal["hot", "cold"]
    index: int
    loads: tuple[float, ...]
    pinch_cps: tuple[float, ...]
    at_pinch: tuple[bool, ...]
    heats: tuple[tuple[float, ...], ...]
    temperatures: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]

    @property
    def load(self) -> float:
        """The period-weighted mean load."""

        return _weighted_mean(self.loads, self.weights)

    @property
    def pinch_cp(self) -> float:
        """The period-weighted mean CP at the pinch end, over active periods."""

        active = [n for n, load in enumerate(self.loads) if load > 0.0]
        if not active:
            return 0.0
        return _weighted_mean(
            [self.pinch_cps[n] for n in active], [self.weights[n] for n in active]
        )

    @property
    def touches_pinch(self) -> bool:
        """Whether the stream reaches the pinch in some period."""

        return any(self.at_pinch)

    def active(self, period: int) -> bool:
        return self.loads[period] > 0.0

    def temperature(self, period: int, heat_from_pinch: float) -> float:
        """The stream temperature ``heat_from_pinch`` kW from its pinch end."""

        return float(
            np.interp(heat_from_pinch, self.heats[period], self.temperatures[period])
        )

    def period_view(self, period: int) -> "SideStream":
        """This stream in one period, as a single-period stream."""

        return replace(
            self,
            loads=(self.loads[period],),
            pinch_cps=(self.pinch_cps[period],),
            at_pinch=(self.at_pinch[period],),
            heats=(self.heats[period],),
            temperatures=(self.temperatures[period],),
            weights=(1.0,),
        )


@dataclass(frozen=True)
class PinchSide:
    """The decision data of one set-up pinch side model.

    ``hot`` and ``cold`` hold the side's active streams. ``approach`` is the
    pair minimum approach ``[period][hot][cold]``; ``match_allowed`` flags the
    recovery pairs the side model can place ``[hot][cold]``, and
    ``heater_allowed`` / ``cooler_allowed`` the utilities each stream can use,
    ``[cold][hot utility]`` and ``[hot][cold utility]``.
    """

    location: Literal["above", "below"]
    required: bool
    hot: tuple[SideStream, ...]
    cold: tuple[SideStream, ...]
    approach: tuple[tuple[tuple[float, ...], ...], ...]
    match_allowed: tuple[tuple[bool, ...], ...]
    heater_allowed: tuple[tuple[int, ...], ...]
    cooler_allowed: tuple[tuple[int, ...], ...]
    hot_count: int
    cold_count: int
    weights: tuple[float, ...]

    @property
    def periods(self) -> int:
        return len(self.weights)

    def period_view(self, period: int) -> "PinchSide":
        """The side in one period: its active streams as single-period ones."""

        return replace(
            self,
            hot=tuple(s.period_view(period) for s in self.hot if s.active(period)),
            cold=tuple(s.period_view(period) for s in self.cold if s.active(period)),
            approach=(self.approach[period],),
            weights=(1.0,),
        )


def pinch_side(model) -> PinchSide:
    """Read a set-up (unsolved) ``PinchDecompModel``'s decision data."""

    weights = tuple(float(value) for value in model.period_weights)
    hot = tuple(
        _side_stream(model, "hot", i, weights)
        for i in range(model.I)
        if model.z_i_active[i]
    )
    cold = tuple(
        _side_stream(model, "cold", j, weights)
        for j in range(model.J)
        if model.z_j_active[j]
    )
    return PinchSide(
        location=model.pinch_loc,
        required=bool(model.side_required),
        hot=hot,
        cold=cold,
        approach=tuple(
            tuple(tuple(float(value) for value in row) for row in period)
            for period in model.dT_r_period
        ),
        match_allowed=tuple(
            tuple(bool(model.z_feasible[i][j][0]) for j in range(model.J))
            for i in range(model.I)
        )
        if model.S > 0
        else tuple(tuple(False for _j in range(model.J)) for _i in range(model.I)),
        heater_allowed=tuple(
            tuple(int(flag) for flag in row) for row in model.z_hu_feasible
        ),
        cooler_allowed=tuple(
            tuple(int(flag) for flag in row) for row in model.z_cu_feasible
        ),
        hot_count=int(model.I),
        cold_count=int(model.J),
        weights=weights,
    )


def _side_stream(model, kind: str, index: int, weights) -> SideStream:
    hot = kind == "hot"
    # The pinch end is the outlet of a hot stream above the pinch and of a
    # cold stream below it; otherwise it is the inlet.
    pinch_at_outlet = hot == (model.pinch_loc == "above")
    loads, cps, at_pinch, heats, temperatures = [], [], [], [], []
    for n in range(model.N_periods):
        active = (model.z_i_active_period if hot else model.z_j_active_period)[n][index]
        load = float((model.Qtot_sh_period if hot else model.Qtot_sc_period)[n][index])
        if not active or load <= 0.0:
            loads.append(0.0)
            cps.append(0.0)
            at_pinch.append(False)
            heats.append((0.0,))
            temperatures.append((0.0,))
            continue
        t_in = float((model.T_h_in_period if hot else model.T_c_in_period)[n][index])
        t_out = float((model.T_h_out_period if hot else model.T_c_out_period)[n][index])
        points_q, points_t, segment_cps = _profile_points(
            model, kind, index, n, t_in, t_out, load
        )
        if pinch_at_outlet:
            points_q = [load - q for q in reversed(points_q)]
            points_t = list(reversed(points_t))
            segment_cps = list(reversed(segment_cps))
        contribution = float(
            (model.T_h_cont_period if hot else model.T_c_cont_period)[n][index]
        )
        pinch = float(model.T_pinch_by_period[n]) + (
            contribution if hot else -contribution
        )
        loads.append(load)
        cps.append(float(segment_cps[0]))
        at_pinch.append(abs(points_t[0] - pinch) <= PINCH_TOLERANCE)
        heats.append(tuple(points_q))
        temperatures.append(tuple(points_t))
    return SideStream(
        kind=kind,
        index=index,
        loads=tuple(loads),
        pinch_cps=tuple(cps),
        at_pinch=tuple(at_pinch),
        heats=tuple(heats),
        temperatures=tuple(temperatures),
        weights=weights,
    )


def _profile_points(model, kind, index, period, t_in, t_out, load):
    """Heat and temperature points from the inlet, and each segment's CP."""

    counts = model.solver_arrays.arrays.get(f"{kind}_segment_count")
    if counts is None or int(counts[index]) <= 1:
        return [0.0, load], [t_in, t_out], [load / abs(t_in - t_out)]
    profile = profile_from_solver_arrays(
        model.solver_arrays, side=kind, parent_index=index, period_index=period
    ).clipped(t_in, t_out)
    points_t = [float(profile.temperatures_in[0])] + [
        float(value) for value in profile.temperatures_out
    ]
    return (
        [float(value) for value in profile.cumulative_duties],
        points_t,
        [float(value) for value in profile.heat_capacity_flowrates],
    )


def _weighted_mean(values, weights) -> float:
    total = float(sum(weights))
    if total <= 0.0:
        return float(np.mean(values)) if len(values) else 0.0
    return float(sum(w * v for w, v in zip(weights, values)) / total)


__all__ = ["PINCH_TOLERANCE", "PinchSide", "SideStream", "pinch_side"]
