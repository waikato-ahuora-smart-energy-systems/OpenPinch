"""Smooth piecewise heat-temperature profiles for the fixed-structure model.

A profile describes one stream (or segmented utility) as ordered segments with
their own heat capacity flowrate, film coefficient, temperature contribution
and price. ``heat(T)`` is the heat released (hot side) or absorbed (cold side)
moving from the supply temperature to ``T``; it is piecewise linear in ``T``.

For the NLP the kinks are rounded with ``smooth_max(x) = (x + sqrt(x**2 +
delta**2)) / 2`` so IPOPT sees a smooth function; a single-segment profile is
exactly linear. Reporting uses the exact piecewise functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..solver.piecewise import PiecewiseThermalProfile

SMOOTHING_K = 0.05
"""Width (K) of the rounding at segment boundaries in solver expressions."""


def smooth_max0(x, delta: float = SMOOTHING_K):
    """Smooth ``max(x, 0)``; exact for plain numbers."""

    if isinstance(x, int | float | np.floating):
        return max(float(x), 0.0)
    return 0.5 * (x + (x * x + delta * delta) ** 0.5)


def smooth_clip(value, lower, upper, delta: float = SMOOTHING_K):
    """Smooth ``min(max(value, lower), upper)`` for ``lower <= upper``."""

    if all(isinstance(v, int | float | np.floating) for v in (value, lower, upper)):
        return min(max(float(value), float(lower)), float(upper))
    above_lower = lower + smooth_max0(value - lower, delta)
    return upper - smooth_max0(upper - above_lower, delta)


@dataclass(frozen=True)
class ThermalProfile:
    """Ordered segments of one hot or cold stream for one period."""

    side: str  # "hot": temperature falls as heat is released; "cold": rises
    identities: tuple[str, ...]
    t_in: tuple[float, ...]
    t_out: tuple[float, ...]
    cp: tuple[float, ...]
    htc: tuple[float, ...]
    dt_cont: tuple[float, ...]
    price: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.identities)
        if count == 0 or any(
            len(values) != count
            for values in (
                self.t_in,
                self.t_out,
                self.cp,
                self.htc,
                self.dt_cont,
                self.price,
            )
        ):
            raise ValueError("thermal profile arrays must share one length.")
        if self.side not in {"hot", "cold"}:
            raise ValueError("thermal profile side must be 'hot' or 'cold'.")

    # -- construction -------------------------------------------------------

    @classmethod
    def single(
        cls,
        side: str,
        identity: str,
        supply: float,
        target: float,
        cp: float,
        htc: float,
        dt_cont: float = 0.0,
        price: float = 0.0,
    ) -> "ThermalProfile":
        return cls(
            side=side,
            identities=(identity,),
            t_in=(float(supply),),
            t_out=(float(target),),
            cp=(float(cp),),
            htc=(float(htc),),
            dt_cont=(float(dt_cont),),
            price=(float(price),),
        )

    @classmethod
    def from_piecewise(cls, side: str, profile: PiecewiseThermalProfile):
        return cls(
            side=side,
            identities=tuple(str(i) for i in profile.identities),
            t_in=tuple(float(v) for v in profile.temperatures_in),
            t_out=tuple(float(v) for v in profile.temperatures_out),
            cp=tuple(float(v) for v in profile.heat_capacity_flowrates),
            htc=tuple(float(v) for v in profile.heat_transfer_coefficients),
            dt_cont=tuple(float(v) for v in profile.temperature_contributions),
            price=tuple(float(v) for v in profile.prices),
        )

    # -- properties ----------------------------------------------------------

    @property
    def supply(self) -> float:
        return self.t_in[0]

    @property
    def target(self) -> float:
        return self.t_out[-1]

    @property
    def segmented(self) -> bool:
        return len(self.identities) > 1

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return self.t_in[1:]

    @property
    def duties(self) -> tuple[float, ...]:
        return tuple(c * abs(a - b) for c, a, b in zip(self.cp, self.t_in, self.t_out))

    @property
    def total(self) -> float:
        return float(sum(self.duties))

    @property
    def lowest(self) -> float:
        return min(self.supply, self.target)

    @property
    def highest(self) -> float:
        return max(self.supply, self.target)

    @property
    def max_contribution(self) -> float:
        return max(self.dt_cont)

    @property
    def mean_htc(self) -> float:
        """Duty-weighted film coefficient used inside the area objective."""

        total = self.total
        if total <= 0.0:
            return float(self.htc[0])
        return float(sum(d * h for d, h in zip(self.duties, self.htc)) / total)

    # -- heat and cost -------------------------------------------------------

    def _distance(self, reference, temperature):
        """Temperature distance travelled from ``reference`` towards target."""

        if self.side == "hot":
            return reference - temperature
        return temperature - reference

    def heat(self, temperature, *, smooth: bool = True):
        """Heat moved from supply to ``temperature`` (piecewise linear)."""

        return self._weighted(self.cp, temperature, smooth=smooth)

    def cost(self, temperature, *, smooth: bool = True):
        """Utility cost from supply to ``temperature`` (price times heat)."""

        rates = tuple(p * c for p, c in zip(self.price, self.cp))
        return self._weighted(rates, temperature, smooth=smooth)

    def _weighted(self, rates: Sequence[float], temperature, *, smooth: bool):
        if not smooth and isinstance(temperature, int | float | np.floating):
            return self._exact(rates, float(temperature))
        expression = rates[0] * self._distance(self.supply, temperature)
        for k in range(1, len(rates)):
            step = rates[k] - rates[k - 1]
            if step == 0.0:
                continue
            expression = expression + step * smooth_max0(
                self._distance(self.t_in[k], temperature)
            )
        return expression

    def _exact(self, rates: Sequence[float], temperature: float) -> float:
        value = rates[0] * self._distance(self.supply, temperature)
        for k in range(1, len(rates)):
            value += (rates[k] - rates[k - 1]) * max(
                self._distance(self.t_in[k], temperature), 0.0
            )
        return float(value)

    def temperature_at(self, heat: float) -> float:
        """Exact inverse of ``heat`` with linear extrapolation at both ends.

        Segments without flow (a stream that is off in a period) carry no
        heat, so they are skipped; a profile without any flow stays at its
        supply temperature.
        """

        heat = float(heat)
        sign = -1.0 if self.side == "hot" else 1.0
        flowing = [k for k, cp in enumerate(self.cp) if cp > 0.0]
        if not flowing:
            return self.supply
        if heat <= 0.0:
            first = flowing[0]
            return self.t_in[first] + sign * heat / self.cp[first]
        cumulative = 0.0
        for k in flowing:
            duty = self.duties[k]
            if heat <= cumulative + duty or k == flowing[-1]:
                return self.t_in[k] + sign * (heat - cumulative) / self.cp[k]
            cumulative += duty
        return self.target  # pragma: no cover - loop always returns

    def contribution_at(self, temperature: float) -> float:
        """Temperature contribution of the segment holding ``temperature``."""

        for k in range(len(self.identities)):
            low, high = sorted((self.t_in[k], self.t_out[k]))
            if low - 1e-9 <= temperature <= high + 1e-9:
                return self.dt_cont[k]
        return (
            self.dt_cont[0]
            if self._distance(self.supply, temperature) < 0
            else (self.dt_cont[-1])
        )

    def piecewise(self, scale: float = 1.0) -> PiecewiseThermalProfile:
        """Exact profile for area slicing, scaled to a branch flow fraction."""

        return PiecewiseThermalProfile(
            identities=self.identities,
            temperatures_in=np.asarray(self.t_in, dtype=float),
            temperatures_out=np.asarray(self.t_out, dtype=float),
            duties=np.asarray(self.duties, dtype=float) * scale,
            heat_capacity_flowrates=np.asarray(self.cp, dtype=float) * scale,
            heat_transfer_coefficients=np.asarray(self.htc, dtype=float),
            prices=np.asarray(self.price, dtype=float),
            temperature_contributions=np.asarray(self.dt_cont, dtype=float),
        )


__all__ = ["SMOOTHING_K", "ThermalProfile", "smooth_clip", "smooth_max0"]
