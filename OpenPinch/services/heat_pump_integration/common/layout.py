"""Shared optimisation-vector layout helpers for HPR targeting backends."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["HPRoptVectorLayout"]


@dataclass(frozen=True)
class HPRoptVectorLayout:
    """Canonical section layout for heat-pump targeting optimisation vectors."""

    n_amb: int = 1
    n_cond: int = 0
    n_evap: int = 0
    n_subcool: int = 0
    n_heat: int = 0
    n_cool: int = 0
    n_ihx: int = 0
    n_misc: int = 0

    def __post_init__(self) -> None:
        for name in (
            "n_amb",
            "n_cond",
            "n_evap",
            "n_subcool",
            "n_heat",
            "n_cool",
            "n_ihx",
            "n_misc",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")

    @property
    def size(self) -> int:
        return sum(
            (
                self.n_amb,
                self.n_cond,
                self.n_evap,
                self.n_subcool,
                self.n_heat,
                self.n_cool,
                self.n_ihx,
                self.n_misc,
            )
        )

    def _slice_for(self, section: str) -> slice:
        start = 0
        for name, count in self._section_sizes():
            stop = start + count
            if name == section:
                return slice(start, stop)
            start = stop
        raise KeyError(section)

    def _section_sizes(self) -> tuple[tuple[str, int], ...]:
        return (
            ("amb", self.n_amb),
            ("cond", self.n_cond),
            ("evap", self.n_evap),
            ("subcool", self.n_subcool),
            ("heat", self.n_heat),
            ("cool", self.n_cool),
            ("ihx", self.n_ihx),
            ("misc", self.n_misc),
        )

    @property
    def amb_slice(self) -> slice:
        return self._slice_for("amb")

    @property
    def cond_slice(self) -> slice:
        return self._slice_for("cond")

    @property
    def evap_slice(self) -> slice:
        return self._slice_for("evap")

    @property
    def subcool_slice(self) -> slice:
        return self._slice_for("subcool")

    @property
    def heat_slice(self) -> slice:
        return self._slice_for("heat")

    @property
    def cool_slice(self) -> slice:
        return self._slice_for("cool")

    @property
    def ihx_slice(self) -> slice:
        return self._slice_for("ihx")

    @property
    def misc_slice(self) -> slice:
        return self._slice_for("misc")

    def pack(
        self,
        *,
        x_amb: float | Sequence[float] = 0.0,
        x_cond: Sequence[float] = (),
        x_evap: Sequence[float] = (),
        x_subcool: Sequence[float] = (),
        x_heat: Sequence[float] = (),
        x_cool: Sequence[float] = (),
        x_ihx: Sequence[float] = (),
        x_misc: Sequence[float] = (),
    ) -> np.ndarray:
        """Pack named optimisation blocks into one canonical vector."""
        blocks = [
            self._coerce_block("x_amb", x_amb, self.n_amb),
            self._coerce_block("x_cond", x_cond, self.n_cond),
            self._coerce_block("x_evap", x_evap, self.n_evap),
            self._coerce_block("x_subcool", x_subcool, self.n_subcool),
            self._coerce_block("x_heat", x_heat, self.n_heat),
            self._coerce_block("x_cool", x_cool, self.n_cool),
            self._coerce_block("x_ihx", x_ihx, self.n_ihx),
            self._coerce_block("x_misc", x_misc, self.n_misc),
        ]
        if not blocks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(blocks).astype(np.float64, copy=False)

    def unpack(self, x: np.ndarray | Sequence[float]) -> dict[str, Any]:
        """Split a flat optimisation vector into its named sections."""
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != self.size:
            raise ValueError(
                f"Expected optimisation vector of size {self.size}, got {vec.size}."
            )

        amb = vec[self.amb_slice]
        return {
            "x_amb": float(amb[0]) if amb.size else 0.0,
            "x_cond": vec[self.cond_slice],
            "x_evap": vec[self.evap_slice],
            "x_subcool": vec[self.subcool_slice],
            "x_heat": vec[self.heat_slice],
            "x_cool": vec[self.cool_slice],
            "x_ihx": vec[self.ihx_slice],
            "x_misc": vec[self.misc_slice],
        }

    def build_bounds(
        self,
        *,
        x_amb: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_cond: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_evap: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_subcool: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_heat: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_cool: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_ihx: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
        x_misc: Sequence[float] | Sequence[Sequence[float]] = (0.0, 1.0),
    ) -> list[tuple[float, float]]:
        """Expand per-section bounds into a flat bounds list."""
        bounds = []
        bounds.extend(self._coerce_bounds("x_amb", x_amb, self.n_amb))
        bounds.extend(self._coerce_bounds("x_cond", x_cond, self.n_cond))
        bounds.extend(self._coerce_bounds("x_evap", x_evap, self.n_evap))
        bounds.extend(self._coerce_bounds("x_subcool", x_subcool, self.n_subcool))
        bounds.extend(self._coerce_bounds("x_heat", x_heat, self.n_heat))
        bounds.extend(self._coerce_bounds("x_cool", x_cool, self.n_cool))
        bounds.extend(self._coerce_bounds("x_ihx", x_ihx, self.n_ihx))
        bounds.extend(self._coerce_bounds("x_misc", x_misc, self.n_misc))
        return bounds

    @staticmethod
    def _coerce_block(
        name: str,
        values: float | Sequence[float],
        size: int,
    ) -> np.ndarray:
        if size == 0:
            return np.empty(0, dtype=np.float64)
        if np.isscalar(values):
            vec = np.array([values], dtype=np.float64)
        else:
            vec = np.asarray(values, dtype=np.float64).reshape(-1)
        if vec.size != size:
            raise ValueError(f"{name} must have size {size}, got {vec.size}.")
        return vec

    @staticmethod
    def _coerce_bounds(
        name: str,
        bounds: Sequence[float] | Sequence[Sequence[float]],
        size: int,
    ) -> list[tuple[float, float]]:
        if size == 0:
            return []

        if len(bounds) == 2 and np.isscalar(bounds[0]) and np.isscalar(bounds[1]):
            lo, hi = bounds
            return [(float(lo), float(hi))] * size

        bounds_ls = [tuple(float(v) for v in bound) for bound in bounds]
        if len(bounds_ls) != size:
            raise ValueError(f"{name} bounds must have size {size}, got {len(bounds_ls)}.")
        return bounds_ls
