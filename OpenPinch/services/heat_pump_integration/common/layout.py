"""Shared optimisation-vector layout helpers for HPR targeting backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = ["HPRoptVectorLayout"]


@dataclass(frozen=True)
class HPRoptVectorLayout:
    """Canonical section layout for Heat Pump targeting optimisation vectors."""

    n_amb: int = 1
    n_cond: int = 0
    n_evap: int = 0
    n_subcool: int = 0
    n_heat: int = 0
    n_cool: int = 0
    n_ihx: int = 0
    n_misc: int = 0
    _size: int = field(init=False, repr=False)
    _amb_slice: slice = field(init=False, repr=False)
    _cond_slice: slice = field(init=False, repr=False)
    _evap_slice: slice = field(init=False, repr=False)
    _subcool_slice: slice = field(init=False, repr=False)
    _heat_slice: slice = field(init=False, repr=False)
    _cool_slice: slice = field(init=False, repr=False)
    _ihx_slice: slice = field(init=False, repr=False)
    _misc_slice: slice = field(init=False, repr=False)

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

        start = 0
        for section_name, count in (
            ("amb", self.n_amb),
            ("cond", self.n_cond),
            ("evap", self.n_evap),
            ("subcool", self.n_subcool),
            ("heat", self.n_heat),
            ("cool", self.n_cool),
            ("ihx", self.n_ihx),
            ("misc", self.n_misc),
        ):
            stop = start + count
            object.__setattr__(self, f"_{section_name}_slice", slice(start, stop))
            start = stop

        object.__setattr__(self, "_size", start)

    @property
    def size(self) -> int:
        return self._size

    @property
    def amb_slice(self) -> slice:
        return self._amb_slice

    @property
    def cond_slice(self) -> slice:
        return self._cond_slice

    @property
    def evap_slice(self) -> slice:
        return self._evap_slice

    @property
    def subcool_slice(self) -> slice:
        return self._subcool_slice

    @property
    def heat_slice(self) -> slice:
        return self._heat_slice

    @property
    def cool_slice(self) -> slice:
        return self._cool_slice

    @property
    def ihx_slice(self) -> slice:
        return self._ihx_slice

    @property
    def misc_slice(self) -> slice:
        return self._misc_slice

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
            raise ValueError(
                f"{name} bounds must have size {size}, got {len(bounds_ls)}."
            )
        return bounds_ls
