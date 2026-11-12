"""Practical operating limits for a vapour-compression stage."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CycleLimits"]


@dataclass(frozen=True)
class CycleLimits:
    """Limits a vapour-compression stage is held to; ``None`` means no limit.

    ``max_discharge_superheat`` (K above the dew point at discharge pressure)
    is met by liquid injection, not penalised. The others are penalised by
    the stage's condenser duty times the relative violation: pressures in Pa,
    ``max_discharge_pressure_co2`` replacing ``max_discharge_pressure`` for
    carbon dioxide, and ``min_lift`` in K from evaporating to condensing.
    """

    max_discharge_superheat: float | None = None
    max_pressure_ratio: float | None = None
    max_discharge_pressure: float | None = None
    max_discharge_pressure_co2: float | None = None
    min_suction_pressure: float | None = None
    min_lift: float = 0.0
