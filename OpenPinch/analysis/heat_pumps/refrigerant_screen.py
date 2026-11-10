"""Single-stage refrigerant screening between evaporating and condensing levels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import CoolProp.CoolProp as _coolprop

from ...contracts.hpr_refrigerant_screen import RefrigerantScreenVerdict
from ...domain.fluids import validate_coolprop_fluid_name

NATURAL_REFRIGERANTS = (
    "Water",
    "Ammonia",
    "CarbonDioxide",
    "n-Propane",
    "n-Butane",
    "IsoButane",
    "n-Pentane",
    "Isopentane",
    "Neopentane",
)

_KELVIN = 273.15


def resolve_refrigerants(selection: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return the candidate fluids for a group name or an explicit list.

    ``None`` and ``"all"`` select every fluid in the installed CoolProp;
    ``"natural"`` selects :data:`NATURAL_REFRIGERANTS`. A list is validated
    name by name and kept in order; mixtures are rejected.
    """
    if selection is None or selection == "all":
        return tuple(_coolprop.get_global_param_string("FluidsList").split(","))
    if selection == "natural":
        return NATURAL_REFRIGERANTS
    if isinstance(selection, str):
        raise ValueError(
            f"Unknown refrigerant group {selection!r}; use 'all', 'natural' or a "
            "list of CoolProp fluid names."
        )
    fluids = tuple(selection)
    if not fluids:
        raise ValueError("Refrigerant screening needs at least one fluid.")
    for fluid in fluids:
        validate_coolprop_fluid_name(fluid)
        if "[" in fluid:
            raise ValueError(
                f"Refrigerant screening takes pure fluids only, not {fluid!r}."
            )
    return fluids


@dataclass(frozen=True)
class RefrigerantScreenThresholds:
    """Limits a fluid must meet to be practical; pressures in Pa, margins in K."""

    dt_cont: float = 0.0
    max_pressure_ratio: float = 15.0
    min_suction_pressure: float = 0.5e5
    critical_margin: float = 15.0


@dataclass(frozen=True)
class RefrigerantScreenEntry:
    """One fluid between one evaporating and one condensing level.

    Temperatures are in degC, pressures in Pa and the suction volumetric
    heating capacity in kJ/m3. Saturation properties are ``None`` when the
    fluid has no saturation state between the levels.
    """

    fluid: str
    T_evap: float
    T_cond: float
    T_crit: float
    verdict: RefrigerantScreenVerdict
    p_evap: float | None = None
    p_cond: float | None = None
    pressure_ratio: float | None = None
    suction_vhc: float | None = None
    rank: int | None = None


def screen_refrigerant(
    fluid: str,
    T_evap: float,
    T_cond: float,
    thresholds: RefrigerantScreenThresholds,
) -> RefrigerantScreenEntry:
    """Screen one fluid as a single-stage cycle between two saturation levels."""
    T_crit = _coolprop.PropsSI("Tcrit", fluid) - _KELVIN
    T_min = _coolprop.PropsSI("Tmin", fluid) - _KELVIN
    row = {"fluid": fluid, "T_evap": T_evap, "T_cond": T_cond, "T_crit": T_crit}
    if T_cond > T_crit - thresholds.dt_cont:
        return RefrigerantScreenEntry(
            **row, verdict=RefrigerantScreenVerdict.ABOVE_CRITICAL
        )
    if T_evap < T_min:
        return RefrigerantScreenEntry(
            **row, verdict=RefrigerantScreenVerdict.BELOW_TRIPLE_POINT
        )
    k_evap, k_cond = T_evap + _KELVIN, T_cond + _KELVIN
    p_evap = _coolprop.PropsSI("P", "T", k_evap, "Q", 1, fluid)
    p_cond = _coolprop.PropsSI("P", "T", k_cond, "Q", 0, fluid)
    latent = _coolprop.PropsSI("H", "T", k_evap, "Q", 1, fluid) - _coolprop.PropsSI(
        "H", "T", k_evap, "Q", 0, fluid
    )
    suction_vhc = latent * _coolprop.PropsSI("D", "T", k_evap, "Q", 1, fluid) / 1e3
    ratio = p_cond / p_evap
    if ratio > thresholds.max_pressure_ratio:
        verdict = RefrigerantScreenVerdict.OVER_RATIO_LIMIT
    elif p_evap < thresholds.min_suction_pressure:
        verdict = RefrigerantScreenVerdict.DEEP_VACUUM_SUCTION
    elif T_crit - T_cond < thresholds.critical_margin:
        verdict = RefrigerantScreenVerdict.NEAR_CRITICAL
    else:
        verdict = RefrigerantScreenVerdict.PRACTICAL
    return RefrigerantScreenEntry(
        **row,
        verdict=verdict,
        p_evap=p_evap,
        p_cond=p_cond,
        pressure_ratio=ratio,
        suction_vhc=suction_vhc,
    )


def screen_refrigerants(
    T_evap_levels: Sequence[float],
    T_cond_levels: Sequence[float],
    fluids: Sequence[str],
    thresholds: RefrigerantScreenThresholds,
) -> tuple[RefrigerantScreenEntry, ...]:
    """Screen every fluid between every evaporating and condensing level.

    Pairs whose condensing level is not above the evaporating level are
    skipped; ``ValueError`` is raised when no pair remains. Each level pair
    lists its practical fluids first, ranked by suction volumetric heating
    capacity, then the others in candidate order.
    """
    entries: list[RefrigerantScreenEntry] = []
    for T_evap in T_evap_levels:
        for T_cond in T_cond_levels:
            if T_cond <= T_evap:
                continue
            pair = [
                screen_refrigerant(fluid, T_evap, T_cond, thresholds)
                for fluid in fluids
            ]
            practical = sorted(
                (e for e in pair if e.verdict is RefrigerantScreenVerdict.PRACTICAL),
                key=lambda e: -e.suction_vhc,
            )
            entries.extend(
                replace(entry, rank=rank) for rank, entry in enumerate(practical, 1)
            )
            entries.extend(
                e for e in pair if e.verdict is not RefrigerantScreenVerdict.PRACTICAL
            )
    if not entries:
        raise ValueError(
            "Refrigerant screening found no evaporating level below a condensing level."
        )
    return tuple(entries)
