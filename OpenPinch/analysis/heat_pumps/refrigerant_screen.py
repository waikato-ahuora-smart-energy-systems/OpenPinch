"""Refrigerant screening of every HPR layout between saturation levels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations

import CoolProp.CoolProp as _coolprop

from ...contracts.hpr import HPRTopologyIdentifier
from ...contracts.hpr_refrigerant_screen import (
    RefrigerantScreenVerdict,
    count_ordered_fluid_sets,
    ordered_fluid_sets,
)
from ...domain.configuration import Configuration
from ...domain.fluids import validate_coolprop_fluid_name
from .targeting.vapour_compression_mvr import MAX_MVR_STAGE_LIFT

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

# An MVR is an open cycle on the process or utility steam, so water is its
# only working fluid.
MVR_FLUID = "Water"
_FEASIBLE = frozenset(
    {RefrigerantScreenVerdict.PRACTICAL, RefrigerantScreenVerdict.NEAR_CRITICAL}
)


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
    """Limits a fluid must meet to be practical; pressures in Pa, margins in K.

    ``dt_cascade_hx`` is not a limit but sets where a cascade or MVR stage
    evaporates relative to the stage below it. A stage lifting more than
    ``max_stage_lift`` is also screened split into a cascade.
    """

    dt_cont: float = 1.0
    max_pressure_ratio: float = 15.0
    min_suction_pressure: float = 0.5e5
    critical_margin: float = 5.0
    dt_cascade_hx: float = 5.0
    max_stage_lift: float = 70.0

    @classmethod
    def from_configuration(cls, config: Configuration) -> "RefrigerantScreenThresholds":
        """Read the limits from ``HPR_*`` configuration (suction floor in bar)."""
        return cls(
            dt_cont=config.hpr.dt_cont,
            max_pressure_ratio=config.hpr.max_pressure_ratio,
            min_suction_pressure=config.hpr.screen_min_suction_pressure * 1e5,
            critical_margin=config.hpr.screen_critical_margin,
            dt_cascade_hx=config.hpr.dt_cascade_hx,
            max_stage_lift=config.hpr.screen_max_stage_lift,
        )


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


def is_feasible(verdict: RefrigerantScreenVerdict) -> bool:
    """Return whether a fluid with ``verdict`` can run the stage.

    Above critical, below the triple point, over the ratio limit and deep
    vacuum suction are hard limits; near critical is feasible.
    """
    return verdict in _FEASIBLE


@dataclass(frozen=True)
class ScreenStage:
    """One compression stage of a layout, with every candidate screened.

    ``kind`` is ``"vapour_compression"`` or ``"mvr"``. The feasible fluids
    come first, ranked from 1 by descending suction volumetric heating
    capacity, then the others in candidate order.
    """

    kind: str
    T_evap: float
    T_cond: float
    entries: tuple[RefrigerantScreenEntry, ...]

    @property
    def feasible(self) -> tuple[str, ...]:
        """The feasible fluids, best ranked first."""
        return tuple(entry.fluid for entry in self.entries if entry.rank is not None)


@dataclass(frozen=True)
class ScreenLayout:
    """One targeting method on one choice of Carnot levels, hottest first.

    ``added_levels`` are the levels a split inserted between Carnot levels.
    """

    method: HPRTopologyIdentifier
    T_cond_levels: tuple[float, ...]
    T_evap_levels: tuple[float, ...]
    stages: tuple[ScreenStage, ...]
    added_levels: tuple[float, ...] = ()

    def _columns(self) -> list[list[tuple[str, float]]]:
        return [
            [(e.fluid, e.T_crit) for e in stage.entries if e.rank is not None]
            for stage in self.stages
            if stage.kind == "vapour_compression"
        ]

    def _mvr_runs(self) -> bool:
        return all(stage.feasible for stage in self.stages if stage.kind == "mvr")

    @property
    def n_fluid_sets(self) -> int:
        """How many fluid sets can run every stage, Tcrit never rising down them."""
        return count_ordered_fluid_sets(self._columns()) if self._mvr_runs() else 0

    def fluid_sets(self) -> tuple[tuple[str, ...], ...]:
        """Every vapour compression fluid set, one fluid per stage, hottest first.

        A set's critical temperature never rises down the stages. MVR stages
        always run on water and are left out of the sets.
        """
        return ordered_fluid_sets(self._columns()) if self._mvr_runs() else ()


def _hottest_first(levels: Sequence[float]) -> tuple[float, ...]:
    return tuple(sorted(dict.fromkeys(float(t) for t in levels), reverse=True))


def _parallel_pairs(T_cond, T_evap) -> list[tuple[float, float]]:
    """Machine ``i`` pairs the ``i``-th hottest levels; the shorter list repeats."""
    n, m = len(T_cond), len(T_evap)
    return [(T_evap[min(i, m - 1)], T_cond[min(i, n - 1)]) for i in range(max(n, m))]


def _cascade_pairs(T_cond, T_evap, dt_hx: float) -> list[tuple[float, float]]:
    """Stage ``i`` lifts from level ``i + 1`` to level ``i``, hottest first.

    A condenser stage above another condenser stage evaporates ``dt_hx``
    below it; an evaporator stage condenses ``dt_hx`` above the level it
    feeds. This is how cascade targeting sets its stage temperatures.
    """
    levels, n = (*T_cond, *T_evap), len(T_cond)
    pairs = []
    for i in range(len(levels) - 1):
        if i < n:
            gap = dt_hx if i + 1 < n else 0.0
            pairs.append((levels[i + 1] - gap, levels[i]))
        else:
            pairs.append((levels[i + 1], levels[i] + dt_hx))
    return pairs


def _intermediate_condensing(
    T_evap: float, T_cond: float, dt_hx: float, max_lift: float
) -> list[float]:
    """Condensing levels, coldest first, that split a stage into equal lifts.

    ``k`` stages of lift ``L`` with a ``dt_hx`` gap between them span
    ``k * L - (k - 1) * dt_hx``; ``k`` is the fewest with ``L <= max_lift``.
    """
    span = T_cond - T_evap
    if span <= max_lift:
        return []
    k = math.ceil((span - dt_hx) / (max_lift - dt_hx) - 1e-9)
    lift = (span + (k - 1) * dt_hx) / k
    return [T_evap + lift + j * (lift - dt_hx) for j in range(k - 1)]


def _split_levels(conds, evaps, dt_hx: float, max_lift: float, is_heat_pumping):
    """Insert levels into a cascade so no stage lifts more than ``max_lift``.

    A stage between condensing levels gains condensing levels and one between
    evaporating levels gains evaporating levels; the stage joining the two
    sides gains condensing levels in a heat pump and evaporating levels in
    refrigeration. Returns the new condensing and evaporating levels and the
    inserted ones.
    """
    levels, n = (*conds, *evaps), len(conds)
    pairs = _cascade_pairs(conds, evaps, dt_hx)
    cond_out, evap_out, added = [], [], []
    for i, level in enumerate(levels):
        (cond_out if i < n else evap_out).append(level)
        if i == len(levels) - 1:
            break
        T_e, T_c = pairs[i]
        inserted = _intermediate_condensing(T_e, T_c, dt_hx, max_lift)[::-1]
        if i < n - 1 or (i == n - 1 and is_heat_pumping):
            cond_out.extend(inserted)
        else:
            inserted = [T - dt_hx for T in inserted]
            evap_out.extend(inserted)
        added.extend(inserted)
    return tuple(cond_out), tuple(evap_out), tuple(added)


def _mvr_pairs(served, T_start: float) -> list[tuple[float, float]]:
    """Equal MVR lifts of at most ``MAX_MVR_STAGE_LIFT`` up to each served level."""
    pairs, low = [], T_start
    for high in reversed(served):
        n = math.ceil((high - low) / MAX_MVR_STAGE_LIFT - 1e-9)
        bounds = [low + (high - low) * j / n for j in range(n)] + [high]
        pairs.extend(zip(bounds, bounds[1:]))
        low = high
    return pairs[::-1]


def screen_layouts(
    T_evap_levels: Sequence[float],
    T_cond_levels: Sequence[float],
    fluids: Sequence[str],
    thresholds: RefrigerantScreenThresholds,
    *,
    topology: str,
    is_heat_pumping: bool = True,
) -> tuple[ScreenLayout, ...]:
    """Screen every layout that a ``topology`` Carnot target's levels allow.

    Every choice of ``n`` condensing and ``m`` evaporating levels, from one
    of each up to all of them, gives a layout: a single stage for one of
    each, otherwise parallel or cascade vapour compression. A cascade heat
    pump also gives every vapour compression plus MVR layout, with the MVR
    serving one or more of the hottest chosen levels on water and parallel
    vapour compression below it. Last come cascades that split each single
    stage or cascade with a stage lifting more than ``max_stage_lift``.
    Layouts with a stage that has no lift are dropped; ``ValueError`` is
    raised when none is left.
    """
    if topology not in ("parallel", "cascade"):
        raise ValueError(
            f"Refrigerant screening topology must be 'parallel' or 'cascade', not "
            f"{topology!r}."
        )
    if thresholds.max_stage_lift <= thresholds.dt_cascade_hx:
        raise ValueError(
            "Refrigerant screening needs a stage lift limit above the cascade "
            "exchanger approach."
        )
    T_cond, T_evap = _hottest_first(T_cond_levels), _hottest_first(T_evap_levels)
    stages: dict[tuple, ScreenStage] = {}

    def stage(kind: str, T_e: float, T_c: float) -> ScreenStage:
        key = (kind, T_e, T_c)
        if key not in stages:
            candidates = (MVR_FLUID,) if kind == "mvr" else fluids
            entries = [
                screen_refrigerant(fluid, T_e, T_c, thresholds) for fluid in candidates
            ]
            feasible = sorted(
                (e for e in entries if is_feasible(e.verdict)),
                key=lambda e: -e.suction_vhc,
            )
            stages[key] = ScreenStage(
                kind,
                T_e,
                T_c,
                (
                    *(replace(e, rank=rank) for rank, e in enumerate(feasible, 1)),
                    *(e for e in entries if not is_feasible(e.verdict)),
                ),
            )
        return stages[key]

    def layout(
        method, conds, evaps, mvr_pairs, vc_pairs, added=()
    ) -> ScreenLayout | None:
        if any(T_c <= T_e for T_e, T_c in (*mvr_pairs, *vc_pairs)):
            return None
        return ScreenLayout(
            method,
            conds,
            evaps,
            (
                *(stage("mvr", *pair) for pair in mvr_pairs),
                *(stage("vapour_compression", *pair) for pair in vc_pairs),
            ),
            added,
        )

    def choices(min_n: int = 1):
        for n in range(min_n, len(T_cond) + 1):
            for m in range(1, len(T_evap) + 1):
                for conds in combinations(T_cond, n):
                    for evaps in combinations(T_evap, m):
                        yield conds, evaps

    dt_hx = thresholds.dt_cascade_hx
    layouts, splits = [], []
    for conds, evaps in choices():
        if len(conds) == len(evaps) == 1:
            method = HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
            pairs = _parallel_pairs(conds, evaps)
        elif topology == "parallel":
            method = HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION
            pairs = _parallel_pairs(conds, evaps)
        else:
            method = HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION
            pairs = _cascade_pairs(conds, evaps, dt_hx)
        layouts.append(layout(method, conds, evaps, (), pairs))
        # A single stage or cascade with an over-lift stage is also a cascade
        # with that stage split; parallel machines have no such method.
        if (
            layouts[-1] is None
            or method is HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION
        ):
            continue
        split_conds, split_evaps, added = _split_levels(
            conds, evaps, dt_hx, thresholds.max_stage_lift, is_heat_pumping
        )
        if added:
            splits.append(
                layout(
                    HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
                    split_conds,
                    split_evaps,
                    (),
                    _cascade_pairs(split_conds, split_evaps, dt_hx),
                    added,
                )
            )
    if topology == "cascade" and is_heat_pumping:
        for conds, evaps in choices(min_n=2):
            for k in range(1, len(conds)):
                layouts.append(
                    layout(
                        HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
                        conds,
                        evaps,
                        _mvr_pairs(conds[:k], conds[k] - thresholds.dt_cascade_hx),
                        _parallel_pairs(conds[k:], evaps),
                    )
                )
    layouts = [item for item in (*layouts, *splits) if item is not None]
    if not layouts:
        raise ValueError(
            "Refrigerant screening found no evaporating level below a condensing level."
        )
    return tuple(layouts)
