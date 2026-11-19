"""A site's utility levels, read from its problem for the site balance.

A hot utility (``"Hot Utility.<name>"``) and a cold utility
(``"Cold Utility.<name>"``) of one name are one main when their
temperatures match in every period (the hot side's inlet is the cold
side's outlet and the reverse, within 1 K, as targeting pairs a level's use
and generation), as a ``Both`` utility splits; otherwise each is a level of
its own, named ``"<name> (hot)"`` and ``"<name> (cold)"`` when the names
clash. Prices are the solver arrays' annual prices ($/kW/y), so costs are
$/y; temperatures are degC per period.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ....contracts.synthesis.site import SiteUtilityLevel, SiteUtilityLevels
from ..solver.utility_slices import utility_slices

HOT_PREFIX = "Hot Utility."
COLD_PREFIX = "Cold Utility."
_KELVIN = 273.15
# Temperatures within this (K) are the same main's, as targeting pairs them.
_MAIN_TOLERANCE = 1.0


@dataclass(frozen=True)
class SiteUtilitySystem:
    """The site's levels and the level each utility serves, by identity."""

    levels: SiteUtilityLevels
    hot_utilities: tuple[tuple[str, str], ...]
    cold_utilities: tuple[tuple[str, str], ...]

    def members(self, solver_arrays) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The level index of each hot and each cold utility of
        ``solver_arrays``; a utility slice serves its parent's level."""
        names = [level.name for level in self.levels.levels]
        slices = utility_slices(solver_arrays)
        result = []
        for axis, pairs in (
            ("hot_utilities", self.hot_utilities),
            ("cold_utilities", self.cold_utilities),
        ):
            level_of = dict(pairs)
            parents = slices.get(axis, {})
            result.append(
                tuple(
                    names.index(level_of[parents.get(identity, (identity,))[0]])
                    for identity in solver_arrays.utility_identities[axis]
                )
            )
        return result[0], result[1]


def site_utility_system(
    solver_arrays,
    *,
    capacities: Mapping[str, Sequence[float | None]] | None = None,
    letdown_price: float = 0.0,
) -> SiteUtilitySystem:
    """The site's levels from its (unsliced) solver arrays.

    ``capacities`` maps a hot utility's identity to its maximum heat flow
    per period (``None`` for no cap), which caps what the site buys into
    its level.
    """

    values = solver_arrays.arrays
    weights = tuple(float(w) for w in values["period_weights"])
    periods = len(weights)
    capacities = capacities or {}

    def side(short, axis, prefix, price_key):
        found = {}
        for u, identity in enumerate(solver_arrays.utility_identities[axis]):
            inlet = [
                float(values[f"T_{short}_in_period"][n][u]) - _KELVIN
                for n in range(periods)
            ]
            outlet = [
                float(values[f"T_{short}_out_period"][n][u]) - _KELVIN
                for n in range(periods)
            ]
            found[identity.removeprefix(prefix)] = {
                "identity": identity,
                "inlet": inlet,
                "outlet": outlet,
                "price": tuple(float(values[price_key][n][u]) for n in range(periods)),
            }
        return found

    hot = side("hu", "hot_utilities", HOT_PREFIX, "hu_cost_period")
    cold = side("cu", "cold_utilities", COLD_PREFIX, "cu_cost_period")
    levels: list[SiteUtilityLevel] = []
    hot_members: list[tuple[str, str]] = []
    cold_members: list[tuple[str, str]] = []

    def span(entry):
        return (
            tuple(max(a, b) for a, b in zip(entry["inlet"], entry["outlet"])),
            tuple(min(a, b) for a, b in zip(entry["inlet"], entry["outlet"])),
        )

    def cap(entry):
        caps = capacities.get(entry["identity"])
        return None if caps is None else tuple(caps)

    for name, entry in hot.items():
        partner = cold.get(name)
        high, low = span(entry)
        if partner is not None and _same_main(entry, partner):
            levels.append(
                SiteUtilityLevel(
                    name=name,
                    kind="both",
                    t_high=high,
                    t_low=low,
                    import_price=entry["price"],
                    dump_price=partner["price"],
                    import_capacity=cap(entry),
                )
            )
            hot_members.append((entry["identity"], name))
            cold_members.append((partner["identity"], name))
            continue
        label = f"{name} (hot)" if partner is not None else name
        levels.append(
            SiteUtilityLevel(
                name=label,
                kind="hot",
                t_high=high,
                t_low=low,
                import_price=entry["price"],
                import_capacity=cap(entry),
            )
        )
        hot_members.append((entry["identity"], label))
    paired = {identity for identity, _name in cold_members}
    for name, entry in cold.items():
        if entry["identity"] in paired:
            continue
        label = f"{name} (cold)" if name in hot else name
        high, low = span(entry)
        levels.append(
            SiteUtilityLevel(
                name=label,
                kind="cold",
                t_high=high,
                t_low=low,
                dump_price=entry["price"],
            )
        )
        cold_members.append((entry["identity"], label))
    return SiteUtilitySystem(
        levels=SiteUtilityLevels(
            levels=tuple(levels), period_weights=weights, letdown_price=letdown_price
        ),
        hot_utilities=tuple(hot_members),
        cold_utilities=tuple(cold_members),
    )


def utility_capacities(zone) -> dict[str, tuple[float | None, ...]]:
    """Each hot utility's ``maximum_heat_flow`` per period, by identity."""
    caps = {}
    for key, stream in zone.hot_utilities.items():
        value = getattr(stream, "maximum_heat_flow", None)
        if value is None:
            continue
        per_period = tuple(
            None if not math.isfinite(float(q)) else float(q)
            for q in value.period_values
        )
        if any(q is not None for q in per_period):
            caps[key] = per_period
    return caps


def _same_main(hot: dict, cold: dict) -> bool:
    return all(
        abs(h_in - c_out) < _MAIN_TOLERANCE and abs(h_out - c_in) < _MAIN_TOLERANCE
        for h_in, h_out, c_in, c_out in zip(
            hot["inlet"], hot["outlet"], cold["inlet"], cold["outlet"], strict=True
        )
    )


__all__ = [
    "COLD_PREFIX",
    "HOT_PREFIX",
    "SiteUtilitySystem",
    "site_utility_system",
    "utility_capacities",
]
