"""Site cost of a site network: capital plus the site utility balance.

Heaters and coolers are grouped by process (the stream-key prefix) and by
level (the utility's name without its ``"Hot Utility."`` or
``"Cold Utility."`` prefix); the site utility accounting
(``utility_system.py``) then balances the levels, so heat a process raises
on a main another process uses is credited rather than charged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

import numpy as np

from ....contracts.synthesis.site import (
    LevelLoad,
    SiteUtilityLevel,
    SiteUtilityLevels,
)
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from .utility_system import account_utility_system

HOT_PREFIX = "Hot Utility."
COLD_PREFIX = "Cold Utility."
_KELVIN = 273.15


def site_utility_levels(solver_arrays, *, letdown_price: float = 0.0):
    """The site's levels from a site problem's solver arrays.

    A utility that is both a hot and a cold utility is a main (``"both"``),
    bought at its hot utility's price and dumped at its cold utility's.
    Prices are the arrays' annual prices ($/kW/y), so costs are $/y;
    temperatures (degC) span the utility's inlet and outlet in period 0.
    """

    values = solver_arrays.arrays
    weights = tuple(float(w) for w in values["period_weights"])
    periods = len(weights)
    found: dict[str, dict] = {}
    for side, prefix, axis, key in (
        ("hu", HOT_PREFIX, "hot_utilities", "hu_cost_period"),
        ("cu", COLD_PREFIX, "cold_utilities", "cu_cost_period"),
    ):
        for u, identity in enumerate(solver_arrays.utility_identities[axis]):
            name = identity.removeprefix(prefix)
            ends = (
                float(values[f"T_{side}_in_period"][0][u]) - _KELVIN,
                float(values[f"T_{side}_out_period"][0][u]) - _KELVIN,
            )
            entry = found.setdefault(name, {"ends": ends})
            prices = tuple(float(values[key][n][u]) for n in range(periods))
            entry["import_price" if side == "hu" else "dump_price"] = prices
    levels = []
    for name, entry in found.items():
        ends = entry.pop("ends")
        imports, dumps = "import_price" in entry, "dump_price" in entry
        kind = "both" if imports and dumps else ("hot" if imports else "cold")
        levels.append(
            SiteUtilityLevel(
                name=name, kind=kind, t_high=max(ends), t_low=min(ends), **entry
            )
        )
    return SiteUtilityLevels(
        levels=tuple(levels), period_weights=weights, letdown_price=letdown_price
    )


def site_loads(
    network: HeatExchangerNetwork, processes: Iterable[str]
) -> dict[str, dict[str, LevelLoad]]:
    """Each process's heater (use) and cooler (generation) duty per level.

    A stream belongs to the longest listed process its key starts with
    (``"<process>.<stream>"``), else to the process ``""``.
    """

    processes = tuple(processes)
    periods = len(network.exchangers[0].period_states) if network.exchangers else 1
    totals: dict[str, dict[str, list[list[float]]]] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            utility, stream, side = exchanger.source_stream, exchanger.sink_stream, 0
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            utility, stream, side = exchanger.sink_stream, exchanger.source_stream, 1
        else:
            continue
        level = utility.removeprefix(HOT_PREFIX).removeprefix(COLD_PREFIX)
        by_level = totals.setdefault(_process_of(stream, processes), {})
        duties = by_level.setdefault(level, [[0.0] * periods, [0.0] * periods])
        for state in exchanger.period_states:
            if state.active:
                duties[side][state.period_idx] += float(state.duty)
    return {
        process: {
            level: LevelLoad(use=tuple(use), generation=tuple(generation))
            for level, (use, generation) in by_level.items()
        }
        for process, by_level in totals.items()
    }


def loads_from_level_duties(
    processes: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, LevelLoad]]:
    """Loads from each process's duty per period by utility name
    (``source_metadata["site_processes"]``)."""

    loads: dict[str, dict[str, LevelLoad]] = {}
    for process, duties in processes.items():
        by_level: dict[str, list[tuple[float, ...]]] = {}
        for utility, values in duties.items():
            periods = len(values)
            level = utility.removeprefix(HOT_PREFIX).removeprefix(COLD_PREFIX)
            pair = by_level.setdefault(level, [(0.0,) * periods, (0.0,) * periods])
            side = 1 if utility.startswith(COLD_PREFIX) else 0
            pair[side] = tuple(float(q) for q in values)
        loads[process] = {
            level: LevelLoad(use=use, generation=generation)
            for level, (use, generation) in by_level.items()
        }
    return loads


def capital_cost(network: HeatExchangerNetwork) -> float:
    """A network's annualised capital: its own, or TAC less utility cost."""
    if network.capital_cost is not None:
        return float(network.capital_cost)
    if network.total_annual_cost is not None and network.utility_cost is not None:
        return float(network.total_annual_cost) - float(network.utility_cost)
    return 0.0


def site_metrics(capital: float, balance) -> dict[str, float]:
    """The site cost summary metrics of a network."""
    return {
        "site_capital_cost": float(capital),
        "site_operating_cost": float(balance.operating_cost),
        "site_total_annual_cost": float(capital) + float(balance.operating_cost),
    }


def priced_site_network(
    network: HeatExchangerNetwork,
    processes: Iterable[str],
    levels: SiteUtilityLevels,
) -> HeatExchangerNetwork:
    """``network`` with its site cost.

    ``summary_metrics`` gain ``site_capital_cost``, ``site_operating_cost``
    and ``site_total_annual_cost``; ``source_metadata["site_utility_balance"]``
    holds the balance. The network's own costs are kept.
    """

    balance = account_utility_system(levels, site_loads(network, processes))
    return network.model_copy(
        update={
            "summary_metrics": network.summary_metrics
            | site_metrics(capital_cost(network), balance),
            "source_metadata": network.source_metadata
            | {"site_utility_balance": balance},
        }
    )


def with_site_prices(solver_arrays, balance):
    """``solver_arrays`` with each utility priced at its level's site price.

    A hot utility costs its level's ``use_price`` and a cold utility its
    level's ``generation_price`` negated, per period: heat raised on a main
    that is short is a credit, and heat raised into a level in surplus
    costs the dump it reaches. A side without a price keeps its own. Use
    never costs less than generation saves, so the priced utility cost of
    the loads ``balance`` was solved at is never below the balance's cost.
    """

    values = dict(solver_arrays.arrays)
    for side, prefix, axis, key in (
        ("use", HOT_PREFIX, "hot_utilities", "hu_cost_period"),
        ("generation", COLD_PREFIX, "cold_utilities", "cu_cost_period"),
    ):
        prices = np.array(values[key], dtype=float, copy=True)
        for u, identity in enumerate(solver_arrays.utility_identities[axis]):
            name = identity.removeprefix(prefix)
            for n in range(prices.shape[0]):
                try:
                    row = balance.level(name, n)
                except KeyError:
                    continue
                price = row.use_price if side == "use" else row.generation_price
                if price is not None:
                    prices[n][u] = price if side == "use" else -price
        values[key] = prices
    return replace(solver_arrays, arrays=values)


def _process_of(stream: str, processes: Sequence[str]) -> str:
    matches = [process for process in processes if stream.startswith(f"{process}.")]
    return max(matches, key=len, default="")


__all__ = [
    "capital_cost",
    "loads_from_level_duties",
    "priced_site_network",
    "site_loads",
    "site_metrics",
    "site_utility_levels",
    "with_site_prices",
]
