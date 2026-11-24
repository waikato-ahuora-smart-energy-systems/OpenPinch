"""The site utility balance inside a site network's stage-wise model.

A site task's model sums each level's heater duties (use) and cooler duties
(generation) per period, closes every level with imports, letdowns and
dumps (``../../site/utility_system.py:add_site_balance``), and costs those
instead of each heater and cooler at its own price: the site pays only for
its net utility use. Reporting and verification read the solved balance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .....contracts.synthesis.site import SiteUtilityLevels, UtilitySystemBalance
from .....domain.heat_exchanger_network import HeatExchangerNetwork
from ...site.utility_system import add_site_balance, period_cost, solved_balance

# A level's solved balance may miss closing by this much (relative to its
# largest flow, or of 1 kW): the solver's feasibility tolerance.
BALANCE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class SiteUtilities:
    """A model's site levels and the level each of its utilities serves."""

    levels: SiteUtilityLevels
    hot_levels: tuple[int, ...]
    cold_levels: tuple[int, ...]


def site_costs(owner, n: int, hot_totals: Sequence[Any], cold_totals: Sequence[Any]):
    """Period ``n``'s site import cost and dump and letdown cost.

    ``hot_totals`` and ``cold_totals`` are each utility's duty in the
    period. The balance rows are added once per period and model.
    """

    blocks = _blocks(owner)
    if n not in blocks:
        site: SiteUtilities = owner.site_utilities
        count = len(site.levels.levels)
        use = [
            owner.m.sum(
                [t for u, t in enumerate(hot_totals) if site.hot_levels[u] == level]
            )
            for level in range(count)
        ]
        generation = [
            owner.m.sum(
                [t for u, t in enumerate(cold_totals) if site.cold_levels[u] == level]
            )
            for level in range(count)
        ]
        blocks[n] = add_site_balance(owner.m, site.levels, n, use, generation)
    return blocks[n].hot_cost, blocks[n].cold_cost


def site_cost_value(owner, side: str, n: int) -> float | None:
    """The solved site cost of one side in period ``n`` (imports for
    ``"hot"``, dumps and letdowns for ``"cold"``); ``None`` without a
    balance in the model."""
    block = _blocks(owner).get(n)
    if block is None:
        return None
    hot, cold = period_cost(owner.site_utilities.levels, block)
    return hot if side == "hot" else cold


def site_balance(model) -> UtilitySystemBalance | None:
    """The solved site balance of a model that carries one, else ``None``."""
    blocks = getattr(model, "_site_balance_blocks", None)
    if not blocks:
        return None
    return solved_balance(
        model.site_utilities.levels, [blocks[n] for n in sorted(blocks)]
    )


def balance_misses(model) -> list[str]:
    """Each level and period whose solved balance does not close."""
    balance = site_balance(model)
    if balance is None:
        return []
    misses = []
    for row in balance.balances:
        inflow = row.generation + row.letdown_in + row.imported
        outflow = row.use + row.letdown_out + row.dump
        scale = max(1.0, inflow, outflow)
        if abs(inflow - outflow) > BALANCE_TOLERANCE * scale:
            misses.append(f"site utility level {row.level} period {row.period_idx}")
    return misses


def with_site_cost(
    network: HeatExchangerNetwork, balance: UtilitySystemBalance
) -> HeatExchangerNetwork:
    """``network`` with its site balance and cost.

    ``summary_metrics`` gain ``site_capital_cost``, ``site_operating_cost``
    and ``site_total_annual_cost`` (scalars, as summary metrics must be);
    ``source_metadata["site_utility_balance"]`` holds the balance.
    """
    capital = float(network.capital_cost or 0.0)
    operating = float(balance.operating_cost)
    return network.model_copy(
        update={
            "summary_metrics": network.summary_metrics
            | {
                "site_capital_cost": capital,
                "site_operating_cost": operating,
                "site_total_annual_cost": capital + operating,
            },
            "source_metadata": network.source_metadata
            | {"site_utility_balance": balance},
        }
    )


def _blocks(owner) -> dict:
    blocks = getattr(owner, "_site_balance_blocks", None)
    if blocks is None:
        blocks = {}
        owner._site_balance_blocks = blocks
    return blocks


__all__ = [
    "BALANCE_TOLERANCE",
    "SiteUtilities",
    "balance_misses",
    "site_balance",
    "site_cost_value",
    "site_costs",
    "with_site_cost",
]
