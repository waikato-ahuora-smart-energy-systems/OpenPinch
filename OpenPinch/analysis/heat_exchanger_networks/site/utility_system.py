"""The site utility balance, as rows and variables of a network's own model.

Each process draws heat from a level (its heaters, ``use``) and supplies it
(its coolers, ``generation``). In each period the site closes every level
``l``:

``use_l + sum_m F[l->m] + X_l = generation_l + sum_h F[h->l] + B_l``

with ``B_l`` bought at the level's import price (hot and both levels,
within its import capacity), ``X_l`` rejected at its dump price (cold and
both levels) and ``F[h->l]`` let down from a level ``h`` wholly above
``l`` in that period, at the letdown price. The block adds ``B``, ``X`` and
``F`` and these rows to the model that holds the heater and cooler duties,
and its cost replaces the heaters' and coolers' prices: the site pays only
for its net utility use. There is no separate accounting problem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ....contracts.synthesis.site import (
    SiteUtilityLevels,
    UtilityLevelBalance,
    UtilitySystemBalance,
)


@dataclass(frozen=True)
class SiteBalancePeriod:
    """One period's balance in a model: its loads, variables and costs.

    ``use`` and ``generation`` are the model's expressions per level;
    ``imports`` and ``dumps`` map a level index to its variable, ``flows`` a
    ``(hotter, colder)`` pair to its letdown variable. ``hot_cost`` is what
    the imports cost, ``cold_cost`` the dumps and letdowns.
    """

    period_idx: int
    use: tuple[Any, ...]
    generation: tuple[Any, ...]
    imports: dict[int, Any]
    dumps: dict[int, Any]
    flows: dict[tuple[int, int], Any]
    hot_cost: Any
    cold_cost: Any


def letdowns(levels: SiteUtilityLevels, n: int) -> list[tuple[int, int]]:
    """Each pair ``(h, l)`` where level h lies wholly above level l in
    period ``n``."""
    items = levels.levels
    return [
        (h, low)
        for h, upper in enumerate(items)
        for low, lower in enumerate(items)
        if h != low and upper.t_low[n] >= lower.t_high[n]
    ]


def add_site_balance(
    m,
    levels: SiteUtilityLevels,
    n: int,
    use: Sequence[Any],
    generation: Sequence[Any],
) -> SiteBalancePeriod:
    """Add period ``n``'s balance rows to model ``m``.

    ``use`` and ``generation`` give each level's heater and cooler duty, in
    the order of ``levels.levels``, as expressions of ``m``.
    """

    items = levels.levels
    imports = {}
    dumps = {}
    for i, level in enumerate(items):
        if level.imports:
            capacity = level.import_capacity
            bound = None if capacity is None else capacity[n]
            imports[i] = m.Var(value=0.0, lb=0.0, ub=bound)
        if level.dumps:
            dumps[i] = m.Var(value=0.0, lb=0.0)
            # A dump credit (a negative price) pays for steam the processes
            # raise; dumping bought or let-down steam earns none, and
            # crediting it would let the site buy steam to sell it.
            if level.dump_price[n] < 0.0:
                m.Equation(dumps[i] <= generation[i])
    flows = {pair: m.Var(value=0.0, lb=0.0) for pair in letdowns(levels, n)}
    for i in range(len(items)):
        out = [flow for (h, _low), flow in flows.items() if h == i]
        into = [flow for (_h, low), flow in flows.items() if low == i]
        m.Equation(
            use[i] + sum(out) + (dumps[i] if i in dumps else 0.0)
            == generation[i] + sum(into) + (imports[i] if i in imports else 0.0)
        )
    hot_cost = sum(items[i].import_price[n] * var for i, var in imports.items())
    cold_cost = sum(items[i].dump_price[n] * var for i, var in dumps.items()) + sum(
        levels.letdown_price * flow for flow in flows.values()
    )
    return SiteBalancePeriod(
        period_idx=n,
        use=tuple(use),
        generation=tuple(generation),
        imports=imports,
        dumps=dumps,
        flows=flows,
        hot_cost=hot_cost,
        cold_cost=cold_cost,
    )


def period_cost(levels: SiteUtilityLevels, block: SiteBalancePeriod) -> tuple:
    """The solved hot (import) and cold (dump and letdown) cost of a period."""
    n = block.period_idx
    items = levels.levels
    hot = sum(items[i].import_price[n] * _value(v) for i, v in block.imports.items())
    cold = sum(items[i].dump_price[n] * _value(v) for i, v in block.dumps.items())
    cold += sum(levels.letdown_price * _value(f) for f in block.flows.values())
    return float(hot), float(cold)


def solved_balance(
    levels: SiteUtilityLevels, blocks: Sequence[SiteBalancePeriod]
) -> UtilitySystemBalance:
    """The balance a solved model reached, per level and period."""

    rows: list[UtilityLevelBalance] = []
    period_costs: list[float] = []
    hot: list[float] = []
    cold: list[float] = []
    for block in blocks:
        n = block.period_idx
        period_rows = []
        for i, level in enumerate(levels.levels):
            imported = _value(block.imports[i]) if i in block.imports else 0.0
            dump = _value(block.dumps[i]) if i in block.dumps else 0.0
            letdown_in = sum(
                _value(f) for (_h, low), f in block.flows.items() if low == i
            )
            letdown_out = sum(
                _value(f) for (h, _low), f in block.flows.items() if h == i
            )
            cost = (
                (level.import_price[n] * imported if level.imports else 0.0)
                + (level.dump_price[n] * dump if level.dumps else 0.0)
                + levels.letdown_price * letdown_in
            )
            period_rows.append(
                UtilityLevelBalance(
                    level=level.name,
                    period_idx=n,
                    use=_value(block.use[i]),
                    generation=_value(block.generation[i]),
                    imported=imported,
                    letdown_in=letdown_in,
                    letdown_out=letdown_out,
                    dump=dump,
                    cost=float(cost),
                )
            )
        rows += period_rows
        period_costs.append(sum(row.cost for row in period_rows))
        hot.append(sum(row.imported for row in period_rows))
        cold.append(sum(row.dump for row in period_rows))
    weights = [float(levels.period_weights[block.period_idx]) for block in blocks]
    return UtilitySystemBalance(
        balances=tuple(rows),
        period_costs=tuple(period_costs),
        operating_cost=sum(w * c for w, c in zip(weights, period_costs, strict=True))
        / sum(weights),
        hot_utility=tuple(hot),
        cold_utility=tuple(cold),
    )


def balance_of_loads(
    levels: SiteUtilityLevels,
    use: Mapping[str, Sequence[float]],
    generation: Mapping[str, Sequence[float]],
) -> UtilitySystemBalance:
    """The cheapest site balance of fixed loads.

    ``use`` and ``generation`` give each level's heater and cooler duty per
    period (kW), by level name; a level left out has none. The balance is a
    small linear model of its own, solved with HiGHS: what a site network's
    model would reach with every duty fixed. Raises ``ValueError`` when the
    loads cannot be balanced (an import capacity too small for them).
    """

    from ..solver import backend
    from ..solver.pyomo_model import PyomoModel

    m = PyomoModel()
    periods = range(len(levels.period_weights))
    blocks = [
        add_site_balance(
            m,
            levels,
            n,
            [m.Param(_load(use, level.name, n)) for level in levels.levels],
            [m.Param(_load(generation, level.name, n)) for level in levels.levels],
        )
        for n in periods
    ]
    m.Minimize(
        sum(
            weight * (block.hot_cost + block.cold_cost)
            for weight, block in zip(levels.period_weights, blocks, strict=True)
        )
    )
    run = backend.solve_pyomo_model(m, solver_name="highs")
    if run.failure_reason is not None:
        raise ValueError(
            f"the site utility balance of these loads has no solution "
            f"({run.failure_reason})."
        )
    return solved_balance(levels, blocks)


def _load(loads: Mapping[str, Sequence[float]], level: str, n: int) -> float:
    duties = loads.get(level)
    return 0.0 if duties is None else float(duties[n])


def _value(expression: Any) -> float:
    """A solved variable's or expression's number (0 for a plain 0)."""
    if isinstance(expression, int | float):
        return float(expression)
    from pyomo.environ import value

    return float(value(expression, exception=False) or 0.0)


__all__ = [
    "SiteBalancePeriod",
    "add_site_balance",
    "balance_of_loads",
    "letdowns",
    "period_cost",
    "solved_balance",
]
