"""Site utility accounting: the cheapest balance of a site's utility levels.

Each process draws heat from a level (its heaters, ``use``) and supplies it
(its coolers, ``generation``). Per period the site closes every level ``l``:

``use_l + sum_m F[l->m] + X_l = generation_l + sum_h F[h->l] + B_l``

with ``B_l`` bought at the level's import price (hot and both levels),
``X_l`` rejected at its dump price (cold and both levels) and ``F[h->l]``
let down from a level ``h`` wholly above ``l`` at the letdown price. The
linear programme is solved with ``scipy.optimize.linprog`` (HiGHS), which
needs no optional solver and runs in Pyodide. Its dual on a level's balance
is the level's marginal price: what one more kW of use there costs, and so
what one more kW of generation saves.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import linprog

from ....contracts.synthesis.site import (
    LevelLoad,
    SiteUtilityLevels,
    UtilityLevelBalance,
    UtilitySystemBalance,
)


def account_utility_system(
    levels: SiteUtilityLevels,
    loads: Mapping[str, Mapping[str, LevelLoad]],
) -> UtilitySystemBalance:
    """The site's cheapest utility balance for the processes' ``loads``.

    ``loads`` maps each process to its load on each level it uses, by level
    name. Raises ``ValueError`` for an unknown level, a load the level
    cannot take (use of a cold level, generation on a hot one), a wrong
    period count, or a balance the levels cannot close (an import capacity
    too small).
    """

    periods = len(levels.period_weights)
    use, generation = _totals(levels, loads, periods)
    rows: list[UtilityLevelBalance] = []
    period_costs: list[float] = []
    hot: list[float] = []
    cold: list[float] = []
    for n in range(periods):
        period_rows = _solve_period(levels, use[:, n], generation[:, n], n)
        rows += period_rows
        period_costs.append(sum(row.cost for row in period_rows))
        hot.append(sum(row.imported for row in period_rows))
        cold.append(sum(row.dump for row in period_rows))
    weights = np.asarray(levels.period_weights, dtype=float)
    return UtilitySystemBalance(
        balances=tuple(rows),
        period_costs=tuple(period_costs),
        operating_cost=float(weights @ np.asarray(period_costs) / weights.sum()),
        hot_utility=tuple(hot),
        cold_utility=tuple(cold),
    )


def _totals(levels, loads, periods: int) -> tuple[np.ndarray, np.ndarray]:
    names = [level.name for level in levels.levels]
    use = np.zeros((len(names), periods))
    generation = np.zeros((len(names), periods))
    for process, by_level in loads.items():
        for name, load in by_level.items():
            if name not in names:
                raise ValueError(f"process {process} names an unknown level {name}")
            if len(load.use) != periods:
                raise ValueError(
                    f"process {process} needs {periods} period values on {name}"
                )
            level = levels.level(name)
            if not level.imports and any(q > 0.0 for q in load.use):
                raise ValueError(f"cold level {name} cannot supply process {process}")
            if not level.dumps and any(q > 0.0 for q in load.generation):
                raise ValueError(
                    f"hot level {name} cannot take process {process}'s heat"
                )
            index = names.index(name)
            use[index] += load.use
            generation[index] += load.generation
    return use, generation


def _letdowns(levels) -> list[tuple[int, int]]:
    """Each pair ``(h, l)`` where level h lies wholly above level l."""
    return [
        (h, low)
        for h, upper in enumerate(levels.levels)
        for low, lower in enumerate(levels.levels)
        if h != low and upper.t_low >= lower.t_high
    ]


def _solve_period(levels, use, generation, n: int) -> list[UtilityLevelBalance]:
    items = levels.levels
    count = len(items)
    imports = [i for i, level in enumerate(items) if level.imports]
    dumps = [i for i, level in enumerate(items) if level.dumps]
    letdowns = _letdowns(levels)
    columns = len(imports) + len(dumps) + len(letdowns)
    costs = np.zeros(columns)
    balance = np.zeros((count, columns))
    bounds: list[tuple[float, float | None]] = []
    for c, i in enumerate(imports):
        costs[c] = items[i].import_price[n]
        balance[i, c] = 1.0
        capacity = items[i].import_capacity
        bounds.append((0.0, None if capacity is None else capacity[n]))
    offset = len(imports)
    for c, i in enumerate(dumps, start=offset):
        costs[c] = items[i].dump_price[n]
        balance[i, c] = -1.0
        bounds.append((0.0, None))
    offset += len(dumps)
    for c, (h, low) in enumerate(letdowns, start=offset):
        costs[c] = levels.letdown_price
        balance[h, c] = -1.0
        balance[low, c] = 1.0
        bounds.append((0.0, None))
    solution = linprog(
        costs,
        A_eq=balance,
        b_eq=use - generation,
        bounds=bounds,
        method="highs",
    )
    if solution.status != 0:
        raise ValueError(
            f"the site utility levels cannot balance period {n}: {solution.message}"
        )
    x = solution.x
    duals = solution.eqlin.marginals
    capped = [(c, bound[1]) for c, bound in enumerate(bounds) if bound[1] is not None]
    prices = _OptimalDuals(balance, costs, use - generation, capped, solution.fun)
    rows = []
    for i, level in enumerate(items):
        imported = sum(x[c] for c, j in enumerate(imports) if j == i)
        dump = sum(x[c] for c, j in enumerate(dumps, start=len(imports)) if j == i)
        flows = list(enumerate(letdowns, start=len(imports) + len(dumps)))
        letdown_in = sum(x[c] for c, (_h, low) in flows if low == i)
        letdown_out = sum(x[c] for c, (h, _low) in flows if h == i)
        cost = (
            (level.import_price[n] * imported if level.imports else 0.0)
            + (level.dump_price[n] * dump if level.dumps else 0.0)
            + levels.letdown_price * letdown_in
        )
        use_price = prices.extreme(i, highest=True) if level.imports else None
        generation_price = prices.extreme(i, highest=False) if level.dumps else None
        sides = [p for p in (use_price, generation_price) if p is not None]
        rows.append(
            UtilityLevelBalance(
                level=level.name,
                period_idx=n,
                use=float(use[i]),
                generation=float(generation[i]),
                imported=float(imported),
                letdown_in=float(letdown_in),
                letdown_out=float(letdown_out),
                dump=float(dump),
                cost=float(cost),
                marginal_price=float(sum(sides) / len(sides) if sides else duals[i]),
                use_price=use_price,
                generation_price=generation_price,
            )
        )
    return rows


class _OptimalDuals:
    """The set of optimal duals of one period's balance.

    Duals ``lambda`` (one per level) and ``mu <= 0`` (one per capped
    column) with ``A' lambda + mu <= c`` and an objective within a hair of
    the optimum. The largest ``lambda_l`` over the set is the cost of one
    more kW of use on level l; the smallest, what one more kW of generation
    there saves. They differ only where the balance is degenerate.
    """

    def __init__(self, balance, costs, rhs, capped, optimum: float) -> None:
        levels, columns = balance.shape
        size = levels + len(capped)
        rows = np.zeros((columns + 1, size))
        rows[:columns, :levels] = balance.T
        for k, (c, _bound) in enumerate(capped):
            rows[c, levels + k] = 1.0
        # Objective at least the optimum (weak duality caps it there).
        rows[columns, :levels] = -rhs
        rows[columns, levels:] = [-bound for _c, bound in capped]
        self._rows = rows
        self._limits = np.append(costs, -optimum + 1e-9 * max(1.0, abs(optimum)))
        self._bounds = [(None, None)] * levels + [(None, 0.0)] * len(capped)
        self._size = size

    def extreme(self, level: int, *, highest: bool) -> float | None:
        """The largest (or smallest) optimal dual of ``level``; ``None``
        when it is unbounded (no more use, or generation, can be taken)."""
        objective = np.zeros(self._size)
        objective[level] = -1.0 if highest else 1.0
        solution = linprog(
            objective,
            A_ub=self._rows,
            b_ub=self._limits,
            bounds=self._bounds,
            method="highs",
        )
        if solution.status != 0:
            return None
        return float(-solution.fun if highest else solution.fun)


__all__ = ["account_utility_system"]
