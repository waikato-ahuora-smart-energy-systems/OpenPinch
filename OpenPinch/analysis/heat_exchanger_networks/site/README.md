# Total Site Heat Exchanger Network Design

## Overview
Helpers for designing one network for a site of several processes linked by
shared utilities. `seeds.py` joins each process's pinch design networks into
site seeds; `utility_system.py` accounts a site's utility levels: given each
process's heater and cooler duty on each level, it finds the cheapest balance
of imports, letdowns and dumps, and the marginal price of each level. The
public entry point is `design.total_site_heat_exchanger_network`
(`OpenPinch/application/_problem/accessors/design.py`).

## Interface
- `process_level_duties`, `site_seeds` (`seeds.py`): a process network's utility duties per level (its level targets when it holds them, otherwise its own heater and cooler duties), and site seeds: one per dt_cont multiplier in ascending order, joining each process's best network at that multiplier (or the nearest one it has) into the recovery matches, the largest stage count and `source_metadata["site_processes"]` (each process's level duties); seeds without recovery are dropped, and `ValueError` is raised when none recovers heat. With `levels`, each seed is priced (its process networks' capital plus the site balance of their level duties, in `summary_metrics` and `source_metadata["site_utility_balance"]`) and the seeds come cheapest first by site total annual cost.
- `site_utility_levels`, `site_loads`, `loads_from_level_duties`, `capital_cost`, `site_metrics`, `priced_site_network` (`costing.py`): a site's levels from its solver arrays (a utility that is both hot and cold is a main; annual prices, so costs are $/y; temperatures in degC); a site network's heater and cooler duties per process and level; the same from `site_processes`; a network's capital (its own, else TAC less utility cost); and a network with its site cost: `summary_metrics` `site_capital_cost`, `site_operating_cost` and `site_total_annual_cost` (scalars, as summary metrics must be), and the balance in `source_metadata["site_utility_balance"]`.
- `account_utility_system(levels, loads)` (`utility_system.py`): `levels` is a `SiteUtilityLevels` and `loads` maps each process to a `LevelLoad` (use and generation per period, kW) by level name; returns a `UtilitySystemBalance` (`OpenPinch/contracts/synthesis/site.py`) with, per level and period, use, generation, import, letdown in and out, dump, cost and the marginal price, and the period costs, the period-weighted operating cost and the site hot and cold utility per period.
- Subdirectories: none.

## Invariants
- Per level and period: `use + letdown_out + dump == generation + letdown_in + imported`. Imports go only into hot and both levels, at the import price and within `import_capacity`; dumps leave only cold and both levels, at the dump price; heat is let down from a level to another only when the first lies wholly above the second (`t_low >= t_high`), at `letdown_price` [utility_system.py:_solve_period].
- A level is a utility's name without its `"Hot Utility."` or `"Cold Utility."` prefix; a stream belongs to the longest listed process its key starts with (`"<process>.<stream>"`), else to the process `""` [costing.py:site_loads].
- The site total annual cost is capital plus the period-weighted site operating cost; the network's own costs are left as they are [costing.py:site_metrics, priced_site_network].
- The balance is the cheapest one: a linear programme per period, solved by `scipy.optimize.linprog(method="highs")`, so it needs no optional solver and runs in Pyodide [utility_system.py:_solve_period].
- A level's `marginal_price` is the dual of its balance (HiGHS's equality marginal): the cost of one more kW of use there, so generation on a level that is short saves its import price, and generation on a level in surplus costs the cheapest dump it reaches. With no import capacity binding, the sum over levels of marginal price times `use - generation` equals the period cost. At a degenerate point (a level exactly balanced) the dual is one vertex of the range of valid prices [utility_system.py:_solve_period].
- `operating_cost` is `sum(w_n * cost_n) / sum(w_n)` over the period weights, as the stage-wise models weight periods [utility_system.py:account_utility_system].
- With import prices rising with temperature and one dump price on every level, the balance's cost and site hot and cold utility equal the Total Site cascade's (`OpenPinch/analysis/targeting/indirect.py:_target_site_utilities`); a Hypothesis property checks it.

## Errors
- `ValueError` for a load on an unknown level, use of a cold level, generation on a hot level, a load with the wrong number of periods, or a balance the levels cannot close in some period (an import capacity too small) [utility_system.py:_totals, _solve_period].
- The contracts reject a level whose kind and prices disagree (a hot level needs `import_price` and takes no `dump_price`, a cold level the reverse, a both level both), non-finite prices or temperatures, `t_high < t_low`, duplicate level names, values not one per period, and negative or non-finite loads [`OpenPinch/contracts/synthesis/site.py`].

## Side Effects
- None: every function is pure.
