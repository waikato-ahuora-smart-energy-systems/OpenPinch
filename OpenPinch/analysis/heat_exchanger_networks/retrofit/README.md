# HEN Retrofit

## Overview
Retrofit of an existing heat exchanger network: a search, derived from the network evolution method, that adds one modification at a time (a new match or added area on an existing unit) to the existing network and evaluates each candidate with the fixed-structure model's "retrofit" goal (`../models/fixed_structure.py`). The best design at each depth makes a frontier over the number of modifications. `search.py` runs the search, `moves.py` holds the structural modifications and the screen that ranks new matches, and `frontier.py` marks dominated designs. Results use the contracts in `OpenPinch/contracts/synthesis/retrofit.py`.

## Interface
- `RetrofitSettings`, `run_retrofit_search(problem, existing, settings)` (`search.py`): search limits and retrofit fixed costs, and the search, returning a `HeatExchangerNetworkRetrofitResult`.
- `NewMatch` (`moves.py`): a new recovery match, hot and cold stream names and a 1-based stage, joining that stage or (`new_stage`) in a stage inserted there.
- `apply_new_match(network, move, exchanger_id)` (`moves.py`): the network with the match added.
- `new_splits(network, move)` (`moves.py`): the streams the match splits.
- `match_potential(network, ends, move, approach=...)`, `screen_new_matches(network, ends, approach=..., limit=...)` (`moves.py`): the heat a new match could recover at its position in a solved network, and the most promising new matches.
- `stage_count(network)` (`moves.py`): the stage count, or the deepest recovery stage.
- `mark_dominated(designs)` (`frontier.py`): the designs with `dominated` set.
- Subdirectories: none.

## Invariants
- Every exchanger of the existing network needs a positive installed `area`, else `ValueError` naming those without one; an existing network the "retrofit" goal cannot solve at its installed areas (widened by `area_tolerance`) raises `ValueError` [search.py:run_retrofit_search].
- Depth 0 is the existing network; its utility cost is the base for savings. Each depth tries, from every design in the beam, added area on each existing unit at its cap (active and needing at least 99 % of its widened area) and the `new_match_candidates` best screened new matches; a structure (recovery matches by stage, and the enlarged units) is solved at most once per search. Children are ranked by net annual benefit, the best `beam_width` form the next beam, and the best is the depth's frontier design. The search stops at `max_modifications` or when a depth has no solved child [search.py:run_retrofit_search, _children, _signature].
- A design is rejected when any of its modifications does nothing: a unit with less than 1e-3 m2 of added area or no duty [search.py:_Evaluator._modification].
- A modification's area cost is `COSTING_HX_AREA_COEFF * added ** COSTING_HX_AREA_EXP` in $; its fixed cost is `new_unit_cost` plus `split_cost` per new split for a new match, `area_addition_cost` for added area. Capital is their sum, annualised with the capital recovery factor of `COSTING_DISCOUNT_RATE` and `COSTING_SERVICE_LIFE`; net annual benefit is savings less annualised capital and payback is capital over savings (`None` without savings) [search.py:_Evaluator.evaluate, _modification].
- Moves use the stream names of the existing network: a solver stream name maps to the existing name it ends with (`H1` for `Process A.H1`) when exactly one matches. New exchangers are named `new:<hot>-><cold>`, with `:2`, `:3` and so on when taken [search.py:_existing_names, _new_id].
- A frontier design is dominated when one with fewer modifications has at least its net annual benefit [frontier.py:mark_dominated].
- Inserting a stage at `s` moves every exchanger (recovery, and utility placed after a stage) with a stage of `s` or more one stage later and adds one to the network's stage count; end utilities (no stage) stay at the stream ends [moves.py:apply_new_match].
- A match joining a stage splits each of its streams that already has a recovery match in that stage; a match in a new stage splits nothing [moves.py:new_splits].
- A hot stream's temperature entering a stage is the source inlet of its first exchanger at or after that point (recovery in that stage or later, coolers after that stage or at the end), else its target; a cold stream's likewise in its own direction (last stage to first). Readings are from the network's first period [moves.py:_hot_temperature, _cold_temperature].
- A match's potential is the smaller heat capacity flowrate times the hot less cold temperature at its position less the approach, at least zero. The screen tries every hot-cold pair joining each stage (except a stage the pair already shares) and in a new stage at each of the `S + 1` positions, keeps positive potentials, and ranks them by potential, fewest new splits, stage, joining before a new stage, then names [moves.py:match_potential, screen_new_matches].

## Side Effects
- `run_retrofit_search` runs direct heat integration through `problem.target` when no targets are cached, and solves each candidate with the configured `HENS_SOLVER_EVM` solver (an external executable via Pyomo), serially in-process [search.py:run_retrofit_search, _Evaluator].
- `moves.py` and `frontier.py` are pure.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_retrofit_moves.py tests/analysis/heat_exchanger_networks/test_retrofit_frontier.py tests/analysis/heat_exchanger_networks/test_retrofit.py` (the last needs Ipopt and is marked `solver`).
