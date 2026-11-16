# HEN Retrofit

## Overview
Retrofit of an existing heat exchanger network: a search, derived from the network evolution method, that adds one modification at a time (a new match or added area on an existing unit) to the existing network and evaluates each candidate with the fixed-structure model's "retrofit" goal (`../models/fixed_structure.py`). `moves.py` holds the structural modifications and the screen that ranks new matches.

## Interface
- `NewMatch` (`moves.py`): a new recovery match, hot and cold stream names and a 1-based stage, joining that stage or (`new_stage`) in a stage inserted there.
- `apply_new_match(network, move, exchanger_id)` (`moves.py`): the network with the match added.
- `new_splits(network, move)` (`moves.py`): the streams the match splits.
- `match_potential(network, ends, move, approach=...)`, `screen_new_matches(network, ends, approach=..., limit=...)` (`moves.py`): the heat a new match could recover at its position in a solved network, and the most promising new matches.
- `stage_count(network)` (`moves.py`): the stage count, or the deepest recovery stage.
- Subdirectories: none.

## Invariants
- Inserting a stage at `s` moves every exchanger (recovery, and utility placed after a stage) with a stage of `s` or more one stage later and adds one to the network's stage count; end utilities (no stage) stay at the stream ends [moves.py:apply_new_match].
- A match joining a stage splits each of its streams that already has a recovery match in that stage; a match in a new stage splits nothing [moves.py:new_splits].
- A hot stream's temperature entering a stage is the source inlet of its first exchanger at or after that point (recovery in that stage or later, coolers after that stage or at the end), else its target; a cold stream's likewise in its own direction (last stage to first). Readings are from the network's first period [moves.py:_hot_temperature, _cold_temperature].
- A match's potential is the smaller heat capacity flowrate times the hot less cold temperature at its position less the approach, at least zero. The screen tries every hot-cold pair joining each stage (except a stage the pair already shares) and in a new stage at each of the `S + 1` positions, keeps positive potentials, and ranks them by potential, fewest new splits, stage, joining before a new stage, then names [moves.py:match_potential, screen_new_matches].

## Side Effects
- None: pure functions on `HeatExchangerNetwork` values.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_retrofit_moves.py`
