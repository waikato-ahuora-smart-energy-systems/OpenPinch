# HPR Cycle Targeting

## Overview
One module per heat pump and refrigeration (HPR) cycle topology. Each defines how the optimisation vector is laid out, decoded to a `HPRParsedState`, and evaluated into an `HPRBackendResult`, and exposes an `optimise_*` entry point that calls `solve_hpr_placement` from `../optimisation_adapter`. Cycle physics lives in `../cycles`, shared encoding and result building in `../common`, and CoolProp and TESPy capability checks in `../performance_maps`. `../_multiperiod/setup.py` reuses the private setup, parse and objective functions.

## Interface
- `optimise_cascade_carnot_heat_pump_placement` (`cascade_carnot.py`): cascade Carnot screening solve.
- `optimise_parallel_carnot_heat_pump_placement` (`parallel_carnot.py`): parallel Carnot screening solve.
- `optimise_cascade_heat_pump_placement` (`cascade_vapour_compression.py`): cascade vapour compression, including the single-stage TESPy path via `prepared_tespy`.
- `optimise_parallel_heat_pump_placement` (`parallel_vapour_compression.py`): parallel vapour compression.
- `optimise_vapour_compression_mvr_heat_pump_placement` (`vapour_compression_mvr.py`): vapour compression low stages with MVR high stages; heat pump only.
- `optimise_brayton_heat_pump_placement` (`brayton.py`): raises `NotImplementedError`; the module is experimental.
- `CarnotTopology`, `carnot_opt_setup`, `parse_carnot_state_variables`, `compute_carnot_objective`, `normalise_carnot_stage_counts` (`_carnot_common.py`): shared Carnot machinery used by the two Carnot modules.
- Private `_get_*_opt_setup`, `_parse_*_state*`, `_compute_*_obj` functions in each module: starting point and bounds, vector decoding, and objective. Other modules import these.

## Invariants
- Carnot objective vectors hold condenser fractions, evaporator fractions and heat-split fractions, all bounded to [0, 1]; the initial point starts every heat split at 1 [_carnot_common.py:carnot_opt_setup].
- Parallel Carnot and parallel vapour compression equalise `n_cond` and `n_evap` to the larger value on `args`, and callers rely on that after the solve [_carnot_common.py:normalise_carnot_stage_counts, parallel_vapour_compression.py:optimise_parallel_heat_pump_placement].
- Condenser temperatures are capped and evaporator temperatures floored through `common.shared` helpers before use [cascade_vapour_compression.py:_parse_cascade_hp_state_variables, parallel_vapour_compression.py:_parse_parallel_hp_state_temperatures].
- Air is a partner for refrigeration heat sink and heat pump heat source only [_carnot_common.py:parse_carnot_state_variables].
- The Carnot warm start is clipped into the refrigerant's subcritical condensing range and seed duties are encoded against availability at the seed temperatures [cascade_vapour_compression.py:_get_cascade_hp_opt_setup, parallel_vapour_compression.py:_get_parallel_hp_opt_setup, vapour_compression_mvr.py:_get_vc_mvr_opt_setup].
- Cascade process-control counts depend on mode: a heat pump has `n_cond` heat controls and `n_evap - 1` cool controls, a refrigerator the reverse [cascade_vapour_compression.py:_cascade_process_control_counts].
- VC+MVR state arrays are ordered [MVR stages, VC stages]; the MVR lift is at most `MAX_MVR_STAGE_LIFT` (20 K) [vapour_compression_mvr.py:_unpack_vc_mvr_state, vapour_compression_mvr.py:_parse_vc_mvr_state_variables].
- VC+MVR is heat-pump-only and requires at least one MVR stage [vapour_compression_mvr.py:optimise_vapour_compression_mvr_heat_pump_placement, vapour_compression_mvr.py:_num_mvr_stages].
- A failed VC+MVR candidate returns a finite penalised objective with `success=False`, which `optimisation_adapter._CachedHPRScalarObjective` scores as `1e30`, so the finite value never ranks candidates; errors after the cycle is evaluated are re-raised as post-solve errors [vapour_compression_mvr.py:_finite_failed_vc_mvr_result, vapour_compression_mvr.py:_compute_vc_mvr_system_obj].
- The TESPy candidate uses fixed superheat (5 K), disables the internal heat exchanger (bounds 0 to 0), and rejects candidates with non-positive allocated duty [cascade_vapour_compression.py:_TESPY_TARGET_SUPERHEAT, cascade_vapour_compression.py:_compute_tespy_cascade_hp_system_obj].
- Final-mode results carry a simulation record; search-mode results do not [cascade_vapour_compression.py:_compute_tespy_cascade_hp_system_obj, vapour_compression_mvr.py:_compute_vc_mvr_system_obj].

## Side Effects
- `args` is mutated: stage counts, `refrigerant_ls` and `mvr_fluid_ls` [parallel_vapour_compression.py:optimise_parallel_heat_pump_placement, cascade_vapour_compression.py:optimise_cascade_heat_pump_placement, vapour_compression_mvr.py:optimise_vapour_compression_mvr_heat_pump_placement].
- CoolProp fluid preflights run before the main search, as does a Carnot seed search when `hpr.initialise_simulated_cycle` is true (the default); a seed failure is ignored and the TESPy path never seeds [cascade_vapour_compression.py:optimise_cascade_heat_pump_placement].
- A TESPy evaluator session is opened and closed around the search [cascade_vapour_compression.py:_optimise_tespy_cascade_heat_pump_placement].
- A debug log line is emitted by the VC+MVR objective only when `debug` is true and the cycle raised `ValueError` before it was evaluated [vapour_compression_mvr.py:_compute_vc_mvr_system_obj].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_cascade_carnot.py tests/analysis/heat_pumps/test_parallel_carnot.py tests/analysis/heat_pumps/test_cascade_vapour_compression.py tests/analysis/heat_pumps/test_parallel_vapour_compression.py tests/analysis/heat_pumps/test_vapour_compression_mvr.py tests/analysis/heat_pumps/test_brayton.py`
