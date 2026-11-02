# HEN Base-Model Helpers

## Overview
Private helper modules that hold the equation, post-processing and solver-execution logic shared by `PinchDecompModel` and `StageWiseModel`. `../base.py` (`BaseHeatExchangerNetworkModel`) delegates to these modules, and `../_stagewise` and `../_pinch_design` import them directly. Every function takes the owning model as its first argument and reads or writes attributes on it; nothing here defines a class.

## Interface
The parent imports `alpha`, `approach`, `execution`, `parameters` and `piecewise`; `area`, `costing` and `superstructure` are used directly by `_stagewise` and `_pinch_design`.
- `get_alpha_values`, `alpha_value`, `set_alpha_dqda_equations` (`alpha.py`): post-optimisation alpha flow-on solve, neutral-alpha lookup, and the alpha and dQ/dA equations.
- `utility_minimum_approach_by_period`, `utility_match_feasibility`, `set_match_restrictions` (`approach.py`): least heater and cooler approach by period, reachable (stream, utility) flags, and the `z_allowed`, `z_hu_allowed`, `z_cu_allowed` restriction arrays. Underscore helpers used by siblings: `_set_minimum_approach_temperatures`, `_recovery_approach_temperature`, `_utility_solved_outlet_temperature`, `_set_multiperiod_utility_approach_equations`, `_weighted_state_average`.
- `solved_utility_duties`, `used_utility_matches`, `recovered_utility_price`, `set_utility_lmtd_and_areas` (`area.py`): solved heater and cooler duties, used-unit flags, recovered-heat price, and utility LMTD and area by period. Used by siblings: `_post_process_lmtd`, `_apply_segment_recovery_areas`, `_segment_exact_dqda`.
- `set_period_totals_and_costs` (`costing.py`): per-period duty totals, operating cost and capital cost.
- `optimise`, `cleanup_model`, `setup_model`, `output_to_cmd_line` (`execution.py`): solve loop, GEKKO directory removal, model and solver creation, diagnostics. Used by siblings: `_solver_value`, `_set_value`.
- `set_blank_input_parameters`, `get_model_parameters_from_solver_arrays` (`parameters.py`): blank parameter arrays, then population from `PreparedSolverArrays` with period validation.
- `create_utility_duty_grids`, `create_match_binaries` (`superstructure.py`): utility duty variables and recovery or utility match binaries.
- `piecewise.py`: underscore-only helpers for segmented streams and utilities (`_set_piecewise_stage_heat_coordinates`, `_set_piecewise_match_outlet_equations`, `_utility_cost_expression`, `_utility_cost_value`, `_parent_profile_duty`, `_recovery_heat_upper_bound`, `_solver_parent_is_segmented`, `_update_piecewise_active_segments`).
- Subdirectories: none.

## Invariants
- `period_ids` and `period_weights` must be present, the weights finite with a positive sum and one per period; the stream and utility `*_period` input arrays are non-empty, 2-D, with one row per period. [parameters.py:_normalise_state_arrays]
- The unsuffixed arrays (`T_h_in`, `f_h`, `dT_r` and similar) are copies of period 0. [parameters.py:_normalise_state_arrays] [approach.py:_set_minimum_approach_temperatures]
- Grid orientation: recovery `[hot][cold][stage]`, heaters `[period][cold][utility]`, coolers `[period][hot][utility]`; each (stream, utility) pair is its own unit. [superstructure.py:create_utility_duty_grids] [approach.py:utility_minimum_approach_by_period]
- Disallowed matches are fixed `Param` objects at 0, not variables. [superstructure.py:create_match_binaries]
- With `integers` false the match binaries are `Param`, so a heater or cooler whose fixed end misses its approach is forced to zero duty. [approach.py:_set_terminal_approach_equations]
- A failed alpha solve returns `None`, is not cached, and ranking falls back to alpha 1.0. [alpha.py:get_alpha_values] [alpha.py:alpha_value]
- A crossed match (non-positive terminal difference) has LMTD 0 and so zero area. [area.py:_post_process_lmtd]
- Design area of every unit is the maximum over periods; period totals are weighted by `period_weights`. [area.py:set_utility_lmtd_and_areas] [costing.py:_weighted_numeric_average]
- Capital cost is `unit_cost * n_units` plus power-law area terms for recovery, heaters and coolers. [costing.py:set_period_totals_and_costs]
- Piecewise solves repeat up to eight times until active segments stop changing; failure to stabilise, or a negative objective, marks the solve failed. [execution.py:_solve_and_post_process]
- `_set_value` clamps to variable bounds. [execution.py:_set_value]

## Side Effects
- Creates a local (non-remote) GEKKO model and configures its solver. [execution.py:setup_model]
- `optimise` always removes the GEKKO run directory afterwards; `cleanup_model` is safe to repeat. [execution.py:optimise] [execution.py:cleanup_model]
- `get_alpha_values` builds and solves a second GEKKO model and cleans it up. [alpha.py:get_alpha_values]
- Writes solver results, logs and many attributes onto the model passed in (`mSuccess`, `solver_run`, `area_*`, `LMTD_*`, `z_allowed`). [execution.py:_solve_and_post_process] [area.py:set_utility_lmtd_and_areas]
- Lazy imports of `OpenPinch.analysis.heat_exchanger_networks.solver.piecewise` inside segmented-stream branches. [area.py:_apply_segment_recovery_areas] [piecewise.py:_utility_cost_expression]
- Emits log records at info, warning and error level. [execution.py:output_to_cmd_line] [alpha.py:get_alpha_values]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_base_model_helpers.py` (there is no `tests/` directory mirroring `models/_base`; `test_utility_approach.py` and `test_segmented_streams.py` in the same folder also exercise it).
