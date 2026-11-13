# HEN Base-Model Helpers

## Overview
Private helper modules that hold the equation, post-processing and solver-execution logic shared by `PinchDecompModel` and `StageWiseModel`. `../base.py` (`BaseHeatExchangerNetworkModel`) delegates to these modules, and `../_stagewise` and `../_pinch_design` import them directly. Every function takes the owning model as its first argument and reads or writes attributes on it; nothing here defines a class.

## Interface
The parent imports `alpha`, `approach`, `execution`, `parameters` and `piecewise`; `area`, `costing` and `superstructure` are used directly by `_stagewise` and `_pinch_design`.
- `get_alpha_values`, `alpha_value` (`alpha.py`): post-optimisation alpha flow-on solve and neutral-alpha lookup.
- `utility_minimum_approach_by_period`, `utility_match_feasibility`, `set_match_restrictions` (`approach.py`): least heater and cooler approach by period, reachable (stream, utility) flags, and the `z_allowed`, `z_hu_allowed`, `z_cu_allowed` restriction arrays. Underscore helpers used by siblings: `_set_minimum_approach_temperatures`, `_recovery_approach_temperature`, `_utility_solved_outlet_temperature`, `_set_multiperiod_utility_approach_equations`, `_weighted_state_average`.
- `solved_utility_duties`, `used_utility_matches`, `recovered_utility_price`, `set_utility_lmtd_and_areas` (`area.py`): solved heater and cooler duties, used-unit flags, recovered-heat price, and utility LMTD and area by period. Used by siblings: `_post_process_lmtd`, `_apply_segment_recovery_areas`, `_segment_exact_dqda`.
- `set_period_totals_and_costs` (`costing.py`): per-period duty totals, operating cost and capital cost.
- `optimise`, `cleanup_model`, `setup_model`, `output_to_cmd_line` (`execution.py`): solve loop, run directory removal, native Pyomo model and solver creation, diagnostics. Used by siblings: `_solver_value`, `_set_value`, `_assign_value`.
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
- Alpha and the gammas are solved from the solved model as one sparse linear system; a singular system, a residual above 1e-8 or a value outside [-1, 1] returns `None`, is not cached, and ranking falls back to alpha 1.0. [alpha.py:get_alpha_values, _solve_flow_on_system] [alpha.py:alpha_value]
- A crossed match (non-positive terminal difference) has LMTD 0 and so zero area. [area.py:_post_process_lmtd]
- Design area of every unit is the maximum over periods; period totals are weighted by `period_weights`. [area.py:set_utility_lmtd_and_areas] [costing.py:_weighted_numeric_average]
- Capital cost is `unit_cost * n_units` plus power-law area terms for recovery, heaters and coolers. [costing.py:set_period_totals_and_costs]
- With a solver in `backend.INTEGER_SOLVERS` (`couenne`, `highs`, `scip`, `mindtpy`) segment temperature and price profiles use interval binaries; with the others they use relaxed selectors that the solve loop updates. Approach-contribution profiles always use relaxed selectors. [piecewise.py:_set_piecewise_stage_heat_coordinates, _utility_cost_expression]
- Piecewise solves repeat up to eight times until active segments stop changing; failure to stabilise, or a negative objective, marks the solve failed. [execution.py:_solve_and_post_process]
- `_set_value` clamps to variable bounds; a parameter (a fixed `PyomoModel` variable) takes the value unclamped. [execution.py:_set_value]
- `_assign_value` writes a value unclamped onto a Pyomo variable or parameter; warm starts, amalgamation, evolution moves and stage reduction write values only through it or `_set_value`. [execution.py:_assign_value]
- The solve loop calls `solve_pyomo_model` and judges success, a negative objective and the logged objective from the `SolverRun` status and objective value. [execution.py:_solve_and_post_process]
- A relaxed match (a fixed parameter) whose fixed end misses its approach forces the unit's duty to zero. [approach.py:_set_terminal_approach_equations, _is_parameter]

## Side Effects
- Checks the Pyomo solver (binary first, then Pyomo) and creates a native `PyomoModel`. [execution.py:setup_model]
- `optimise` always removes the model's run directory afterwards; `cleanup_model` is safe to repeat. [execution.py:optimise] [execution.py:cleanup_model]
- A solve that stopped before converging (`solver_run.converged` false) keeps `mSuccess` and its values, but is logged as a warning (`[Unconverged]`) rather than `[Success]`. [execution.py:_solve_and_post_process]
- A failed solve sets `mSuccess` to 0 and is logged at ERROR, or at the model's `failure_log_level` when a caller that expects failures (heuristic proposals, the minimum-unit search) sets one. A passing status with no objective value is a failure (`solver returned no objective value`). [execution.py:_solve_and_post_process]
- `get_alpha_values` writes the flow-fraction grids (`P_h`, `P_c`, `beta_h`, `beta_c`, `z_i`, `z_j`) and `alpha` onto the model. [alpha.py:get_alpha_values]
- Writes solver results, logs and many attributes onto the model passed in (`mSuccess`, `solver_run`, `area_*`, `LMTD_*`, `z_allowed`). [execution.py:_solve_and_post_process] [area.py:set_utility_lmtd_and_areas]
- Lazy imports of `OpenPinch.analysis.heat_exchanger_networks.solver.piecewise` inside segmented-stream branches. [area.py:_apply_segment_recovery_areas] [piecewise.py:_utility_cost_expression]
- Emits log records at info, warning and error level. [execution.py:output_to_cmd_line] [alpha.py:get_alpha_values]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_base_model_helpers.py` (there is no `tests/` directory mirroring `models/_base`; `test_utility_approach.py` and `test_segmented_streams.py` in the same folder also exercise it).
